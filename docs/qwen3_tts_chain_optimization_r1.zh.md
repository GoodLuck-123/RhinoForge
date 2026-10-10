# Qwen3-TTS 0.6B Base：R1 输出头与 codec 优化

2026-10-10，`feat/qwen3-audio-rpu`。沿用冻结 checkpoint、prepared source、RPU FP16 Talker/Predictor/输出头与 CPU FP32 codec；batch=1、greedy、上限 64，不改变语音长度或生成参数。关联[此前分阶段评测](qwen3_audio_thor_stages.zh.md)及[Thor TensorRT 对照](qwen3_audio_thor_tensorrt.zh.md)。

**完整请求中位耗时从 3.366 秒降至 3.043 秒（−9.6%），生成长度仍为 2.48 秒；10 条中英文转写全部正确。**

## 改动与执行位置

- **输出头仍在 RPU**：单 token decoder 完成后，输出头直接消费该次原 RPU hidden，省去“CPU FP32 hidden → FP16 → 重新上传 RPU”。官方生成组合所需的 CPU hidden 仍保留。绑定精确的 CPU 输出对象、仅消费一次，且下一 decoder 调用前清空；旧输出或多 token prefill 使用原路径。CPU logits 独立存储，生成语义不变。
- **codec 仍在 CPU FP32**：6 个 `ConvTranspose1d` 用等价的矩阵乘＋`fold` 重叠相加；普通 `Conv1d` 保持原实现。stride、padding、bias、因果裁剪和后续激活保持不变。补丁仅安装在 speech decoder 内；关闭模型时恢复原方法。
- **主体仍为 RPU 推理**：Talker 与 Predictor 继续使用现有 fused decoder、KV cache 和 Graph，每组件 Graph size=1。没有将主体改回 CPU，也没有删层、截短输出、改变精度或缓存整段语音。

codec 的主要 CPU 瓶颈是卷积，attention 在固定输入的诊断中约 3 ms。固定 codec 输入的消融（2 次预热＋3 次测量）中，原路径中位 1.418 秒、只替换转置卷积 1.150 秒、额外替换普通卷积 1.172 秒；后者没有净收益，因此最终删去普通卷积改动及其额外权重布局副本。首层转置卷积（1536→768、kernel=16、stride=8）局部实测从 268.46 ms 降到 30.35 ms；前段上采样转置卷积从 18.09 ms 降到 1.33 ms。局部成绩只用于选方案，完整请求结果单独测量。

输出头尝试加入独立 Graph 时，被已有的 fresh DDR / mutable DMA 边界明确拒绝；没有绕过该保护。也检查了现有 `causal_decoder_set_lm_head`：启用后 decode 仅返回 logits，Talker 却仍需 normalized hidden 供下一帧 Predictor 使用；同时返回 hidden 的 exact-rerank 分支目前仅接受 Qwen3-VL-2B W4。Predictor 的 15 个 head 若逐步调用 setter 切换，又会触发 `invalidate_model_state()`，无法保持稳定 REPLAY。因此最终输出头保持 eager 调用，decoder 的 Graph 契约不变；进一步融合需要明确的“双输出”及多 head 选择接口，不能直接套用现有路径。

## 测量口径

R1，8 RPU cores，CPU 8 线程，PyTorch 2.10.0+cpu、Transformers 5.3.0、运行资产包 `RhinoForge-runtime-v1.1.0`、Launch 1.0.0（`613f2da`）与相同算子资产。正式测速前原生库编译已结束，每项任务持 `/mnt/ssd_data/rhino-deploy-platform/rpu.lock`，CPU 评分与硬件测速也串行。

固定句“你好，欢迎使用语音助手。”及相同克隆参考：2 次预热＋5 次完整请求计时，包含参考文件读取、生成、codec 与 CPU 波形返回；加载、结果检查和写 WAV 不在计时内。所有重复输出逐点一致。新旧路径均生成 32 步、2.48 秒音频；RTF 采用 5 次总耗时÷总音频时长。阶段诊断另外预热 2 次、测量 3 次，不拿 hooks 计时计算正式加速比。

| 配置 | 中位数（秒） | 最小–最大（秒） | RTF |
|---|---:|---:|---:|
| 同日原路径 | 3.366 | 3.340–3.399 | 1.357 |
| 优化后 | 3.043 | 3.038–3.188 | 1.239 |

完整请求中位耗时降低 **9.6%**。以下为另行测量的互斥阶段均值，单位毫秒；RPU/CPU 传输与同步归入所属阶段。

| 阶段 | 原路径 | 优化后 |
|---|---:|---:|
| 参考音频读取 | 0.97 | 1.01 |
| Talker decoder | 446.00 | 448.05 |
| Predictor decoder | 1048.99 | 1062.67 |
| 输出头 | 230.51 | 214.33 |
| 生成组合/调度 | 196.94 | 204.27 |
| codec 解码 | 1429.18 | 1156.33 |
| 其他 | 2.04 | 2.03 |

输出头阶段降低约 7.0%，codec 降低约 19.1%；decoder 的小幅差异是本轮观察值，不视为该路径的优化收益。

此前 Thor TensorRT 主体路径的输出头为 41.83 ms、codec 为 50.85 ms，二者仍由 PyTorch GPU 执行。本轮 R1 仍明显较慢：输出头保留逐步独立下发，codec 保留 CPU。Thor 使用 BF16、采样及 28 步输出，R1 使用 FP16 主体、FP32 codec、greedy 及 32 步输出，这些阶段数值只能定位差距，不能直接视为等工作量的硬件比值；本轮未重新测试 Thor。

## 数值与任务质量

10 条中英文输入全部保持原生成步数和波形长度；codec 的 FP32 运算重排带来普通舍入差异，逐句波形 MSE 最大 2.31e-13、max-abs 最大 1.19e-05、相对 L2 最大 2.82e-06，输出形状、dtype、finite 均正常。这些数字作为数值观察，不设端到端任意误差阈值。

独立的官方 CPU FP32 ASR 对全部 10 条新波形重新转写：中文 CER=0%、英文 WER=0%，中英各 5/5 句正确。沿用之前的归一化与编辑距离规则，没有以波形接近替代任务评分。

原有中文 1 条、英文 2 条达到 64 步上限，本轮情况不变。转写评分只验证文字内容，未测试人工音质、音色相似度或 MOS。

## 验证与复现

- TTS 定向 host checks 32 项通过，覆盖单 token hidden 的一次性消费/旧输出拒绝，以及 CPU 转置卷积不同长度、重叠、恢复与安装前几何拒绝。
- 正式 2＋5 重复及 10 条不同输入通过；每组件 Graph size=1、REPLAY 递增、invariant 正常，波形由调用方独立持有。
- 最终完整 2＋5 测速、10 条质量样本和阶段诊断均使用整合 ASR 改动的新原生库，Graph 生命周期正常。整合库 SHA256 见证据文件。
- [机器可读证据](evidence/qwen3_tts_chain_optimization_r1/summary.json)记录逐次耗时、逐阶段、波形差异和独立评分。
- R1 候选目录：`/home/agi/rhinoforge_audio_runtime/tts_chain_20261010/`；脚本、原始日志、WAV 与 JSON 保留在设备。未将权重、内部激活或音频大文件加入仓库。

复现沿用[原有持锁环境](qwen3_audio_benchmark_r1.zh.md#复现与证据)，将 `PYTHONPATH` 首项替换为候选目录的 `python`，运行该目录 `benchmark.py --task tts`，使用原 checkpoint/source/manifest，附加 `--threads 8` 与新的 `--output`。阶段脚本为 `stages_after.py tts --threads 8 --output NEW.json`。旧版对照的 Python 来自 `stage_optimization_20261009/python`，不覆盖部署目录。

最终 Python 与整合 native 已持锁更新到 `/home/agi/rhinoforge_audio_runtime/`，部署文件与测试候选的 SHA256 一致。旧文件备份在其下 `backup_ffcbf9d_audio_chain_20261010/`；模型权重及共享依赖保持原版本。

当前仍未达到实时播放速度。没有新增 codec 的 RPU 实现，也未测试流式首包、并发、长文本和人工音质；跨 Thor 对照仍有精度、采样和生成长度差异，不能视为等工作量的硬件比值。
