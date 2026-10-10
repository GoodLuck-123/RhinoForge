# Qwen3 ASR / TTS：Thor 对比与 R1 分阶段优化

2026-10-09，分支 `feat/qwen3-audio-rpu`。本轮基线为 `4f099b3`，沿用[上轮优化](qwen3_audio_optimization_r1.zh.md)的权重、输入和评分规则。ASR 主体及输出头在 RPU；TTS Talker、Predictor 和输出头在 RPU，音频 codec 在 CPU。

**R1 ASR 耗时从 0.639 秒降到 0.426 秒（−33.3%）；TTS 从 3.836 秒降到 3.588 秒（−6.5%），使用 8 个 CPU 线程后为 3.402 秒（−11.3%）。20 条 ASR token 不变，10 条 TTS 波形逐点一致。TTS 仍慢于播放速度。**

2026-10-10 补充：[Thor TensorRT 实测](qwen3_audio_thor_tensorrt.zh.md)，包含同日 PyTorch 对照、完整请求耗时和质量核验；下文保留 10 月 9 日的原始结果。

## 测量口径

- R1：8 RPU cores；PyTorch 2.10.0+cpu、Transformers 5.3.0，沿用原生库、Launch v1.1.0 与原算子资产。只有 Python 适配/示例改变。
- Thor：`Thor1`，NVIDIA Thor，计算能力 11.0；驱动 580.00、CUDA 13.0、PyTorch 2.10.0+cu130；ASR Transformers 5.3.0，正式 TTS 使用隔离的官方 Transformers 4.57.3；`nvpmodel` 120W/mode 1，未修改功耗或锁频。独立目录使用既有环境，未替换系统 PyTorch。
- 同一冻结 checkpoint，加载器验证其身份；R1 使用 prepared TTS source，Thor 正式 TTS 使用相同来源提交的未修改官方源码。batch=1、生成上限 64；默认 4 个 CPU 线程，8 线程作为独立配置。R1/ASR 为 greedy；Thor TTS 为 BF16、SDPA、Talker/Predictor 均开启采样，每请求重置随机种子为 0。采用原 generation_config：两级 top-k=50、top-p=1.0、temperature=0.9，Talker repetition_penalty=1.05。TF32 关闭；未使用 TensorRT、CUDA Graph、torch.compile 或外部 FlashAttention 包，SDPA 的内部 kernel 由 PyTorch 选择。
- ASR：WAV 读取在计时外，特征提取、生成、文本解码在内。TTS：参考 WAV 读取/验证/哈希、生成及 CPU 波形返回都在内；固定输入重复测量命中说话人缓存。CUDA 计时开始/结束均 synchronize，避免只计入异步提交。
- 每个正式配置预热 2 次、固定输入测量 5 次，再测 ASR 20 条或 TTS 10 条不同输入；模型串行持设备锁。阶段诊断单独预热 2 次、测量 3 次，不用带 hooks 的数据计算正式加速比。
- ASR Thor 使用 GPU FP16 encoder/decoder/head；R1 卷积为 CPU FP32 计算、FP16 边界。Thor 正式 TTS 使用 GPU BF16 模型与 codec、官方生成流程；R1 为 RPU FP16 主体＋CPU FP32 codec，使用固定 greedy Predictor 循环。初期 Thor greedy 对照未通过质量检查，单独列在失败试验中。

## 稳态总耗时

ASR 固定录音为 4.2039375 秒；TTS 固定句为“你好，欢迎使用语音助手。”。RTF=耗时/音频时长，低于 1 才表示该离线任务快于播放，不能当作流式首包延迟。

| 任务/配置 | 中位数（秒） | 最小–最大（秒） | 输入/生成音频（秒） | RTF | 生成步数 |
|---|---:|---:|---:|---:|---:|
| ASR R1 上轮 | 0.639 | 0.630–0.651 | 4.204 | 0.152 | 11 |
| ASR R1 本轮 | 0.426 | 0.424–0.433 | 4.204 | 0.102 | 11 |
| ASR Thor FP16 | 0.245 | 0.243–0.252 | 4.204 | 0.059 | 11 |
| TTS R1 上轮，4 线程 | 3.836 | 3.822–3.877 | 2.480 | 1.551 | 32 |
| TTS R1 本轮，4 线程 | 3.588 | 3.551–3.653 | 2.480 | 1.452 | 32 |
| TTS R1 本轮，8 线程 | 3.402 | 3.349–3.431 | 2.480 | 1.367 | 32 |
| TTS Thor BF16＋采样 | 2.543 | 2.540–2.549 | 2.320 | 1.097 | 30 |

ASR 三条路径固定输入均生成 11 token，Thor 比本轮 R1 快约 1.74 倍。TTS R1 固定句生成 2.48 秒，Thor BF16 生成 2.32 秒，均自然结束。TTS 的精度、采样方式、生成长度均不同，因此这里是两套可用部署流程的耗时对照，不能解释为等工作量的硬件加速比。两者固定句 RTF 均大于 1，尚未实时。

加载耗时单列，不含包导入，未清空文件缓存，不做冷盘启动性能结论。R1 包含逐文件身份校验和布局变换；Thor 官方加载不重复计入此前的同源资产校验，加载列不用于计算硬件加速比：

| 配置 | 加载（秒） | 两次预热（秒） |
|---|---:|---|
| asr-r1 | 23.538 | 0.903, 0.416 |
| asr-thor | 15.743 | 0.814, 0.254 |
| tts-r1 | 32.083 | 4.114, 3.608 |
| tts8-r1 | 32.029 | 3.774, 3.393 |
| tts-bf16-thor | 6.097 | 3.222, 2.546 |

## ASR 各阶段

单位毫秒，3 次诊断均值。RPU/CPU 传输和同步计入其所属阶段；各列互斥阶段相加为诊断总耗时。Thor 每阶段增加 CUDA 同步，诊断总时间可高于无 hooks 的正式时间。

| 阶段 | R1 上轮 | R1 本轮 | Thor FP16 | 处理 |
|---|---:|---:|---:|---|
| 音频特征/分词 | 4.01 | 3.84 | 4.79 | 约 4 ms，保留输入校验 |
| 音频卷积 | 107.17 | 106.04 | 1.53 | 沿用已验证的 CPU FP32 卷积优化 |
| 音频 111 个 Linear | 57.76 | 54.36 | 6.32 | 继续全部运行于 RPU |
| 音频 attention/norm/激活等 | 107.12 | 106.54 | 7.56 | 剩余 CPU 瓶颈，未改数值语义 |
| 文本 prefill | 13.35 | 13.32 | — | 保持已有 fused/Graph 路径 |
| 文本 decode | 94.67 | 91.22 | — | 保持 KV 复用与 Graph replay |
| 词表输出头 | 233.07 | 26.52 | 17.98 | 改为 RPU FP16 权重＋FP32 累加 |
| 组合/argmax/校验/其他 | 25.07 | 20.63 | 4.37 | 保留必要校验和文本输出 |
| 文本 decoder 合计 | 108.02 | 104.54 | 212.61 | 此行为上两项小计，不重复加总 |

输出头从 233.07 ms 降到 26.52 ms。直接使用 RPU 最后一个 hidden position，避免先回 CPU 再投影；输入 embedding 留在 CPU，单独转换输出头一次。增加约 296.75 MiB 的 RPU 权重存储。11 步 logits 诊断的最大 max-abs 为 0.015625、MSE 最大约 3.02e−8，top-1 均相同；这些是数值观察，不是另设的准入阈值。

## TTS 各阶段

单位毫秒，R1 均为 4 线程；Thor 不同生成长度对应不同工作量，表格用于定位各自瓶颈，不能逐项当作等量算子加速比。

| 阶段 | R1 上轮 | R1 本轮 | Thor BF16＋采样 |
|---|---:|---:|---:|
| 参考音频读取 | 0.96 | 0.96 | 0.66 |
| 说话人编码（缓存命中） | 0.00 | 0.00 | 0.00 |
| Talker decoder | 453.50 | 447.13 | 576.65 |
| Code Predictor decoder | 1092.07 | 1057.81 | 1488.92 |
| 输出头 | 252.97 | 237.34 | 48.51 |
| 生成调度与组合 | 435.15 | 211.81 | 404.97 |
| 音频 codec 解码 | 1709.49 | 1654.74 | 49.86 |
| 其他 | 2.04 | 2.04 | 1.61 |

- **生成调度**：固定 Predictor 每帧只生成 15 个 greedy 子码，直接循环调用已安装 decoder/head，省去重复构造通用 HF generate 状态。调度由 435.15 ms 降至 211.81 ms。Talker 的 EOS、重复惩罚、64-token 上限及必要有限值检查保留。
- **Talker/Predictor/输出头**：沿用 RPU fused decoder、KV reset、固定 Graph；主体的小幅差异包含 Python 调度变化和测量波动，不宣称单独优化了算子。
- **codec 解码**：仍是 R1 最大单项。独立固定 codec 输入，2/4/8 线程各测 2 次，均值分别约 2.096/1.666/1.450 秒，波形逐点一致；8 线程比 4 线程快约 13.0%。完整 8 线程请求又按 2+5 正式复测，并核验全部 10 条波形。
- **参考音频/说话人**：读取约 1 ms，保留逐次校验；首次请求说话人编码约 179 ms，后续同内容命中缓存。参考内容改变仍重算，未缓存整段模型输出。
- **未采用**：2 线程 codec 更慢；没有缩短输入/输出、降低 64-token 上限或删除模型层；没有绕过数据同步，普通语音 decode 也未强开 RhinoVLA 专用 fast-replay。

库 API 不修改调用方全局 CPU 线程数；TTS CLI 与评测脚本新增 `--threads`，默认仍为 4。R1 单请求可使用 `--threads 8`；并发服务需另测线程争用。

## 识别和生成质量

ASR 沿用 AISHELL 中文 10 条（165 字、7 位说话人）、LibriSpeech dummy 英文 10 条（124 词、1 位说话人）。R1 优化前后及 Thor 的 20 条完整 token 序列均相同。

| 路径 | 中文 CER / 1−CER | 英文 WER / 1−WER | 整句一致（中/英） |
|---|---:|---:|---:|
| R1 上轮、本轮、Thor | 1.82% / 98.18% | 5.65% / 94.35% | 9/10、4/10 |

R1 TTS 4/8 线程的全部 10 条波形与上轮逐点一致，复用相同波形的原独立 CPU FP32 ASR 评分：中文 CER=0%、英文 WER=0%。中文 1 条、英文 2 条达到生成上限，已有文本完整，但更长输入可能截断。Thor BF16 的 10 条输出重新用同一冻结官方 CPU FP32 ASR 转写，归一化与编辑距离算法不变：

| Thor TTS 配置 | 中文 CER | 英文 WER | 整句一致（中/英） | 触及上限（中/英） |
|---|---:|---:|---:|---:|
| BF16＋采样 | 0.00% | 0.00% | 5/5、5/5 | 0/5、0/5 |

逐条转写与错误计数保留在机器可读证据中。转写一致仅验证有限样本的文字内容，不认证音色、自然度、静音占比或 MOS。不同精度的自回归结果可能改变结束位置；不能把多生成的音频当作有效吞吐提升。

## 不同输入与验证

| 任务/配置 | 中文平均秒 / RTF | 英文平均秒 / RTF |
|---|---:|---:|
| asr-r1 | 0.911 / 0.168 | 0.982 / 0.202 |
| asr-thor | 0.326 / 0.060 | 0.433 / 0.089 |
| tts-r1 | 4.242 / 1.403 | 5.505 / 1.404 |
| tts8-r1 | 3.992 / 1.320 | 4.989 / 1.273 |
| tts-bf16-thor | 2.486 / 1.110 | 2.657 / 1.114 |

不同输入各测一次，ASR 长度变化包含必要的 Graph BUILD，因此与固定长度重复耗时不同。

- 相关主机检查 69 项通过，补充 Predictor generate 恢复检查 1 项通过，共 70 项；语法检查、diff 空白检查通过。
- ASR 固定长度 Graph size=2，REPLAY 递增；TTS 两组件 Graph size 各为 1，REPLAY 递增；全部 invariant 正常。固定输入输出重复一致。
- ASR 同长度换内容、0.5/8 秒边界、切回原输入和关闭重载通过，token 与既有参考一致；TTS 更换参考、切回、保留旧输出和关闭重载通过。
- 权重身份/精度边界、关闭失败时保留资源的契约未放宽。没有重新跑无关模型矩阵。

## Thor 未通过质量检查的试验

初期严格模仿 R1 greedy 设置的 Thor 路径发生生成退化。这些耗时只保留作诊断，不列为有效性能结果：

| 配置 | 固定句耗时 / 音频秒 | 中文 CER | 英文 WER | 有正确转写句数 |
|---|---:|---:|---:|---:|
| 5.3，FP16 主体＋FP32 codec，eager，greedy | 5.980 / 5.04 | 100.00% | 100.00% | 0/10 |
| 5.3，全 FP32，eager，greedy | 2.751 / 2.64 | 79.59% | 82.86% | 2/10 |

其中 FP32 的另外 8 条接近静音（RMS 约 1.11e−5）；FP16 混合路径虽然非零，但 10 条都没有可识别转写。另用官方 4.57.3、FP32、SDPA 对“请把杯子放在桌子上。”做单条复核，也达到 64-token 上限，不能简单归因于 Transformers 5.3。最终采用官方 BF16＋采样配置并重新完成全部质量评分。这里同时改变了精度、采样及运行实现，尚未把生成退化归因到某一个数值或缓存因素。

失败路径的逐条结果和阶段计时仍保留在证据中，不能将其较低 RTF 或多生成的静音当作优化收益。

## 文件与复现

- [机器可读证据](evidence/qwen3_audio_thor_stages/summary.json)：逐次计时、分阶段数据、转写评分、源文件 SHA256、Graph 与生命周期结果。
- R1：`/home/agi/rhinoforge_audio_runtime/stage_optimization_20261009/`；候选 Python、正式 `asr/tts/tts8`、分阶段 JSON、质量对照、生命周期检查和日志均保留。
- Thor：`/home/hx/rhinoforge_audio_20261009/`；同源 assets、`thor_bench.py`、`thor_official_bf16.py`、`asr/tts-bf16` 及失败试验 `tts/tts-fp32` 和对应 stages 目录保留。大权重、音频和原始日志不入仓库。

R1 使用[原有持锁环境](qwen3_audio_benchmark_r1.zh.md#复现与证据)，将 PYTHONPATH 首项设为本轮目录的 `python`。执行 `examples/benchmark_qwen3_audio.py --task asr|tts`，沿用原 checkpoint/source/manifest，输出选择新目录；TTS 8 线程附加 `--threads 8`。分阶段脚本为同目录 `stages_after.py asr|tts --output NEW.json`。

Thor 使用 `/home/hx/miniconda3/envs/3.12/bin/python`，`PYTHONPATH=/home/hx/rhinoforge_audio_20261009/python:/home/hx/rhinoforge_audio_20261009/deps`，`TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1`，持有该目录 `gpu.lock`。运行 `thor_bench.py asr|tts --output NEW_DIR`；全 FP32 TTS 加 `--fp32-tts`；分阶段另开进程加 `--stages`。

Thor 正式 TTS 使用 `thor_official_bf16.py tts --output NEW_DIR`，PYTHONPATH 在上述路径之前增加专用目录 `official/tts-ref-deps:official/Qwen3-TTS`（均展开为绝对路径）；阶段诊断附加 `--stages`。参考克隆仍为相同音频、speaker-only 模式，并缓存其 embedding。

当前结果不是 Thor 最优部署性能，也不是两块设备的等功耗比较。未测试 TensorRT/编译、流式首包、并发、完整公开数据集、噪声/远场或人工音质评分。R1 TTS 存在到达长度上限的样本，Thor 正式 BF16 的这 10 条均未达到上限；当前短句输入限制继续生效。
