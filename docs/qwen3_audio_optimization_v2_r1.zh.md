# ASR / TTS：卷积布局、输出头和预填充优化

2026-10-10，在 `feat/qwen3-audio-rpu` 分支落实[上一轮优化空间评估](qwen3_audio_next_optimization.zh.md)。基线代码为 `40e4574`；沿用冻结 checkpoint、精度、输入范围、8 RPU cores 和已有 native，不新增设备程序或依赖。ASR 使用 4 CPU 线程，TTS 使用 8 CPU 线程，全部测速及独立评分持 R1 共享设备锁串行执行。

## 保留的改动

1. **ASR 卷积布局**：CPU FP32 卷积的输入和预备权重使用 channels_last，层间保持布局。FP16 舍入权重、FP32 卷积、FP16 输出及原 GELU 均不变。三层卷积仍在 CPU，18 层音频 encoder 仍连续驻留 RPU。
2. **ASR 输出头融合**：复用已有 causal-LM wrapper 和 native fused head，decode 在同一 Graph 中完成词表投影；prefill 只投影最后位置。保持 FP32 累加、CPU embedding、logits finite 拒绝及 greedy argmax。没有把输出 logits 的接口冒充返回 hidden 的 base model，也没有删除原校验。
3. **TTS codec**：FP32 转置卷积继续用矩阵乘，但将通用 `fold` 替换为冻结几何对应的重叠相加；kernel=stride 时直接重排，kernel=2×stride 时只累加相邻块。普通 kernel=1、stride=1、无 padding 的卷积改用等价矩阵乘；其余普通卷积保留。安装前检查全部目标几何，关闭时恢复原方法。
4. **TTS Talker prefill**：开头的多 token 一次执行，不再拆成逐 token decode。复用 Qwen3 28 层/1024 维的已验证 prefill envelope，并收紧到当前 128-position cache；补齐仅在尾部，KV 位置按逻辑 token 数推进。固定句从 49 次 Talker token 执行变为 1 次 prefill + 31 次 decode，输出 32 步音频码的语义不变。
5. **TTS 换长度保留 decode Graph**：只用现成 `GraphCache.evict` 删除旧 prefill，保留 decode 和有界规划缓存。Talker 最多两张图，Predictor 仍一张；不同逻辑长度的 prefill 保留各自语义签名，没有强行合并。

## 完整请求速度

计时包括完整公开 API 请求，排除加载和结果检查；固定输入 2 次预热、5 次测量。ASR 为同一 4.204 秒录音、11 个 token；TTS 为同一句中文、greedy、32 步、2.48 秒波形。输入和输出工作量均保持一致。

| 配置 | ASR 中位数，ms | TTS 中位数，s |
|---|---:|---:|
| 本轮开始的正式版本 | 294.22 | 3.043 |
| 本轮布局 / codec 单项 | 262.40 | 2.943 |
| 无编译干扰、同进程重测旧 TTS 路径 | — | **3.023** |
| 最终：布局 + fused head / codec + Talker prefill | **258.05** | **2.797** |

ASR 完整请求相对本轮开始降低约 **12.3%**。其布局项贡献最大；fused head 叠加后的中位数只再降低约 4.35 ms，短测量有波动，不将这部分夸大为大幅收益。

TTS 最终采用无编译干扰的同进程对照：保持同一份驻留权重，分别切换旧/新 decoder 与 codec 路径，切换时清 Graph，各做 2+5；两者均输出 32 步、2.48 秒波形。完整请求中位数 **3.02279→2.79677 s，降低 7.48%**；平均数 3.01812→2.79416 s，新版本测量范围 2.750–2.847 s，RTF **1.127**，尚未达到实时生成。加载不计时，各次请求后检查没有编译进程。

TTS 的 codec 单项消融另在同一进程中测得 3.017→2.943 s，降低 **2.5%**；对应固定 codes 子系统消融为 1.179→1.027 s，降低 **12.9%**。子系统收益不能直接当作整句加速。

较早的联合版本和缓存淘汰复测曾分别得到 2.823 s、3.011 s。核查发现本轮后半段 R1 另有 Qwen3.6 native 编译占用 CPU；这两轮联合计时及当时的 TTS 阶段数据仅保留为诊断。没有暂停或修改其他任务；等编译结束后完成上述同进程对照，正式结论以该对照为准。

本轮没有重测 Thor。历史 TensorRT 主体路径为 ASR 154.21 ms、TTS 1.057 s；TTS 两机的精度、采样和生成步数不同，仍不能当作等工作量的硬件比值。见 [Thor 测试报告](qwen3_audio_thor_tensorrt.zh.md)。

## 分阶段耗时

以下为另行 2 次预热、5 次测量的诊断均值，单位 ms，含阶段内传输和同步。它们不是上表无 hooks 的中位数。ASR decode 的输出头现已融合，因此采用合并行比较，不把独立 head 的减少误报为矩阵乘计算消失。

| ASR 阶段 | 本轮之前 | 本轮 |
|---|---:|---:|
| 特征提取/分词 | 4.08 | 3.68 |
| CPU 三层卷积 | 107.84 | **81.60** |
| RPU 音频 encoder + 最终 norm + proj1 | 6.35 | 6.26 |
| conv_out / proj2 边界 Linear | 1.58 | 1.58 |
| 其余音频激活/位置/布局 | 17.72 | 17.75 |
| 文本 prefill + decode + 词表输出头 | 131.52 | **128.94** |
| 其他生成/校验/调度 | 20.81 | 20.05 |
| 诊断总计 | 289.91 | 259.85 |

本轮文本合并行细分：prefill 13.30、prefill 独立 head 2.42、含 fused head 的 decode 113.22 ms。不能直接拿最后一项与旧的纯 decode 91.64 ms 比较。

| TTS 阶段 | 本轮之前 | 本轮联合版本 |
|---|---:|---:|
| 参考音频读取 | 1.01 | 0.97 |
| Talker decoder | 448.05 | **298.36** |
| Predictor decoder | 1062.67 | 1058.20 |
| 输出头 | 214.33 | 206.83 |
| 生成组合/调度 | 204.27 | 201.12 |
| codec | 1156.33 | **1015.37** |
| 其他 | 2.03 | 2.06 |
| 诊断总计 | 3088.69 | 2782.92 |

上表使用编译结束后重新测量的联合阶段数据。codec 单独优化后的另一次阶段测量为 1038.55 ms，曾受编译干扰的联合诊断为 1126.05 ms；这些原始观察均保留在证据中。完整请求结论采用上述同进程 2+5 数据，不拼接不同次阶段测量构造一个更快的总时。Predictor 和其余生成阶段没有实现算法改动，差异只作观察。

## 放弃或保留原路径的方向

- TTS 的 GEMM 输出布局加 fold 方案，固定 codes 中位 1.214 s；K7 卷积预打包为 1.666 s，均比 1.179 s 基线慢，未保留。
- ASR 第一层 GELU 改 FP32 的候选有舍入差异且收益较小，本轮不改变激活精度。此前独立 RPU 卷积路径也明显更慢，继续使用 CPU channels_last 卷积。
- Predictor 只有 5 层，现成 prefill envelope 不含这个几何；没有套用 Talker 的 28 层声明，也没有解除 native 的拒绝保护。它的每帧两个起始 token 继续顺序 decode。
- TTS 输出头继续复用当前 RPU hidden 后独立下发。进一步融合需要 Talker 的 hidden+logits 双输出及 Predictor 多 head 选择接口；本轮没有新增这些 native 接口。不能每步重新设置 head 导致 Graph 失效。
- SnakeBeta、codec 的连续 RPU CNN 路径尚未实施；本轮不近似 sin、不降低 codec 精度，也不改变生成长度。

## 验证与复现

ASR 的 20 条中英文录音 token 全部与基线一致：中文 CER 1.818%（165 字）、英文 WER 5.645%（124 词），均没有触及 token 上限。中文→同长度英文→0.5 秒→8 秒→原输入，以及关闭重载均通过；音频 Graph 保持一张，文本 prefill/decode 两张，缓存 invariant 正常。

另在同一固定录音的相同 11 个生成位置比较完整词表 logits，形状 `[1,11,151936]`、FP16、全部有限；各位置 MSE 和 max-abs 均为 0。该样本中布局和 head 融合没有改变词表输出；不将单条数值相同扩展为任意输入保证。

TTS 的 10 条录音样本数和生成步数全部与基线一致，原有 3 条达到 64 步上限的情况不变。联合版本相对基线的波形 MSE 最大 1.79e−14、max-abs 最大 2.40e−6；这些是 FP32 重排的数值观察，不作为任意质量阈值。同 token 数不同文本→原文本、不同长度→原文本及关闭重载均可复现，KV 存储复用、输出独立、Graph invariant 正常。

独立冻结官方 CPU FP32 ASR 对这 10 条新波形重新转写：中文 5 条、49 字，CER=0%；英文 5 条、35 词，WER=0%，全部整句正确。这仅验证当前小样本的文字内容，不代表人工音质或音色相似度。

缓存淘汰改动后再次完成 2+5 及 10 条输入：Talker decode 的 `graph_lifetime_id`/`build_generation` 全程不变，回放计数增长；不同长度仅重建 prefill，Graph 始终为 Talker 2 / Predictor 1。10 条波形与已独立评分的波形逐点相同，因此复用该评分，没有再次识别相同音频。

检查包括 ASR 42 项、TTS 33 项、codec 29 项 host tests，以及评分函数 2 项；换长度定向淘汰另复核对应 host case。没有扩大为不相关模型的完整认证。

R1 候选及脚本保留在 `/home/agi/rhinoforge_audio_runtime/optimization_20261010/`，codec 消融在 `tts_codec_20261010_v2/`。公开完整请求入口仍是 [benchmark_qwen3_audio.py](../examples/benchmark_qwen3_audio.py)，TTS 的 Graph 检查已更新为 Talker 2 / Predictor 1。独立转写继续使用冻结官方 CPU FP32 ASR 和隔离的参考依赖。

验证后的两个 API、codec 和 benchmark 脚本已持锁更新到 `/home/agi/rhinoforge_audio_runtime/`，部署文件与测试候选一致；原版备份在 `backup_40e4574_audio_v2_20261010/`。没有覆盖其他模型的构建或运行目录。

[机器可读证据](evidence/qwen3_audio_optimization_v2_r1/summary.json)记录消融、完整请求、干净环境匹配对照、阶段、质量、缓存生命周期和部署身份。受编译干扰的测量明确归入诊断，不与正式结果混用；没有将波形、中间激活或底层 trace 加入仓库。

本轮沿用的 native SHA256 为 `d2cd78227982c68d3576de06f26712763b26af6b7283f3107b55cd2a9f4521f4`。没有修改权重、共享依赖或设备算子资产。加载时间、新长度首次 BUILD 与稳态请求仍需分别看待；本轮没有新增流式、长文本、并发、人工音质或音色相似度评测。
