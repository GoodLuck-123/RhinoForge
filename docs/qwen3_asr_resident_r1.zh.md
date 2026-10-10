# Qwen3-ASR：音频编码器连续驻留 RPU

2026-10-10，在 `feat/qwen3-audio-rpu` 分支完成现有 0.6B 有限输入配置的优化。固定 4.204 秒录音的稳态中位耗时从 **426.04 ms 降到 294.22 ms**，降低 **30.94%**，速度为原来的 **1.45 倍**。20 条中英文录音的输出 token 全部与优化前一致。

本轮使用仓库 `rhinoforge-port` 流程，属于已有模型的 **Runtime extension**：复用已有双向编码器、Graph、SPM 和算子资产，增加 ASR 专用 host 模式；未新增设备程序，也未扩大模型或输入范围。TTS 的并行优化见[单独报告](qwen3_tts_chain_optimization_r1.zh.md)。

## 计算位置

| 阶段 | 当前执行位置与边界 |
|---|---|
| 音频读取、特征提取、分词 | CPU；正式计时不含 WAV 文件解码，包含特征提取 |
| 三层卷积及其 GELU | CPU；保留 FP16 权重舍入、FP32 卷积、FP16 输出边界 |
| `conv_out` | RPU Linear，随后回 CPU 处理布局及位置 embedding |
| 18 层音频 encoder | RPU 连续执行 LayerNorm、Q/K/V/O、双向 attention、ERF GELU、MLP、残差、FP16 clamp；层间不返回 CPU |
| `ln_post` 与 `proj1` | 接在同一 RPU Graph 内 |
| `proj1 → GELU → proj2` | Graph 输出回 CPU 做 GELU，`proj2` 在 RPU；此处仍有一次往返 |
| 28 层文本 decoder、词表输出头 | RPU，沿用已验证路径；embedding、生成组合与 argmax 在 CPU |

原来的 111 个音频 Linear 保持全部在 RPU，其中 **109 个**与编码器连接，只保留 `conv_out`、`proj2` 两次独立 Linear。`audio_linear_calls` 继续统计逻辑投影次数，不代表 111 次独立下发。

复用 SigLIP 的双向 LayerNorm 编码器框架，增加严格的 ASR 模式：18 层、896 维残差、3584 维 MLP、14 个有效 attention heads，每个 64 维。Q/K/V 补两个零 head 至 16，O 投影对应列补零；归一化仍在原 896 维执行，attention scale 保留 `1/sqrt(64)`。使用原模型 ERF GELU 及 FP16 clamp，线性采用 FP32 累加。没有把 SigLIP 的近似 GELU 或 head scale 套到音频模型。

## 速度与分阶段结果

R1、4 CPU 线程、batch 1、greedy、最多 64 token；固定输入预热 2 次、测量 5 次。模型加载不计入请求耗时。对照来自上轮同配置的[阶段报告](qwen3_audio_thor_stages.zh.md)，本轮没有改变录音、分词或生成步数。测试持同一设备锁，正式数据不与编译或 TTS 测试并发。

| 指标 | 优化前 | 连续驻留后 |
|---|---:|---:|
| 固定录音稳态中位数 | 426.04 ms | 294.22 ms |
| 固定录音平均数 | 427.42 ms | 296.53 ms |
| 固定录音 RTF（总耗时/总音频） | 0.10167 | 0.07054 |
| 中文 10 条平均耗时，含形状切换 BUILD | 910.65 ms | 729.94 ms |
| 英文 10 条平均耗时，含形状切换 BUILD | 982.30 ms | 827.68 ms |
| 模型加载 | 23.54 s | 23.50 s |

不同长度输入会重新 BUILD，不能将 294 ms 稳态数值当作任意新长度请求的耗时。首个请求约 771 ms，第二次预热约 299 ms。

独立分阶段诊断，本轮 2 次预热、5 次测量，单位 ms，含所属阶段的数据传输。分阶段均值与无 hooks 的正式中位数口径不同。

| 阶段 | 优化前 | 连续驻留后 |
|---|---:|---:|
| 特征提取/分词 | 3.84 | 4.08 |
| 三层 CPU 卷积 | 106.04 | 107.84 |
| 111 个音频 Linear 的独立调用 | 54.36 | — |
| 音频 attention/norm/激活等 CPU 工作 | 106.54 | — |
| 18 层编码器 + 最终 LayerNorm + `proj1`，RPU Graph | — | **6.35** |
| 两个边界 Linear：`conv_out`、`proj2` | — | 1.58 |
| 剩余音频 CPU 激活/位置/布局等 | — | 17.72 |
| 文本 prefill | 13.32 | 13.30 |
| 文本 decode | 91.22 | 91.64 |
| 输出头 | 26.52 | 26.58 |
| 其他生成/校验/调度 | 20.63 | 20.81 |
| 诊断总计 | 422.47 | 289.91 |

原来的 106.54 ms 分类同时含编码器和卷积前端的 CPU 工作，不能将其全部理解为 attention 的耗时。优化后编码器主体单独计时，剩余 CPU 边界明确列出。

Thor 历史同录音端到端结果为 PyTorch **240.41 ms**、TensorRT 文本主体 **154.21 ms**，见[TensorRT 报告](qwen3_audio_thor_tensorrt.zh.md)。本轮 R1 分别约为其 1.22 倍、1.91 倍；本轮未重新测试 Thor。Thor 的音频 encoder 仍是 PyTorch GPU 路径，不能把所有收益归因于 TensorRT。R1 当前主要差距在 CPU 卷积，而非已驻留的 18 层音频编码器。

## 卷积 RPU 路径评估

直接调用已有 `conv2d_mc_nhwc` 八核算子，以同一录音的三个真实层输入分别对照，2 次预热、5 次测量。输入、输出布局转换和主机传输计入耗时，权重预先上传。为符合算子约束，batch 5 补到 8，输出通道 480 补到 512，首层输入通道 1 补到 16；结果裁回原形状。该微测试每层使用相同 CPU 参考输入，不累积前层数值差异。

| 卷积层 | CPU FP32，ms | RPU 八核，ms |
|---|---:|---:|
| 1 | 23.28 | 86.68 |
| 2 | 51.95 | 244.86 |
| 3 | 19.71 | 61.15 |
| 三层中位数之和 | 94.94 | 392.69 |

形状、有限值检查通过，但独立 RPU 路径约慢 4.14 倍，故**不接入正式推理**。它包含补齐计算、布局和边界传输，不能据此推断所有可能的融合卷积实现都慢；当前已有接口不能直接把这些 DDR 卷积与编码器 Graph 连起来。评估后释放卷积 SPM 缓存，不让它干扰后续融合模型。

初次探测与 native 编译重叠，仅用于诊断；上表来自编译结束后的互斥复测。CPU 激活切换 FP32 的小试验仅在第一层获益，其余层未稳定获益，本轮保留原激活精度边界。

## 质量与生命周期

20 条公开录音，中文 165 字、英文 124 词，沿用之前的正规化和评分规则：

| 指标 | 优化前与本轮 |
|---|---:|
| 中文 CER / 1−CER | 1.82% / 98.18% |
| 英文 WER / 1−WER | 5.65% / 94.35% |
| 中文/英文整句正确 | 9/10、4/10 |
| 与原 R1 的 token 完全相同 | 20/20 |
| 达到生成上限 | 0/20 |

相同长度请求的音频 Graph 仅 BUILD 一次，稳态 replay 依次为 2、3、4、5、6，size 始终为 1，invariant 为 true；文本 Graph size 保持 2，replay 为 31、42、53、64、75。不同长度先清旧图，再 BUILD，未累计旧长度缓存。

另验证中文→等长度英文→0.5 秒→8 秒→原中文及关闭重载，全部 token 与旧版对应输入相同；相同长度换内容正确使用新输入，切回及重载输出一致。音频 Graph size 保持 1，音频和文本的 cache invariant 均通过。

固定录音的最终音频特征形状为 `[55,1024]`。与匹配的 CPU FP16 路径对照（卷积按相同 FP32 边界计算），MSE 为 `1.62e−9`、max-abs 为 `4.33e−4`；将这些已舍入权重和输入提升至 FP32 的参考，MSE 为 `1.14e−9`、max-abs 为 `3.05e−4`。形状、dtype、有限值正常。这是累计浮点误差记录，不是质量阈值，也没有将提升后的 FP32 参考冒充重新加载原始精度权重的整模型结果。

ASR 定向 host tests **39 项**通过；另有既有 SigLIP attention eligibility 和音频评分测试 **3 项**通过。没有扩大到不相关模型的完整硬件认证。

## 复现与范围

沿用[原始语音配置](qwen3_audio_r1.zh.md)的 checkpoint、依赖、Launch 与 operator assets；需要重编译包含 `siglip_set_audio_mode` 的 native extension，不能仅复制 Python 到旧二进制。旧二进制在加载完整权重前报错。正式 benchmark 入口为 `examples/benchmark_qwen3_audio.py --task asr`，新增音频 Graph 的稳定性检查。

新 native `.so` SHA256：`d2cd78227982c68d3576de06f26712763b26af6b7283f3107b55cd2a9f4521f4`。R1 的构建、诊断和复现脚本位于 `/home/agi/rhinoforge_audio_runtime/asr_resident_20261010/`。

验证后的 Python 与 native 已持锁更新到 `/home/agi/rhinoforge_audio_runtime/`，部署文件与测试候选的 SHA256 一致。原文件备份在 `backup_ffcbf9d_audio_chain_20261010/`，没有覆盖模型权重或共享依赖。

[原始数值与运行摘要](evidence/qwen3_asr_resident_r1/summary.json)记录正式样本、阶段计时、卷积评估、生命周期及 native 身份。没有把模型权重、中间激活或硬件 trace 放入仓库。

这仍是单请求、16 kHz、0.5–8 秒的有限配置。未扩大长音频、多语言、流式、并发或完整数据集质量认证；小样本 CER/WER 不能代表完整识别率。
