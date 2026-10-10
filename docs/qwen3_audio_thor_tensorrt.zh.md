# Qwen3 ASR / TTS：Thor TensorRT 实测

2026-10-10，分支 `feat/qwen3-audio-rpu`，仓库基线 `dbcc369`。沿用[上一轮 Thor 对比](qwen3_audio_thor_stages.zh.md)的冻结权重、音频及评分规则。本轮仅测试 Thor；没有修改 RPU 实现或 R1 部署。

**完整请求中位耗时：ASR 从 240 ms 降至 154 ms（−35.9%）；TTS 从 2.574 s 降至 1.057 s（−58.9%）。这里是 TensorRT 解码主体＋PyTorch 其余步骤的端到端结果。**

## 实际执行范围

| 模型 | TensorRT | 仍由 PyTorch 执行 |
|---|---|---|
| Qwen3-ASR-0.6B | 完整 28 层文本 decoder，含 attention、MLP、norm 和 KV 更新 | 音频 encoder、embedding、词表 head、特征处理及生成调度 |
| Qwen3-TTS-12Hz-0.6B-Base | 完整 28 层 Talker decoder、5 层 Code Predictor decoder，含 KV 更新 | speaker encoder、embedding、输出 head、采样/调度、音频 codec |

PyTorch 神经网络组件也在 Thor GPU 上；CPU 负责分词、文件处理、Python 调度和最终波形返回。没有 CPU 模型主体替代，也没有把单层试验当作完整模型成绩。

采用已有 Torch-TensorRT 的 `torch.export` / Dynamo AOT 转换，每个组件分别构建 prefill/decode 引擎；`require_full_compilation=True`，每个导出边界均确认只有一个 `TorchTensorRTModule`，禁止该边界内静默回退。模型权重及原 decoder 层的数学定义保持不变，用显式 tensor KV 输入输出连接原生成流程。

- ASR 为 FP16；TTS 模型与 codec 均 BF16，保持官方 Transformers 4.57.3、SDPA、采样设置。没有量化，也未开启 TF32。
- batch=1；ASR KV 容量 256，Talker 128，Predictor 32。ASR/Talker prefill 固定右补至 128 token，Predictor prefill 为 2，decode 为 1。
- 真实前缀使用因果 mask；补齐位置不能被真实前缀关注，返回值截去补齐部分，逻辑 KV 位置只增加真实 token 数。后续生成按真实位置覆盖 KV。只接受无 padding 的单样本输入；不支持任意批量或无限长度。
- 动态 prefill 的初次 ASR 转换未通过完整转换检查；TTS 动态构建探索主动停止，改用上述固定形状。失败/探索时间不计入有效引擎构建时间或推理时间。
- 正式 ASR 27 次请求实际调用 prefill 27 次、decode 386 次；TTS Talker prefill 17 次、decode 488 次，Predictor prefill 488 次、decode 6832 次。引擎在各次请求间复用，KV 按请求重置。

转换方法参考 [Torch-TensorRT 2.10 quick start](https://docs.pytorch.org/TensorRT/v2.10.0/getting_started/quick_start.html) 与[动态形状说明](https://docs.pytorch.org/TensorRT/v2.10.0/user_guide/dynamic_shapes.html)。这是上述冻结短输入的测试路径，不是新增通用 TensorRT 后端支持声明。

## 测量口径

- 设备：`Thor1` / NVIDIA Thor，sm110；驱动 580.00、CUDA 13.0、PyTorch 2.10.0+cu130、TensorRT **10.13.3.9**、Torch-TensorRT **2.10.0**。沿用既有环境，未安装/升级系统依赖。
- `nvpmodel` 为 120W / mode 1，未改功耗或锁频；4 个 CPU 线程，独占测试锁，GPU 测试串行执行。
- 同日重新测量 PyTorch 基线和 TensorRT 路径；每个配置 2 次预热＋5 次固定输入计时，再测 ASR 中英各 10 条、TTS 中英各 5 条。CUDA 计时起止均同步。
- ASR WAV 文件读取在计时外；特征提取、音频 encoder、文本生成、文本解码在内。固定录音长 4.2039375 秒。
- TTS 参考音频读取/校验/哈希、语音生成、codec 及 CPU 波形返回均在内，重复请求命中相同说话人 embedding 缓存；不缓存生成结果。固定句为“你好，欢迎使用语音助手。”，speaker-only 克隆。
- 两种 TTS 路径均每请求 seed=0，Talker/Predictor 都采样：top-k=50、top-p=1、temperature=0.9，Talker repetition_penalty=1.05；生成上限 64。ASR 仍为 greedy、上限 64。
- RTF 为计时总和 / 音频总时长。RTF<1 表示离线处理快于音频播放，不代表流式首包延迟。

## 稳态完整请求

| 模型 / 路径 | 中位耗时 | 最小–最大 | 音频秒 | 生成步数 | RTF |
|---|---:|---:|---:|---:|---:|
| ASR PyTorch FP16 | 0.240 s | 0.237–0.243 s | 4.204（输入） | 11 | 0.057 |
| ASR TensorRT 主体 FP16 | **0.154 s** | 0.144–0.155 s | 4.204（输入） | 11 | **0.036** |
| TTS PyTorch BF16 | 2.574 s | 2.498–2.590 s | 2.320（输出） | 30 | 1.099 |
| TTS TensorRT 主体 BF16 | **1.057 s** | 1.055–1.067 s | 2.160（输出） | 28 | **0.490** |

ASR 对相同输入和相同 token 工作量的请求加速为 **1.56 倍**。TTS 固定文本请求为 **2.44 倍**；编译后的浮点计算差异使采样轨迹及结束位置发生变化，输出比基线短约 6.9%，因此 2.44 倍不能当作等 token 的算子加速比。按音频长度归一化的 RTF 改善约 **2.24 倍**，且两条路径都自然结束。

不同输入各一次的统计：

| 模型 / 路径 | 中文平均耗时 / RTF | 英文平均耗时 / RTF |
|---|---:|---:|
| ASR PyTorch | 0.317 s / 0.058 | 0.427 s / 0.088 |
| ASR TensorRT 主体 | 0.161 s / 0.030 | 0.198 s / 0.041 |
| TTS PyTorch | 2.462 s / 1.099 | 2.615 s / 1.097 |
| TTS TensorRT 主体 | 1.153 s / 0.500 | 1.215 s / 0.490 |

## 质量与一致性

ASR 正式运行的全部 27 组输出 token（包含预热、固定重复和 20 条不同录音）与同日 PyTorch 逐项相同。中文测试共 165 字、英文共 124 词：

| 路径 | 中文 CER / 1−CER | 英文 WER / 1−WER | 整句一致（中 / 英） |
|---|---:|---:|---:|
| ASR PyTorch、TensorRT | 1.82% / 98.18% | 5.65% / 94.35% | 9/10、4/10 |

TTS 的 10 条 TensorRT 输出全部重新经过冻结官方 CPU FP32 ASR 转写，评分规则与上一轮一致。同日 PyTorch 基线中 8 条波形与上一轮逐点相同，复用原评分；另外 2 条重新转写。因此本轮新评分共 12 条，没有用 TensorRT ASR 自评 TensorRT TTS。

| TTS 路径 | 中文 CER | 英文 WER | 整句一致（中 / 英） |
|---|---:|---:|---:|
| PyTorch BF16 | 0.00% | 0.00% | 5/5、5/5 |
| TensorRT 主体 BF16 | 0.00% | 0.00% | 5/5、5/5 |

两个模型所有正式样本均未触及 64-token 上限；TTS 同一进程内固定种子的预热/稳态波形逐点一致。编译时检查原层包装前后的形状和有限值、TensorRT 输出的形状/dtype/有限值；随机张量上的 MSE/max-abs 仅作为诊断保留在证据中，不能取代任务质量。TTS 生成波形不要求与 PyTorch bitwise 相同。

## TensorRT 路径各阶段

另开进程预热 2 次、测量 3 次，单位 ms。每个被测阶段都加 CUDA 同步，特别是 TTS 会放大数百次调用的同步成本；下面只用于定位瓶颈，正式总耗时使用前述无 hooks 测量。

| ASR 阶段 | 耗时 |
|---|---:|
| 特征提取 / 分词 | 6.54 |
| 音频卷积（PyTorch） | 1.72 |
| 音频 Linear（PyTorch） | 9.62 |
| 音频 attention / norm 等（PyTorch） | 14.96 |
| 文本 decoder（TensorRT，含边界准备） | 104.73 |
| 输出 head（PyTorch） | 22.62 |
| 其他 / 调度 | 7.19 |
| 带 hooks 总计 | 167.38 |

| TTS 阶段 | 耗时 |
|---|---:|
| 参考音频读取 | 0.58 |
| Talker decoder（TensorRT，含边界准备） | 175.23 |
| Code Predictor decoder（TensorRT，含边界准备） | 613.53 |
| 输出 head（PyTorch） | 41.83 |
| 生成调度与组合 | 501.27 |
| codec（PyTorch） | 50.85 |
| 其他 | 1.67 |
| 带 hooks 总计 | 1384.97 |

说话人缓存命中，稳态不再执行 speaker encoder。TTS 剩余主要开销是频繁的 Predictor 调用与通用生成调度；ASR 仍有 decoder、词表 head 和未编译音频 encoder 的成本。本轮没有继续更换生成算法或削减输出长度。

## 构建与加载

构建与首次调用均不混入稳态结果。构建列为成功引擎的 export＋compile 调用耗时之和，不含保存、权重加载或前期失败探索；文件大小为包含引擎的 `.ts` 归档，不是运行显存占用。

| 模型 | 引擎数 | 构建合计 | 归档大小 | 已有引擎时加载 | 两次预热 |
|---|---:|---:|---:|---:|---|
| ASR | 2 | 84.70 s | 2.196 GiB | 20.09 s | 0.642、0.156 s |
| TTS | 4 | 109.95 s | 2.588 GiB | 11.63 s | 1.861、1.070 s |

同日 PyTorch 加载 ASR 13.63 s、TTS 4.28 s。加载不含 Python 包导入，未清空文件缓存，不解释为冷盘启动性能。TensorRT 路径仍加载原模型，未优化常驻内存。引擎只在该 Thor 环境验证，不宣称跨设备/版本可移植，参见 [NVIDIA 引擎兼容性说明](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/engine-compatibility.html)。

## 文件与复现

- [机器可读证据](evidence/qwen3_audio_thor_tensorrt/summary.json)：逐次计时、完整 ASR token、TTS 转写评分、阶段耗时、编译诊断、归档大小和脚本 SHA256。
- Thor 原始目录：`/home/hx/rhinoforge_audio_20261009/tensorrt_20261010/`。保留 `trt_decoder.py`、`bench_asr.py`、`bench_tts.py`、日志、音频、引擎及 `asr-{baseline,trt,stages}`、`tts-{baseline,trt,stages}` 结果目录。模型资产继续使用上一级 `assets/`。
- R1 独立 CPU FP32 评分：`/home/agi/rhinoforge_audio_runtime/tensorrt_thor_20261010/`。只上传少量生成音频和清单，没有替换 RPU 代码、权重或部署。

使用 `/home/hx/miniconda3/envs/3.12/bin/python`，`TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1`，持有上一级 `gpu.lock`。ASR 的 PYTHONPATH 为上一级 `python:deps` 的绝对路径；TTS 再在最前面加入 `official/tts-ref-deps:official/Qwen3-TTS` 的绝对路径，与[上一轮](qwen3_audio_thor_stages.zh.md#文件与复现)相同。

运行 `bench_asr.py asr --output NEW_DIR` 或 `bench_tts.py tts --output NEW_DIR`；加 `--baseline` 走 PyTorch，加 `--stages` 单独诊断各阶段。输出目录必须不存在。默认加载已保存引擎，避免重复构建；重建只应使用与记录匹配的冻结资产与环境，不能把此测试脚本当作任意模型的通用加载器。

未测试 TensorRT-LLM、FP8/INT8/INT4、CUDA Graph、流式首包、并发、长音频/长文本或完整公开测试集；没有人工 MOS、音色相似度或自然度评分。有限样本上的转写一致只能验证文字内容，不能认证整体音质。本结果不代表 Thor 的最高性能，也不是完整模型所有阶段均 TensorRT 化。
