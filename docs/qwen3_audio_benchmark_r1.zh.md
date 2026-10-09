# Qwen3 语音模型：R1 速度与识别质量评测

评测日期：2026-10-09。模型实现为 `feat/qwen3-audio-rpu` 分支的 `7fd9eb4`，模型和运行时身份沿用[迁移报告](qwen3_audio_r1.zh.md)。本次不修改模型实现。以下是固定协议的小样本实测，不能直接当作完整 AISHELL/LibriSpeech 榜单成绩。

## 方法与指标

- R1，batch=1、串行请求、8 RPU cores、PyTorch CPU 线程数 4，关闭 profiler。ASR/TTS 各用独立进程；RPU 测量与 CPU 对照串行执行，无其他模型进程竞争。
- ASR：CPU/RPU FP16；最多生成 64 token，greedy。CPU 对照用同一 checkpoint 的官方 ASR 源码 `7c6daf77a2421100f5fb066495372c00129d39ff`、隔离的 Transformers 4.57.6、FP16。混合路径用仓库固定的 Transformers 5.3.0，组件语义一致性已在迁移阶段核验。
- TTS：RPU FP16 Talker/Predictor/heads，CPU FP32 speaker/codec/glue；中英文、non-streaming、speaker-embedding-only、greedy、最多 64 token。使用与迁移阶段相同的公开 `clone.wav` 参考音频。
- **稳定速度**：每个模型选清单第一条输入，在相同进程完整预热 2 次，随后测 5 次；核验输出一致、固定 Graph 数量及递增 REPLAY。给出中位数、均值和范围，不对 5 个样本报告 P95/P99。
- **不同输入**：ASR 20 条、TTS 10 条，每条测一次完整请求。ASR 不同长度会清理旧 Graph，因此该组耗时包含需要的重新 BUILD，与固定输入的稳定速度分开。
- 计时覆盖完整公开 API，直至 CPU 文本或波形返回。ASR 输入 WAV 解码放在计时外，音频特征提取在计时内；TTS 的参考音频读取、speaker 提取、codec 编解码都在 API 内，计入耗时。结果评分、Graph 状态查询、输出 WAV 写盘不计时。
- “加载”只计 `from_pretrained`，包含其配置检查、权重读取/转换和上传，排除 Python/包导入；是新进程加载，**没有清空操作系统文件缓存**。CPU 参考加载不执行适配器的完整权重哈希，不能把加载耗时差异直接解释为硬件优劣。

`RTF = 总推理耗时 / 总音频时长`。ASR 分母为输入音频，TTS 为生成音频；RTF < 1 才表示该离线请求路径的平均处理速度快于音频播放。倒数表示每秒墙钟时间处理/生成多少秒音频，不等同于流式首包延迟。

识别质量采用最短编辑距离：`CER/WER = (替换 S + 删除 D + 插入 I) / 参考字符或词数 N`。同时报告 `1−CER/WER`（含插入惩罚的正确率）及归一化后整句完全一致率。按整组总错误/总参考长度汇总，不平均每句错误率。

归一化规则预先固定：Unicode NFKC、小写；中文保留文字和数字，移除标点/空白；英文去掉撇号，其他标点转为空白，按词分割。不做数字口语化、缩写展开、繁简转换、同义词替换或事后修改参考。分词/书写差异同样计入错误。

## ASR 样本

| 语言 | 来源 | 条数 | 说话人数 | 音频合计 | 时长范围 |
|---|---|---:|---:|---:|---:|
| 中文 | AISHELL-1 test，第 1/3 个 Parquet 分片的固定子集 | 10 | 7 | 54.292 秒 | 3.526–7.379 秒 |
| 英文 | LibriSpeech ASR dummy，clean/validation 固定子集 | 10 | **1** | 48.685 秒 | 1.740–7.100 秒 |

中文下载自 [carlot/AIShell](https://huggingface.co/datasets/carlot/AIShell/tree/692ef58020d79b21f54eb25b15a4813d4f9650d7)，revision `692ef58020d79b21f54eb25b15a4813d4f9650d7`。分片共 2,392 条，2,268 条满足单声道 16 kHz、0.5–8 秒；它们的文本全部与 [AISHELL 官方仓库](https://huggingface.co/datasets/AISHELL/AISHELL-1/tree/bbe295d530192a4cd41644b711c9aecd087df653)原始 transcript 去空白后核对一致。AISHELL 来源和许可见 [OpenSLR 33](https://www.openslr.org/33/)。

英文来自 [HF 官方测试数据](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/tree/5be91486e11a2d616f4ec5db8d3fd248585ac07a)，revision `5be91486e11a2d616f4ec5db8d3fd248585ac07a`。73 条中 55 条满足时长范围；语料源头与许可见 [LibriSpeech / OpenSLR 12](https://www.openslr.org/12/)。它是开发测试小集，英文仅一位说话人，不能推断多人或不同口音的水平。

两种语言分别将满足输入范围的完整录音按原始 ID 排序，取索引 `floor(i×(N−1)/9)`，`i=0…9`。不裁剪音频、不按文本难易或模型结果筛选。清单保留原始 ID、参考文本、时长、来源版本和输入文件哈希。

## ASR 结果

固定输入是 4.204 秒中文录音（`zh-BAC009S0764W0121`）。两条路径分别预热 2 次，再各测 5 次：

| 指标 | RPU＋CPU 混合 | 官方 CPU FP16 对照 |
|---|---:|---:|
| 新进程加载（不含包导入） | 23.454 秒 | 2.316 秒 |
| 稳定请求耗时中位数 | 10.465 秒 | 17.462 秒 |
| 稳定请求耗时均值 | 10.491 秒 | 17.470 秒 |
| 最小–最大 | 10.453–10.598 秒 | 17.447–17.498 秒 |
| 总耗时/总输入音频 RTF | 2.496 | 4.156 |
| 每秒墙钟时间处理音频（秒） | 0.401 | 0.241 |

在该固定输入、相同 4 个 CPU 线程配置下，按耗时中位数计算，混合路径相对官方 CPU FP16 路径加速 **1.67 倍**。这是两套完整实现的实测对比，不是纯 RPU 算子加速比，也不是充分调优后的 CPU 最佳性能。混合路径 RTF **2.496 > 1**，仍未达到实时识别。

不同输入的速度如下；每条一次，不与上述固定输入重复测量混为一组：

| 语言 | 路径 | 条数 | 耗时中位数 | 耗时均值 | 总耗时/总输入音频 RTF |
|---|---|---:|---:|---:|---:|
| 中文 | RPU＋CPU | 10 | 11.354 秒 | 10.600 秒 | 1.952 |
| 中文 | CPU FP16 | 10 | 19.700 秒 | 19.069 秒 | 3.512 |
| 英文 | RPU＋CPU | 10 | 11.366 秒 | 10.598 秒 | 2.177 |
| 英文 | CPU FP16 | 10 | 19.694 秒 | 18.933 秒 | 3.889 |

识别率按总字符/词数加权。下表的“正确率”定义为 `1−CER/WER`，不等同于整句正确率：

| 语言 | 路径 | 参考单位 N | S / D / I | 错误率 | 正确率 | 整句完全一致 |
|---|---|---:|---:|---:|---:|---:|
| 中文 | RPU＋CPU | 165 | 3 / 0 / 0 | CER **1.82%** | **98.18%** | 9/10（90%） |
| 中文 | CPU FP16 | 165 | 3 / 0 / 0 | CER **1.82%** | **98.18%** | 9/10（90%） |
| 英文 | RPU＋CPU | 124 | 5 / 2 / 0 | WER **5.65%** | **94.35%** | 4/10（40%） |
| 英文 | CPU FP16 | 124 | 5 / 2 / 0 | WER **5.65%** | **94.35%** | 4/10（40%） |

20 条音频中，RPU 混合路径与 CPU FP16 对照的原始转写文本有 **20/20** 完全相同，生成 token 序列有 **20/20** 完全相同。RPU ASR 没有触及 64-token 上限。

RPU 错误涉及以下 7 条输入；完整参考和两条路径的输出保存在证据 JSON：

| 输入 ID（清单中的 ID） | 参考 → RPU 输出差异 | S / D / I |
|---|---|---:|
| zh-BAC009S0766W0437 | “间”→“斤”两处；“增”→“升”一处 | 3 / 0 / 0 |
| en-1272-128104-0000 | MISTER → Mr. | 1 / 0 / 0 |
| en-1272-128104-0013 | MISTER → Mr. | 1 / 0 / 0 |
| en-1272-135031-0020 | ANN → Anne | 1 / 0 / 0 |
| en-1272-141231-0003 | “instant of panic” → “instant panic” | 0 / 1 / 0 |
| en-1272-141231-0026 | BRION → Brienne | 1 / 0 / 0 |
| en-1272-141231-0032 | “thrust it aside” → “thrusted aside” | 1 / 1 / 0 |

这些错误既包含量词/用词偏差，也包含缩写或人名拼写差异；统一按预先固定规则计分，没有为提高分数而删除样本或更改归一化。

## TTS 结果

TTS 使用 5 条中文和 5 条英文人工编写的短句，涵盖欢迎、物体放置、避让、返航充电及出发提醒。该组文本是功能小集，不是标准音质测试集。

固定输入“你好，欢迎使用语音助手。”生成 2.48 秒音频。预热耗时分别为 5.434、5.328 秒；随后 5 次测量：

| 项目 | 结果 |
|---|---:|
| 新进程模型加载（不含包导入） | 32.299 秒 |
| 稳定请求耗时中位数 | **5.237 秒** |
| 稳定请求耗时均值 | 5.258 秒 |
| 最小–最大 | 5.213–5.359 秒 |
| 总耗时/总音频时长 RTF | **2.120** |
| 每秒墙钟时间生成音频 | 0.472 秒 |

这条混合推理路径尚未达到实时生成。不同短句的实测如下，均包含参考音频处理及 CPU codec：

| 语言 | 请求数 | 请求耗时中位数 | 请求耗时均值 | 生成音频合计 | 总耗时/总音频 RTF |
|---|---:|---:|---:|---:|---:|
| 中文 | 5 | 5.799 秒 | 6.033 秒 | 15.12 秒 | 1.995 |
| 英文 | 5 | 6.607 秒 | 7.321 秒 | 19.60 秒 | 1.868 |

内容一致性用**官方 Qwen3-ASR-0.6B CPU FP32**独立转写，24 kHz 输出以 `resample_poly(2,3)` 转为 16 kHz 后评分；重采样和识别耗时均不计入 TTS 生成速度。没有使用待测 RPU ASR 作评分器。

| 语言 | 参考单位 | S / D / I | CER / WER | 整句完全一致 |
|---|---:|---:|---:|---:|
| 中文 | 49 字 | 0 / 0 / 0 | CER **0%** | **5/5** |
| 英文 | 35 词 | 0 / 0 / 0 | WER **0%** | **5/5** |

该结果只说明这 10 句经上述识别器得到完整文本，不是人工听评，也不证明音色相似度、自然度或没有多余静音。中文 1 条、英文 2 条达到 64-token 上限；这三条的文本仍完整，但未验证其自然结束，不能据此承诺更长句子不会截断。RTF 使用完整返回波形时长，未裁剪尾部静音；不能将其直接解释为有效讲话速度。没有对这些样本重试或从统计中删除。

## 运行检查

ASR 固定输入的 5 次测量保持 Graph size=2，REPLAY 为 31→42→53→64→75；每次音频 encoder 的 111 个 Linear 都有 RPU 调用。TTS 的 Talker/Predictor 各保持一个 Graph，5 次 Talker REPLAY 为 146→195→244→293→342，Predictor 为 1487→1983→2479→2975→3471；16 个输出头每次均调用 RPU。全部检查的 Graph invariant 为 true，固定输入重复输出一致。主机端评分/统计测试 2 项通过，持久化后的 CPU 参考源码与隔离依赖导入检查通过。ASR 混合路径共 27 次请求、TTS 共 17 次请求完成，未剔除失败样本。

## 复现与证据

结果、输入音频和生成波形保存在 R1：`/home/agi/rhinoforge_audio_runtime/benchmark_20261009/`。执行时必须独占已有设备锁，CPU 对照也应与 RPU 计时串行。

仓库中保留[汇总及逐条结果](evidence/qwen3_audio_benchmark_r1/summary.json)、[ASR 输入清单](evidence/qwen3_audio_benchmark_r1/asr-manifest.json)、[TTS 文本清单](evidence/qwen3_audio_benchmark_r1/tts-manifest.json)及[数据筛选记录](evidence/qwen3_audio_benchmark_r1/dataset-provenance.json)。脚本和参考音频身份见[复现文件记录](evidence/qwen3_audio_benchmark_r1/reproduction-files.json)。原始 `report.json`、`runs.jsonl` 在 R1 的 `asr-rpu/`、`asr-cpu-fp16/`、`tts/`、`tts-score/` 子目录，生成 WAV 在 `tts/`，日志在评测根目录；不把整份数据集和波形放进代码仓库。

公开接口测量脚本：[benchmark_qwen3_audio.py](../examples/benchmark_qwen3_audio.py)。它固定 2 次预热、5 次测量、最多 20 条清单输入；评分函数有[独立单元测试](../tests/test_audio_benchmark_metrics.py)。清单路径与 `--output` 要按实际环境设置，输出目录必须尚不存在。

在 R1 已部署环境中，以下命令可重新评测 ASR；新结果目录不得与原结果重名：

```bash
root=/home/agi/rhinoforge_audio_runtime
bench="$root/benchmark_20261009"
runtime=/home/agi/rhinoforge_bench_20261009/RhinoForge-runtime-v1.1.0
python=/home/agi/rhinoforge_bench_20261009/env/bin/python
sudo -n env TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1 \
  PYTHONPATH="$root/python:$root/deps" \
  RPU_KERNEL_LIB_PATH="$runtime/rhinoOpLib_rhinoforge_v1.1.0.ref" \
  LD_LIBRARY_PATH="$runtime/launch/lib" \
  flock -n /mnt/ssd_data/rhino-deploy-platform/rpu.lock \
  "$python" "$bench/benchmark_qwen3_audio.py" --task asr \
  --checkpoint /home/agi/rhinoforge_audio_models/Qwen3-ASR-0.6B \
  --manifest "$bench/asr-manifest.json" --output "$bench/asr-rerun"
```

测 TTS 时沿用同一环境/锁，将最后的脚本参数换成：

```bash
--task tts \
--checkpoint /mnt/user_data/rhinoforge_audio_models/Qwen3-TTS-12Hz-0.6B-Base \
--source "$root/source/qwen_tts" \
--manifest "$bench/tts-manifest.json" --output "$bench/tts-rerun"
```

CPU 参考脚本 `reference_audio_benchmark.py`、冻结上游代码 `reference_source/` 和隔离依赖 `reference_deps/` 均保留在 R1 的评测目录。该副本只将首次执行脚本的临时源码路径改为持久路径，推理和评分代码不变。原执行版本也保留为 `reference_audio_benchmark.original.py`。CPU 对照命令：

```bash
sudo -n env TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1 \
  PYTHONPATH="$bench/reference_deps:$root/deps" \
  flock -n /mnt/ssd_data/rhino-deploy-platform/rpu.lock \
  "$python" "$bench/reference_audio_benchmark.py" \
  --manifest "$bench/asr-manifest.json" --output "$bench/asr-cpu-rerun" \
  --dtype float16 --steady
```

TTS 转写评分使用 `tts-score-manifest.json`、新输出目录和 `--dtype float32`，不传 `--steady`；重新生成波形时先将评分清单的音频路径指向新波形。不要把 Transformers 4.57.6 安装进正式的 5.3.0 环境。

未测完整公开测试集、噪声/远场录音、英文多说话人、长音频、并发、TTS MOS 和音色相似度。没有测 TTS 同文本 CPU 稳定性能，因此不声称 TTS 相对 CPU 的加速倍数。本报告按本次任务要求记录实测结果，不扩展原模型的输入或质量支持范围。
