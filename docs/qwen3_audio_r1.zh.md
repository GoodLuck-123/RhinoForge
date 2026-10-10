# Qwen3 ASR / TTS：当前推理状态与使用说明

本页汇总 `feat/qwen3-audio-rpu` 的最终实现、运行限制和最近实测结果。
这是限定输入范围内的 CPU/RPU 混合推理接口，不是通用语音质量或实时性能认证。
两个接口独立于 CausalLM/VLM registry 和 `run_model.py`。

## 模型与环境

| 项目 | 固定版本 |
|---|---|
| ASR checkpoint | `Qwen/Qwen3-ASR-0.6B`，`5eb144179a02acc5e5ba31e748d22b0cf3e303b0` |
| ASR 参考源码 | `QwenLM/Qwen3-ASR`，`7c6daf77a2421100f5fb066495372c00129d39ff` |
| TTS checkpoint | `Qwen/Qwen3-TTS-12Hz-0.6B-Base`，`5d83992436eae1d760afd27aff78a71d676296fc`，含 `speech_tokenizer` |
| TTS 参考源码 | `QwenLM/Qwen3-TTS`，`022e286b98fbec7e1e916cb940cdf532cd9f488e` |
| R1 | Linux aarch64，24 CPU，61 GiB 内存，8 RPU cores |
| 依赖 | Python 3.12、PyTorch 2.10.0、Transformers 5.3.0；安装项目 `audio` extra |
| Runtime | RhinoForge-runtime-v1.1.0；Launch 1.0.0，commit `613f2da`；配套 operator asset 及 `.kernels` |

加载器在完整权重加载前核对 checkpoint、配置和 tokenizer 的固定哈希。
ASR 身份见 [`qwen3_asr.py`](../python/rpu_backend/api/qwen3_asr.py)，TTS 见
[`_manifest.py`](../python/rpu_backend/adapters/qwen3_tts/_manifest.py)。不接受其他大小、
精度或 tokenizer 覆盖文件。转换失败后必须关闭并重新加载干净模型，不能重复转换权重。
权重、参考音频和 restricted runtime 不随仓库发布。

## 输入与执行范围

| 模型 | 输入限制 | 生成方式 |
|---|---|---|
| ASR | batch 1，单声道 16 kHz，0.5–8 秒 | greedy，最多 64 token；无 context、时间戳和流式接口 |
| TTS Base | batch 1，中/英文；文本 ≤128 字符且 ≤32 tokenizer token；参考音频单声道 24 kHz、0.5–20 秒 | 2–64 token，非流式 greedy，仅 speaker embedding 克隆；Talker prompt 加生成上限 ≤128 位置 |

TTS 必须提供参考音频，不提供 ICL transcript、CustomVoice 或 VoiceDesign 模式。
达到生成上限可能截断句子。CPU 胶水保持 FP32，RPU 主体为 FP16；codec 前段的
FP16 配方是明确的精度变化，不能当作整个 codec 的 FP32 等价迁移。

## 当前各阶段设备

| 模型 | 阶段 | 设备与执行方式 |
|---|---|---|
| ASR | 音频读取、mel 特征、分块 | CPU |
| ASR | 三层卷积、层间 GELU、末尾布局 | CPU，channels-last；FP16 舍入权重以 FP32 卷积，再转 FP16 |
| ASR | conv_out、位置相加、有效行选择 | RPU；有效行是连续前缀 |
| ASR | 18 层音频 attention/norm/GELU/Linear、ln_post、proj1、ERF GELU、proj2 | 单张连续 RPU Graph，FP32 Linear 累加；含 110 个逻辑 Linear，加图外 conv_out 共 111 个 |
| ASR | prefill 文本 embedding 和音频特征组装 | RPU |
| ASR | 28 层文本 decoder、词表输出头 | RPU；decode 输出头并入 Graph，prefill 只投影最后位置 |
| ASR | 单 token embedding、finite 检查、argmax、文本解码 | CPU；查表后上传 |
| TTS | Talker 28 层、Predictor 5 层、16 个输出头 | RPU；输出头直接消费当前 RPU hidden |
| TTS | codec 输入/输出投影、8 层 Transformer | RPU FP16，FP32 Linear 累加；native 缺少显式 codec 模式时保留 CPU |
| TTS | codec 码本查表/求和、卷积/上采样、SnakeBeta、波形输出 | CPU FP32；适用卷积使用矩阵乘及 overlap-add |
| TTS | embedding、参考音频处理、speaker encoder、生成控制和选词 | CPU FP32；只缓存最新参考音频的 speaker embedding |

保留 CPU 的原因：ASR 现有独立卷积和单 token RPU 查表路径没有实测收益；TTS
后段 codec 的 FP32 运算、通道对齐与长序列 SPM 容量尚无完整原生 RPU 路径。
已知边界明确选择 CPU，未知执行错误仍向调用方报告，不进行任意异常的静默重试。

## Graph、资源和公共代码隔离

- ASR 保留一张音频图及最多两张文本图。尾部 GELU/proj2 使用原音频图，不增加图数。
  proj1 中间存储由模型持有，最终输出每次独立分配并更新 mutable DMA 地址。
- ASR 的 SigLIP 复用只在显式 `siglip_set_audio_mode` 下启用，检查冻结 encoder 几何；
  非音频模型维持原 TANH GELU、投影宽度和布局 hash。ASR 加载前检查 audio-mode、
  audio-projection 和 assembly 接口，缺少接口会拒绝，需重编译匹配的 native。
- TTS codec 的逐 head K RoPE 仅由 `causal_decoder_set_tts_codec_mode(handle)` 开启。
  native 默认关闭，检查 8 层、hidden 512、16 个 Q/KV head、head_dim 64、无 Q/K norm
  等专用几何。其他 decoder 保持原 K RoPE 参数，不自动按几何切换路径；codec 模式
  参与布局 hash，开启后禁止替换权重。
- codec 在 FP32 中将 LayerScale 折入投影，再转换 FP16；RoPE 使用原 FP32 频率。
  仅接受 1–63 帧，严格低于上游 72 帧滑窗。旧 native 没有显式 setter 时，前段
  fallback 到 CPU，`execution_stats()` 返回原因；旧的全局能力标记不再代表可用。
- TTS 冷启动用已有 owner policy 申请 5 个 arena：Talker prefill/decode 两个、
  Predictor 一个、codec 一个、即时算子一个。框架全局默认预算不变。
- 同长度 BUILD 一次后 REPLAY，长度变化淘汰对应旧签名。codec、Talker、Predictor
  的 Graph 共用同一 owner policy；销毁时先清 Graph，再释放 handle 和权重。
- 一个进程只有一个存活的 RPU 模型/策略；请求、统计和关闭串行。两类 API 支持
  `with`、`close()` 和 `execution_stats()`，返回结果有独立存储。
- speech loader 在首次设备分配前选择 caching allocator；已冻结为不兼容 allocator
  或 arena 预算的进程需要重启。CPU 线程数由调用方/CLI 设置，API 不隐式更改它。
- `audio` 是可选依赖；TTS 兼容补丁只作用于校验后的独立源码副本，不修改全局
  Transformers 类、其他 adapter 或已安装的 upstream 包。

## 最近完成的硬件测量

以下为 2026-10-10、`e1f890f` ASR Graph 尾部版本和相同 TTS codec 配方的已有记录。
本次显式 codec 模式已完成 native 重编译和下述定向硬件回归；未重新跑完整性能
及识别率语料。因此下表保留最近一次标准测量，不作为新二进制的重新计时结果。

R1：8 RPU cores；ASR 4 CPU 线程、TTS 8 CPU 线程；2 次预热后测 5 次，
不含加载，包含完整 CPU/RPU 请求。ASR 文件读取在计时外；TTS 参考读取在计时内。

| 模型 | 固定工作量 | 完整请求中位数 | RTF |
|---|---|---:|---:|
| ASR | 4.20394 秒音频、11 个生成 token | 252.968 ms（交替诊断）；260.703 ms（独立加载顺序测量） | 约 0.060–0.062 |
| TTS | 同一中文短句、32 步、2.48 秒音频 | 2.69821 s（同模型配对）；2.71871 s（标准 benchmark） | 约 1.09–1.10 |

ASR 不同测量方式存在毫秒级波动，不能把最低值当作稳定生产延迟。
音频 encoder 加图尾均值 6.349–6.383 ms；交替测量中的 CPU 卷积均值 75.175 ms。
Graph 尾部局部收益约 0.8 ms，不能推导为整句大幅提速。

TTS 同模型阶段计时均值如下，含各段传输和调度；不能与完整请求中位数直接相加比较。

| 阶段 | 均值 |
|---|---:|
| Talker | 298.22 ms |
| Predictor | 1060.45 ms |
| 16 个输出头 | 206.74 ms |
| codec 前段 | 4.01 ms |
| codec 后段 | 924.34 ms |
| 参考处理、生成控制等其余工作 | 206.76 ms |
| 完整诊断请求 | 2700.52 ms |

Thor 最近 TensorRT 主体结果：ASR 154.21 ms，TTS 1.057 s。它们是文本 decoder /
Talker / Predictor 使用 TensorRT，其余步骤使用 PyTorch 的完整请求；不是全模型
TensorRT。Thor TTS 使用 BF16 和采样，输出 28 步、2.16 秒；与 R1 的精度、采样和
步数不匹配，不能据此计算等工作量硬件加速倍数。本次没有重测 Thor。

已有质量与生命周期证据：

- ASR 20 条短录音，中文 CER 1.818%（165 字），英文 WER 5.645%（124 词）；
  Graph 尾部版本 20/20 输出 token 与其前一正式版一致。
- TTS 10 条生成音频经独立官方 CPU FP32 ASR 转写，10/10 内容一致；中文 49 字
  CER 0%，英文 35 词 WER 0%。三句达到生成上限；未评价自然度或音色相似度。
- ASR 0.5/8 秒边界、同长换内容、旧输出持有、长度往返与重载通过；音频图 1 项、
  文本图最多 2 项，replay 递增且 invariant 正常。
- TTS codec 1/31/63 帧及整模型 Graph 共存、换输入、独立波形、关闭重载通过。
  codec FP16 前段相对原 FP32 波形的固定样本相对 L2 为 0.4851%，不声称逐位等价。

## 当前版本回归结果

2026-10-10 在 R1 独立目录构建 Release native，持设备锁串行验证，现有部署未替换。
回归结果如下：

- 主机定向测试 161 项通过；native 编译和加载通过。
- ASR：中文、同长英文、0.5/8 秒边界及返回原输入，5 次请求的 token 均与原记录一致；
  关闭重载结果一致，音频 Graph 1 项、文本最多 2 项，invariant 正常。
- TTS：同长换文本、中英文切换、返回原文本及关闭重载通过；固定中文样本
  59,520 个波形采样与先前 FLOAT WAV 逐样本一致。Talker/Predictor/codec 的
  Graph 数分别为 2/1/1，codec 确认在 RPU，无 CPU fallback。
- codec：1/31/63 帧、换内容、旧输出持有、确定性重放和关闭重载通过；缓存大小为 1。
  相对匹配 FP16 参考的数值差异仅作观测，不用任意误差阈值认证质量。
- 公共 decoder：未开启 codec 模式的固定 GQA 算子用例，KV_FIRST 与 SEQUENTIAL
  各运行 3 次（1 BUILD + 2 REPLAY），输出与更改前备份 native 逐元素一致；
  错误几何开启 codec 模式被拒绝。此项是共享路径回归，不是其他完整模型的认证。

使用更新后的 Python 代码时须配套重编译 native；旧 native 缺少显式 setter 时，
TTS codec 前段会退回 CPU。未重新运行全模型矩阵、Thor、完整语料和并发压力。

## 安装与运行

先按 [runtime 资产](runtime_assets.md) 配置匹配的 Launch/operator 路径，再从当前
源码构建并安装 `audio` extra（参见[构建说明](development_build.md)）。ASR 与 TTS
应分别在独立进程运行；无需将上游 Transformers 4.57.x 安装进正式 5.3.0 环境。

ASR 直接使用 Transformers 内已有组件，无需导入官方 ASR 源码。TTS 需要显式准备已审阅的外部源码：

```bash
git clone https://github.com/QwenLM/Qwen3-TTS /path/Qwen3-TTS
git -C /path/Qwen3-TTS checkout 022e286b98fbec7e1e916cb940cdf532cd9f488e
python -m rpu_backend.adapters.qwen3_tts \
  --source /path/Qwen3-TTS --output /path/qwen3-tts-prepared
cp /path/Qwen3-TTS/LICENSE /path/qwen3-tts-prepared/LICENSE
```

prepare 仅复制固定哈希的 15 个源码文件并应用可审阅补丁，不导入原始源码、不修改原 checkout、不安装上游包。输出目录必须尚不存在；加载时再检查 prepared 代码清单和哈希。`source` 参数指向生成的 `qwen_tts` 子目录。保留上游许可证。上游导入链会加载 25Hz 模块的依赖，即使当前只使用 12Hz codec；`audio` extra 因而包含 ONNX Runtime、TorchAudio 和 python-sox。未安装 SoX 可执行文件可能产生上游警告，本次 12Hz 路径不调用它。

```bash
python examples/qwen3_asr.py \
  --checkpoint /models/Qwen3-ASR-0.6B --audio /audio/mono-16k-pcm16.wav
python examples/qwen3_tts.py \
  --checkpoint /models/Qwen3-TTS-12Hz-0.6B-Base \
  --source /path/qwen3-tts-prepared/qwen_tts \
  --reference-audio /audio/mono-24k.wav \
  --text '你好，欢迎使用。' --language Chinese --output /output/hello.wav
```

ASR Python API 接受 CPU 浮点数组；CLI 只读 PCM16 WAV。TTS CLI 不覆盖已存在输出，输出浮点 WAV。Python 用法和状态接口见 [API reference](api_reference.md#bounded-speech-apis)。

## 性能和内容评测

使用 [`benchmark_qwen3_audio.py`](../examples/benchmark_qwen3_audio.py)，准备本地 JSON
清单：ASR 行含 `id/language/audio/text`，TTS 行含 `id/language/text/reference_audio`。
`language` 为 `Chinese` 或 `English`，清单为 1–20 条唯一 ID。音频路径需指向本地文件。

```bash
python examples/benchmark_qwen3_audio.py --task asr --threads 4 \
  --checkpoint /models/Qwen3-ASR-0.6B \
  --manifest /audio/asr-manifest.json --output /results/asr-new
python examples/benchmark_qwen3_audio.py --task tts --threads 8 \
  --checkpoint /models/Qwen3-TTS-12Hz-0.6B-Base \
  --source /path/qwen3-tts-prepared/qwen_tts \
  --manifest /audio/tts-manifest.json --output /results/tts-new
```

输出目录必须尚不存在。持设备锁串行执行，固定输入预热 2 次、计时 5 次，再遍历
清单。加载单独计时，结果检查和 WAV 写盘在请求计时外。保存逐次耗时、RTF、生成
长度、Graph 统计及独立输出检查。ASR 评分先 NFKC/casefold 归一化；中文按字符、
英文按单词计算总编辑错误数 / 总参考长度，不平均逐句错误率。
TTS 内容评分还需独立 ASR 转写，不能把生成成功自动记为内容正确。

主机回归入口：`tests/test_qwen3_asr.py`、`tests/test_qwen3_tts*.py`、
`tests/test_causal_decoder_no_qknorm_rope.py` 和 `tests/test_audio_benchmark_metrics.py`。
本次隔离改动的定向主机检查共 161 项通过，覆盖默认/codec RoPE 分支、ASR/TTS
输入和生命周期、旧 native fallback、SigLIP 规划及公共 decoder preload 路由。
本次 native 与板卡验证范围见上文。未测长音频、多语言、MOS/音色相似度和实时
压力，不从这组短输入结果扩展模型支持范围。
