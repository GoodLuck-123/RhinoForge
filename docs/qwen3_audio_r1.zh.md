# Qwen3 ASR / TTS：R1 混合推理评估与迁移

首轮先用仓库 `rhinoforge-port` 完成真实权重运行评估及独立复核，两个模型均归类为 `Adapter-only`，随后才从 `c5712839cecf11cef942876765ed35f1e5b9e261` 创建 `feat/qwen3-audio-rpu` 分支实施。首轮沿用现有 RPU 算子、decoder 和 Graph。后续 [ASR 连续驻留优化](qwen3_asr_resident_r1.zh.md) 增加了小范围 C++ host 扩展，需要重编译 native；没有新增设备程序。

这是有限输入范围内的运行验证，不是通用语音质量认证。ASR 是语音识别；TTS Base 是借助参考音频生成短句，必须提供参考音频。

## 冻结模型与运行环境

| 项目 | 精确版本 |
|---|---|
| ASR checkpoint | `Qwen/Qwen3-ASR-0.6B`，`5eb144179a02acc5e5ba31e748d22b0cf3e303b0` |
| ASR 参考源码 | `QwenLM/Qwen3-ASR`，`7c6daf77a2421100f5fb066495372c00129d39ff` |
| TTS checkpoint | `Qwen/Qwen3-TTS-12Hz-0.6B-Base`，`5d83992436eae1d760afd27aff78a71d676296fc`，含 `speech_tokenizer` |
| TTS 参考源码 | `QwenLM/Qwen3-TTS`，`022e286b98fbec7e1e916cb940cdf532cd9f488e` |
| R1 | Linux aarch64，24 CPU，61 GiB 内存，8 RPU cores |
| 正式依赖 | Python 3.12.13、PyTorch 2.10.0+cpu、Transformers 5.3.0；音频依赖见 `pyproject.toml` 的 `audio` extra |
| Launch | 1.0.0，commit `613f2da`；SDK `EA_R1SDK_2025_51T-cf33df86`，API `2026-06-17` |
| Operator assets | `RhinoForge-runtime-v1.1.0/rhinoOpLib_rhinoforge_v1.1.0.ref` 及配套 `.kernels` |
| Native 构建 | 首轮为上述 base commit，wheel SHA256 `9a72cdb8228d7372332d4b301d47e8e72a5c442a55ac567d4250b95b3cd0e782`；连续驻留版本的 native 身份见后续 ASR 报告 |

checkpoint 配置、分词器和原始权重 SHA256 均由加载器固定校验。ASR 身份清单在 [`qwen3_asr.py`](../python/rpu_backend/api/qwen3_asr.py)，TTS 在 [`_manifest.py`](../python/rpu_backend/adapters/qwen3_tts/_manifest.py)。不接受改名模型、其他精度、其他大小或分词器覆盖文件。校验在完整权重加载及转换前执行；转换失败需关闭后重新加载干净模型。

模型来源：[ASR](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/5eb144179a02acc5e5ba31e748d22b0cf3e303b0)、[TTS Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/tree/5d83992436eae1d760afd27aff78a71d676296fc)。权重与官方参考音频分别取得，不随仓库发布。

## 主体计算位置与范围

| | RPU | CPU | 当前输入限制 |
|---|---|---|---|
| ASR | FP16，28 层文本 decoder、输出头；音频 18 层 attention/norm/ERF GELU/残差及 111 个 Linear，其中 109 个 Linear 与编码器连续执行 | 卷积用 FP16 权重提升至 FP32 计算，输出回 FP16；卷积后激活、布局/位置处理、最终两投影间激活、embedding、音频预处理 | batch 1，单声道 16 kHz，0.5–8 秒，最多生成 64 token |
| TTS | FP16，28 层 Talker、5 层 Code Predictor、16 个输出头 | FP32 embedding/text projection、speaker encoder、音频 codec、组合和采样 | batch 1，中/英文，非流式、greedy、仅 speaker embedding 克隆；文本 ≤128 字符且 ≤32 tokenizer token；参考音频单声道 24 kHz、0.5–20 秒；生成 2–64 token |

ASR 初始适配（输出头尚在 CPU）的官方中文样本按 Linear/Conv2d 实际调用统计，RPU 占 **66.59%** MAC；该统计不含 QK/AV、norm、激活或数据搬运，不是性能占比。TTS 的自回归生成主体全部在 RPU；CPU codec 仍有较大计算量，不能将此描述为整个冷请求的大多数 MAC 均在 RPU。

ASR 不提供 context、时间戳或流式识别。TTS 不提供 ICL transcript 模式、CustomVoice/VoiceDesign、流式或批量生成。Talker prompt 加生成上限必须不超过 128 位置；短 prompt 沿已验证的 decode 路径逐 token 处理。生成长度上限可能截断句子，建议先使用短句。

一个进程只允许一个存活的 RPU 模型/策略，ASR/TTS 共用该所有权规则。请求、状态查询和关闭必须串行。`with` 或 `close()` 先释放 Graph，再释放模型；返回结果不引用可被下一次 REPLAY 改写的存储。若底层资源清理失败，保留其依赖并阻止当前进程继续加载新模型，需重启进程。

## 首轮运行证据与已解决问题

主机检查共 65 项通过：ASR 34、TTS 24、TTS 源码准备/准入 7；包含无板卡输入拒绝、生命周期故障注入和输出独立存储。

- ASR：正式 API 的中文重复请求、等长度英文、0.5 秒及 8 秒边界、切回原输入和关闭重载均通过。四组音频的文本与官方 CPU FP16 参考一致。固定长度 Graph size=2，连续请求 replay 为 9→20→31→42，invariant 始终为 true。音频长度变化时先清理旧图，避免占满 Launch arena；切回旧长度重新 BUILD，缓存内 replay 计数重新开始。
- ASR 数值参考：Transformers 5.3 原生组件与官方 4.57.6 的真实 FP32/FP16 权重、processor 和已测边界逐位一致。混合路径音频特征 MSE 约 `8.6e-10`，文本 prefill MSE 约 `0.00544`、max-abs 约 `1.77`，仅作差异记录，不设任意数值门槛。
- TTS CPU 加载：处理 5.3 的 decorator、RoPE、tokenizer 等兼容差异；避免 HF meta/from_pretrained 路径错误重初始化已加载权重，采用正常 CPU 构造和 strict safetensors 加载。974 个持久 tensor、3 个 RoPE buffer 与冻结参考一致；5.3 FP32 完整波形及已测中间边界与官方 4.57.3 FP32 逐位一致。
- TTS 混合路径：两个 decoder 各自一个 Graph、所有输出头实走 RPU。早期 32-token 三次请求生成相同的有限波形，但测试句末尾被长度上限截断；改用 64-token 上限后自然结束，输出 24 kHz、71,040 采样点（2.96 秒），再用独立 CPU ASR 识别得到 “Hello. This is a short speech test.”。
- TTS 数值限制：官方全 CPU FP16 的完整生成出现非有限值，CPU FP16 codec 也很慢，因此正式方案明确采用 CPU FP32＋RPU FP16。RPU body 与冻结 FP16 权重的 FP32 anchor 存在正常数值/argmax 差异，未将其隐藏或声明逐位相等。生成结果一致性和语音内容交叉检查不代表音色或 MOS 认证。

正式 TTS API 的 7 次连续请求及关闭重载已通过：三次英文输出逐位一致，中文短句、参考音频 0.5/20 秒及生成下限 2 token 均产生有限波形。修复后又验证了 32-token 文本配 64-token 生成上限、关闭重载和英文波形与原结果一致。两个 Graph size 均保持 1；前三次 Talker replay 为 56→113→170，Predictor 为 591→1183→1775，invariant 始终为 true。保留的旧输出未被后续请求改写。参考音频上下边界只验证可运行，不保证音色或发音质量：0.5 秒截取和 20 秒补静音参考上的 `Hello.` 生成均到达上限，独立 ASR 未识别出文本；不要将这些边界样本当作质量通过。匹配的 CPU FP32 对照也生成到上限，输出接近静音（RMS `1.11e-5`；RPU 为 `0.00194`），因此该边界不能视为质量通过，两路径波形也不相等。应使用包含清晰连续人声的参考音频。中文常规参考生成可重新识别为“你好，欢迎使用。”。

单次中文 ASR 约 10–13 秒，早期短句 TTS 混合请求约 5–7 秒（均不含模型加载）。这些是本次机器上的运行观察，不是并发、长文本或实时性能承诺。严格权重哈希和冷加载另有开销。

## 速度与识别质量评测

后续按 2 次预热、5 次固定输入测量及中英文多输入协议完成了[速度与识别质量评测](qwen3_audio_benchmark_r1.zh.md)。该报告包含 RTF、加载/推理耗时、ASR CER/WER、CPU 对照及 TTS 内容一致性，替代上文零散耗时观察作为本次性能结论。

最新性能与计算位置见 [ASR 连续驻留](qwen3_asr_resident_r1.zh.md)及 [TTS 输出头与 codec 优化](qwen3_tts_chain_optimization_r1.zh.md)。此前 [Thor 对比及分阶段优化](qwen3_audio_thor_stages.zh.md)记录 ASR 输出头迁至 RPU、TTS 固定 Predictor 生成循环和 CPU 线程配置。

首轮优化参考 RhinoVLA 复用分配器和 KV 存储，减少无用计算；ASR 卷积精度边界按上表更新。该轮速度与原样本质量对比见[推理优化报告](qwen3_audio_optimization_r1.zh.md)，前述迁移记录和首次性能报告保留为历史基线。两模型加载前需要尚未固定为 direct allocator 的新进程。

## 在新环境安装与运行

先按[运行时资产](runtime_assets.md)配置匹配的 R1/Launch/operator assets，再安装本分支构建的包及 `audio` extra。不要安装官方 ASR/TTS 包覆盖仓库固定的 Transformers 5.3.0。

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

## 本次 R1 交付

机器 `agi@10.10.66.42` 上为独立目录，不修改原 Python 环境：

- 运行包、音频依赖、prepared TTS 源码和示例：`/home/agi/rhinoforge_audio_runtime/`。
- ASR 权重：`/home/agi/rhinoforge_audio_models/Qwen3-ASR-0.6B`。
- TTS 权重：`/mnt/user_data/rhinoforge_audio_models/Qwen3-TTS-12Hz-0.6B-Base`。
- `run` 设置上述依赖及已安装 runtime，使用设备锁串行运行。R1 的 `/dev/mem` 需要现有 `sudo -n` 权限；不在仓库或脚本保存密码。运行目录仍依赖 R1 已有的持久 Python 环境及 restricted runtime 路径。

```bash
/home/agi/rhinoforge_audio_runtime/run asr \
  --checkpoint /home/agi/rhinoforge_audio_models/Qwen3-ASR-0.6B \
  --audio /home/agi/rhinoforge_audio_runtime/audio/asr_zh.wav
/home/agi/rhinoforge_audio_runtime/run tts \
  --checkpoint /mnt/user_data/rhinoforge_audio_models/Qwen3-TTS-12Hz-0.6B-Base \
  --source /home/agi/rhinoforge_audio_runtime/source/qwen_tts \
  --reference-audio /home/agi/rhinoforge_audio_runtime/audio/clone.wav \
  --text 'Hello, this is a short speech test.' --language English \
  --output /home/agi/rhinoforge_audio_runtime/my-speech.wav
```

已补充 20 条公开音频的 ASR CER/WER 和 10 条 TTS 内容一致性评测；没有进行完整 WER/CER 数据集、MOS/音色相似度评测、长音频、多语言质量、并发或实时压力测试。后续扩大范围应单独评估，不能从本次短输入结果直接推断支持。

正式 ASR/TTS CLI 均在持久环境运行成功，TTS CLI 波形与 API 样本逐位一致。交付示例已生成 `/home/agi/rhinoforge_audio_runtime/hello.wav`；重新运行请使用新的输出文件名。

运行记录：[ASR 正式请求](evidence/qwen3_audio_r1/asr-public-validation.json)、[TTS 正式请求](evidence/qwen3_audio_r1/tts-public-validation.json)。[修复后上界与重载](evidence/qwen3_audio_r1/tts-final-validation.json)、[依赖版本](evidence/qwen3_audio_r1/environment.json)、[短参考音频反例](evidence/qwen3_audio_r1/tts-short-reference-comparison.json)。R1 的 `evidence/` 另外保存日志与参考对照，`audio/tts-public-0.wav` 和 `audio/tts-public-3.wav` 为英文、中文生成样本。
