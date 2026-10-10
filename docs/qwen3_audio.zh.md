# Qwen3 ASR / TTS 使用指南

简体中文 | [English](qwen3_audio.md)

`Qwen3ASRModel` 将本地音频转成文本，`Qwen3TTSModel` 根据短文本和参考音频生成语音。
两者使用独立 Python API 和 CLI，不经过文本/视觉 TOML runner 或 CausalLM/VLM registry。

## 模型与输入

| 项目 | 固定来源 |
|---|---|
| ASR checkpoint | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/5eb144179a02acc5e5ba31e748d22b0cf3e303b0) |
| ASR 参考源码 | [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR/tree/7c6daf77a2421100f5fb066495372c00129d39ff)；运行时直接使用 Transformers 组件，无需安装该源码 |
| TTS checkpoint | [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/tree/5d83992436eae1d760afd27aff78a71d676296fc)，包含 `speech_tokenizer/` |
| TTS 配套源码 | [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS/tree/022e286b98fbec7e1e916cb940cdf532cd9f488e)，按下文准备兼容副本 |

| 模型 | 输入范围 | 生成方式与输出 |
|---|---|---|
| ASR | batch 1；单声道 16 kHz、0.5–8 秒；有限浮点样本 | greedy，`max_new_tokens=1..64`；返回文本、含语言元数据的原始文本和 token ID；无 context、时间戳或流式接口 |
| TTS Base | batch 1；`Chinese` / `English`；文本 1–128 字符且最多 32 tokenizer token；参考音频单声道 24 kHz、0.5–20 秒 | 非流式 greedy，`max_new_tokens=2..64`；返回独立存储的 float32 单声道波形和 24000 Hz 采样率 |

TTS 只使用参考音频的 speaker embedding；不接受 ICL transcript、CustomVoice 或
VoiceDesign。Talker prompt 加生成上限须在 128 个位置内；达到上限可能截断句子。

加载器先核对固定 checkpoint、配置和 tokenizer 身份，再转换权重。ASR 身份见
[`qwen3_asr.py`](../python/rpu_backend/api/qwen3_asr.py)，TTS 身份见
[`_manifest.py`](../python/rpu_backend/adapters/qwen3_tts/_manifest.py)。保留原 tokenizer
文件，不添加覆盖文件。当前接口固定使用八个设备核，仅接受 `dtype=torch.float16`。

## 1. 安装

按[源码构建](development_build.md)创建 Python 3.12 环境，并配置
[匹配的运行时资产](runtime_assets.zh.md)。在仓库根目录安装 `audio` 可选依赖：

```bash
python -m pip install '.[audio]' --no-build-isolation \
  -Ccmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  -Cbuild.tool-args=-j2
python -m pip check
```

依赖以 [pyproject.toml](../pyproject.toml) 为准，包括 PyTorch 2.10.0、Transformers
5.3.0。保留该 Transformers 版本，不在此环境安装上游 TTS 包。
`audio` 包含 TorchAudio、ONNX Runtime 和 python-sox，以满足上游导入链；当前 12Hz
路径不调用 SoX 可执行文件，未安装该程序时可能收到上游警告。

## 2. 下载权重并准备 TTS 源码

```bash
export RPU_MODEL_CACHE="${RPU_MODEL_CACHE:-$HOME/.cache/rhinoforge/models}"
mkdir -p "$RPU_MODEL_CACHE"
hf download Qwen/Qwen3-ASR-0.6B \
  --revision 5eb144179a02acc5e5ba31e748d22b0cf3e303b0 \
  --local-dir "$RPU_MODEL_CACHE/Qwen3-ASR-0.6B"
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --revision 5d83992436eae1d760afd27aff78a71d676296fc \
  --local-dir "$RPU_MODEL_CACHE/Qwen3-TTS-12Hz-0.6B-Base"

export QWEN3_TTS_SOURCE=/path/to/Qwen3-TTS
export QWEN3_TTS_PREPARED=/path/to/qwen3-tts-prepared
git clone https://github.com/QwenLM/Qwen3-TTS.git "$QWEN3_TTS_SOURCE"
git -C "$QWEN3_TTS_SOURCE" checkout 022e286b98fbec7e1e916cb940cdf532cd9f488e
python -m rpu_backend.adapters.qwen3_tts \
  --source "$QWEN3_TTS_SOURCE" --output "$QWEN3_TTS_PREPARED"
cp "$QWEN3_TTS_SOURCE/LICENSE" "$QWEN3_TTS_PREPARED/LICENSE"
```

将 `/path/to/...` 换成实际目录。prepare 输出目录必须尚不存在；它复制固定哈希的
15 个源码文件并应用兼容补丁，不执行原始源码、不修改原 checkout。
加载时检查 prepared 清单和文件哈希；API/CLI 的 `source` 指向生成的 `qwen_tts/`
子目录。保留随源码交付的许可证。

## 3. 运行

确认设备空闲并取得平台设备锁，ASR 和 TTS 分别在独立进程运行。

ASR CLI 读取单声道 16 kHz、16-bit PCM WAV，向标准输出写入 JSON：

```bash
python examples/qwen3_asr.py \
  --checkpoint "$RPU_MODEL_CACHE/Qwen3-ASR-0.6B" \
  --audio /path/to/mono-16k-pcm16.wav \
  --max-new-tokens 64 --threads 4
```

TTS CLI 读取本地参考音频，输出 FLOAT WAV；目标文件必须尚不存在：

```bash
python examples/qwen3_tts.py \
  --checkpoint "$RPU_MODEL_CACHE/Qwen3-TTS-12Hz-0.6B-Base" \
  --source "$QWEN3_TTS_PREPARED/qwen_tts" \
  --reference-audio /path/to/mono-24k.wav \
  --text '你好，欢迎使用。' --language Chinese \
  --max-new-tokens 64 --threads 4 --output /path/to/output/hello.wav
```

两种 CLI 的 `--threads` 均为正整数，默认 4；Python API 不修改进程线程数。
ASR Python API 可直接接收一维 CPU 浮点数组或 Tensor；接口和返回值见
[语音 API](api_reference.zh.md#bounded-speech-apis)。

## 执行与资源

| 模型 | CPU | Rhino 设备 |
|---|---|---|
| ASR | 音频读取、mel/分块、卷积与层间激活、单 token embedding、有限值检查、argmax、文本解码 | conv_out、位置和有效行处理、音频 encoder 与末尾投影、prefill 文本/音频组装、文本 decoder 和输出头 |
| TTS | embedding、参考音频、speaker encoder、生成控制、码本、卷积/上采样、SnakeBeta、波形输出 | Talker、Predictor 和输出头；具备对应 native 接口时还包括 codec 输入/输出投影和 Transformer 前段 |

ASR 卷积使用 FP16 舍入权重进行 CPU FP32 计算，再转回 FP16；卷积采用 channels-last。
设备音频 Graph 使用 FP32 Linear 累加。TTS CPU 组件保持 FP32，设备组件使用 FP16、
FP32 Linear 累加；codec 前段 LayerScale 在 FP32 中折入投影后转 FP16，RoPE 保留 FP32 频率。
codec 前段只接受 1–63 帧，低于其 72 帧滑窗。

- ASR 需要重编译的 native 提供 `siglip_set_audio_mode`、`siglip_set_audio_projection`
  和 `assemble_inputs_embeds`，缺少接口时加载失败。
- TTS codec 通过 `causal_decoder_set_tts_codec_mode` 显式启用设备路径；缺少 setter
  时前段留在 CPU。`execution_stats()` 返回设备和原因；其他执行错误直接报告。
- 一个进程只保留一个存活模型/策略；请求、`execution_stats()` 和 `close()` 串行调用。
  两类 API 均支持 `with` 和幂等 `close()`，输出存储独立于后续重放。
- 加载器在设备分配前选择 caching allocator。TTS 同时申请 owner 专属五个 arena：
  四个保留 Graph 和一个串行即时调用。allocator 或 arena 计划已冻结且不兼容时，启动新进程。
- ASR 保留当前长度的一张音频图和最多两张文本图；TTS 保留 Talker prefill/decode、
  Predictor decode 及当前帧数的 codec 图。长度变化按组件淘汰旧签名，同长度重放。
- TTS 只缓存最新参考音频内容的 speaker embedding，每次请求仍读取和检查文件。
  转换失败后关闭模型，从干净权重重新加载；释放时先清 Graph，再释放 handle 和权重。
