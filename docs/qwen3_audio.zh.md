# Qwen3 ASR / TTS 使用指南

简体中文 | [English](qwen3_audio.md)

`Qwen3ASRModel` 将本地音频转成文本，`Qwen3TTSModel` 根据文本和参考音频生成语音。
两者保留独立 Python API，并提供 [ASR](../examples/configs/qwen3_asr/README.md) /
[TTS](../examples/configs/qwen3_tts/README.md) TOML 入口，通过 `python -m examples.run --config PATH`
运行；不经过 CausalLM/VLM registry。

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
| TTS Base | batch 1；`Chinese` / `English`；文本 1–1024 字符；参考音频为有限、非全零的单声道 24 kHz、0.5–20 秒 | 非流式采样，每段 `max_new_tokens=2..256`（默认 256）；返回独立存储的 float32 单声道波形和 24000 Hz 采样率 |

TTS 只使用参考音频的 speaker embedding；不接受 ICL transcript、CustomVoice 或
VoiceDesign。每段 Talker prompt 加生成上限须在 512 个位置内。采样沿用固定
checkpoint 配置：top-k 50、temperature 0.9、top-p 1.0。未生成 EOS 时在波形
解码前报错；空、非有限值或全零输出会被拒绝。

不超过 128 字符且 32 tokenizer token 的文本保持单段。长文本按标点分段，原始
片段最多 128 字符，目标 12 token；不超过八个字符的短句尾可与前一分句合并至
16 token。段尾补句号可能改变 token 数、停顿和语调。每段从本次请求初始 CPU 随机状态开始，后续分段
恢复第一段留下的调用方状态。各段独立复制，全部成功后拼接返回；失败不返回部分音频。

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
加载时检查 prepared 清单和文件哈希；API 的 `source` 指向生成的 `qwen_tts/`
子目录。保留随源码交付的许可证。

## 3. 运行

确认设备空闲并取得平台设备锁，ASR 和 TTS 分别在独立进程运行。

ASR API 接收单声道 16 kHz 浮点采样。以下代码读取本地 WAV，并以 JSON 输出转写结果：

```python
import json
import os
from dataclasses import asdict
from pathlib import Path

import soundfile as sf
import torch
from rpu_backend.api import Qwen3ASRModel

torch.set_num_threads(4)
checkpoint = Path(os.environ["RPU_MODEL_CACHE"]) / "Qwen3-ASR-0.6B"
audio, sample_rate = sf.read("/path/to/mono-16k.wav", dtype="float32")
with Qwen3ASRModel.from_pretrained(checkpoint, dtype=torch.float16) as model:
    result = model.transcribe(audio, sample_rate=sample_rate, max_new_tokens=64)
print(json.dumps(asdict(result), ensure_ascii=False))
```

TTS API 接收本地参考音频。以下代码创建新的 FLOAT WAV，输出目录须已存在：

```python
import os
from pathlib import Path

import soundfile as sf
import torch
from rpu_backend.api import Qwen3TTSModel

torch.set_num_threads(16)
checkpoint = Path(os.environ["RPU_MODEL_CACHE"]) / "Qwen3-TTS-12Hz-0.6B-Base"
source = Path(os.environ["QWEN3_TTS_PREPARED"]) / "qwen_tts"
with Qwen3TTSModel.from_pretrained(checkpoint, source=source, dtype=torch.float16) as model:
    torch.manual_seed(0)
    waveform, sample_rate = model.synthesize(
        "Hello, welcome.", reference_audio="/path/to/mono-24k.wav",
        language="English", max_new_tokens=256,
    )
with open("/path/to/output/hello.wav", "xb") as output:
    sf.write(output, waveform, sample_rate, format="WAV", subtype="FLOAT")
```

加载前按主机配置设置线程数，API 不会修改该值；加载后、每次请求前设置种子，
可在同一执行配置下复现采样。多次合成可复用同一个模型上下文。ASR 也接受一维
CPU 浮点 Tensor。
函数签名和返回类型见[语音 API](api_reference.zh.md#语音-api)。

## 执行与资源

| 模型 | CPU | Rhino 设备 |
|---|---|---|
| ASR | 音频读取、mel/分块、卷积与层间激活、单 token embedding、有限值检查、argmax、文本解码 | conv_out、位置和有效行处理、音频 encoder 与末尾投影、prefill 文本/音频组装、文本 decoder 和输出头 |
| TTS | embedding、参考音频、speaker encoder、生成控制、码本、卷积/上采样、SnakeBeta、波形输出 | Talker、Predictor 和输出头；具备对应 native 接口时还包括 codec 输入/输出投影和 Transformer 前段 |

ASR 卷积使用 FP16 舍入权重进行 CPU FP32 计算，再转回 FP16；卷积采用 channels-last。
设备音频 Graph 使用 FP32 Linear 累加。TTS CPU 组件保持 FP32，设备组件使用 FP16、
FP32 Linear 累加；codec 前段 LayerScale 在 FP32 中折入投影后转 FP16，RoPE 保留 FP32 频率。
设备 codec 前段接受 1–63 帧；64–255 帧使用原始 CPU FP32 前段及其 72 帧滑窗注意力。

- ASR 需要重编译的 native 提供 `siglip_set_audio_mode`、`siglip_set_audio_projection`
  和 `assemble_inputs_embeds`，缺少接口时加载失败。
- TTS codec 通过 `causal_decoder_set_tts_codec_mode` 显式启用设备路径；缺少 setter
  时前段留在 CPU。`execution_stats()` 返回最后一段的设备和原因，以及总生成步数、
  EOS 和每段步数；其他执行错误直接报告。
- 一个进程只保留一个存活模型/策略；请求、`execution_stats()` 和 `close()` 串行调用。
  两类 API 均支持 `with` 和幂等 `close()`，输出存储独立于后续重放。
- 加载器在设备分配前选择 caching allocator。TTS 同时申请 owner 专属五个 arena：
  四个保留 Graph 和一个串行即时调用。allocator 或 arena 计划已冻结且不兼容时，启动新进程。
- ASR 保留当前长度的一张音频图和最多两张文本图；TTS 保留 Talker prefill/decode、
  Predictor decode 及当前帧数的 codec 图。长度变化按组件淘汰旧签名，同长度重放。
- Predictor 最后一行设备 hidden 只供紧接着的输出头使用一次；请求 hidden states
  时仍返回独立 CPU 存储。
- Predictor 复用 native checked decode replay：仅匹配五层 FP16、32 位置缓存、
  一维 RoPE 和无融合输出头的固定路径。首次记录完整算子序列，后续校验权重版本、
  KV 所有者、Graph 和内存布局，再更新位置及 attention 参数，跳过重复的主机算子
  调度。输入上传、设备计算、输出头和原采样继续执行；无法建立证明时走普通路径。
  此优化需要重新编译匹配的 native，不能用通用 Graph 跳过开关代替。
- TTS 只缓存最新参考音频内容的 speaker embedding，每次请求仍读取和检查文件。
  转换失败后关闭模型，从干净权重重新加载；释放时先清 Graph，再释放 handle 和权重。

## 实验性 RPU 声码器卷积

在 `Qwen3TTSModel.from_pretrained` 中设置 `codec_backend="rpu"`，或使用
`examples/configs/qwen3_tts/0_6b/base_rpu_codec.toml`。默认仍为 `cpu`。
RPU 模式将 12 层残差 Conv1d（kernel 7；通道 768/384/192/96；dilation 1/3/9）
转换成显式八核 Linear，输入/权重 FP16、累加 FP32。权重只转换一次，保留原 CPU FP32
权重；输出通道补齐到 128 对齐后裁回原大小，再加原 FP32 bias。

CPU 仍执行列展开、码本、SnakeBeta、其余卷积和上采样，因此属于部分卸载。
复用同步 immediate Linear 和第五个共享 arena，不新增常驻 Graph。几何不匹配会拒绝
安装，执行错误直接抛出，不静默退回 CPU。`execution_stats().codec_convolutions`
记录 `backend`、安装的 `layers` 和累计 RPU `calls`。前段 Transformer 的 63 帧设备
边界和更长序列的 CPU 滑窗注意力保持不变。

FP16 舍入会改变波形，应使用同一份 codec tokens 对照 CPU FP32 并核验语音内容。
该模式为实验选项，不继承默认配置的音质结论，运行成功不等于音质认证。
