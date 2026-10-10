# Qwen3 ASR / TTS guide

[简体中文](qwen3_audio.zh.md) | English

`Qwen3ASRModel` transcribes local audio. `Qwen3TTSModel` generates speech from
short text and reference audio. Both use dedicated Python APIs and CLIs,
separate from the text/vision TOML runner and CausalLM/VLM registry.

## Models and inputs

| Asset | Pinned source |
|---|---|
| ASR checkpoint | [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/5eb144179a02acc5e5ba31e748d22b0cf3e303b0) |
| ASR reference source | [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR/tree/7c6daf77a2421100f5fb066495372c00129d39ff); inference uses Transformers components directly and does not require this checkout |
| TTS checkpoint | [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/tree/5d83992436eae1d760afd27aff78a71d676296fc), including `speech_tokenizer/` |
| TTS source | [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS/tree/022e286b98fbec7e1e916cb940cdf532cd9f488e), prepared as described below |

| Model | Inputs | Generation and output |
|---|---|---|
| ASR | Batch one; finite mono floating-point samples at 16 kHz, 0.5–8 seconds | Greedy, `max_new_tokens=1..64`; text, raw text with language metadata, and token IDs; no context prompt, timestamps or streaming |
| TTS Base | Batch one; `Chinese` / `English`; 1–128 text characters and at most 32 tokenizer tokens; mono 24 kHz reference audio, 0.5–20 seconds | Non-streaming greedy, `max_new_tokens=2..64`; independent float32 mono waveform and sample rate 24000 |

TTS uses only the reference speaker embedding. ICL transcripts, CustomVoice and
VoiceDesign are unavailable. Talker prompt plus the generation limit must fit
128 positions; reaching the limit can truncate speech.

Loaders check the pinned checkpoint, configuration and tokenizer before weight
transformation. ASR identities are in
[`qwen3_asr.py`](../python/rpu_backend/api/qwen3_asr.py); TTS identities are in
[`_manifest.py`](../python/rpu_backend/adapters/qwen3_tts/_manifest.py).
Keep the original tokenizer files without overrides. Both APIs use eight device
cores and accept only `dtype=torch.float16`.

## 1. Install

Create a Python 3.12 environment using the [source build guide](development_build.md)
and configure [matching runtime assets](runtime_assets.md). From the repository
root, install the `audio` extra:

```bash
python -m pip install '.[audio]' --no-build-isolation \
  -Ccmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  -Cbuild.tool-args=-j2
python -m pip check
```

[pyproject.toml](../pyproject.toml) pins the dependencies, including PyTorch 2.10.0
and Transformers 5.3.0. Keep this Transformers version and prepare the TTS source
without installing the upstream TTS package. The `audio` extra includes
TorchAudio, ONNX Runtime and python-sox for the upstream import chain. The current
12Hz path does not invoke the SoX executable; its absence may cause an upstream warning.

## 2. Download weights and prepare TTS source

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

Replace `/path/to/...` with local directories. The prepared output directory must
not exist. Preparation copies 15 hash-pinned source files and applies compatibility
patches without executing the original source or modifying its checkout. Loading
checks the prepared manifest and file hashes. The API/CLI `source` argument points
to the generated `qwen_tts/` subdirectory. Keep the upstream license with the copy.

## 3. Run

Use an idle device with the platform's device lock. Run ASR and TTS in separate
processes.

The ASR CLI reads mono 16 kHz, 16-bit PCM WAV and writes JSON to standard output:

```bash
python examples/qwen3_asr.py \
  --checkpoint "$RPU_MODEL_CACHE/Qwen3-ASR-0.6B" \
  --audio /path/to/mono-16k-pcm16.wav \
  --max-new-tokens 64 --threads 4
```

The TTS CLI reads a local reference file and writes FLOAT WAV. The output file
must not already exist:

```bash
python examples/qwen3_tts.py \
  --checkpoint "$RPU_MODEL_CACHE/Qwen3-TTS-12Hz-0.6B-Base" \
  --source "$QWEN3_TTS_PREPARED/qwen_tts" \
  --reference-audio /path/to/mono-24k.wav \
  --text 'Hello, welcome.' --language English \
  --max-new-tokens 64 --threads 4 --output /path/to/output/hello.wav
```

Both CLIs accept a positive `--threads` value, defaulting to 4. The Python APIs do
not change the process thread count. The ASR API also accepts a one-dimensional
CPU floating-point array or Tensor. See [speech APIs](api_reference.md#bounded-speech-apis)
for signatures and return types.

## Execution and resources

| Model | CPU | Rhino device |
|---|---|---|
| ASR | Audio reading, mel features/chunking, convolutions and their activations, single-token embedding, finite checks, argmax, text decoding | conv_out, positions and valid rows, audio encoder and output projections, prefill text/audio composition, text decoder and vocabulary head |
| TTS | Embedding, reference processing, speaker encoder, generation control, codebook, convolutions/upsampling, SnakeBeta, waveform output | Talker, Predictor and output heads; codec input/output projections and Transformer frontend when supported by native interfaces |

ASR convolutions use channels-last storage and FP16-rounded weights in CPU FP32
computation, then round outputs to FP16. The device audio Graph uses FP32 Linear
accumulation. TTS CPU components stay FP32; device components use FP16 with FP32
Linear accumulation. Codec LayerScale is folded into projections in FP32 before
conversion to FP16, while RoPE retains FP32 frequencies. The codec frontend accepts
1–63 frames, below its 72-frame sliding window.

- ASR requires a rebuilt native extension exposing `siglip_set_audio_mode`,
  `siglip_set_audio_projection` and `assemble_inputs_embeds`; loading fails if
  these interfaces are absent.
- TTS explicitly enables the device codec path with
  `causal_decoder_set_tts_codec_mode`. Without that setter, the frontend stays on
  CPU; `execution_stats()` reports the device and reason. Other execution errors
  propagate to the caller.
- Keep one live model/policy per process. Serialize requests, `execution_stats()`
  and `close()`. Both APIs support `with`, idempotent `close()` and outputs with
  storage independent of later replay.
- Loaders select the caching allocator before device allocation. TTS requests
  five owner-specific arenas: four retained Graphs and one serial immediate call.
  Start a new process if its frozen allocator or arena plan is incompatible.
- ASR retains one audio Graph and at most two text Graphs for current lengths.
  TTS retains Talker prefill/decode, Predictor decode and the current codec frame
  count. Components evict old signatures when their length changes and replay
  matching signatures.
- TTS caches only the latest reference audio's speaker embedding by decoded
  content; every request still reads and validates the file. After a failed
  transformation, close the model and reload clean weights. Retire Graphs before
  releasing handles and weights.
