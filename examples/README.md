# Model examples

[简体中文](README.zh.md) · [Config index](configs/CONFIGURATIONS.md) · [Conversion commands](configs/WEIGHTS.md#quantization) · [Weight sources](configs/WEIGHTS.md) · [Support matrix](../docs/model_support.md)

Choose a model and recommended TOML, then prepare weights and inputs using its guide. Current model templates use [`run.py`](run.py); `--config` is required.

Legacy model scripts remain compatibility wrappers. See the [v1.2.0 upgrade guide](../docs/upgrading_1_2.md) for old paths and input migrations; `python examples/migrate_config.py --help` lists the migration CLI.

## Choose a model

| Model and config guide | Recommended config |
|---|---|
| [DINOv3](configs/dinov3/README.md) | [ViT-B FP16](configs/dinov3/vit_b/fp16.toml) |
| [G0.5](configs/g05/README.md) | [3cam FP16](configs/g05/3cam/fp16.toml) |
| [Gemma4](configs/gemma4/README.md) | [E4B FP16](configs/gemma4/e4b/fp16.toml) |
| [GR00T](configs/gr00t/README.md) | [224 FP16](configs/gr00t/n1_7_droid/fp16.toml) |
| [HyVLA](configs/hy_vla/README.md) | [3cam FP16](configs/hy_vla/3cam/fp16.toml) |
| [HY Embodied UMI](configs/hy_embodied/README.md) | [UMI FP16](configs/hy_embodied/umi/fp16.toml) |
| [Llama](configs/llama/README.md) | [3.2-1B FP16](configs/llama/3_2_1b/fp16.toml) |
| [Pi0.5](configs/pi05/README.md) | [3cam W8A16](configs/pi05/3cam/w8a16.toml) |
| [Qwen3](configs/qwen3/README.md) | [0.6B FP16](configs/qwen3/text/0_6b/fp16.toml) |
| [Qwen3.5 text](configs/qwen3_5/README.md) | [0.8B FP16](configs/qwen3_5/text/0_8b/fp16.toml) |
| [Qwen3.5 vision](configs/qwen3_5/README.md) | [2B FP16](configs/qwen3_5/vl/2b/fp16.toml) |
| [Qwen3-VL](configs/qwen3_vl/README.md) | [2B FP16](configs/qwen3_vl/vl/2b/fp16.toml) |
| [RhinoVLA](configs/rhinovla/README.md) | [v1 FP16](configs/rhinovla/v1/official_fp16.toml) |
| [Wall-OSS base](configs/wall_oss/README.md) | [3cam 224 FP16](configs/wall_oss/base/3cam/224/fp16.toml) |
| [Wall-OSS RTC](configs/wall_oss/rtc/README.md) | [FP16](configs/wall_oss/rtc/fp16.toml) |
| [Wall-OSS sorting](configs/wall_oss/sorting/README.md) | [W8A16](configs/wall_oss/sorting/w8a16.toml) |
| [InternVLA](configs/internvla/README.md) | [NavDP FP16](configs/internvla/navdp/fp16.toml) |
| [LingBot v2](configs/lingbot/v2/README.md) | [3cam T3 W8A16](configs/lingbot/v2/3cam_t3/w8a16.toml) |
| [LingBot v1 (deferred)](configs/lingbot/README.md) | [4B mixed](configs/lingbot/v1/4b_mixed.toml) |
| [HALO (deferred)](configs/halo/README.md) | [FP16 Expert](configs/halo/action_expert/fp16.toml) |

HALO and LingBot v1 remain deferred. Qwen3.5 35B MoE uses its controlled profiles.

## Run

Install the backend, copy the chosen TOML, and fill in asset paths, inputs, and `run.output_dir` as described in the model guide.

```bash
cp examples/configs/g05/3cam/fp16.toml /tmp/g05.toml
# Edit paths and parameters as described in the config guide
python examples/run.py --config /tmp/g05.toml --check-config
python examples/run.py --config /tmp/g05.toml
```

`--check-config` validates configuration without importing Torch or accessing a device; `--help` lists the unified CLI options. Edit supported ACC, text length, CPU threads, and sample counts in the same TOML; the model guide lists legal values.

<a id="logging-and-performance-output"></a>
## Logging

Add this top-level table beside `[run]`; these are also the defaults:

```toml
[logging]
level = "ERROR"
summary = true
```

`level` accepts the uppercase values below. It filters diagnostics; `summary=true` keeps `[MODEL]` configuration and `[PERF]` runtime summaries at every level.

| Level | Diagnostics shown |
|---|---|
| `CRITICAL` | Fatal errors only; ordinary ERROR messages are hidden |
| `ERROR` | ERROR and CRITICAL; default |
| `WARNING` | Warnings, including tagged fallback and Graph messages, plus errors |
| `INFO` | Loading, ordinary `print()` output and generated text, plus warnings/errors |
| `DEBUG` | Existing Graph diagnostics, complete reports and native TRACE output |

Ordinary `print()` is INFO; recognized tags select their own level. Exceptions propagate. `summary=false` hides only `[MODEL]` and `[PERF]`. There is no `logging.file` field. Native output is saved as `native-*.stdout.log` / `native-*.stderr.log` under `run.output_dir` and replayed with the selected filter when the process finishes. `report.json` is saved at every level.

`[MODEL]` lists precision, inputs, device settings, CPU threads, warmup/runs and output location. `[PERF]` labels the runner's actual call boundary and separates complete-model, component and diagnostic calls. LLM/VLM summaries use actual input tokens, decode calls and output counts; prefill produces the first output token. Unmeasured Vision or TTFT fields are omitted.

## Results

`run.output_dir` selects the output location; some profiles create subdirectories. Read `report.json` for timings, checks, and actual artifact paths. Multi-length G05 results use one `text_N/` directory per length.

[Runtime setup](../docs/runtime_assets.md)

## Qwen3 speech

- [`qwen3_asr.py`](qwen3_asr.py): local mono 16 kHz PCM16 WAV to transcript.
- [`qwen3_tts.py`](qwen3_tts.py): short Chinese/English text and mono 24 kHz reference audio to WAV.

Install the `audio` extra and follow the [speech guide](../docs/qwen3_audio.md)
for pinned weights, source preparation, commands and input limits. These scripts
use dedicated APIs, separate from the TOML runner.
