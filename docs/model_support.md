# Model entries and execution scope

[简体中文](model_support.zh.md) | English

Choose a TOML from [model examples](../examples/README.md), prepare its checkpoint and inputs, then run `python examples/run.py --config PATH --check-config` before execution. This command validates configuration only.

Each profile specifies checkpoint, precision, core count, input shape and length. Loaders and runtime admission enforce those limits; controlled entries also require the template's opt-ins. See the [configuration index](../examples/configs/CONFIGURATIONS.md).

Supply model weights and request data separately. Follow [model assets](model_assets.md) and [runtime assets](runtime_assets.md) for dependencies, and [running models](model_testing.md), [API](api_reference.md) and [quantization](quantization.md) for usage.

## Qwen3 speech

`Qwen3ASRModel` and `Qwen3TTSModel` provide dedicated ASR and TTS entry points
with CPU and device components. Use the [speech guide](qwen3_audio.md) for pinned
assets, input limits, source preparation and process ownership. These interfaces
are separate from the text/vision TOML runner and CausalLM/VLM registry.
