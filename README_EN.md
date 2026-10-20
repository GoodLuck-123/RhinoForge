<div align="center">

<a href="https://github.com/HUIXI-AI/RhinoVLA"><img src="https://raw.githubusercontent.com/HUIXI-AI/RhinoVLA/3bfdbbfe8370be42f3a936eec616d065219e1825/assets/huixi_logo_cropped.png" alt="Huixi Intelligence" height="72" /></a>

# RhinoForge

**PyTorch inference and model deployment for the Rhino Processing Unit**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)

**English** | [简体中文](README.md)

</div>

RhinoForge exposes the Rhino device as `torch.rpu`. It combines
model adapters, fused model runtimes, graph capture and replay, SPM memory
management, multi-core execution, offline quantization, and a model-porting
template in one PyTorch backend.

The distribution name is `rhinoforge`; the Python import remains
`rpu_backend`. RhinoForge is inference-only and does not support training or
fine-tuning.

This source tree targets **v1.3.0.dev0** and depends on archived **Rhino Launch v1.2.0**.
Use the published components specified in the [runtime asset guide](docs/runtime_assets.md).
See the [release policy](docs/release_policy.md), [changelog](CHANGELOG.md)
and [upgrade guide](docs/upgrading_1_2.md) before updating an existing installation.

## Highlights

- **PyTorch integration:** use the device through the standard custom-device path
  and `torch.rpu` APIs.
- **Model deployment:** run text, vision-language, vision, and VLA profiles
  through model-specific adapters and TOML examples.
- **Extensible runtime:** reuse the public fused-model, graph, SPM, multi-core,
  batch, and adapter mechanisms when porting another model.
- **Profile-scoped support:** every status is tied to an exact checkpoint,
  precision, input envelope, configuration, and runtime asset set.

## AI-assisted model porting

The checked-in [`rhinoforge-port` skill](.agents/skills/rhinoforge-port/SKILL.md)
supports two modes. **Assess** produces a read-only capability review with an
outcome and independent certification; **Port** starts only from an accepted,
certified, non-Blocked assessment and loads the applicable public family
playbook. See [Model porting](docs/model_porting.md) for the canonical workflow.

## Examples

[`examples/`](examples/README.md) provides the unified runner and TOML
configurations organized by model for text, image-text, vision components, and
VLA workflows. See the [configuration index](examples/configs/README.md) for
the directory layout, precision choices, and input requirements.

Supply your own model weights and requests. Configurations and runtime admission
checks constrain each checkpoint, precision, core count, and input envelope.

Qwen3 ASR/TTS use dedicated speech entry points. Follow the
[speech guide](docs/qwen3_audio.md) for downloads and commands.

## Quick start

Follow the [unified installation, inference and development guide](docs/getting_started.md)
to install Miniconda3, create Python 3.12, install dependencies and run
Qwen3-0.6B FP16. The first deployment and later editable/wheel development use
one `RHINOFORGE_ENV` and the same guide.

For slow downloads or offline boards, use the guide's
[complete Conda environment bundle](docs/getting_started.md#6-deploy-a-complete-environment-bundle)
to download and unpack the matching Python, dependencies and native runtime
at once. Prepare model weights, the board OS/driver and any SDK components not
listed in the bundle separately. Download availability and compatible platforms
follow the provider's actual delivery.

## Documentation

| Topic | Document |
|---|---|
| v1.2 changes and installation | [Changelog](CHANGELOG.md) · [Upgrade guide](docs/upgrading_1_2.md) |
| Installation, inference and development | [Unified guide](docs/getting_started.md) |
| Speech recognition and synthesis | [Qwen3 ASR/TTS](docs/qwen3_audio.md) |
| Examples | [Example directory and entry points](examples/README.md) |
| Restricted Launch and operator assets | [Restricted runtime assets](docs/runtime_assets.md) |
| Model checkpoints | [Model assets](docs/model_assets.md) |
| Runtime and TOML parameters | [Runtime configuration](docs/runtime_config.md) |
| Model execution, testing, and profiling | [Run and debug models](docs/model_testing.md) |
| Proportionate validation | [Model validation policy](docs/validation_policy.md) |
| Runtime design | [Architecture](docs/architecture.md) |
| Python and C++ APIs | [API reference](docs/api_reference.md) |
| Port another model | [Model porting](docs/model_porting.md) |
| Offline quantization | [Quantization](docs/quantization.md) |
| Report a vulnerability | [Security policy](SECURITY.md) |
| Contribute | [Contributing](CONTRIBUTING.md) |

## License

RhinoForge source code is licensed under the
[Apache License 2.0](LICENSE). Model checkpoints, Rhino Launch, the main and
expansion operator assets, and other third-party materials are governed by their own terms
and are not licensed by this repository's `LICENSE` file.
