# Third-party notices

RhinoForge is licensed under Apache-2.0. That license does not replace the
licenses or distribution terms of the software and assets used with it.

The following dependencies and adapted upstream sources are declared by
`pyproject.toml`, linked by the checked-in CMake build, or cited by adapter
source. Consult the linked upstream license text for the exact version used.

| Component | Use | License or terms |
|---|---|---|
| [Hugging Face Accelerate](https://github.com/huggingface/accelerate) | Device-aware Hugging Face model loading | [Apache License 2.0](https://github.com/huggingface/accelerate/blob/main/LICENSE) |
| [PyTorch](https://github.com/pytorch/pytorch) | Python and C++ tensor runtime | [Upstream license](https://github.com/pytorch/pytorch/blob/main/LICENSE) |
| [NumPy](https://github.com/numpy/numpy) | Array utilities | [Upstream license](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| [safetensors](https://github.com/safetensors/safetensors) | Checkpoint I/O | [Upstream license](https://github.com/safetensors/safetensors/blob/main/LICENSE) |
| [Transformers](https://github.com/huggingface/transformers) | Hugging Face model APIs | [Upstream license](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Checkpoint retrieval used by quantized loaders | [Upstream license](https://github.com/huggingface/huggingface_hub/blob/main/LICENSE) |
| [Pillow](https://github.com/python-pillow/Pillow) | Optional vision input support | [Upstream license](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| [TorchVision](https://github.com/pytorch/vision) | Optional vision transforms | [Upstream license](https://github.com/pytorch/vision/blob/main/LICENSE) |
| [Diffusers](https://github.com/huggingface/diffusers) | Optional NavDP scheduling | [Upstream license](https://github.com/huggingface/diffusers/blob/main/LICENSE) |
| [LeRobot](https://github.com/huggingface/lerobot) | Optional Pi0.5/VLA integration | [Upstream license](https://github.com/huggingface/lerobot/blob/main/LICENSE) |
| [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T/tree/23ace64f17aa5015259b8609d371eb61a357c776) | Optional GR00T adapter reference and public preprocessing contract | [Apache License 2.0](https://github.com/NVIDIA/Isaac-GR00T/blob/23ace64f17aa5015259b8609d371eb61a357c776/LICENSE) |
| [NVIDIA GR00T-N1.7 checkpoints](https://huggingface.co/nvidia/GR00T-N1.7-3B) | User-supplied model assets; not redistributed by RhinoForge | NVIDIA Open Model License and the access terms delivered by the model owner |
| [NVIDIA Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B) | Gated, user-supplied GR00T processor/reference dependency; not redistributed by RhinoForge | NVIDIA Open Model License; access and license acceptance are required from the model owner |
| [OpenGalaxea G05 / GalaxeaVLA](https://github.com/OpenGalaxea/GalaxeaVLA) | Optional public G0.5 continuous/discrete/CoT adapter reference and user-supplied checkpoint | G0.5 Community License Agreement (Non-Commercial + Limited Patent License); see the pinned upstream [LICENSE-G0.5](https://github.com/OpenGalaxea/GalaxeaVLA/blob/main/LICENSE-G0.5) and [NOTICE](https://github.com/OpenGalaxea/GalaxeaVLA/blob/main/NOTICE). Not redistributed by RhinoForge |
| [LingBot-VLA-V2 Normalizer](https://github.com/Robbyant/lingbot-vla-v2/blob/951475ae1b1d87553e7dc47c97b53a3d695c0d13/lingbotvla/data/vla_data/transform.py) | Adapted action de-normalization formulas | [Apache License 2.0](https://github.com/Robbyant/lingbot-vla-v2/blob/951475ae1b1d87553e7dc47c97b53a3d695c0d13/LICENSE) |
| [InternNav](https://github.com/InternRobotics/InternNav/tree/7a5c62400ac45b313d9b709c740b64191556a242) | InternVLA goal conditioning and optional NavDP integration | [MIT, Copyright 2025 Intern Robotics](LICENSES/InternNav-MIT.txt) |
| [OmegaConf](https://github.com/omry/omegaconf) | Optional VLA configuration | [Upstream license](https://github.com/omry/omegaconf/blob/main/LICENSE) |
| [PyYAML](https://github.com/yaml/pyyaml) | Optional VLA configuration loading | [Upstream license](https://github.com/yaml/pyyaml/blob/main/LICENSE) |
| [SciPy](https://github.com/scipy/scipy) | Optional VLA action conversion | [Upstream license](https://github.com/scipy/scipy/blob/main/LICENSE.txt) |
| [scikit-build-core](https://github.com/scikit-build/scikit-build-core) | Python/CMake build backend | [Upstream license](https://github.com/scikit-build/scikit-build-core/blob/main/LICENSE) |
| [build](https://github.com/pypa/build), [pytest](https://github.com/pytest-dev/pytest) | Optional development tools | See each upstream repository's license |

## Adapted source

- `python/rpu_backend/adapters/internvla_n1/goal_glue.py` adapts positional
  encoding and token compression from InternNav's
  [`navdp_backbone.py`](https://github.com/InternRobotics/InternNav/blob/7a5c62400ac45b313d9b709c740b64191556a242/internnav/model/encoder/navdp_backbone.py).
  RhinoForge uses explicit attention projections and strict checkpoint loading.
  The original MIT license is included in [LICENSES/InternNav-MIT.txt](LICENSES/InternNav-MIT.txt).
- `python/rpu_backend/api/action_denorm.py` adapts LingBot-VLA-V2 Normalizer
  formulas from the revision listed above, adding action-group metadata and
  checkpoint-stat validation. Its Apache-2.0 license text is included in [LICENSE](LICENSE).
- The separately obtained InternVLA NextDiT source includes Apache-2.0 code
  credited to Copyright 2024 Alpha-VLLM Authors and The HuggingFace Team in
  [`nextdit_traj.py`](https://github.com/InternRobotics/InternNav/blob/7a5c62400ac45b313d9b709c740b64191556a242/internnav/model/basemodel/internvla_n1/nextdit_traj.py).
  RhinoForge does not redistribute that external model source.

## External assets

Rhino Launch 1.0.0 is a separately distributed restricted prerequisite. It is
not included in this repository and is governed by the terms delivered with
that package. The versioned main and expansion operator assets named by the matching
runtime set are also distributed separately under its accompanying terms.

RhinoForge assumes a preconfigured board environment. System components
provided by that environment remain governed by the platform terms and are not
delivered by this repository.

Model checkpoints, tokenizers, processors, datasets, and generated quantized
checkpoints are not covered by RhinoForge's Apache-2.0 license. See
[`docs/model_assets.md`](docs/model_assets.md) and the relevant model owner.

## Optional Qwen3 speech integration

ASR audio placeholder and length handling adapt
[QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR/tree/7c6daf77a2421100f5fb066495372c00129d39ff)
source credited to Copyright 2026 The Qwen team, Alibaba Group and the HuggingFace Inc. team,
under [Apache-2.0](https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/LICENSE).
The TTS helper prepares a user-supplied
[QwenLM/Qwen3-TTS checkout](https://github.com/QwenLM/Qwen3-TTS/tree/022e286b98fbec7e1e916cb940cdf532cd9f488e)
with compatibility edits, under
[Apache-2.0, Copyright 2026 Alibaba Cloud](https://github.com/QwenLM/Qwen3-TTS/blob/022e286b98fbec7e1e916cb940cdf532cd9f488e/LICENSE).
Keep upstream copyright and license files with prepared source. Upstream model
source, checkpoint weights and reference audio are obtained separately.

| Audio dependency | License |
|---|---|
| librosa | [ISC](https://github.com/librosa/librosa/blob/0.11.0/LICENSE.md) |
| SoundFile | [BSD-3-Clause](https://github.com/bastibe/python-soundfile/blob/0.13.1/LICENSE) |
| python-sox | [BSD-3-Clause](https://github.com/marl/pysox/blob/master/LICENSE) |
| ONNX Runtime | [MIT](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) |
| einops | [MIT](https://github.com/arogozhnikov/einops/blob/main/LICENSE) |
| TorchAudio | [BSD-2-Clause](https://github.com/pytorch/audio/blob/v2.10.0/LICENSE) |

These packages and their dependencies retain their upstream licenses.
