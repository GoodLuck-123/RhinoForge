# Getting started

[简体中文](getting_started.zh.md) | English

This guide installs the current RhinoForge source in a separate Python
environment and runs the public Qwen3-0.6B example. Models and runtime assets are
provided separately. The [examples directory](../examples/README.md) describes
configuration and usage.

## 1. Prepare the runtime and environment

Use Python 3.12, a C++17 compiler and a configured board SDK/runtime. Obtain the
matching external assets and set `RHINO_LAUNCH_ROOT`, `RPU_KERNEL_LIB_PATH`, `RPU_KERNEL_LIB_PATH_EXPANSION` and
library paths as described in [Runtime assets](runtime_assets.md).

Create a new environment for this checkout, separate from other RhinoForge or
`rpu_backend` installations. Run the commands in one shell:

```bash
export RHINOFORGE_SOURCE="/absolute/path/to/RhinoForge"
export RHINOFORGE_ENV="/absolute/path/to/new/rhinoforge-env"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
python3.12 -m venv "$RHINOFORGE_ENV"
. "$RHINOFORGE_ENV/bin/activate"
cd "$RHINOFORGE_SOURCE"

python -m pip install "torch==2.10.0" "scikit-build-core>=0.12,<0.13" "cmake==4.1.3" "ninja==1.13.0"
python -m pip install . -c constraints/security.txt --no-build-isolation \
  -Ccmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  -Cbuild.tool-args=-j2
python -m pip check
```

Project dependencies come from [pyproject.toml](../pyproject.toml). Use
`'.[vision]'` for vision, `'.[vla]'` for policies, or `'.[audio]'` for
[speech models](qwen3_audio.md) in the install command.
For editable installs and building wheels, see [Development build](development_build.md).

To apply the validated network and image dependency updates to an existing environment:

```bash
python -m pip install --upgrade -r constraints/security.txt
python -m pip check
```

## 2. Prepare the public model

Download the public checkpoint to the directory used by the example alias:

```bash
export RPU_MODEL_CACHE="${RPU_MODEL_CACHE:-$HOME/.cache/rhinoforge/models}"
mkdir -p "$RPU_MODEL_CACHE"
hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir "$RPU_MODEL_CACHE/Qwen3-0.6B"
cp examples/configs/qwen3/text/0_6b/fp16.toml qwen3.local.toml
cat >> qwen3.local.toml <<'TOML'

[logging]
level = "INFO"
TOML
python -m examples.run --config qwen3.local.toml --check-config
```

An existing complete copy of that checkpoint can be placed at the same location.
Model licenses and access terms are separate from RhinoForge's source license.
All example configurations are organized by model under
[`examples/configs`](../examples/README.md).

## 3. Run inference on an idle device

Check that the local device is idle and acquire the platform's device lock before
running. On a shared host, coordinate with the board's other users. Then run:

```bash
python -m examples.run --config qwen3.local.toml
```

The `INFO` setting above displays generated text. Without it, the default
`ERROR` level hides ordinary output; `[logging].summary` controls the model
summary. See [example logging](../examples/RESULTS.md).
If imports come from another checkout, activate the intended environment and
run `python -m pip show rhinoforge` to check the installation location. If the
model alias cannot resolve, check `RPU_MODEL_CACHE/Qwen3-0.6B`; runtime asset
errors are covered in [Runtime assets](runtime_assets.md).

Continue with [Model assets](model_assets.md),
[Runtime configuration](runtime_config.md), and
[Run and debug models](model_testing.md).
