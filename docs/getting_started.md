# Getting started

[简体中文](getting_started.zh.md) | English

This guide installs the current RhinoForge source in a new Python environment
and runs Qwen3-0.6B FP16. The board OS, driver and SDK remain an explicit platform
prerequisite; they are not installed by the Python package or runtime bundle.

## 1. Prepare the runtime and environment

1. Clone the source into a new directory and record `git rev-parse HEAD`.
2. Follow [Python and compiler preparation](development_build.md#prepare-python-and-the-compiler)
   to install Python 3.12 with headers/venv and select `RHINOFORGE_PYTHON`, `CC`
   and `CXX`. The guide includes a source installation for systems whose package
   repositories do not supply Python 3.12.
3. Follow [runtime verification and nested extraction](runtime_assets.md#configure-paths)
   to set `RHINO_LAUNCH_ROOT`, `RPU_KERNEL_LIB_PATH`,
   `RPU_KERNEL_LIB_PATH_EXPANSION` and library paths. Check the provider's SDK
   installation and [device access](runtime_assets.md#board-sdk-and-device-access).

Use new absolute paths below, and continue in the same shell. Do not use
`--system-site-packages`. Clear inherited overrides even when using a new venv:

```bash
export RHINOFORGE_SOURCE="/absolute/path/to/RhinoForge"
export RHINOFORGE_ENV="/absolute/path/to/new/rhinoforge-env"
export RHINOFORGE_BUILD="/absolute/path/to/new/rhinoforge-build"
unset PYTHONPATH PYTHONHOME RPU_BACKEND_SO
export PYTHONNOUSERSITE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
"$RHINOFORGE_PYTHON" -m venv "$RHINOFORGE_ENV"
. "$RHINOFORGE_ENV/bin/activate"
cd "$RHINOFORGE_SOURCE"

python -m pip install -c constraints/security.txt "torch==2.10.0" "scikit-build-core>=0.12,<0.13" "cmake==4.1.3" "ninja==1.13.0"
python -m pip install . -c constraints/security.txt --no-build-isolation \
  -Cbuild-dir="$RHINOFORGE_BUILD" \
  -Ccmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  -Cbuild.tool-args=-j2
python -m pip check
python - <<'PY'
import sys
from importlib.metadata import version
from importlib.util import find_spec
assert sys.prefix != sys.base_prefix
print("Python:", sys.executable)
print("RhinoForge:", version("rhinoforge"))
print("Package source:", find_spec("rpu_backend").origin)
PY
```

The last check reads package metadata without loading the native extension.
Project dependencies come from [pyproject.toml](../pyproject.toml). Use
`'.[vision]'` for vision, `'.[vla]'` for policies, or `'.[audio]'` for
[speech models](qwen3_audio.md); Qwen3 text FP16 needs none of these extras.
Keep dependency resolution enabled. Editable installs and wheels are covered in
[Development build](development_build.md). If the board has no outbound network,
transfer verified downloads from an allowed host, using packages compatible with
its architecture and Python 3.12; record any mirror used.

## 2. Prepare the public model and local configuration

Download the complete checkpoint (config, tokenizer and weights) at the pinned
revision into a new cache. Its stored weights are BF16; this example loads them
as FP16 and needs no offline conversion. The `hf` CLI is supplied by the project's
`huggingface-hub` dependency.

```bash
export RPU_MODEL_CACHE="/absolute/path/to/new/model-cache"
export RHINOFORGE_RESULTS="/absolute/path/to/new/qwen3-results"
export RHINOFORGE_CONFIG="$RHINOFORGE_RESULTS/qwen3.local.toml"
mkdir -p "$RPU_MODEL_CACHE" "$RHINOFORGE_RESULTS"
hf download Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --local-dir "$RPU_MODEL_CACHE/Qwen3-0.6B"
python - <<'PY'
import json, os
from pathlib import Path
text = Path("examples/configs/qwen3/text/0_6b/fp16.toml").read_text()
text = text.replace("[run]\n", "[run]\noutput_dir = " + json.dumps(os.environ["RHINOFORGE_RESULTS"]) + "\n", 1)
text += '\n[logging]\nlevel = "INFO"\n'
Path(os.environ["RHINOFORGE_CONFIG"]).write_text(text)
PY
python -m examples.run --config "$RHINOFORGE_CONFIG" --check-config
```

`--check-config` checks syntax and profile relationships only; it does not load
weights, resolve shared libraries or test device access. Model licenses and
access terms are separate from RhinoForge's source license. An existing complete
copy at the same pinned revision can also be used, but is not a fresh download.

## 3. Run inference on an idle device

Use the platform scheduler/status tools and coordinate with other users to
confirm the board is idle. Acquire the **same lock used by other board jobs**;
creating an unrelated lock does not protect them. For a platform using `flock`,
set its administrator-provided lock path (for example `/run/lock/rpu.device`
only where that is the established convention):

```bash
export RPU_DEVICE_LOCK="/absolute/path/to/platform/device.lock"
flock --nonblock "$RPU_DEVICE_LOCK" \
  "$RHINOFORGE_ENV/bin/python" -m examples.run --config "$RHINOFORGE_CONFIG"
```

Lock acquisition failure means do not run. Platforms with a scheduler or a
separate launcher should use that mechanism instead. The lock covers the whole
Python process, including native initialization and teardown. Device permissions
must allow `O_RDWR` on an RPU node and `/dev/mem`; see the
[access check](runtime_assets.md#board-sdk-and-device-access). A requested RPU
CausalLM load fails before loading weights when device access or the native
backend is unavailable; CPU-only loading remains supported.

If the administrator's approved execution policy requires `sudo`, use the
**same new environment, assets and model** with an explicit environment. From
the source directory, this is an alternative to the unprivileged command:

```bash
sudo env -i \
  HOME="$HOME" PATH="$RHINOFORGE_ENV/bin:/usr/bin:/bin" \
  PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  RHINO_LAUNCH_ROOT="$RHINO_LAUNCH_ROOT" \
  RPU_KERNEL_LIB_PATH="$RPU_KERNEL_LIB_PATH" \
  RPU_KERNEL_LIB_PATH_EXPANSION="$RPU_KERNEL_LIB_PATH_EXPANSION" \
  RPU_MODEL_CACHE="$RPU_MODEL_CACHE" \
  GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0="$RHINOFORGE_SOURCE" \
  /usr/bin/flock --nonblock "$RPU_DEVICE_LOCK" \
  "$RHINOFORGE_ENV/bin/python" -m examples.run --config "$RHINOFORGE_CONFIG"
```

Review `LD_LIBRARY_PATH` beforehand so it contains only the selected Launch and
required SDK paths. The process-local Git setting permits provenance recording
for this checkout. Privileged execution can create root-owned private result
subdirectories; after the process exits, return only this newly created task's
results to the calling user if the platform permits it:

```bash
sudo chown -R "$(id -u):$(id -g)" "$RHINOFORGE_RESULTS"
```

For an installation-origin check, use the same account and environment as
inference. If using the approved privileged path, retain its explicit `sudo env
-i` environment for this probe as well:

```bash
flock --nonblock "$RPU_DEVICE_LOCK" "$RHINOFORGE_ENV/bin/python" - <<'PYPROBE'
from pathlib import Path
import torch
import rpu_backend

print("Python package:", rpu_backend.__file__)
mapped = {
    line.split(maxsplit=5)[-1]
    for line in Path("/proc/self/maps").read_text().splitlines()
    if "rpu_backend.so" in line
}
print("Loaded native libraries:", *sorted(mapped), sep="\n")
assert torch.rpu.is_available()
PYPROBE
```

The Linux process map lists the native libraries actually loaded. This imports
the native runtime, so keep the platform lock and an idle board. `pip show`
alone does not prove which native library loaded. Restart Python after fixing
permissions or replacing a runtime asset.

## 4. Read the short-case result

The template performs two warmups and three measured inference calls (W2/R3).
With the pinned tokenizer its raw prompt has seven tokens (P7), and D4 means
four decode calls after the prefill token: **at most five generated tokens**,
possibly fewer on EOS. It does not apply a chat template and is a short text
continuation smoke test; repeating part of the prompt is not by itself a device
failure or a full answer to the prompt. Timing excludes loading and warmups.

`INFO` shows generated text; the default `ERROR` hides ordinary output. Normal
Graph BUILD/REPLAY messages are informational. Native output is captured and
replayed by the runner; stderr alone is not an execution-failure signal.
See [example logging and results](../examples/RESULTS.md).

Each run creates a `qwen3-0.6b.fp16.text-*` child under `RHINOFORGE_RESULTS` with
`report.json` and `output.pt` (the last measured output). Native logs are in the
parent results directory. Check the process exit status and report: `PASS` /
`execution.status=COMPLETED` describes execution. The ordinary runner records
`correctness=not_run` and `formal_certification=false`; it does not perform a CPU
reference comparison or establish full numerical/task certification. See
[model validation](validation_policy.md) for those separate checks.

Continue with [Model assets](model_assets.md),
[Runtime configuration](runtime_config.md), and
[Run and debug models](model_testing.md).
