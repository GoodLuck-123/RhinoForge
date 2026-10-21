# Install, run and develop RhinoForge

[简体中文](getting_started.zh.md) | English

Use this single guide and one `RHINOFORGE_ENV` for the first model run and later
development. First obtain a supported board platform, then choose one of the two
application installation paths below. Both paths build RhinoForge from a GitHub
checkout and continue with Qwen3-0.6B FP16.

## Platform prerequisites

**The platform provider must supply and confirm all four prerequisites before
application installation begins.** This guide starts from an empty application
environment; it does not provision a bare board.

| Prerequisite | What the platform provider supplies and confirms |
|---|---|
| Board OS | Supported Linux distribution, architecture and kernel combination |
| Driver | Matching RPU driver, device nodes, inference-account permissions and device scheduling/lock convention |
| Board SDK | The complete SDK, including its user-space libraries, installation instructions and library paths |
| Runtime assets | Matching Rhino Launch, main/expansion operator assets and their `.kernels` sidecars, checksums and distribution terms |

Our recommended, validated SDK baseline is **`EA_R1SDK_2026_08F-2156007a`** on
R1 Linux aarch64, Ubuntu 22.04, glibc 2.35 and GCC 11.4. This records the tested
combination; it is not a minimum-version claim or a guarantee for other SDK
versions. Ask the platform provider to confirm the matching OS, kernel and
driver. **Rhino Launch 1.2.0 is a separate runtime version, not the SDK version.**

See [runtime assets and platform setup](runtime_assets.md). The platform supplies
the complete SDK; we do not package SDK libraries. Rhino Launch and the operator
assets are delivered separately from the public dependency environment, under
their own terms. Model weights are prepared separately after application
installation.

## Choose one installation path

| Path | Steps | Use it when |
|---|---|---|
| [A. Build from source](#source-build) | Install Miniconda3, create the environment, install dependencies and compile RhinoForge | Step-by-step dependency installation or a custom environment |
| [B. Use our dependency environment bundle](#bundle-deploy) | Obtain and activate the dependency bundle, then clone and compile RhinoForge | Avoids repeated downloads of Python and third-party dependencies |

Both paths use the same `RHINOFORGE_ENV` for installation, inference and later
development. Path B replaces environment creation and dependency downloads;
it still requires the GitHub source, host compiler, platform SDK and separately
delivered runtime assets. The current environment is
[`deps-2026.10.1` (preview)](https://github.com/HUIXI-AI/RhinoForge/releases/tag/deps-2026.10.1).
See the [environment bundle guide](environment_bundle.md) for contents, known
limitations and the relationship to source versions.

<a id="source-build"></a>
<a id="1-install-miniconda3-and-build-the-source"></a>

## 1. Path A: install Miniconda3 and build from source

### A1. Download and install Miniconda3

These commands target a Linux **aarch64** RPU board using Bash. Check `uname -m`
before downloading; an x86_64 installer or environment bundle cannot run on it.
Use new, user-owned directories with enough space for downloads, the environment,
source, build and model. The examples use `$HOME`; a writable data disk also works.
On Ubuntu/Debian, install the host tools first:

```bash
set -e
sudo apt-get update
sudo apt-get install --yes build-essential pkg-config git curl ca-certificates bzip2 xz-utils

export RHINOFORGE_MINICONDA="$HOME/miniconda3-rhinoforge"
export RHINOFORGE_DOWNLOADS="$HOME/rhinoforge-downloads"
export RHINOFORGE_WORKDIR="$HOME/rhinoforge"
export RHINOFORGE_SOURCE="$RHINOFORGE_WORKDIR/source"
export RHINOFORGE_ENV="$RHINOFORGE_WORKDIR/env"
export RHINOFORGE_BUILD="$RHINOFORGE_WORKDIR/build"
test "$(uname -m)" = aarch64
mkdir -p "$RHINOFORGE_DOWNLOADS"
cd "$RHINOFORGE_DOWNLOADS"
curl --fail --location --retry 3 --continue-at - \
  https://repo.anaconda.com/miniconda/Miniconda3-py312_26.9.3-0-Linux-aarch64.sh \
  --output Miniconda3-py312_26.9.3-0-Linux-aarch64.sh
printf '%s  %s\n' a7631abb0af9f2b28bb2778db6229be982471a6032a16fc5817542ec0b2de91a \
  Miniconda3-py312_26.9.3-0-Linux-aarch64.sh | sha256sum -c -
bash Miniconda3-py312_26.9.3-0-Linux-aarch64.sh -b -p "$RHINOFORGE_MINICONDA"
source "$RHINOFORGE_MINICONDA/etc/profile.d/conda.sh"
conda --version
```

The installer and checksum come from the [official Miniconda archive](https://repo.anaconda.com/miniconda/).
Review the installer terms before using the unattended `-b` option. This installs
under your own prefix, without replacing system Python or editing shell startup
files. In a new shell, source the same `conda.sh` again before using `conda`.

### A2. Create the application environment

Clear overrides from previous installations, then create and activate the
application environment. `--override-channels` makes the selected package source
explicit rather than inheriting a user's channel list. Miniconda supplies the
manager; this command obtains Python from the community `conda-forge` channel.
`CONDA_PKGS_DIRS` keeps this installation's package cache under its own workspace
without changing the global Conda configuration or other users' caches. For audio
workloads, append `ffmpeg libsndfile` to the `conda create` command below, or add
them to this same environment using the [audio tools appendix](#build-bundle).
That appendix also provides the separately built SoX executable for aarch64.

```bash
unset PYTHONPATH PYTHONHOME RHINO_LAUNCH_ROOT CMAKE_PREFIX_PATH LD_LIBRARY_PATH
while IFS= read -r name; do
  case "$name" in RPU_*|LKN_*) unset "$name" ;; esac
done < <(compgen -v)
export PYTHONNOUSERSITE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export CONDA_PKGS_DIRS="$RHINOFORGE_WORKDIR/conda-pkgs"
mkdir -p "$CONDA_PKGS_DIRS"
conda create --yes --override-channels --channel conda-forge \
  --prefix "$RHINOFORGE_ENV" python=3.12.15 "pip>=23.1" \
  "packaging>=24.2,<26" "setuptools>=71,<81"
conda activate "$RHINOFORGE_ENV"
python - <<'PY'
import os, pathlib, ssl, sys, sysconfig
from importlib.metadata import version
from packaging.version import Version
assert sys.version_info[:2] == (3, 12), sys.version
assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(os.environ["RHINOFORGE_ENV"]).resolve()
header = pathlib.Path(sysconfig.get_path("include")) / "Python.h"
assert header.is_file(), f"Missing Python development header: {header}"
print("Python:", sys.executable, sys.version)
assert Version(version("pip")) >= Version("23.1")
assert Version("24.2") <= Version(version("packaging")) < Version("26")
assert Version("71") <= Version(version("setuptools")) < Version("81")
print("pip:", version("pip"))
print("packaging:", version("packaging"))
print("setuptools:", version("setuptools"))
print("Headers:", header)
print("TLS:", ssl.OPENSSL_VERSION)
PY
```

Do not run these installs in Conda's `base` environment. Conda environments do
not use the `sys.prefix != sys.base_prefix` test for venvs. The check above
verifies the actual prefix instead. Use `python -m pip` throughout so pip belongs
to this interpreter. `--config-settings` below is pip's build-setting option;
`-c` is its separate dependency-constraints option.

Install Conda packages before pip packages. LeRobot 0.5.1 requires
`packaging>=24.2,<26` and `setuptools>=71,<81`; these ranges are selected by Conda
even for the base text environment, so adding the full extras later preserves
the same environment. Installing newer Conda versions of those packages and
letting pip downgrade them overwrites Conda-managed files, which makes
`conda-pack` fail. Keep pip, packaging and setuptools managed by Conda; the
selected pip already supports the required build options and needs no pip-driven
upgrade. A successful `pip check` alone does not detect overwritten Conda files.

### A3. Obtain the source and configure the supplied runtime

```bash
git clone https://github.com/HUIXI-AI/RhinoForge.git "$RHINOFORGE_SOURCE"
cd "$RHINOFORGE_SOURCE"
git rev-parse HEAD
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
"$CC" --version
"$CXX" --version
printf '#include <optional>\nint main() { std::optional<int> x{1}; return *x - 1; }\n' \
  | "$CXX" -std=c++17 -x c++ -fsyntax-only -
```

Read the guide from the checked-out revision. For a provider's pinned source
release, check out that revision before proceeding and retain its commit ID.
Use a new build directory after changing the compiler. Python's default compiler
settings alone do not establish that the compiler exists on this board.

Follow [runtime verification and nested extraction](runtime_assets.md#configure-paths)
to set `RHINO_LAUNCH_ROOT`, `RPU_KERNEL_LIB_PATH`,
`RPU_KERNEL_LIB_PATH_EXPANSION` and the selected Launch/SDK library paths.
Current `1.3.0.dev0` source requires the archived sanitized **Release Launch 1.2.0**
pinned in [runtime-dependencies.json](../release/runtime-dependencies.json),
not a same-version development package. Verify both operator assets and their
adjacent `.kernels` files against that manifest. Complete the provider's
[SDK and device access checks](runtime_assets.md#board-sdk-and-device-access).
A Conda environment does not install the SDK or kernel driver.

### A4. Install dependencies and build RhinoForge

Continue in the same shell and source directory. This compiles and installs
RhinoForge into the selected environment. The current [pyproject.toml](../pyproject.toml)
requires PyTorch **2.10.0** and Transformers **5.3.0**. Keep dependency resolution
enabled during the first install; `constraints/security.txt` is a security
constraint file, not a complete lock of every dependency.

```bash
cd "$RHINOFORGE_SOURCE"
python -m pip install -c constraints/security.txt \
  "torch==2.10.0" "scikit-build-core>=0.12,<0.13" "cmake==4.1.3" "ninja==1.13.0"
python -m pip install . -c constraints/security.txt --no-build-isolation \
  --config-settings=build-dir="$RHINOFORGE_BUILD" \
  --config-settings=cmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  --config-settings=build.tool-args=-j2
python -m pip check
python - <<'PY'
from importlib.metadata import version
from importlib.util import find_spec
assert version("transformers") == "5.3.0"
assert version("torch").split("+")[0] == "2.10.0"
print("RhinoForge:", version("rhinoforge"))
print("Python package:", find_spec("rpu_backend").origin)
PY
```

The last check does not load the native extension or initialize a device. Two
build workers limit compiler memory use. For vision, VLA or speech, replace `.`
in the source-install command with `'.[vision]'`, `'.[vla]'` or `'.[audio]'`;
`'.[vision,vla,audio]'` installs all three optional dependency groups. Qwen3 text
FP16 requires none of them. See the [speech guide](qwen3_audio.md) for its inputs.

If downloads are slow or the board is offline, a mirror or wheel cache must
supply packages compatible with Python 3.12 and the target architecture; record
its source instead of changing the pinned Torch/Transformers versions. The
dependency bundle in [path B](#bundle-deploy) avoids individual downloads. The source installation also supports
[editable development and wheel builds](#development).

Path A is complete. Skip path B and continue with [model preparation](#model-setup).

<a id="bundle-deploy"></a>
<a id="6-deploy-a-complete-environment-bundle"></a>

## 2. Path B: use the dependency Conda environment bundle

### B1. Obtain source and its matching environment version

Dependency bundles have independent GitHub Releases. The full profile contains
Python 3.12, default/vision/vla/audio/dev dependencies and build/media tools.
RhinoForge, SDK libraries, Launch, operators and models remain separate; build
and develop RhinoForge in your own Git checkout.

Use the [GitHub Release](https://github.com/HUIXI-AI/RhinoForge/releases/tag/deps-2026.10.1)
identified by the checkout's [`release/environment.json`](../release/environment.json).
This first version is a preview; read its `KNOWN_ISSUES.md` before deployment.
Its support boundary uses reviewed local models, media and datasets.

The host needs Bash, Python 3.10 or newer and system C/C++ build tools, but no
Miniconda manager or conda create step. On Ubuntu/Debian, prepare these tools
and clone the source. If the provider specifies a tag/commit, check it out before
reading its environment binding:

```bash
set -e
sudo apt-get update
sudo apt-get install --yes build-essential pkg-config git curl ca-certificates bzip2 xz-utils python3
export RHINOFORGE_WORKDIR="$HOME/rhinoforge"
export RHINOFORGE_SOURCE="$RHINOFORGE_WORKDIR/source"
export RHINOFORGE_BUILD="$RHINOFORGE_WORKDIR/build"
export RHINOFORGE_DEPENDENCIES="$RHINOFORGE_WORKDIR/dependencies"
test "$(uname -m)" = aarch64
mkdir -p "$RHINOFORGE_WORKDIR"
git clone https://github.com/HUIXI-AI/RhinoForge.git "$RHINOFORGE_SOURCE"
cd "$RHINOFORGE_SOURCE"
git rev-parse HEAD
```

The checkout's `release/environment.json` binds source to the environment ID,
platform, requirements digest, download URL and SHA256. Source changes that
preserve the normalized dependency recipe and ABI can reuse the same archive.

### B2. Download, verify and deploy dependencies

Continue in the same shell. Read the actual URL and checksum from the checkout;
the standalone installer downloads, checks, extracts, relocates and runs pip
check. The installer uses host Python; builds and inference use bundled Python 3.12:

```bash
cd "$RHINOFORGE_SOURCE"
mapfile -t rf_bundle_fields < <(python3 - <<'PYENV'
import json
from pathlib import Path
record = json.loads(Path("release/environment.json").read_text())
print(record["archive_url"])
print(record["archive_sha256"])
PYENV
)
python3 scripts/install_dependency_env.py \
  --archive "${rf_bundle_fields[0]}" \
  --sha256 "${rf_bundle_fields[1]}" \
  --prefix "$RHINOFORGE_DEPENDENCIES"
unset rf_bundle_fields
source "$RHINOFORGE_DEPENDENCIES/activate.sh"
python scripts/dependency_environment.py \
  --source "$RHINOFORGE_SOURCE" --prefix "$RHINOFORGE_ENV"
```

The installation directory must be new, with an existing writable parent. For an
offline board, transfer the archive and source first, then use the local archive
path with the same published SHA256. Installation does not download pip/Conda
packages, install RhinoForge or access the device.

Activation sets `RHINOFORGE_ENV="$RHINOFORGE_DEPENDENCIES/env"`, clears stale
Python/native overrides and preserves platform library paths. In a new shell,
set source/work paths and activate again. The installer checks archive identity,
architecture and glibc; confirm SDK, driver and device permissions separately.
Install directly at the final path; to move later, reinstall from the archive.

### B3. Configure separate runtime assets and build RhinoForge

Follow the [runtime guide](runtime_assets.md#configure-paths) to set
`RHINO_LAUNCH_ROOT`, `RPU_KERNEL_LIB_PATH`, `RPU_KERNEL_LIB_PATH_EXPANSION` and
platform SDK loader paths, and complete the [SDK/access checks](runtime_assets.md#board-sdk-and-device-access).
The dependency environment supplies none of those components.

After dependency compatibility checks pass, build in the same shell.
`--no-index --no-deps --no-build-isolation` uses the verified environment without
redownloading Torch or silently changing dependencies:

```bash
cd "$RHINOFORGE_SOURCE"
export CC=/usr/bin/gcc CXX=/usr/bin/g++
test -r "$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch/rhino_launchConfig.cmake"
python -m pip install '.[vision,vla,audio,dev]' --no-index --no-deps --no-build-isolation \
  --config-settings=build-dir="$RHINOFORGE_BUILD" \
  --config-settings=cmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  --config-settings=build.tool-args=-j2
python -m pip check
python -c 'from importlib.metadata import version; print("RhinoForge:", version("rhinoforge"))'
```

One-command deployment covers dependencies only. RF still needs a build and
separate assets; offline use also requires system tools, SDK, Launch/operators,
models and inputs in advance. Continue with [model preparation](#model-setup)
and inference, then [editable development](#development) in your own Git branch
using the same environment. See [bundle contents, redistribution materials and
known limitations](environment_bundle.md).

<a id="model-setup"></a>
<a id="2-prepare-the-public-model-and-local-configuration"></a>

## 3. Prepare the public model and local configuration

Download the complete checkpoint (config, tokenizer and weights) at the pinned
revision into a new cache. Its stored weights are BF16; this example loads them
as FP16 and needs no offline conversion. The `hf` CLI is supplied by the project's
`huggingface-hub` dependency.

```bash
cd "$RHINOFORGE_SOURCE"
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

<a id="run-inference"></a>
<a id="3-run-inference-on-an-idle-device"></a>

## 4. Run inference on an idle device

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

Review `LD_LIBRARY_PATH` beforehand: retain the current `$RHINOFORGE_ENV/lib`,
selected Launch and required SDK paths, and remove stale environment paths. The process-local Git setting permits provenance recording
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

<a id="read-results"></a>
<a id="4-read-the-short-case-result"></a>

## 5. Read the short-case result

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

<a id="development"></a>
<a id="5-develop-in-the-same-environment"></a>

## 6. Develop in the same environment

Keep the same `RHINOFORGE_ENV`, source checkout and matching runtime. To make
Python edits take effect directly from the checkout, replace the regular install
with an editable one; native changes still require rerunning the build. Keep the
same system compiler settings, `CC=/usr/bin/gcc` and `CXX=/usr/bin/g++`. Create your
development branch from the selected GitHub tag/commit; the dependency archive
does not contain a separate source snapshot. After updating the checkout, check
its dependency and runtime requirements before rebuilding.

Path B already includes `dev`: rerun the compatibility check from B2, then add
`--no-index --no-deps` to the following `pip install` command to keep the bundled
dependencies unchanged. Path A uses the command as shown to resolve and install
any missing development dependencies.

```bash
cd "$RHINOFORGE_SOURCE"
python -m pip install -e '.[dev]' -c constraints/security.txt --no-build-isolation \
  --config-settings=build-dir="$RHINOFORGE_BUILD" \
  --config-settings=cmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  --config-settings=build.tool-args=-j2
python -m pip check
```

Use `'.[dev,vision]'`, `'.[dev,vla]'` or `'.[dev,audio]'` when the selected entry
requires those extras. Do not create a second development environment or change
the `RHINOFORGE_ENV` name. To build a distributable wheel after dependencies are
installed, select an artifact directory outside the source tree:

```bash
export RHINOFORGE_WHEELS="$RHINOFORGE_WORKDIR/wheels"
python -m pip wheel . -c constraints/security.txt --no-build-isolation --no-deps \
  --wheel-dir "$RHINOFORGE_WHEELS" \
  --config-settings=build-dir="$RHINOFORGE_BUILD" \
  --config-settings=cmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  --config-settings=build.tool-args=-j2
```

Record the source commit/diff, resolved dependencies, wheel/native hashes and
Launch/operator/SDK identities. Validate the wheel in a fresh deployment prefix.
Rebuild it after Python as well as native changes; an unchanged `.so` hash does
not mean an older wheel contains the current Python code. Keep wheels and
editable checkouts separate from the public dependency bundle: that bundle
must not contain any installed RhinoForge package or source.

<a id="build-bundle"></a>

## Appendix: optional audio tools and bundle maintenance

The following commands extend the Miniconda-managed application environment
from [path A](#source-build) for audio/video workloads. Qwen3 text FP16 does not
require these tools. Install Conda packages before Python extras so their
dependency resolution finishes first. Keep path A's packaging/setuptools
ranges; do not let pip replace Conda's copies. These commands install RhinoForge
with optional extras for local use; they are not a public-bundle build recipe:

```bash
conda install --yes --override-channels --channel conda-forge \
  --prefix "$RHINOFORGE_ENV" ffmpeg libsndfile \
  "packaging>=24.2,<26" "setuptools>=71,<81"
cd "$RHINOFORGE_SOURCE"
python -m pip install '.[vision,vla,audio]' -c constraints/security.txt --no-build-isolation \
  --config-settings=build-dir="$RHINOFORGE_BUILD" \
  --config-settings=cmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  --config-settings=build.tool-args=-j2
ffmpeg -version
```

The validated aarch64 setup builds SoX separately rather than obtaining it from
`conda-forge`. Build the [official SoX 14.4.2 source](https://sourceforge.net/projects/sox/files/sox/14.4.2/)
into the **same** `RHINOFORGE_ENV` instead of relying on a pre-existing system
executable. This build embeds static `libsox` in the CLI, provides built-in
PCM/WAV and effects/resampling, and disables external codecs, plugins and audio
devices. It does not add MP3 or other disabled codecs to SoX. Use the separately
packaged FFmpeg/libsndfile for the formats supported by those libraries; their
presence does not enable codecs in this SoX build.

The subshell below keeps Conda's active environment unchanged while preventing
SoX's build from picking up unrelated libraries. Retain the source archive and
license files inside the environment for later inventory and redistribution review.

```bash
export RHINOFORGE_DOWNLOADS="$RHINOFORGE_WORKDIR/downloads"
export RHINOFORGE_SOX_BUILD="$RHINOFORGE_WORKDIR/sox-build"
mkdir "$RHINOFORGE_SOX_BUILD"
mkdir -p "$RHINOFORGE_DOWNLOADS"
curl --fail --location --retry 3 \
  https://downloads.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.gz \
  --output "$RHINOFORGE_DOWNLOADS/sox-14.4.2.tar.gz"
printf '%s  %s\n' b45f598643ffbd8e363ff24d61166ccec4836fea6d3888881b8df53e3bb55f6c \
  "$RHINOFORGE_DOWNLOADS/sox-14.4.2.tar.gz" | sha256sum -c -
(
  export PATH=/usr/bin:/bin
  unset CPPFLAGS LDFLAGS LIBS CFLAGS LD_LIBRARY_PATH
  tar -xzf "$RHINOFORGE_DOWNLOADS/sox-14.4.2.tar.gz" -C "$RHINOFORGE_SOX_BUILD"
  cd "$RHINOFORGE_SOX_BUILD/sox-14.4.2"
  CC=/usr/bin/gcc CFLAGS=-O2 PKG_CONFIG=/usr/bin/false ./configure \
    --prefix="$RHINOFORGE_ENV" \
    --disable-shared --enable-static --disable-openmp \
    --without-libltdl --without-magic --without-png --without-ladspa \
    --without-mad --without-id3tag --without-lame --without-twolame \
    --without-oggvorbis --without-opus --without-flac \
    --without-amrwb --without-amrnb --without-wavpack \
    --without-sndio --without-coreaudio --without-alsa --without-ao \
    --without-pulseaudio --without-waveaudio --without-sndfile \
    --without-oss --without-sunaudio --without-mp3 \
    --without-gsm --without-lpc10 \
    --with-distro='RhinoForge portable PCM tooling'
  make -j2
  make install
  mkdir -p "$RHINOFORGE_ENV/share/licenses/sox" "$RHINOFORGE_ENV/share/sources/sox"
  cp COPYING LICENSE.GPL LICENSE.LGPL "$RHINOFORGE_ENV/share/licenses/sox/"
  cp "$RHINOFORGE_DOWNLOADS/sox-14.4.2.tar.gz" "$RHINOFORGE_ENV/share/sources/sox/"
  cat > "$RHINOFORGE_ENV/share/sources/sox/SOURCE.txt" <<'SOX_SOURCE'
SoX 14.4.2
Source: https://downloads.sourceforge.net/project/sox/sox/14.4.2/sox-14.4.2.tar.gz
SHA256: b45f598643ffbd8e363ff24d61166ccec4836fea6d3888881b8df53e3bb55f6c
Static libsox linked into the CLI; optional external codecs, device backends,
plugins and OpenMP disabled. Built-in PCM/WAV formats and effects are included.
See share/licenses/sox/COPYING, LICENSE.GPL and LICENSE.LGPL.
SOX_SOURCE
)
"$RHINOFORGE_ENV/bin/sox" --version
"$RHINOFORGE_ENV/bin/sox" -n -r 16000 -c 1 -b 16 \
  "$RHINOFORGE_SOX_BUILD/tone-16k.wav" synth 0.05 sine 440
"$RHINOFORGE_ENV/bin/sox" "$RHINOFORGE_SOX_BUILD/tone-16k.wav" \
  -r 8000 "$RHINOFORGE_SOX_BUILD/tone-8k.wav"
"$RHINOFORGE_ENV/bin/sox" --i "$RHINOFORGE_SOX_BUILD/tone-8k.wav"
```

The short command sequence writes 800 samples of 16-bit mono PCM at 16 kHz,
then resamples to 400 samples at 8 kHz. It checks this CLI path only; it does not
certify every audio model or codec.

For public bundle maintenance, create a **new, clean dependency environment
that has never had RhinoForge installed**. Resolve its third-party dependencies
from the selected source revision without installing the project itself. Do not
strip the existing internal full bundle or pack this application environment:
removing a few directories does not establish that code, runtime assets and SDK
libraries are absent.

Keep `CONDA_PKGS_DIRS` and the downloaded package cache until packing finishes:
`conda-pack` may need those records to repair prefixes. Retain the resolved
package/build inventory, source URLs, checksums, license notices and required
corresponding-source materials. Build tools and media libraries have their own
redistribution terms. Public availability of a dependency alone does not settle
the requirements for redistributing its binary build.

The [environment bundle design](environment_bundle.md) defines the separate
release version, source compatibility mapping, archive exclusions and validation
required before publishing. Validate relocation into a different absolute
prefix, build a fresh GitHub checkout there with the platform SDK and separately
provided runtime assets, then exercise the intended model paths. The existing
full-bundle scripts do not implement that public dependency-only workflow.
