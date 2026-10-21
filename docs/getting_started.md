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
| [A. Build from source](#source-build) | Install Miniconda3, create the environment, install dependencies and compile RhinoForge | Available now, including for development |
| [B. Use our dependency environment bundle](#bundle-deploy) | Obtain and activate the dependency bundle, then clone and compile RhinoForge | Planned: avoids repeated downloads of Python and third-party dependencies |

Both paths use the same `RHINOFORGE_ENV` for installation, inference and later
development. Path B will replace environment creation and dependency downloads;
it still requires the GitHub source, host compiler, platform SDK and separately
delivered runtime assets. **The new dependency-only bundle is not yet built or
published. Use path A for executable installation instructions today.** See the
[environment bundle design](environment_bundle.md) for its release and source
version relationship.

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
planned dependency bundle in [path B](#bundle-deploy) will cover this use case
once its release is available. The source installation also supports
[editable development and wheel builds](#development).

Path A is complete. Skip path B and continue with [model preparation](#model-setup).

<a id="bundle-deploy"></a>
<a id="6-deploy-a-complete-environment-bundle"></a>

## 2. Path B: use our dependency environment bundle (planned)

The public bundle will contain **Python 3.12 and third-party dependencies and
tools only**. Its `full` profile covers the default, `vision`, `vla`, `audio` and
`dev` dependencies, plus build and media tools such as CMake, Ninja, conda-pack,
SoX, FFmpeg and libsndfile. It will contain no RhinoForge source, installed
RhinoForge package, native extension or wheel; no SDK libraries; and no Rhino
Launch, operator assets or model weights. The host compiler remains a platform
prerequisite. A complete dependency profile does not establish validation of
every model.

### B1. Obtain a compatible dependency release

**No new dependency-only archive, public download URL or installer is available
yet.** The intended delivery location is this repository's GitHub Releases after
a clean dependency-only build, redistribution review and relocation/build
validation. The environment will have its own version. Each supported RhinoForge
source revision will identify the compatible environment release, manifest and
SHA256; a source change will not automatically require a new environment release.
See [environment bundle contents and release design](environment_bundle.md).

The existing internal full bundle contains installed RhinoForge and restricted
runtime assets. It is not the planned public dependency bundle, and must not be
uploaded as one. The current `scripts/pack_conda_bundle.py` and
`scripts/install_conda_bundle.py` implement that older full-bundle format;
they are not installation instructions for the new format.

### B2. Planned deployment and development flow

Once the release and its installer are available:

1. Obtain the dependency archive and its published checksum and manifest for
   the board architecture. Install into a new final prefix, complete relocation,
   and activate it as `RHINOFORGE_ENV`. The target will not need a separate
   Miniconda manager to use the unpacked environment.
2. Clone RhinoForge from GitHub, check out the tag or commit selected for the
   deployment, and verify that its environment requirement matches the archive.
   Keep the source and build directories outside the dependency prefix.
3. Install the host build tools from [A1](#a1-download-and-install-miniconda3),
   select the compiler, and configure the platform SDK and separately delivered
   Launch/operator assets as in [A3](#a3-obtain-the-source-and-configure-the-supplied-runtime).
4. Build and install that RhinoForge checkout in `RHINOFORGE_ENV` using its
   verified dependency set, then continue with [model preparation](#model-setup).
5. For development, use the same checkout and environment with an
   [editable installation](#development). Rebuild after native changes. When
   updating the source, recheck its dependency and runtime requirements before
   reusing the environment.

The release instructions will supply the exact download, checksum verification,
relocation and build commands. This section describes the agreed design, not a
working installation route today. An aarch64 environment is not portable to
x86_64 or every Linux/SDK combination. Moving a relocated environment will
require a fresh installation into its new final prefix; see
[conda-pack's relocation constraints](https://conda.github.io/conda-pack/).

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
with an editable one; native changes still require rerunning the build. A future
dependency-bundle installation will use this same workflow. After updating the
checkout, check its dependency and runtime requirements before rebuilding.

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
