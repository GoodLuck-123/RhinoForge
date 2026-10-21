# Install, run and develop RhinoForge

[简体中文](getting_started.zh.md) | English

Use this single guide and one `RHINOFORGE_ENV` for the first model run and later
development. It installs Miniconda3, creates a Python 3.12 environment, builds the
current source, and runs Qwen3-0.6B FP16. The board OS, driver and SDK must already
be provisioned by the platform provider.

If your provider supplies a matching [complete environment bundle](#6-deploy-a-complete-environment-bundle),
use that route to avoid downloading Torch and other dependencies separately or
compiling on the target board. It already contains Python; skip the Miniconda
installation, environment creation and source build below. Both routes use the
same environment variable names and continue at step 2.

## 1. Install Miniconda3 and build the source

### Install Miniconda3

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

### Create the one environment

Clear overrides from previous installations, then create and activate the
application environment. `--override-channels` makes the selected package source
explicit rather than inheriting a user's channel list. Miniconda supplies the
manager; this command obtains Python from the community `conda-forge` channel.
`CONDA_PKGS_DIRS` keeps this installation's package cache under its own workspace
without changing the global Conda configuration or other users' caches. For a
`full` bundle, append `ffmpeg libsndfile` to the `conda create` command below;
otherwise add them to this same environment in step 6. Do not request `sox` from
`conda-forge` on aarch64; its separate build is covered in step 6.

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

### Obtain the source and matching runtime

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

### Install dependencies and RhinoForge

Continue in the same shell and source directory. This installs a regular wheel,
which can later be packed and relocated. The current [pyproject.toml](../pyproject.toml)
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

If downloads are slow or the board is offline, use the
[complete environment bundle](#6-deploy-a-complete-environment-bundle) built for
the same architecture and runtime baseline. A mirror or wheel cache must supply
packages compatible with Python 3.12 and the target architecture; record its
source instead of changing the pinned Torch/Transformers versions. Once the
source installation works, the same environment supports
[editable development and wheel builds](#5-develop-in-the-same-environment).

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

## 5. Develop in the same environment

Keep the same `RHINOFORGE_ENV`, source checkout and matching runtime. To make
Python edits take effect directly from the checkout, replace the regular install
with an editable one; native changes still require rerunning the build. If you
started from a bundle, first install the host build tools listed in step 1 and
set `CC=/usr/bin/gcc` and `CXX=/usr/bin/g++`; keep the installed environment.


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
not mean an older wheel contains the current Python code. Before packing an
environment, reinstall the chosen wheel normally: `conda-pack` cannot relocate
an editable checkout dependency.

## 6. Deploy a complete environment bundle

A bundle is a snapshot of a working **Conda environment**, plus the matching
source/examples, compiled RhinoForge, sanitized Launch and the main/expansion
operator assets with their sidecars. The `full` profile includes the `vision`,
`vla` and `audio` Python extras, the SoX executable and the Conda packages
`ffmpeg` and `libsndfile`, including the audio/video command-line tools and native
libraries. The Python `sox` package alone does not provide the SoX executable.
The `text` profile includes the base text dependencies only.
Consult the supplied `bundle-manifest.json` for the actual profile, versions,
architecture and build platform. Optional SDK user-space libraries are included
only when the manifest lists them. Model weights, the OS and kernel driver are
not part of this environment package.

The bundle is a local validation artifact until a provider publishes its download
URL, SHA256 and supported board/SDK baseline. This repository does not invent a
public download URL. Restricted runtime components retain their own distribution
terms; a source-code license does not grant permission to redistribute them.
A full dependency profile also does not establish validation of every model.

The supplied examples run against the matching installed wheel; no local
`build/` directory or editable installation is needed. Prepare each model's
checkpoint, input/reference fixtures and any required third-party source
separately, following that model's README and supported profile.

### Download and install in one command

Obtain `scripts/install_conda_bundle.py` from the same source revision (or the
provider's delivery) and put it outside the destination directory. The installer
requires system Python 3.10 or newer; the bundle supplies its own Python 3.12 for
inference, so no Miniconda installation or `conda create` is needed on the target.
Set the actual provider URL and checksum, or use a local archive path after
transferring the file to an offline board:

```bash
export RHINOFORGE_WORKDIR="$HOME/rhinoforge"
export RHINOFORGE_BUNDLE_ARCHIVE="/absolute/path/to/rhinoforge-conda.tar.gz"
export RHINOFORGE_BUNDLE_SHA256="REPLACE_WITH_PROVIDER_SHA256"
python3 /absolute/path/to/install_conda_bundle.py \
  --archive "$RHINOFORGE_BUNDLE_ARCHIVE" \
  --sha256 "$RHINOFORGE_BUNDLE_SHA256" \
  --prefix "$RHINOFORGE_WORKDIR"
source "$RHINOFORGE_WORKDIR/activate.sh"
cd "$RHINOFORGE_SOURCE"
python -m pip check
printf 'Python environment: %s\nSource: %s\n' "$RHINOFORGE_ENV" "$RHINOFORGE_SOURCE"
```

`--archive` also accepts an HTTP(S) URL: the same command downloads, verifies and
installs it. The destination must not already exist. The installer does not
install packages from the network, compile RhinoForge or access the RPU device.
Activation selects the packaged paths, including
`RHINOFORGE_ENV="$RHINOFORGE_WORKDIR/env"`. Continue with step 2 for the model and
step 3 for the board lock and inference account; those requirements still apply.

The installed layout is:

```text
rhinoforge/
  activate.sh, run-example.sh, bundle-manifest.json
  env/                         # Python, dependencies and installed RhinoForge
  source/                      # matching source, examples and documentation
  runtime/launch/              # sanitized Release Launch
  runtime/operators/           # main/expansion assets and sidecars
  runtime/sdk/                 # SDK user-space libraries, when supplied
  runtime/notices/              # accompanying runtime distribution terms
```

Install directly into its final absolute path. The installer runs `conda-unpack`
to repair environment prefixes once; moving the installed directory afterwards
is unsupported. To change its location, install the original archive into another
new directory. Follow the manifest's OS/glibc, architecture and SDK requirements;
an aarch64 bundle is not portable to x86_64 or every Linux/SDK combination. See
[conda-pack's relocation constraints](https://conda.github.io/conda-pack/).

### Build a bundle from a validated environment

Use a regular, non-editable RhinoForge installation from step 1 or a wheel from
step 5. For `--profile full`, install the audio/video libraries and all three
Python extras in the same environment. Install the Conda packages before the
Python extras so their dependency resolution finishes first. Keep the same
packaging/setuptools ranges from step 1; do not let pip replace Conda's copies:

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

The `conda-forge` channel has no `sox` package for `linux-aarch64`. Build the
[official SoX 14.4.2 source](https://sourceforge.net/projects/sox/files/sox/14.4.2/)
into the **same** `RHINOFORGE_ENV` instead of relying on a pre-existing system
executable. This build embeds static `libsox` in the CLI, provides built-in
PCM/WAV and effects/resampling, and disables external codecs, plugins and audio
devices. It does not add MP3 or other disabled codecs to SoX. Use the separately
packaged FFmpeg/libsndfile for the formats supported by those libraries; their
presence does not enable codecs in this SoX build.

The subshell below keeps Conda's active environment unchanged while preventing
SoX's build from picking up unrelated libraries. Retain the source archive and
license files inside the environment so they travel with the bundle.

```bash
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

Use `--profile text` for a base-only environment; the Qwen3 text example does
not require these extras or audio/video tools. Check dependencies and run the
intended model before packing. Keep `CONDA_PKGS_DIRS` and its downloaded package
cache until packing finishes: `conda-pack` may need those records to repair
prefixes. Do not run `conda clean` or remove that cache before packing.
`--runtime-notices` must contain the provider's original `RELEASE.txt` and
`DISTRIBUTION_TERMS.txt`. Provide the exact runtime paths used for validation:

```bash
python -m pip install "conda-pack==0.8.1"
python -m pip check
python scripts/pack_conda_bundle.py \
  --env "$RHINOFORGE_ENV" --source "$RHINOFORGE_SOURCE" \
  --launch "$RHINO_LAUNCH_ROOT" \
  --main-ref "$RPU_KERNEL_LIB_PATH" \
  --expansion-ref "$RPU_KERNEL_LIB_PATH_EXPANSION" \
  --sdk-version "REPLACE_WITH_VALIDATED_SDK_VERSION" \
  --runtime-notices /absolute/path/to/provider-runtime-notices \
  --profile full \
  --output /absolute/path/to/new/rhinoforge-conda.tar.gz
```

Use `--sdk-libs-dir /absolute/path/to/redistributable-sdk-libraries` only for the
provider-approved SDK user-space libraries; otherwise the target must supply
them. Keep model weights and private caches outside the environment. Retain the
bundle, checksum and manifest together. Validate installation at a **different
absolute prefix** and run Qwen3-0.6B FP16 with that installed environment before
delivery. A successful build alone does not demonstrate relocation or board
compatibility.
