# Build the current development source

This guide covers editable installs and wheel builds for the current source.
For a first model run, use [Getting started](getting_started.md). Build
requirements follow the current `pyproject.toml` and matching external runtime
assets. For existing installations, follow the [v1.2.0 upgrade guide](upgrading_1_2.md).

## Prepare Python and the compiler

Use Python **3.12**, its development headers, `venv` and a C++17 compiler.
Ubuntu 22.04's default Python is 3.10; installing `python3-dev` there does not
satisfy this requirement. On a distribution that supplies Python 3.12, install
its matching interpreter, development and venv packages. For example, on
Ubuntu 24.04:

```bash
sudo apt-get update
sudo apt-get install build-essential python3.12 python3.12-dev python3.12-venv
export RHINOFORGE_PYTHON=/usr/bin/python3.12
```

If the platform has no Python 3.12 package, install a verified Python 3.12
distribution with headers into a new user-owned prefix. One option is to build
the [official CPython 3.12.15 source](https://www.python.org/downloads/release/python-31215/).
The following Ubuntu/Debian example uses `altinstall` and leaves the system
Python in place; see [CPython's Unix build guide](https://docs.python.org/3.12/using/unix.html#building-python).
Installing OS build dependencies requires the platform administrator's policy:

```bash
sudo apt-get update
sudo apt-get install build-essential pkg-config curl ca-certificates xz-utils \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev
export RHINOFORGE_BOOTSTRAP="/absolute/path/to/new/python-bootstrap"
export RHINOFORGE_PYTHON_PREFIX="/absolute/path/to/new/python-3.12"
mkdir -p "$RHINOFORGE_BOOTSTRAP"
cd "$RHINOFORGE_BOOTSTRAP"
curl -fLO https://www.python.org/ftp/python/3.12.15/Python-3.12.15.tar.xz
printf '%s  %s\n' c2c4321961fab0fb999d66e0cecf521c2ab3994c7992873ea99e306c1094fd5a \
  Python-3.12.15.tar.xz | sha256sum -c -
tar -xf Python-3.12.15.tar.xz
cd Python-3.12.15
./configure --prefix="$RHINOFORGE_PYTHON_PREFIX" --with-ensurepip=install
make -j2
make altinstall
export RHINOFORGE_PYTHON="$RHINOFORGE_PYTHON_PREFIX/bin/python3.12"
```

For a standalone binary distribution, also check its architecture, checksum and
included headers. Its Python `sysconfig` may name a compiler absent on the board
(for example `clang++ -pthread`). Explicitly select installed C and C++ compilers
before configuring RhinoForge. This GCC example applies to either Python route:

```bash
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
"$CC" --version
"$CXX" --version
unset PYTHONPATH PYTHONHOME RPU_BACKEND_SO
export PYTHONNOUSERSITE=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
"$RHINOFORGE_PYTHON" - <<'PY'
import pathlib, ssl, sys, sysconfig, venv
assert sys.version_info[:2] == (3, 12), sys.version
header = pathlib.Path(sysconfig.get_path("include")) / "Python.h"
assert header.is_file(), f"Missing Python development header: {header}"
print(sys.executable, sys.version, header, ssl.OPENSSL_VERSION, sep="\n")
PY
printf '#include <optional>\nint main() { std::optional<int> x{1}; return *x - 1; }\n' \
  | "$CXX" -std=c++17 -x c++ -fsyntax-only -
```

Use a new CMake build directory after changing compilers; an existing CMake cache
can retain the old selection. A venv alone does not clear `PYTHONPATH` or
`RPU_BACKEND_SO`, which can select code or a native extension from another checkout.

## Select matching external runtime assets

The board OS, kernel driver and SDK are separate platform prerequisites; a new
Python environment does not install them. Follow the provider's platform setup
and the [library and device preflight](runtime_assets.md#board-sdk-and-device-access).
Obtain
the following matching assets from the runtime provider before building:

- A sanitized **Release** Rhino Launch installation with its headers, libraries
  and `lib/cmake/rhino_launch` package. Current CMake requires
  `graph-arena-pool-v2`, `register-state-token`, sanitized Chrome hwperf support and no development Program
  API. The package version or library filename alone does not prove compatibility.
- The main and expansion opaque operator assets selected for this source revision, with its
  adjacent `.kernels` sidecar and provider-supplied integrity values. The current
  backend resolves each operator against its designated main or expansion asset.
- The matching board SDK/runtime and its required shared libraries. Keep the
  build-time and run-time Launch installation consistent.

Launch and operator assets remain external to this repository and wheel. Do not
substitute an older incompatible runtime, copy its payload into the package,
merge operator payloads locally, or generate a replacement sidecar.
Use the [runtime asset request entry](runtime_assets.md#request-runtime-assets)
if you need the matching package.

Set paths to the selected source, a new environment and a dedicated build
directory. Keep this environment separate from any installed `rpu_backend` or
other RhinoForge checkout; do not use `--system-site-packages`.
First set the Launch and operator paths using the complete
[download, verification and nested extraction steps](runtime_assets.md#configure-paths).

```bash
export RHINOFORGE_SOURCE="/absolute/path/to/current/RhinoForge"
export RHINOFORGE_DEV_ENV="/absolute/path/to/new/rhinoforge-dev-env"
export RHINOFORGE_DEV_BUILD="/absolute/path/to/new/rhinoforge-dev-build"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

test -r "$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch/rhino_launchConfig.cmake"
test -r "$RPU_KERNEL_LIB_PATH"
test -r "$RPU_KERNEL_LIB_PATH.kernels"
test -r "$RPU_KERNEL_LIB_PATH_EXPANSION"
test -r "$RPU_KERNEL_LIB_PATH_EXPANSION.kernels"
export LD_LIBRARY_PATH="$RHINO_LAUNCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Verify the supplied asset digests before use. The existence checks above only
check paths. Remove conflicting Launch directories from an inherited library
search path; retain the platform SDK paths required by the selected runtime.

## Install from source

The current [pyproject.toml](../pyproject.toml) is the dependency authority:
Python 3.12, PyTorch 2.10.0 and Transformers **5.3.0**. Its optional `vision` extra
adds TorchVision 0.25.0; the `vla` extra also selects LeRobot 0.5.1 and Diffusers
0.33.1. Speech uses the `audio` extra; see the [Qwen3 ASR/TTS guide](qwen3_audio.md).
Install the extra needed by the intended entry point.

```bash
"$RHINOFORGE_PYTHON" -m venv "$RHINOFORGE_DEV_ENV"
. "$RHINOFORGE_DEV_ENV/bin/activate"
cd "$RHINOFORGE_SOURCE"

python -m pip install -c constraints/security.txt "torch==2.10.0" "scikit-build-core>=0.12,<0.13" "cmake==4.1.3" "ninja==1.13.0"
python -m pip install -e . -c constraints/security.txt --no-build-isolation \
  -Cbuild-dir="$RHINOFORGE_DEV_BUILD" \
  -Ccmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  -Cbuild.tool-args=-j2
python -m pip check
python -c 'import sys; assert sys.prefix != sys.base_prefix; print(sys.executable)'
```

For vision, VLA or speech, replace `-e .` with `-e '.[vision]'`,
`-e '.[vla]'` or `-e '.[audio]'` in that install command. Keep dependency resolution enabled; the
initial source install must not use `--no-deps`. Two build workers bound compiler
memory use; adjust this limit to the host's available resources.

Inspect the dependency versions and selected Python source without loading the
native extension or issuing a device operation:

```bash
python - <<'PY'
from importlib.metadata import version
from importlib.util import find_spec

assert version("transformers") == "5.3.0"
assert version("torch").split("+")[0] == "2.10.0"
print("rhinoforge:", version("rhinoforge"))
print("Python source:", find_spec("rpu_backend").origin)
PY
```

An editable install follows Python edits in this checkout. Native changes still
require rebuilding and reinstalling with the same selected Launch installation.

## Build a reviewable wheel

After installing and checking the dependencies above, build into a new artifact
directory. This command reuses the dedicated build directory and does not replace
the installed backend:

```bash
python -m pip wheel . -c constraints/security.txt --no-build-isolation --no-deps \
  --wheel-dir /absolute/path/to/new/wheel-artifacts \
  -Cbuild-dir="$RHINOFORGE_DEV_BUILD" \
  -Ccmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  -Cbuild.tool-args=-j2
```

Record the source commit and local diff, resolved dependencies, wheel/native
digests, and selected Launch/operator/SDK identities with the build. Install the
wheel in another fresh environment and check package/native origins and
dependencies before device validation. Rebuild the wheel after Python changes
as well as native changes: an unchanged `rpu_backend.so` digest or `1.0.0` package
version does not establish that an older wheel contains the current Python code.
