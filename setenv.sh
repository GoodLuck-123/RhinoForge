#!/usr/bin/env bash
export RHINOFORGE_SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RHINOFORGE_ENV="$RHINOFORGE_SOURCE/.venv"
export RHINOFORGE_RUNTIME="/mnt/user_data/hao.wu/RhinoForge-runtime-v1.2.0"
export RHINO_LAUNCH_ROOT="$RHINOFORGE_RUNTIME/rhino-launch-kernel-v1.2.0-linux-aarch64"
export RPU_KERNEL_LIB_PATH="$RHINOFORGE_RUNTIME/rhinoOpLib_current.ref"
export RPU_KERNEL_LIB_PATH_EXPANSION="$RHINOFORGE_RUNTIME/rhinoExpansionOpLib_current.ref"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PATH="${HOME}/.local/bin:${PATH}"
export PIP_INDEX_URL="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
export UV_INDEX_URL="${UV_INDEX_URL:-$PIP_INDEX_URL}"
case ":${CMAKE_PREFIX_PATH:-}:" in
  *":$RHINO_LAUNCH_ROOT:"*) ;;
  *) export CMAKE_PREFIX_PATH="$RHINO_LAUNCH_ROOT${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}" ;;
esac
case ":${LD_LIBRARY_PATH:-}:" in
  *":$RHINO_LAUNCH_ROOT/lib:"*) ;;
  *) export LD_LIBRARY_PATH="$RHINO_LAUNCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
esac
source "$RHINOFORGE_ENV/bin/activate" || return 1
cd "$RHINOFORGE_SOURCE" || return 1
