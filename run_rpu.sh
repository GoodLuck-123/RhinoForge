#!/usr/bin/env bash
# Run the repository Python with board permissions and exclusive device access.
set -e
source "$(dirname -- "${BASH_SOURCE[0]}")/setenv.sh"
if (( $# == 0 )); then
  echo "Usage: bash run_rpu.sh <python-script> [arguments...]" >&2
  exit 2
fi
caller_uid="${SUDO_UID:-$(id -u)}"
caller_gid="${SUDO_GID:-$(id -g)}"
exec sudo env \
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
  RPU_CACHING_SEGMENT_MB="${RPU_CACHING_SEGMENT_MB:-2}" \
  LD_PRELOAD="$(g++ -print-file-name=libstdc++.so.6)" \
  LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  RPU_KERNEL_LIB_PATH="$RPU_KERNEL_LIB_PATH" \
  RPU_KERNEL_LIB_PATH_EXPANSION="$RPU_KERNEL_LIB_PATH_EXPANSION" \
  RPU_MODEL_CACHE="${RPU_MODEL_CACHE:-/nfs3-sh/LLM}" \
  HF_HOME="${HF_HOME:-$RHINOFORGE_SOURCE/.cache/huggingface}" \
  flock -n /var/lock/rpu.device bash -c '
    owner_uid=$1; owner_gid=$2; results=$3; shift 3
    cleanup_outputs() {
      [[ -e "$results" ]] || return 0
      if [[ -L "$results" || ! -d "$results" ]]; then
        echo "Refusing to repair a symlink or non-directory: $results" >&2
        return 1
      fi
      # Do not follow symlinks or change files owned by another ordinary user.
      find -P "$results" -xdev -uid 0 \! -type l \
        -exec chmod u+rwX -- {} + \
        -exec chown -h -- "$owner_uid:$owner_gid" {} +
    }
    if fuser /dev/rpu /dev/mem >/dev/null 2>&1; then
      echo "RPU is in use; refusing to start another process." >&2
      exit 75
    fi
    on_exit() {
      rc=$?
      trap - EXIT
      cleanup_outputs || echo "Output ownership repair failed" >&2
      exit "$rc"
    }
    trap on_exit EXIT
    "$@"
  ' bash "$caller_uid" "$caller_gid" "$RHINOFORGE_SOURCE/perf_results" \
    "$RHINOFORGE_ENV/bin/python" "$@"
