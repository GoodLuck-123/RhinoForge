#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

FAIL=0

VIOLATIONS="$(
  python3 - <<'PY'
import re
from pathlib import Path

FORBIDDEN = re.compile(
    r'\b(llama_common|rhino_runtime/[^\s]*flash_attn[^\s]*)\b'
)

TRACKED_GLOBS = (
    'src/**/*.cpp', 'src/**/*.h', 'src/**/*.hpp', 'src/**/*.cc',
    'python/**/*.cpp', 'python/**/*.h', 'python/**/*.hpp', 'python/**/*.cc',
    'python/**/*.py',
)
files = set()
for glob in TRACKED_GLOBS:
    files.update(Path('.').glob(glob))
files = {p for p in files
         if '/tests/' not in p.as_posix() and '/debug/' not in p.as_posix()}


def strip_noncode(line: str, is_py: bool) -> str:
    """Remove string literals + comments so only real code survives.
    Conservative: false-positives here just mean the guard MIGHT flag
    a benign mention (fixable by rephrasing). False-negatives are
    dangerous; favor stripping less.

    Two exceptions to string stripping:
      * `#include "..."`  — the string IS the code (header path).
      * `#import` / `#pragma` (rare) — same.
    """
    line = re.sub(r'/\*.*?\*/', '', line)
    stripped_leading = line.lstrip()
    is_include_directive = stripped_leading.startswith((
        '#include', '#import',
    ))
    if not is_include_directive:
        line = re.sub(r'"(?:[^"\\]|\\.)*"', '', line)
    cc_idx = line.find('//')
    if cc_idx >= 0:
        line = line[:cc_idx]
    if is_py:
        hash_idx = line.find('#')
        if hash_idx >= 0:
            line = line[:hash_idx]
    return line


violations = []
for path in sorted(files):
    is_py = path.suffix == '.py'
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        continue
    in_block_comment = False
    for lineno, raw in enumerate(text.splitlines(), 1):
        line_for_scan = raw
        if in_block_comment:
            close = raw.find('*/')
            if close < 0:
                continue  # entire line is still inside the block comment
            line_for_scan = raw[close + 2:]
            in_block_comment = False
        stripped_for_state = re.sub(r'/\*.*?\*/', '', line_for_scan)
        open_idx = stripped_for_state.rfind('/*')
        close_idx = stripped_for_state.rfind('*/')
        if open_idx > close_idx:
            in_block_comment = True
            line_for_scan = stripped_for_state[:open_idx]

        if not FORBIDDEN.search(line_for_scan):
            continue
        code_only = strip_noncode(line_for_scan, is_py)
        if FORBIDDEN.search(code_only):
            violations.append(f"{path}:{lineno}:{raw.rstrip()}")

print('\n'.join(violations))
PY
)"

if [[ -n "$VIOLATIONS" ]]; then
  echo "R27 violation: runtime _dp reference in fused dispatch path:" >&2
  echo "$VIOLATIONS" >&2
  echo "" >&2
  echo "  The fused SDPA path must go through choose_tile_k" >&2
  echo "  (src/core/rpu_helpers.h). Do not import llama_common or" >&2
  echo "  rhino_runtime's flash_attn helpers — they use the" >&2
  echo "  legacy tile_k = tile_m heuristic that v5 replaces." >&2
  FAIL=1
fi

if ! grep -q 'kSdpaTilePolicyVersion' src/core/rpu_helpers.h; then
  echo "R27 violation: kSdpaTilePolicyVersion not defined in" >&2
  echo "  src/core/rpu_helpers.h. The compile-time tile-policy version" >&2
  echo "  stamp must exist so a policy/algorithm change is observable." >&2
  FAIL=1
fi

if ! grep -q 'kSdpaTilePolicyVersion' src/core/rpu_pybind.inc; then
  echo "R27 violation: kSdpaTilePolicyVersion not exposed to Python via" >&2
  echo "  src/core/rpu_pybind.inc. Post-refactor the C++ GraphCacheKey was" >&2
  echo "  retired and Python-side GraphSignature owns cache-key construction;" >&2
  echo "  the version must stay observable across the pybind boundary." >&2
  FAIL=1
fi

if [[ "$FAIL" -ne 0 ]]; then
  exit 1
fi

echo "R27 OK: no runtime _dp references; policy version wired through."
