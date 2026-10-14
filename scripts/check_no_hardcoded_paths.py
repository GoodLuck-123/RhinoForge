#!/usr/bin/env python3
"""Check CI tests for machine-specific paths."""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Files or Python-only directories CI executes. HARD-FAIL scope. Keep in sync
# with the jobs' `script:` blocks in .gitlab-ci.yml.
CI_EXECUTED: tuple[str, ...] = (
    "tests", "python/rpu_backend/tests/acceptance",
    "python/rpu_backend/tests/core", "python/rpu_backend/tests/graph",
    "python/rpu_backend/tests/library",
    "python/rpu_backend/tests/model", "python/rpu_backend/tests/quant",
)



def iter_ci_executed_files():
    """Yield every Python file covered by a CI path entry."""
    for rel in CI_EXECUTED:
        path = REPO_ROOT / rel
        if path.is_dir():
            for child in sorted(path.rglob("*.py")):
                yield child.relative_to(REPO_ROOT).as_posix(), child
        else:
            yield rel, path

# Roots scanned for the WARN pass (legacy / dev-only; not gated).
WARN_ROOTS: tuple[str, ...] = (
    "python/rpu_backend/tests",
    "scripts",
)

# (pattern, human reason). Narrow the interpreter rule to env/dev interpreters
# (/mnt, /home, /opt/.../bin/python) so a portable `/usr/bin/env python` or a
# bare `python` is never flagged.
FORBIDDEN: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"/mnt/nvme(?![A-Za-z0-9_])"), "hardcoded /mnt/nvme dev root"),
    (re.compile(r"/mnt/miniconda"), "hardcoded /mnt/miniconda root"),
    (re.compile(r"/(?:mnt|home|opt)/[^\s\"'`]*/bin/python[0-9.]*"),
     "hardcoded python interpreter path (use sys.executable)"),
]

# Per-line opt-out for the rare legitimate mention (kept tight).
EXEMPT_LINE = re.compile(r"#\s*lint-exempt:\s*hardcoded-path")


def _scan(path: pathlib.Path) -> list[tuple[int, str, str]]:
    """Return [(lineno, reason, snippet)] for forbidden matches. Skips pure
    comment lines (a path in prose doesn't break CI) and opted-out lines."""
    hits: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return hits
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("#") or EXEMPT_LINE.search(line):
            continue
        for pat, why in FORBIDDEN:
            if pat.search(line):
                hits.append((i, why, line.strip()[:100]))
                break
    return hits


def _lint() -> int:
    ci_resolved: set[pathlib.Path] = set()
    hard_fail = 0

    # HARD-FAIL pass over the CI-executed files.
    ci_files = tuple(iter_ci_executed_files())
    for rel, p in ci_files:
        ci_resolved.add(p.resolve())
        if not p.exists():
            print(f"FAIL {rel}: CI_EXECUTED file missing (renamed/removed? "
                  f"update CI_EXECUTED)", file=sys.stderr)
            hard_fail += 1
            continue
        for ln, why, snippet in _scan(p):
            print(f"FAIL {rel}:{ln}: {why}\n    {snippet}", file=sys.stderr)
            hard_fail += 1

    # WARN pass over everything else (one warn per file is enough).
    self_path = pathlib.Path(__file__).resolve()  # this checker holds the patterns
    warn_files = 0
    for root in WARN_ROOTS:
        for p in sorted((REPO_ROOT / root).rglob("*.py")):
            if p.resolve() in ci_resolved or p.resolve() == self_path:
                continue
            first = _scan(p)
            if first:
                ln, why, _ = first[0]
                print(f"warn {p.relative_to(REPO_ROOT)}:{ln}: {why} "
                      f"(+{len(first) - 1} more)" if len(first) > 1
                      else f"warn {p.relative_to(REPO_ROOT)}:{ln}: {why}",
                      file=sys.stderr)
                warn_files += 1

    print(f"[check_no_hardcoded_paths] hard-fail={hard_fail} "
          f"(CI-executed, {len(ci_files)} files) | warn={warn_files} "
          f"(legacy, not gated)", file=sys.stderr)
    if hard_fail:
        print("[check_no_hardcoded_paths] FAIL — fix with model_registry."
              "model_path() / sys.executable / env override, or add "
              "'# lint-exempt: hardcoded-path' if genuinely required.",
              file=sys.stderr)
        return 1
    print("[check_no_hardcoded_paths] PASS (CI-executed files portable)",
          file=sys.stderr)
    return 0


def _self_test() -> int:
    """Assert the patterns fire / don't fire on representative lines."""
    should_hit = [
        'model_dir = "/mnt/nvme/test-user/model"',
        "PY = '/mnt/miniconda3/envs/torch_3.12/bin/python'",
        'subprocess.run(["/mnt/nvme/miniconda3/envs/torch_ci/bin/python", ...])',
        'p = "/home/test-user/venv/bin/python3.12"',
    ]
    should_miss = [
        'model_path("qwen3-0.6b")',
        'sys.executable',
        '#!/usr/bin/env python3',
        'root = "/var/cache/models"',
        'PYBIN = "/usr/bin/python3"',
        'model_dir = "/mnt/nvme/test-user/x"  # lint-exempt: hardcoded-path',
    ]
    bad = 0
    for s in should_hit:
        if s.lstrip().startswith("#") or EXEMPT_LINE.search(s):
            print(f"self-test BAD (exempt should not apply): {s}", file=sys.stderr)
            bad += 1
            continue
        if not any(pat.search(s) for pat, _ in FORBIDDEN):
            print(f"self-test BAD (expected HIT): {s}", file=sys.stderr)
            bad += 1
    for s in should_miss:
        exempt = s.lstrip().startswith("#") or EXEMPT_LINE.search(s)
        if not exempt and any(pat.search(s) for pat, _ in FORBIDDEN):
            print(f"self-test BAD (expected MISS): {s}", file=sys.stderr)
            bad += 1
    if bad:
        print(f"[check_no_hardcoded_paths --self-test] {bad} FAIL", file=sys.stderr)
        return 1
    print("[check_no_hardcoded_paths --self-test] PASS", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="check the forbidden/allowed patterns, then exit")
    args = ap.parse_args()
    return _self_test() if args.self_test else _lint()


if __name__ == "__main__":
    sys.exit(main())
