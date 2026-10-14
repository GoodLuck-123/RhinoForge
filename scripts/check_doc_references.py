#!/usr/bin/env python3
"""Verify that intra-repo doc cross-references resolve to real files."""
from __future__ import annotations

import argparse
import pathlib
import re
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# Explicit scan set — the operating-manual surface, NOT the append-only history
# under docs/roadmap, docs/superpowers, docs/kernels (full of intentionally
# time-stamped/stale refs). Keep this list tight and load-bearing.
SCAN_GLOBS = (
    "README.md",              # customer-facing entry doc
    "README_EN.md",
    "AGENTS.md",
    "docs/*.md",
    "docs/guidelines/*.md",
    "examples/**/*.md",
    ".agents/skills/**/*.md",
    "knowledge/INDEX.md",     # KB catalog: its markdown infra links must resolve
                              # (the [[wiki]] page links are dangling-by-design
                              #  per knowledge/SCHEMA.md and are NOT matched here)
)

_MD_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
_AT_IMPORT = re.compile(r"^@(\S+\.md)\s*$")
_SKIP_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://|^(mailto|tel):")
# Template / glob / shell-var markers — a target carrying any of these is not a
# literal path.
_TEMPLATE_CHARS = set("<>*{}$|` ")


def _iter_files() -> list[pathlib.Path]:
    files: list[pathlib.Path] = []
    for glob in SCAN_GLOBS:
        files.extend(p for p in REPO_ROOT.glob(glob) if p.is_file())
    return sorted(set(files))


def _clean_target(raw: str) -> str | None:
    """Return a repo file path to verify, or None to skip this reference."""
    t = raw.strip()
    if not t or t.startswith("#"):            # empty or same-file anchor
        return None
    if t.startswith("//") or _SKIP_SCHEME.match(t):   # URL / protocol-relative
        return None
    t = t.split("#", 1)[0].split("?", 1)[0].strip()   # drop anchor / query
    if not t:
        return None
    if any(c in _TEMPLATE_CHARS for c in t):  # templated / glob / var
        return None
    # Vendored submodule (third_party/): checkout-state varies (populated in CI,
    # often not in a worktree) and its docs are the submodule's own concern, not
    # our doc surface — don't gate on it.
    if "third_party" in pathlib.PurePosixPath(t).parts:
        return None
    return t


def _broken_refs(path: pathlib.Path) -> list[tuple[int, str]]:
    """Return [(lineno, target)] for references in `path` that do not resolve."""
    broken: list[tuple[int, str]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    base = path.parent
    for lineno, line in enumerate(text.splitlines(), 1):
        targets = list(_MD_LINK.findall(line))
        m = _AT_IMPORT.match(line.strip())
        # @import resolves relative to the repo root, not the containing file.
        at_targets = [(REPO_ROOT, m.group(1))] if m else []
        for raw in targets:
            t = _clean_target(raw)
            if t is None:
                continue
            resolved = (base / t).resolve()
            if not resolved.is_relative_to(REPO_ROOT.resolve()):
                continue  # External workspace/board provenance, not an intra-repo link.
            if not resolved.exists():
                broken.append((lineno, raw.strip()))
        for root, raw in at_targets:
            resolved = (root / raw).resolve()
            if not resolved.exists():
                broken.append((lineno, "@" + raw))
    return broken


def _run(files: list[pathlib.Path]) -> int:
    total = 0
    for path in files:
        for lineno, target in _broken_refs(path):
            rel = path.relative_to(REPO_ROOT)
            print(f"FAIL: {rel}:{lineno} -> broken reference {target!r}",
                  file=sys.stderr)
            total += 1
    if total:
        print(f"FAIL: {total} broken doc reference(s)", file=sys.stderr)
        return 1
    print(f"PASS: all doc references resolve ({len(files)} files scanned)")
    return 0


def _self_test() -> int:
    import tempfile
    global REPO_ROOT
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "real.md").write_text("target\n")
    (d / "sub").mkdir()
    (d / "sub" / "child.md").write_text("hi\n")
    doc = d / "doc.md"
    doc.write_text(
        "[ok](real.md)\n"                       # resolves
        "[ok-anchor](real.md#sec)\n"            # resolves (anchor stripped)
        "[ok-rel](sub/child.md)\n"              # resolves
        "[url](https://example.com/x.md)\n"     # skipped (URL)
        "[tmpl](adapters/<model>/x.md)\n"       # skipped (template)
        "[glob](perf_results/*/r.md)\n"         # skipped (glob)
        "[submod](../third_party/foo/bar.md)\n" # skipped (vendored submodule)
        "[anchor](#section)\n"                   # skipped (same-file anchor)
        "[bad](does_not_exist.md)\n"            # BROKEN
        "[bad2](sub/missing.md)\n"              # BROKEN
        "[external](../external-workspace/missing.md)\n"  # outside repo scope
    )
    saved = REPO_ROOT
    REPO_ROOT = d
    try:
        broken = {t for _, t in _broken_refs(doc)}
        nested = d / "examples/configs/model/README.md"
        nested.parent.mkdir(parents=True)
        nested.write_text("[bad](missing.toml)\n")
        performance = d / "docs/example.md"
        performance.parent.mkdir(parents=True)
        performance.write_text("[bad](missing.json)\n")
        nested_ok = {nested, performance}.issubset(_iter_files())
        nested_ok = nested_ok and _broken_refs(nested) == [(1, "missing.toml")]
    finally:
        REPO_ROOT = saved
    expected = {"does_not_exist.md", "sub/missing.md"}
    ok = broken == expected
    print(("PASS" if ok else "FAIL")
          + f": self-test broken={sorted(broken)} expected={sorted(expected)}")
    # also exercise @import resolution relative to repo-root-like base
    imp = d / "imp.md"
    imp.write_text("@real.md\n@nope.md\n")
    # temporarily point resolution root at d by monkeypatching module global
    saved = REPO_ROOT
    REPO_ROOT = d
    try:
        imp_broken = {t for _, t in _broken_refs(imp)}
    finally:
        REPO_ROOT = saved
    ok2 = imp_broken == {"@nope.md"}
    print(("PASS" if ok2 else "FAIL")
          + f": self-test @import broken={sorted(imp_broken)} expected=['@nope.md']")
    print(("PASS" if nested_ok else "FAIL") + ": self-test nested example/document discovery")
    return 0 if (ok and ok2 and nested_ok) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    return _run(_iter_files())


if __name__ == "__main__":
    sys.exit(main())
