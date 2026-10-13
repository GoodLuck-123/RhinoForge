#!/usr/bin/env python3
"""scripts/legacy_converter_inventory.py — LEG-01 / Phase 5 D-E1..E6 audit."""
from __future__ import annotations
import argparse
import ast
import pathlib
import sys
from typing import Iterable, Optional

import markdown_it


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "python" / "rpu_backend"
DOCS_ROOT = REPO_ROOT / "docs"


REMOVED_MODULES = frozenset({
    # v4.0 Phase 5 (LEG-01): 3 named legacy converters physically deleted.
    "rpu_backend.pi05_converter",
    "rpu_backend.qwenpi05_converter",
    "rpu_backend.qwenpi05_fused_converter",
    # v4.1 Phase 12 D-K1: 4 sibling converter shims tombstone-trajectory.
    # Phase 12 keeps them importable as thin re-exports; Phase 13 SIB-01 deletes them.
    # The script's --include-self toggle (default OFF) skips owner-file self-references via SELF_FILES below.
    "rpu_backend.model_converter",
    "rpu_backend.gemma_converter",
    "rpu_backend.adarms_converter",
    "rpu_backend.siglip_converter",
})

# Short module names (last dotted segment) for `from rpu_backend import <name>`
# and `getattr(rpu_backend, "<name>")` style detection.
REMOVED_MODULE_SHORT = frozenset({
    # v4.0 Phase 5 (LEG-01):
    "pi05_converter",
    "qwenpi05_converter",
    "qwenpi05_fused_converter",
    # v4.1 Phase 12 D-K1:
    "model_converter",
    "gemma_converter",
    "adarms_converter",
    "siglip_converter",
})

REMOVED_TOP_NAMES = frozenset({
    # Symbol-level matches surfaced by Plan 05-01 grep audit + the executor's
    # `grep -rn` of legacy callsites in tests/. Hardcoded per D-E1; NOT a
    # manifest file. Update only if a new external callsite surfaces.
    #
    # NB (Plan 05-01 MED-5 close-out / `decisions.helper_naming_propagation`):
    # the LEADING-UNDERSCORE versions below are the OLD names from
    # `pi05_converter.py:991/1060`. They MUST stay in REMOVED_TOP_NAMES so
    # the inventory flags any forgotten `_patch_*` callsite. The RENAMED
    # symbols `patch_siglip_embeddings_for_rpu` /
    # `patch_siglip_vision_model_for_rpu_fused` (no leading underscore) live
    # in the SURVIVING `_internal/patches/siglip_pi05.py` and are NOT in this
    # set (otherwise the inventory would falsely flag the surviving home).
    "_patch_siglip_embeddings_for_rpu",
    "_patch_siglip_vision_model_for_rpu_fused",
    "_patch_denoise_step",
    "_patch_pi_gemma_rmsnorm",
    "_patch_gemma_model_for_rpu_fused",
    "load_pi05_model",
    "convert_pi05_to_rpu_fused",
    "_remap_and_save",
    "_RPURotaryEmbWrapper",
    "run_qwenpi05_action_on_rpu",
    "convert_qwenpi05_action_model_for_rpu",
    # NOTE: `_convert_siglip_weights_for_rpu` is NOT in this set — the symbol
    # exists in BOTH the surviving siglip_converter.py AND the deleted
    # pi05_converter.py; we match against the *module path* via ImportFrom.module
    # so the script does not false-flag the surviving siblings shim.
})

SELF_FILES = frozenset({
    # Files the AST walk EXCLUDES when --include-self is OFF (D-E3).
    # v4.0 Phase 5 (LEG-01):
    "python/rpu_backend/pi05_converter.py",
    "python/rpu_backend/qwenpi05_converter.py",
    "python/rpu_backend/qwenpi05_fused_converter.py",
    # v4.1 Phase 12 D-K1: through Phase 12 these 4 shims still exist on disk
    # and self-reference (gemma_converter.py re-exports patch_gemma_*,
    # siglip_converter.py re-exports _convert_siglip_weights_for_rpu, etc.).
    # --include-self OFF (default) skips them so the script reports clean.
    # Phase 13 SIB-01 deletes them; SELF_FILES entries become harmless dead
    # branches but stay for forward-compat with the v4.0 entries above.
    "python/rpu_backend/model_converter.py",
    "python/rpu_backend/gemma_converter.py",
    "python/rpu_backend/adarms_converter.py",
    "python/rpu_backend/siglip_converter.py",
})

SKIP_DIRS = frozenset({
    ".planning", "rpu_deploy", "notebooks", "__pycache__", ".git",
})

# Substring patterns used for prose-level WARN scans + non-literal dynamic
# imports. Matches the short module names (more specific than `rpu_backend`).
# v4.1 Phase 12 D-K1: extended with 4 sibling shim short names for prose-level WARN scans.
# NOTE: PROSE_NEEDLES is a TUPLE (not a frozenset); the script's prose-scan iterates
# this collection, and tuple ordering is irrelevant for substring-membership tests.
PROSE_NEEDLES = (
    "pi05_converter", "qwenpi05_converter", "qwenpi05_fused_converter",
    "model_converter", "gemma_converter", "adarms_converter", "siglip_converter",
)


# ----------------------------------------------------------------------
# Per-file AST visitor + name-binding pre-pass.
# ----------------------------------------------------------------------


class _ImportAliases:
    """Module-scope name bindings discovered in the file's import block.

    rpu_aliases: names that resolve to ``rpu_backend`` (e.g., {"rpu_backend",
        "rb"} after ``import rpu_backend as rb``).
    import_module_aliases: names that resolve to ``importlib.import_module``
        (e.g., {"import_module", "im"} after ``from importlib import
        import_module as im``).
    """

    def __init__(self) -> None:
        self.rpu_aliases: set[str] = set()
        self.import_module_aliases: set[str] = set()

    def collect(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "rpu_backend":
                        self.rpu_aliases.add(alias.asname or "rpu_backend")
            elif isinstance(node, ast.ImportFrom):
                if node.module == "importlib":
                    for alias in node.names:
                        if alias.name == "import_module":
                            self.import_module_aliases.add(
                                alias.asname or "import_module"
                            )


class _AuditVisitor(ast.NodeVisitor):
    """Flags any reference to a removed module / symbol per D-E1 + LOW-8.

    Each finding is appended to ``self.findings`` as a tuple
    ``(severity, lineno, message)`` where severity is ``"FAIL"`` or
    ``"WARN"``.
    """

    def __init__(self, aliases: _ImportAliases) -> None:
        self.aliases = aliases
        self.findings: list[tuple[str, int, str]] = []

    # --- Pattern (1) + (2a) + (2b) + (2c) -----------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name in REMOVED_MODULES:
                self._fail(
                    node.lineno,
                    f"`import {alias.name}` references removed module",
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # (2a) `from rpu_backend.pi05_converter import …` (any names).
        if node.module in REMOVED_MODULES:
            self._fail(
                node.lineno,
                f"`from {node.module} import …` references removed module",
            )
        elif node.module == "rpu_backend":
            for alias in node.names:
                # (2b) `from rpu_backend import pi05_converter`.
                if alias.name in REMOVED_MODULE_SHORT:
                    self._fail(
                        node.lineno,
                        f"`from rpu_backend import {alias.name}` references "
                        f"removed module",
                    )
                # (2c) `from rpu_backend import _patch_denoise_step`.
                elif alias.name in REMOVED_TOP_NAMES:
                    self._fail(
                        node.lineno,
                        f"`from rpu_backend import {alias.name}` references "
                        f"removed top-level symbol",
                    )
        self.generic_visit(node)

    # --- Pattern (3), (4), (5), (6) -----------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        self._check_importlib_call(node)
        self._check_dunder_import_call(node)
        self._check_getattr_call(node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self._check_dotted_alias_attr(node)
        self.generic_visit(node)

    # --- Helpers ------------------------------------------------------

    def _check_importlib_call(self, node: ast.Call) -> None:
        # importlib.import_module("rpu_backend.pi05_converter")
        # OR (after `from importlib import import_module`) bare import_module(...)
        is_import_module = False
        if isinstance(node.func, ast.Attribute):
            if (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "importlib"
                and node.func.attr == "import_module"
            ):
                is_import_module = True
        elif isinstance(node.func, ast.Name):
            if node.func.id in self.aliases.import_module_aliases:
                is_import_module = True
        if not is_import_module:
            return
        if not node.args:
            return
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            if first.value in REMOVED_MODULES:
                self._fail(
                    node.lineno,
                    f"`import_module({first.value!r})` references removed module",
                )
        else:
            # Non-literal arg → WARN if it textually mentions one of the modules.
            text = ast.unparse(first) if hasattr(ast, "unparse") else ""
            if any(needle in text for needle in PROSE_NEEDLES):
                self._warn(
                    node.lineno,
                    f"`import_module(<non-literal>)` mentions a removed "
                    f"converter in its expression: {text!r}",
                )

    def _check_dunder_import_call(self, node: ast.Call) -> None:
        if not (isinstance(node.func, ast.Name) and node.func.id == "__import__"):
            return
        if not node.args:
            return
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            if first.value in REMOVED_MODULES:
                self._fail(
                    node.lineno,
                    f"`__import__({first.value!r})` references removed module",
                )

    def _check_getattr_call(self, node: ast.Call) -> None:
        if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
            return
        if len(node.args) < 2:
            return
        base, attr = node.args[0], node.args[1]
        if not isinstance(base, ast.Name):
            return
        if base.id not in self.aliases.rpu_aliases:
            return
        if isinstance(attr, ast.Constant) and isinstance(attr.value, str):
            attrname = attr.value
            if attrname in REMOVED_MODULE_SHORT:
                self._fail(
                    node.lineno,
                    f"`getattr({base.id}, {attrname!r})` references removed "
                    f"submodule",
                )
            elif attrname in REMOVED_TOP_NAMES:
                self._fail(
                    node.lineno,
                    f"`getattr({base.id}, {attrname!r})` references removed "
                    f"top-level symbol",
                )
        else:
            # Non-literal getattr → WARN per D-E4 §12 accepted debt.
            self._warn(
                node.lineno,
                f"`getattr({base.id}, <non-literal>)` against rpu_backend — "
                f"manual review recommended",
            )

    def _check_dotted_alias_attr(self, node: ast.Attribute) -> None:
        # rb.pi05_converter (whether in a Call or as an Attribute access).
        if not isinstance(node.value, ast.Name):
            return
        if node.value.id not in self.aliases.rpu_aliases:
            return
        if node.attr in REMOVED_MODULE_SHORT:
            self._fail(
                node.lineno,
                f"`{node.value.id}.{node.attr}` references removed submodule",
            )
        elif node.attr in REMOVED_TOP_NAMES:
            self._fail(
                node.lineno,
                f"`{node.value.id}.{node.attr}` references removed top-level "
                f"symbol",
            )

    def _fail(self, lineno: int, message: str) -> None:
        self.findings.append(("FAIL", lineno, message))

    def _warn(self, lineno: int, message: str) -> None:
        self.findings.append(("WARN", lineno, message))


# ----------------------------------------------------------------------
# File walking + dispatch.
# ----------------------------------------------------------------------


def _is_skipped(path: pathlib.Path) -> bool:
    """Return True if any path component is in SKIP_DIRS."""
    return any(part in SKIP_DIRS for part in path.parts)


def _audit_python_source(
    source: str,
    relpath: str,
    line_offset: int = 0,
) -> list[tuple[str, int, str]]:
    """Parse + audit a single Python source string.

    line_offset shifts every reported line by N (used for fenced markdown
    blocks where the AST inside the fence is line-1-based but the markdown
    fence starts at a non-zero line in the .md file).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [(
            "WARN",
            (exc.lineno or 1) + line_offset,
            f"SyntaxError in {relpath}: {exc.msg}; skipping AST audit",
        )]
    aliases = _ImportAliases()
    aliases.collect(tree)
    visitor = _AuditVisitor(aliases)
    visitor.visit(tree)
    if line_offset:
        return [
            (sev, lineno + line_offset, msg)
            for sev, lineno, msg in visitor.findings
        ]
    return list(visitor.findings)


def _walk_python_files(
    root: pathlib.Path,
    include_self: bool,
) -> Iterable[pathlib.Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(REPO_ROOT)
        if _is_skipped(rel):
            continue
        if not include_self and rel.as_posix() in SELF_FILES:
            continue
        yield path


def _walk_markdown_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(REPO_ROOT)
        if _is_skipped(rel):
            continue
        yield path


def _audit_markdown(path: pathlib.Path) -> list[tuple[str, int, str]]:
    """Walk fenced-code-blocks of a markdown file and audit Python ones.

    Per D-E4: typed `python` fences AND untyped fences (no info string) are
    parsed as Python; fence info strings starting with anything else are
    skipped. Prose mention of a removed module's name → WARN.
    """
    findings: list[tuple[str, int, str]] = []
    relpath = path.relative_to(REPO_ROOT).as_posix()
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        findings.append((
            "WARN", 1, f"could not read {relpath}: {exc}; skipping",
        ))
        return findings

    md = markdown_it.MarkdownIt()
    tokens = md.parse(text)
    for token in tokens:
        if token.type != "fence":
            continue
        info = (token.info or "").strip().lower()
        is_python_fence = info.startswith(("python", "py", "python3"))
        is_untyped_fence = info == ""
        if not (is_python_fence or is_untyped_fence):
            continue
        # token.map is [start_line, end_line) (0-indexed). The opening ```
        # is on start_line; code starts on start_line + 1.
        if token.map is None:
            line_offset = 0
        else:
            line_offset = token.map[0] + 1  # 1-based + skip the ``` line
        try:
            ast.parse(token.content)
        except SyntaxError:
            if is_untyped_fence:
                # Untyped fence that isn't valid Python — informational WARN.
                findings.append((
                    "WARN",
                    line_offset,
                    f"untyped fence in {relpath} is not parseable as Python; "
                    f"skipping",
                ))
                continue
            # Typed `python` fence with a parse error — emit WARN with the
            # parse error so the doc author can fix it.
            findings.append((
                "WARN",
                line_offset,
                f"typed `python` fence in {relpath} did not parse; skipping",
            ))
            continue
        block_findings = _audit_python_source(
            token.content, relpath, line_offset=line_offset
        )
        findings.extend(block_findings)

    # Prose-level WARN: any literal mention of a removed-converter short
    # name OUTSIDE a fenced code block. We approximate by scanning the
    # full text and emitting one WARN per matched line.
    seen_lines: set[int] = set()
    for ln, line in enumerate(text.splitlines(), start=1):
        if any(needle in line for needle in PROSE_NEEDLES):
            # Skip lines that already triggered FAIL inside a fence (the
            # block_findings path covers code; this just covers prose).
            if ln in seen_lines:
                continue
            seen_lines.add(ln)
            # Only emit WARN if the line isn't strictly inside a fence body
            # we already audited above (the AST path emits FAIL there).
            findings.append((
                "WARN",
                ln,
                f"prose mention of removed converter in {relpath}",
            ))
    return findings


# ----------------------------------------------------------------------
# CLI.
# ----------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "AST + markdown-it-py audit for legacy converter references "
            "(LEG-01 / Phase 5 D-E1..E6). Exit 0 = zero FAILs."
        ),
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help=(
            "Include the to-be-deleted converter files in the AST walk. "
            "Default OFF (Plan 05-03 gate); ON for Plan 05-02 dry-run."
        ),
    )
    args = parser.parse_args(argv)

    fail_count = 0
    warn_count = 0

    # Python files under python/rpu_backend/.
    for path in _walk_python_files(SRC_ROOT, include_self=args.include_self):
        relpath = path.relative_to(REPO_ROOT).as_posix()
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"WARN: {relpath}:1: could not read file: {exc}; skipping")
            warn_count += 1
            continue
        findings = _audit_python_source(source, relpath)
        for sev, lineno, msg in findings:
            print(f"{sev}: {relpath}:{lineno}: {msg}")
            if sev == "FAIL":
                fail_count += 1
            else:
                warn_count += 1

    # Markdown files under docs/.
    for path in _walk_markdown_files(DOCS_ROOT):
        relpath = path.relative_to(REPO_ROOT).as_posix()
        findings = _audit_markdown(path)
        for sev, lineno, msg in findings:
            print(f"{sev}: {relpath}:{lineno}: {msg}")
            if sev == "FAIL":
                fail_count += 1
            else:
                warn_count += 1

    if fail_count == 0:
        print(f"OK: 0 FAIL, {warn_count} WARN")
        return 0
    print(f"FAIL: {fail_count} FAIL, {warn_count} WARN")
    return 1


if __name__ == "__main__":
    sys.exit(main())
