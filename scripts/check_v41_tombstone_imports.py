#!/usr/bin/env python3
"""scripts/check_v41_tombstone_imports.py — codex Round-2 B-08 + Round-1 (post-execute) B-01 6-form coverage"""
from __future__ import annotations
import ast
import sys
from pathlib import Path
from typing import Optional


# 7 v4.1 tombstoned top-level names (post-Phase-13 SIB-04 D-A1; matches
# `[r.name for r in _DEPRECATED_ROWS if r.removed_in == "4.1"]`).
_V41_TOMBSTONED_NAMES = frozenset({
    "patch_qwen3_model_for_rpu",
    "patch_qwen3_model_for_rpu_all_layers_once",
    "patch_gemma_model_for_rpu_all_layers_once",
    "patch_adarms_model_for_rpu_all_layers_once",
    "patch_siglip_model_for_rpu_all_layers_once",
    "set_chunk_size",
    "set_spm_mode",
})


# Files that REFERENCE v4.1 names by design — exclude from scan.
# Mirrors the SELF_FILES idiom in scripts/check_import_graph.py +
# legacy_converter_inventory.py. These are forbidden-list-host files +
# the registry source-of-truth + the tombstone scaffold + tests that
# validate the tombstone behavior.
_EXEMPT_PATHS = frozenset({
    Path("python/rpu_backend/_internal/_deprecations.py"),
    Path("python/rpu_backend/_legacy_tombstones.py"),
    Path("python/rpu_backend/tests/acceptance/test_tombstones.py"),
    Path("python/rpu_backend/tests/acceptance/test_deprecation_shims.py"),
    Path("python/rpu_backend/tests/acceptance/test_v4_public_api.py"),
})


class _ImportAliases:
    """Module-scope name bindings discovered in the file's import block.

    Mirrors scripts/legacy_converter_inventory.py:142 _ImportAliases.

    rpu_aliases: names that resolve to ``rpu_backend`` (e.g., {"rpu_backend",
        "rb"} after ``import rpu_backend as rb``).
    getattr_aliases: names bound to the built-in ``getattr`` (e.g., {"getattr",
        "ga"} after ``from builtins import getattr as ga`` — rare but possible).
    """

    def __init__(self) -> None:
        # `rpu_backend` is always its own alias (Python name resolution).
        self.rpu_aliases: set[str] = {"rpu_backend"}
        # `getattr` is always callable as itself.
        self.getattr_aliases: set[str] = {"getattr"}

    def feed_import(self, node: ast.Import) -> None:
        """`import rpu_backend [as alias]`."""
        for alias in node.names:
            if alias.name == "rpu_backend":
                # `import rpu_backend` (alias.asname is None) → name is "rpu_backend"
                # `import rpu_backend as rb` → alias.asname is "rb"
                self.rpu_aliases.add(alias.asname or alias.name)

    def feed_import_from(self, node: ast.ImportFrom) -> None:
        """`from builtins import getattr as ga` rebinds the `getattr` name."""
        if node.module == "builtins":
            for alias in node.names:
                if alias.name == "getattr":
                    self.getattr_aliases.add(alias.asname or alias.name)

    def feed_assign(self, node: ast.AST) -> None:
        """Form F — assignment-style alias from `__import__("rpu_backend")`.

        Catches:
          rb = __import__("rpu_backend")           # ast.Assign
          rb: types.ModuleType = __import__("rpu_backend")  # ast.AnnAssign

        After this binding, `rb.<NAME>` is the same as `rpu_backend.<NAME>`
        and Form B/C dispatch picks it up via `rpu_aliases`.
        """
        # Extract RHS + targets generically across Assign / AnnAssign.
        if isinstance(node, ast.Assign):
            value = node.value
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value = node.value
            targets = [node.target]
        else:
            return
        # RHS must be `__import__("rpu_backend")` (or aliased `__import__`).
        if not (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "__import__"
            and len(value.args) >= 1
            and isinstance(value.args[0], ast.Constant)
            and value.args[0].value == "rpu_backend"
        ):
            return
        # LHS must be a single Name (skip tuple/list unpacking — out of scope).
        for target in targets:
            if isinstance(target, ast.Name):
                self.rpu_aliases.add(target.id)


class _V41HitVisitor(ast.NodeVisitor):
    """Walk an AST and accumulate hits across all 6 forms (A-F)."""

    def __init__(self, path: Path, aliases: _ImportAliases) -> None:
        self.path = path
        self.aliases = aliases
        self.hits: list[str] = []

    # Form A — `from rpu_backend import <NAME>` (handles multiline + aliased)
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module == "rpu_backend":
            for alias in node.names:
                if alias.name in _V41_TOMBSTONED_NAMES:
                    as_clause = f" as {alias.asname}" if alias.asname else ""
                    self.hits.append(
                        f"{self.path}:{node.lineno} [Form A] "
                        f"`from rpu_backend import {alias.name}{as_clause}`"
                    )
        self.generic_visit(node)

    # Form B/C — `<rpu_alias>.<NAME>` attribute access
    def visit_Attribute(self, node: ast.Attribute) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in self.aliases.rpu_aliases
            and node.attr in _V41_TOMBSTONED_NAMES
        ):
            form = "B" if node.value.id == "rpu_backend" else "C"
            self.hits.append(
                f"{self.path}:{node.lineno} [Form {form}] "
                f"`{node.value.id}.{node.attr}` (alias={node.value.id})"
            )
        self.generic_visit(node)

    # Form D/E — `<getattr_alias>(<rpu_alias>, "<NAME>")` string-literal call
    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in self.aliases.getattr_aliases
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in self.aliases.rpu_aliases
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
            and node.args[1].value in _V41_TOMBSTONED_NAMES
        ):
            base = node.args[0].id
            name = node.args[1].value
            form = "D" if base == "rpu_backend" else "E"
            self.hits.append(
                f"{self.path}:{node.lineno} [Form {form}] "
                f"`{node.func.id}({base}, \"{name}\")` (base_alias={base})"
            )
        self.generic_visit(node)


def _scan_file(path: Path) -> list[str]:
    """Return list of `path:lineno description` hit strings for `path`.

    Two-pass:
      1. Walk all top-level Import / ImportFrom nodes to populate aliases.
         (Function-scope re-imports are rare in this codebase; module-scope
         is the dominant pattern. Per-function alias rebinding is intentionally
         out of scope — false-negative risk acknowledged.)
      2. Walk the full AST with _V41HitVisitor using the resolved aliases.
    """
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"{path}:0 ReadError: {exc}"]
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno or 0} SyntaxError: {exc.msg}"]

    # Pass 1 — module-scope import alias collection (Forms A-E + F binding).
    aliases = _ImportAliases()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.feed_import(node)
        elif isinstance(node, ast.ImportFrom):
            aliases.feed_import_from(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # Form F — `rb = __import__("rpu_backend")` aliases (codex Round-2 B-03 fix)
            aliases.feed_assign(node)

    # Pass 2 — visit + collect hits.
    visitor = _V41HitVisitor(path, aliases)
    visitor.visit(tree)
    return visitor.hits


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    scan_root = repo_root / "python" / "rpu_backend"
    all_hits: list[str] = []
    for py_path in sorted(scan_root.rglob("*.py")):
        rel = py_path.relative_to(repo_root)
        if rel in _EXEMPT_PATHS:
            continue
        if "__pycache__" in py_path.parts:
            continue
        all_hits.extend(_scan_file(py_path))
    if all_hits:
        print(
            "FAIL (codex Round-2 B-08 + Round-1-post-execute B-01): "
            "v4.1-tombstoned name imports/access found:",
            file=sys.stderr,
        )
        for hit in all_hits:
            print(f"  {hit}", file=sys.stderr)
        return 1
    print(
        "OK: zero v4.1 tombstoned imports under python/rpu_backend/ "
        "(6-form coverage: A from-import, B rpu_backend.X, C aliased rb.X, "
        "D getattr literal, E aliased getattr, F implicit rb-binding via __import__)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
