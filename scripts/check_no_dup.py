#!/usr/bin/env python3
"""Enforce one production definition site for shared runtime primitives."""
from __future__ import annotations

import ast
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "python" / "rpu_backend"
CANONICAL = PACKAGE_ROOT / "runtime" / "weights.py"
GRAPH_CANONICAL = PACKAGE_ROOT / "graph" / "_runtime.py"
SHARED_NAMES = frozenset({
    "NUM_CORES",
    "LinearPartition",
    "assert_fp16",
    "convert_linear_weights_inplace",
    "get_linear_partition",
    "swizzle_linear",
    "swizzle_model_inplace",
    "tp_col_swizzle_mc_weight",
    "tp_row_swizzle_mc_weight",
    "transform_linear_weight",
})
EXCLUDED_PARTS = frozenset({"tests", "debug", "_template"})
GRAPH_NAMES = frozenset({
    "GraphCache",
    "GraphSignature",
    "get_default_graph_cache",
})


def _definition_sites(
    path: pathlib.Path, names: frozenset[str]
) -> dict[str, int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    sites: dict[str, int] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in names:
                sites[node.name] = node.lineno
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    sites[target.id] = node.lineno
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id in names:
                sites[node.target.id] = node.lineno
    return sites


def main() -> int:
    if not CANONICAL.is_file():
        print(f"FAIL: canonical weights module missing: {CANONICAL}", file=sys.stderr)
        return 1

    definitions: dict[str, list[tuple[pathlib.Path, int]]] = {
        name: [] for name in SHARED_NAMES
    }
    try:
        for path in PACKAGE_ROOT.rglob("*.py"):
            rel = path.relative_to(PACKAGE_ROOT)
            if EXCLUDED_PARTS & set(rel.parts):
                continue
            for name, lineno in _definition_sites(path, SHARED_NAMES).items():
                if name == "NUM_CORES" and path != CANONICAL:
                    # Architecture-local planners may pin the hardware core
                    # count; the shared conversion default remains canonical.
                    continue
                definitions[name].append((rel, lineno))
    except (OSError, SyntaxError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    failures = []
    canonical_rel = CANONICAL.relative_to(PACKAGE_ROOT)
    for name, sites in sorted(definitions.items()):
        if len(sites) != 1 or sites[0][0] != canonical_rel:
            rendered = [f"{path}:{line}" for path, line in sites]
            failures.append(f"{name}: expected only {canonical_rel}, got {rendered}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    graph_definitions: dict[str, list[tuple[pathlib.Path, int]]] = {
        name: [] for name in GRAPH_NAMES
    }
    try:
        for path in PACKAGE_ROOT.rglob("*.py"):
            rel = path.relative_to(PACKAGE_ROOT)
            if EXCLUDED_PARTS & set(rel.parts):
                continue
            for name, lineno in _definition_sites(path, GRAPH_NAMES).items():
                graph_definitions[name].append((rel, lineno))
    except (OSError, SyntaxError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    graph_canonical_rel = GRAPH_CANONICAL.relative_to(PACKAGE_ROOT)
    graph_failures = []
    for name, sites in sorted(graph_definitions.items()):
        if len(sites) != 1 or sites[0][0] != graph_canonical_rel:
            rendered = [f"{path}:{line}" for path, line in sites]
            graph_failures.append(
                f"{name}: expected only {graph_canonical_rel}, got {rendered}"
            )
    if graph_failures:
        for failure in graph_failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print(
        f"PASS: {len(SHARED_NAMES)} shared weight primitives have one definition "
        f"site in {canonical_rel}; {len(GRAPH_NAMES)} structured graph primitives "
        f"have one definition site in {graph_canonical_rel}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
