from __future__ import annotations
import ast
from pathlib import Path
_GRAPH_NAMES = {'Graph', 'GraphCache', 'GraphSignature'}

def _deprecated_graph_accesses(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    package_aliases = set()
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'rpu_backend':
                    package_aliases.add(alias.asname or 'rpu_backend')
        elif isinstance(node, ast.ImportFrom) and node.module == 'rpu_backend':
            for alias in node.names:
                if alias.name in _GRAPH_NAMES:
                    violations.append(f'{path}:{node.lineno}: from rpu_backend import {alias.name}')
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _GRAPH_NAMES and isinstance(node.value, ast.Name) and (node.value.id in package_aliases):
            violations.append(f'{path}:{node.lineno}: {node.value.id}.{node.attr}')
    return violations

def test_runtime_and_adapters_use_graph_namespace_directly():
    package_root = Path(__file__).resolve().parents[2]
    violations = []
    for subtree in ('runtime', 'adapters'):
        for path in sorted((package_root / subtree).rglob('*.py')):
            violations.extend(_deprecated_graph_accesses(path))
    assert violations == [], 'production code must use rpu_backend.graph.{Graph,GraphCache,GraphSignature}; deprecated top-level access remains:\n' + '\n'.join(violations)
