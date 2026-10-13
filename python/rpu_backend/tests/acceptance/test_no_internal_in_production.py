from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path
PRODUCTION_ROOT = 'python/rpu_backend'
EXCLUDED_PATH_PARTS = {'tests', 'debug', '__pycache__'}
_IMPORT_PATTERN = re.compile('^\\s*(from|import)\\s+rpu_backend\\._internal(\\b|\\.)', re.MULTILINE)

def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]

def _walk_production_files(root: Path):
    prod_root = root / PRODUCTION_ROOT
    for path in prod_root.rglob('*.py'):
        rel_parts = set(path.relative_to(prod_root).parts)
        if rel_parts & EXCLUDED_PATH_PARTS:
            continue
        yield path

def test_no_internal_in_production_code():
    root = _project_root()
    hits: list[str] = []
    for path in _walk_production_files(root):
        text = path.read_text(encoding='utf-8', errors='replace')
        for m in _IMPORT_PATTERN.finditer(text):
            line_no = text.count('\n', 0, m.start()) + 1
            line_text = text.splitlines()[line_no - 1].strip()
            hits.append(f'{path.relative_to(root)}:{line_no}: {line_text}')
    assert not hits, f'NS-03b violation: {len(hits)} production-tree import(s) of rpu_backend._internal found:\n  ' + '\n  '.join(hits)

def test_no_internal_in_qwen3_adapter_imports():
    import rpu_backend.adapters.qwen3
    leaked = [m for m in sys.modules if m.startswith('rpu_backend._internal')]
    assert not leaked, f'rpu_backend._internal modules leaked into adapters.qwen3: {leaked}'

def test_no_internal_in_llama_adapter_imports():
    import rpu_backend.adapters.llama
    leaked = [m for m in sys.modules if m.startswith('rpu_backend._internal')]
    assert not leaked, f'rpu_backend._internal modules leaked into adapters.llama: {leaked}'
