from __future__ import annotations
import ast
from pathlib import Path
from rpu_backend.adapters.halo import text_cache, vit_cache

def test_halo_runtime_modules_have_no_optimized_away_assertions():
    for module in (text_cache, vit_cache):
        tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
        assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)], module.__file__
