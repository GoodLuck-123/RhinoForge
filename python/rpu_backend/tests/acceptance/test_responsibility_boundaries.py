from __future__ import annotations
import ast
import pathlib
import pytest
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
SRC_ROOT = REPO_ROOT / 'python' / 'rpu_backend'
PI05_DIR = SRC_ROOT / 'adapters' / 'pi05'
COMPONENTS_DIR = SRC_ROOT / 'core' / 'components'
FIVE_FILES = ['loader.py', 'weights.py', 'patches.py', 'runtime.py', '__init__.py']
OWNED_SYMBOLS = {'loader.py': {'_load_and_fp16_cast', 'load_and_construct_pi05_policy'}, 'weights.py': {'swizzle_projector_inplace', '_precompute_adarms_cond_all'}, 'patches.py': {'install_class_patches'}, 'runtime.py': {'install_sample_actions_patch', '_run_denoise_fused'}, '__init__.py': {'Pi05Adapter', '_pi05_runtime_state'}}
FORBIDDEN_FOR_COMPONENTS = ('rpu_backend.model_converter', 'rpu_backend.pi05_converter', 'rpu_backend.gemma_converter', 'rpu_backend.adarms_converter', 'rpu_backend.siglip_converter', 'rpu_backend.qwenpi05_', 'rpu_backend.transformers', 'rpu_backend.adapters', 'rpu_backend.models')

def _extract_imports(path: pathlib.Path):
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield (alias.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            yield (node.module, node.lineno)

def test_pi05_canonical_responsibilities_have_one_owner():
    definitions: dict[str, list[str]] = {}
    for filename in FIVE_FILES:
        path = PI05_DIR / filename
        assert path.exists(), f'{filename} missing — Pi0.5 module boundary incomplete'
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                definitions.setdefault(node.name, []).append(filename)
    for (expected_file, symbols) in OWNED_SYMBOLS.items():
        for symbol in symbols:
            assert definitions.get(symbol) == [expected_file], f'{symbol} must be defined exactly once in {expected_file}; found in {definitions.get(symbol, [])}'

def test_components_no_outward_imports():
    failures = []
    if not COMPONENTS_DIR.exists():
        pytest.skip('core/components/ does not exist (v5-03 / C1 deleted it)')
    for path in COMPONENTS_DIR.rglob('*.py'):
        for (modname, lineno) in _extract_imports(path):
            for forbidden in FORBIDDEN_FOR_COMPONENTS:
                if modname.startswith(forbidden):
                    failures.append(f'{path.relative_to(REPO_ROOT)}:{lineno}: imports `{modname}` (AM-6 forbids — adapters/transformers/sibling direction is inward only)')
    assert not failures, 'core.components AM-6 violations:\n  ' + '\n  '.join(failures)

def test_pi05_init_exports_pi05_adapter():
    import rpu_backend.adapters.pi05 as pkg
    assert hasattr(pkg, 'Pi05Adapter'), 'adapters/pi05/__init__.py must export Pi05Adapter'
    assert hasattr(pkg, '_SWIZZLE_LOCK'), 'adapters/pi05/__init__.py must export _SWIZZLE_LOCK (Q7 lock)'
    if hasattr(pkg, '__all__'):
        assert 'Pi05Policy' not in pkg.__all__, f'adapters/pi05/__all__ leaks Pi05Policy: {pkg.__all__}; Pi05Policy must come from rpu_backend.api.policy'
