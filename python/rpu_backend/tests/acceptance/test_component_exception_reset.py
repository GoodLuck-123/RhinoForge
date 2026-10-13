from __future__ import annotations
import ast
import importlib
import inspect
import pytest
_LIFTED_PATCHES = [('rpu_backend.adapters.pi05.gemma', 'patch_gemma_model_for_rpu_all_layers_once', 'gemma_forward'), ('rpu_backend.adapters.pi05.adarms', 'patch_adarms_model_for_rpu_all_layers_once', 'adarms_forward'), ('rpu_backend.adapters.siglip', 'patch_siglip_model_for_rpu_all_layers_once', 'siglip_forward'), ('rpu_backend.adapters.paligemma.gemma2', 'patch_gemma2_model_for_rpu_all_layers_once', 'gemma2_forward')]

def _is_torch_ops_rpu_call(node: ast.AST, expected_attr: str) -> bool:
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    a3 = node.func
    if a3.attr != expected_attr:
        return False
    a2 = a3.value
    if not isinstance(a2, ast.Attribute) or a2.attr != 'rpu':
        return False
    a1 = a2.value
    if not isinstance(a1, ast.Attribute) or a1.attr != 'ops':
        return False
    a0 = a1.value
    return isinstance(a0, ast.Name) and a0.id == 'torch'

def _first_call_lineno(fn: ast.FunctionDef, expected_attr: str) -> int | None:
    for node in ast.walk(fn):
        if _is_torch_ops_rpu_call(node, expected_attr):
            return node.lineno
    return None

@pytest.mark.parametrize('module_path,patch_fn_name,fused_op_attr', _LIFTED_PATCHES)
def test_patch_function_body_contains_spm_reset_call(module_path, patch_fn_name, fused_op_attr):
    mod = importlib.import_module(module_path)
    src = inspect.getsource(mod)
    tree = ast.parse(src)
    patch_fn = next((n for n in ast.iter_child_nodes(tree) if isinstance(n, ast.FunctionDef) and n.name == patch_fn_name), None)
    assert patch_fn is not None, f'{patch_fn_name} not found at top level of {module_path}'
    nested = [n for n in ast.walk(patch_fn) if isinstance(n, ast.FunctionDef) and n is not patch_fn and n.name.startswith('rpu_') and n.name.endswith('forward')]
    assert nested, f'{module_path}.{patch_fn_name} defines no nested `rpu_*_forward` closure; v3 contract requires at least one (lifted from pi05_all_layers_once.py per B4)'
    for inner in nested:
        if fused_op_attr == 'gemma_forward':
            assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_run_gemma_prefill' for n in ast.walk(inner))
            inner = next(n for n in ast.walk(patch_fn) if isinstance(n, ast.FunctionDef) and n.name == '_run_gemma_prefill')
            dispatch = next(n for n in ast.walk(inner) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'native_forward' for t in n.targets))
            assert isinstance(dispatch.value, ast.IfExp)
            assert dispatch.value.body.attr == 'gemma_prefill_kv_only'
            assert dispatch.value.orelse.attr == fused_op_attr
            fused_lineno = next(n.lineno for n in ast.walk(inner) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'native_forward')
        else:
            fused_lineno = _first_call_lineno(inner, fused_op_attr)
        reset_lineno = _first_call_lineno(inner, 'spm_alloc_reset_temporary')
        assert fused_lineno is not None, f'{module_path}.{patch_fn_name}::{inner.name} body does not call `torch.ops.rpu.{fused_op_attr}(...)` — fused-op emission missing'
        assert reset_lineno is not None, f'{module_path}.{patch_fn_name}::{inner.name} body does not call `torch.ops.rpu.spm_alloc_reset_temporary()` — v3 SPM-reset postcondition violated'
        assert reset_lineno > fused_lineno, f'{module_path}.{patch_fn_name}::{inner.name} calls spm_alloc_reset_temporary() at line {reset_lineno} which is NOT after the fused `torch.ops.rpu.{fused_op_attr}` call at line {fused_lineno} — v3 subsystem-boundary postcondition violated (reset must fire AFTER the forward op)'
