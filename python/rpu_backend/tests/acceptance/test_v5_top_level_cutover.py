import pytest
import torch

def test_cut05_top_level_all_byte_identical():
    import rpu_backend
    expected = ['__version__', 'RPUCache', 'RPUModelForCausalLM', 'reset_graph_cache']
    assert isinstance(rpu_backend.__all__, list), f'__all__ must be a list, got {type(rpu_backend.__all__).__name__}'
    assert rpu_backend.__all__ == expected, f'CUT-05 broken: got {rpu_backend.__all__!r}; expected {expected!r}'

def test_distribution_version_and_api_level_are_explicit():
    import tomllib
    from pathlib import Path
    import rpu_backend
    from rpu_backend.api import API_VERSION
    repo_root = Path(__file__).resolve().parents[4]
    distribution_version = (repo_root / 'VERSION').read_text().strip()
    assert rpu_backend.__version__ == distribution_version
    assert API_VERSION == '5.0.0'
DEPRECATED_FROM_IMPORT_NAMES = ['build_rpu_cache', 'build_cache', 'set_chunk_size', 'set_spm_mode', 'set_fuse_lm_head', 'get_fuse_lm_head', 'set_fuse_lm_head_weights', 'convert_linear_weights_inplace', 'model_converter', 'patch_rmsnorm_class', 'patch_attention_for_rpu', 'patch_decoder_layer_for_rpu_fused']

@pytest.mark.parametrize('name', DEPRECATED_FROM_IMPORT_NAMES)
def test_cut01a_from_import_raises_ImportError(name):
    with pytest.raises(ImportError):
        exec(f'from rpu_backend import {name}')

@pytest.mark.parametrize('name', DEPRECATED_FROM_IMPORT_NAMES)
def test_cut01b_attr_access_raises_AttributeError(name):
    import rpu_backend
    with pytest.raises(AttributeError) as excinfo:
        _ = getattr(rpu_backend, name)
    msg = str(excinfo.value)
    assert 'rpu_backend' in msg and name in msg, f"AttributeError message missing 'rpu_backend' or name {name!r}: {msg!r}"
DELETED_TORCH_RPU_SETTERS = ['set_chunk_size', 'set_spm_mode', 'set_fuse_lm_head', 'get_fuse_lm_head', 'set_fuse_lm_head_weights']

@pytest.mark.parametrize('name', DELETED_TORCH_RPU_SETTERS)
def test_cut02_and_adr10_15_torch_rpu_setters_deleted(name):
    assert getattr(torch.rpu, name, None) is None, f'torch.rpu.{name} MUST NOT exist in v5 (CUT-02 / ADR §10 #15 broken)'

def test_cut03_build_rpu_cache_function_deleted():
    with pytest.raises(ImportError):
        from rpu_backend.transformers._causal_lm import build_rpu_cache

def test_cut03_build_cache_function_deleted():
    with pytest.raises(ImportError):
        from rpu_backend.transformers._causal_lm import build_cache

def test_cut03_top_level_build_cache_unbound():
    import rpu_backend
    with pytest.raises(ImportError):
        from rpu_backend import build_cache
    with pytest.raises(AttributeError):
        _ = rpu_backend.build_cache
EXT02_HINT_MATRIX = [('build_rpu_cache', 'RPUCache.from_model'), ('build_cache', 'RPUCache.from_model'), ('model_converter', 'swizzle_model_inplace'), ('convert_linear_weights_inplace', 'swizzle_model_inplace'), ('Pi05Policy', 'rpu_backend.api'), ('set_chunk_size', 'causal_decoder_set_chunk_size_override'), ('set_spm_mode', '_rpu_spm_mode'), ('set_fuse_lm_head', 'causal_decoder_set_lm_head'), ('RPUKVCache', 'RPUCache'), ('qwenpi05_fused_converter', 'supported policy conversion API')]

@pytest.mark.parametrize('name,expected_substring', EXT02_HINT_MATRIX)
def test_ext02_attr_access_message_contains_migration_hint(name, expected_substring):
    import rpu_backend
    with pytest.raises(AttributeError) as excinfo:
        _ = getattr(rpu_backend, name)
    msg = str(excinfo.value)
    assert expected_substring in msg, f'EXT-02 broken for {name!r}: expected substring {expected_substring!r} missing from message {msg!r}'
    assert 'Did you mean' in msg, f"EXT-02 prefix missing for {name!r}: message must contain 'Did you mean' per REQUIREMENTS line 138 (R1 MEDIUM-1); got msg={msg!r}"

def test_v5_04_core_weights_cache_deleted():
    with pytest.raises(ImportError):
        from rpu_backend.core.weights.cache import RPUCache as _CoreRPUCache

def test_transformers_pi05_module_gone():
    with pytest.raises(ImportError):
        from rpu_backend.transformers.pi05 import Pi05Policy
