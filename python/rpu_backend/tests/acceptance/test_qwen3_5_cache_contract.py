from __future__ import annotations
from types import SimpleNamespace
import pytest
import torch
from rpu_backend.api import Qwen3_5Cache

def _cache_kwargs() -> dict:
    return {'num_layers': 2, 'batch_size': 1, 'max_seq_len': 17, 'num_kv_heads': 1, 'head_dim': 16, 'layer_types': ['full_attention', 'linear_attention'], 'num_v_heads': 2, 'key_head_dim': 4, 'value_head_dim': 4, 'conv_dim': 16, 'conv_kernel_dim': 3, 'device': 'cpu', 'dtype': torch.float16}

def test_hybrid_cache_allocates_parallel_layer_slots():
    cache = Qwen3_5Cache(**_cache_kwargs())
    assert cache.max_seq_len == 17
    assert cache.allocated_max_seq_len == 144
    assert len(cache.k_caches) == len(cache.recurrent_states) == 2
    assert cache.k_caches[0].numel() > 0
    assert cache.recurrent_states[0].numel() == 0
    assert cache.k_caches[1].numel() == 0
    assert tuple(cache.recurrent_states[1].shape) == (2, 4, 4)
    assert tuple(cache.conv_states[1].shape) == (8, 3, 2)

@pytest.mark.parametrize(('overrides', 'message'), [({'layer_types': ['full_attention']}, 'length'), ({'layer_types': ['full_attention', 'unknown']}, 'unsupported layer_types'), ({'conv_dim': 15}, 'divisible by 8'), ({'num_v_heads': 0}, 'num_v_heads'), ({'conv_kernel_dim': 0}, 'conv_kernel_dim')])
def test_hybrid_cache_rejects_invalid_structure(overrides, message):
    kwargs = _cache_kwargs()
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        Qwen3_5Cache(**kwargs)

def test_hybrid_cache_rewind_does_not_truncate_non_integer_position():
    cache = Qwen3_5Cache(**_cache_kwargs())
    with pytest.raises(ValueError, match='must be an integer'):
        cache.reset_to_position(0.5)

def test_from_config_validates_before_layout_math():
    config = SimpleNamespace(num_hidden_layers=2, num_key_value_heads=0, head_dim=16, layer_types=['full_attention', 'linear_attention'], linear_num_value_heads=2, linear_num_key_heads=2, linear_key_head_dim=4, linear_value_head_dim=4, linear_conv_kernel_dim=3)
    with pytest.raises(ValueError, match='num_key_value_heads'):
        Qwen3_5Cache.from_config(config, max_seq_len=17, device='cpu')
