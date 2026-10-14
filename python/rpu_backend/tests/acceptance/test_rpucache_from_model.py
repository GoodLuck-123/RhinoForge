import pytest
import torch
from types import SimpleNamespace

def _make_dummy_model(*, num_hidden_layers: int=4, hidden_size: int=256, num_attention_heads: int=8, num_key_value_heads: int=4, head_dim: int | None=None, model_type: str | None=None):
    cfg = SimpleNamespace(num_hidden_layers=num_hidden_layers, hidden_size=hidden_size, num_attention_heads=num_attention_heads, num_key_value_heads=num_key_value_heads)
    if head_dim is not None:
        cfg.head_dim = head_dim
    if model_type is not None:
        cfg.model_type = model_type
    return SimpleNamespace(config=cfg)

def test_from_model_input_ids_path_device_cpu(monkeypatch):
    from rpu_backend.api.cache import RPUCache
    monkeypatch.delenv('RPU_CAUSAL_PREFILL_PADDING_BUDGET', raising=False)
    model = _make_dummy_model()
    ids = torch.zeros((1, 17), dtype=torch.long)
    cache = RPUCache.from_model(model, input_ids=ids, max_new_tokens=11, device='cpu')
    assert isinstance(cache, RPUCache)
    assert cache.max_seq_len == 17 + 64
    assert cache.batch_size == 1
    assert cache.num_layers == 4
    assert str(cache.k_caches[0].device) == 'cpu'
    assert str(cache.v_caches[0].device) == 'cpu'

def test_from_model_max_seq_len_path_device_cpu():
    from rpu_backend.api.cache import RPUCache
    model = _make_dummy_model()
    cache = RPUCache.from_model(model, max_seq_len=128, device='cpu')
    assert isinstance(cache, RPUCache)
    assert cache.max_seq_len == 128, f'max_seq_len path: expected 128, got {cache.max_seq_len}'
    assert cache.batch_size == 1
    assert cache.num_layers == 4
    assert str(cache.k_caches[0].device) == 'cpu'

def test_from_model_both_slots_raises_value_error():
    from rpu_backend.api.cache import RPUCache
    model = _make_dummy_model()
    ids = torch.zeros((1, 8), dtype=torch.long)
    with pytest.raises(ValueError, match='exactly one of'):
        RPUCache.from_model(model, input_ids=ids, max_new_tokens=4, max_seq_len=64, device='cpu')

def test_from_model_neither_slot_raises_value_error():
    from rpu_backend.api.cache import RPUCache
    model = _make_dummy_model()
    with pytest.raises(ValueError, match='exactly one of'):
        RPUCache.from_model(model, device='cpu')

def test_from_model_input_ids_without_max_new_tokens_raises_value_error():
    from rpu_backend.api.cache import RPUCache
    model = _make_dummy_model()
    ids = torch.zeros((1, 8), dtype=torch.long)
    with pytest.raises(ValueError, match='max_new_tokens'):
        RPUCache.from_model(model, input_ids=ids, device='cpu')

@pytest.mark.parametrize(('overrides', 'error', 'message'), [({'num_layers': 0}, ValueError, 'num_layers'), ({'max_seq_len': 0}, ValueError, 'max_seq_len'), ({'num_kv_heads': 0}, ValueError, 'num_kv_heads'), ({'head_dim': 15}, ValueError, 'divisible by 16'), ({'attn_tp': 9}, ValueError, 'attn_tp must be <= 8'), ({'attn_tp': 3}, ValueError, 'layout requires'), ({'dtype': torch.float32}, ValueError, 'torch.float16'), ({'device': 'cuda'}, ValueError, 'device must be')])
def test_direct_constructor_rejects_invalid_layout(overrides, error, message):
    from rpu_backend.api.cache import RPUCache
    kwargs = {'num_layers': 1, 'batch_size': 1, 'max_seq_len': 16, 'num_kv_heads': 1, 'head_dim': 16, 'device': 'cpu', 'dtype': torch.float16}
    kwargs.update(overrides)
    with pytest.raises(error, match=message):
        RPUCache(**kwargs)

def test_direct_constructor_allocates_independent_batch_slots():
    from rpu_backend.api.cache import RPUCache
    cache = RPUCache(num_layers=1, batch_size=2, max_seq_len=16, num_kv_heads=1, head_dim=16, device='cpu')
    assert cache.k_caches[0].shape[0] == 2
    assert cache.v_caches[0].shape[0] == 2

def test_from_model_batch_slots_are_qwen3_only():
    from rpu_backend.api.cache import RPUCache
    ids = torch.zeros((2, 8), dtype=torch.long)
    qwen3 = _make_dummy_model(model_type='qwen3')
    cache = RPUCache.from_model(qwen3, input_ids=ids, max_new_tokens=1, batch_size=2, device='cpu')
    assert cache.batch_size == 2
    llama = _make_dummy_model(model_type='llama')
    with pytest.raises(NotImplementedError, match='only.*qwen3'):
        RPUCache.from_model(llama, input_ids=ids, max_new_tokens=1, batch_size=2, device='cpu')

@pytest.mark.parametrize('value', [None, True, 1.5, '4'])
def test_from_model_rejects_non_integer_lengths(value):
    from rpu_backend.api.cache import RPUCache
    model = _make_dummy_model()
    ids = torch.zeros((1, 8), dtype=torch.long)
    with pytest.raises(ValueError, match='non-negative integer'):
        RPUCache.from_model(model, input_ids=ids, max_new_tokens=value, device='cpu')

def test_from_model_validates_input_shape_before_allocation():
    from rpu_backend.api.cache import RPUCache
    model = _make_dummy_model()
    with pytest.raises(ValueError, match='\\[1, seq_len\\]'):
        RPUCache.from_model(model, input_ids=torch.zeros(8, dtype=torch.long), max_new_tokens=1, device='cpu')

def test_cache_position_updates_are_bounded():
    from rpu_backend.api.cache import RPUCache
    cache = RPUCache(num_layers=1, batch_size=1, max_seq_len=16, num_kv_heads=1, head_dim=16, device='cpu')
    cache.update_position(12)
    assert cache.position == 12
    with pytest.raises(ValueError, match='overflow'):
        cache.update_position(5)
    with pytest.raises(ValueError, match='\\[0, 16\\]'):
        cache.reset_to_position(-1)
    cache.reset_to_position(4)
    assert cache.position == cache.seen_tokens == 4

@pytest.mark.parametrize('max_new_tokens', [3, 8, 11])
def test_from_model_uses_larger_of_decode_and_auto_padding(max_new_tokens):
    from rpu_backend.api.cache import RPUCache
    model = _make_dummy_model()
    model._rpu_execution = {'prefill': {'padding_rows': 'auto', 'padding_budget': 8}}
    ids = torch.zeros((1, 17), dtype=torch.long)
    cache = RPUCache.from_model(model, input_ids=ids, max_new_tokens=max_new_tokens, device='cpu')
    assert cache.max_seq_len == 17 + max(8, max_new_tokens)

@pytest.mark.parametrize(('stage', 'expected_room'), [({'padding_rows': 0}, 0), ({'padding_rows': 7}, 7), ({'padding_rows': 'auto', 'padding_budget': 9}, 9), ({'padding_budget': 11}, 11)])
def test_from_model_honors_exact_and_auto_padding(stage, expected_room):
    from rpu_backend.api.cache import RPUCache
    model = _make_dummy_model()
    model._rpu_execution = {'prefill': stage}
    ids = torch.zeros((1, 17), dtype=torch.long)
    cache = RPUCache.from_model(model, input_ids=ids, max_new_tokens=0, device='cpu')
    assert cache.max_seq_len == 17 + expected_room

@pytest.mark.parametrize(('stage', 'expected_max_seq_len'), [({'padding_rows': 'auto', 'padding_budget': 0}, 320), ({'padding_rows': 'auto', 'padding_budget': 64}, 384), ({'padding_rows': 7}, 320)])
def test_from_model_reserves_mandatory_alignment_before_optional_padding(stage, expected_max_seq_len):
    from rpu_backend.api.cache import RPUCache
    model = _make_dummy_model()
    model._rpu_execution = {'prefill': stage}
    model._rpu_prefill_execution_alignment = 16
    ids = torch.zeros((1, 313), dtype=torch.long)
    cache = RPUCache.from_model(model, input_ids=ids, max_new_tokens=0, device='cpu')
    assert cache.max_seq_len == expected_max_seq_len

def test_from_model_rejects_invalid_installed_prefill_alignment():
    from rpu_backend.api.cache import RPUCache
    model = _make_dummy_model()
    model._rpu_prefill_execution_alignment = 0
    with pytest.raises(ValueError, match='execution_alignment'):
        RPUCache.from_model(model, input_ids=torch.zeros((1, 17), dtype=torch.long), max_new_tokens=0, device='cpu')

def test_from_model_auto_padding_uses_environment(monkeypatch):
    from rpu_backend.api.cache import RPUCache
    monkeypatch.setenv('RPU_CAUSAL_PREFILL_PADDING_BUDGET', '23')
    model = _make_dummy_model()
    model._rpu_execution = {'prefill': {'padding_rows': 'auto'}}
    ids = torch.zeros((1, 17), dtype=torch.long)
    cache = RPUCache.from_model(model, input_ids=ids, max_new_tokens=0, device='cpu')
    assert cache.max_seq_len == 40

@pytest.mark.parametrize(('value', 'match'), [('bad', 'integer'), ('-1', 'non-negative')])
def test_from_model_rejects_invalid_padding_budget_environment(monkeypatch, value, match):
    from rpu_backend.api.cache import RPUCache
    monkeypatch.setenv('RPU_CAUSAL_PREFILL_PADDING_BUDGET', value)
    model = _make_dummy_model()
    ids = torch.zeros((1, 17), dtype=torch.long)
    with pytest.raises(ValueError, match=match):
        RPUCache.from_model(model, input_ids=ids, max_new_tokens=0, device='cpu')
    assert RPUCache.from_model(model, max_seq_len=17, device='cpu').max_seq_len == 17

def test_padding_rows_are_rewound_before_reserved_decode_rows_are_used():
    from rpu_backend.api.cache import RPUCache
    model = _make_dummy_model()
    model._rpu_execution = {'prefill': {'padding_rows': 32}}
    logical_len = 17
    decode_rows = 4
    cache = RPUCache.from_model(model, input_ids=torch.zeros((1, logical_len), dtype=torch.long), max_new_tokens=decode_rows, device='cpu')
    cache.update_position(logical_len + 32)
    cache.reset_to_position(logical_len)
    cache.update_position(decode_rows)
    assert cache.position == logical_len + decode_rows
    assert cache.max_seq_len == logical_len + 32
