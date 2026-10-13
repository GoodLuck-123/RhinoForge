"""32B cache storage grouping preserves the per-layer 7D ABI and ownership."""
from types import SimpleNamespace as NS

import pytest
import torch

from rpu_backend.api.cache import RPUCache


def model(bank_size=None):
    result = NS(config=NS(model_type="qwen3_vl_text", num_hidden_layers=64,
        hidden_size=5120, intermediate_size=25600, num_attention_heads=64,
        num_key_value_heads=8, head_dim=128))
    if bank_size is not None:
        result._rpu_kv_cache_layer_bank_size = bank_size
    return result


@pytest.mark.parametrize("bank_size,storages", [(None, 128), (8, 16)])
def test_factory_all_layers_are_disjoint_stable_and_resettable(bank_size, storages):
    cache = RPUCache.from_model(model(bank_size), max_seq_len=17, device="cpu")
    tensors = [*cache.k_caches, *cache.v_caches]
    assert len({t.untyped_storage().data_ptr() for t in tensors}) == storages
    addresses = [t.data_ptr() for t in tensors]
    assert len(set(addresses)) == 128
    intervals = sorted((t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
                       for t in tensors)
    assert all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:]))
    for i, tensor in enumerate(tensors):
        assert tensor.shape == (1, 2, 1, 8, 8, 16, 16)
        assert tensor.is_contiguous() and tensor.dtype == torch.float16
        tensor.fill_(i + 1)
    cache.update_position(17)
    cache.reset_to_position(1)
    for i, tensor in enumerate(tensors):
        assert torch.all(tensor == i + 1)
    # Export must interpret each layer independently despite shared storage.
    exported = cache.to_dynamic_cache()
    for i in (0, 31, 63):
        k, v = exported.layers[i].keys, exported.layers[i].values
        assert k.shape == v.shape == (1, 8, 1, 128)
        assert torch.all(k == i + 1) and torch.all(v == i + 65)
    cache.reset()
    assert cache.position == 0 and [t.data_ptr() for t in tensors] == addresses
    assert all(torch.count_nonzero(t) == 0 for t in tensors)


def test_partial_last_bank_keeps_all_layers():
    cache = RPUCache(5, 1, 16, 8, 128, device="cpu", _layer_bank_size=3)
    assert len(cache.k_caches) == len(cache.v_caches) == 5
    assert len({t.untyped_storage().data_ptr() for t in cache.k_caches}) == 2
    for i, (k, v) in enumerate(cache):
        k.fill_(i + 1)
        v.fill_(10 + i)
    for i, (k, v) in enumerate(cache):
        assert torch.all(k == i + 1) and torch.all(v == 10 + i)


@pytest.mark.parametrize("bad", [True, False, 0, -1, 8.0, "8"])
def test_invalid_factory_marker_fails_before_allocation(monkeypatch, bad):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid storage policy allocated data")
    monkeypatch.setattr(torch, "zeros", unexpected)
    with pytest.raises(ValueError, match="_layer_bank_size"):
        RPUCache.from_model(model(bad), max_seq_len=17, device="cpu")


def test_report_capacity_groups_128_allocations_into_16_without_allocating_payload(monkeypatch):
    calls = []
    def meta_zeros(shape, **kwargs):
        calls.append(shape)
        return torch.empty(shape, device="meta", dtype=kwargs["dtype"])
    monkeypatch.setattr(torch, "zeros", meta_zeros)
    cache = RPUCache.from_model(model(8), max_seq_len=4352, device="cpu")
    assert len(calls) == 16
    assert all(t.numel() * 2 == 8_912_896 for t in cache.k_caches + cache.v_caches)
    assert all(t.untyped_storage().nbytes() == 71_303_168 for t in cache.k_caches)


def test_long_capacity_reduces_grouping_without_changing_7d_shape(monkeypatch):
    calls = []
    def meta_zeros(shape, **kwargs):
        calls.append(shape)
        return torch.empty(shape, device="meta", dtype=kwargs["dtype"])
    monkeypatch.setattr(torch, "zeros", meta_zeros)
    cache = RPUCache.from_model(model(8), max_seq_len=131072, device="cpu")
    assert len(calls) == 64  # 256 MiB per layer, two layers per 512 MiB bank
    assert all(t.shape == (1, 8192, 1, 8, 8, 16, 16) for t in cache.k_caches)
    assert all(t.untyped_storage().nbytes() == 512 * 1024 * 1024 for t in cache.k_caches)


def test_mrope_bank_policy_publication_replacement_rollback_and_cleanup(monkeypatch):
    from rpu_backend.runtime import decoder
    from rpu_backend.api import _execution
    monkeypatch.setattr(_execution, "_require_execution_process_safe", lambda: None)
    monkeypatch.setattr(decoder, "_validate_mrope_decoder_topology", lambda *a, **k: None)
    monkeypatch.setattr(decoder, "build_mrope_cos_sin_tables", lambda *a, **k: (None, None))
    monkeypatch.setattr(decoder, "_new_decoder_graph_cache", lambda *a, **kw: object())
    monkeypatch.setattr(decoder, "_install_causal_decoder_forward", lambda *a, **k: 73)
    owner = model()
    owner.config.rope_parameters = {"mrope_section": [24, 20, 20]}
    args = dict(arch="qwen3_vl_text", chunk_envelope_for=lambda *a: None)
    assert decoder.install_mrope_text_decoder_for_rpu(owner,
        _kv_cache_layer_bank_size=8, **args) == 73
    assert owner._rpu_kv_cache_layer_bank_size == 8
    # A successful ordinary replacement restores the default policy.
    decoder.install_mrope_text_decoder_for_rpu(owner, **args)
    assert owner._rpu_kv_cache_layer_bank_size == 1
    before = dict(vars(owner))
    def fail(*args, **kwargs):
        raise RuntimeError("native installation failed")
    monkeypatch.setattr(decoder, "_install_causal_decoder_forward", fail)
    with pytest.raises(RuntimeError, match="installation failed"):
        decoder.install_mrope_text_decoder_for_rpu(owner,
            _kv_cache_layer_bank_size=8, **args)
    assert vars(owner) == before
    decoder._cleanup_causal_decoder_install(owner, had_instance_forward=False,
        original_instance_forward=None)
    assert not hasattr(owner, "_rpu_kv_cache_layer_bank_size")
