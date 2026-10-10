"""The shared runner rejects logical overflow before native cache writes."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.api.cache import RPUCache
from rpu_backend.runtime.decoder import _run_causal_decoder_forward


def _cache(rows, position):
    cache = RPUCache(1, 1, rows, 8, 128, device="cpu")
    cache.position = position
    return cache


@pytest.mark.parametrize("position,allowed", [(512, True), (513, False), (-1, False)])
def test_decode_uses_logical_capacity_before_native_write(monkeypatch, position, allowed):
    cache = _cache(513, position)
    assert cache.k_caches[0].shape[1] * 16 == 528
    calls = []

    def native(_handle, hidden, *_args):
        calls.append(cache.position)
        return hidden.clone()

    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_forward", native, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_get_resolved_chunk_size",
                        lambda _handle: 1, raising=False)
    model = SimpleNamespace(_rpu_decode_stage_descriptor=(11, 22))
    args = dict(inputs_embeds=torch.zeros(1, 1, 16, dtype=torch.float16), past_key_values=cache)
    if allowed:
        raw, returned = _run_causal_decoder_forward(model, 1, **args)
        assert raw.shape == (1, 1, 16) and returned is cache
        assert calls == [position] and cache.position == position + 1
    else:
        with pytest.raises(ValueError, match="position overflow before execution"):
            _run_causal_decoder_forward(model, 1, **args)
        assert calls == [] and cache.position == position


@pytest.mark.parametrize("capacity,allowed", [(25, False), (26, True)])
def test_prefill_checks_temporary_rows_but_advances_logical_rows(monkeypatch, capacity, allowed):
    cache = _cache(capacity, 10)
    calls = []

    def native(_handle, hidden, *_args):
        calls.append(cache.position)
        return hidden.clone()

    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_forward", native, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_get_resolved_chunk_size",
                        lambda _handle: 16, raising=False)
    plan = SimpleNamespace(graph_mode=0, selected=SimpleNamespace(
        execution_len=16, padding_rows=3,
        stage_tuple=SimpleNamespace(compute_chunk=16, physical_descriptor=(11, 22))))
    args = dict(inputs_embeds=torch.zeros(1, 16, 16, dtype=torch.float16),
                past_key_values=cache, prefill_plan=(16, 16, plan))
    if allowed:
        raw, returned = _run_causal_decoder_forward(SimpleNamespace(), 1, **args)
        assert raw.shape == (1, 13, 16) and returned is cache
        assert calls == [10] and cache.position == 23
    else:
        with pytest.raises(ValueError, match="position overflow before execution"):
            _run_causal_decoder_forward(SimpleNamespace(), 1, **args)
        assert calls == [] and cache.position == 10


def test_padding_preserves_real_rows_and_retained_input_with_logical_kv(monkeypatch):
    cache = _cache(26, 10)
    hidden = torch.arange(13 * 16, dtype=torch.float16).view(1, 13, 16)
    saved = hidden.clone()
    calls = []
    def native(_handle, value, *_args):
        calls.append(value)
        assert value.shape == (1, 16, 16) and value.is_contiguous()
        assert torch.equal(value[:, :13], saved)
        assert torch.count_nonzero(value[:, 13:]).item() == 0
        return value.clone()
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_forward", native, raising=False)
    plan = SimpleNamespace(graph_mode=0, selected=SimpleNamespace(
        execution_len=16, padding_rows=3,
        stage_tuple=SimpleNamespace(compute_chunk=16, physical_descriptor=(11, 22))))
    raw, returned = _run_causal_decoder_forward(SimpleNamespace(), 1,
        inputs_embeds=hidden, past_key_values=cache, prefill_plan=(16, 16, plan))
    assert returned is cache and cache.position == 23
    assert torch.equal(raw, saved) and torch.equal(hidden, saved)
    assert calls[0].data_ptr() != hidden.data_ptr()


@pytest.mark.parametrize("source_device", ["cpu", "rpu"])
@pytest.mark.parametrize("layout", ["matrix", "thw", "strided"])
def test_mrope_runner_preserves_current_cpu_positions_before_native(
        monkeypatch, source_device, layout):
    """Real runner; device metadata and the final native computation are delegates."""
    class RpuPositions(torch.Tensor):
        @property
        def device(self):
            return SimpleNamespace(type="rpu")
    matrix = torch.zeros(3, 3, dtype=torch.int32 if layout == "matrix" else torch.int64)
    if layout == "thw":
        positions = matrix.T.unsqueeze(1)
    elif layout == "strided":
        positions = torch.zeros(3, 6, dtype=torch.int64)[:, ::2]
    else:
        positions = matrix
    if source_device == "rpu":
        positions = positions.as_subclass(RpuPositions)
    raw_alias = positions.as_subclass(torch.Tensor).numpy()
    version = positions._version
    original_to = torch.Tensor.to
    def no_upload(tensor, *args, **kwargs):
        assert not ((args and str(args[0]) == "rpu") or str(kwargs.get("device")) == "rpu")
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", no_upload)
    seen = []
    def native(handle, hidden, keys, values, mask, position, causal, pos, *_args):
        assert pos.device.type == source_device and pos.dtype == torch.int32
        assert pos.shape == (3, 3) and pos.is_contiguous()
        actual = pos.as_subclass(torch.Tensor)
        assert torch.equal(actual, torch.full_like(actual, current))
        seen.append(actual.clone())
        return hidden.clone()
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_forward", native, raising=False)
    plan = SimpleNamespace(graph_mode=0, selected=SimpleNamespace(
        execution_len=3, padding_rows=0,
        stage_tuple=SimpleNamespace(compute_chunk=3, physical_descriptor=(11, 22))))
    cache = _cache(64, 0)
    for current in (5, 9, 5):
        raw_alias.fill(current)
        assert positions._version == version
        cache.position = 0
        output, returned = _run_causal_decoder_forward(SimpleNamespace(), 1,
            inputs_embeds=torch.zeros(1, 3, 16, dtype=torch.float16),
            position_ids=positions, past_key_values=cache, prefill_plan=(3, 3, plan))
        assert output.shape == (1, 3, 16) and returned is cache and cache.position == 3
    assert torch.equal(seen[0], seen[2]) and not torch.equal(seen[0], seen[1])
