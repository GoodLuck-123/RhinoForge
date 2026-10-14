"""KV-cache position bookkeeping contracts — correctness-matrix axes 22/23/24.

The checks cover capacity, reset, and repeated prefix reuse:

  axis 23 (capacity)  — the cache RAISES at max_seq_len, it does not wrap.
                        A wrap would silently overwrite live prefix K/V.
  axis 22 (reset)     — reset() vs reset_to_position(pos), including the
                        out-of-range and wrong-type rejections.
  axis 24 (multi-round) — prefill -> decode -> reset_to_position(P), repeated.
                        This is the Pi0.5 / VLA shared-prefix pattern, where
                        reset_to_position MUST preserve the [0, P) prefix data
                        (insert kernels overwrite [P, ...) before SDPA reads it).

Hardware-free by construction: RPUCache accepts device="cpu" and every branch
under test is Python-side bookkeeping in rpu_backend/api/cache.py. The K/V
tensors here total ~2 MB, so this belongs in the host lint job rather than on
the single shared device runner.
"""
import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from rpu_backend.api.cache import RPUCache
from rpu_backend.adapters.gemma4.cache import Gemma4KVCache
from rpu_backend.adapters.gemma4.geometry import layer_geometry

MAX_SEQ = 8192


def _cache(max_seq_len=MAX_SEQ, num_layers=1):
    return RPUCache(
        num_layers=num_layers, batch_size=1, max_seq_len=max_seq_len,
        num_kv_heads=2, head_dim=64, device="cpu",
    )


def test_hybrid_cache_never_allocates_unused_kv_layers(monkeypatch):
    from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache
    allocated_kv = []
    original_zeros = torch.zeros

    def zeros(*args, **kwargs):
        tensor = original_zeros(*args, **kwargs)
        if tensor.ndim == 7:
            allocated_kv.append(tensor)
        return tensor

    monkeypatch.setattr(torch, "zeros", zeros)
    layer_types = ["linear_attention"] * 3 + ["full_attention"]
    cache = Qwen3_5Cache(64, 1, 65, 24, 16, layer_types * 16,
                        8, 1, 1, 16, 4, device="cpu")
    assert len(allocated_kv) == 32  # Two tensors for each of 16 full layers.
    addresses = [tensor.data_ptr() for tensor in cache.k_caches + cache.v_caches]
    for i in range(64):
        assert bool(cache.k_caches[i].numel()) == (i % 4 == 3)
        assert bool(cache.v_caches[i].numel()) == (i % 4 == 3)
    for tensor in allocated_kv + cache.recurrent_states + cache.conv_states:
        tensor.fill_(1)
    cache.update_position(17)
    cache.reset()
    assert cache.position == 0
    assert all(not tensor.any() for tensor in allocated_kv + cache.recurrent_states + cache.conv_states)
    assert addresses == [tensor.data_ptr() for tensor in cache.k_caches + cache.v_caches]


class _HostStorageRpuTensor(torch.Tensor):
    """Only device dispatch is mocked; storage and alias versions are real."""
    @property
    def device(self):
        return SimpleNamespace(type="rpu")


def _aligned_rpu_tensor(elements=128, *, misalign=0):
    if elements == 0:
        return torch.empty(0, dtype=torch.float16).as_subclass(_HostStorageRpuTensor)
    raw = bytearray(elements * 2 + 256)
    address = ctypes.addressof(ctypes.c_char.from_buffer(raw))
    offset = (-address) % 256 + misalign
    tensor = torch.frombuffer(memoryview(raw)[offset:offset + elements * 2], dtype=torch.float16)
    return tensor.as_subclass(_HostStorageRpuTensor)


def _reset_cache(kind):
    from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache
    from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache

    cls = {"plain": RPUCache, "dense": Qwen3_5Cache, "moe": Qwen3_5MoeCache}[kind]
    cache = object.__new__(cls)
    cache.position = cache._seen_tokens = 17
    cache._reset_groups = None
    empty = _aligned_rpu_tensor(0)
    # Tiny allocations preserve a real MoE's 10 full + 30 GDN layer topology.
    layers = 40 if kind == "moe" else 4
    cache.k_caches, cache.v_caches = [], []
    cache.recurrent_states, cache.conv_states = [], []
    for index in range(layers):
        full = kind == "plain" or index % 4 == 3
        cache.k_caches.append(_aligned_rpu_tensor() if full else empty)
        cache.v_caches.append(_aligned_rpu_tensor() if full else empty)
        if kind != "plain":
            cache.recurrent_states.append(empty if full else _aligned_rpu_tensor())
            cache.conv_states.append(empty if full else _aligned_rpu_tensor())
    cache.gdn_states = cache.recurrent_states
    return cache


def _reset_tensors(cache):
    return ([tensor for pair in zip(cache.k_caches, cache.v_caches) for tensor in pair]
            + cache.recurrent_states + cache.conv_states)


def _gemma4_generation_cache(monkeypatch, *, capacity=64, inference=False):
    config = SimpleNamespace(
        num_hidden_layers=42, num_kv_shared_layers=18, head_dim=256,
        global_head_dim=512, num_attention_heads=8, num_key_value_heads=2,
        layer_types=["full_attention" if i % 6 == 5 else "sliding_attention"
                     for i in range(42)], sliding_window=512, vocab_size=3)
    geom = layer_geometry(config)

    def allocate(shape, *, dtype, device):
        assert dtype == torch.float16 and device == "rpu"
        import math
        return _aligned_rpu_tensor(math.prod(shape)).view(shape)

    # Run the real mixed-width allocator/source-map constructor, substituting
    # only aligned host storage for device allocation.
    with monkeypatch.context() as allocator, torch.inference_mode(inference):
        allocator.setattr(torch, "zeros", allocate)
        cache = Gemma4KVCache(geom, max_seq_len=capacity)
    model = SimpleNamespace(
        _rpu_gemma4_geom=geom,
        config=SimpleNamespace(architectures=["Gemma4ForConditionalGeneration"],
                               text_config=config))
    return cache, model


def _generation_runner():
    path = Path(__file__).resolve().parents[1] / "examples/common/generation.py"
    spec = importlib.util.spec_from_file_location("gemma4_generation_reset_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("tokens,decode_steps", [(1, 0), (13, 3), (13, 4), (16, 1), (31, 1)])
@pytest.mark.parametrize("inference,reset_inference", [(False, False), (False, True), (True, True)])
def test_gemma4_generation_prefix_clears_all_lanes_and_preserves_shared_owners(
        monkeypatch, tokens, decode_steps, inference, reset_inference):
    cache, model = _gemma4_generation_cache(monkeypatch, inference=inference)
    runner = _generation_runner()
    owners = [tensors[i] for i in cache._owner_layers
              for tensors in (cache.k_caches, cache.v_caches)]
    public_tensors = cache.k_caches + cache.v_caches
    pointers = [tensor.data_ptr() for tensor in public_tensors]
    ranges, blocks = [], (tokens + decode_steps + 15) // 16

    def batch(views, block_index):
        assert block_index == 0 and len(views) == 48
        assert cache.position == cache._seen_tokens == 19
        for view, owner in zip(views, owners):
            assert view.data_ptr() == owner.data_ptr()
            assert view.shape == (1, 1, owner[:, :blocks].numel())
            if not view.is_inference():
                torch.autograd.graph.increment_version(view)
            ctypes.memset(view.data_ptr(), 0, view.nbytes)
        ranges.append([(view.data_ptr(), view.nbytes) for view in views])

    monkeypatch.setattr(torch.ops.rpu, "zero_contiguous_tensor_blocks_sized", batch, raising=False)
    with torch.inference_mode(reset_inference):
        aliases = [tensor.view_as(tensor) for tensor in public_tensors]
        for byte in (0x11, 0x22, 0x11):
            versions = None if inference else [t._version for t in public_tensors]
            for owner in owners:
                ctypes.memset(owner.data_ptr(), byte, owner.nbytes)
            if not inference:
                assert [t._version for t in public_tensors] == versions
            cache.position = cache._seen_tokens = 19
            runner._reset_generation_cache(
                model, cache, torch.ones((1, tokens), dtype=torch.long), decode_steps)
            assert cache.position == cache._seen_tokens == 0
            assert [t.data_ptr() for t in public_tensors] == pointers
            for index, (current, original) in enumerate(zip(cache.k_caches + cache.v_caches, public_tensors)):
                assert current is original
                assert not current[:, :blocks].count_nonzero()
                tail = current[:, blocks:]
                assert ctypes.string_at(tail.data_ptr(), tail.nbytes) == bytes([byte]) * tail.nbytes
                if not inference:
                    assert current._version == versions[index] + 1
                    assert aliases[index]._version == current._version
            # Public reset must still clear the entire allocation, including
            # the dirty tail intentionally excluded by the private entry.
            before_public = None if inference else [t._version for t in public_tensors]
            cache.reset()
            assert all(not tensor.count_nonzero() for tensor in public_tensors)
            if not inference:
                assert [t._version for t in public_tensors] == [v + 1 for v in before_public]
    assert len(ranges) == 3
    assert all(sum(size for _, size in call) == blocks * 16 * 224 * 1024 for call in ranges)


@pytest.mark.parametrize("mode", ["per_call", "legacy_dense_vl"])
def test_gemma4_generation_uses_actual_prompt_and_requested_decode_bound(monkeypatch, mode):
    runner = _generation_runner()
    cache, config_model = _gemma4_generation_cache(monkeypatch)
    ranges, calls = [], []

    def batch(views, index):
        assert index == 0
        ranges.append(sum(view.nbytes for view in views))
        for view in views:
            torch.autograd.graph.increment_version(view)
            ctypes.memset(view.data_ptr(), 0, view.nbytes)

    monkeypatch.setattr(torch.ops.rpu, "zero_contiguous_tensor_blocks_sized", batch, raising=False)

    def forward(**kwargs):
        calls.append(kwargs["input_ids"].shape[1])
        cache.update_position(kwargs["input_ids"].shape[1])
        return SimpleNamespace(logits=torch.tensor([[[0., 2., 1.]]]))

    model = Mock(side_effect=forward)
    model.config, model._rpu_gemma4_geom = config_model.config, config_model._rpu_gemma4_geom
    for tokens, decode_steps in [(13, 3), (13, 4), (17, 16)]:
        result = runner._generate(
            model, cache, torch.ones((1, tokens), dtype=torch.long), decode_steps,
            stop_on_eos=False, timing_mode=mode,
            input_metadata={"prefill_tokens": 1, "decode_steps": 0, "prompt_file": "not-the-input"})
        assert result["generation"]["prefill_tokens"] == tokens
        assert result["generation"]["decode_calls"] == decode_steps
        assert cache.position == tokens + decode_steps
    assert ranges == [16 * 224 * 1024, 32 * 224 * 1024, 48 * 224 * 1024]
    assert calls == [13] + [1] * 3 + [13] + [1] * 4 + [17] + [1] * 16


@pytest.mark.parametrize("mutation", [
    "architecture", "subclass", "geometry", "source_map", "owners", "shared_alias",
    "overlap", "attn_tp", "batch", "capacity", "cpu", "dtype", "shape", "offset",
    "strided", "partial_storage", "unaligned", "float_ids", "empty_ids", "batch_ids",
    "non_tensor_ids", "bool_decode", "float_decode", "negative_decode",
])
def test_gemma4_generation_prefix_falls_back_to_public_reset(monkeypatch, mutation):
    runner = _generation_runner()
    cache, model = _gemma4_generation_cache(monkeypatch)
    ids, decode = torch.ones((1, 13), dtype=torch.long), 3
    i = cache._owner_layers[-1]
    old = cache.k_caches[i]
    if mutation == "architecture":
        model.config.architectures = ["Gemma3ForConditionalGeneration"]
    elif mutation == "subclass":
        class OtherCache(Gemma4KVCache):
            pass
        cache.__class__ = OtherCache
    elif mutation == "geometry":
        model._rpu_gemma4_geom = []
    elif mutation == "source_map":
        cache.kv_source_layer_for_layer[-1] = 0
    elif mutation == "owners":
        cache._owner_layers = list(reversed(cache._owner_layers))
    elif mutation == "shared_alias":
        cache.k_caches[-1] = old.view_as(old)
    elif mutation == "overlap":
        cache.v_caches[i] = old
        for j, source in enumerate(cache.kv_source_layer_for_layer):
            if source == i:
                cache.v_caches[j] = old
    elif mutation == "attn_tp":
        cache.attn_tp = 4
    elif mutation == "batch":
        cache.batch_size = 2
    elif mutation == "capacity":
        decode = 52
    elif mutation == "float_ids":
        ids = ids.float()
    elif mutation == "empty_ids":
        ids = ids[:, :0]
    elif mutation == "batch_ids":
        ids = ids.expand(2, -1)
    elif mutation == "non_tensor_ids":
        ids = [[1, 2]]
    elif mutation.endswith("_decode"):
        decode = {"bool_decode": True, "float_decode": 3., "negative_decode": -1}[mutation]
    else:
        if mutation == "cpu":
            current = old.as_subclass(torch.Tensor)
        elif mutation == "dtype":
            current = old.float()
        elif mutation == "shape":
            current = old.flatten()
        elif mutation == "offset":
            current = _aligned_rpu_tensor(old.numel() + 128)[128:].view(old.shape)
        elif mutation == "strided":
            current = old.transpose(-1, -2)
        elif mutation == "partial_storage":
            current = _aligned_rpu_tensor(old.numel() + 128)[:old.numel()].view(old.shape)
        else:
            current = _aligned_rpu_tensor(old.numel(), misalign=2).view(old.shape)
        for j, source in enumerate(cache.kv_source_layer_for_layer):
            if source == i:
                cache.k_caches[j] = current
    cache.reset = Mock(wraps=cache.reset)
    monkeypatch.setattr(torch.ops.rpu, "zero_contiguous_tensor_blocks_sized",
                        lambda *_: pytest.fail("invalid private reset entered native batch"), raising=False)
    runner._reset_generation_cache(model, cache, ids, decode)
    cache.reset.assert_called_once_with()
    assert cache.position == cache._seen_tokens == 0


@pytest.mark.parametrize("fault", ["requires_grad", "inference_outside", "native_failure"])
def test_gemma4_generation_prefix_preserves_mutation_and_native_errors(monkeypatch, fault):
    runner = _generation_runner()
    cache, model = _gemma4_generation_cache(monkeypatch, inference=fault == "inference_outside")
    cache.position = cache._seen_tokens = 19
    if fault == "requires_grad":
        cache.k_caches[0].requires_grad_()
    cache.reset = Mock(wraps=cache.reset)

    def batch(*_):
        assert fault == "native_failure"
        raise RuntimeError("native validation failure")

    monkeypatch.setattr(torch.ops.rpu, "zero_contiguous_tensor_blocks_sized", batch, raising=False)
    with pytest.raises(RuntimeError, match="leaf Variable|Inference tensors|inference tensor|native validation failure"):
        runner._reset_generation_cache(model, cache, torch.ones((1, 13), dtype=torch.long), 3)
    assert cache.reset.call_count == (fault != "native_failure")
    assert cache.position == (19 if fault == "native_failure" else 0)


@pytest.mark.parametrize("kind", ["plain", "dense", "moe"])
@pytest.mark.parametrize("inference,reset_inference", [(False, False), (False, True), (True, True)],
                         ids=["ordinary", "ordinary_in_inference", "inference"])
def test_whole_storage_reset_batches_current_dirty_aliases_and_versions(
        monkeypatch, kind, inference, reset_inference):
    with torch.inference_mode(inference):
        cache = _reset_cache(kind)
    with torch.inference_mode(reset_inference):
        tensors = _reset_tensors(cache)
        aliases = [tensor.view_as(tensor) for tensor in tensors]
        pointers = [tensor.data_ptr() for tensor in tensors]
        publications = []

        def batch(blocks, index):
            assert index == 0 and len(blocks) == len(tensors)
            assert [tensor.data_ptr() for tensor in blocks] == pointers
            assert all(tensor.shape == (1, 1, original.numel())
                       for tensor, original in zip(blocks, tensors))
            for tensor in blocks:
                if not tensor.is_inference():
                    torch.autograd.graph.increment_version(tensor)
                if tensor.numel():
                    ctypes.memset(tensor.data_ptr(), 0, tensor.nbytes)
            # The compiled native-body test checks the real force gate/sync.
            publications.append([(t.data_ptr(), t.nbytes) for t in blocks if t.numel()])

        monkeypatch.setattr(torch.ops.rpu, "zero_contiguous_tensor_blocks_sized", batch, raising=False)
        for byte in (0x11, 0x22, 0x11):
            versions = None if inference else [t._version for t in tensors]
            for tensor in tensors:
                if tensor.numel():
                    ctypes.memset(tensor.data_ptr(), byte, tensor.nbytes)
            if not inference:
                assert [t._version for t in tensors] == versions
            cache.position = cache._seen_tokens = 17
            cache.reset()
            assert cache.position == cache._seen_tokens == 0
            assert [t.data_ptr() for t in tensors] == pointers
            assert all(not t.count_nonzero() for t in tensors)
            if not inference:
                for index, tensor in enumerate(tensors):
                    occurrences = sum(other is tensor for other in tensors)
                    assert tensor._version == versions[index] + occurrences
                    assert aliases[index]._version == tensor._version
        assert len(publications) == 3
        assert all(len(ranges) == (80 if kind == "moe" else 8) for ranges in publications)


@pytest.mark.parametrize("kind", ["plain", "dense", "moe"])
@pytest.mark.parametrize("mutation", ["offset", "strided", "transpose", "partial", "empty_view",
                                     "unaligned", "short_storage", "cpu", "groups", "replace"])
def test_reset_guard_preserves_current_views_neighbours_and_retired_owners(monkeypatch, kind, mutation):
    cache = _reset_cache(kind)
    # Put the mutation in an actual full K owner, rather than a GDN placeholder.
    index = 0 if kind == "plain" else 3
    retired = cache.k_caches[index]
    retired.fill_(7)
    if mutation == "offset":
        current = retired[1:]
    elif mutation == "strided":
        current = retired[::2]
    elif mutation == "transpose":
        current = retired.view(8, 16).T
    elif mutation == "partial":
        current = retired[:64]
    elif mutation == "empty_view":
        current = retired[:0]
    elif mutation == "unaligned":
        current = _aligned_rpu_tensor(misalign=2)
    elif mutation == "short_storage":
        current = _aligned_rpu_tensor(127)
    elif mutation == "cpu":
        current = torch.ones(128, dtype=torch.float16)
    elif mutation == "replace":
        current = _aligned_rpu_tensor()
    else:
        current = retired
        # Retain the existing grouped reset route; a stale witness falls back
        # to current lists just as before this optimization.
        cache._reset_groups = (([], ()), ([], ()))
    cache.k_caches[index] = current
    current.fill_(5)
    expected = retired.as_subclass(torch.Tensor).clone()
    expected_current = ({"offset": lambda x: x[1:], "strided": lambda x: x[::2],
                         "transpose": lambda x: x.view(8, 16).T,
                         "partial": lambda x: x[:64],
                         "empty_view": lambda x: x[:0]}.get(mutation, lambda x: x))(expected)
    if mutation not in {"replace", "unaligned", "short_storage", "cpu"}:
        expected_current.zero_()
    calls = []

    def batch(blocks, block_index):
        calls.append(blocks)
        for tensor in blocks:
            tensor.zero_()

    monkeypatch.setattr(torch.ops.rpu, "zero_contiguous_tensor_blocks_sized", batch, raising=False)
    version = current._version
    cache.reset()
    assert current._version == version + 1
    assert not current.count_nonzero() and cache.position == cache._seen_tokens == 0
    assert torch.equal(retired.as_subclass(torch.Tensor), expected)
    assert len(calls) == (1 if mutation == "replace" else 0)


@pytest.mark.parametrize("kind", ["plain", "dense", "moe"])
@pytest.mark.parametrize("invalid", ["requires_grad", "inference_outside"])
def test_reset_guard_keeps_inplace_mutation_errors(monkeypatch, kind, invalid):
    with torch.inference_mode(invalid == "inference_outside"):
        cache = _reset_cache(kind)
    if invalid == "requires_grad":
        cache.k_caches[0] = _aligned_rpu_tensor().requires_grad_()
    monkeypatch.setattr(torch.ops.rpu, "zero_contiguous_tensor_blocks_sized",
                        lambda *_: pytest.fail("invalid scalar mutation entered batch"), raising=False)
    with pytest.raises(RuntimeError, match="leaf Variable|Inference tensors|inference tensor"):
        cache.reset()
    assert cache.position == cache._seen_tokens == 0


@pytest.mark.parametrize("kind", ["plain", "dense", "moe"])
def test_reset_keeps_unpaired_cache_list_tail_unchanged(monkeypatch, kind):
    cache = _reset_cache(kind)
    tail = _aligned_rpu_tensor().fill_(7)
    cache.k_caches.append(tail)
    version = tail._version
    monkeypatch.setattr(torch.ops.rpu, "zero_contiguous_tensor_blocks_sized",
                        lambda *_: pytest.fail("unpaired lists entered batch"), raising=False)
    cache.reset()
    assert torch.all(tail == 7) and tail._version == version
    assert cache.position == cache._seen_tokens == 0
    assert all(not tensor.count_nonzero() for tensor in _reset_tensors(cache))


# --- axis 23: capacity ------------------------------------------------------

@pytest.mark.parametrize("first", [8191, 8192])
def test_fill_up_to_capacity_is_accepted(first):
    c = _cache()
    c.update_position(first)
    assert c.position == first
    assert c.get_seq_length() == first
    assert c.get_max_length() == MAX_SEQ


def test_one_token_past_capacity_raises_and_leaves_position_intact():
    c = _cache()
    c.update_position(MAX_SEQ)
    with pytest.raises(ValueError, match="position overflow"):
        c.update_position(1)
    # the rejected call must not have advanced anything (no wrap, no partial)
    assert c.position == MAX_SEQ
    assert c.get_seq_length() == MAX_SEQ


def test_single_oversized_advance_raises():
    c = _cache()
    with pytest.raises(ValueError, match="position overflow"):
        c.update_position(MAX_SEQ + 1)
    assert c.position == 0


@pytest.mark.parametrize("bad", [-1, True, 1.0, "16", None])
def test_update_position_rejects_bad_values(bad):
    c = _cache()
    with pytest.raises(ValueError):
        c.update_position(bad)
    assert c.position == 0


# --- axis 22: reset semantics ----------------------------------------------

@pytest.mark.parametrize("pos", [0, 1, 4096, MAX_SEQ])
def test_reset_to_position_accepts_the_closed_range(pos):
    c = _cache()
    c.reset_to_position(pos)
    assert c.position == pos
    assert c.get_seq_length() == pos


@pytest.mark.parametrize("bad", [MAX_SEQ + 1, -1, True, 1.0, "0", None])
def test_reset_to_position_rejects_out_of_range_and_non_int(bad):
    c = _cache()
    c.update_position(64)
    with pytest.raises(ValueError):
        c.reset_to_position(bad)
    assert c.position == 64          # rejected call changes nothing


def test_reset_zeroes_data_but_reset_to_position_preserves_it():
    """The whole point of reset_to_position: keep the prefix K/V alive."""
    c = _cache()
    c.update_position(128)
    c.k_caches[0].fill_(1.0)
    c.v_caches[0].fill_(2.0)

    c.reset_to_position(64)
    assert c.position == 64
    assert torch.all(c.k_caches[0] == 1.0), "reset_to_position must not clear K"
    assert torch.all(c.v_caches[0] == 2.0), "reset_to_position must not clear V"

    c.reset()
    assert c.position == 0
    assert torch.all(c.k_caches[0] == 0.0), "reset() must clear K"
    assert torch.all(c.v_caches[0] == 0.0), "reset() must clear V"


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("inference_owner", [False, True])
def test_banked_reset_preserves_layer_addresses_and_clears_each_owner_once(
        monkeypatch, batch_size, inference_owner):
    with torch.inference_mode(inference_owner):
        cache = RPUCache(5, batch_size, 32, 2, 64, device="cpu", _layer_bank_size=3)
    values = cache.k_caches + cache.v_caches
    addresses = [value.data_ptr() for value in values]
    with torch.inference_mode():
        for index, value in enumerate(values):
            value.fill_(index + 1)
    calls, zero = [], torch.Tensor.zero_

    def tracked(value):
        calls.append(value)
        return zero(value)

    monkeypatch.setattr(torch.Tensor, "zero_", tracked)
    cache.update_position(17)
    with torch.inference_mode():
        cache.reset()
    assert len(calls) == 4  # Two owning banks per K/V kind, including the tail.
    assert [value.data_ptr() for value in values] == addresses
    assert all(not value.count_nonzero() for value in values)
    assert cache.position == cache._seen_tokens == 0


@pytest.mark.parametrize("mutation", ["replace", "reshape", "partial", "reorder", "append", "remove", "bank"])
def test_banked_reset_falls_back_to_current_views_without_clearing_retired_rows(monkeypatch, mutation):
    cache = RPUCache(3, 1, 32, 2, 64, device="cpu", _layer_bank_size=3)
    retired = cache.k_caches[0]
    for value in cache.k_caches + cache.v_caches:
        value.fill_(7)
    if mutation == "replace":
        cache.k_caches[0] = retired.clone()
    elif mutation == "reshape":
        retired.squeeze_(0)
    elif mutation == "partial":
        cache.k_caches[0] = retired.flatten()[256:]
    elif mutation == "reorder":
        cache.k_caches.reverse()
    elif mutation == "append":
        cache.k_caches.append(retired.clone())
        cache.v_caches.append(cache.v_caches[0].clone())
    elif mutation == "remove":
        cache.k_caches.pop(0)
        cache.v_caches.pop(0)
    else:
        # The bank's own geometry can drift independently of its layer views.
        cache._reset_groups[0][0][0][0].transpose_(0, 1)
    calls, zero = [], torch.Tensor.zero_

    def tracked(value):
        calls.append(value)
        return zero(value)

    monkeypatch.setattr(torch.Tensor, "zero_", tracked)
    cache.reset()
    assert all(not value.count_nonzero() for value in cache.k_caches + cache.v_caches)
    assert all(any(value is current for current in cache.k_caches) for value in calls[:len(cache.k_caches)])
    if mutation in {"replace", "remove"}:
        assert torch.all(retired == 7)
    elif mutation == "partial":
        assert torch.all(retired.flatten()[:256] == 7)
    if mutation not in {"append", "remove"}:
        assert len(calls) == 4  # Current K views, still-intact V bank.


# --- axis 24: multi-round growth -------------------------------------------

def test_three_rounds_of_prefix_reuse_track_position():
    """Pi0.5 pattern: one prefill, then N rounds of decode rewound to P."""
    prefix_len, decode_steps, rounds = 100, 5, 3
    c = _cache()
    c.update_position(prefix_len)
    c.k_caches[0].fill_(3.0)                       # stand-in for prefix K

    for _ in range(rounds):
        for step in range(decode_steps):
            c.update_position(1)
            assert c.position == prefix_len + step + 1
        c.reset_to_position(prefix_len)
        assert c.position == prefix_len
        assert torch.all(c.k_caches[0] == 3.0), "prefix K lost across rounds"

    assert c.position == prefix_len


def test_rounds_still_respect_capacity():
    """Rewinding must not let repeated rounds sneak past max_seq_len."""
    c = _cache(max_seq_len=256)
    c.update_position(200)
    for _ in range(3):
        c.update_position(56)                      # exactly fills 256
        assert c.position == 256
        with pytest.raises(ValueError, match="position overflow"):
            c.update_position(1)
        c.reset_to_position(200)
    assert c.position == 200


def test_multi_layer_caches_share_one_position():
    c = _cache(num_layers=3)
    c.update_position(48)
    assert c.get_seq_length(0) == c.get_seq_length(2) == 48
    assert len(c.k_caches) == len(c.v_caches) == 3
