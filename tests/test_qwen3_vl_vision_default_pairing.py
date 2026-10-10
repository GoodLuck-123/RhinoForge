"""Controlled W8 pairs only consecutive complete N256 image grids."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.qwen3_vl import vision
from rpu_backend.runtime.execution_planner import PlannerRejectError
from test_qwen3_vl_merger_lifecycle import cpu_transfers  # noqa: F401


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("route", ["borrow", "public", "multi", "mixed", "adaptive", "eager", "one_ds",
    "eager_public", "eager_multi", "eager_mixed", "eager_cpu_public",
    "adaptive_public", "adaptive_multi", "adaptive_mixed", "adaptive_partial", "adaptive_one_ds",
    "adaptive_chunked_w8", "adaptive_chunked_fp16"])
def test_private_merged_consumer_preserves_native_slot_and_public_ownership(
        cpu_transfers, monkeypatch, inference, route):
    """Actual shared forward; device kernels/capture are explicit CPU doubles."""
    import ctypes

    class Visual(SimpleNamespace):
        def fast_pos_embed_interpolate(self, grid):
            return torch.zeros(int(grid.prod(-1).sum()), 2, dtype=torch.float16)

    active, events, slots, clone_sources, signatures = [False], [], {}, [], []
    raw_slots, raw_clones, public_raw = {}, [], []
    adaptive = route.startswith("adaptive")
    eager = route.startswith("eager")
    merged_only = route in ("adaptive", "adaptive_chunked_w8", "adaptive_chunked_fp16")
    public = route in ("public", "adaptive_public", "eager_public", "eager_cpu_public")
    @contextmanager
    def capture(signature):
        assert not active[0]
        active[0] = True
        signatures.append(signature)
        try:
            yield
        finally:
            active[0] = False
            events.append("capture_end")

    ds_count = 1 if route in ("one_ds", "adaptive_one_ds") else 3
    visual = Visual(
        _rpu_vision_handle=11, _rpu_vision_spatial_merge_size=2,
        _rpu_vision_num_layers=1, _rpu_vision_hidden_size=2,
        _rpu_vision_kv_cache=SimpleNamespace(k_caches=[None], v_caches=[None],
            max_seq_len=2048, reset_to_position=lambda position: None),
        _rpu_vision_graph_cache=SimpleNamespace(capture=capture),
        _rpu_vision_deepstack_hash=1, _rpu_vision_max_hw=40,
        _rpu_vision_host_fp32_patch=False, _rpu_vision_patch_embed_w=torch.eye(2).half(),
        _rpu_vision_patch_embed_b=None, _rpu_vision_position_idx_keepalive=torch.zeros(2048, 2).short(),
        _rpu_vision_deepstack_indexes=list(range(ds_count)),
        _rpu_vision_fused_merger=not eager,
        _rpu_vision_fused_merger_w8a16=adaptive and route != "adaptive_chunked_fp16",
        _rpu_vision_chunked_merger_w8a16=route == "adaptive_chunked_w8",
        _rpu_vision_chunked_merger_fp16=route == "adaptive_chunked_fp16",
        _rpu_vision_merger_on_device=route != "eager_cpu_public",
        _rpu_vision_batch_enabled=route not in ("multi", "adaptive_multi", "eager_multi"), _rpu_vision_batch_cap=3,
        merger=lambda values: values[::4] + 10,
        deepstack_merger_list=[lambda values: values[::4] + 10 for _ in range(ds_count)],
    )
    grid = torch.tensor([[1, 14, 14]] * 3)
    if eager and route not in ("eager_multi", "eager_mixed"):
        grid = grid[:1]
    if route == "adaptive_partial":
        grid = grid[:1]
    if route in ("adaptive_chunked_w8", "adaptive_chunked_fp16"):
        grid = torch.tensor([[1, 30, 40]])
    if route == "eager_mixed":
        grid[:2] = torch.tensor([1, 16, 16])
    elif route in ("mixed", "adaptive_mixed"):
        grid[2] = torch.tensor([1, 16, 16])
    rows = grid.prod(-1).tolist()
    pixels = torch.cat([torch.full((n, 2), float(i + 1)).half() for i, n in enumerate(rows)])
    original_pixels = pixels.clone()
    pixel_version = pixels._version
    aliases = (ctypes.c_uint16 * pixels.numel()).from_address(pixels.data_ptr())
    def chunk(patches):
        return (608 if route in ("adaptive_chunked_w8", "adaptive_chunked_fp16")
                else (patches + 1) // 2 if route == "adaptive_partial" else patches)
    def plan(handle, patches, batch, *args, **kwargs):
        return SimpleNamespace(selected=SimpleNamespace(execution_len=patches,
            stage_tuple=SimpleNamespace(compute_chunk=chunk(patches), physical_descriptor=(patches, batch),
                physical_metadata=(("manifest_state", 1), ("raw_attention_site_count", 1),
                                   ("capability_attention_ddr_site_count", 0)))),
            graph_key_words=lambda: (patches, batch), graph_mode="GRAPH_COMPOSITE_CHILD",
            selection_scope="TEST", physical_plan_digest="test", plan_digest="test")
    monkeypatch.setattr(vision, "_plan_qwen3_vl_vision_execution", plan)
    monkeypatch.setattr(vision, "_deepstack_output_cls", lambda: SimpleNamespace)
    monkeypatch.setattr(vision.rpu_backend.graph, "GraphSignature", lambda **kw: tuple((k, str(v)) for k, v in kw.items()))
    current = [None]
    def forward(handle, values, keys, vals, patches, batch, descriptor):
        assert active[0] and descriptor == [patches, batch]
        current[0] = patches
        if patches not in slots:
            # Native stable output owners are installed ordinary tensors.
            with torch.inference_mode(False):
                slots[patches] = [torch.empty((patches // 4, 2), dtype=torch.float16)
                                  for _ in range(1 + ds_count)]
                raw_slots[patches] = torch.empty((1, patches, 2), dtype=torch.float16)
        for ordinal, tensor in enumerate(slots[patches]):
            tensor.copy_(values[0, ::4] + (ordinal + 1) * 10)
        raw_slots[patches].copy_(values)
        return raw_slots[patches]
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_forward", forward, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_get_resolved_chunk_size", lambda h: chunk(current[0]), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_pop_merged", lambda h: slots[current[0]], raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_pop_deepstack_snapshots",
                        lambda h: [torch.ones(current[0], 2).half() for _ in range(ds_count)], raising=False)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: events.append("reset"), raising=False)
    monkeypatch.setattr(vision, "_merger_forward_on_device",
                        lambda merger, values, **kw: (values[::4] + 10).to(
                            torch.float16 if kw.get("keep_fp16") else torch.float32))
    original_clone = torch.Tensor.clone
    def track_clone(value, *args, **kwargs):
        if any(value is slot for group in slots.values() for slot in group):
            clone_sources.append(value.data_ptr())
        if any(value.data_ptr() == raw.data_ptr() and value.numel() == raw.numel()
               for raw in raw_slots.values()):
            raw_clones.append(value.data_ptr())
        return original_clone(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "clone", track_clone)
    raw_cats = []
    original_cat = torch.cat
    def track_cat(values, *args, **kwargs):
        if eager and values[0].dtype == torch.float16 and values[0].ndim == 2:
            raw_cats.append(len(values))
        return original_cat(values, *args, **kwargs)
    monkeypatch.setattr(torch, "cat", track_cat)
    consumed = []
    def consume(merged, ds):
        assert not active[0] and events[-1] == "reset"
        assert visual._rpu_last_execution_plan["dry_forward_agreement"]
        sources = [merged, *ds]
        if adaptive:
            assert all(value.device.type == "cpu" for value in sources)
            assert all(value.dtype == torch.float16 for value in sources)
        native = slots[current[0]]
        if route == "borrow":
            assert all(a is b for a, b in zip(sources, native))
        else:
            assert all(a.data_ptr() != b.data_ptr() for a, b in zip(sources, native))
        # CPU stand-in for synchronous native composition into a fresh owner.
        output = torch.stack(sources)
        assert all(output.data_ptr() != value.data_ptr() for value in sources)
        consumed.append([value.data_ptr() for value in sources])
        events.append("consume")
        return output

    retained = []
    for delta in (0, 100, 0):
        bits = (original_pixels + delta).view(torch.uint16).reshape(-1).tolist()
        for index, bit in enumerate(bits):
            aliases[index] = bit
        with torch.inference_mode(inference):
            result = vision._rpu_vision_forward(visual, pixels, grid,
                _rpu_keep_merged_fp16=adaptive,
                **({} if public else {"_rpu_merged_consumer": consume}))
        values = torch.stack([result.pooler_output, *result.deepstack_features]) if public else result
        if route in ("adaptive_public", "eager_public", "eager_cpu_public"):
            public_raw.append((result.last_hidden_state, original_clone(result.last_hidden_state)))
            assert all(torch.equal(value, saved) for value, saved in public_raw)
            assert all(value.data_ptr() != raw.data_ptr()
                       for value, _ in public_raw for raw in raw_slots.values())
            if eager:
                assert torch.equal(result.last_hidden_state, original_pixels + delta)
                assert torch.equal(result.pooler_output, (original_pixels + delta)[::4].float() + 10)
        retained.append((values, original_clone(values)))
        assert all(torch.equal(value, saved) for value, saved in retained)
    assert pixels._version == pixel_version
    assert torch.equal(retained[0][0], retained[-1][0])
    assert not torch.equal(retained[0][0], retained[1][0])
    if eager:
        group_count = 3 if route == "eager_multi" else 2 if route == "eager_mixed" else 1
        assert len(raw_clones) == 3 * group_count
        assert raw_cats == ([group_count] * 3 if group_count > 1 else [])
    if adaptive:
        assert len(raw_clones) == (0 if merged_only else
                                 9 if route == "adaptive_multi" else
                                 6 if route == "adaptive_mixed" else 3)
    if route in ("borrow", "adaptive"):
        if route == "borrow":
            assert clone_sources == [] and len(set(map(tuple, consumed))) == 1
        assert len(signatures) == 3 and signatures[0] == signatures[1] == signatures[2]
        # A consumer exception propagates after Graph exit; the next request
        # must consume freshly written slots rather than a retained result.
        def fail(*args):
            raise RuntimeError("consumer failed")
        with pytest.raises(RuntimeError, match="consumer failed"):
            vision._rpu_vision_forward(visual, pixels, grid, _rpu_merged_consumer=fail,
                                      _rpu_keep_merged_fp16=adaptive)
        assert not active[0]
        assert torch.equal(vision._rpu_vision_forward(visual, pixels, grid,
                          _rpu_merged_consumer=consume,
                          _rpu_keep_merged_fp16=adaptive), retained[0][0])
        for fault in (lambda x: x[:3], lambda x: [x[0].float(), *x[1:]],
                      lambda x: [x[0][:-1], *x[1:]],
                      lambda x: [x[0].T.contiguous().T, *x[1:]]):
            monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_pop_merged", lambda h: fault(slots[current[0]]))
            before = len(consumed)
            with pytest.raises(RuntimeError, match="merged count|slot geometry"):
                vision._rpu_vision_forward(visual, pixels, grid, _rpu_merged_consumer=consume,
                                          _rpu_keep_merged_fp16=adaptive)
            assert len(consumed) == before and not active[0]
    elif route in ("public", "multi", "mixed", "one_ds"):
        assert clone_sources
    with pytest.raises(TypeError, match="callable"):
        vision._rpu_vision_forward(visual, pixels, grid, _rpu_merged_consumer=True)
    with pytest.raises(ValueError, match="prefix-scatter"):
        vision._rpu_vision_forward(visual, pixels, grid,
            _rpu_merged_consumer=consume, _rpu_prefix_scatter_consume_only=True)


@pytest.mark.parametrize("grids,cap,capacity,expected", [
    ([[1, 16, 16]] * 2, 3, 2048, [(0, 2)]),
    ([[1, 16, 16]] * 3, 3, 2048, [(0, 2), (2, 1)]),
    ([[1, 16, 16]] * 4, 3, 2048, [(0, 2), (2, 2)]),
    ([[1, 16, 16], [1, 8, 32]], 3, 2048, [(0, 1), (1, 1)]),
    ([[1, 16, 16], [1, 8, 32], [1, 16, 16]], 3, 2048, [(0, 1), (1, 1), (2, 1)]),
    ([[1, 16, 32]] * 2, 3, 2048, [(0, 1), (1, 1)]),
    ([[1, 24, 32]] * 2, 3, 2048, [(0, 1), (1, 1)]),
    ([[2, 8, 16]] * 2, 3, 2048, [(0, 1), (1, 1)]),
    ([[1, 16, 16]] * 2, 1, 2048, [(0, 1), (1, 1)]),
    ([[1, 16, 16]] * 2, 3, 256, [(0, 1), (1, 1)]),
])
def test_actual_grouping_bounds(grids, cap, capacity, expected):
    grid = torch.tensor(grids)
    assert vision._vision_image_groups(
        grid, grid.prod(-1).tolist(), batch_cap=cap, cache_capacity=capacity,
        exact_chunk=None, fused_merger=False, default_pair_n256=True) == expected


def test_legacy_explicit_batch_keeps_three_images():
    grid = torch.tensor([[1, 16, 16]] * 3)
    assert vision._vision_image_groups(
        grid, [256] * 3, batch_cap=3, cache_capacity=2048,
        exact_chunk=None, fused_merger=True) == [(0, 3)]


@pytest.mark.parametrize("default_pair,fp16,compact,rows,chunk,expected_groups", [
    (True, False, False, [256] * 3, "auto", [(512, 2), (256, 1)]),
    (False, False, False, [196] * 3, "auto", [(588, 3)]),
    (False, False, False, [196, 196, 256], "auto", [(392, 2), (256, 1)]),
    (True, False, False, [196] * 3, "auto", [(196, 1)] * 3),
    (False, False, False, [256] * 3, "auto", [(256, 1)] * 3),
    (False, True, False, [1200] * 3, "auto", [(1200, 1)] * 3),
    (False, True, False, [1200] * 3, 608, [(1200, 1)] * 3),
    (False, True, False, [1200] * 3, 144, [(1200, 1)] * 3),
    (False, True, False, [256, 1200, 256], "auto", [(256, 1), (1200, 1), (256, 1)]),
    (False, True, True, [1200] * 3, "auto", [(1200, 1)] * 3),
    (False, True, True, [1200] * 3, 608, [(1200, 1)] * 3),
    (False, True, True, [256, 1200, 256], "auto", [(256, 1), (1200, 1), (256, 1)]),
    (False, False, True, [1200], "auto", [(1200, 1)]),
    (False, False, True, [1200] * 3, "auto", [(1200, 1)] * 3),
    (False, False, True, [1200] * 3, 608, [(1200, 1)] * 3),
    (False, False, True, [1200] * 3, 144, [(1200, 1)] * 3),
    (False, False, True, [256, 1200, 256], "auto", [(256, 1), (1200, 1), (256, 1)]),
    (True, False, True, [256] * 3, "auto", [(512, 2), (256, 1)]),
])
def test_actual_forward_keeps_image_order_positions_and_retained_outputs(
        cpu_transfers, monkeypatch, default_pair, fp16, compact, rows, chunk, expected_groups):
    class Visual(SimpleNamespace):
        def fast_pos_embed_interpolate(self, grid):
            return torch.zeros(int(grid.prod(-1).sum()), 2, dtype=torch.float16)
    legacy_merger = 196 in rows
    signatures, calls, positions = [], [], []
    @contextmanager
    def capture(signature):
        signatures.append(signature)
        yield
    visual = Visual(
        _rpu_vision_handle=11, _rpu_vision_spatial_merge_size=2,
        _rpu_vision_linear_acc32=fp16,
        _rpu_vision_num_layers=1, _rpu_vision_hidden_size=2,
        _rpu_vision_kv_cache=SimpleNamespace(k_caches=[None], v_caches=[None],
            max_seq_len=2048, reset_to_position=lambda position: None),
        _rpu_vision_graph_cache=SimpleNamespace(capture=capture),
        _rpu_vision_deepstack_hash=1, _rpu_vision_max_hw=40,
        _rpu_vision_host_fp32_patch=False, _rpu_vision_patch_embed_w=torch.eye(2).half(),
        _rpu_vision_patch_embed_b=None, _rpu_vision_position_idx_keepalive=torch.zeros(2048, 2).short(),
        _rpu_vision_deepstack_indexes=[0], _rpu_vision_fused_merger=True,
        _rpu_vision_fused_merger_w8a16=not fp16 and not legacy_merger, _rpu_vision_merger_on_device=True,
        _rpu_vision_chunked_merger_fp16=fp16, _rpu_vision_execution_chunk_size=chunk,
        _rpu_vision_chunked_merger_w8a16=not fp16,
        _rpu_vision_compact_encoder_fp16=compact and fp16,
        _rpu_vision_compact_encoder_w8a16=compact and not fp16,
        _rpu_vision_batch_enabled=fp16 or (196 in rows and not default_pair), _rpu_vision_batch_cap=3,
        _rpu_vision_default_pair_n256=default_pair,
        merger=object(), deepstack_merger_list=[object()],
    )
    def resolved(patches):
        return ((1200 if compact else 608) if patches == 1200 else (patches + 15) // 16 * 16) if chunk == "auto" else chunk
    def plan(handle, patches, batch, *args, **kwargs):
        descriptor = (patches, batch)
        return SimpleNamespace(selected=SimpleNamespace(execution_len=patches,
            stage_tuple=SimpleNamespace(compute_chunk=resolved(patches), physical_descriptor=descriptor,
                physical_metadata=(("manifest_state", 1), ("raw_attention_site_count", 1),
                                   ("capability_attention_ddr_site_count", 0)))),
            graph_key_words=lambda: descriptor, graph_mode="GRAPH_COMPOSITE_CHILD",
            selection_scope="TEST", physical_plan_digest="test", plan_digest="test")
    monkeypatch.setattr(vision, "_plan_qwen3_vl_vision_execution", plan)
    monkeypatch.setattr(vision, "_deepstack_output_cls", lambda: SimpleNamespace)
    monkeypatch.setattr(vision.rpu_backend.graph, "GraphSignature", lambda **kw: SimpleNamespace(**kw))
    slots, current = {}, {}
    def forward(handle, values, k, v, patches, batch, descriptor):
        calls.append((patches, batch))
        positions.append(visual._rpu_vision_position_idx_keepalive[:patches].clone())
        assert descriptor == [patches, batch]
        current["patches"] = patches
        raw, pooler, ds = slots.setdefault(patches, (
            torch.empty(1, patches, 2).half(), torch.empty(patches // 4, 2).half(),
            torch.empty(patches // 4, 2).half()))
        raw.copy_(values)
        pooler.copy_(values[0, ::4] + 10)
        ds.copy_(values[0, ::4] + 20)
        return raw
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_forward", forward, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_get_resolved_chunk_size",
                        lambda handle: resolved(current["patches"]), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_pop_merged",
                        lambda handle: slots[current["patches"]][1:], raising=False)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: None, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_pop_deepstack_snapshots",
                        lambda handle: [slots[current["patches"]][0].squeeze(0)], raising=False)
    eager_calls = []
    def eager(merger, values, *, keep_fp16=False, linear_acc32=False):
        assert linear_acc32 is fp16
        eager_calls.append(len(values))
        return values[::4].to(torch.float16 if keep_fp16 else torch.float32) + (
            10 if merger is visual.merger else 20)
    monkeypatch.setattr(vision, "_merger_forward_on_device", eager)
    pixels = torch.cat([torch.full((n, 2), value) for n, value in zip(rows, (1, 2, 3))]).half()
    grid = torch.tensor([[1, 30, 40] if n == 1200 else [1, 14, 14] if n == 196 else [1, 16, 16] for n in rows])
    output = vision._rpu_vision_forward(visual, pixels, grid)
    assert calls == expected_groups
    fused_fp16 = fp16 and rows == [1200] * 3 and chunk in ("auto", 608)
    assert eager_calls == ([] if not fp16 or fused_fp16 else [sum(rows)] * 2)
    if fp16:
        assert visual._rpu_last_execution_plan["dispatch_merger_routes"] == (
            ("chunked_fused_fp16" if fused_fp16 else "eager_fp16"),) * 3
        assert [sig.dyn_dims[2] for sig in signatures] == [int(fused_fp16)] * 3
    elif not legacy_merger:
        assert visual._rpu_last_execution_plan["dispatch_merger_routes"] == tuple(
            "chunked_fused_w8a16" if (n == 1200 and compact and chunk == "auto")
            or resolved(n) < n else "fused_w8a16" for n, _ in expected_groups)
    assert torch.equal(output.last_hidden_state, pixels[::4] + 10 if legacy_merger else pixels)
    assert torch.equal(output.pooler_output, pixels[::4].float() + 10)
    assert torch.equal(output.deepstack_features[0], pixels[::4].float() + 20)
    for position, (packed_rows, count) in zip(positions, expected_groups):
        per_image = packed_rows // count
        for image in range(1, count):
            assert torch.equal(position[:per_image],
                               position[image * per_image:(image + 1) * per_image])
    if visual._rpu_vision_batch_enabled or any(count > 1 for _, count in expected_groups):
        assert [sig.dyn_dims[3] for sig in signatures] == [count for _, count in expected_groups]
    if len(set(expected_groups)) > 1:
        assert signatures[0] != signatures[1]
    # A later call overwrites every internal slot; the previous public values survive.
    vision._rpu_vision_forward(visual, pixels + 100, grid)
    assert torch.equal(output.last_hidden_state, pixels[::4] + 10 if legacy_merger else pixels)
    assert torch.equal(output.pooler_output, pixels[::4].float() + 10)
    assert torch.equal(output.deepstack_features[0], pixels[::4].float() + 20)


@pytest.mark.parametrize("grids,cap,capacity,admit,expected,probes", [
    ([[1,14,14]]*3, 3, 2048, True, [(0,3)], [(0,2),(0,3)]),
    ([[1,14,14]]*3, 3, 2048, False, [(0,1),(1,1),(2,1)], [(0,2),(1,2)]),
    ([[1,14,14]]*3, 3, 400, True, [(0,2),(2,1)], [(0,2)]),
    ([[1,14,14]]*3, 1, 2048, True, [(0,1),(1,1),(2,1)], []),
    ([[1,14,14],[1,7,28]], 3, 2048, True, [(0,1),(1,1)], []),
    ([[1,12,12]]*3, 3, 2048, True, [(0,1),(1,1),(2,1)], []),
    ([[1,16,16]]*3, 3, 2048, True, [(0,3)], []),
])
def test_unaligned_packing_requires_native_admission_and_whole_grids(
        grids, cap, capacity, admit, expected, probes):
    grid = torch.tensor(grids)
    seen = []
    def native(first, count):
        seen.append((first, count))
        return admit
    assert vision._vision_image_groups(
        grid, grid.prod(-1).tolist(), batch_cap=cap, cache_capacity=capacity,
        exact_chunk=None, fused_merger=True, raw_group_supported=native) == expected
    assert seen == probes


@pytest.mark.parametrize("fault", [None, "old_ddr", "incomplete", "padding", "capacity", "native_error", "stale"])
def test_unaligned_native_plan_fallback_and_errors(fault):
    meta = dict(manifest_state=1, raw_attention_site_count=1,
                capability_attention_ddr_site_count=0)
    if fault == "old_ddr":
        meta.update(raw_attention_site_count=0, capability_attention_ddr_site_count=1)
    if fault == "incomplete":
        meta["manifest_state"] = 0
    result = SimpleNamespace(selected=SimpleNamespace(
        execution_len=592 if fault == "padding" else 588,
        stage_tuple=SimpleNamespace(physical_metadata=tuple(meta.items()))))
    def resolve():
        if fault == "capacity":
            raise PlannerRejectError("NO_FEASIBLE")
        if fault == "native_error":
            raise RuntimeError("native handle was retired")
        if fault == "stale":
            raise PlannerRejectError("STALE_DESCRIPTOR")
        return result
    if fault in ("native_error", "stale"):
        with pytest.raises(RuntimeError):
            vision._vision_raw_group_plan(resolve, 588)
    else:
        selected = vision._vision_raw_group_plan(resolve, 588)
        assert selected is (result if fault is None else None)
