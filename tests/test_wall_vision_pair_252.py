"""Semantic checks for the exact paired route with two or three cameras."""

import ast
import sys
from contextlib import nullcontext
from pathlib import Path
from types import MethodType, SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F


SOURCE = Path(__file__).resolve().parents[1] / "python/rpu_backend/adapters/wall_oss/vision.py"


def helpers():
    tree = ast.parse(SOURCE.read_text())
    functions = {"_get_window_index", "_block_diag_mask", "_compute_vision_position_idx_cpu",
                 "_pair_252_layout", "_ragged_mask_plan_identity"}
    methods = {"forward", "_forward", "_forward_parts", "_forward_pair_252", "_plan_native_execution",
               "_patch_embed_pair_252", "_patch_embed", "_consume_patch_embed_input",
               "_patch_perm_ids", "_prepare_patch_pixels", "_index_rpu",
               "_refresh_position_keepalive",
               "_record_native_execution", "_validate_execution_chunks", "_native_chunk_size"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in functions]
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WallOssVision")
    nodes += [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    ns = dict(torch=torch, F=F, Sequence=list, GRAPH_COMPOSITE_CHILD="COMPOSITE_CHILD",
              _LAYER_GROUP_TOTAL_SEQ=1728)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


def test_pair_layout_preserves_real_pixels_positions_and_merger_order():
    h = helpers()
    grid = torch.tensor([[1, 18, 18]] * 2)
    pos, win, rev, window, full = h["_pair_252_layout"](grid)
    assert torch.equal(win.sort().values, torch.arange(168))
    pixels = torch.arange(648).reshape(2, 81, 4)
    padded = torch.full((2, 84, 4), -1)
    padded[:, :81] = pixels
    reordered = padded.reshape(168, 4)[win]
    assert torch.equal(reordered.sum(1)[rev], pixels.reshape(162, 4).sum(1))
    expected_pos = h["_compute_vision_position_idx_cpu"](grid[:1], 2)
    for image in range(2):
        actual = pos[image*336:(image+1)*336].reshape(84, 4, 2)
        assert torch.equal(actual[rev[image*81:(image+1)*81] - image*84].reshape(324, 2),
                           expected_pos)
    for mask in (window, full):
        assert torch.isneginf(mask[:324, 324:]).all()
        assert torch.isneginf(mask[324:, :324]).all()
        assert torch.isfinite(mask).any(1).all()
    with pytest.raises(ValueError, match="18x18"):
        h["_pair_252_layout"](torch.tensor([[1, 12, 27]] * 2))


@pytest.mark.parametrize("full_attention", [False, True])
def test_pair_attention_matches_unpadded_and_isolates_cameras_and_dummy_rows(full_attention):
    h = helpers()
    _, _, _, window, full = h["_pair_252_layout"](torch.tensor([[1, 18, 18]] * 2))
    mask = (full if full_attention else window).double()
    rng = torch.Generator().manual_seed(31)
    q, k, v = [torch.randn(2, 2, 336, 16, dtype=torch.float64, generator=rng) for _ in range(3)]
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    reference = F.scaled_dot_product_attention(q[:, :, :324], k[:, :, :324], v[:, :, :324],
                                             attn_mask=mask[:324, :324])
    torch.testing.assert_close(out[:, :, :324], reference, rtol=1e-12, atol=1e-12)
    k[:, :, 324:] *= 100
    v[:, :, 324:] += 1000
    changed_pads = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.equal(out[:, :, :324], changed_pads[:, :, :324])
    v[1] += 1
    changed_camera = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    assert torch.equal(out[0, :, :324], changed_camera[0, :, :324])
    assert not torch.equal(out[1, :, :324], changed_camera[1, :, :324])


@pytest.mark.parametrize("cameras", [2, 3])
def test_public_forward_uses_pair_and_keeps_debug_off_and_other_shapes(monkeypatch, cameras):
    h = helpers()
    # The extracted adapter forward imports its lifetime guard lazily. Execute
    # that real guard without importing the device backend on this CPU test.
    retirement = ast.parse((SOURCE.parent / "_retirement.py").read_text())
    guard = next(n for n in retirement.body
                 if isinstance(n, ast.FunctionDef) and n.name == "_require_open")
    guard_ns = {"_execution": NS(_require_execution_process_safe=lambda: None)}
    exec(compile(ast.Module(body=[guard], type_ignores=[]), str(SOURCE.parent / "_retirement.py"), "exec"), guard_ns)
    monkeypatch.setitem(sys.modules, "rpu_backend.adapters.wall_oss._retirement",
                        NS(_require_open=guard_ns["_require_open"]))
    monkeypatch.setattr(torch, "rpu", NS(get_debug_export=lambda: False), raising=False)
    calls = []
    owner = NS(_pair_252=True, _batch_vision=True, _batch_max_seq=768,
               _layer_group=0, _weight_outer=False, _window=True, _fused_assemble=True,
               _device_patch_embed=False, _patch_w_rpu=None)
    owner._forward = MethodType(h["_forward"], owner)
    owner._validate_execution_chunks = lambda chunks: calls.append(tuple(chunks))
    owner._native_chunk_size = lambda grid: 16*((int(grid.prod(-1).sum())+15)//16)
    def encode(pixels, grid, **kwargs):
        calls.append((len(grid), int(pixels[0, 0]), len(pixels)))
        return pixels.reshape(-1, 4, 1).sum(1)
    owner._forward_pair_252 = owner._forward_one = owner._forward_group = encode
    grid = torch.tensor([[1, 18, 18]] * cameras)
    pixels = torch.arange(324 * cameras).reshape(-1, 1).float()
    result = h["forward"](owner, pixels, grid)
    expected = [(672,), (2, 0, 648)] if cameras == 2 else [(672, 336), (2, 0, 648), (1, 648, 324)]
    assert calls == expected
    assert torch.equal(result, pixels.reshape(-1, 4, 1).sum(1))
    assert owner._pending_rev is None and owner._last_forward_plans == []
    for off, debug in [(False, True), (True, False)]:
        calls.clear(); owner._pair_252 = not off
        monkeypatch.setattr(torch.rpu, "get_debug_export", lambda: debug)
        h["forward"](owner, pixels, grid)
        assert [x[0] for x in calls[1:]] == ([1] * cameras if off else [2] + ([1] if cameras == 3 else []))
    owner._pair_252 = True
    calls.clear()
    h["forward"](owner, torch.arange(768).reshape(-1, 1).float(), torch.tensor([[1,16,16]]*3))
    assert calls[1][0] == 3
    owner._batch_max_seq = 640
    with pytest.raises(ValueError, match="max_seq"):
        h["forward"](owner, pixels, grid)
    # An explicit C672 is legal only when no third C336 invocation is needed.
    owner._batch_max_seq = 768
    owner._rpu_execution = {"vision": {"chunk_size": 672}}
    owner._validate_execution_chunks = lambda chunks: h["_validate_execution_chunks"](owner, chunks)
    calls.clear()
    if cameras == 2:
        assert torch.equal(h["forward"](owner, pixels, grid), result)
        assert calls == [(2, 0, 648)]
    else:
        with pytest.raises(ValueError, match="native=\\(672, 336\\)"):
            h["forward"](owner, pixels, grid)
        assert calls == []


@pytest.mark.parametrize("inference_mode", [False, True])
def test_pair_native_boundary_refreshes_frames_and_preserves_returned_storage(monkeypatch, inference_mode):
    h = helpers(); trace = []; signatures = []; copied_positions = []
    monkeypatch.setitem(sys.modules, "rpu_backend.adapters.wall_oss._retirement",
                        NS(_require_open=lambda _: None))
    monkeypatch.setattr(torch, "rpu", NS(get_debug_export=lambda: False), raising=False)
    original_to = torch.Tensor.to
    def cpu_device_boundary(t, *args, **kwargs):
        if t.dtype == torch.int16 and tuple(t.shape) == (672, 2):
            raise AssertionError("pair positions must copy directly from their CPU source")
        if args and args[0] == "rpu":
            return t
        return original_to(t, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", cpu_device_boundary)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: trace.append("reset"), raising=False)
    def native(handle, embed, k, v, seq, window, batch, cu, descriptor, valid, full,
               *, position_published=False):
        assert (seq, batch, valid) == (672, 2, 324)
        assert position_published is True
        assert descriptor == [99]
        assert torch.isneginf(window[:324,324:]).all()
        assert torch.isneginf(full[:324,324:]).all()
        assert torch.equal(owner._keepalive[:672], copied_positions[-1])
        trace.append("native")
        # CPU transport/consumer double: use the last actual copy_ payload,
        # rather than reading the caller's table again inside the consumer.
        positions = copied_positions[-1].reshape(168, 4, 2).sum((1, 2)).reshape(168, 1)
        return embed.reshape(168,4,-1).sum(1) + positions
    monkeypatch.setattr(torch.ops.rpu, "qwen25vl_vision_forward", native, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "gather_embedding", lambda out, ids: out[ids.long()], raising=False)
    h["rpu_backend"] = NS(graph=NS(GraphSignature=lambda **kwargs: kwargs))
    owner = NS(hidden_size=1280, _window=True, _fused_merger=True, _mixed_precision=False,
               _per_window_sdpa=False, _layer_group=0, _weight_outer=False, _native_window_sdpa=False,
               _sms=2, _window_size=112, _patch_size=14, _host_cache=True, _group_layout_cache={},
               _device_patch_embed=False, _pair_252=True, _batch_vision=True,
               _batch_max_seq=768, _fused_assemble=True,
               _keepalive=torch.full((1024,2), -41, dtype=torch.int16), num_layers=1, _handle=5,
               cache=NS(max_seq_len=1024, reset_to_position=lambda _: None, k_caches=[None], v_caches=[None]),
               _graph_cache=NS(capture=lambda sig: signatures.append(sig) or nullcontext()))
    owner._forward = MethodType(h["_forward"], owner)
    owner._patch_embed = lambda pv, win, seq, smu: pv.reshape(168,4,-1)[win].reshape(seq,-1).clone()
    owner._patch_embed_pair_252 = lambda pv, win: h["_patch_embed_pair_252"](owner, pv, win)
    def plan(seq, batch, mask, cu, **kwargs):
        assert kwargs == dict(valid_patches_per_image=324)
        return 672, NS(graph_key_words=lambda: [648,672,24],
                       selected=NS(execution_len=672, padding_rows=24,
                                   stage_tuple=NS(compute_chunk=672, physical_descriptor=[99])),
                       graph_mode="COMPOSITE_CHILD", selection_scope="test_native",
                       physical_plan_digest="physical", plan_digest="plan"), [99], None
    owner._plan_native_execution = plan
    for name in ("_record_native_execution", "_validate_execution_chunks", "_forward_pair_252"):
        setattr(owner, name, MethodType(h[name], owner))
    grid = torch.tensor([[1,18,18]]*2)
    a = torch.arange(648).float().reshape(-1,1)
    pair_key = ("pair252", *grid.flatten().tolist())
    owner._group_layout_cache[pair_key] = h["_pair_252_layout"](grid)
    pair_pos = owner._group_layout_cache[pair_key][0]
    position_alias = pair_pos.numpy()
    frame_alias = a.numpy()
    source_versions = (pair_pos._version, a._version)
    keepalive_ptr = owner._keepalive.data_ptr()
    keepalive_tail = owner._keepalive[672:].clone()
    original_copy = torch.Tensor.copy_

    def copy_boundary(dst, src, *args, **kwargs):
        result = original_copy(dst, src, *args, **kwargs)
        if dst.data_ptr() == keepalive_ptr and tuple(dst.shape) == (672, 2):
            assert src.device.type == "cpu" and src.data_ptr() == pair_pos.data_ptr()
            copied_positions.append(src.clone())
        return result

    monkeypatch.setattr(torch.Tensor, "copy_", copy_boundary)
    # A previous/third ordinary camera has already cached its 324-row layout.
    # The pair overwrites that same keepalive; restoring the ordinary layout
    # must copy current rows instead of treating its old snapshot as a hit.
    ordinary_pos = h["_compute_vision_position_idx_cpu"](grid[:1], 2)
    owner._position_keepalive_snapshot = None
    h["_refresh_position_keepalive"](owner, ordinary_pos)

    def pair_then_ordinary():
        # A caller can dirty an exposed keepalive without a Tensor version bump.
        # Every pair call must overwrite and publish its current CPU position rows.
        owner._keepalive.numpy()[:672] = -73
        with torch.inference_mode(inference_mode):
            result = h["forward"](owner, a, grid)
        assert owner._keepalive.data_ptr() == keepalive_ptr
        assert torch.equal(owner._keepalive[672:], keepalive_tail)
        assert (pair_pos._version, a._version) == source_versions
        assert not torch.equal(owner._keepalive[:324], ordinary_pos)
        h["_refresh_position_keepalive"](owner, ordinary_pos)
        assert torch.equal(owner._keepalive[:324], ordinary_pos)
        return result

    first = pair_then_ordinary()
    saved = first.clone()
    frame_alias[324:] += 3
    position_alias[336:] += 1
    second = pair_then_ordinary()
    assert torch.equal(first, saved)
    assert torch.equal(first[:81], second[:81])
    assert torch.equal(second[81:], first[81:]+20)
    frame_alias[324:] -= 3
    position_alias[336:] -= 1
    restored = pair_then_ordinary()
    assert torch.equal(first, restored)
    assert len(copied_positions) == 3
    assert torch.equal(copied_positions[0], copied_positions[2])
    assert not torch.equal(copied_positions[0], copied_positions[1])
    assert trace == ["reset", "native", "reset"]*3
    assert signatures[0] == signatures[1] == signatures[2]
    assert len(owner._last_forward_plans) == 1
    receipt = owner._last_forward_plans[0]
    assert (receipt["logical_len"], receipt["execution_len"], receipt["padding_rows"],
            receipt["image_batch_count"], receipt["chunk_size"]) == (648, 672, 24, 2, 672)
    assert receipt["dry_forward_agreement"] and owner._pending_rev is None
    owner._mixed_precision = True
    with pytest.raises(ValueError, match="uniform"):
        h["_forward_pair_252"](owner, a, grid)


def test_pair_planner_binds_real_rows_padding_and_native_signature(monkeypatch):
    h = helpers(); calls = []
    native = lambda *args: calls.append(args) or [123]
    monkeypatch.setattr(torch.ops.rpu, "qwen25vl_vision_resolve_stage_domain", native, raising=False)
    def plan(real, physical, budget, **kwargs):
        assert (real, physical, budget, kwargs["padding_rows"]) == (648,672,0,24)
        assert kwargs["plan_signature"] == (2, True, None, 324)
        assert kwargs["resolve_stage_domain"](672) == [123]
        kwargs["plan_result_sink"](NS(selected=NS(stage_tuple=NS(physical_descriptor=[7]))))
        return 672,672
    h["plan_bounded_prefill_execution"] = plan
    owner = NS(_handle=5, _graph_cache=object())
    out = h["_plan_native_execution"](owner,672,2,torch.zeros(336,336),None,
                                      valid_patches_per_image=324)
    assert out[0] == 672 and out[2] == [7]
    assert calls == [(5,672,2,True,None,0,324)]


def test_cold_pair_admission_rejects_conflicting_schedules():
    tree = ast.parse(SOURCE.read_text())
    builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "build_wall_oss_vision")
    guard = next(n for n in builder.body if isinstance(n, ast.If)
                 and 'RPU_WALL_OSS_VISION_PAIR_252' in ast.unparse(n.test))
    code = compile(ast.Module(body=[guard], type_ignores=[]), str(SOURCE), "exec")
    cold = dict(rpu_env_bool=lambda _: True, experimental_schedule=(0,False),
                window=True, fused_merger=True, sorting=NS(silu_mul=True),
                mixed_precision=False, per_window_sdpa=False, native_window_sdpa=False,
                rmsnorm_fixed32=False)
    exec(code, dict(cold))
    for changes in [dict(mixed_precision=True), dict(per_window_sdpa=True),
                    dict(native_window_sdpa=True), dict(experimental_schedule=(32,False)),
                    dict(experimental_schedule=(0,True)), dict(fused_merger=False),
                    dict(sorting=NS(silu_mul=False))]:
        with pytest.raises(ValueError, match="paired Vision"):
            exec(code, dict(cold, **changes))
    exec(code, dict(cold, rpu_env_bool=lambda _: False, mixed_precision=True))
