"""Rhino vision input allocation reuse must never reuse image contents."""
import ast
import ctypes
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from rpu_backend.adapters.rhinovla import vision


def _owner():
    return NS(
        _rpu_vision_patch_embed_w_rpu=torch.empty(128, 32, dtype=torch.int8),
        _rpu_vision_patch_input_cache={},
        _rpu_vision_spatial_merge_size=2,
        _rpu_vision_max_hw=48,
        _rpu_vision_position_idx_keepalive=torch.full((128, 2), -17, dtype=torch.int16),
        _rpu_vision_batched_pos_idx_cache={},
        _rpu_vision_batch_position_witness=None,
    )


def test_patch_input_aba_and_inplace_updates_refresh_both_slabs(monkeypatch):
    owner = _owner()
    a = torch.randn(24, 32)
    b = torch.randn_like(a)
    first = vision._prepare_vision_cached_patch_input(owner, a, 24)
    resident_ptr = first.data_ptr()
    host, resident = next(iter(owner._rpu_vision_patch_input_cache.values()))
    host_ptr = host.data_ptr()
    real_copy = torch.Tensor.copy_
    calls = []
    def copy(dst, source, *args, **kwargs):
        calls.append((dst.data_ptr(), source.dtype, dst.dtype))
        return real_copy(dst, source, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "copy_", copy)
    for pixels in (b, a):
        result = vision._prepare_vision_cached_patch_input(owner, pixels, 24)
        assert result is resident and result.data_ptr() == resident_ptr
        assert host.data_ptr() == host_ptr
        assert torch.equal(result, pixels.half())
    a.add_(0.75)
    result = vision._prepare_vision_cached_patch_input(owner, a, 24)
    assert torch.equal(result, a.half())
    assert calls == [(host_ptr, torch.float32, torch.float16),
                     (resident_ptr, torch.float16, torch.float16)] * 3


def test_patch_input_accepts_noncontiguous_and_half_without_recaching():
    owner = _owner()
    source = torch.randn(32, 24).T
    assert not source.is_contiguous()
    first = vision._prepare_vision_cached_patch_input(owner, source, 24)
    assert torch.equal(first, source.half())
    half = source.half().contiguous()
    second = vision._prepare_vision_cached_patch_input(owner, half, 24)
    assert first is second and torch.equal(second, half)
    assert len(owner._rpu_vision_patch_input_cache) == 1


def test_patch_cache_shape_bound_and_inference_to_normal_transition():
    owner = _owner()
    with torch.inference_mode():
        first = vision._prepare_vision_cached_patch_input(owner, torch.ones(24, 32), 24)
    second = vision._prepare_vision_cached_patch_input(owner, torch.full((24, 32), 2.0), 24)
    assert first is second and torch.all(second == 2)
    for rows in range(1, vision._RPU_VISION_INPUT_CACHE_MAX_ENTRIES + 2):
        output = vision._prepare_vision_cached_patch_input(owner, torch.full((rows, 32), float(rows)), rows)
        assert output.shape == (rows, 32) and torch.all(output == rows)
        assert len(owner._rpu_vision_patch_input_cache) <= vision._RPU_VISION_INPUT_CACHE_MAX_ENTRIES
    assert all(key[0] != (24, 32) for key in owner._rpu_vision_patch_input_cache)


@pytest.mark.parametrize("pixels,rows", [
    (torch.ones(24, 31), 24), (torch.ones(23, 32), 24),
    (torch.ones(1, 24, 32), 24), (torch.ones(24, 32, dtype=torch.int32), 24),
    (torch.ones(0, 32), 0), (None, 24),
])
def test_bad_patch_inputs_do_not_change_existing_cache(pixels, rows):
    owner = _owner()
    output = vision._prepare_vision_cached_patch_input(owner, torch.ones(24, 32), 24)
    version = output._version
    keys = tuple(owner._rpu_vision_patch_input_cache)
    with pytest.raises(ValueError, match="floating CPU or device"):
        vision._prepare_vision_cached_patch_input(owner, pixels, rows)
    assert tuple(owner._rpu_vision_patch_input_cache) == keys
    assert output._version == version and torch.all(output == 1)


def _prime(owner, grid):
    rows = int(grid.prod(-1).sum())
    vision._prime_vision_cached_batch_positions(owner, grid, rows)
    actual = owner._rpu_vision_position_idx_keepalive[:rows]
    expected = vision._compute_vision_position_idx_cpu(grid, 2)
    assert torch.equal(actual, expected)


def test_batch_positions_grid_aba_owner_replacement_and_mutations():
    owner = _owner()
    a = torch.tensor([[1, 2, 4]] * 3)
    b = torch.tensor([[1, 4, 2]] * 3)
    keepalive = owner._rpu_vision_position_idx_keepalive
    for grid in (a, b, a):
        before = keepalive._version
        _prime(owner, grid)
        assert keepalive._version > before
        after = keepalive._version
        _prime(owner, grid)
        assert keepalive._version == after
    assert torch.all(keepalive[24:] == -17)
    # Same input tensor object, changed grid contents with identical row count.
    a.copy_(b)
    _prime(owner, a)
    keepalive[0, 0] = -9
    _prime(owner, a)
    replacement = torch.full_like(keepalive, -7)
    owner._rpu_vision_position_idx_keepalive = replacement
    _prime(owner, a)
    assert replacement._version > 0 and torch.all(replacement[24:] == -7)
    # A modified private cache entry cannot poison a later prime.
    source = owner._rpu_vision_batch_position_witness[3]
    source.fill_(-13)
    _prime(owner, a)
    assert owner._rpu_vision_batch_position_witness[3] is not source
    _prime(owner, torch.tensor([[1, 4, 4]] * 3))


def test_batch_position_cache_is_bounded_and_inference_owner_is_refreshed():
    owner = _owner()
    for width in (2, 4, 6, 8, 10, 12):
        _prime(owner, torch.tensor([[1, 2, width]]))
        assert len(owner._rpu_vision_batched_pos_idx_cache) <= vision._RPU_VISION_INPUT_CACHE_MAX_ENTRIES
    with torch.inference_mode():
        owner._rpu_vision_position_idx_keepalive = torch.zeros(128, 2, dtype=torch.int16)
        grid = torch.tensor([[1, 2, 4]] * 3)
        _prime(owner, grid)
        owner._rpu_vision_position_idx_keepalive.fill_(-23)
        _prime(owner, grid)
    assert owner._rpu_vision_batch_position_witness[2] is None


@pytest.mark.parametrize("grid,rows", [
    (torch.tensor([[1, 3, 4]]), 12), (torch.tensor([[1, -2, 4]]), -8),
    (torch.tensor([[1.0, 2.0, 4.0]]), 8), (torch.empty(0, 3, dtype=torch.int64), 0),
    (torch.tensor([[1, 2]]), 2), (torch.tensor([[1, 2, 4]]), 9),
    (torch.tensor([[1, 50, 2]]), 100), (torch.tensor([[1, 16, 16]]), 256),
])
def test_bad_grid_never_overwrites_position_keepalive(grid, rows):
    owner = _owner()
    _prime(owner, torch.tensor([[1, 2, 4]] * 3))
    old = owner._rpu_vision_position_idx_keepalive.clone()
    witness = owner._rpu_vision_batch_position_witness
    with pytest.raises(ValueError):
        vision._prime_vision_cached_batch_positions(owner, grid, rows)
    assert torch.equal(owner._rpu_vision_position_idx_keepalive, old)
    assert owner._rpu_vision_batch_position_witness is witness


@pytest.mark.parametrize("batched", [False, True])
def test_uncached_position_preparation_refreshes_dirty_alias_without_upload(monkeypatch, batched):
    path = Path(vision.__file__)
    forward = next(node for node in ast.parse(path.read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name == "_rpu_vision_forward")
    branches = [node for node in ast.walk(forward)
                if isinstance(node, ast.If)
                and ast.unparse(node.test) == "getattr(self, '_rpu_vision_rope_disable', False)"
                and "keepalive.narrow" in ast.unparse(node)]
    branch = next(node for node in branches
                  if ("total_patches" in ast.unparse(node)) == batched)
    prepare = compile(ast.Module([branch], type_ignores=[]), str(path), "exec")
    copies, uploads = [], []

    class PositionTensor(torch.Tensor):
        @property
        def device(self):
            return NS(type="rpu")

        def copy_(self, source, *args, **kwargs):
            assert source.device.type == "cpu"
            plain = self.as_subclass(torch.Tensor)
            torch.Tensor.copy_(plain, source, *args, **kwargs)
            copies.append((self.data_ptr(), self.nbytes, plain.clone()))
            return self

    original_to = torch.Tensor.to

    def observe_to(value, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        if getattr(target, "type", None) == "rpu" or target == "rpu":
            uploads.append(tuple(value.shape))
            return value.clone().as_subclass(PositionTensor)
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", observe_to)
    keepalive = torch.full((784 if batched else 272, 2), 777,
                          dtype=torch.int16).as_subclass(PositionTensor)
    plain = keepalive.as_subclass(torch.Tensor)
    address = keepalive.data_ptr()
    for height, width in [(16, 16), (16, 16), (8, 16), (16, 16)]:
        grid = torch.tensor([[1, height, width]] * 3)
        expected_one = torch.tensor([
            (row, col)
            for block_row in range(0, height, 2)
            for block_col in range(0, width, 2)
            for row in range(block_row, block_row + 2)
            for col in range(block_col, block_col + 2)
        ], dtype=torch.int16)
        expected = expected_one.repeat((3 if batched else 1), 1)
        rows = expected.size(0)
        for image in range(1 if batched else 3):
            tail = plain[rows:].clone()
            version = keepalive._version
            ctypes.memset(address, 0x7F, rows * 4)
            assert keepalive._version == version
            count = len(copies)
            namespace = dict(vision.__dict__, self=NS(), keepalive=keepalive,
                             input_cache_enabled=False, pos_idx_rpu_by_image=None,
                             grid_thw_cpu=grid, patches_per_image=[height * width] * 3,
                             spatial_merge_size=2, total_patches=height * width * 3,
                             n_i=height * width, i=image)
            exec(prepare, namespace)
            assert keepalive.data_ptr() == address and len(copies) == count + 1
            assert copies[-1][:2] == (address, rows * 4)
            assert torch.equal(copies[-1][2], expected)
            assert torch.equal(plain[:rows], expected)
            assert torch.equal(plain[rows:], tail)
    assert uploads == []
    if not batched:
        before = plain.clone()
        count = len(copies)
        namespace["n_i"] -= 1
        with pytest.raises(RuntimeError, match="position_idx rows"):
            exec(prepare, namespace)
        assert len(copies) == count and torch.equal(plain, before)


def test_cache_flag_is_cold_and_failed_replacement_preserves_retained_owners(monkeypatch):
    from rpu_backend.tests.acceptance import test_vla_vision_install_transaction as tx
    state = tx.native_install.__wrapped__(NS(param=vision), monkeypatch)
    model = tx._model()
    monkeypatch.setenv("RPU_RHINOVLA_VISION_INPUT_CACHE", "1")
    try:
        tx._install(state, model)
        resource = model._rpu_vision_retirement_state
        pixel_cache = model._rpu_vision_patch_input_cache
        position_cache = model._rpu_vision_batched_pos_idx_cache
        pixel_cache["held"] = torch.ones(1)
        def retains(container, target):
            return container is target or (isinstance(container, tuple) and
                   any(retains(item, target) for item in container))
        assert retains(resource.keepalive, pixel_cache)
        assert retains(resource.keepalive, position_cache)
        monkeypatch.setenv("RPU_RHINOVLA_VISION_INPUT_CACHE", "0")
        assert model._rpu_vision_input_cache_enabled is True
        state["fail_set"].add(2)
        with pytest.raises(RuntimeError, match="synthetic set_weights"):
            tx._install(state, model)
        assert model._rpu_vision_input_cache_enabled is True
        assert model._rpu_vision_patch_input_cache is pixel_cache
        assert model._rpu_vision_batched_pos_idx_cache is position_cache
        assert model._rpu_vision_retirement_state is resource
        assert state["alive"] == {1}
        resource.retire()
        assert resource.keepalive == () and not state["alive"]
    finally:
        tx._finalize(model)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("inference", [False, True])
def test_cached_cpu_positions_refresh_actual_forward_and_replay(monkeypatch, batched, inference):
    """Actual Python forward; CPU ATen storage, deferred native/Graph doubles.

    The doubles check publication order and owners, not SDK cache coherence or
    hardware Graph execution. No cached INT16 source may be uploaded to device.
    """
    import sys
    import weakref
    from contextlib import contextmanager

    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_vl.modeling_qwen3_vl",
                        NS(BaseModelOutputWithDeepstackFeatures=NS))
    copies, consumed, retained = [], [], []

    class PositionTensor(torch.Tensor):
        @property
        def device(self):
            return NS(type="rpu")

        def copy_(self, source, *args, **kwargs):
            assert not graph.active
            assert source.device.type == "cpu" and source.dtype == torch.int16
            assert source.is_contiguous()
            plain = self.as_subclass(torch.Tensor)
            torch.Tensor.copy_(plain, source, *args, **kwargs)
            copies.append((self.data_ptr(), self.nbytes, plain.clone()))
            return self

    original_to = torch.Tensor.to

    def to(value, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        if getattr(target, "type", None) == "rpu" or target == "rpu":
            assert value.dtype != torch.int16, "cached positions must stay on CPU"
            # Simulate the existing patch-embedding upload only.
            return value.clone()
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)

    class DeferredGraph:
        def __init__(self):
            self.active = False
            self.entries = set()
            self.builds = self.replays = 0
            self.pending = []

        @contextmanager
        def capture(self, sig):
            assert not self.active
            self.active = True
            if sig in self.entries:
                self.replays += 1
            else:
                self.entries.add(sig)
                self.builds += 1
            try:
                yield
                for consume in self.pending:
                    consume()
            finally:
                self.pending.clear()
                self.active = False

    graph = DeferredGraph()
    monkeypatch.setattr(vision.rpu_backend.graph, "GraphSignature", lambda **kw:
                        (kw["op_id"], tuple(kw["shapes"]), tuple(kw["dyn_dims"])))
    monkeypatch.setattr(vision, "_execution_graph_key_words", lambda model: ())
    monkeypatch.setattr(vision, "_plan_rhinovla_vision_execution", lambda model, rows, batch:
                        NS(selected=NS(stage_tuple=NS(physical_descriptor=(rows, batch),
                                                      compute_chunk=16)),
                           graph_key_words=lambda: (rows, batch)))
    monkeypatch.setattr(vision, "_vision_execution_receipt", lambda *args: {"host_double": True})

    class Model(NS):
        def fast_pos_embed_interpolate(self, grid):
            return torch.zeros(int(grid.prod(-1).sum()), 4, dtype=torch.float16)

    def native_forward(handle, hidden, keys, values, rows, batch, descriptor):
        assert graph.active and handle == 7
        assert tuple(descriptor) == (rows, batch)
        owner = model._rpu_vision_position_idx_keepalive
        assert copies[-1][:2] == (owner.data_ptr(), rows * 4)
        hidden_ref, owner_ref = weakref.ref(hidden), weakref.ref(owner)
        out = torch.empty_like(hidden)
        published = copies[-1][2]

        def consume():
            # Delayed until capture exit: Python callers must retain both live
            # input and the current stable owner, including after replacement.
            current, live = owner_ref(), hidden_ref()
            assert current is not None and live is not None
            positions = current.as_subclass(torch.Tensor)[:rows]
            assert torch.equal(positions, published)
            out.copy_(live + (positions[:, 0] * 16 + positions[:, 1]).half()[None, :, None])
            consumed.append((current.data_ptr(), positions.clone()))

        graph.pending.append(consume)
        return out

    def reset_temporary():
        assert not graph.active and not graph.pending

    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_forward", native_forward, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_get_resolved_chunk_size", lambda handle: 16,
                        raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_pop_deepstack_snapshots", lambda handle: [],
                        raising=False)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", reset_temporary, raising=False)

    def positions(height, width):
        # Independent explicit merge order; do not use the production helper.
        return torch.tensor([(row, col)
                             for br in range(0, height, 2) for bc in range(0, width, 2)
                             for row in range(br, br + 2) for col in range(bc, bc + 2)],
                            dtype=torch.int16)

    with torch.inference_mode(inference):
        model = Model(
            _rpu_vision_handle=7, _rpu_vision_spatial_merge_size=2,
            _rpu_vision_num_layers=1, _rpu_vision_hidden_size=4,
            _rpu_vision_kv_cache=NS(k_caches=[None], v_caches=[None], reset_to_position=lambda p: None),
            _rpu_vision_graph_cache=graph, _rpu_vision_deepstack_hash=0,
            _rpu_vision_prep_cache_enabled=True, _rpu_vision_input_cache_enabled=False,
            _rpu_vision_max_hw=48, _rpu_vision_batch_views=batched,
            _rpu_vision_position_idx_keepalive=torch.full((64, 2), 777, dtype=torch.int16)
                .as_subclass(PositionTensor),
            _rpu_vision_patch_embed_w=torch.eye(4, dtype=torch.float16),
            _rpu_vision_patch_embed_b=torch.zeros(4, dtype=torch.float16),
            _rpu_vision_pos_embed_cache={}, _rpu_vision_pos_idx_cache={},
            _rpu_vision_deepstack_indexes=[], deepstack_merger_list=[], merger=lambda x: x,
        )
        grid = torch.tensor([[1, 2, 4]] * 3)
        pixels = torch.zeros(24, 4, dtype=torch.float16)
        expected_a = positions(2, 4).repeat(3, 1)

        def run(expected, data=pixels):
            owner = model._rpu_vision_position_idx_keepalive
            plain = owner.as_subclass(torch.Tensor)
            rows = expected.size(0) if batched else int(grid[0].prod())
            tail = plain[rows:].clone()
            version = vision._vision_tensor_version(owner)
            # CPU aliases bypass TensorVersion, exactly the dirty-owner case.
            plain.numpy()[:rows] = -19
            assert vision._vision_tensor_version(owner) == version
            before = len(copies)
            result = vision._rpu_vision_forward(model, data, grid)
            want = data + (expected[:, 0] * 16 + expected[:, 1]).half()[:, None]
            assert torch.equal(result.last_hidden_state, want)
            assert torch.equal(result.pooler_output, want.float())
            assert len(copies) - before == (1 if batched else 3)
            assert torch.equal(plain[rows:], tail)
            assert all(t.device.type == "cpu" and t.dtype == torch.int16 and t.is_contiguous()
                       for pair in model._rpu_vision_pos_idx_cache.values() for t in pair)
            for old, snapshot in retained:
                assert torch.equal(old, snapshot)
            retained.append((result.pooler_output, result.pooler_output.clone()))
            return result.pooler_output

        first = run(expected_a)
        key_a = vision._grid_thw_cache_key(grid, 2)
        cached_a = model._rpu_vision_pos_idx_cache[key_a]
        run(expected_a)
        assert model._rpu_vision_pos_idx_cache[key_a] is cached_a
        assert graph.builds == 1 and graph.replays == (1 if batched else 5)
        version = vision._vision_tensor_version(cached_a[0])
        cached_a[0].numpy()[:, 0] += 1
        assert vision._vision_tensor_version(cached_a[0]) == version
        changed = expected_a.clone()
        changed[:8, 0] += 1
        assert not torch.equal(run(changed), first)
        cached_a[0].numpy()[:, 0] -= 1
        assert torch.equal(run(expected_a), first)

        version = vision._vision_tensor_version(grid)
        grid.numpy()[:, 1:] = (4, 2)
        assert vision._vision_tensor_version(grid) == version
        run(positions(4, 2).repeat(3, 1))
        assert graph.builds == 1  # Changed semantic grid, same Graph signature.
        grid.numpy()[:, 1:] = (2, 4)
        run(expected_a)
        assert model._rpu_vision_pos_idx_cache[key_a] is cached_a
        grid.numpy()[:, 1:] = (2, 8)
        run(positions(2, 8).repeat(3, 1), torch.zeros(48, 4, dtype=torch.float16))
        assert graph.builds == 2
        grid.numpy()[:, 1:] = (2, 4)

        old_owner = model._rpu_vision_position_idx_keepalive
        model._rpu_vision_position_idx_keepalive = torch.full((64, 2), 333, dtype=torch.int16) \
            .as_subclass(PositionTensor)
        new_address = model._rpu_vision_position_idx_keepalive.data_ptr()
        assert new_address != old_owner.data_ptr()
        run(expected_a)
        assert consumed[-1][0] == new_address and graph.builds == 2
        version = vision._vision_tensor_version(pixels)
        pixels.numpy()[:] = 2
        assert vision._vision_tensor_version(pixels) == version
        assert torch.equal(run(expected_a), first + 2)
        pixels.numpy()[:] = 0
        assert torch.equal(run(expected_a), first)


def test_fused_merger_preserves_sequential_equal_resolution_views(monkeypatch):
    original_to = torch.Tensor.to

    def to(value, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        if getattr(target, "type", None) == "rpu" or target == "rpu":
            return value.clone()
        return original_to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(vision, "_execution_graph_key_words", lambda model: ())
    monkeypatch.setattr(vision, "_plan_rhinovla_vision_execution", lambda model, rows, batch:
                        NS(selected=NS(stage_tuple=NS(physical_descriptor=(rows, batch),
                                                      compute_chunk=rows)),
                           graph_key_words=lambda: (rows, batch)))
    monkeypatch.setattr(vision, "_vision_execution_receipt", lambda *args: {})

    calls = []

    def native_forward(handle, hidden, keys, values, rows, batch, descriptor):
        calls.append((rows, batch))
        return hidden

    def pop_merged(handle):
        marker = float(len(calls))
        return [torch.full((2, 4), marker, dtype=torch.float16),
                torch.full((2, 4), marker + 10, dtype=torch.float16)]

    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_forward", native_forward, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_get_resolved_chunk_size",
                        lambda handle: 8, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_pop_merged", pop_merged, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3vl_vision_pop_deepstack_snapshots",
                        lambda handle: pytest.fail("fused merger must consume snapshots"),
                        raising=False)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: None, raising=False)

    class Model(NS):
        def fast_pos_embed_interpolate(self, grid):
            return torch.zeros(int(grid.prod(-1).sum()), 4, dtype=torch.float16)

    model = Model(
        _rpu_vision_handle=7, _rpu_vision_spatial_merge_size=2,
        _rpu_vision_num_layers=1, _rpu_vision_hidden_size=4,
        _rpu_vision_kv_cache=NS(k_caches=[None], v_caches=[None],
                                reset_to_position=lambda position: None),
        _rpu_vision_graph_cache=NS(), _rpu_vision_deepstack_hash=0,
        _rpu_vision_prep_cache_enabled=False, _rpu_vision_input_cache_enabled=False,
        _rpu_vision_max_hw=48, _rpu_vision_batch_views=False,
        _rpu_vision_graph_disable=True,
        _rpu_vision_position_idx_keepalive=torch.zeros(64, 2, dtype=torch.int16),
        _rpu_vision_patch_embed_w=torch.eye(4, dtype=torch.float16),
        _rpu_vision_patch_embed_b=torch.zeros(4, dtype=torch.float16),
        _rpu_vision_deepstack_indexes=[0], deepstack_merger_list=[object()],
        merger=object(), _rpu_vision_fused_merger=True,
    )
    grid = torch.tensor([[1, 2, 4]] * 3)
    result = vision._rpu_vision_forward(
        model, torch.zeros(24, 4, dtype=torch.float16), grid,
    )

    assert calls == [(8, 1)] * 3
    assert torch.equal(result.pooler_output[:, 0],
                       torch.tensor([1, 1, 2, 2, 3, 3], dtype=torch.float16))
    assert torch.equal(result.deepstack_features[0][:, 0], result.pooler_output[:, 0] + 10)
