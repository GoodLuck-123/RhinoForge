"""Host memory/adapter contracts; no native import, device execution or model load.

Execute the complete production forward_embed and original CPU packing through
an device memory double. Publication events below describe the reviewed copy_
contract, not hardware measurements or Graph proof.
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "python/rpu_backend/adapters/g05/runtime.py"
VISION = ROOT / "python/rpu_backend/adapters/qwen3_5/vision.py"
HIDDEN = 2048


class RpuMemory:
    """Own CPU backing bytes while exposing the original device view protocol."""

    def __init__(self, data, events):
        self.data = data
        self.events = events
        self.device = SimpleNamespace(type="rpu")

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, index):
        return RpuMemory(self.data[index], self.events)

    def unsqueeze(self, dim):
        return RpuMemory(self.data.unsqueeze(dim), self.events)

    def reshape(self, *shape):
        return RpuMemory(self.data.reshape(*shape), self.events)

    def narrow(self, dim, start, length):
        return RpuMemory(self.data.narrow(dim, start, length), self.events)

    def clone(self):
        self.events.append({"op": "fresh_hidden_clone"})
        return RpuMemory(self.data.clone(), self.events)

    def to(self, *, device=None, dtype=None):
        self.events.append({"op": "rpu_to", "device": device, "dtype": dtype})
        assert device in (None, "rpu"), "fixture never manufactures CPU readbacks"
        return RpuMemory(self.data.to(dtype=dtype or self.dtype), self.events)

    def contiguous(self):
        return RpuMemory(self.data.contiguous(), self.events)

    def copy_(self, source):
        src = source.data if isinstance(source, RpuMemory) else source
        assert self.data.is_contiguous()
        assert src.is_contiguous()
        assert self.data.shape == src.shape
        self.events.append({
            "op": "state_slice_copy",
            "source_device": source.device.type,
            "source_dtype": src.dtype,
            "source_contiguous": src.is_contiguous(),
            "bytes": src.numel() * src.element_size(),
            "offset_bytes": self.data.storage_offset() * self.data.element_size(),
            "destination_ptr": self.data.data_ptr(),
            "publication_count": 1 if source.device.type == "cpu" else 2,
        })
        self.data.copy_(src)
        return self


def _extract(path, names, scope):
    source = path.read_text()
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    for node in nodes:
        exec(compile(ast.get_source_segment(source, node), str(path), "exec"), scope)


@pytest.fixture
def host_adapter(monkeypatch):
    events = []
    original_to = torch.Tensor.to

    def tensor_to(tensor, *args, **kwargs):
        if kwargs.get("device") == "rpu":
            dtype = kwargs.get("dtype", tensor.dtype)
            data = original_to(tensor, dtype=dtype).contiguous().clone()
            events.append({"op": "temporary_RPU_upload", "bytes": data.numel() * data.element_size()})
            return RpuMemory(data, events)
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", tensor_to)
    tokens = SimpleNamespace(IMAGE_TOKEN_INDEX=1, PROPRIO_TOKEN_INDEX=2)
    rpu_ops = SimpleNamespace(spm_alloc_reset_temporary=lambda: None)
    torch_proxy = SimpleNamespace(
        Tensor=torch.Tensor, LongTensor=torch.LongTensor,
        float16=torch.float16, float32=torch.float32,
        int32=torch.int32, long=torch.long,
        tensor=torch.tensor, cat=torch.cat, equal=torch.equal,
        ops=SimpleNamespace(rpu=rpu_ops),
    )
    scope = {
        "torch": torch_proxy,
        "_g05_token_index": lambda model: tokens,
        "_g05_packed_vision_capacity_enabled": lambda: False,
    }
    _extract(RUNTIME, {"_pack_g05_vision", "_g05_packed_camera_batch_count", "_rpu_g05_forward_embed"}, scope)
    visual_scope = {"torch": torch}
    _extract(VISION, {"_visual_run_starts"}, visual_scope)
    for name in ("rpu_backend", "rpu_backend.adapters", "rpu_backend.adapters.qwen3_5"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    visual = ModuleType("rpu_backend.adapters.qwen3_5.vision")
    visual._visual_run_starts = visual_scope["_visual_run_starts"]
    monkeypatch.setitem(sys.modules, visual.__name__, visual)
    return scope["_rpu_g05_forward_embed"], events


def _case(events, cameras=2, *, positions=None, state_rows=1, state_dtype=torch.float16, state_device="cpu", noncontiguous=False):
    length, first_state = (161, 155) if cameras == 2 else (225, 219)
    positions = [first_state] if positions is None else list(positions)
    ids = torch.full((1, length), 7, dtype=torch.long)
    mask = torch.full_like(ids, 3)
    starts = [8 + 72 * index for index in range(cameras)]
    for start in starts:
        ids[:, start:start + 64] = 9
        mask[:, start:start + 64] = 1
    for position in positions:
        ids[0, position] = 11
        mask[0, position] = 2
    data = torch.arange(length * HIDDEN, dtype=torch.float32).reshape(1, length, HIDDEN).remainder(97).half()
    initial = data.clone()
    state_data = torch.arange(state_rows * HIDDEN, dtype=torch.float32).reshape(state_rows, HIDDEN).remainder(43).sub_(20)
    state_data.flatten()[:5] = torch.tensor([0., -0., 2.**-25, -2.**-24, 1.00048828125])
    state_data = state_data.to(dtype=state_dtype)
    if noncontiguous:
        backing = torch.empty(state_rows, HIDDEN * 2, dtype=state_dtype)
        backing[:, ::2].copy_(state_data)
        state_data = backing[:, ::2]
        assert not state_data.is_contiguous()
    state = RpuMemory(state_data, events) if state_device == "rpu" else state_data
    calls = {"mlp": 0, "vision": 0}
    model = SimpleNamespace(
        image_token_index=9,
        proprio_embedder=SimpleNamespace(state_token_id=11),
        _rpu_g05_camera_names=tuple(str(index) for index in range(cameras)),
        _rpu_g05_embedding_cache=(ids.clone(), RpuMemory(data, events)),
    )

    def mlp(proprio):
        calls["mlp"] += 1
        assert proprio.device.type == "cpu" and proprio.dtype == torch.float16
        return model.current_state

    def vision(patches, grid, **kwargs):
        calls["vision"] += 1
        assert patches.shape == (cameras * 256, 1536)
        assert patches.device.type == "cpu" and patches.dtype == torch.float16
        assert torch.equal(grid, torch.tensor([[1, 16, 16]] * cameras))
        assert kwargs["_rpu_temporal_num_frames"] == 1
        assert kwargs["_rpu_camera_batch_count"] == 1
        assert kwargs["_rpu_fusion_run_starts"] == starts
        target = kwargs["_rpu_fusion_target"].data
        for index, start in enumerate(starts):
            target[:, start:start + 64].fill_(index + 100)

    model.current_state = state
    model._rpu_g05_proprio_mlp = mlp
    model.vision_tower = SimpleNamespace(
        patch_size=16, spatial_merge_size=2,
        config=SimpleNamespace(temporal_patch_size=2),
        _rpu_vision_step0=False, _rpu_g05_vision_forward_impl=vision,
    )
    pixels = {str(index): torch.full((1, 1, 3, 256, 256), index / 10, dtype=torch.float32)
              for index in range(cameras)}
    proprio = torch.zeros(1, 32)
    expected = initial.clone()
    for index, start in enumerate(starts):
        expected[:, start:start + 64].fill_(index + 100)
    if positions:
        expected[0, positions[0]:positions[0] + state_rows].copy_(state_data.half())
    return SimpleNamespace(model=model, ids=ids, mask=mask, pixels=pixels, proprio=proprio,
                           initial=initial, expected=expected, positions=positions, state=state_data, calls=calls)


def _run(forward, case, proprio=True):
    return forward(case.model, case.ids, case.mask, case.pixels,
                   case.proprio if proprio else None)


def _bits_equal(actual, expected):
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("cameras", [2, 3])
@pytest.mark.parametrize("dtype,noncontiguous", [(torch.float16, False), (torch.float32, False), (torch.float16, True)])
def test_complete_forward_cpu_state_copies_current_exact_bytes_only_into_state_rows(host_adapter, cameras, dtype, noncontiguous):
    forward, events = host_adapter
    case = _case(events, cameras, state_dtype=dtype, noncontiguous=noncontiguous)
    out = _run(forward, case)
    _bits_equal(out.data, case.expected)
    _bits_equal(case.model._rpu_g05_embedding_cache[1].data, case.initial)
    copies = [event for event in events if event["op"] == "state_slice_copy"]
    assert len(copies) == 1
    assert copies[0]["source_device"] == "cpu"
    assert copies[0]["source_dtype"] == torch.float16 and copies[0]["source_contiguous"]
    assert copies[0]["bytes"] == HIDDEN * 2
    assert copies[0]["offset_bytes"] == case.positions[0] * HIDDEN * 2
    assert copies[0]["destination_ptr"] == out.data.data_ptr() + copies[0]["offset_bytes"]
    assert copies[0]["publication_count"] == 1  # modeled copy_ contract, not hardware PASS
    assert not any(event["op"] == "temporary_RPU_upload" for event in events)
    assert case.calls == {"mlp": 1, "vision": 1}


@pytest.mark.parametrize("cameras", [2, 3])
def test_existing_rpu_state_preserves_original_to_rpu_and_rpu_copy_fallback(host_adapter, cameras):
    forward, events = host_adapter
    case = _case(events, cameras, state_device="rpu")
    out = _run(forward, case)
    _bits_equal(out.data, case.expected)
    assert any(event["op"] == "rpu_to" and event["device"] == "rpu" and event["dtype"] == torch.float16 for event in events)
    copy = next(event for event in events if event["op"] == "state_slice_copy")
    assert copy["source_device"] == "rpu" and copy["publication_count"] == 2


def test_complete_forward_multiple_adjacent_state_rows_and_no_state(host_adapter):
    forward, events = host_adapter
    case = _case(events, positions=[153, 154], state_rows=2)
    _bits_equal(_run(forward, case).data, case.expected)
    copy = next(event for event in events if event["op"] == "state_slice_copy")
    assert copy["bytes"] == 8192 and copy["offset_bytes"] == 153 * 4096
    events.clear()
    case = _case(events, positions=[])
    _bits_equal(_run(forward, case).data, case.expected)
    assert case.calls == {"mlp": 0, "vision": 1}
    assert not any(event["op"] == "state_slice_copy" for event in events)


@pytest.mark.parametrize("cameras", [2, 3])
def test_same_shape_state_a_b_a_keeps_fresh_hidden_and_all_previous_returns(host_adapter, cameras):
    forward, events = host_adapter
    case = _case(events, cameras)
    a = _run(forward, case)
    saved_a = a.data.clone()
    case.model.current_state = case.state.add(1)
    b = _run(forward, case)
    saved_b = b.data.clone()
    assert not torch.equal(a.data.view(torch.int16), b.data.view(torch.int16))
    case.model.current_state = case.state
    restored = _run(forward, case)
    _bits_equal(restored.data, saved_a)
    _bits_equal(a.data, saved_a)
    _bits_equal(b.data, saved_b)
    _bits_equal(case.model._rpu_g05_embedding_cache[1].data, case.initial)
    assert len({a.data.data_ptr(), b.data.data_ptr(), restored.data.data_ptr()}) == 3
    assert case.calls == {"mlp": 3, "vision": 3}


@pytest.mark.parametrize("positions,rows", [([153, 155], 2), ([155], 2)])
def test_original_state_token_count_and_contiguous_run_errors_precede_copy(host_adapter, positions, rows):
    forward, events = host_adapter
    case = _case(events, positions=positions, state_rows=rows)
    with pytest.raises(NotImplementedError, match="contiguous state-token run"):
        _run(forward, case)
    assert case.calls["vision"] == 0
    assert not any(event["op"] == "state_slice_copy" for event in events)


def test_missing_proprio_and_original_mask_profile_admission_remain_fail_closed(host_adapter):
    forward, events = host_adapter
    case = _case(events)
    with pytest.raises(ValueError, match="requires proprio"):
        _run(forward, case, proprio=False)
    assert case.calls == {"mlp": 0, "vision": 0}
    assert not any(event["op"] == "state_slice_copy" for event in events)
    case.mask[0, 0] = 0
    with pytest.raises(NotImplementedError, match="does not support padding"):
        _run(forward, case)
    case.mask[0, 0] = 3
    case.mask[0, 155] = 3
    with pytest.raises(ValueError, match="state token IDs and token-type mask disagree"):
        _run(forward, case)
    case.mask[0, 155] = 2
    case.model._rpu_g05_camera_names = ("unexpected",)
    with pytest.raises(ValueError, match="installed cold profile"):
        _run(forward, case)
