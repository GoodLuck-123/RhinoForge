"""Run the whole G05 embedding entry and real CPU camera pack/run discovery.

CPU payloads and expected complete tensors use real ATen. device device labels,
embedding/proprio/Vision kernels, SDK CPU/DDR visibility, current SPM admission,
and Graph submission are explicit boundary doubles. This is not a physical DMA,
cache-instruction, kernel, Graph-replay, traffic, or performance measurement.
"""
import ast
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TWO = ("head_rgb", "left_wrist_rgb")
THREE = (*TWO, "right_wrist_rgb")
IMAGE_ID, STATE_ID = 3, 4
IMAGE_TYPE, STATE_TYPE = 2, 3


class _Bank:
    def __init__(self, rows, width, dtype=torch.float16):
        flat = torch.empty(rows * width + 256, dtype=dtype)
        shift = ((-flat.data_ptr()) % 256) // flat.element_size()
        self.cpu = flat.narrow(0, shift, rows * width).view(rows, width)
        self.cpu.zero_()
        self.ddr = self.cpu.clone()
        self.dirty = False
        self.reported_bytes = self.cpu.numel() * self.cpu.element_size()


class _RpuTensor:
    def __init__(self, case, bank, shape, first=0, device="rpu"):
        self.case, self.bank, self.shape, self.first = case, bank, tuple(shape), first
        self.device = types.SimpleNamespace(type=device)
        self.dtype, self.layout = bank.cpu.dtype, torch.strided
        self.requires_grad = False
        self.contiguous_value, self.conj_value, self.neg_value = True, False, False
        self.offset_value, self.pointer_delta = 0, 0

    def numel(self):
        result = 1
        for value in self.shape:
            result *= value
        return result

    def element_size(self):
        return self.bank.cpu.element_size()

    def data_ptr(self):
        return self.bank.cpu.data_ptr() + self.first * self.element_size() + self.pointer_delta

    def storage_offset(self):
        return self.first + self.offset_value

    def untyped_storage(self):
        return types.SimpleNamespace(nbytes=lambda: self.bank.reported_bytes)

    def is_contiguous(self):
        return self.contiguous_value

    def is_conj(self):
        return self.conj_value

    def is_neg(self):
        return self.neg_value

    def _view(self, device_store=False):
        store = self.bank.ddr if device_store else self.bank.cpu
        return store.reshape(-1).narrow(0, self.first, self.numel()).view(self.shape)

    def clone(self):
        self.case.events.append("clone")
        self.case.flush(self)
        bank = _Bank(self.shape[-2], self.shape[-1], self.dtype)
        bank.cpu.copy_(self._view().reshape(bank.cpu.shape))
        bank.ddr.copy_(bank.cpu)
        return _RpuTensor(self.case, bank, self.shape, device=self.device.type)

    def squeeze(self, dim):
        assert dim == 0 and self.shape[0] == 1
        return _RpuTensor(self.case, self.bank, self.shape[1:], self.first)

    def unsqueeze(self, dim):
        assert dim == 0
        return _RpuTensor(self.case, self.bank, (1, *self.shape), self.first)

    def __getitem__(self, index):
        assert index == 0 and self.shape[0] == 1
        return self.squeeze(0)

    def narrow(self, dim, start, length):
        assert dim == 0 and len(self.shape) == 2
        return _RpuTensor(self.case, self.bank, (length, self.shape[1]),
                          self.first + start * self.shape[1])

    def copy_(self, value):
        self.case.events.append("state_cpu_copy")
        self._view().copy_(value)
        if self.case.force:
            self._view(True).copy_(self._view())
        return self


class _UploadedIds:
    def __init__(self, values):
        self.values = values

    def contiguous(self):
        return self


class _GeometryIds:
    def __init__(self, rows, device):
        self.shape, self.device, self.dtype = (rows,), device, torch.int32

    def __getattr__(self, name):
        if name in ("item", "tolist", "to", "copy_", "fill_", "zero_", "values"):
            raise AssertionError("prefix-copy dummy IDs must never be read or initialized")
        raise AttributeError(name)


def _functions(path, names):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    return nodes


def _case(monkeypatch, *, cameras=2, rows=None, width=2048, normal=False,
          force=True, fault=None, frames=1, size=256, names=None):
    names = names or (TWO if cameras == 2 else THREE)
    rows = rows or (161 if cameras == 2 else 225)
    image_rows = (size // 32) ** 2
    assert cameras * image_rows + 2 <= rows
    case = types.SimpleNamespace(events=[], force=force, normal=normal,
                                 fail=None, generation=0, vision_build=0,
                                 vision_replay=0, source=None)
    ids = torch.full((1, rows), 1, dtype=torch.long)
    mask = torch.ones_like(ids)
    for index in range(cameras):
        start = 1 + index * image_rows
        ids[0, start:start + image_rows] = IMAGE_ID
        mask[0, start:start + image_rows] = IMAGE_TYPE
    ids[0, -1], mask[0, -1] = STATE_ID, STATE_TYPE
    case.ids, case.mask = ids, mask
    columns = torch.arange(width, dtype=torch.float32).remainder(31).sub(15).div(16)
    vocab = torch.arange(8, dtype=torch.float32).unsqueeze(1).div(8) + columns
    case.vocab = vocab.to(torch.float16)

    def flush(value):
        if not case.force:
            return
        if value.bank.dirty:
            value._view(True).copy_(value._view())
        else:
            value._view().copy_(value._view(True))
        value.bank.dirty = False
    case.flush = flush

    original_to = torch.Tensor.to

    def tensor_to(value, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if device == "rpu":
            assert value.ndim == 1 and kwargs.get("dtype") == torch.int32
            case.events.append("ids_upload")
            return _UploadedIds(original_to(value, dtype=torch.int32))
        return original_to(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", tensor_to)

    def gather(weight, uploaded):
        assert weight is case.vocab and isinstance(uploaded, _UploadedIds)
        case.events.append("gather")
        bank = _Bank(rows, width)
        bank.ddr.copy_(case.vocab[uploaded.values.long()])
        value = _RpuTensor(case, bank, (rows, width))
        case.source = value
        return value

    def publish(values):
        assert len(values) == 1 and values[0].bank is case.source.bank
        case.events.append("source_publish")
        if case.fail == "publish":
            raise RuntimeError("mandatory source publication rejected")
        flush(values[0])

    def empty(shape, *, dtype, device):
        if dtype == torch.int32:
            assert shape == rows and device.type == "rpu"
            case.events.append("geometry_ids")
            return _GeometryIds(shape, device)
        assert shape == (0, 2) and dtype == torch.int64 and device == "cpu"
        case.events.append("empty_runs")
        return torch.empty(shape, dtype=dtype)

    def assemble(prefix, uploaded, parts, runs, *, prefix_is_embeddings=False):
        assert prefix_is_embeddings is True and parts == []
        assert uploaded.shape == (rows,) and uploaded.dtype == torch.int32
        assert uploaded.device == prefix.device
        assert runs.shape == (0, 2) and runs.dtype == torch.int64
        assert prefix.shape == (rows, width)
        assert case.events[-3:] == ["source_publish", "geometry_ids", "empty_runs"]
        case.events.append("dma_admission")
        if case.fail in ("Graph", "SPM", "owner", "SDK"):
            raise RuntimeError("native " + case.fail + " admission/submission rejected")
        bank = _Bank(rows, width)
        # Reading only DDR exposes any omitted CPU-dirty source publication.
        bank.ddr.copy_(prefix._view(True))
        case.events.append("dma_completed")
        return _RpuTensor(case, bank, (rows, width))

    def vision(patches, grid, **kwargs):
        case.events.append("vision")
        if case.fail == "vision":
            raise RuntimeError("Vision producer rejected")
        assert patches.shape == (cameras * frames * (size // 16) ** 2, 1536)
        assert patches.dtype == torch.float16 and patches.is_contiguous()
        assert grid.tolist() == [[1, size // 16, size // 16]] * (cameras * frames)
        assert kwargs["_rpu_temporal_num_frames"] == frames
        assert kwargs["_rpu_camera_batch_count"] == 1
        target = kwargs["_rpu_fusion_target"]
        starts = kwargs["_rpu_fusion_run_starts"]
        assert starts == [1 + index * image_rows for index in range(cameras)]
        for index, start in enumerate(starts):
            # A declared merger boundary, independent from packing arithmetic.
            value = case.pixels[names[index]].reshape(-1)[0].to(torch.float16)
            target._view(True)[0, start:start + image_rows].fill_(value)
        if case.vision_build == 0:
            case.vision_build += 1
        else:
            case.vision_replay += 1

    vision._g05_k1_camera_group_wrapper = True
    vision._g05_k1_camera_batch = cameras
    tower = types.SimpleNamespace(patch_size=16, spatial_merge_size=2,
                                  config=types.SimpleNamespace(temporal_patch_size=2),
                                  _rpu_vision_step0=True,
                                  _rpu_g05_vision_forward_impl=vision)
    model = types.SimpleNamespace(image_token_index=IMAGE_ID,
                                  cfg=types.SimpleNamespace(num_input_images=cameras),
                                  _rpu_g05_embed_w_rpu=case.vocab, vision_tower=tower,
                                  proprio_embedder=types.SimpleNamespace(state_token_id=STATE_ID))
    # The real configure_g05_camera_profile default THREE returns without a
    # marker. Only the explicit two-camera selection publishes this attribute.
    if cameras == 2 or names != THREE:
        model._rpu_g05_camera_names = names
    model._rpu_g05_proprio_mlp = lambda proprio: proprio[:, :1].expand(-1, width).clone()
    ops = types.SimpleNamespace(spm_alloc_reset_temporary=lambda: case.events.append("reset_spm"),
                                gather_embedding=gather, qwen3vl_publish_inputs=publish,
                                assemble_inputs_embeds_parts=assemble)

    class _Torch:
        Tensor = _RpuTensor

        def __getattr__(self, name):
            return getattr(torch, name)

    facade = _Torch()
    facade.ops = types.SimpleNamespace(rpu=ops)
    facade.rpu = types.SimpleNamespace(get_ddr_flush=lambda: case.normal,
                                       get_ddr_flush_force=lambda: case.force)
    facade.empty = empty
    namespace = {"torch": facade, "_G05_TWO_CAMERA_NAMES": TWO,
                 "_G05_RELEASE_CAMERA_NAMES": THREE,
                 "_g05_token_index": lambda model: types.SimpleNamespace(
                     IMAGE_TOKEN_INDEX=IMAGE_TYPE, PROPRIO_TOKEN_INDEX=STATE_TYPE),
                 "_g05_packed_vision_capacity_enabled": lambda: False}
    source = ROOT / "python/rpu_backend/adapters/g05/runtime.py"
    nodes = _functions(source, ("_rpu_g05_forward_embed", "_pack_g05_vision",
                                "_g05_packed_camera_batch_count"))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    visual_source = ROOT / "python/rpu_backend/adapters/qwen3_5/vision.py"
    visual_namespace = {"torch": torch}
    exec(compile(ast.Module(body=_functions(visual_source, ("_visual_run_starts",)),
                            type_ignores=[]), str(visual_source), "exec"), visual_namespace)
    visual_module = types.ModuleType("rpu_backend.adapters.qwen3_5.vision")
    visual_module._visual_run_starts = visual_namespace["_visual_run_starts"]
    monkeypatch.setitem(sys.modules, visual_module.__name__, visual_module)

    def change_fault():
        value = model._rpu_g05_embedding_cache[1]
        if fault == "dtype":
            value.dtype = torch.float32
        elif fault == "layout":
            value.layout = torch.sparse_coo
        elif fault == "noncontiguous":
            value.contiguous_value = False
        elif fault == "conj":
            value.conj_value = True
        elif fault == "neg":
            value.neg_value = True
        elif fault == "grad":
            value.requires_grad = True
        elif fault == "offset":
            value.offset_value = 1
        elif fault == "alignment":
            value.pointer_delta = 2
        elif fault == "range":
            value.bank.reported_bytes = value.numel() * value.element_size() - 1
        elif fault == "zero_pointer":
            value.pointer_delta = -value.data_ptr()
        elif fault == "cpu":
            value.device = types.SimpleNamespace(type="cpu")

    def pixels(value):
        return {name: torch.full((1, frames, 3, size, size),
                                 value + index / 8, dtype=torch.float32)
                for index, name in enumerate(names)}

    def call(value=1):
        case.pixels = case.pixels_for(value)
        return namespace["_rpu_g05_forward_embed"](
            model, case.ids, case.mask, case.pixels,
            proprio=torch.tensor([[value / 4]], dtype=torch.float32))

    case.call, case.model, case.pixels_for = call, model, pixels
    case.reference = lambda value, source=None: _reference(
        case.vocab[case.ids[0]] if source is None else source,
        case.ids, case.pixels_for(value), value / 4, image_rows)
    case.change_fault = change_fault
    return case


def _reference(source, ids, pixels, proprio, image_rows):
    expected = source.clone().view(1, ids.shape[1], -1)
    expected[0, -1].fill_(torch.tensor(proprio).to(torch.float16))
    for index, camera in enumerate(pixels.values()):
        start = 1 + index * image_rows
        expected[0, start:start + image_rows].fill_(camera.flatten()[0].to(torch.float16))
    return expected


@pytest.mark.parametrize("cameras,rows", [(2, 161), (2, 181), (2, 193), (2, 257), (2, 321), (2, 385), (3, 225), (3, 245), (3, 257), (3, 321), (3, 385), (3, 449)])
def test_complete_embed_entry_device_copy_preserves_aba_cache_and_retention(monkeypatch, cameras, rows):
    case = _case(monkeypatch, cameras=cameras, rows=rows)
    retained = []
    for value in (-1, 7, -1):
        out = case.call(value)
        assert out.bank is not case.source.bank
        assert out.dtype == torch.float16 and out.shape == case.reference(value).shape
        assert torch.equal(out._view(True), case.reference(value))
        assert torch.equal(case.source._view(True), case.vocab[case.ids[0]])
        for old, saved in retained:
            assert torch.equal(old._view(True), saved)
        retained.append((out, out._view(True).clone()))
    assert torch.equal(retained[0][1], retained[2][1])
    assert len({out.data_ptr() for out, _ in retained}) == 3
    assert case.events.count("gather") == case.events.count("ids_upload") == 1
    assert case.events.count("source_publish") == case.events.count("dma_completed") == 3
    assert case.vision_build == 1 and case.vision_replay == 2
    assert "clone" not in case.events


@pytest.mark.parametrize("cameras,rows", [(2, 161), (2, 181), (2, 193), (2, 257), (2, 321), (2, 385), (3, 225), (3, 245), (3, 257), (3, 321), (3, 385), (3, 449)])
def test_cached_source_dirty_cpu_alias_keeps_full_publication(monkeypatch, cameras, rows):
    case = _case(monkeypatch, cameras=cameras, rows=rows)
    old = case.call(1)
    old_bytes = old._view(True).clone()
    alias = case.source._view()
    alias.fill_(-9)
    case.source.bank.dirty = True
    copied = case.call(7)
    expected = case.reference(7, torch.full_like(case.vocab[case.ids[0]], -9))
    assert torch.equal(copied._view(True), expected)
    assert torch.equal(old._view(True), old_bytes)
    saved = copied._view(True).clone()
    alias.fill_(3)
    case.source.bank.dirty = True
    later = case.call(-1)
    expected = case.reference(-1, torch.full_like(case.vocab[case.ids[0]], 3))
    assert torch.equal(later._view(True), expected)
    assert torch.equal(copied._view(True), saved)


@pytest.mark.parametrize("fault", ["dtype", "layout", "noncontiguous", "conj", "neg",
                                   "grad", "offset", "alignment", "range", "zero_pointer", "cpu"])
@pytest.mark.parametrize("cameras,rows", [(2, 161), (2, 385), (3, 449)])
def test_nonplain_or_incomplete_cached_source_keeps_original_clone(monkeypatch, fault, cameras, rows):
    case = _case(monkeypatch, cameras=cameras, rows=rows, fault=fault)
    case.call()
    case.change_fault()
    case.events.clear()
    case.call()
    assert "clone" in case.events and "source_publish" not in case.events
    assert "dma_admission" not in case.events


@pytest.mark.parametrize("normal,force", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("cameras,rows", [(2, 161), (2, 385), (3, 449)])
def test_original_flush_controls_keep_original_clone(monkeypatch, normal, force, cameras, rows):
    case = _case(monkeypatch, cameras=cameras, rows=rows, normal=normal, force=force)
    case.call()
    assert "clone" in case.events and "source_publish" not in case.events


@pytest.mark.parametrize("options", [dict(rows=160), dict(cameras=3, rows=226),
                                      dict(width=1024), dict(frames=6), dict(size=224),
                                      dict(names=("other_rgb", "left_wrist_rgb")),
                                      dict(cameras=3, names=("other_rgb", "left_wrist_rgb", "right_wrist_rgb")),
                                       dict(rows=225), dict(rows=245), dict(rows=449),
                                       dict(cameras=3, rows=224), dict(cameras=3, rows=226),
                                       dict(cameras=3, rows=258), dict(cameras=3, rows=448),
                                       dict(cameras=3, rows=450)])
def test_noncanonical_camera_or_prefix_geometry_keeps_original_clone(monkeypatch, options):
    case = _case(monkeypatch, **options)
    case.call()
    assert "clone" in case.events and "source_publish" not in case.events


@pytest.mark.parametrize("failure", ["publish", "Graph", "SPM", "owner", "SDK"])
@pytest.mark.parametrize("cameras,rows", [(2, 161), (2, 385), (3, 449)])
def test_existing_native_errors_propagate_before_vision_without_clone_retry(monkeypatch, failure, cameras, rows):
    case = _case(monkeypatch, cameras=cameras, rows=rows)
    case.fail = failure
    with pytest.raises(RuntimeError):
        case.call()
    assert "source_publish" in case.events
    assert "clone" not in case.events and "vision" not in case.events
    assert "dma_completed" not in case.events


@pytest.mark.parametrize("failure", ["batch", "padding", "camera_names", "image_mask", "state_mask", "fragmented_image", "vision"])
def test_original_entry_error_order_and_cache_owner_preserved(monkeypatch, failure):
    case = _case(monkeypatch)
    old = case.call()
    saved = old._view(True).clone()
    cache = case.model._rpu_g05_embedding_cache
    case.events.clear()
    if failure == "batch":
        case.ids = case.ids.expand(2, -1)
    elif failure == "padding":
        case.mask[0, 0] = 0
    elif failure == "camera_names":
        case.model._rpu_g05_camera_names = THREE
    elif failure == "image_mask":
        case.mask[0, 1] = 1
    elif failure == "state_mask":
        case.mask[0, -1] = 1
    elif failure == "fragmented_image":
        case.ids[0, 1], case.ids[0, -2] = 1, IMAGE_ID
        case.mask[0, 1], case.mask[0, -2] = 1, IMAGE_TYPE
    else:
        case.fail = "vision"
    with pytest.raises((RuntimeError, ValueError, NotImplementedError)):
        case.call()
    assert torch.equal(old._view(True), saved)
    if failure in ("batch", "padding", "camera_names", "image_mask"):
        assert case.events == [] and case.model._rpu_g05_embedding_cache is cache
    elif failure == "state_mask":
        assert "dma_completed" in case.events and "state_cpu_copy" not in case.events
    elif failure == "fragmented_image":
        assert "dma_completed" in case.events and "vision" not in case.events


@pytest.mark.parametrize("cameras,rows", [(2, 161), (2, 181), (2, 193), (2, 257), (2, 321), (2, 385), (3, 225), (3, 245), (3, 257), (3, 321), (3, 385), (3, 449)])
def test_changed_ids_regather_new_cached_owner_and_keep_saved_outputs(monkeypatch, cameras, rows):
    case = _case(monkeypatch, cameras=cameras, rows=rows)
    first = case.call(-1)
    first_bytes, old_source = first._view(True).clone(), case.source.bank
    case.ids[0, 0] = 5
    second = case.call(7)
    assert case.source.bank is not old_source
    assert case.events.count("gather") == case.events.count("ids_upload") == 2
    assert torch.equal(second._view(True), case.reference(7))
    assert torch.equal(first._view(True), first_bytes)



@pytest.mark.parametrize("rows", [225, 245, 257, 321, 385, 449])
def test_real_three_camera_default_without_marker_uses_fresh_copy(monkeypatch, rows):
    case = _case(monkeypatch, cameras=3, rows=rows)
    assert not hasattr(case.model, "_rpu_g05_camera_names")
    assert case.model.cfg.num_input_images == 3
    out = case.call(-1)
    assert "dma_completed" in case.events and "clone" not in case.events
    assert torch.equal(out._view(True), case.reference(-1))


@pytest.mark.parametrize("rows", [161, 181, 193, 257, 321, 385])
def test_missing_explicit_two_camera_marker_keeps_original_clone(monkeypatch, rows):
    case = _case(monkeypatch, rows=rows)
    del case.model._rpu_g05_camera_names
    case.call()
    assert "clone" in case.events and "source_publish" not in case.events


@pytest.mark.parametrize("fault", ["config", "order"])
@pytest.mark.parametrize("rows", [225, 245, 257, 321, 385, 449])
def test_default_three_camera_config_and_order_are_bounded(monkeypatch, fault, rows):
    case = _case(monkeypatch, cameras=3, rows=rows)
    if fault == "config":
        case.model.cfg.num_input_images = 2
    else:
        original = case.pixels_for
        case.pixels_for = lambda value: dict(reversed(tuple(original(value).items())))
    case.call()
    assert "clone" in case.events and "source_publish" not in case.events
