"""Host contract for the real G05 VLM return, with separate CPU/device stores.

The function under test is the complete production entry. Only device labels,
the completed text producer and SDK transport are doubled. This does not test
cache instructions, device math, Graph replay, physical traffic or performance.
"""
import ast
from collections.abc import Mapping
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


class _Bank:
    def __init__(self, rows, width=2048, dtype=torch.float16):
        flat = torch.empty(rows * width + 256, dtype=dtype)
        shift = ((-flat.data_ptr()) % 256) // flat.element_size()
        self.cpu = flat.narrow(0, shift, rows * width).view(rows, width)
        self.cpu.zero_()
        self.ddr = self.cpu.clone()
        self.dirty = False
        self.reported_bytes = self.cpu.numel() * self.cpu.element_size()


class _Tensor:
    def __init__(self, bank, shape, device="rpu"):
        self.bank, self.shape = bank, tuple(shape)
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
        return self.bank.cpu.data_ptr() + self.pointer_delta

    def storage_offset(self):
        return self.offset_value

    def untyped_storage(self):
        return types.SimpleNamespace(nbytes=lambda: self.bank.reported_bytes)

    def is_contiguous(self):
        return self.contiguous_value

    def is_conj(self):
        return self.conj_value

    def is_neg(self):
        return self.neg_value

    def contiguous(self):
        return self

    def _view(self, device_store=False):
        store = self.bank.ddr if device_store else self.bank.cpu
        return store.reshape(-1)[:self.numel()].view(self.shape)

    def to(self, *args, device=None, dtype=None):
        target = args[0] if args else device
        if target == "cpu" and self.device.type == "rpu":
            self.case.flush(self)
            # Preserve the actual old CPU-return shared-Storage behavior.
            return self._view()
        if target is None:
            return self
        self.device = types.SimpleNamespace(type=target)
        return self

    def clone(self):
        self.case.events.append("clone")
        self.case.flush(self)
        bank = _Bank(self.shape[-2], self.shape[-1], self.dtype)
        bank.cpu.copy_(self._view().reshape(bank.cpu.shape))
        bank.ddr.copy_(bank.cpu)
        return self.case.tensor(bank, self.shape)

    def squeeze(self, dim):
        assert dim == 0 and self.shape[0] == 1
        return self.case.tensor(self.bank, self.shape[1:])

    def unsqueeze(self, dim):
        assert dim == 0
        return self.case.tensor(self.bank, (1, *self.shape))


class _GeometryIds:
    def __init__(self, rows, device):
        self.shape, self.device = (rows,), device
        self.dtype = torch.int32

    def __getitem__(self, index):
        raise AssertionError("prefix mode must not read dummy IDs")

    def __getattr__(self, name):
        if name in ("item", "tolist", "to", "copy_", "fill_", "zero_"):
            raise AssertionError("dummy IDs must not be read, copied or filled")
        raise AttributeError(name)


def _case(monkeypatch, *, rows=161, width=2048, dtype=torch.float16,
          normal=False, force=True, fault=None):
    source = (ROOT / "python/rpu_backend/adapters/g05/runtime.py").read_text()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "_rpu_g05_vlm_forward")
    node.decorator_list = []
    case = types.SimpleNamespace(events=[], value=1, normal=normal, force=force,
                                 fail_publish=False, fail_dma=False)
    case.bank = _Bank(((rows + 63) // 64) * 64, width, dtype)

    def tensor(bank, shape, device="rpu"):
        out = _Tensor(bank, shape, device)
        out.case = case
        return out
    case.tensor = tensor

    def flush(value):
        if not case.force:
            return
        # A cache-maintenance double makes public dirty CPU aliases observable.
        if value.bank.dirty:
            value._view(True).copy_(value._view())
        else:
            value._view().copy_(value._view(True))
        value.bank.dirty = False
    case.flush = flush

    def publish(values):
        assert len(values) == 1 and values[0].bank is case.bank
        assert case.events[-1] == "text_capture_completed"
        case.events.append("publish")
        if case.fail_publish:
            raise RuntimeError("source publication failure")
        flush(values[0])

    def empty(shape, *, dtype, device):
        if dtype == torch.int32:
            assert isinstance(shape, int) and device.type == "rpu"
            case.events.append("empty_geometry_ids")
            return _GeometryIds(shape, device)
        assert dtype == torch.int64 and shape == (0, 2) and device == "cpu"
        case.events.append("empty_runs")
        return torch.empty(shape, dtype=dtype)

    def assemble(prefix, ids, parts, runs, *, prefix_is_embeddings=False):
        assert prefix_is_embeddings is True and parts == []
        assert ids.shape == (rows,) and ids.dtype == torch.int32
        assert ids.device == prefix.device
        assert runs.shape == (0, 2) and runs.dtype == torch.int64
        assert prefix.shape == (rows, width)
        assert case.events[-3:] == ["publish", "empty_geometry_ids", "empty_runs"]
        case.events.append("dma")
        if case.fail_dma:
            raise RuntimeError("managed owner/SPM/SDK DMA rejection")
        bank = _Bank(rows, width, dtype)
        bank.ddr.copy_(prefix._view(True))
        # The actual native helper completes before its scratch/leases return.
        case.events.append("dma_completed")
        return tensor(bank, (rows, width))

    class Cache:
        position = 0
        recurrent_states, conv_states = [], []

        @staticmethod
        def reset_to_position(cache, position):
            cache.position = position

    cache_module = types.ModuleType("rpu_backend.api.cache")
    cache_module.RPUCache = Cache
    monkeypatch.setitem(sys.modules, cache_module.__name__, cache_module)
    text_module = types.ModuleType("rpu_backend.adapters.qwen3_5.text")

    def native_text(model, hidden, cache, *, position_ids):
        assert hidden.device.type == "rpu" and position_ids is None
        case.events.append("text")
        case.bank.ddr.fill_(case.value)
        cache.position += rows
        out = tensor(case.bank, (1, rows, width))
        if fault == "dtype":
            out.dtype = torch.float32
        elif fault == "layout":
            out.layout = torch.sparse_coo
        elif fault == "noncontiguous":
            out.contiguous_value = False
        elif fault == "conj":
            out.conj_value = True
        elif fault == "neg":
            out.neg_value = True
        elif fault == "grad":
            out.requires_grad = True
        elif fault == "offset":
            out.offset_value = 1
        elif fault == "alignment":
            out.pointer_delta = 2
        elif fault == "range":
            case.bank.reported_bytes = rows * width * out.element_size() - 1
        elif fault == "zero_pointer":
            out.pointer_delta = -out.data_ptr()
        case.events.append("text_capture_completed")
        return out

    text_module.run_qwen3_5_text = native_text
    text_module._normalize_prefill_position_ids = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, text_module.__name__, text_module)
    facade_torch = types.SimpleNamespace(
        Tensor=_Tensor, LongTensor=torch.Tensor, float16=torch.float16,
        int32=torch.int32, int64=torch.int64, strided=torch.strided, empty=empty,
        rpu=types.SimpleNamespace(get_ddr_flush=lambda: case.normal,
                                  get_ddr_flush_force=lambda: case.force),
        ops=types.SimpleNamespace(rpu=types.SimpleNamespace(
            qwen3vl_publish_inputs=publish, assemble_inputs_embeds_parts=assemble,
            qwen3_5_set_mrope_position_delta=lambda *a: None)),
    )
    namespace = {"torch": facade_torch, "Mapping": Mapping, "types": types,
                 "_check_causal_mask": lambda *a, **k: None,
                 "_g05_cache_num_items": lambda facade: facade._rpu_qwen3_5_cache.position}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / "runtime.py"), "exec"), namespace)
    owner = types.SimpleNamespace(training=False)
    model = types.SimpleNamespace(_rpu_g05_owner=lambda: owner, _rpu_g05_cache=Cache(),
                                  _rpu_qwen3_5=types.SimpleNamespace(handle=1))

    def call(device="rpu"):
        facade = types.SimpleNamespace(num_items=lambda: 0)
        inputs = tensor(_Bank(rows, width), (1, rows, width), device)
        return namespace["_rpu_g05_vlm_forward"](
            model, inputs, None, None, kv_cache=facade, return_kv_cache=True)[0]
    case.call = call
    return case


@pytest.mark.parametrize("rows", [161, 181, 193, 225, 245, 257, 321, 385, 449])
def test_complete_vlm_entry_returns_fresh_aba_outputs(monkeypatch, rows):
    case = _case(monkeypatch, rows=rows)
    retained = []
    for value in (1, 7, 1):
        case.value = value
        out = case.call()
        assert out.shape == (1, rows, 2048) and out.dtype == torch.float16
        assert out.bank is not case.bank and out.is_contiguous()
        assert torch.equal(out._view(True), torch.full(out.shape, value, dtype=torch.float16))
        for old, saved in retained:
            assert torch.equal(old._view(True), saved)
        retained.append((out, out._view(True).clone()))
    assert torch.equal(retained[0][1], retained[2][1])
    assert case.events.count("publish") == case.events.count("dma_completed") == 3
    assert "clone" not in case.events


@pytest.mark.parametrize("rows", [161, 181, 193, 225, 245, 257, 321, 385, 449])
def test_prior_cpu_return_dirty_alias_preserves_source_boundary(monkeypatch, rows):
    case = _case(monkeypatch, rows=rows)
    alias = case.call("cpu")
    assert alias.untyped_storage().data_ptr() == case.bank.cpu.untyped_storage().data_ptr()
    alias.fill_(9)
    case.bank.dirty = True
    case.value = 7
    copied = case.call()
    # The source maintenance must retain the old clone's dirty-alias semantics.
    assert torch.equal(copied._view(True), torch.full(copied.shape, 9, dtype=torch.float16))
    saved = copied._view(True).clone()
    alias.fill_(3)
    case.bank.dirty = True
    case.value = 1
    later = case.call()
    assert torch.equal(copied._view(True), saved)
    assert torch.equal(later._view(True), torch.full(later.shape, 3, dtype=torch.float16))


@pytest.mark.parametrize("fault", ["dtype", "layout", "noncontiguous", "conj", "neg", "grad", "offset", "alignment", "range", "zero_pointer"])
@pytest.mark.parametrize("rows", [161, 385, 449])
def test_layout_range_and_lazy_rejection_keeps_original_clone(monkeypatch, fault, rows):
    case = _case(monkeypatch, rows=rows, fault=fault)
    case.call()
    assert case.events[-1] == "clone"
    assert "publish" not in case.events and "dma" not in case.events


@pytest.mark.parametrize("normal,force", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("rows", [161, 385, 449])
def test_existing_flush_controls_keep_original_clone(monkeypatch, normal, force, rows):
    case = _case(monkeypatch, rows=rows, normal=normal, force=force)
    case.call()
    assert case.events[-1] == "clone"
    assert "publish" not in case.events and "empty_geometry_ids" not in case.events


@pytest.mark.parametrize("rows,width", [(1, 2048), (160, 2048), (226, 2048), (161, 1024),
                                         (192, 2048), (258, 2048), (448, 2048),
                                         (450, 2048), (449, 1024)])
def test_unadmitted_geometry_keeps_original_clone(monkeypatch, rows, width):
    case = _case(monkeypatch, rows=rows, width=width)
    case.call()
    assert case.events[-1] == "clone" and "publish" not in case.events


@pytest.mark.parametrize("where", ["publish", "dma"])
@pytest.mark.parametrize("rows", [161, 385, 449])
def test_publication_and_dma_errors_propagate_without_clone_retry(monkeypatch, where, rows):
    case = _case(monkeypatch, rows=rows)
    case.fail_publish, case.fail_dma = where == "publish", where == "dma"
    with pytest.raises(RuntimeError):
        case.call()
    assert "clone" not in case.events
    if where == "publish":
        assert "empty_geometry_ids" not in case.events
