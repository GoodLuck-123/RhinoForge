"""Actual swizzle/storage and install-failure checks for bounded decoder banks."""
import copy
import gc
import weakref
from types import SimpleNamespace as NS

import pytest
import torch

from rpu_backend.quant import load
from rpu_backend.runtime import weights
from rpu_backend.adapters import qwen3_vl as adapter
from rpu_backend.tests.acceptance.test_qwen3_vl_install_lifecycle import install_runtime  # noqa: F401
from test_qwen3_vl_merger_lifecycle import _model_2b, _model_4b


ROLES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
         "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def tiny_model(dtype):
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    text = model.model.language_model = torch.nn.Module()
    text.embed_tokens = torch.nn.Embedding(256, 256, dtype=torch.float16)
    model.lm_head = torch.nn.Linear(256, 256, bias=False, dtype=torch.float16)
    model.lm_head.weight = text.embed_tokens.weight
    text.layers = torch.nn.ModuleList()
    generator = torch.Generator().manual_seed(217)
    for _ in range(2):
        layer = torch.nn.Module()
        layer.self_attn, layer.mlp = torch.nn.Module(), torch.nn.Module()
        for role in ROLES:
            parent, name = role.split(".")
            linear = torch.nn.Linear(256, 256, bias=False, dtype=torch.float16)
            linear.weight = torch.nn.Parameter(torch.randint(
                -120, 120, (256, 256), generator=generator).to(dtype),
                requires_grad=dtype == torch.float16)
            if dtype == torch.int8:
                linear.register_buffer("weight_scale", torch.linspace(.01, .5, 256).half())
            getattr(layer, parent).add_module(name, linear)
        layer.norm = torch.nn.LayerNorm(256, dtype=torch.float16)
        text.layers.append(layer)
    model._rpu_swizzle_started = True
    return model


@pytest.fixture
def host_bank_transport(monkeypatch):
    original = torch.empty
    state = {"allocations": [], "moves": [], "banks": []}
    def empty(*args, **kwargs):
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
            tensor = original(*args, **kwargs)
            state["allocations"].append((tensor.numel(), tensor.dtype))
            state["banks"].append(weakref.ref(tensor))
            return tensor
        return original(*args, **kwargs)
    def move(layer, device):
        assert device == "rpu"
        state["moves"].append(layer)
        return layer
    monkeypatch.setattr(torch, "empty", empty)
    monkeypatch.setattr(torch.nn.Module, "to", move)
    return state


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
@pytest.mark.parametrize("per_layer", [False, True])
def test_real_converter_and_parameters_preserve_bytes_scales_tied_head_and_lifetime(host_bank_transport, dtype, per_layer):
    model = tiny_model(dtype)
    source_refs = [weakref.ref(layer.get_submodule(role).weight)
                   for layer in model.model.language_model.layers for role in ROLES]
    reference = copy.deepcopy(model)
    for layer in reference.model.language_model.layers:
        weights.convert_linear_weights_inplace(layer, skip_names=set())
    load._move_materialized_decoder_for_rpu(model, dtype=dtype, per_layer=per_layer)
    assert host_bank_transport["allocations"] == (
        [(7 * 256 * 256, dtype)] * 2 if per_layer else [(14 * 256 * 256, dtype)])
    assert host_bank_transport["moves"] == list(model.model.language_model.layers)
    assert all(ref() is None for ref in source_refs)
    storage_ids = set()
    for layer, original in zip(model.model.language_model.layers, reference.model.language_model.layers):
        layer_storage_ids = set()
        for role in ROLES:
            actual, expected = layer.get_submodule(role), original.get_submodule(role)
            assert torch.equal(actual.weight, expected.weight)
            assert actual._rpu_linear_partition == expected._rpu_linear_partition
            assert actual.weight.is_contiguous() and not actual.weight.requires_grad
            assert actual.weight.storage_offset() * actual.weight.element_size() % 256 == 0
            storage_ids.add(actual.weight.untyped_storage().data_ptr())
            layer_storage_ids.add(actual.weight.untyped_storage().data_ptr())
            if dtype == torch.int8:
                assert torch.equal(actual.weight_scale, expected.weight_scale)
        assert torch.equal(layer.norm.weight, original.norm.weight)
        assert len(layer_storage_ids) == 1
    assert len(storage_ids) == (2 if per_layer else 1)
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    assert torch.equal(model.lm_head.weight, reference.lm_head.weight)
    with pytest.raises(ValueError, match="unswizzled CPU"):
        load._move_materialized_decoder_for_rpu(model, dtype=dtype, per_layer=per_layer)
    retained = model.model.language_model.layers[-1].mlp.down_proj.weight
    expected = reference.model.language_model.layers[-1].mlp.down_proj.weight
    del model, reference, layer, original, actual
    host_bank_transport["moves"].clear()
    gc.collect()
    assert all(ref() is None for ref in host_bank_transport["banks"])
    assert torch.equal(retained, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
@pytest.mark.parametrize("fail_second", [False, True])
def test_layer_cpu_owners_are_reclaimed_before_next_bank(
        host_bank_transport, monkeypatch, dtype, fail_second):
    model = tiny_model(dtype)
    originals = [[weakref.ref(layer.get_submodule(role).weight) for role in ROLES]
                 for layer in model.model.language_model.layers]
    transformed = []
    reclaimed = []
    transform = weights.transform_linear_weight
    allocate = torch.empty

    def observe_transform(*args, **kwargs):
        result = transform(*args, **kwargs)
        transformed.append(weakref.ref(result))
        return result

    def reclaim():
        index = len(reclaimed)
        # Check before GC: no validation local, old Parameter or swizzle
        # temporary may keep CPU storage alive at the page-release boundary.
        assert all(ref() is None for ref in originals[index])
        assert all(ref() is None for ref in transformed)
        reclaimed.append(index)

    def next_bank(*args, **kwargs):
        if kwargs.get("device") == "rpu" and host_bank_transport["allocations"]:
            assert reclaimed == [0]
            assert all(ref() is not None for ref in originals[1])
            if fail_second:
                raise RuntimeError("second device bank allocation failed")
        return allocate(*args, **kwargs)

    monkeypatch.setattr(weights, "transform_linear_weight", observe_transform)
    monkeypatch.setattr(weights, "_release_cpu_weight_pages", reclaim)
    monkeypatch.setattr(torch, "empty", next_bank)
    if fail_second:
        with pytest.raises(RuntimeError, match="second device bank"):
            load._move_materialized_decoder_for_rpu(model, dtype=dtype, per_layer=True)
        assert reclaimed == [0]
        assert model._rpu_swizzle_started is True
        with pytest.raises(ValueError, match="unswizzled CPU"):
            load._move_materialized_decoder_for_rpu(model, dtype=dtype, per_layer=True)
    else:
        load._move_materialized_decoder_for_rpu(model, dtype=dtype, per_layer=True)
        assert reclaimed == [0, 1]


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
@pytest.mark.parametrize("fault", ["unowned", "last_dtype", "last_swizzled"])
@pytest.mark.parametrize("per_layer", [False, True])
def test_bad_inventory_fails_before_allocation_or_first_swizzle(host_bank_transport, monkeypatch, dtype, fault, per_layer):
    model = tiny_model(dtype)
    if fault == "unowned":
        del model._rpu_swizzle_started
    elif fault == "last_dtype":
        model.model.language_model.layers[-1].mlp.down_proj.weight = torch.nn.Parameter(torch.ones(256, 256))
    else:
        model.model.language_model.layers[-1].mlp.down_proj._rpu_linear_partition = 0
    monkeypatch.setattr(weights, "convert_linear_weights_inplace", lambda *a, **k: pytest.fail("early swizzle"))
    with pytest.raises((RuntimeError, ValueError), match="ownership/poison|unswizzled CPU"):
        load._move_materialized_decoder_for_rpu(model, dtype=dtype, per_layer=per_layer)
    assert host_bank_transport["allocations"] == []


@pytest.mark.parametrize("dtype", [torch.float16, torch.int8])
def test_full_2b_layout_uses_wide_byte_offsets_without_allocating_gigabytes(monkeypatch, dtype):
    shapes = {"self_attn.q_proj": (2048, 2048), "self_attn.k_proj": (1024, 2048),
              "self_attn.v_proj": (1024, 2048), "self_attn.o_proj": (2048, 2048),
              "mlp.gate_proj": (6144, 2048), "mlp.up_proj": (6144, 2048),
              "mlp.down_proj": (2048, 6144)}
    raw = {f"model.language_model.layers.{i}.{role}.weight": torch.empty(shape, device="meta", dtype=dtype)
           for i in range(28) for role, shape in shapes.items()}
    names = tuple(tuple(f"model.language_model.layers.{i}.{role}.weight" for role in ROLES) for i in range(28))
    original = torch.empty
    allocations = []
    def empty(*args, **kwargs):
        assert kwargs["device"] == "rpu"
        allocations.append(args[0])
        return original(*args, **(kwargs | {"device": "meta"}))
    monkeypatch.setattr(torch, "empty", empty)
    views = load._allocate_decoder_projection_views(NS(get_parameter=raw.__getitem__), names, dtype=dtype)
    assert allocations == [1409286144]
    assert len({v.untyped_storage()._cdata for v in views.values()}) == 1
    end = 0
    for name in (name for layer in names for name in layer):
        view = views[name]
        start = view.storage_offset() * view.element_size()
        assert start == end and start % 256 == 0
        end = start + view.numel() * view.element_size()
    assert end == 1409286144 * (2 if dtype == torch.float16 else 1)
    if dtype == torch.float16:
        assert views[names[-1][-1]].storage_offset() * 2 > 2**31


def test_full_4b_fp16_layout_retains_36_separate_layer_banks_without_gigabyte_allocations(monkeypatch):
    shapes = {"self_attn.q_proj": (4096, 2560), "self_attn.k_proj": (1024, 2560),
              "self_attn.v_proj": (1024, 2560), "self_attn.o_proj": (2560, 4096),
              "mlp.gate_proj": (9728, 2560), "mlp.up_proj": (9728, 2560),
              "mlp.down_proj": (2560, 9728)}
    raw = {f"model.language_model.layers.{i}.{role}.weight": torch.empty(shape, device="meta", dtype=torch.float16)
           for i in range(36) for role, shape in shapes.items()}
    names = tuple(tuple(f"model.language_model.layers.{i}.{role}.weight" for role in ROLES) for i in range(36))
    original = torch.empty
    allocations = []
    def empty(*args, **kwargs):
        assert kwargs == {"dtype": torch.float16, "device": "rpu"}
        allocations.append(args[0] * 2)
        return original(*args, **(kwargs | {"device": "meta"}))
    monkeypatch.setattr(torch, "empty", empty)
    views = [load._allocate_decoder_projection_views(
        NS(get_parameter=raw.__getitem__), (layer,), dtype=torch.float16) for layer in names]
    assert allocations == [201850880] * 36
    assert sum(allocations) == 7266631680
    assert len({v.untyped_storage()._cdata for layer in views for v in layer.values()}) == 36
    for layer_names, layer_views in zip(names, views, strict=True):
        assert len({v.untyped_storage()._cdata for v in layer_views.values()}) == 1
        end = 0
        for name in layer_names:
            view = layer_views[name]
            start = view.storage_offset() * 2
            assert tuple(view.shape) == tuple(raw[name].shape)
            assert start == end and start % 256 == 0
            end = start + view.numel() * 2
        assert end == 201850880 < 2**32


@pytest.mark.parametrize("text_w8", [False, True])
def test_actual_adapter_selects_banks_without_changing_vision_precision_or_tied_head(
    install_runtime, monkeypatch, text_w8,
):
    model = _model_2b()
    if text_w8:
        model.config.quant_config = copy.deepcopy(load._QWEN3_VL_2B_W8A16_QUANT_CONFIG)
        # Metadata selects the real 2B W8 route; the tiny installer double has
        # no Vision/decoder inventory. Real projection conversion is above.
        monkeypatch.setattr(adapter, "_validate_qwen3_vl_2b_w8a16_model", lambda model: True)
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    original_embedding = model.model.language_model.embed_tokens.weight
    before = original_embedding.detach().clone()
    calls = []
    def move(owner, *, dtype):
        assert owner is model and owner._rpu_swizzle_started
        assert install_runtime["allocator_enabled"]
        calls.append(dtype)
    monkeypatch.setattr(load, "_move_materialized_decoder_for_rpu", move)
    owner = adapter.Qwen3VLAdapter(model)
    assert owner.to_rpu() is model
    assert calls == [torch.int8 if text_w8 else torch.float16]
    assert install_runtime["vision_kwargs"]["_pack_fp16_block_weights"] is True
    assert not install_runtime["vision_kwargs"].get("w8a16", False)
    assert not install_runtime["text_kwargs"].get("runtime_align_w8a16", False)
    assert model.model.language_model.embed_tokens.weight is original_embedding
    assert torch.equal(original_embedding, before)
    assert model.lm_head.weight.data_ptr() != original_embedding.data_ptr()
    assert model.lm_head.weight.dtype == torch.float16
    assert owner.to_rpu() is model and len(calls) == 1
    owner.close()


def test_exact_4b_fp16_adapter_selects_per_layer_decoder_banks(install_runtime, monkeypatch):
    model = _model_4b()
    calls = []
    def move(value, *, dtype, per_layer=False):
        assert value is model and value._rpu_swizzle_started
        calls.append((dtype, per_layer))
    monkeypatch.setattr(load, "_move_materialized_decoder_for_rpu", move)
    owner = adapter.Qwen3VLAdapter(model)
    assert owner.to_rpu() is model
    assert calls == [(torch.float16, True)]
    assert not install_runtime["vision_kwargs"].get("w8a16", False)
    assert install_runtime["vision_kwargs"]["_pack_fp16_block_weights"] is True
    assert model.lm_head.weight.dtype == torch.float16
    assert owner.to_rpu() is model and len(calls) == 1
    owner.close()


@pytest.mark.parametrize(("per_layer", "fail_at"), [
    (False, "allocation"), (False, "second_layer_copy"),
    (True, "allocation"), (True, "second_layer_allocation"), (True, "second_layer_copy"),
])
def test_real_adapter_keeps_failed_bank_install_poisoned(install_runtime, host_bank_transport, monkeypatch, fail_at, per_layer):
    model = _model_4b() if per_layer else _model_2b()
    actual_tree = tiny_model(torch.float16)
    model.model.language_model.layers = actual_tree.model.language_model.layers
    model.get_parameter = actual_tree.get_parameter
    model.get_submodule = actual_tree.get_submodule
    owner = adapter.Qwen3VLAdapter(model)
    if fail_at in ("allocation", "second_layer_allocation"):
        original = torch.empty
        count = 0
        def empty(*args, **kwargs):
            nonlocal count
            if kwargs.get("device") == "rpu":
                assert model._rpu_swizzle_started
                count += 1
                if count == (2 if fail_at == "second_layer_allocation" else 1):
                    raise RuntimeError("bank allocation failed")
            return original(*args, **kwargs)
        monkeypatch.setattr(torch, "empty", empty)
    else:
        original = torch.Tensor.copy_
        count = 0
        def copy_(self, other, *args, **kwargs):
            nonlocal count
            if self.shape == (256, 256) and self.untyped_storage().nbytes() == (7 if per_layer else 14) * 256 * 256 * 2:
                count += 1
                if count == 8:
                    raise RuntimeError("bank copy failed")
            return original(self, other, *args, **kwargs)
        monkeypatch.setattr(torch.Tensor, "copy_", copy_)
    with pytest.raises(RuntimeError, match="bank .* failed"):
        owner.to_rpu()
    assert model._rpu_swizzle_started and not getattr(model, "_rpu_swizzled", False)
    assert ("claim", "model") in install_runtime["events"]
    assert ("install", "vision") not in install_runtime["events"]
    if fail_at in ("second_layer_copy", "second_layer_allocation"):
        retained = actual_tree.model.language_model.layers[0].self_attn.q_proj.weight
        assert retained.untyped_storage().nbytes() == (7 if per_layer else 14) * 256 * 256 * 2
        assert torch.isfinite(retained).all()
    with pytest.raises(adapter.RPUBackendError, match="prior swizzle attempt"):
        owner.to_rpu()
