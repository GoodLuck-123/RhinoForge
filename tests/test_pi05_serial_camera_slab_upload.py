"""Exercise the real Pi runtime and serial prefix adapter at the upload boundary.

The only device double is Tensor.to("rpu"): it copies real CPU ATen bytes to a
fresh owner on fallback paths. Private serial inputs remain CPU views; native
publication/Graph behavior is tested separately and still requires hardware.
"""
from collections.abc import Mapping
import gc
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import weakref

import pytest
import torch

from rpu_backend.api._execution import (
    PI05_EXECUTION_COMPONENTS,
    PI05_EXECUTION_SUPPORTED,
    normalize_rpu_execution,
)


ROOT = Path(__file__).resolve().parents[1]


def _raw(value):
    if type(value) is not torch.Tensor:
        value = value.as_subclass(torch.Tensor)
    value = value.resolve_neg().resolve_conj().detach().contiguous()
    return value.view(torch.uint8).numpy().tobytes()


def _marked(slab):
    images = [slab.narrow(0, index, 1) for index in range(slab.shape[0])]
    for index, image in enumerate(images):
        image._pi05_batch_slab = slab
        image._pi05_batch_index = index
    return images


def _owner(cores):
    # The installed Pi model receives this actual deeply frozen cold root,
    # not the plain input dictionary passed by the caller.
    execution = normalize_rpu_execution(
        {"model": {"num_cores": cores}},
        entry_point="Pi05Adapter serial camera upload host fixture",
        supported=PI05_EXECUTION_SUPPORTED,
        supported_components=PI05_EXECUTION_COMPONENTS,
    )
    assert isinstance(execution, Mapping) and type(execution) is not dict
    assert isinstance(execution["model"], Mapping) and type(execution["model"]) is not dict
    model = SimpleNamespace(
        _rpu_siglip_batch_n=1,
        _rpu_execution=execution,
        training=False,
    )
    assert vars(model)["_rpu_execution"] is execution
    return model


def _signed_slab():
    pattern = torch.tensor(
        [0.0, -0.0, 1.0, -1.0, 0.33325, -0.33325,
         2**-24, -(2**-24), 65504.0, -65504.0], dtype=torch.float16)
    camera = pattern.repeat((3 * 224 * 224 + 9) // 10)[:3 * 224 * 224]
    camera = camera.reshape(1, 3, 224, 224)
    return torch.cat([camera, -camera, camera.roll(1, dims=-1)], dim=0)


@pytest.fixture
def adapter(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "_pi05_serial_upload_runtime_under_test",
        ROOT / "python/rpu_backend/adapters/pi05/runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    # Import complete production modules, not extracted or mirrored functions.
    from rpu_backend.adapters.pi05 import patches
    from rpu_backend.adapters import siglip

    original_to = torch.Tensor.to
    uploads = []

    def upload(value, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        if str(target) != "rpu":
            return original_to(value, *args, **kwargs)
        plain = value if type(value) is torch.Tensor else value.as_subclass(torch.Tensor)
        result = plain.clone(memory_format=torch.preserve_format)
        uploads.append(dict(source=weakref.ref(value), owner=weakref.ref(result),
                            shape=tuple(value.shape), bytes=_raw(value),
                            source_ptr=value.data_ptr(), destination_ptr=result.data_ptr()))
        return result

    monkeypatch.setattr(torch.Tensor, "to", upload)
    return runtime, patches, siglip, uploads


class _SerialVisionBoundary:
    """Retain each actual input view until the deferred native-consumer seam."""
    def __init__(self):
        self.leases = []
        self.shapes = []

    def embed_image(self, image):
        self.shapes.append(tuple(image.shape))
        self.leases.append(image)
        # A transparent CPU result lets the real prefix adapter assemble rows
        # and masks. This is a device/consumer double, not a model oracle.
        return image.flatten(2).transpose(1, 2)[:, :256, :].clone()

    def embed_language_tokens(self, tokens):
        return torch.ones(tokens.shape[0], tokens.shape[1], 3, dtype=torch.float16)

    def consume_deferred(self):
        return [_raw(image) for image in self.leases]


@pytest.mark.parametrize("cores", [4, 6])
def test_actual_serial_adapter_retains_current_cpu_three_cameras_and_keeps_order(adapter, cores):
    runtime, patches, siglip, uploads = adapter
    owner = _owner(cores)
    boundary = _SerialVisionBoundary()
    owner.paligemma_with_expert = boundary
    slab = _signed_slab()
    original = slab.clone()
    alternate = original.flip(-1)
    masks = [torch.tensor([True]), torch.tensor([False]), torch.tensor([True])]
    tokens = torch.tensor([[2, 7]])
    token_mask = torch.tensor([[True, False]])
    retained = []
    expected_outputs = []
    for request in (original, alternate, original):
        slab.copy_(request)  # Same CPU source address, different complete bytes.
        expected = [_raw(request.narrow(0, index, 1)) for index in range(3)]
        input_images = _marked(slab)
        assert siglip._shared_siglip_image_slab(input_images) is slab
        before = len(uploads)
        views, returned_masks = runtime._prepare_prefill_siglip_inputs(
            owner, input_images, masks, "rpu")
        assert returned_masks is masks
        assert len(uploads) == before and views is input_images
        assert all(view.device.type == 'cpu' and view._pi05_batch_slab is slab for view in views)
        assert [view.storage_offset() for view in views] == [0, 150528, 301056]
        assert all(tuple(view.shape) == (1, 3, 224, 224) and view.dtype == torch.float16
                   and view.is_contiguous() for view in views)
        assert [_raw(view) for view in views] == expected
        start = len(boundary.shapes)
        prefix, pad, attention = patches._patched_embed_prefix(
            owner, views, masks, tokens, token_mask)
        assert boundary.shapes[start:] == [(1, 3, 224, 224)] * 3
        assert boundary.consume_deferred()[-3:] == expected
        assert prefix.shape == (1, 770, 3) and prefix.dtype == torch.float16
        assert torch.isfinite(prefix).all()
        assert torch.equal(pad, torch.cat([mask[:, None].expand(1, 256) for mask in masks]
                                         + [token_mask], dim=1))
        assert attention.shape == (1, 770) and not bool(attention.any())
        retained.append((views, [_raw(view) for view in views], prefix.clone()))
        expected_outputs.append(_raw(prefix))
        # CPU views intentionally follow their producer; completed prefixes do not.
        prefix_before = _raw(prefix)
        slab.fill_(11)
        assert _raw(prefix) == prefix_before
    assert expected_outputs[0] == expected_outputs[2] != expected_outputs[1]
    assert len({item[0][0].data_ptr() for item in retained}) == 1
    assert [_raw(prefix) for _,_,prefix in retained] == expected_outputs
    assert uploads == [] and owner._rpu_siglip_batch_n == 1
    assert owner._rpu_execution == {"model": {"num_cores": cores}}


@pytest.mark.parametrize("cores", [4, 6])
def test_plain_dictionary_cold_config_remains_compatible(adapter, cores):
    runtime, _, _, uploads = adapter
    owner = _owner(cores)
    owner._rpu_execution = {"model": {"num_cores": cores}}
    slab = _signed_slab()
    masks = [torch.tensor([True])] * 3
    views, returned_masks = runtime._prepare_prefill_siglip_inputs(
        owner, _marked(slab), masks, "rpu")
    assert returned_masks is masks and uploads == []
    assert all(view._pi05_batch_slab is slab for view in views)
    assert [view.storage_offset() for view in views] == [0, 150528, 301056]
    assert [_raw(view) for view in views] == [_raw(slab.narrow(0, index, 1)) for index in range(3)]


@pytest.mark.parametrize("cores", [4, 6])
def test_cpu_views_keep_producer_alive_until_existing_serial_consumer_releases(adapter, cores):
    runtime, patches, _, uploads = adapter
    owner = _owner(cores)
    boundary = _SerialVisionBoundary()
    owner.paligemma_with_expert = boundary
    slab = _signed_slab()
    images = _marked(slab)
    expected = [_raw(image) for image in images]
    views, masks = runtime._prepare_prefill_siglip_inputs(
        owner, images, [torch.tensor([True])] * 3, "rpu")
    assert uploads == []
    owner_ref = weakref.ref(slab)
    prefix, _, _ = patches._patched_embed_prefix(
        owner, views, masks, torch.tensor([[1]]), torch.tensor([[True]]))
    del views, images, slab
    gc.collect()
    assert owner_ref() is not None and boundary.consume_deferred() == expected
    prefix_before = _raw(prefix)
    boundary.leases.clear()
    gc.collect()
    assert owner_ref() is None and _raw(prefix) == prefix_before


class _PublicRpuTensor(torch.Tensor):
    @property
    def device(self):
        return SimpleNamespace(type="rpu")


@pytest.mark.parametrize("cores", [4, 6])
@pytest.mark.parametrize("case", [
    "unmarked", "forged_index", "swapped", "changed_view_stride", "strided_slab",
    "offset_slab", "allocation_tail", "fp32", "slab_grad", "view_grad",
    "lazy_negative_slab", "lazy_negative_view", "conjugate", "public_rpu", "two_cameras",
])
def test_noneligible_inputs_preserve_original_per_camera_upload(adapter, cores, case):
    runtime, _, _, uploads = adapter
    slab = _signed_slab()
    if case == "strided_slab":
        slab = torch.zeros(3, 3, 224, 448, dtype=torch.float16)[..., ::2]
    elif case in ("offset_slab", "allocation_tail"):
        bank = torch.zeros(4, 3, 224, 224, dtype=torch.float16)
        slab = bank.narrow(0, int(case == "offset_slab"), 3)
    elif case == "fp32":
        slab = slab.float()
    elif case == "slab_grad":
        slab.requires_grad_(True)
    elif case == "lazy_negative_slab":
        slab = torch._neg_view(slab)
    elif case == "conjugate":
        slab = slab.to(torch.complex64).conj()
    elif case == "two_cameras":
        slab = slab[:2].clone()
    images = _marked(slab)
    if case == "unmarked":
        for image in images:
            del image._pi05_batch_slab
    elif case == "forged_index":
        images[1]._pi05_batch_index = 0
    elif case == "swapped":
        images[0], images[1] = images[1], images[0]
    elif case in ("changed_view_stride", "lazy_negative_view", "public_rpu"):
        changed = (images[0].transpose(2, 3) if case == "changed_view_stride"
                   else torch._neg_view(images[0]) if case == "lazy_negative_view"
                   else images[0].as_subclass(_PublicRpuTensor))
        changed._pi05_batch_slab, changed._pi05_batch_index = slab, 0
        images[0] = changed
    elif case == "view_grad":
        images[0].requires_grad_(True)
    expected = [_raw(image) for image in images]
    masks = [torch.tensor([True])] * len(images)
    prepared, returned_masks = runtime._prepare_prefill_siglip_inputs(
        _owner(cores), images, masks, "rpu")
    assert returned_masks is masks and len(uploads) == len(images)
    assert [_raw(image) for image in prepared] == expected
    assert [entry["shape"] for entry in uploads] == [tuple(image.shape) for image in images]


@pytest.mark.parametrize("case", ["missing_execution", "invalid_model", "float_cores",
                                      "unknown_cores", "training", "TP8_serial", "wrong_batch", "missing_batch"])
def test_unknown_or_other_profiles_keep_original_per_camera_upload(adapter, case):
    runtime, _, _, uploads = adapter
    owner = _owner(4)
    if case == "missing_execution":
        owner._rpu_execution = None
    elif case == "invalid_model":
        owner._rpu_execution = {"model": None}
    elif case == "float_cores":
        owner._rpu_execution = {"model": {"num_cores": 4.0}}
    elif case == "unknown_cores":
        owner._rpu_execution = {"model": {"num_cores": 2}}
    elif case == "training":
        owner.training = True
    elif case == "TP8_serial":
        owner._rpu_execution = {"model": {"num_cores": 8}}
    elif case == "wrong_batch":
        owner._rpu_siglip_batch_n = 2
    elif case == "missing_batch":
        del owner._rpu_siglip_batch_n
    images = _marked(_signed_slab())
    prepared, _ = runtime._prepare_prefill_siglip_inputs(owner, images, [], "rpu")
    assert len(uploads) == 3 and [_raw(image) for image in prepared] == [_raw(image) for image in images]


def test_TP8_private_packed_cpu_views_and_none_inputs_are_unchanged(adapter):
    runtime, _, _, uploads = adapter
    owner = _owner(8)
    owner._rpu_siglip_batch_n = 3
    images = _marked(_signed_slab())
    masks = [torch.tensor([True])] * 3
    prepared, returned_masks = runtime._prepare_prefill_siglip_inputs(owner, images, masks, "rpu")
    assert prepared is images and returned_masks is masks and uploads == []
    empty, returned_masks = runtime._prepare_prefill_siglip_inputs(owner, None, masks, "rpu")
    assert empty is None and returned_masks is masks and uploads == []
