"""CPU-only mapped W8 loading; tiny storage is not full-model/device evidence."""
import copy
import gc
import hashlib
import json

import pytest
import torch
from safetensors import safe_open
from torch import nn
from torch.multiprocessing.reductions import StorageWeakRef
from transformers import Qwen3VLConfig

from rpu_backend.api import conditional_generation as public
from rpu_backend.quant import convert_qwen3_vl as convert
from rpu_backend.quant import load, load_qwen3_vl_runtime as runtime
from rpu_backend.runtime.weights import transform_linear_weight
from test_qwen3_vl_runtime_quant_loading import (
    TinyStorage, config, tiny_schema, write_tiny_source,
)


@pytest.fixture(autouse=True)
def bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("mapped-runtime-w8")
    source, destination = root / "source", root / "converted"
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        write_tiny_source(source, config(32, 8, source=True), explicit_head=True)
        raw_config = json.loads((source / "config.json").read_text())
        raw_config["text_config"].pop("tie_word_embeddings")
        (source / "config.json").write_text(json.dumps(raw_config))
        # Only physical test dimensions are reduced. Admission, complete
        # Text7 inventory, converter and safetensors serialization stay real.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(convert, "_checkpoint_tensor_shapes", tiny_schema)
            stats = convert.convert_checkpoint(source, destination, bits=8)
    finally:
        torch.set_num_threads(previous)
    assert stats["n_quantized"] == 64 * 7 + 1
    (destination / "model.safetensors").chmod(0o444)
    cfg = Qwen3VLConfig.from_dict(json.loads((destination / "config.json").read_text()))
    return destination, cfg


def _load_checkpoint(checkpoint, monkeypatch, **kwargs):
    directory, cfg = checkpoint
    monkeypatch.setattr(runtime.AutoModelForImageTextToText, "from_config", TinyStorage)
    return runtime.load_qwen3_vl_runtime_imagetext(
        copy.deepcopy(cfg), str(directory), local_files_only=True, **kwargs,
    )


@pytest.mark.parametrize("mapped", [False, True])
def test_real_checkpoint_keeps_only_selected_weights_mapped_after_handles_close(
    checkpoint, monkeypatch, mapped,
):
    opened, closed, sources = [], [], {}

    class ObservedOpen:
        def __init__(self, *args, **kwargs):
            self.handle = safe_open(*args, **kwargs)

        def __enter__(self):
            self.handle.__enter__()
            opened.append(id(self))
            return self

        def __exit__(self, *args):
            try:
                return self.handle.__exit__(*args)
            finally:
                closed.append(id(self))
                self.handle = None

        def keys(self):
            return self.handle.keys()

        def get_slice(self, name):
            return self.handle.get_slice(name)

        def get_tensor(self, name):
            value = self.handle.get_tensor(name)
            sources[name] = value.data_ptr()
            return value

    monkeypatch.setattr(runtime, "safe_open", ObservedOpen)
    model = _load_checkpoint(
        checkpoint, monkeypatch, **({"_mmap_w8_weights": True} if mapped else {}),
    )
    gc.collect()
    assert opened and opened == closed
    assert runtime._validate_qwen3_vl_runtime_model(model)
    quantized = {name + ".weight" for name, _, _ in runtime._quantized_modules(model)}
    assert len(quantized) == 449 and "lm_head.weight" in quantized
    shard = checkpoint[0] / "model.safetensors"
    with safe_open(shard, framework="pt") as reader:
        for name, value in (*model.named_parameters(), *model.named_buffers()):
            assert value.device.type == "cpu" and value.is_contiguous()
            assert torch.equal(value, reader.get_tensor(name)), name
            if mapped and name in quantized:
                assert value.dtype == torch.int8 and value.data_ptr() == sources[name]
            else:
                # Default weights, scales and all floating companions remain
                # independent copies, including in the mapped weight mode.
                assert value.data_ptr() != sources[name], name


@pytest.mark.parametrize("mapped", [False, True])
def test_cpu_mutation_preserves_checkpoint_and_invalidates_inventory(
    checkpoint, monkeypatch, mapped,
):
    model = _load_checkpoint(checkpoint, monkeypatch, _mmap_w8_weights=mapped)
    shard = checkpoint[0] / "model.safetensors"
    before = hashlib.sha256(shard.read_bytes()).hexdigest()
    original = int(model.lm_head.weight[0, 0])
    model.lm_head.weight[0, 0] = original + 1 if original < 127 else original - 1
    assert hashlib.sha256(shard.read_bytes()).hexdigest() == before
    with safe_open(shard, framework="pt") as reader:
        assert int(reader.get_tensor("lm_head.weight")[0, 0]) == original
    with pytest.raises(RuntimeError, match="inventory"):
        runtime._validate_qwen3_vl_runtime_model(model)


def test_existing_swizzles_preserve_mapped_source_and_release_replaced_storage(
    checkpoint, monkeypatch,
):
    model = _load_checkpoint(checkpoint, monkeypatch, _mmap_w8_weights=True)
    shard = checkpoint[0] / "model.safetensors"
    before = hashlib.sha256(shard.read_bytes()).hexdigest()
    leaf = model.model.language_model.layers[0].self_attn.q_proj
    original = leaf.weight.detach().clone()
    source_storage = StorageWeakRef(leaf.weight.untyped_storage())
    version, pointer = leaf.weight._version, leaf.weight.data_ptr()
    for partition in (0, 1):
        transformed = transform_linear_weight(leaf.weight.data, partition, 8)
        expected = transform_linear_weight(original, partition, 8)
        assert torch.equal(transformed, expected)
        assert transformed.data_ptr() != pointer
        assert leaf.weight._version == version and torch.equal(leaf.weight, original)
        del transformed, expected
    # Mirror the real converter's replacement, then drop its temporary.
    transformed = transform_linear_weight(leaf.weight.data, 1, 8)
    leaf.weight.data = transformed
    del transformed
    gc.collect()
    assert source_storage.expired()
    assert hashlib.sha256(shard.read_bytes()).hexdigest() == before


@pytest.mark.parametrize("size,bits", [(2, 8), (8, 8), (8, 4), (32, 4)])
def test_mapped_mode_rejects_other_runtime_profiles_before_checkpoint_resolution(
    monkeypatch, size, bits,
):
    monkeypatch.setattr(load, "_resolve_checkpoint_dir", lambda *a, **k: pytest.fail("checkpoint read"))
    with pytest.raises(ValueError, match="exact 32B W8"):
        runtime.load_qwen3_vl_runtime_imagetext(config(size, bits), "unused", _mmap_w8_weights=True)


def test_mapped_mode_rejects_legacy_32b_metadata_before_checkpoint_resolution(monkeypatch):
    cfg = config(32, 8)
    cfg.quant_config = copy.deepcopy(load._QWEN3_VL_32B_W8A16_QUANT_CONFIG)
    monkeypatch.setattr(load, "_resolve_checkpoint_dir", lambda *a, **k: pytest.fail("checkpoint read"))
    with pytest.raises(ValueError, match="exact registered runtime quant profile"):
        runtime.load_qwen3_vl_runtime_imagetext(cfg, "unused", _mmap_w8_weights=True)


@pytest.mark.parametrize("fault", ["weight_dtype", "weight_stride", "scale_dtype", "zero", "nan", "inf"])
def test_omitting_weight_copy_keeps_raw_weight_and_scale_validation(fault):
    weight = torch.zeros((128, 512), dtype=torch.int8)
    scale = torch.ones(128, dtype=torch.float16)
    if fault == "weight_dtype":
        weight = weight.half()
    elif fault == "weight_stride":
        weight = weight.t().contiguous().t()
    elif fault == "scale_dtype":
        scale = scale.float()
    else:
        scale[0] = {"zero": 0, "nan": float("nan"), "inf": float("inf")}[fault]
    with pytest.raises(ValueError, match="runtime W8 (weight|scale)"):
        runtime._copy_w8_projection(weight, scale, n=128, k=512, copy_weight=False)


@pytest.mark.parametrize("size,bits", [(32, 8), (8, 8), (32, 4)])
@pytest.mark.parametrize("device", [None, "cpu", "rpu", "rpu:0"])
def test_public_loader_selects_mapping_only_for_one_step_32b_w8(
    monkeypatch, size, bits, device,
):
    cfg, calls, moved = config(size, bits), [], []
    model = nn.Module()
    model.config = cfg

    class Adapter:
        def __init__(self, value):
            assert value is model

        def to_rpu(self):
            moved.append(True)
            return model

    def loader(actual_cfg, path, **kwargs):
        assert actual_cfg is cfg and path == "unused/checkpoint"
        calls.append(kwargs)
        return model

    monkeypatch.setattr(public.AutoConfig, "from_pretrained", lambda *a, **k: cfg)
    monkeypatch.setattr(public, "get_adapter", lambda arch: Adapter)
    monkeypatch.setattr(runtime, "load_qwen3_vl_runtime_imagetext", loader)
    options = {"revision": "fixed-revision", "subfolder": "weights", "local_files_only": True}
    actual = public.RPUModelForConditionalGeneration.from_pretrained(
        "unused/checkpoint", **({"device": device} if device is not None else {}), **options,
    )
    expected = {"dtype": torch.float16, **options}
    on_rpu = device in ("rpu", "rpu:0")
    if on_rpu and (size, bits) == (32, 8):
        expected["_mmap_w8_weights"] = True
    assert actual is model and calls == [expected]
    assert moved == ([True] if on_rpu else [])


@pytest.mark.parametrize("device", [None, "rpu"])
def test_internal_mapping_switch_is_not_a_public_hf_loading_kwarg(monkeypatch, device):
    cfg = config(32, 8)
    monkeypatch.setattr(public.AutoConfig, "from_pretrained", lambda *a, **k: cfg)
    monkeypatch.setattr(public, "get_adapter", lambda arch: type("Adapter", (), {}))
    monkeypatch.setattr(runtime, "load_qwen3_vl_runtime_imagetext", lambda *a, **k: pytest.fail("weight load"))
    with pytest.raises(ValueError, match="unsupported.*_mmap_w8_weights"):
        public.RPUModelForConditionalGeneration.from_pretrained(
            "unused", device=device, _mmap_w8_weights=True,
        )
