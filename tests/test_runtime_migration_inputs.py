"""Asset identity, quantization input and cached-mask regression checks."""
import hashlib
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.rhinovla import factory
from rpu_backend.adapters.rhinovla.fused import _cached_rhino_attention_bias
from rpu_backend.quant.int4_pgrp_pack import quantize_int4_group_wise


@pytest.mark.parametrize("group", [0, -1, 16, 256, True, 32.0])
def test_int4_rejects_invalid_group_before_quantization(group):
    with pytest.raises(ValueError, match="group_size"):
        quantize_int4_group_wise(torch.ones(4, 128), group)


@pytest.mark.parametrize("value", [torch.ones(4, 128, dtype=torch.int8),
                                  torch.full((4, 128), float("nan")),
                                  torch.full((4, 128), float("inf"))])
def test_int4_rejects_nonfinite_or_integer_weights(value):
    with pytest.raises(ValueError, match="finite floating"):
        quantize_int4_group_wise(value, 32)


@pytest.mark.parametrize("group", [32, 64, 128])
def test_int4_accepted_groups_keep_expected_layout(group):
    codes, scales, _ = quantize_int4_group_wise(torch.arange(512).reshape(4, 128).float(), group)
    assert codes.shape == (4, 128) and codes.dtype == torch.int8
    assert scales.shape == (128 // group, 4)
    assert bool(torch.isfinite(scales).all())


def test_v3_asset_identity_does_not_relax_original_profile(monkeypatch, tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"fixture")
    source = tmp_path / "source"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(factory.subprocess, "check_output", lambda args, **kwargs:
                        "new-revision\n" if "rev-parse" in args else "")
    factory._validate_official_assets(source, checkpoint, source_sha="new-revision",
                                     checkpoint_sha256=digest)
    with pytest.raises(ValueError, match="clean reviewed"):
        factory._validate_official_assets(source, checkpoint)
    with pytest.raises(ValueError, match="SHA-256"):
        factory._validate_official_assets(source, checkpoint, source_sha="new-revision",
                                         checkpoint_sha256="different")
    monkeypatch.setattr(factory.subprocess, "check_output", lambda args, **kwargs:
                        "new-revision\n" if "rev-parse" in args else "?? extra.py")
    with pytest.raises(ValueError, match="clean reviewed"):
        factory._validate_official_assets(source, checkpoint, source_sha="new-revision",
                                         checkpoint_sha256=digest)


def test_factory_entrypoints_preserve_separate_asset_selection(monkeypatch):
    calls = []
    monkeypatch.setattr(factory, "_runtime_factory", lambda config, **kwargs:
                        calls.append((config, kwargs)))
    factory.official_runtime_factory({"test": 1}, rpu_execution={"test": 2})
    factory.v3_runtime_factory({"test": 1}, rpu_execution={"test": 2})
    assert calls == [({"test": 1}, {"rpu_execution": {"test": 2}}),
                     ({"test": 1}, {"rpu_execution": {"test": 2}, "v3": True})]
    assert factory.OFFICIAL_SOURCE_SHA != factory.V3_SOURCE_SHA
    assert factory.OFFICIAL_CHECKPOINT_SHA256 != factory.V3_CHECKPOINT_SHA256


def test_v3_factory_loads_tensor_inputs_from_file_before_assets(monkeypatch, tmp_path):
    path = tmp_path / "qin.pt"
    torch.save({"input_ids": torch.ones(1, 2, dtype=torch.long),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
                "pixel_values": torch.ones(1, 4),
                "image_grid_thw": torch.tensor([[1, 1, 1]])}, path)
    load = torch.load
    calls = []
    def record(*args, **kwargs):
        calls.append(kwargs)
        return load(*args, **kwargs)
    monkeypatch.setattr(torch, "load", record)
    monkeypatch.setattr(factory, "_reject_unsafe_diagnostics", lambda: None)
    def stop_before_assets(*args, **kwargs):
        raise RuntimeError("asset boundary reached")
    monkeypatch.setattr(factory, "_validate_official_assets", stop_before_assets)
    with pytest.raises(RuntimeError, match="asset boundary reached"):
        factory.v3_runtime_factory({"qin": str(path), "model_root": str(tmp_path),
                                    "checkpoint": str(path), "trust_model_code": True})
    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_shared_attention_mask_preserves_padding_and_mutated_inputs():
    calls = []
    def make(prefix, suffix):
        calls.append(True)
        return torch.cat((prefix, suffix), dim=1)[:, None].expand(-1, suffix.shape[1], -1)
    def bias(mask, dtype):
        return torch.where(mask, torch.tensor(0, dtype=dtype),
                           torch.tensor(torch.finfo(dtype).min, dtype=dtype))[:, None]
    owner = SimpleNamespace()
    with torch.inference_mode():
        prefix = torch.tensor([[True, True, False]])
        suffix = torch.tensor([[True, False]])
        first = _cached_rhino_attention_bias(owner, prefix, suffix, make, bias)
        assert _cached_rhino_attention_bias(owner, prefix.clone(), suffix.clone(), make, bias) is first
        assert first.shape == (1, 1, 2, 16) and not first.is_inference()
        assert bool((first[..., 4:] < 0).all())
        prefix[0, 1] = False
        second = _cached_rhino_attention_bias(owner, prefix, suffix, make, bias)
        assert second is not first and len(calls) == 2
        assert bool((second[..., 1] < 0).all()) and bool((first[..., 1] == 0).all())


@pytest.mark.parametrize("logical,stage,chunk,padding", [
    (1904, {}, 176, 32), (100, {}, None, "auto"),
    (1904, {"chunk_size": 160, "padding_rows": 0}, 160, 0),
])
def test_internvla_default_prefill_keeps_explicit_overrides(monkeypatch, logical, stage, chunk, padding):
    from rpu_backend.adapters.internvla_n1 import backbone
    captured = {}
    def plan(*args, **kwargs):
        captured.update(kwargs)
        kwargs["plan_result_sink"](SimpleNamespace(selected=SimpleNamespace(
            stage_tuple=SimpleNamespace(physical_descriptor=(1,)))))
        return 1936, chunk or 16
    monkeypatch.setattr(backbone, "plan_bounded_prefill_execution", plan)
    owner = SimpleNamespace(_handle=1, cache=SimpleNamespace(max_seq_len=4096),
                            _rpu_execution={"prefill": stage}, _graph_cache=None)
    backbone.Qwen25VLBackbone._prefill_execution_plan(owner, logical)
    assert captured["exact_chunk_size"] == chunk
    assert captured["padding_rows"] == padding


def test_pi05_probe_requires_explicit_directory_and_saves_tensor(monkeypatch, tmp_path):
    from rpu_backend.adapters.pi05.runtime import _pi05_probe_save
    monkeypatch.delenv("RPU_PI05_PROBE_DIR", raising=False)
    _pi05_probe_save("input", torch.ones(2, dtype=torch.float16))
    assert not list(tmp_path.iterdir())
    monkeypatch.setenv("RPU_PI05_PROBE_DIR", str(tmp_path))
    value = torch.arange(4, dtype=torch.float16)
    _pi05_probe_save("input", value)
    value.zero_()
    saved = torch.load(tmp_path / "input.pt", weights_only=True)
    assert saved.dtype == torch.float32 and torch.equal(saved, torch.arange(4).float())
