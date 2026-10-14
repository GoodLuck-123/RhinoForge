"""Small real safetensors conversion and existing-loader checks; no model/device."""
import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from rpu_backend.quant import convert_qwen3_5_moe_rtn as converter
from rpu_backend.adapters.qwen3_5_moe.rtn_checkpoint import RtnCheckpoint
from rpu_backend.quant.qwen3_5_moe_rtn import quantize_rtn_group128


def source(tmp_path):
    directory = tmp_path / "parent"
    directory.mkdir()
    config = {"model_type": "qwen3_5_moe", "text_config": {
        "num_hidden_layers": 1, "num_experts": 2, "hidden_size": 128,
        "moe_intermediate_size": 128}}
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "generation_config.json").write_text('{"eos_token_id":[11,12]}')
    (directory / "tokenizer_config.json").write_text('{}')
    prefix = "model.language_model.layers.0.mlp.experts."
    tensors = {prefix + "gate_up_proj": torch.cat((
        torch.ones(2, 128, 128), torch.full((2, 128, 128), 2.)), 1).bfloat16(),
        prefix + "down_proj": torch.full((2, 128, 128), -3.).bfloat16(),
        "model.language_model.norm.weight": torch.ones(128).bfloat16(),
        "mtp.layers.0.mlp.experts.0.gate_proj.weight": torch.ones(8, 128).bfloat16()}
    files = [dict(list(tensors.items())[:1]), dict(list(tensors.items())[1:])]
    weight_map = {}
    for number, values in enumerate(files):
        filename = f"part-{number}.safetensors"
        save_file(values, directory / filename)
        weight_map.update({key: filename for key in values})
    (directory / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return directory, tensors


def test_sharded_converter_preserves_split_regular_bytes_and_loader_contract(tmp_path, monkeypatch):
    src, original = source(tmp_path)
    # Reduce geometry only for serialization: production still calls exact admission.
    monkeypatch.setattr(converter, "validate_exact_profile", lambda *a, **kw: None)
    dst = tmp_path / "output"
    manifest = converter.convert_checkpoint(src, dst)
    reader = RtnCheckpoint(dst, num_layers=1, num_experts=2, hidden_size=128, intermediate_size=128)
    assert manifest["completed"] is True and manifest["quantized_matrices"] == 6
    assert len(reader.tensor_shards) == 26
    tensors = {}
    for shard in reader.shards:
        tensors.update(load_file(dst / shard))
        row = manifest["output_files"][shard]
        assert hashlib.sha256((dst / shard).read_bytes()).hexdigest() == row["sha256"]
    prefix = "model.language_model.layers.0.mlp.experts."
    for expert in range(2):
        for role, weight in (("gate_proj", original[prefix + "gate_up_proj"][expert, :128]),
                             ("up_proj", original[prefix + "gate_up_proj"][expert, 128:]),
                             ("down_proj", original[prefix + "down_proj"][expert])):
            expected = quantize_rtn_group128(weight)
            for suffix, value in expected.items():
                assert torch.equal(tensors[f"{prefix}{expert}.{role}.{suffix}"], value)
    for key in original:
        if not key.startswith(prefix):
            assert tensors[key].dtype == original[key].dtype
            assert torch.equal(tensors[key], original[key])
    assert (dst / "generation_config.json").read_bytes() == (src / "generation_config.json").read_bytes()
    with pytest.raises(FileExistsError):
        converter.convert_checkpoint(src, dst)


@pytest.mark.parametrize("fault,match", [("shape", "expected"), ("nan", "nonfinite"),
    ("missing", "complete fused"), ("quantized", "unquantized"),
    ("index", "inventory mismatch")])
def test_bad_sources_fail_without_publishing_output(tmp_path, monkeypatch, fault, match):
    src, tensors = source(tmp_path)
    monkeypatch.setattr(converter, "validate_exact_profile", lambda *a, **kw: None)
    path = src / "part-0.safetensors"
    name = next(iter(tensors))
    if fault in ("shape", "nan"):
        value = tensors[name].clone()
        if fault == "shape":
            value = value[:, :-1]
        else:
            value[0, 0, 0] = float("nan")
        save_file({name: value.contiguous()}, path)
    elif fault == "quantized":
        config = json.loads((src / "config.json").read_text())
        config["quantization_config"] = None
        (src / "config.json").write_text(json.dumps(config))
    else:
        index = json.loads((src / "model.safetensors.index.json").read_text())
        index["weight_map"].pop(name if fault == "missing" else "model.language_model.norm.weight")
        (src / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match=match):
        converter.convert_checkpoint(src, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output.partial-*"))


def test_cli_geometry_gate_rejects_small_or_wrong_model_before_output(tmp_path):
    src, _ = source(tmp_path)
    with pytest.raises(ValueError, match="exact 35B-A3B"):
        converter.convert_checkpoint(src, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_required_generation_sidecar_fails_before_conversion(tmp_path):
    src, _ = source(tmp_path)
    (src / "generation_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="generation_config"):
        converter.convert_checkpoint(src, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_changed_parent_is_not_published_with_old_sha(tmp_path, monkeypatch):
    src, tensors = source(tmp_path)
    monkeypatch.setattr(converter, "validate_exact_profile", lambda *a, **kw: None)
    save = converter.save_file

    def mutate_after_output(values, path, **kwargs):
        save(values, path, **kwargs)
        if path.name == "part-0.safetensors":
            key = next(iter(tensors))
            save({key: tensors[key] + 1}, src / path.name)

    monkeypatch.setattr(converter, "save_file", mutate_after_output)
    with pytest.raises(ValueError, match="source changed during conversion"):
        converter.convert_checkpoint(src, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".output.partial-*"))
