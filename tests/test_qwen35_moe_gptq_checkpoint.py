"""Official source isolation and code-preserving normalization, without hardware."""
import hashlib
import json
import os
import shutil

import pytest
from safetensors.torch import load_file, save_file
import torch

from rpu_backend.adapters.qwen3_5_moe.quant_scope import GPTQ_PROFILE_ID, RTN_PROFILE_ID
from rpu_backend.quant.grouped_uint4 import (
    pack_uint4_qweight, unpack_gptq_sym8_matrix,
)
from rpu_backend.quant.qwen3_5_moe_rtn import unpack_rtn_matrix
from test_qwen35_moe_rtn_weights import _checkpoint, _reader


def _quartet():
    codes = torch.arange(16, dtype=torch.uint8).repeat(16, 8)
    # Include subnormal, +0 and -0; the decoder must preserve stored scale bits.
    scales = torch.tensor([[0., -0., 2**-24, .25] * 4], dtype=torch.float16)
    return codes, {
        "qweight": pack_uint4_qweight(codes), "scales": scales,
        "qzeros": torch.full((1, 2), -2004318072, dtype=torch.int32),
        "g_idx": torch.zeros(128, dtype=torch.int32),
    }


def test_official_direct8_preserves_codes_scale_bits_and_effective_zero():
    expected, tensors = _quartet()
    codes, scales = unpack_gptq_sym8_matrix(**tensors, n=16, k=128)
    assert torch.equal(codes, expected)
    assert scales.data_ptr() == tensors["scales"].data_ptr()
    assert torch.equal(scales.view(torch.int16), tensors["scales"].view(torch.int16))
    dequant = (codes.float() - 8) * scales.t()
    assert dequant[3, :16].tolist() == [v * .25 for v in range(-8, 8)]
    with pytest.raises(ValueError, match="zero-minus-one"):
        unpack_rtn_matrix(**tensors, n=16, k=128)


@pytest.mark.parametrize("fault,match", [
    ("legacy_zero", "direct zero 8"), ("one_bad_zero", "direct zero 8"),
    ("act_order", "act-order"), ("negative_scale", "nonnegative"),
    ("nonfinite_scale", "finite"), ("wrong_dtype", "scales"),
])
def test_official_values_are_validated_before_runtime_pack(fault, match):
    _, tensors = _quartet()
    if fault == "legacy_zero":
        tensors["qzeros"].fill_(0x77777777)
    elif fault == "one_bad_zero":
        tensors["qzeros"][0, 1] += 1
    elif fault == "act_order":
        tensors["g_idx"][5] = 1
    elif fault == "negative_scale":
        tensors["scales"][0, 0] = -1
    elif fault == "nonfinite_scale":
        tensors["scales"][0, 0] = float("inf")
    elif fault == "wrong_dtype":
        tensors["scales"] = tensors["scales"].bfloat16()
    with pytest.raises(ValueError, match=match):
        unpack_gptq_sym8_matrix(**tensors, n=16, k=128)


@pytest.mark.parametrize("conflict", [
    "rtn_null", "fp8_null", "manifest", "rtn_shard", "explicit_format",
])
def test_official_source_refuses_rtn_residue_and_unreviewed_formats(tmp_path, conflict):
    _checkpoint(tmp_path, profile="gptq")
    path = tmp_path / "config.json"
    config = json.loads(path.read_text())
    if conflict == "rtn_null":
        config["rtn_metadata"] = None
    elif conflict == "fp8_null":
        config["quant_config"] = None
    elif conflict == "manifest":
        (tmp_path / "derivation_manifest.json").write_text("{}")
    elif conflict == "rtn_shard":
        shard = tmp_path / "part-7.safetensors"
        tensors = load_file(shard)
        save_file(tensors, shard, metadata={"quantization_algorithm": "RTN"})
    elif conflict == "explicit_format":
        config["quantization_config"]["checkpoint_format"] = "gptq"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="conflict|RTN|checkpoint_format"):
        _reader(tmp_path, "gptq")


def test_identity_is_path_independent_metadata_fingerprint_not_manifest_requirement(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _checkpoint(source, profile="gptq")
    checkpoint = _reader(source, "gptq")
    descriptor = checkpoint.identity_descriptor
    canonical = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    assert hashlib.sha256(canonical.encode()).hexdigest() == checkpoint.identity
    assert descriptor["profile_id"] == GPTQ_PROFILE_ID
    assert descriptor["identity_kind"] == "cold_metadata_fingerprint_not_payload_hash"
    assert str(source) not in canonical
    assert descriptor["config_sha256"] == hashlib.sha256(
        (source / "config.json").read_bytes()).hexdigest()
    assert descriptor["index_sha256"] == hashlib.sha256(
        (source / "model.safetensors.index.json").read_bytes()).hexdigest()
    assert [v["filename"] for v in descriptor["shards"]] == [
        "part-6.safetensors", "part-7.safetensors"]
    copied = tmp_path / "copied"
    shutil.copytree(source, copied)
    assert _reader(copied, "gptq").identity == checkpoint.identity
    (copied / "download_manifest.json").write_text("optional provenance, not parsed")
    assert _reader(copied, "gptq").identity == checkpoint.identity
    with (copied / "config.json").open("a") as stream:
        stream.write("\n")
    assert _reader(copied, "gptq").identity != checkpoint.identity


@pytest.mark.parametrize("profile", ["rtn", "gptq"])
def test_lazy_owner_retains_bound_source_profile(tmp_path, profile):
    _checkpoint(tmp_path, profile=profile)
    checkpoint = _reader(tmp_path, profile)
    source = checkpoint.layer_source(0)
    checkpoint.profile_id = RTN_PROFILE_ID if profile == "gptq" else GPTQ_PROFILE_ID
    with pytest.raises(ValueError, match="identity/profile"):
        source.prepare()


def test_rtn_manifest_identity_is_unchanged(tmp_path):
    _checkpoint(tmp_path)
    checkpoint = _reader(tmp_path)
    assert checkpoint.profile_id == RTN_PROFILE_ID
    assert checkpoint.identity_descriptor is None
    assert checkpoint.identity == hashlib.sha256(
        (tmp_path / "derivation_manifest.json").read_bytes()).hexdigest()


@pytest.mark.parametrize("profile", ["rtn", "gptq"])
def test_source_change_during_read_is_rejected_before_upload(tmp_path, profile):
    _checkpoint(tmp_path, profile=profile)
    checkpoint = _reader(tmp_path, profile)
    filename = next(iter(checkpoint.shards))
    shard = checkpoint.shards[filename]
    with pytest.raises(ValueError, match="changed after admission"):
        with checkpoint.open_shards([filename]):
            stat = shard.path.stat()
            os.utime(shard.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
