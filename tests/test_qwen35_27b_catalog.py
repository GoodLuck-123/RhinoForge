"""Official dense27B configs bind exact precision/core identity without hardware."""
import importlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "examples/configs/qwen3_5"
CASES = [(component, bits, cores) for component in ("text", "vl")
         for bits, cores in ((8, 8), (8, 4), (8, 6), (4, 8))]


@pytest.fixture
def common(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module("examples.common.config")


def _path(component, bits, cores):
    suffix = "" if cores == 8 else f"_tp{cores}"
    return FAMILY / f"{component}/27b/w{bits}a16{suffix}.toml"


@pytest.fixture
def metadata():
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5", "tie_word_embeddings": False,
        "image_token_id": 248056,
        "text_config": {
            "model_type": "qwen3_5_text", "hidden_size": 5120,
            "intermediate_size": 17408, "num_hidden_layers": 64,
            "num_attention_heads": 24, "num_key_value_heads": 4, "head_dim": 256,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "vocab_size": 248320,
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "hidden_act": "silu", "attention_bias": False, "attn_output_gate": True,
            "rms_norm_eps": 1e-6, "use_cache": True, "full_attention_interval": 4,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                "partial_rotary_factor": 0.25, "mrope_interleaved": True,
                "mrope_section": [11, 11, 10]},
        },
        "vision_config": {"depth": 27, "num_heads": 16, "hidden_size": 1152,
            "intermediate_size": 4304, "out_hidden_size": 5120,
            "in_channels": 3, "patch_size": 16, "temporal_patch_size": 2,
            "spatial_merge_size": 2, "num_position_embeddings": 2304,
            "hidden_act": "gelu_pytorch_tanh", "deepstack_visual_indexes": []},
    }


@pytest.mark.parametrize("component,bits,cores", CASES)
def test_official_27b_profiles_match_exact_config_topology(common, component, bits, cores, tmp_path):
    path = _path(component, bits, cores)
    config = common.load_config(path)
    profile = common.planner_profiles()[config["example"]["profile_id"]]
    assert profile["asset_alias"] == f"qwen3_5-27b-w{bits}a16"
    assert profile["input_envelope"]["limits"]["num_cores"] == cores
    assert profile["hw_status"] == "IMPLEMENTED_UNVERIFIED_HW"
    assert profile["support"]["status"] == "CONTROLLED_EVALUATION"
    assert config["rpu_execution"]["model"]["num_cores"] == cores
    assert config["rpu_execution"]["prefill"]["linear_acc32"] is False
    assert config["run"]["warmup"] == 2 and config["run"]["runs"] == 5
    bad_core = 4 if cores == 8 else 8
    bad = tmp_path / "wrong_core.toml"
    bad.write_text(path.read_text().replace(f"num_cores = {cores}", f"num_cores = {bad_core}"))
    with pytest.raises(common.ConfigError, match=r"model\.num_cores"):
        common.load_config(bad)
    for gate in config["example"]["opt_in"]:
        bad.write_text(path.read_text().replace(f'{gate} = "1"\n', ""))
        with pytest.raises(common.ConfigError, match="exact opt_in"):
            common.load_config(bad)


def test_all_27b_runtime_templates_load_with_explicit_core_and_precision(common):
    paths = sorted((FAMILY / "text/27b").glob("*.toml")) + sorted((FAMILY / "vl/27b").glob("*.toml"))
    assert len(paths) >= len(CASES)
    for path in paths:
        config = common.load_config(path)
        assert config["rpu_execution"]["model"]["num_cores"] in (4, 6, 8)
        assert isinstance(config["rpu_execution"]["prefill"]["linear_acc32"], bool)
        assert config["input"]["decode_steps"] > 0
        assert "fp16" not in path.stem


@pytest.mark.parametrize("component,bits,version", [(component, bits, version)
    for component in ("text", "vl") for bits in (4, 8) for version in (1, 2)])
def test_checkpoint_preflight_accepts_equivalent_uniform_quant_semantics(common, metadata, component, bits, version, tmp_path):
    metadata["text_config"]["layer_types"] *= 16
    method = "w8a16" if bits == 8 else "w4a16_pgrp"
    spec = {"method": method, **({"group_size": 32} if bits == 4 else {})}
    metadata["quant_config"] = ({"format_version": 2, "default": spec}
                                if version == 2 else {"format_version": 1, **spec})
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    common.validate_checkpoint_profile(common.load_config(_path(component, bits, 8)), str(tmp_path))


@pytest.mark.parametrize("mutation", ["other_size", "legacy", "wrong_precision", "missing_quant", "wrong_architecture"])
def test_checkpoint_preflight_rejects_relabelling(common, metadata, mutation, tmp_path):
    metadata["text_config"]["layer_types"] *= 16
    scope = common._backend_leaf("adapters/qwen3_5/quant_scope.py")
    metadata["quant_config"] = {"method": "w8a16"}
    if mutation == "other_size":
        metadata["text_config"]["hidden_size"] = 4096
    elif mutation == "legacy":
        metadata["quant_config"] = scope.legacy_w8a16_quant_config()
    elif mutation == "wrong_precision":
        metadata["quant_config"] = {"method": "w4a16_pgrp", "group_size": 32}
    elif mutation == "missing_quant":
        del metadata["quant_config"]
    else:
        metadata["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    (tmp_path / "config.json").write_text(json.dumps(metadata))
    for component in ("text", "vl"):
        with pytest.raises(common.ConfigError, match="rejects checkpoint"):
            common.validate_checkpoint_profile(common.load_config(_path(component, 8, 8)), str(tmp_path))


def test_official_assets_do_not_replace_archived_legacy(common):
    profiles = common.planner_profiles()
    aliases = common._backend_leaf("model_registry.py").MODELS
    assert aliases["qwen3_8-27b-w8a16"] == "Qwen3.8-27B-w8a16"
    assert aliases["qwen3_5-27b-w8a16"] != aliases["qwen3_8-27b-w8a16"]
    assert aliases["qwen3_5-27b-w4a16"] != aliases["qwen3_5-27b-w8a16"]
    for suffix in ("", "_tp4", "_tp6"):
        config = common.load_config(FAMILY / f"legacy/27b/w8a16{suffix}.toml")
        assert profiles[config["example"]["profile_id"]]["asset_alias"] == "qwen3_8-27b-w8a16"
