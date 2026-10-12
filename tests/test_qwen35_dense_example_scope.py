"""Dense example labels bind the checkpoint size and uniform projection recipe."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]


def metadata(size, precision):
    layers, hidden, intermediate, heads, kv_heads = {
        "0_8b": (24, 1024, 3584, 8, 2),
        "2b": (24, 2048, 6144, 8, 2),
        "4b": (32, 2560, 9216, 16, 4),
        "9b": (32, 4096, 12288, 16, 4),
    }[size]
    result = {
        "model_type": "qwen3_5",
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "text_config": {
            "num_hidden_layers": layers, "hidden_size": hidden,
            "intermediate_size": intermediate, "num_attention_heads": heads,
            "num_key_value_heads": kv_heads, "head_dim": 256,
            "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * (layers // 4),
        },
    }
    if precision != "fp16":
        result["quant_config"] = {
            "format_version": 1, "architecture": "qwen3_5",
            "method": "w8a16" if precision == "w8a16" else "w4a16_pgrp",
            "storage": "int8", "scale_dtype": "float16",
        }
        if precision == "w4a16":
            result["quant_config"]["group_size"] = 32
    return result


def config(size, precision, component="text"):
    from examples.common.config import load_config

    return load_config(ROOT / f"examples/configs/qwen3_5/{component}/{size}/{precision}.toml")


def write_metadata(tmp_path, value):
    (tmp_path / "config.json").write_text(json.dumps(value))
    return str(tmp_path)


@pytest.mark.parametrize("size", ["0_8b", "2b", "4b", "9b"])
@pytest.mark.parametrize("precision", ["fp16", "w8a16", "w4a16"])
def test_matching_dense_checkpoint_is_accepted(tmp_path, size, precision):
    from examples.common.config import validate_checkpoint_profile

    validate_checkpoint_profile(config(size, precision), write_metadata(tmp_path, metadata(size, precision)))


@pytest.mark.parametrize("component", ["text", "vl"])
def test_uniform_v2_semantics_are_accepted(tmp_path, component):
    from examples.common.config import validate_checkpoint_profile

    value = metadata("2b", "w4a16")
    value["quant_config"] = {"format_version": 2, "default": {"method": "w4a16_pgrp", "group_size": 32}}
    validate_checkpoint_profile(config("2b", "w4a16", component), write_metadata(tmp_path, value))


@pytest.mark.parametrize("mutation", ["floating", "other_size", "hybrid", "wrong_precision", "group64", "nested_quant", "truncated_layers"])
def test_another_recipe_cannot_inherit_the_uniform_label(tmp_path, mutation):
    from examples.common.config import ConfigError, validate_checkpoint_profile

    value = metadata("2b", "w4a16")
    if mutation == "floating":
        value.pop("quant_config")
    elif mutation == "other_size":
        value = metadata("4b", "w4a16")
    elif mutation == "hybrid":
        value["quant_config"] = {
            "format_version": 2, "profile": "w4_main_w8_gdn",
            "default": {"method": "w4a16_pgrp", "group_size": 32},
            "projection_overrides": {
                "linear_attn.in_proj_a": {"method": "fp16"},
                "linear_attn.in_proj_b": {"method": "fp16"},
                "linear_attn.in_proj_qkv": {"method": "w8a16"},
                "linear_attn.in_proj_z": {"method": "w8a16"},
                "linear_attn.out_proj": {"method": "w8a16"},
            },
        }
    elif mutation == "wrong_precision":
        value = metadata("2b", "w8a16")
    elif mutation == "group64":
        value["quant_config"]["group_size"] = 64
    elif mutation == "nested_quant":
        value["text_config"]["quantization_config"] = {"quant_method": "awq"}
    else:
        value["text_config"]["layer_types"].pop()
    for component in ("text", "vl"):
        with pytest.raises(ConfigError, match="rejects checkpoint"):
            validate_checkpoint_profile(config("2b", "w4a16", component), write_metadata(tmp_path, value))


def test_fp16_rejects_quantized_checkpoint(tmp_path):
    from examples.common.config import ConfigError, validate_checkpoint_profile

    with pytest.raises(ConfigError, match="rejects checkpoint"):
        validate_checkpoint_profile(config("2b", "fp16"), write_metadata(tmp_path, metadata("2b", "w8a16")))


def test_wrong_recipe_rejects_before_tokenizer_and_weights(tmp_path, monkeypatch):
    from examples.common.config import ConfigError
    from examples.common.text import text_inputs
    import transformers

    tokenizer = Mock(side_effect=AssertionError("tokenizer must not load"))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    with pytest.raises(ConfigError, match="rejects checkpoint"):
        text_inputs(config("2b", "w8a16"), write_metadata(tmp_path, metadata("2b", "fp16")), 4)
    tokenizer.assert_not_called()
