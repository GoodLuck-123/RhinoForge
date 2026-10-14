"""Tensor-contract tests for the Qwen3.5 mixed checkpoint loader."""
from __future__ import annotations

import json

import torch
import torch.nn as nn
import pytest
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, Qwen3_5Config

from rpu_backend.adapters.qwen3_5.loader import (
    _pop_projection_tensors, load_qwen3_5_model,
)
from rpu_backend.adapters.qwen3_5.quant_scope import (
    FP16, W4A16_PGRP, W8A16, QuantSpec, scale_key, resolve_quant_specs,
)
from rpu_backend.api.errors import RPUBackendError
from rpu_backend.quant.convert_qwen3_5 import convert_checkpoint


PREFIX = "model.language_model.layers.0."


def _case():
    expected = {
        PREFIX + "linear_attn.in_proj_b.weight": QuantSpec(FP16),
        PREFIX + "linear_attn.in_proj_qkv.weight": QuantSpec(W8A16),
        PREFIX + "mlp.gate_proj.weight": QuantSpec(W4A16_PGRP, 32),
    }
    modules = {
        key.removesuffix(".weight"): nn.Linear(64, 64, bias=False)
        for key in expected
    }
    state = {
        next(key for key, spec in expected.items() if spec.method == FP16):
            torch.randn(64, 64, dtype=torch.float16),
        next(key for key, spec in expected.items() if spec.method == W8A16):
            torch.randint(-127, 128, (64, 64), dtype=torch.int8),
        next(key for key, spec in expected.items() if spec.method == W4A16_PGRP):
            torch.randint(-8, 8, (64, 64), dtype=torch.int8),
    }
    for key, spec in expected.items():
        if spec.method == W8A16:
            state[scale_key(key)] = torch.ones(64, dtype=torch.float16)
        elif spec.method == W4A16_PGRP:
            state[scale_key(key)] = torch.ones(2, 64, dtype=torch.float16)
    return state, expected, modules


def test_mixed_loader_consumes_only_quantized_tensors():
    state, expected, modules = _case()
    fp16_key = next(key for key, spec in expected.items() if spec.method == FP16)

    consumed, seen = _pop_projection_tensors(state, expected, modules)

    assert seen == set(expected)
    assert fp16_key in state
    assert set(consumed) == {
        key.removesuffix(".weight")
        for key, spec in expected.items() if spec.method != FP16
    }
    assert not any(key.endswith(".weight_scale") for key in state)


@pytest.mark.parametrize("failure", ["missing_w8_scale", "fp16_scale", "bad_w4_scale"])
def test_mixed_loader_rejects_tensor_contract_mismatch(failure):
    state, expected, modules = _case()
    key = next(
        key for key, spec in expected.items()
        if ((failure == "missing_w8_scale" and spec.method == W8A16)
            or (failure == "fp16_scale" and spec.method == FP16)
            or (failure == "bad_w4_scale" and spec.method == W4A16_PGRP))
    )
    if failure == "missing_w8_scale":
        state.pop(scale_key(key))
    elif failure == "fp16_scale":
        state[scale_key(key)] = torch.ones(64, dtype=torch.float16)
    else:
        state[scale_key(key)] = torch.ones(64, dtype=torch.float16)

    with pytest.raises(RPUBackendError):
        _pop_projection_tensors(state, expected, modules)


@pytest.mark.parametrize("bad_code", [-9, 8])
def test_mixed_loader_rejects_out_of_range_w4_codes(bad_code):
    state, expected, modules = _case()
    key = next(
        key for key, spec in expected.items() if spec.method == W4A16_PGRP)
    state[key].flatten()[0] = bad_code

    with pytest.raises(RPUBackendError, match=r"codes must be in \[-8, 7\]"):
        _pop_projection_tensors(state, expected, modules)


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_mixed_loader_rejects_nonpositive_or_nonfinite_scale(bad_scale):
    state, expected, modules = _case()
    key = next(key for key, spec in expected.items() if spec.method == W8A16)
    state[scale_key(key)].flatten()[0] = bad_scale

    with pytest.raises(RPUBackendError, match="finite, positive"):
        _pop_projection_tensors(state, expected, modules)


@pytest.mark.parametrize("source_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("profile", [None, "w8_main_w8_gdn"])
def test_w8_conversion_and_loading_preserve_source_precision(tmp_path, source_dtype, profile):
    """Exercise actual HF skeleton/loading with BF16-pinned composite metadata."""
    torch.manual_seed(5)
    config = Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=dict(
            dtype="bfloat16", vocab_size=32, hidden_size=64, intermediate_size=64,
            num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
            num_attention_heads=2, num_key_value_heads=1, head_dim=32,
            linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=32, linear_value_head_dim=32,
        ),
        vision_config=dict(
            dtype="bfloat16", depth=1, hidden_size=64, intermediate_size=64,
            num_heads=2, out_hidden_size=64, num_position_embeddings=16,
        ),
    )
    model = AutoModelForImageTextToText.from_config(config)
    source = {name: value.detach().to(source_dtype).contiguous()
              for name, value in model.state_dict().items()}
    projection_keys = {
        f"model.language_model.{name}.weight"
        for name in resolve_quant_specs([False, True])
    }
    # With scale=1, direct FP16 quantizes 1.4990234375 to 1. A BF16
    # intermediate rounds it to 1.5, changing the INT8 code to 2.
    for name in projection_keys:
        source[name].fill_(1.4990234375)
        source[name][:, 0] = 127
    norm_key = PREFIX + "linear_attn.norm.weight"
    source[norm_key] = torch.full_like(source[norm_key], 0.955810546875,
                                       dtype=torch.float32)
    assert not torch.equal(source[norm_key].half(), source[norm_key].bfloat16().half())
    src, dst = tmp_path / "source", tmp_path / "w8"
    src.mkdir()
    config.save_pretrained(src)
    save_file(source, src / "model.safetensors", metadata={"format": "pt"})
    source_config = (src / "config.json").read_bytes()

    convert_checkpoint(src, dst, profile=profile)
    converted = load_file(dst / "model.safetensors")
    loaded = load_qwen3_5_model(dst)
    actual = loaded.state_dict()
    quant = json.loads((dst / "config.json").read_text())["quant_config"]
    specs = resolve_quant_specs([False, True], quant)
    quantized = {f"model.language_model.{name}.weight"
                 for name, spec in specs.items() if spec.method == W8A16}
    for name, original in source.items():
        if name in quantized:
            expected = torch.full_like(original, 2 if source_dtype == torch.bfloat16 else 1,
                                       dtype=torch.int8)
            expected[:, 0] = 127
            scale = torch.ones(original.shape[0], dtype=torch.float16)
            assert torch.equal(converted[scale_key(name)], scale), name
            assert torch.equal(actual[scale_key(name)], scale), name
        else:
            expected = original.half()
        assert converted[name].dtype == actual[name].dtype == expected.dtype, name
        assert torch.equal(converted[name], expected), name
        assert torch.equal(actual[name], expected), name
    assert (src / "config.json").read_bytes() == source_config
    assert not any(t.is_meta for t in (*loaded.parameters(), *loaded.buffers()))
