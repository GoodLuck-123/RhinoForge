"""Host checks for the new precision recipe and bounded codec admission."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.qwen3_tts import codec_transformer as codec


def _core():
    config = SimpleNamespace(num_hidden_layers=8, hidden_size=512, intermediate_size=1024,
        num_attention_heads=16, num_key_value_heads=16, head_dim=64,
        sliding_window=72, hidden_act="silu", attention_bias=False)
    layers = [SimpleNamespace(
        self_attn=SimpleNamespace(q_norm=torch.nn.Identity(), k_norm=torch.nn.Identity(),
                                 o_proj=SimpleNamespace(bias=None)),
        mlp=SimpleNamespace(down_proj=SimpleNamespace(bias=None)),
        attention_type="sliding_attention",
        input_layernorm=SimpleNamespace(variance_epsilon=1e-5),
        post_attention_layernorm=SimpleNamespace(variance_epsilon=1e-5),
        self_attn_layer_scale=SimpleNamespace(scale=torch.ones(512)),
        mlp_layer_scale=SimpleNamespace(scale=torch.ones(512))) for _ in range(8)]
    return SimpleNamespace(config=config, layers=layers, norm=SimpleNamespace(variance_epsilon=1e-5),
        input_proj=SimpleNamespace(weight=SimpleNamespace(shape=(512, 1024))),
        output_proj=SimpleNamespace(weight=SimpleNamespace(shape=(1024, 512))),
        parameters=lambda: iter([torch.zeros(1)]))


@pytest.mark.parametrize("field,value", [
    ("num_hidden_layers", 7), ("hidden_size", 1024), ("intermediate_size", 2048),
    ("num_attention_heads", 8), ("num_key_value_heads", 8), ("head_dim", 128),
    ("sliding_window", 32), ("hidden_act", "gelu"), ("attention_bias", True),
])
def test_changed_geometry_rejected(field, value):
    core = _core()
    codec._validate_core(core)
    setattr(core.config, field, value)
    with pytest.raises(ValueError, match="geometry"):
        codec._validate_core(core)


def test_norm_bias_and_transformed_weight_rejection():
    for change in (
        lambda core: setattr(core.layers[0].self_attn, "k_norm", torch.nn.LayerNorm(64)),
        lambda core: setattr(core.layers[0].mlp.down_proj, "bias", torch.ones(512)),
        lambda core: setattr(core.layers[0].post_attention_layernorm, "variance_epsilon", 1e-6),
        lambda core: setattr(core, "parameters", lambda: iter([torch.ones(1).half()])),
    ):
        core = _core()
        change(core)
        with pytest.raises(ValueError):
            codec._validate_core(core)


def test_layerscale_fold_is_fp32_before_half_and_keeps_source():
    torch.manual_seed(9)
    linear = torch.nn.Linear(7, 5, bias=False)
    scale = torch.randn(5)
    weight_before, scale_before = linear.weight.clone(), scale.clone()
    actual = codec._folded_weight(linear, scale)
    assert torch.equal(actual, (weight_before * scale[:, None]).half())
    assert torch.equal(linear.weight, weight_before) and torch.equal(scale, scale_before)
    assert actual.dtype == torch.float16 and actual.data_ptr() != linear.weight.data_ptr()


@pytest.mark.parametrize("capability", [None, lambda: False, lambda: True])
def test_old_native_falls_back_without_allocation(monkeypatch, capability):
    native = SimpleNamespace()
    if capability is not None:
        native.causal_decoder_has_per_head_k_rope = capability
    monkeypatch.setattr(torch.ops, "rpu", native)
    owner = SimpleNamespace(_components={}, _restore=[])
    assert codec.install_codec_transformer(owner, _core()) is False
    assert owner._components == {} and owner._restore == []


@pytest.mark.parametrize("hidden", [None, torch.zeros(1, 0, 1024), torch.zeros(1, 64, 1024),
    torch.zeros(2, 3, 1024), torch.zeros(1, 3, 512), torch.zeros(1, 3, 1024).half(),
    torch.full((1, 3, 1024), float("nan"))])
def test_input_rejection_precedes_native_access(hidden):
    with pytest.raises(ValueError, match="1–63"):
        codec._forward({}, hidden)
