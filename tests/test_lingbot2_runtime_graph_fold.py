"""Host regressions for the cold language graph weight transform."""

import copy
import importlib.util
from pathlib import Path

import pytest
import torch
from torch import nn


_PATH = Path(__file__).resolve().parents[1] / "python/rpu_backend/adapters/lingbot_vla_v2/runtime_graph.py"
_SPEC = importlib.util.spec_from_file_location("lingbot2_runtime_graph_host", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
fold = _MODULE.fold_lingbot_language_rmsnorm_


def model():
    root = nn.Module()
    root.layers = nn.ModuleList()
    for _ in range(2):
        layer = nn.Module()
        layer.input_layernorm = nn.RMSNorm(8, eps=1e-6)
        layer.post_attention_layernorm = nn.RMSNorm(8, eps=1e-6)
        layer.input_layernorm.weight.data.copy_(torch.linspace(.2, 1.7, 8))
        layer.post_attention_layernorm.weight.data.copy_(torch.linspace(1.4, -.3, 8))
        layer.self_attn = nn.Module()
        for name, width in (("q_proj", 16), ("k_proj", 4), ("v_proj", 4), ("o_proj", 8)):
            setattr(layer.self_attn, name, nn.Linear(8, width, bias=True))
        layer.self_attn.q_norm = nn.RMSNorm(4)
        layer.self_attn.k_norm = nn.RMSNorm(4)
        layer.mlp = nn.Module()
        layer.mlp.gate_proj = nn.Linear(8, 12, bias=True)
        layer.mlp.up_proj = nn.Linear(8, 12, bias=True)
        layer.mlp.down_proj = nn.Linear(12, 8, bias=True)
        root.layers.append(layer)
    return root


def test_fold_preserves_actual_norm_projection_and_mlp():
    torch.manual_seed(61)
    baseline = model()
    candidate = copy.deepcopy(baseline)
    fold(candidate)
    x = torch.randn(5, 8)
    for old, new in zip(baseline.layers, candidate.layers):
        for name in ("q_proj", "k_proj", "v_proj"):
            torch.testing.assert_close(
                getattr(new.self_attn, name)(new.input_layernorm(x)),
                getattr(old.self_attn, name)(old.input_layernorm(x)),
            )
        def mlp(layer):
            h = layer.post_attention_layernorm(x)
            return layer.mlp.down_proj(torch.nn.functional.silu(layer.mlp.gate_proj(h)) * layer.mlp.up_proj(h))
        torch.testing.assert_close(mlp(new), mlp(old))
        for name in ("self_attn.q_norm.weight", "self_attn.k_norm.weight",
                     "self_attn.o_proj.weight", "mlp.down_proj.weight"):
            torch.testing.assert_close(new.get_parameter(name), old.get_parameter(name), rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["half", "nan", "scale", "alias", "gamma_alias"])
def test_invalid_late_layer_does_not_mutate_earlier_layers(invalid):
    candidate = model()
    late = candidate.layers[-1]
    if invalid == "half":
        late.mlp.up_proj.half()
    elif invalid == "nan":
        late.post_attention_layernorm.weight.data[0] = float("nan")
    elif invalid == "scale":
        late.mlp.up_proj.register_buffer("weight_scale", torch.ones(12))
    elif invalid == "gamma_alias":
        late.post_attention_layernorm.weight = candidate.layers[0].input_layernorm.weight
    else:
        late.mlp.up_proj.weight = late.mlp.gate_proj.weight
    before = copy.deepcopy(candidate.state_dict())
    with pytest.raises(ValueError):
        fold(candidate)
    for name, weight in candidate.state_dict().items():
        torch.testing.assert_close(weight, before[name], rtol=0, atol=0, equal_nan=True)
    assert not getattr(candidate, "_lingbot_runtime_norm_folded", False)


def test_double_fold_and_installed_owner_are_rejected():
    candidate = model()
    fold(candidate)
    with pytest.raises(ValueError, match="already folded"):
        fold(candidate)
    candidate = model()
    candidate._rpu_decoder_handle = 7
    with pytest.raises(ValueError, match="precede native"):
        fold(candidate)
