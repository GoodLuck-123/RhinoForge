from __future__ import annotations
import types
import pytest
import torch
import torch.nn as nn
from rpu_backend.adapters.qwen3_5 import text as qwen35_text
from rpu_backend.adapters.qwen3_5.quant_scope import FP16, FULL_ATTENTION, LINEAR_ATTENTION, W4A16_PGRP, W8A16, is_full_from_layer_types, resolve_quant_specs
from rpu_backend.api.errors import RPUBackendError

def _mixed_policy():
    return {'format_version': 2, 'architecture': 'qwen3_5', 'activation_dtype': 'float16', 'default': {'method': W4A16_PGRP, 'group_size': 32}, 'projection_overrides': {'linear_attn.in_proj_qkv': {'method': W8A16}, 'linear_attn.in_proj_z': {'method': W8A16}, 'linear_attn.out_proj': {'method': W8A16}, 'linear_attn.in_proj_b': {'method': FP16}, 'linear_attn.in_proj_a': {'method': FP16}}}

def _projection_tensors():
    is_full = is_full_from_layer_types([LINEAR_ATTENTION, FULL_ATTENTION])
    specs = resolve_quant_specs(is_full, _mixed_policy())
    projections = []
    for (name, spec) in specs.items():
        linear = nn.Linear(64, 64, bias=False).half()
        if spec.method != FP16:
            linear.weight = nn.Parameter(torch.randint(-8, 8, (64, 64), dtype=torch.int8), requires_grad=False)
            scale = torch.ones(64, dtype=torch.float16) if spec.method == W8A16 else torch.ones(2, 64, dtype=torch.float16)
            linear.register_buffer('weight_scale', scale)
        projections.append((linear, name))
    return (is_full, specs, projections)

def test_adapter_accepts_config_driven_mixed_projection_tensors(monkeypatch):
    (is_full, specs, projections) = _projection_tensors()
    monkeypatch.setattr(qwen35_text, '_iter_text_projections', lambda inner, layer_flags: iter(projections))
    resolved = qwen35_text._validate_projection_specs(object(), is_full, _mixed_policy())
    assert resolved == specs

def test_adapter_rejects_w4_group_size_mismatch(monkeypatch):
    (is_full, _, projections) = _projection_tensors()
    (linear, _) = next((item for item in projections if getattr(item[0], 'weight_scale', torch.empty(0)).dim() == 2))
    linear.weight_scale = torch.ones(1, 64, dtype=torch.float16)
    monkeypatch.setattr(qwen35_text, '_iter_text_projections', lambda inner, layer_flags: iter(projections))
    with pytest.raises(RPUBackendError, match='scale encodes group_size=64'):
        qwen35_text._validate_projection_specs(object(), is_full, _mixed_policy())

def _linear_with_scale():
    linear = nn.Linear(64, 64, bias=False).half()
    linear.register_buffer('weight_scale', torch.ones(64, dtype=torch.float16))
    return linear

def _norm():
    return types.SimpleNamespace(weight=nn.Parameter(torch.ones(64, dtype=torch.float16)))

def _mlp():
    return types.SimpleNamespace(gate_proj=_linear_with_scale(), up_proj=_linear_with_scale(), down_proj=_linear_with_scale())

def _full_layer():
    return types.SimpleNamespace(self_attn=types.SimpleNamespace(q_proj=_linear_with_scale(), k_proj=_linear_with_scale(), v_proj=_linear_with_scale(), o_proj=_linear_with_scale(), q_norm=_norm(), k_norm=_norm()), mlp=_mlp(), input_layernorm=_norm(), post_attention_layernorm=_norm())

def _gdn_layer():
    return types.SimpleNamespace(linear_attn=types.SimpleNamespace(in_proj_qkv=_linear_with_scale(), in_proj_z=_linear_with_scale(), in_proj_b=_linear_with_scale(), in_proj_a=_linear_with_scale(), out_proj=_linear_with_scale(), conv1d=nn.Conv1d(64, 64, 4, groups=64, bias=False).half(), norm=_norm(), A_log=nn.Parameter(torch.ones(64, dtype=torch.float16)), dt_bias=nn.Parameter(torch.ones(64, dtype=torch.float16))), mlp=_mlp(), input_layernorm=_norm(), post_attention_layernorm=_norm())

def _projection_modules(inner):
    (full, gdn) = inner.layers
    return [full.self_attn.q_proj, full.self_attn.k_proj, full.self_attn.v_proj, full.self_attn.o_proj, full.mlp.gate_proj, full.mlp.up_proj, full.mlp.down_proj, gdn.linear_attn.in_proj_qkv, gdn.linear_attn.in_proj_z, gdn.linear_attn.in_proj_b, gdn.linear_attn.in_proj_a, gdn.linear_attn.out_proj, gdn.linear_attn.conv1d, gdn.mlp.gate_proj, gdn.mlp.up_proj, gdn.mlp.down_proj]

def test_drop_raw_hf_text_weights_releases_complete_decoder_state(monkeypatch):
    monkeypatch.delenv('RPU_QWEN3_5_FREE_HF_WEIGHTS', raising=False)
    inner = types.SimpleNamespace(layers=[_full_layer(), _gdn_layer()], norm=_norm(), embed_tokens=nn.Embedding(64, 64).half())
    projections = _projection_modules(inner)
    direct_install_alias = inner.layers[1].linear_attn.norm.weight.data
    qwen35_text._drop_raw_hf_text_weights(inner, [1, 0])
    for module in projections:
        assert module.weight.numel() == 0
        scale = getattr(module, 'weight_scale', None)
        if scale is not None:
            assert scale.numel() == 0
    (full, gdn) = inner.layers
    for norm in (full.self_attn.q_norm, full.self_attn.k_norm, full.input_layernorm, full.post_attention_layernorm, gdn.linear_attn.norm, gdn.input_layernorm, gdn.post_attention_layernorm, inner.norm):
        assert norm.weight.numel() == 0
    assert gdn.linear_attn.A_log.numel() == 0
    assert gdn.linear_attn.dt_bias.numel() == 0
    assert inner.embed_tokens.weight.numel() > 0
    assert direct_install_alias.numel() > 0

def test_drop_raw_hf_text_layer_weights_is_incremental(monkeypatch):
    monkeypatch.delenv('RPU_QWEN3_5_FREE_HF_WEIGHTS', raising=False)
    full = _full_layer()
    later_gdn = _gdn_layer()
    qwen35_text._drop_raw_hf_text_layer_weights(full, True)
    assert all((module.weight.numel() == 0 for module in (full.self_attn.q_proj, full.self_attn.k_proj, full.self_attn.v_proj, full.self_attn.o_proj, full.mlp.gate_proj, full.mlp.up_proj, full.mlp.down_proj)))
    assert all((module.weight.numel() > 0 for module in (later_gdn.linear_attn.in_proj_qkv, later_gdn.linear_attn.in_proj_z, later_gdn.linear_attn.out_proj, later_gdn.mlp.gate_proj, later_gdn.mlp.up_proj, later_gdn.mlp.down_proj)))
