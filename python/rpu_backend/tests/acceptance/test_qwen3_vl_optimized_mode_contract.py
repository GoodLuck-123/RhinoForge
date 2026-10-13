from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from rpu_backend.adapters.qwen3_vl import text, vision

def test_qwen3_vl_adapter_has_no_optimized_away_assertions():
    for module in (text, vision):
        tree = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
        assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)], module.__file__

def test_qwen3_vl_rope_table_contracts_are_board_free():
    (vision_cos, vision_sin) = vision.build_vision_rope_tables(64, 48, device='cpu')
    assert tuple(vision_cos.shape) == (48, 16)
    assert tuple(vision_sin.shape) == (48, 16)
    config = SimpleNamespace(head_dim=128, rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0})
    (text_cos, text_sin) = text.build_mrope_cos_sin_tables(config, max_seq_len=32, device='cpu')
    assert tuple(text_cos.shape) == (32, 64)
    assert tuple(text_sin.shape) == (32, 64)
    assert text_cos.dtype is torch.float16
    assert text_sin.dtype is torch.float16

def test_qwen3_vl_qkv_weight_mismatch_fails_before_swizzle():
    block = SimpleNamespace(attn=SimpleNamespace(qkv=torch.nn.Linear(4, 11, bias=True)))
    with pytest.raises(ValueError, match='fused QKV weight.*\\(12, 4\\)'):
        vision._convert_vision_block_weights_for_rpu(block, num_heads=2, hidden_size=4)

def test_qwen3_vl_qkv_bias_mismatch_fails_before_swizzle():
    qkv = torch.nn.Linear(4, 12, bias=True)
    qkv.bias = torch.nn.Parameter(torch.zeros(11))
    block = SimpleNamespace(attn=SimpleNamespace(qkv=qkv))
    with pytest.raises(ValueError, match='fused QKV bias.*\\(12,\\)'):
        vision._convert_vision_block_weights_for_rpu(block, num_heads=2, hidden_size=4)

def test_qwen3_vl_fp16_conversion_casts_fp32_before_swizzle(monkeypatch):
    block = SimpleNamespace(attn=SimpleNamespace(qkv=torch.nn.Linear(8, 24, bias=True, dtype=torch.float32), proj=torch.nn.Linear(8, 8, bias=True, dtype=torch.float32)), mlp=SimpleNamespace(linear_fc1=torch.nn.Linear(8, 16, bias=True, dtype=torch.float32), linear_fc2=torch.nn.Linear(16, 8, bias=True, dtype=torch.float32)))
    seen_dtypes = []

    def record(weight, *args, **kwargs):
        seen_dtypes.append(weight.dtype)
        return weight.clone()
    monkeypatch.setattr(vision, 'tp_col_swizzle_mc_weight', record)
    monkeypatch.setattr(vision, 'tp_row_swizzle_mc_weight', record)
    vision._convert_vision_block_weights_for_rpu(block, num_heads=2, hidden_size=8)
    assert seen_dtypes == [torch.float16] * 6
    assert block._rpu_q_w.dtype is torch.float16
    assert block._rpu_k_w.dtype is torch.float16
    assert block._rpu_v_w.dtype is torch.float16
    assert block.attn.proj.weight.dtype is torch.float16
    assert block.mlp.linear_fc1.weight.dtype is torch.float16
    assert block.mlp.linear_fc2.weight.dtype is torch.float16
