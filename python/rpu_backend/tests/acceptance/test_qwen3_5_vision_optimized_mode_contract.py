from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from rpu_backend.adapters.qwen3_5 import vision

def test_qwen3_5_vision_has_no_optimized_away_assertions():
    tree = ast.parse(Path(vision.__file__).read_text(encoding='utf-8'))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]

def test_qwen3_5_vision_rope_table_contract_is_board_free():
    (cos, sin) = vision.build_vision_rope_tables(64, 48, device='cpu')
    assert tuple(cos.shape) == (48, 16)
    assert tuple(sin.shape) == (48, 16)
    assert cos.dtype is torch.float16
    assert sin.dtype is torch.float16

@pytest.mark.parametrize('num_heads, expected_tp', [(12, 4), (16, 8)])
def test_qwen3_5_vision_attention_tp_is_shape_derived(num_heads, expected_tp):
    assert vision._resolve_qwen3_5_vision_attention_tp(num_heads) == expected_tp

def test_qwen3_5_vision_admission_separates_runtime_and_candidate_profiles():
    assert vision._QWEN3_5_VISION_PROFILES == {(24, 16, 1024, 4096)}
    assert vision._QWEN3_5_VISION_CANDIDATE_PROFILES == {(12, 12, 768, 3072), (27, 16, 1152, 4304)}

@pytest.mark.parametrize('real, padded', [(64, 64), (72, 80)])
def test_qwen3_5_vision_head_dim_layout_alignment(real, padded):
    assert vision._qwen3_5_vision_padded_head_dim(real) == padded

@pytest.mark.parametrize('real, padded', [(3072, 3072), (4096, 4096), (4304, 4352)])
def test_qwen3_5_vision_mlp_layout_alignment(real, padded):
    assert vision._qwen3_5_vision_padded_intermediate_size(real) == padded

@pytest.mark.parametrize('num_layers, num_heads, padded_head_dim, expected_head_vx', [(12, 12, 64, 3), (24, 16, 64, 2), (27, 16, 80, 2)])
def test_qwen3_5_vision_cache_layout_matches_attention_tp(num_layers, num_heads, padded_head_dim, expected_head_vx):
    cache = vision._make_dummy_vision_kv_caches(num_layers=num_layers, max_seq_len=16, num_heads=num_heads, head_dim=padded_head_dim, device='cpu')
    assert len(cache.k_caches) == num_layers
    assert cache.nKVHeadChunk == 8
    assert cache.nKVHeadVx == expected_head_vx
    assert tuple(cache.k_caches[0].shape) == (1, 1, expected_head_vx, padded_head_dim // 16, 8, 16, 16)

def test_qwen3_5_vision_conversion_uses_tp4_only_for_attention(monkeypatch):
    hidden_size = 12
    block = SimpleNamespace(attn=SimpleNamespace(qkv=torch.nn.Linear(hidden_size, 3 * hidden_size, bias=True), proj=torch.nn.Linear(hidden_size, hidden_size, bias=True)), mlp=SimpleNamespace(linear_fc1=torch.nn.Linear(hidden_size, 24, bias=True), linear_fc2=torch.nn.Linear(24, hidden_size, bias=True)))
    calls = []

    def col(weight, num_cores=8):
        calls.append(('col', num_cores, tuple(weight.shape)))
        return weight.clone()

    def row(weight, num_cores=8):
        calls.append(('row', num_cores, tuple(weight.shape)))
        return weight.clone()
    monkeypatch.setattr(vision, 'tp_col_swizzle_mc_weight', col)
    monkeypatch.setattr(vision, 'tp_row_swizzle_mc_weight', row)
    vision._convert_vision_block_weights_for_rpu(block, num_heads=12, hidden_size=hidden_size)
    assert [core_count for (kind, core_count, _) in calls if kind == 'col'] == [4, 4, 4, 8]
    assert [core_count for (kind, core_count, _) in calls if kind == 'row'] == [4, 8]

def test_qwen3_5_vision_qkv_weight_mismatch_fails_before_swizzle():
    block = SimpleNamespace(attn=SimpleNamespace(qkv=torch.nn.Linear(4, 11, bias=True)))
    with pytest.raises(ValueError, match='fused QKV weight.*\\(12, 4\\)'):
        vision._convert_vision_block_weights_for_rpu(block, num_heads=2, hidden_size=4)

def test_qwen3_5_vision_qkv_bias_mismatch_fails_before_swizzle():
    qkv = torch.nn.Linear(4, 12, bias=True)
    qkv.bias = torch.nn.Parameter(torch.zeros(11))
    block = SimpleNamespace(attn=SimpleNamespace(qkv=qkv))
    with pytest.raises(ValueError, match='fused QKV bias.*\\(12,\\)'):
        vision._convert_vision_block_weights_for_rpu(block, num_heads=2, hidden_size=4)
