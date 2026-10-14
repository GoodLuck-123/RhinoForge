from __future__ import annotations
import ast
from pathlib import Path
import pytest
import torch
from rpu_backend.runtime import weights

def test_weight_runtime_has_no_optimized_away_assertions():
    tree = ast.parse(Path(weights.__file__).read_text(encoding='utf-8'))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]

@pytest.mark.parametrize(('in_features', 'out_features', 'expected'), [(128, 128, 1), (128, 16, 0), (16, 128, 1), (17, 17, -1)])
def test_partition_selection_matches_native_contract(in_features, out_features, expected):
    assert weights.get_linear_partition(in_features, out_features) == expected

@pytest.mark.parametrize(('args', 'message'), [((0, 128), 'in_features'), ((128, 0), 'out_features'), ((128, 128, 3), 'divide 32'), ((128, 128, 2, 0), 'num_cores'), ((128, 128, 2, 9), '<= 8')])
def test_partition_selection_rejects_invalid_layout(args, message):
    with pytest.raises((TypeError, ValueError), match=message):
        weights.get_linear_partition(*args)

def test_row_and_col_swizzle_keep_the_verified_byte_permutations():
    row = torch.arange(16 * 128, dtype=torch.float16).reshape(16, 128)
    expected_row = row.view(1, 16, 8, 1, 16).permute(0, 3, 2, 1, 4).reshape(16, 128).contiguous()
    assert torch.equal(weights.tp_row_swizzle_mc_weight(row), expected_row)
    col = torch.arange(128 * 16, dtype=torch.float16).reshape(128, 16)
    expected_col = col.view(8, 1, 16, 1, 16).permute(1, 3, 0, 2, 4).reshape(128, 16).contiguous()
    assert torch.equal(weights.tp_col_swizzle_mc_weight(col), expected_col)

@pytest.mark.parametrize('call', [lambda : weights.tp_row_swizzle_mc_weight(torch.zeros(16, 127)), lambda : weights.tp_col_swizzle_mc_weight(torch.zeros(127, 16)), lambda : weights.tp_col_swizzle_mc_weight(torch.zeros(128, 16, dtype=torch.float32)), lambda : weights.transform_linear_weight(torch.zeros(128, 16), 2), lambda : weights.transform_linear_weight([], 1)])
def test_swizzle_rejects_invalid_layout_before_permutation(call):
    with pytest.raises((TypeError, ValueError)):
        call()

def test_convert_and_safe_load_round_trip(tmp_path):
    source = torch.nn.Sequential(torch.nn.Linear(16, 128)).float()
    artifact = tmp_path / 'converted.pt'
    partition_info = weights.convert_model_for_rpu(source, str(artifact), device='cpu', half=True)
    assert partition_info == {'0.weight': 1}
    payload = torch.load(artifact, map_location='cpu', weights_only=True)
    assert payload['state_dict']['0.weight'].dtype is torch.float16
    target = torch.nn.Sequential(torch.nn.Linear(16, 128)).half()
    loaded = weights.load_rpu_model(target, str(artifact), device='cpu')
    assert torch.equal(loaded.state_dict()['0.weight'], payload['state_dict']['0.weight'])

def test_convert_without_half_rejects_fp32_before_swizzle(tmp_path):
    source = torch.nn.Sequential(torch.nn.Linear(16, 128)).float()
    with pytest.raises(TypeError, match='expected torch.float16'):
        weights.convert_model_for_rpu(source, str(tmp_path / 'bad.pt'), device='cpu', half=False)

def test_load_rejects_unconverted_artifact(tmp_path):
    artifact = tmp_path / 'unconverted.pt'
    torch.save({'state_dict': torch.nn.Linear(16, 128).state_dict(), 'rpu_converted': False}, artifact)
    with pytest.raises(ValueError, match='not marked as converted'):
        weights.load_rpu_model(torch.nn.Linear(16, 128), str(artifact), device='cpu')

def test_load_validates_partition_metadata_before_model_mutation(tmp_path):
    target = torch.nn.Linear(16, 128).half()
    original = {name: value.clone() for (name, value) in target.state_dict().items()}
    replacement = torch.nn.Linear(16, 128).half().state_dict()
    artifact = tmp_path / 'invalid-partition.pt'
    torch.save({'state_dict': replacement, 'partition_info': {'weight': 7}, 'rpu_converted': True}, artifact)
    with pytest.raises(TypeError, match='invalid partition_info'):
        weights.load_rpu_model(target, str(artifact), device='cpu')
    for (name, value) in target.state_dict().items():
        assert torch.equal(value, original[name])
