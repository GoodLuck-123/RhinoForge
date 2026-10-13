from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from rpu_backend.adapters.wall_oss import vision

def test_wall_oss_vision_has_no_optimized_away_assertions():
    tree = ast.parse(Path(vision.__file__).read_text(encoding='utf-8'))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]

def test_vision_rope_table_contract_is_explicit_and_board_free():
    (cos, sin) = vision.build_vision_rope_tables(80, 64, device='cpu')
    assert tuple(cos.shape) == (64, 20)
    assert tuple(sin.shape) == (64, 20)
    assert cos.dtype is torch.float16
    assert sin.dtype is torch.float16

def test_gate_up_split_accepts_raw_and_per_half_zero_padding():
    raw = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    stored = torch.tensor([[1.0], [2.0], [0.0], [3.0], [4.0], [0.0]])
    padded = torch.tensor([[1.0], [2.0], [0.0], [0.0], [3.0], [4.0], [0.0], [0.0]])
    (raw_gate, raw_up) = vision._split_gate_up(raw, 2, 4, 'gate_up')
    (stored_gate, stored_up) = vision._split_gate_up(stored, 2, 4, 'gate_up')
    (pad_gate, pad_up) = vision._split_gate_up(padded, 2, 4, 'gate_up')
    assert torch.equal(vision._pad_rows(raw_gate, 4), pad_gate)
    assert torch.equal(vision._pad_rows(raw_up, 4), pad_up)
    assert torch.equal(vision._pad_rows(stored_gate, 4), pad_gate)
    assert torch.equal(vision._pad_rows(stored_up, 4), pad_up)
    scales = torch.tensor([1.0, 2.0, 0.125, 3.0, 4.0, 0.25])
    (gate_scale, up_scale) = vision._split_gate_up(scales, 2, 4, 'gate_up_scale', check_padding=False)
    assert torch.equal(gate_scale, scales[:3])
    assert torch.equal(up_scale, scales[3:])
    padded[3] = 1.0
    with pytest.raises(ValueError, match='padded rows must be zero'):
        vision._split_gate_up(padded, 2, 4, 'gate_up')

@pytest.mark.parametrize('rows', [2, 3, 10])
def test_gate_up_split_rejects_invalid_stored_half(rows):
    with pytest.raises(ValueError, match='two halves'):
        vision._split_gate_up(torch.zeros(rows), 2, 4, 'gate_up')

@pytest.mark.parametrize('layer_group', ['16', '32.0', '+32', 'true'])
def test_experimental_schedule_rejects_ambiguous_layer_group(monkeypatch, layer_group):
    monkeypatch.setenv('RPU_WALL_OSS_VISION_LAYER_GROUP', layer_group)
    monkeypatch.delenv('RPU_WALL_OSS_VISION_WEIGHT_OUTER', raising=False)
    with pytest.raises(ValueError, match='accepts only 0.*32'):
        vision._experimental_schedule_from_env()

def test_experimental_schedule_rejects_conflicts_and_accepts_weight_outer(monkeypatch):
    monkeypatch.setenv('RPU_WALL_OSS_VISION_LAYER_GROUP', '32')
    monkeypatch.setenv('RPU_WALL_OSS_VISION_WEIGHT_OUTER', '1')
    with pytest.raises(ValueError, match='mutually exclusive'):
        vision._experimental_schedule_from_env()
    monkeypatch.setenv('RPU_WALL_OSS_VISION_LAYER_GROUP', '0')
    assert vision._experimental_schedule_from_env() == (0, True)

def _construct_vision(monkeypatch, *, window=False):
    monkeypatch.setattr(torch.ops.rpu, 'qwen25vl_vision_destroy', lambda handle: None, raising=False)
    return vision.WallOssVision(handle=0, cache=object(), graph_cache=SimpleNamespace(clear=lambda : None, cache_invariant_ok=lambda : True), keepalive=torch.empty(0), patch_w=torch.empty(0), merger=lambda value: value, hidden_size=8, num_layers=1, spatial_merge_size=2, window=window, window_size=112, patch_size=14)

def test_experimental_schedule_is_process_startup_only(monkeypatch):
    monkeypatch.setattr(vision, '_PROCESS_EXPERIMENTAL_SCHEDULE', None)
    monkeypatch.setenv('RPU_WALL_OSS_VISION_LAYER_GROUP', '0')
    monkeypatch.setenv('RPU_WALL_OSS_VISION_WEIGHT_OUTER', '0')
    _construct_vision(monkeypatch).close()
    monkeypatch.setenv('RPU_WALL_OSS_VISION_LAYER_GROUP', '32')
    with pytest.raises(RuntimeError, match='process-startup-only'):
        _construct_vision(monkeypatch)

def test_native_window_sdpa_is_explicit_opt_in(monkeypatch):
    monkeypatch.setattr(vision, '_PROCESS_EXPERIMENTAL_SCHEDULE', None)
    monkeypatch.setenv('RPU_WALL_OSS_VISION_LAYER_GROUP', '0')
    monkeypatch.setenv('RPU_WALL_OSS_VISION_WEIGHT_OUTER', '1')
    monkeypatch.setenv('RPU_WALL_OSS_VISION_MIXED_PRECISION', '1')
    monkeypatch.setenv('RPU_WALL_OSS_VISION_FUSED_MERGER', '1')
    monkeypatch.delenv('RPU_WALL_OSS_VISION_NATIVE_WINDOW_SDPA', raising=False)
    model = _construct_vision(monkeypatch, window=True)
    assert model._native_window_sdpa is False
    model.close()
    monkeypatch.setenv('RPU_WALL_OSS_VISION_NATIVE_WINDOW_SDPA', '1')
    model = _construct_vision(monkeypatch, window=True)
    assert model._native_window_sdpa is True
    model.close()
    with pytest.raises(ValueError, match='requires the exact W8A16'):
        vision.build_wall_oss_vision('/must-not-be-read', window=True, w8a16=False)

def test_native_window_boundary_selection_is_fail_closed():
    model = vision.WallOssVision.__new__(vision.WallOssVision)
    canonical = torch.tensor(vision._NATIVE_WINDOW_BOUNDARIES)
    model._native_window_sdpa = False
    assert model._select_native_window_boundaries(True, canonical) is None
    model._native_window_sdpa = True
    assert model._select_native_window_boundaries(True, canonical) is canonical
    with pytest.raises(ValueError, match='exact 3x576 canonical boundaries'):
        model._select_native_window_boundaries(False, canonical)
    with pytest.raises(ValueError, match='exact 3x576 canonical boundaries'):
        model._select_native_window_boundaries(True, canonical[:-1])

@pytest.mark.parametrize('pixel_values,grid,match', [(torch.zeros(1, 4), [], 'grid_thw must be a torch.Tensor'), (torch.zeros(1, 4), torch.zeros(3), '\\[N, 3\\]'), (torch.zeros(1, 4), torch.zeros(1, 2), '\\[N, 3\\]'), ([], torch.tensor([[1, 1, 1]]), 'pixel_values must be a torch.Tensor'), (torch.zeros(2, 4), torch.tensor([[1, 1, 1]]), 'rows 2.*patches 1')])
def test_vision_forward_rejects_bad_shapes_before_rpu_access(monkeypatch, pixel_values, grid, match):
    model = vision.WallOssVision.__new__(vision.WallOssVision)
    monkeypatch.setenv('RPU_WALL_OSS_VISION_LAYER_GROUP', '0')
    monkeypatch.setenv('RPU_WALL_OSS_VISION_WEIGHT_OUTER', '0')
    with pytest.raises((TypeError, ValueError), match=match):
        model.forward(pixel_values, grid)

def test_layer_group_requires_true_batch_switch_before_rpu_access(monkeypatch):
    model = vision.WallOssVision.__new__(vision.WallOssVision)
    model._layer_group = 32
    model._weight_outer = False
    model._batch_vision = False
    monkeypatch.setenv('RPU_WALL_OSS_VISION_LAYER_GROUP', '32')
    monkeypatch.setenv('RPU_WALL_OSS_VISION_WEIGHT_OUTER', '0')
    monkeypatch.setenv('RPU_WALL_OSS_VISION_MIXED_PRECISION', '0')
    monkeypatch.setenv('RPU_WALL_OSS_BATCH_VISION', 'false')
    with pytest.raises(ValueError, match='BATCH_VISION=1'):
        model.forward(torch.zeros(3, 4), torch.ones(3, 3, dtype=torch.int64))

def test_group_forward_requires_equal_image_geometry_before_rpu_access():
    model = vision.WallOssVision.__new__(vision.WallOssVision)
    grid = torch.tensor([[1, 2, 2], [1, 2, 3]], dtype=torch.int64)
    with pytest.raises(ValueError, match='equal-size images'):
        model._forward_group(torch.zeros(10, 4), grid)

def test_normalize_fold_requires_homogeneous_bias_column():
    model = vision.WallOssVision.__new__(vision.WallOssVision)
    model._patch_w = torch.zeros(2, 3)
    model._pe_k = 3
    with pytest.raises(RuntimeError, match='no pad column'):
        model.enable_normalize_fold(1.0, [0.0] * 3, [1.0] * 3)

@pytest.mark.parametrize('kwargs,match', [({'window': 'false'}, 'window must be bool'), ({'w8a16': 'false'}, 'w8a16 must be bool'), ({'per_window_sdpa': 'false'}, 'per_window_sdpa must be bool')])
def test_vision_builder_rejects_ambiguous_booleans_before_checkpoint_access(kwargs, match):
    with pytest.raises(ValueError, match=match):
        vision.build_wall_oss_vision(**kwargs)

def _execution_vision(monkeypatch, chunk_size, *, layer_group=0):
    model = vision.WallOssVision.__new__(vision.WallOssVision)
    model._execution_chunk_size = chunk_size
    model._layer_group = layer_group
    model._weight_outer = False
    model._batch_vision = True
    model._batch_max_seq = vision._MAX_BATCH_SEQ
    model._fused_assemble = False
    model._mixed_precision = False
    model._per_window_sdpa = False
    monkeypatch.setenv('RPU_WALL_OSS_VISION_LAYER_GROUP', str(layer_group))
    monkeypatch.setenv('RPU_WALL_OSS_VISION_WEIGHT_OUTER', '0')
    monkeypatch.setenv('RPU_WALL_OSS_VISION_MIXED_PRECISION', '0')
    monkeypatch.setenv('RPU_WALL_OSS_BATCH_VISION', '1')
    monkeypatch.delenv('RPU_WALL_OSS_BATCH_MAX_SEQ', raising=False)
    return model

def test_exact_vision_chunk_uses_native_single_and_packed_geometry(monkeypatch):
    calls = []
    model = _execution_vision(monkeypatch, 256)
    model._forward_one = lambda pixels, grid, fused=False: calls.append(('one', int(grid.prod().item()))) or torch.empty(0)
    model._forward_group = lambda pixels, grid, fused=False: calls.append(('group', int(grid.prod(-1).sum().item()))) or torch.empty(0)
    model.forward(torch.zeros(256, 1), torch.tensor([[1, 16, 16]]))
    assert calls == [('one', 256)]
    model._execution_chunk_size = 112
    model.forward(torch.zeros(100, 1), torch.tensor([[1, 10, 10]]))
    assert calls[-1] == ('one', 100)
    model._execution_chunk_size = 512
    model.forward(torch.zeros(512, 1), torch.tensor([[1, 16, 16], [1, 16, 16]]))
    assert calls[-1] == ('group', 512)

def test_mixed_vision_chunks_fail_before_any_subgroup_dispatch(monkeypatch):
    calls = []
    model = _execution_vision(monkeypatch, 256)
    model._forward_one = lambda *args, **kwargs: calls.append('one')
    model._forward_group = lambda *args, **kwargs: calls.append('group')
    with pytest.raises(ValueError, match='native=\\(256, 512\\)'):
        model.forward(torch.zeros(768, 1), torch.tensor([[1, 16, 16], [1, 16, 32]]))
    assert calls == []

def test_exact_vision_chunk_for_special_three_image_schedule_is_per_image(monkeypatch):
    calls = []
    model = _execution_vision(monkeypatch, 576, layer_group=32)
    model._forward_group = lambda pixels, grid, fused=False: calls.append(int(grid[0].prod().item())) or torch.empty(0)
    model.forward(torch.zeros(3 * 576, 1), torch.tensor([[1, 24, 24]] * 3))
    assert calls == [576]

@pytest.mark.parametrize('field', ['padding_rows', 'padding_budget'])
def test_vision_builder_rejects_padding_before_checkpoint_access(field):
    with pytest.raises(ValueError, match='not supported'):
        vision.build_wall_oss_vision(execution_config={'vision': {field: 0}})

def test_wall_builder_keeps_per_window_sdpa_disabled_by_default():
    import inspect
    assert inspect.signature(vision.build_wall_oss_vision).parameters['per_window_sdpa'].default is False

@pytest.mark.parametrize('mask,admitted', [('0', False), ('2', False), ('1', True), ('7', True)])
def test_native_window_fp16_requires_vision_transfer_bit(monkeypatch, mask, admitted):
    monkeypatch.setattr(vision, '_PROCESS_EXPERIMENTAL_SCHEDULE', None)
    monkeypatch.setenv('RPU_WALL_OSS_FP16_TRANSFER_MASK', mask)
    for name in ('WEIGHT_OUTER', 'MIXED_PRECISION', 'FUSED_MERGER', 'NATIVE_WINDOW_SDPA'):
        monkeypatch.setenv('RPU_WALL_OSS_VISION_' + name, '1')
    monkeypatch.setenv('RPU_WALL_OSS_VISION_LAYER_GROUP', '0')

    def checkpoint_access(*_):
        raise RuntimeError('admitted before checkpoint access')
    monkeypatch.setattr(vision, '_resolve_ckpt', checkpoint_access)
    error = RuntimeError if admitted else ValueError
    match = 'admitted before checkpoint' if admitted else 'requires the exact'
    with pytest.raises(error, match=match):
        vision.build_wall_oss_vision('/must-not-be-read', window=True)
