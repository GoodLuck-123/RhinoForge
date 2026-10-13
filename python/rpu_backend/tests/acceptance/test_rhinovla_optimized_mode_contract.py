from __future__ import annotations
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from rpu_backend.adapters.rhinovla.vision import _convert_vision_block_weights_for_rpu, _validate_vision_grid_bounds, build_vision_rope_tables
from rpu_backend.adapters.rhinovla.fused import _validate_rhino_prefix_mask, _validate_rhino_prefix_rope_contract

def test_vision_rope_table_contract_is_explicit_and_board_free():
    (cos, sin) = build_vision_rope_tables(64, 48, device='cpu')
    assert tuple(cos.shape) == (48, 16)
    assert tuple(sin.shape) == (48, 16)
    assert cos.dtype is torch.float16
    assert sin.dtype is torch.float16

def test_qkv_shape_mismatch_fails_before_weight_swizzle():
    block = SimpleNamespace(attn=SimpleNamespace(qkv=torch.nn.Linear(4, 11, bias=True)))
    with pytest.raises(ValueError, match='fused QKV weight.*\\(12, 4\\)'):
        _convert_vision_block_weights_for_rpu(block, num_heads=2, hidden_size=4)

def test_vision_grid_is_bounded_by_the_installed_rope_table():
    _validate_vision_grid_bounds(torch.tensor([[1, 48, 32]]), 48)
    with pytest.raises(ValueError, match='H/W max 49.*max_hw 48'):
        _validate_vision_grid_bounds(torch.tensor([[1, 49, 32]]), 48)

def test_vision_grid_guard_precedes_patch_embed_and_native_work():
    vision_path = Path(__file__).resolve().parents[2] / 'adapters/rhinovla/vision.py'
    source = vision_path.read_text(encoding='utf-8')
    forward = source[source.index('def _rpu_vision_forward'):]
    guard = forward.index('_validate_vision_grid_bounds(')
    assert guard < forward.index('# ----- Input prep + patch_embed')
    assert guard < forward.index('qwen3vl_vision_forward(')

def _fixed_rope_inputs(prefix_len=16, suffix_len=3):
    return {'runtime_prefix_len': prefix_len, 'rope_max_seq_len': 64, 'suffix_len': suffix_len, 'batch_size': 1, 'position_ids': torch.arange(prefix_len, prefix_len + suffix_len).view(1, 1, suffix_len).expand(3, 1, suffix_len), 'prefix_mask': torch.ones((1, prefix_len), dtype=torch.bool), 'suffix_mask': torch.ones((1, suffix_len), dtype=torch.bool)}

def test_action_rope_splits_logical_position_from_physical_prefix_row():
    inputs = _fixed_rope_inputs()
    assert _validate_rhino_prefix_rope_contract(**inputs) == 16
    variable = _fixed_rope_inputs(prefix_len=15)
    assert _validate_rhino_prefix_rope_contract(**variable) == 15
    padded = {**inputs, 'prefix_mask': inputs['prefix_mask'].clone()}
    padded['prefix_mask'][0, 12:] = False
    padded['position_ids'] = torch.arange(12, 15).view(1, 1, 3).expand(3, 1, 3)
    assert _validate_rhino_prefix_rope_contract(**padded) == 12
    interior = {**padded, 'prefix_mask': padded['prefix_mask'].clone()}
    interior['prefix_mask'][0, 13] = True
    with pytest.raises(ValueError, match='right-padding'):
        _validate_rhino_prefix_rope_contract(**interior)
    shifted = {**inputs, 'position_ids': inputs['position_ids'] + 1}
    with pytest.raises(NotImplementedError, match='contiguous range'):
        _validate_rhino_prefix_rope_contract(**shifted)
    too_long = {**inputs, 'rope_max_seq_len': 18}
    with pytest.raises(RuntimeError, match='16\\+3>18'):
        _validate_rhino_prefix_rope_contract(**too_long)

def test_action_prefix_mask_helper_is_shared_with_denoise_unroll():
    mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)
    assert _validate_rhino_prefix_mask(runtime_prefix_len=5, batch_size=1, prefix_mask=mask) == 3
    source = (Path(__file__).resolve().parents[2] / 'adapters/rhinovla/pipeline.py').read_text(encoding='utf-8')
    loop = source[source.index('    def _denoise_loop('):]
    loop = loop[:loop.index('    @torch.no_grad()')]
    assert '_rhino_suffix_rope_start(' in loop
    assert 'physical_prefix_len' in loop
    assert 'self.P, rope_position' not in loop

def test_action_rope_accepts_explicit_official_continuation_delta():
    inputs = _fixed_rope_inputs(prefix_len=64, suffix_len=31)
    inputs['prefix_mask'][:, 48:] = False
    inputs['rope_max_seq_len'] = 96
    inputs['prefix_rope_deltas'] = torch.tensor([[-32]])
    inputs['position_ids'] = torch.arange(16, 47).view(1, 1, 31).expand(3, 1, 31)
    assert _validate_rhino_prefix_rope_contract(**inputs) == 16
    with pytest.raises(NotImplementedError, match='contiguous range'):
        _validate_rhino_prefix_rope_contract(**{**inputs, 'prefix_rope_deltas': None})
    with pytest.raises(NotImplementedError, match='contiguous range'):
        _validate_rhino_prefix_rope_contract(**{**inputs, 'prefix_rope_deltas': -31})
    with pytest.raises(ValueError, match='non-negative'):
        _validate_rhino_prefix_rope_contract(**{**inputs, 'prefix_rope_deltas': -49})
    with pytest.raises(RuntimeError, match='RoPE table'):
        _validate_rhino_prefix_rope_contract(**{**inputs, 'rope_max_seq_len': 46})

@pytest.mark.parametrize('delta', [True, 0.5, torch.tensor([[0.5]]), torch.zeros(3, 1, dtype=torch.long)])
def test_action_rope_rejects_noncanonical_delta(delta):
    with pytest.raises((TypeError, ValueError), match='prefix_rope_deltas'):
        _validate_rhino_prefix_rope_contract(**{**_fixed_rope_inputs(), 'prefix_rope_deltas': delta})

@pytest.mark.parametrize(('kwargs', 'match'), [({'prefix_len': -1, 'suffix_len': 3, 'max_seq_len': 16}, 'prefix_len'), ({'prefix_len': 1.5, 'suffix_len': 3, 'max_seq_len': 16}, 'prefix_len'), ({'prefix_len': 1, 'suffix_len': 0, 'max_seq_len': 16}, 'suffix_len'), ({'prefix_len': 14, 'suffix_len': 3, 'max_seq_len': 16}, 'exceeds')])
def test_action_install_geometry_fails_before_expert_access(kwargs, match):
    from rpu_backend.adapters.rhinovla.fused import patch_rhino_vla_for_rpu
    with pytest.raises(ValueError, match=match):
        patch_rhino_vla_for_rpu(object(), **kwargs)

def test_action_rope_guard_precedes_rpu_copy_cache_fill_and_graph():
    fused_path = Path(__file__).resolve().parents[2] / 'adapters/rhinovla/fused.py'
    source = fused_path.read_text(encoding='utf-8')
    forward = source[source.index('def rpu_rhino_vla_fused_forward'):]
    guard = forward.index('_validate_rhino_prefix_rope_contract(')
    setter = forward.index('rhino_vla_set_rope_position(')
    assert guard < forward.index('hidden_states = suffix_embeds')
    assert guard < forward.index('_prefill_kv_cache(')
    assert guard < forward.index('torch.ops.rpu.rhino_vla_forward(')
    assert guard < setter < forward.index('torch.ops.rpu.rhino_vla_forward(')

def test_action_direct_prefix_reuse_requires_an_explicit_cache_reset():
    fused_path = Path(__file__).resolve().parents[2] / 'adapters/rhinovla/fused.py'
    source = fused_path.read_text(encoding='utf-8')
    forward = source[source.index('def rpu_rhino_vla_fused_forward'):]
    stale_guard = forward.index('_rhino_prefix_source_matches(')
    typed_plan = forward.index('_plan_rhino_prefix_kv_execution(')
    cache_fill = forward.index('_prefill_kv_cache(')
    native_forward = forward.index('torch.ops.rpu.rhino_vla_forward(')
    assert typed_plan < stale_guard < cache_fill < native_forward
    assert 'prefix_plan_key_words=(' in forward
    assert 'prefix_plan.graph_key_words()' in forward

def test_action_hot_reconfigure_rebuilds_direct_prefix_cache():
    runtime_path = Path(__file__).resolve().parents[2] / 'adapters/rhinovla/runtime.py'
    source = runtime_path.read_text(encoding='utf-8')
    reset = source[source.index('def _reset(self, changed):'):]
    action = reset[reset.index('if RHINOVLA_ACTION_COMPONENT in changed:'):]
    assert action.index('_rpu_graph_cache') < action.index('cache.reset_to_position(0)')
    assert action.index('cache.reset_to_position(0)') < action.index('vars(self.action_expert)["_rpu_prefix_source_snapshot"] = None')

def test_action_direct_prefix_snapshot_detects_offset_replacement_and_mutation():
    from rpu_backend.adapters.rhinovla.fused import _rhino_prefix_source_matches, _snapshot_rhino_prefix_source
    owner = object()
    k = torch.zeros(1, 2, 3, 4)
    v = torch.ones(1, 2, 3, 4)
    original = _snapshot_rhino_prefix_source(owner, 4, [(k, v)])
    assert _rhino_prefix_source_matches(original, _snapshot_rhino_prefix_source(owner, 4, [(k, v)]))
    assert not _rhino_prefix_source_matches(original, _snapshot_rhino_prefix_source(owner, 5, [(k, v)]))
    assert not _rhino_prefix_source_matches(original, _snapshot_rhino_prefix_source(owner, 4, [(k.clone(), v)]))
    k.add_(1)
    assert not _rhino_prefix_source_matches(original, _snapshot_rhino_prefix_source(owner, 4, [(k, v)]))

def test_action_direct_prefix_snapshot_binds_route_and_generation():
    from rpu_backend.adapters.rhinovla.fused import _rhino_prefix_source_matches, _snapshot_rhino_prefix_source
    owner = object()
    pair = [(torch.zeros(1, 2, 3, 4), torch.ones(1, 2, 3, 4))]
    original = _snapshot_rhino_prefix_source(owner, 0, pair, route_argument_digest='route-a', execution_generation=7)
    assert not _rhino_prefix_source_matches(original, _snapshot_rhino_prefix_source(owner, 0, pair, route_argument_digest='route-b', execution_generation=7))
    assert not _rhino_prefix_source_matches(original, _snapshot_rhino_prefix_source(owner, 0, pair, route_argument_digest='route-a', execution_generation=8))

def test_prefix_kv_fixed_plan_digests_route_and_generation(monkeypatch):
    from rpu_backend.adapters.rhinovla.convert import _plan_rhino_prefix_kv_execution
    route = [2, 2, 48, 48, 1, 2, 0, 0, 48, *[0] * 8, 2, 6, 3, 123, 0]
    monkeypatch.setattr(torch.rpu, 'resolve_kvcache_arena_plan', lambda position, rows, num_kv_heads, head_dim: list(route), raising=False)
    pairs = [(torch.zeros(1, 8, 33, 128), torch.ones(1, 8, 33, 128))]
    (first, arguments, digest) = _plan_rhino_prefix_kv_execution(pairs, component='action_expert', generation=3, queue_owner_id=9)
    (second, _, second_digest) = _plan_rhino_prefix_kv_execution(pairs, component='action_expert', generation=4, queue_owner_id=9)
    assert arguments == tuple(route)
    assert len(digest) == 64 and digest == second_digest
    assert first.selected is not None
    assert first.selected.execution_len == 48
    assert first.selected.padding_rows == 15
    metadata = dict(first.selected.stage_tuple.physical_metadata)
    assert metadata['num_kv_heads'] == 8
    assert metadata['head_dim'] == 128
    assert first.physical_plan_digest != second.physical_plan_digest

def test_prefix_kv_fill_rejects_missing_route_before_cache_reset():
    from rpu_backend.adapters.rhinovla.convert import _prefill_kv_cache
    cache = SimpleNamespace(resets=[], reset_to_position=lambda position: cache.resets.append(int(position)))
    pair = [(torch.zeros(1, 8, 16, 128), torch.ones(1, 8, 16, 128))]
    with pytest.raises(RuntimeError, match='exact typed KV route'):
        _prefill_kv_cache(cache, pair, ())
    assert cache.resets == []

def test_rhinovla_adapter_has_no_optimized_away_assertions():
    adapter_root = Path(__file__).resolve().parents[2] / 'adapters/rhinovla'
    offenders = []
    for path in sorted(adapter_root.rglob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        if any((isinstance(node, ast.Assert) for node in ast.walk(tree))):
            offenders.append(path.relative_to(adapter_root).as_posix())
    assert offenders == []
