import os
import pytest
from rpu_backend.tests.fixtures.pi05_fuse_fixture import pi05_fuse_fixture
from rpu_backend.graph._runtime import _graph_fnv1a_64
pytestmark = pytest.mark.skipif(os.environ.get('PI05_HARDWARE') != '1', reason='Requires hardware + Pi0.5 weights — set PI05_HARDWARE=1 with sudo.')

def _count_entries_for_op(cache, op_id: str) -> int:
    h = _graph_fnv1a_64(op_id)
    return sum((1 for e in cache.snapshot() if e.signature.op_id == h))

def _prepared_hit_for_op(cache, op_id: str) -> int:
    h = _graph_fnv1a_64(op_id)
    return sum((e.prepared_segment_hit_total for e in cache.snapshot() if e.signature.op_id == h))

def test_graph_cache_1_build_2_replays_for_3_calls(pi05_fuse_fixture):
    import torch
    fx = pi05_fuse_fixture
    cfg = fx['config']
    (bs, cs, mad) = (1, cfg.chunk_size, cfg.max_action_dim)
    x_t_rpu = torch.zeros(bs, cs, mad, dtype=torch.float16, device='rpu')
    cond = fx['adarms_cond_all'].select(0, 0).contiguous()
    v_t_buf = torch.empty(bs, cs, mad, dtype=torch.float16, device='rpu')
    entries_before = _count_entries_for_op(fx['cache'], 'pi05_denoise_step')
    hits_before = _prepared_hit_for_op(fx['cache'], 'pi05_denoise_step')
    for _ in range(3):
        fx['step'](x_t_rpu, cond, v_t_buf)
    entries_after = _count_entries_for_op(fx['cache'], 'pi05_denoise_step')
    hits_after = _prepared_hit_for_op(fx['cache'], 'pi05_denoise_step')
    new_entries = entries_after - entries_before
    new_hits = hits_after - hits_before
    assert new_entries == 1, f"expected 1 new BUILD entry, got {new_entries} (snapshot: {fx['cache'].snapshot()})"
    assert new_hits >= 2, f"expected ≥2 prepared_segment_hits (2 REPLAYs × ≥1 seg), got {new_hits} (snapshot: {fx['cache'].snapshot()})"
