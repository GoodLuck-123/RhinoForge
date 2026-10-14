from __future__ import annotations
import pytest
import torch
import rpu_backend

def _rpu_hw_probe() -> bool:
    try:
        torch.zeros(1, device='rpu')
        return True
    except Exception:
        return False

@pytest.fixture
def rpu_runtime():
    if not _rpu_hw_probe():
        pytest.skip('needs device device')
    torch.rpu.set_debug(False)
    torch.rpu.set_profile(False)
    torch.rpu.set_caching_allocator(True)
    torch.rpu.set_ddr_flush(True)
    try:
        torch._dynamo.reset()
    except Exception:
        pass

def _cache_stats(backend):
    snap = backend.cache.snapshot()
    return {'size': backend.cache.size(), 'replay_total': sum((e.replay_count for e in snap)), 'recapture_total': sum((e.recapture_count for e in snap)), 'non_replayable': [e.non_replayable_reason for e in snap if e.non_replayable_reason]}

def _gelu_chain(x):
    return torch.nn.functional.gelu(torch.nn.functional.gelu(x))

def test_gelu_chain_compile_matches_eager_and_replays(rpu_runtime):
    torch.manual_seed(0)
    x = torch.randn(1, 8, 64, dtype=torch.float16).to('rpu')
    out_eager = _gelu_chain(x).to('cpu')
    from rpu_backend.graph.dynamo_backend import RpuDynamoBackend
    backend = RpuDynamoBackend(max_entries=2, debug=False)
    compiled = torch.compile(_gelu_chain, backend=backend)
    out_c1 = compiled(x).to('cpu')
    stats_1 = _cache_stats(backend)
    out_c2 = compiled(x).to('cpu')
    stats_2 = _cache_stats(backend)
    assert torch.allclose(out_eager, out_c1, atol=0.001)
    assert torch.allclose(out_c1, out_c2, atol=0.001)
    assert torch.equal(out_eager, out_c1)
    assert torch.equal(out_c1, out_c2)
    assert stats_1['size'] == 1
    assert stats_1['replay_total'] == 0
    assert stats_2['replay_total'] == 1
    assert stats_2['recapture_total'] == 0
    assert not stats_2['non_replayable']
