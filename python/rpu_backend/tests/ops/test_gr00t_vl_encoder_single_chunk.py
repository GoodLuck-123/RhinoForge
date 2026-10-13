from __future__ import annotations
import pytest
import torch
import rpu_backend
from rpu_backend.api import RPUCache
pytestmark = pytest.mark.skipif(not hasattr(torch.ops.rpu, 'gr00t_vl_encoder_create'), reason='needs the native backend (device device nodes)')
(H, NQ, HD, FF, NL) = (2048, 32, 64, 8192, 1)
EPS = 1e-06

def _rpu(*shape):
    return torch.zeros(*shape, dtype=torch.float16, device='rpu')

def _lists(n, *shape):
    return [_rpu(*shape) for _ in range(n)]

@pytest.fixture
def handle():
    h = torch.ops.rpu.gr00t_vl_encoder_create()
    try:
        torch.ops.rpu.gr00t_vl_encoder_set_weights(h, _lists(NL, H), _lists(NL, H), _lists(NL, NQ * HD, H), _lists(NL, NQ * HD), _lists(NL, NQ * HD, H), _lists(NL, NQ * HD), _lists(NL, NQ * HD, H), _lists(NL, NQ * HD), _lists(NL, H, NQ * HD), _lists(NL, H), _lists(NL, H), _lists(NL, H), _lists(NL, FF, H), _lists(NL, FF), _lists(NL, H, FF), _lists(NL, H), _rpu(H), _rpu(H), _rpu(H), NQ, HD, H, FF, EPS)
    except Exception:
        torch.ops.rpu.gr00t_vl_encoder_destroy(h)
        raise
    yield h
    torch.ops.rpu.gr00t_vl_encoder_destroy(h)

def _caches(seq):
    c = RPUCache(num_layers=NL, batch_size=1, max_seq_len=seq, num_kv_heads=NQ, head_dim=HD)
    return ([c.k_caches[i] for i in range(NL)], [c.v_caches[i] for i in range(NL)], c)
SEQ_SINGLE_CHUNK = 128
SEQ_MULTI_CHUNK = 512

def test_a_supported_shape_runs_as_one_chunk(handle):
    (k, v, _c) = _caches(SEQ_SINGLE_CHUNK)
    out = torch.ops.rpu.gr00t_vl_encoder_forward(handle, _rpu(1, SEQ_SINGLE_CHUNK, H), k, v)
    assert tuple(out.shape) == (1, SEQ_SINGLE_CHUNK, H)

def test_the_largest_single_chunk_shape_still_passes(handle):
    (k, v, _c) = _caches(256)
    out = torch.ops.rpu.gr00t_vl_encoder_forward(handle, _rpu(1, 256, H), k, v)
    assert tuple(out.shape) == (1, 256, H)

def test_a_multi_chunk_plan_triggers_the_guard(handle):
    (k, v, _c) = _caches(SEQ_MULTI_CHUNK)
    with pytest.raises(RuntimeError, match='requires a single chunk'):
        torch.ops.rpu.gr00t_vl_encoder_forward(handle, _rpu(1, SEQ_MULTI_CHUNK, H), k, v)

def test_the_refusal_reports_the_plan_it_refused(handle):
    (k, v, _c) = _caches(SEQ_MULTI_CHUNK)
    with pytest.raises(RuntimeError) as exc:
        torch.ops.rpu.gr00t_vl_encoder_forward(handle, _rpu(1, SEQ_MULTI_CHUNK, H), k, v)
    msg = str(exc.value)
    assert 'got 2' in msg, msg
    assert 'chunk_size=256' in msg, msg
    assert 'cross-patch attention' in msg, 'the message must say WHAT breaks, not just that it is disallowed'
