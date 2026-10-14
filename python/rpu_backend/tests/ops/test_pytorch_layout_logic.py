import pytest
import torch
import torch.nn.functional as F
import rpu_backend
(BATCH, DIM1, DIM2) = (1, 8, 64)
SEED = 0
SAME = 0.999
DIFFERENT = 0.9

def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()

def _qkv(head_dim):
    torch.manual_seed(SEED)
    return tuple((torch.randn(BATCH, DIM1, DIM2, head_dim, dtype=torch.float16) for _ in range(3)))

def _cpu_reference(q, k, v, *, unified):
    if unified:
        out = F.scaled_dot_product_attention(*(t.float().transpose(1, 2).contiguous() for t in (q, k, v)), is_causal=True)
        return out.transpose(1, 2).contiguous().half()
    return F.scaled_dot_product_attention(*(t.float() for t in (q, k, v)), is_causal=True).half()

def _via_aten(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)

def _via_unified(q, k, v):
    return torch.ops.rpu.scaled_dot_product_attention_unified(q, k, v, None, 0.0, True, None, False)
ENTRY_POINTS = {'aten': (_via_aten, False), 'rpu_unified': (_via_unified, True)}

@pytest.mark.parametrize('head_dim', [64, 128])
@pytest.mark.parametrize('entry', sorted(ENTRY_POINTS))
def test_entry_point_reads_its_registered_layout(entry, head_dim):
    (call, unified) = ENTRY_POINTS[entry]
    (q, k, v) = _qkv(head_dim)
    mine = _cpu_reference(q, k, v, unified=unified)
    theirs = _cpu_reference(q, k, v, unified=not unified)
    assert mine.shape == theirs.shape
    separation = _cos(mine, theirs)
    assert separation < DIFFERENT, f'the two layout readings agree at this shape (cos={separation:.6f}); pick dims where they differ or this test proves nothing'
    got = call(q.to('rpu'), k.to('rpu'), v.to('rpu')).cpu()
    assert got.shape == mine.shape
    (right, wrong) = (_cos(got, mine), _cos(got, theirs))
    own = 'unified' if unified else 'PyTorch'
    other = 'PyTorch' if unified else 'unified'
    assert right > SAME, f'{entry} head_dim={head_dim}: cos={right:.6f} against its own {own}-layout reference (cos={wrong:.6f} against the {other}-layout one) — the entry point is not honouring its registered layout'
    assert wrong < DIFFERENT, f'{entry} head_dim={head_dim}: output matches the {other}-layout reading (cos={wrong:.6f}) — layouts have been confused'

@pytest.mark.parametrize('head_dim', [64, 128])
def test_the_two_entry_points_disagree_on_identical_input(head_dim):
    (q, k, v) = _qkv(head_dim)
    args = (q.to('rpu'), k.to('rpu'), v.to('rpu'))
    from_aten = _via_aten(*args).cpu()
    from_unified = _via_unified(*args).cpu()
    assert from_aten.shape == from_unified.shape
    agreement = _cos(from_aten, from_unified)
    assert agreement < DIFFERENT, f'head_dim={head_dim}: aten and device-unified returned the SAME answer (cos={agreement:.6f}) for identical input. They declare layouts that differ by a dim1<->dim2 transpose, so one of them is now reading the wrong one — most likely the layout is being inferred from the shapes again instead of coming from the registration point'
