import os
import pytest
import torch
import rpu_backend
pytestmark = pytest.mark.skipif(os.environ.get('RPU_HW_AVAILABLE') != '1', reason='Requires a device; set RPU_HW_AVAILABLE=1')

def check_host_argmax_edges() -> None:
    neon_tie = torch.zeros((2, 13), dtype=torch.float16)
    neon_tie[0, [2, 8]] = 4.0
    neon_tie[1, [0, 12]] = 7.0
    neon_nan = torch.arange(26, dtype=torch.float16).view(2, 13)
    neon_nan[0, 3] = float('nan')
    neon_nan[0, 10] = float('nan')
    neon_nan[1, 12] = float('nan')
    cases = (torch.tensor([[1.0, 3.0, 3.0, 2.0]], dtype=torch.float16), torch.tensor([[1.0, float('nan'), 2.0]], dtype=torch.float16), torch.arange(13, dtype=torch.float16).view(1, 13), neon_tie, neon_nan)
    for cpu in cases:
        want = cpu.argmax(dim=-1)
        got_cpu = torch.ops.rpu.argmax_lastdim_host(cpu.contiguous())
        got_rpu = torch.ops.rpu.argmax_lastdim_host(cpu.to('rpu').contiguous())
        if not torch.equal(got_cpu, want) or not torch.equal(got_rpu, want):
            raise RuntimeError(f'argmax_lastdim_host edge mismatch: want={want.tolist()} cpu={got_cpu.tolist()} rpu={got_rpu.tolist()}')
    try:
        torch.ops.rpu.argmax_lastdim_host(torch.empty((1, 0), dtype=torch.float16, device='rpu'))
    except RuntimeError as exc:
        if 'empty last dimension' not in str(exc):
            raise
    else:
        raise RuntimeError('argmax_lastdim_host accepted an empty last dimension')

def test_host_argmax_edges():
    check_host_argmax_edges()
