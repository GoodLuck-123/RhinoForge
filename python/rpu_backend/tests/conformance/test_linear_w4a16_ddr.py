import os
import pytest
import torch
from rpu_backend.quant.int4_pgrp_pack import swizzle_int4_pgrp_scale, swizzle_pack_int4_pgrp
pytestmark = pytest.mark.skipif(os.environ.get('RPU_HW_AVAILABLE') != '1', reason='requires device hardware')

def _operands(rows=17, n=128, k=256):
    generator = torch.Generator().manual_seed(2019)
    x = (torch.rand((rows, k), generator=generator) * 0.2 - 0.1).half()
    q = torch.randint(-8, 8, (n, k), generator=generator, dtype=torch.int8)
    logical = (torch.rand((k // 32, n), generator=generator) * 0.02 + 0.01).half()
    dequant = q.float().view(n, k // 32, 32) * logical.T.float().unsqueeze(-1)
    packed = swizzle_pack_int4_pgrp(q, 1, 8).reshape(n, k // 2).to('rpu')
    scale = swizzle_int4_pgrp_scale(logical, 32, 1, 8).to('rpu')
    return (x, dequant.reshape(n, k), packed, scale)

@pytest.mark.parametrize('rows,n,k,bias', [(1, 128, 2048, False), (17, 512, 2560, True)])
def test_w4_ddr_matches_dequantized_reference_and_returns_fresh(rows, n, k, bias):
    (x, weight, packed, scale) = _operands(rows, n, k)
    b = torch.linspace(-0.1, 0.1, n).half() if bias else None
    br = b.to('rpu') if bias else None
    xr = x.to('rpu')
    first = torch.ops.rpu.linear_w4a16(xr, packed, scale, br)
    saved = first.cpu().clone()
    second = torch.ops.rpu.linear_w4a16((-x).to('rpu'), packed, scale, br)
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first.cpu(), saved)
    reference = x.float() @ weight.T
    if bias:
        reference += b.float()
    other = -reference + 2 * b.float() if bias else -reference
    for (actual, expected) in ((saved, reference), (second.cpu(), other)):
        assert actual.dtype == torch.float16 and actual.shape == (rows, n)
        assert torch.isfinite(actual).all()
        relative = torch.linalg.vector_norm(actual.float() - expected) / torch.linalg.vector_norm(expected)
        assert relative <= 0.01, float(relative)

def test_w4_ddr_rejects_invalid_scale_shape_and_capture():
    from rpu_backend.graph import GraphCache, GraphSignature
    (x, _, packed, scale) = _operands()
    xr = x.to('rpu')
    with pytest.raises(RuntimeError, match='scale'):
        torch.ops.rpu.linear_w4a16(xr, packed, scale[:, :-1].contiguous())
    with pytest.raises(RuntimeError, match='packed weight'):
        torch.ops.rpu.linear_w4a16(xr[:, :128].contiguous(), packed, scale)
    cache = GraphCache()
    signature = GraphSignature(op_id='w4-ddr-eager-boundary', shapes=(17, 256), dtypes=(5,))
    with pytest.raises(RuntimeError, match='eager'):
        with cache.capture(signature):
            torch.ops.rpu.linear_w4a16(xr, packed, scale)
    assert cache.cache_invariant_ok()
    assert torch.isfinite(torch.ops.rpu.linear_w4a16(xr, packed, scale).cpu()).all()
