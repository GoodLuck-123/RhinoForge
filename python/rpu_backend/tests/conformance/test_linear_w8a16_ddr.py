import os
import pytest
import torch
from rpu_backend.runtime.weights import transform_linear_weight
pytestmark = pytest.mark.skipif(os.environ.get('RPU_HW_AVAILABLE') != '1', reason='requires device hardware')

def _operands(m, n, k):
    g = torch.Generator().manual_seed(1609 + m)
    x = (torch.rand((m, k), generator=g) * 0.2 - 0.1).half()
    w = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int8)
    scale = (torch.rand((n,), generator=g) * 0.002 + 0.001).half()
    packed = transform_linear_weight(w, 1, num_cores=8).contiguous().to('rpu')
    return (x, w, scale, x.to('rpu'), packed, scale.to('rpu'))

@pytest.mark.parametrize('m,n,k,bias', [(1, 1024, 1536, True), (17, 1024, 1536, False), (256, 1024, 1536, True), (32, 151936, 2560, False)])
def test_w8_ddr_matches_cpu_and_retains_distinct_outputs(m, n, k, bias):
    torch.set_num_threads(8)
    (x, w, scale, xr, wr, sr) = _operands(m, n, k)
    b = torch.linspace(-0.1, 0.1, n).half() if bias else None
    br = b.to('rpu') if bias else None
    reference = x.float() @ (w.float() * scale.float()[:, None]).T
    if bias:
        reference += b.float()
    first = torch.ops.rpu.linear_w8a16(xr, wr, sr, br)
    saved = first.cpu()
    second = torch.ops.rpu.linear_w8a16((-x).to('rpu'), wr, sr, br)
    assert first.data_ptr() != second.data_ptr()
    assert torch.equal(first.cpu(), saved)
    expected_second = -reference + 2 * b.float() if bias else -reference
    for (actual, expected) in ((saved, reference), (second.cpu(), expected_second)):
        assert actual.dtype == torch.float16 and actual.shape == (m, n)
        assert torch.isfinite(actual).all()
        cos = torch.nn.functional.cosine_similarity(actual.float(), expected, dim=-1)
        rel = torch.linalg.vector_norm(actual.float() - expected) / torch.linalg.vector_norm(expected)
        assert cos.min() >= 0.999, (m, n, k, float(cos.min()))
        assert rel <= 0.01, (m, n, k, float(rel))
    out = torch.empty_like(second)
    torch.ops.rpu.linear_w8a16_into(xr, wr, sr, br, out)
    assert torch.equal(out.cpu(), saved)

def test_w8_ddr_rejects_invalid_scale_overlap_and_graph_capture():
    from rpu_backend.graph import GraphCache, GraphSignature
    (_, _, _, x, w, scale) = _operands(2, 128, 128)
    with pytest.raises(RuntimeError, match='scale'):
        torch.ops.rpu.linear_w8a16(x, w, scale[:64], None)
    with pytest.raises(RuntimeError, match='K divisible by 32'):
        torch.ops.rpu.linear_w8a16(torch.zeros((1, 16), dtype=torch.float16, device='rpu'), torch.zeros((128, 16), dtype=torch.int8, device='rpu'), scale, None)
    with pytest.raises(RuntimeError, match='single memory location|overlap'):
        torch.ops.rpu.linear_w8a16_into(x, w, scale, None, x)
    cache = GraphCache()
    sig = GraphSignature(op_id='w8-ddr-eager-boundary', shapes=(2, 128), dtypes=(5,))
    with pytest.raises(RuntimeError, match='eager'):
        with cache.capture(sig):
            torch.ops.rpu.linear_w8a16(x, w, scale, None)
    assert cache.cache_invariant_ok()
    assert torch.isfinite(torch.ops.rpu.linear_w8a16(x, w, scale, None).cpu()).all()

def test_w8_weight_view_beyond_two_gibibytes_preserves_device_address():
    (x, w, scale, xr, _, sr) = _operands(1, 128, 128)
    offset = 1 << 31
    storage = torch.empty(offset + w.numel(), dtype=torch.int8, device='rpu')
    view = storage.narrow(0, offset, w.numel()).view_as(w)
    view.copy_(transform_linear_weight(w, 1, num_cores=8).contiguous())
    assert view.is_contiguous() and view.data_ptr() - storage.data_ptr() == offset
    expected = x.float() @ (w.float() * scale.float()[:, None]).T
    result = torch.ops.rpu.linear_w8a16(xr, view, sr, None).cpu().float()
    assert torch.isfinite(result).all()
    assert torch.nn.functional.cosine_similarity(result, expected, dim=-1).min() >= 0.999
    assert torch.linalg.vector_norm(result - expected) / torch.linalg.vector_norm(expected) <= 0.01
