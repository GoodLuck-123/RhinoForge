from __future__ import annotations
import pytest
import torch
import torch.nn.functional as F
import rpu_backend
pytestmark = pytest.mark.hardware

@pytest.mark.parametrize(('shape', 'normalized_shape'), [((8, 512), (512,)), ((2, 17, 1024), (1024,)), ((196, 1280), (1280,)), ((2, 3, 4), (3, 4))])
def test_eager_rmsnorm_matches_fp32_reference(shape, normalized_shape):
    torch.manual_seed(20260801)
    x = (torch.randn(shape, dtype=torch.float32) * 10).half()
    weight = (torch.randn(normalized_shape, dtype=torch.float32) * 0.1 + 1).half()
    x_float = x.float()
    dims = tuple(range(x.dim() - len(normalized_shape), x.dim()))
    variance = x_float.square().mean(dims, keepdim=True)
    expected = (weight.float() * x_float * torch.rsqrt(variance + 1e-06)).half()
    actual = torch.ops.rpu.rms_norm(x.to('rpu'), normalized_shape, weight.to('rpu'), 1e-06).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0.002)

@pytest.mark.parametrize(('shape', 'dim', 'dtype'), [((1, 17, 64), 2, None), ((1, 17, 64), -1, None), ((2, 17, 64), 2, None), ((1, 5, 31), -1, None), ((1, 2, 2064), -1, None), ((2, 3, 4, 17), -1, None), ((1, 5, 32), -1, torch.float32), ((1, 5, 32), 1, None)])
def test_eager_softmax_respects_shape_dim_and_dtype(shape, dim, dtype, record_property):
    torch.manual_seed(20260801)
    x = torch.randn(shape, dtype=torch.float16)
    expected = torch.softmax(x, dim=dim, dtype=dtype)
    actual = torch.softmax(x.to('rpu'), dim=dim, dtype=dtype).cpu()
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape and actual.numel()
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    cosine = F.cosine_similarity(actual.float().reshape(-1), expected.float().reshape(-1), dim=0).item()
    assert cosine >= 0.99999
    torch.testing.assert_close(actual, expected, rtol=0.002, atol=0.0005)
    if shape == (1, 17, 64) and dim == 2 and (dtype is None):
        pass
