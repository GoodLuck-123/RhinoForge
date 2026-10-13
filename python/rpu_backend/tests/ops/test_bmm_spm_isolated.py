import torch
import rpu_backend

def test_bmm_eager_parity(record_property):
    """Run before any operation initializes the shared memory pool."""
    from rpu_backend.tests._seed import input_generator
    from rpu_backend.tests.conformance._numeric import require_same_semantic
    generator = input_generator()
    for (batch, M, K, N) in ((1, 16, 64, 16), (2, 64, 96, 64), (1, 64, 128, 128), (2, 32, 256, 64)):
        a = torch.randn(batch, M, K, generator=generator).half()
        b = torch.randn(batch, K, N, generator=generator).half()
        ref = torch.bmm(a.float(), b.float())
        got = torch.bmm(a.to('rpu'), b.to('rpu')).cpu()
        metrics = require_same_semantic(got, ref, actual_dtype=torch.float16, cosine_min=0.999, relative_l2_max=0.02, row_relative_l2_max=0.02, label=f'eager bmm b{batch} M{M} K{K} N{N}')
        relative_l1 = ((ref - got.float()).abs().mean() / ref.abs().mean().clamp_min(1e-06)).item()
        assert metrics['cosine'] > 0.999 and relative_l1 < 0.02
