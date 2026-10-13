import os
import pytest
import torch

@pytest.mark.skipif(os.environ.get('RPU_HW_AVAILABLE') != '1', reason='requires RPU hardware (set RPU_HW_AVAILABLE=1)')
def test_layernorm_eager_nonzero_matches_cpu(record_property):
    import rpu_backend
    from rpu_backend.tests.conformance._numeric import require_same_semantic
    torch.manual_seed(0)
    N = 4
    eps = 1e-05
    for C in (768, 256):
        x = torch.randn(N, C, dtype=torch.float16)
        w = torch.randn(C, dtype=torch.float16)
        b = torch.randn(C, dtype=torch.float16)
        ref = torch.layer_norm(x.float(), (C,), w.float(), b.float(), eps)
        got = torch.layer_norm(x.to('rpu'), (C,), w.to('rpu'), b.to('rpu'), eps).cpu()
        require_same_semantic(got, ref, actual_dtype=torch.float16, cosine_min=0.99, relative_l2_max=0.02, row_relative_l2_max=0.02, label=f'eager affine layer_norm C={C}')
        assert got.abs().sum().item() > 0, f'device eager layer_norm C={C} collapsed to all-zeros (reg ABI bug)'
        cos = torch.nn.functional.cosine_similarity(ref.half().flatten().float(), got.flatten().float(), dim=0).item()
        assert cos > 0.99, f'device eager layer_norm C={C} cos={cos:.5f} vs CPU (expected >0.99)'
        print(f'[PASS] eager layer_norm C={C}: nonzero + cos={cos:.6f} vs CPU fp32')

@pytest.mark.skipif(os.environ.get('RPU_HW_AVAILABLE') != '1', reason='requires RPU hardware (set RPU_HW_AVAILABLE=1)')
def test_eager_mean_last_dim_matches_fp32(record_property):
    import rpu_backend
    from rpu_backend.tests._seed import input_generator
    from rpu_backend.tests.conformance._numeric import require_same_semantic
    generator = input_generator()
    for shape in ((16, 32), (2, 16, 256)):
        x = (0.25 + 1.75 * torch.rand(shape, generator=generator)).half()
        for keepdim in (False, True):
            ref = torch.mean(x.float(), dim=-1, keepdim=keepdim)
            got = torch.mean(x.to('rpu'), dim=-1, keepdim=keepdim).cpu()
            require_same_semantic(got, ref, actual_dtype=torch.float16, cosine_min=0.999, relative_l2_max=0.01, row_relative_l2_max=0.01, label=f'eager mean FP16 {shape} keepdim={keepdim}')
if __name__ == '__main__':
    test_layernorm_eager_nonzero_matches_cpu(lambda *_: None)
    test_eager_mean_last_dim_matches_fp32(lambda *_: None)
