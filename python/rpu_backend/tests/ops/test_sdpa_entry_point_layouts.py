import pytest
import torch
import torch.nn.functional as F
import rpu_backend
(BATCH, HEADS, SEQ) = (1, 8, 64)

def _cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()

@pytest.mark.parametrize('head_dim', [64, 128])
def test_aten_entry_point_is_pytorch_layout(head_dim):
    torch.manual_seed(0)
    (q, k, v) = (torch.randn(BATCH, HEADS, SEQ, head_dim, dtype=torch.float16) for _ in range(3))
    ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True).half()
    got = F.scaled_dot_product_attention(q.to('rpu'), k.to('rpu'), v.to('rpu'), is_causal=True).cpu()
    assert got.shape == ref.shape
    cos = _cos(ref, got)
    assert cos > 0.999, f'aten SDPA head_dim={head_dim} was read as unified layout: cos={cos:.6f}'

@pytest.mark.parametrize('head_dim', [64, 128])
def test_unified_entry_point_is_unified_layout(head_dim):
    torch.manual_seed(0)
    (q, k, v) = (torch.randn(BATCH, SEQ, HEADS, head_dim, dtype=torch.float16) for _ in range(3))
    ref = F.scaled_dot_product_attention(q.float().transpose(1, 2).contiguous(), k.float().transpose(1, 2).contiguous(), v.float().transpose(1, 2).contiguous(), is_causal=True).half().transpose(1, 2).contiguous()
    got = torch.ops.rpu.scaled_dot_product_attention_unified(q.to('rpu'), k.to('rpu'), v.to('rpu'), None, 0.0, True, None, False).cpu()
    assert got.shape == ref.shape
    cos = _cos(ref, got)
    assert cos > 0.999, f'unified SDPA head_dim={head_dim} was read as PyTorch layout: cos={cos:.6f}'
