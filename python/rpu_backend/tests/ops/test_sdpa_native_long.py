import pytest
import torch
import rpu_backend
from rpu_backend.tests._seed import input_generator
DEVICE = 'rpu'

def _unified_qkv(b, h, hkv, s, hd):
    g = input_generator()
    q = torch.randn(b, s, h, hd, generator=g, dtype=torch.float16)
    k = torch.randn(b, s, hkv, hd, generator=g, dtype=torch.float16)
    v = torch.randn(b, s, hkv, hd, generator=g, dtype=torch.float16)
    return (q, k, v)

def _cpu_reference_unified(q, k, v, is_causal, enable_gqa=False):
    out = torch.nn.functional.scaled_dot_product_attention(q.float().transpose(1, 2).contiguous(), k.float().transpose(1, 2).contiguous(), v.float().transpose(1, 2).contiguous(), is_causal=is_causal, enable_gqa=enable_gqa)
    return out.to(torch.float16).transpose(1, 2).contiguous()

@pytest.mark.parametrize('seq_q', [16, 32, 64, 128, 256, 512])
def test_mha_ltm_matches_cpu_reference(seq_q):
    (q, k, v) = _unified_qkv(b=1, h=8, hkv=8, s=seq_q, hd=128)
    out = torch.ops.rpu.scaled_dot_product_attention_unified(q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), None, 0.0, True, None, False)
    ref = _cpu_reference_unified(q, k, v, is_causal=True)
    assert out.shape == (1, seq_q, 8, 128)
    assert (out.to('cpu') != 0).any(dim=2).any(dim=1).all(), f'seq_q={seq_q}: some output rows are entirely zero'
    diff = (out.to('cpu') - ref).abs().max().item()
    assert diff < 0.1, f'MHA seq_q={seq_q} diverges from CPU fp32: max_abs={diff}'

def test_gqa_is_honoured_not_silently_dropped():
    (q, k, v) = _unified_qkv(b=1, h=32, hkv=8, s=512, hd=128)
    out = torch.ops.rpu.scaled_dot_product_attention_unified(q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), None, 0.0, True, None, True)
    ref = _cpu_reference_unified(q, k, v, is_causal=True, enable_gqa=True)
    assert out.shape == (1, 512, 32, 128)
    diff = (out.to('cpu') - ref).abs().max().item()
    assert diff < 0.1, f'GQA output diverges from CPU fp32: max_abs={diff}'
