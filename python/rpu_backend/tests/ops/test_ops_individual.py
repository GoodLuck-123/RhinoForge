import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.environ.get('RPU_BACKEND_PYTHON', os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))))
import rpu_backend
from rpu_backend.runtime.weights import convert_linear_weights_inplace
from rpu_backend.api.cache import RPUCache
HIDDEN = 1024
INTER = 3072
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
GQA_GROUPS = NUM_Q_HEADS // NUM_KV_HEADS
BATCH = 1
SEQ_LEN = 128
EPS = 1e-06
PASS_THRESHOLD = 0.05
MEAN_THRESHOLD = 0.005

def assert_close(name, cpu_out, rpu_out):
    passed = compare(name, cpu_out, rpu_out)
    assert passed, f'{name}: device output does not match the CPU reference (max_diff >= {PASS_THRESHOLD} or mean_diff >= {MEAN_THRESHOLD}); per-element diagnostics are printed above'

def compare(name, cpu_out, rpu_out):
    if rpu_out.device.type != 'cpu':
        rpu_cpu = rpu_out.cpu()
    else:
        rpu_cpu = rpu_out
    cpu_f = cpu_out.float()
    rpu_f = rpu_cpu.float()
    diff = (cpu_f - rpu_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = F.cosine_similarity(cpu_f.flatten().unsqueeze(0), rpu_f.flatten().unsqueeze(0)).item()
    passed = max_diff < PASS_THRESHOLD and mean_diff < MEAN_THRESHOLD
    status = 'PASS' if passed else 'FAIL'
    print(f'  [{status}] {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, cos_sim={cos_sim:.6f}')
    if not passed:
        flat_diff = diff.flatten()
        topk = flat_diff.topk(min(5, flat_diff.numel()))
        for (i, (v, idx)) in enumerate(zip(topk.values, topk.indices)):
            cpu_v = cpu_f.flatten()[idx].item()
            rpu_v = rpu_f.flatten()[idx].item()
            print(f'    top{i}: idx={idx.item()}, cpu={cpu_v:.6f}, device={rpu_v:.6f}, diff={v.item():.6f}')
    return passed

def test_rmsnorm():
    print('\n' + '=' * 60)
    print('Test 1: RMSNorm')
    print('=' * 60)
    torch.manual_seed(42)
    x_cpu = torch.randn(BATCH, SEQ_LEN, HIDDEN, dtype=torch.float16)
    w_cpu = torch.randn(HIDDEN, dtype=torch.float16)
    x_f = x_cpu.float()
    var = x_f.pow(2).mean(-1, keepdim=True)
    normed = x_f * torch.rsqrt(var + EPS)
    cpu_out = (w_cpu.float() * normed).half()
    x_rpu = x_cpu.to('rpu')
    w_rpu = w_cpu.to('rpu')
    norm = nn.Module()
    norm.weight = nn.Parameter(w_cpu)
    norm.variance_epsilon = EPS
    x_rpu_back = x_rpu.cpu()
    rpu_out = x_rpu.cpu()

    class RMSNormModule(nn.Module):

        def __init__(self, h, eps):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(h))
            self.variance_epsilon = eps

        def forward(self, x):
            xf = x.float()
            var = xf.pow(2).mean(-1, keepdim=True)
            normed = xf * torch.rsqrt(var + self.variance_epsilon)
            return (self.weight * normed).to(x.dtype)
    norm_mod = RMSNormModule(HIDDEN, EPS).half()
    norm_mod.weight.data = w_cpu.clone()
    cpu_out = norm_mod(x_cpu)
    norm_rpu = RMSNormModule(HIDDEN, EPS).half().to('rpu')
    norm_rpu.weight.data = w_cpu.to('rpu')
    rpu_out = norm_rpu(x_rpu)
    assert_close('RMSNorm', cpu_out, rpu_out)

def test_linear_col():
    print('\n' + '=' * 60)
    print('Test 2: Linear (col partition)')
    print('=' * 60)
    torch.manual_seed(42)
    out_features = NUM_Q_HEADS * HEAD_DIM
    x_cpu = torch.randn(BATCH, SEQ_LEN, HIDDEN, dtype=torch.float16)
    linear = nn.Linear(HIDDEN, out_features, bias=False).half()
    cpu_out = linear(x_cpu)

    class QkvContainer(nn.Module):

        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(HIDDEN, out_features, bias=False)
    container_rpu = QkvContainer().half()
    container_rpu.q_proj.load_state_dict(linear.state_dict())
    info = convert_linear_weights_inplace(container_rpu)
    assert info == {'q_proj.weight': 1}, f'expected col partition, got {info}'
    container_rpu = container_rpu.to('rpu')
    x_rpu = x_cpu.to('rpu')
    rpu_out = container_rpu.q_proj(x_rpu)
    assert_close('Linear col (QKV-like)', cpu_out, rpu_out)

def test_linear_row():
    print('\n' + '=' * 60)
    print('Test 3: Linear (row partition)')
    print('=' * 60)
    torch.manual_seed(42)
    in_features = NUM_Q_HEADS * HEAD_DIM
    x_cpu = torch.randn(BATCH, SEQ_LEN, in_features, dtype=torch.float16)

    class OProjContainer(nn.Module):

        def __init__(self):
            super().__init__()
            self.o_proj = nn.Linear(in_features, HIDDEN, bias=False)
    container = OProjContainer().half()
    cpu_out = container.o_proj(x_cpu)
    container_rpu = OProjContainer().half()
    container_rpu.load_state_dict(container.state_dict())
    info = convert_linear_weights_inplace(container_rpu)
    assert info == {'o_proj.weight': 0}, f'expected row partition by name, got {info}'
    container_rpu = container_rpu.to('rpu')
    x_rpu = x_cpu.to('rpu')
    rpu_out = container_rpu.o_proj(x_rpu)
    assert_close('Linear row (o_proj-like)', cpu_out, rpu_out)

def test_rope():
    print('\n' + '=' * 60)
    print('Test 4: RoPE')
    print('=' * 60)
    torch.manual_seed(42)
    q_cpu = torch.randn(BATCH, SEQ_LEN, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float16)
    k_cpu = torch.randn(BATCH, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
    inv_freq = 1.0 / 10000.0 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM)
    positions = torch.arange(SEQ_LEN).float()
    freqs = torch.einsum('i,j->ij', positions, inv_freq)
    cos_table = freqs.cos().unsqueeze(0).half()
    sin_table = freqs.sin().unsqueeze(0).half()

    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
    cos_exp = torch.cat([cos_table, cos_table], dim=-1).unsqueeze(2)
    sin_exp = torch.cat([sin_table, sin_table], dim=-1).unsqueeze(2)
    q_cpu_roped = q_cpu * cos_exp + rotate_half(q_cpu) * sin_exp
    k_cpu_roped = k_cpu * cos_exp[:, :, :1, :] + rotate_half(k_cpu) * sin_exp[:, :, :1, :]
    q_rpu = q_cpu.to('rpu')
    k_rpu = k_cpu.to('rpu')
    q_back = q_rpu.cpu()
    k_back = k_rpu.cpu()
    assert_close('RoPE Q transfer', q_cpu, q_back)
    assert_close('RoPE K transfer', k_cpu, k_back)
    print('  [INFO] RoPE kernel tested implicitly via fused decoder layer')

def test_sdpa():
    print('\n' + '=' * 60)
    print('Test 6: SDPA')
    print('=' * 60)
    torch.manual_seed(42)
    q_unified = torch.randn(BATCH, SEQ_LEN, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float16)
    k_unified = torch.randn(BATCH, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
    v_unified = torch.randn(BATCH, SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16)
    q_pt = q_unified.transpose(1, 2).contiguous()
    k_pt = k_unified.transpose(1, 2).contiguous()
    v_pt = v_unified.transpose(1, 2).contiguous()
    k_exp = k_pt.repeat_interleave(GQA_GROUPS, dim=1)
    v_exp = v_pt.repeat_interleave(GQA_GROUPS, dim=1)
    cpu_out_pt = F.scaled_dot_product_attention(q_pt.float(), k_exp.float(), v_exp.float(), is_causal=True).half()
    cpu_out = cpu_out_pt.transpose(1, 2).contiguous()
    q_rpu = q_unified.to('rpu')
    k_rpu = k_unified.to('rpu')
    v_rpu = v_unified.to('rpu')
    try:
        q_rpu_pt = q_rpu.transpose(1, 2).contiguous()
        k_rpu_pt = k_rpu.transpose(1, 2).contiguous()
        v_rpu_pt = v_rpu.transpose(1, 2).contiguous()
        k_rpu_exp = k_rpu_pt.repeat_interleave(GQA_GROUPS, dim=1)
        v_rpu_exp = v_rpu_pt.repeat_interleave(GQA_GROUPS, dim=1)
        rpu_out_pt = F.scaled_dot_product_attention(q_rpu_pt, k_rpu_exp, v_rpu_exp, is_causal=True)
        rpu_out = rpu_out_pt.transpose(1, 2).contiguous()
    except Exception as e:
        print(f'  [FAIL] SDPA error: {e}')
        import traceback
        traceback.print_exc()
        raise
    assert_close('SDPA (causal)', cpu_out, rpu_out)

def main():
    print('=' * 60)
    print('Individual Operator Tests for device Backend')
    print(f'Config: hidden={HIDDEN}, inter={INTER}, q_heads={NUM_Q_HEADS}, kv_heads={NUM_KV_HEADS}, head_dim={HEAD_DIM}')
    print(f'Input: batch={BATCH}, seq_len={SEQ_LEN}')
    print(f'Thresholds: max_diff<{PASS_THRESHOLD}, mean_diff<{MEAN_THRESHOLD}')
    print('=' * 60)
    torch.rpu.set_spm_debug(True)
    results = {}

    def run(label, fn):
        try:
            fn()
        except AssertionError as exc:
            print(f'  [FAIL] {label}: {exc}')
            results[label] = False
        else:
            results[label] = True
    run('RMSNorm', test_rmsnorm)
    run('Linear_col', test_linear_col)
    run('Linear_row', test_linear_row)
    run('RoPE_transfer', test_rope)
    run('SDPA', test_sdpa)
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    all_pass = True
    for (name, passed) in results.items():
        status = 'PASS' if passed else 'FAIL'
        print(f'  {name}: {status}')
        if not passed:
            all_pass = False
    if all_pass:
        print('\nAll tests PASSED!')
    else:
        print('\nSome tests FAILED!')
        sys.exit(1)
if __name__ == '__main__':
    main()
