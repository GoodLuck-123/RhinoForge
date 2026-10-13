import torch
import rpu_backend
from rpu_backend.tests._seed import input_generator

def precompute_freqs_cis(dim: int, end: int, theta: float=10000.0):
    freqs = 1.0 / theta ** (torch.arange(0, dim, 2)[:dim // 2].float() / dim)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return (freqs_cos, freqs_sin)

def expand_cos_sin_for_cpu(freqs_cos, freqs_sin, head_dim):
    cos = freqs_cos.view(-1, 1, head_dim // 2).expand(-1, 2, -1).reshape(-1, head_dim)
    sin = freqs_sin.view(-1, 1, head_dim // 2).expand(-1, 2, -1).reshape(-1, head_dim)
    return (cos, sin)

def rotate_half_cpu(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope_single_cpu(x, freqs_cos, freqs_sin, pos, head_dim):
    (cos, sin) = expand_cos_sin_for_cpu(freqs_cos, freqs_sin, head_dim)
    cos_pos = cos[pos]
    sin_pos = sin[pos]
    return x * cos_pos + rotate_half_cpu(x) * sin_pos

def test_rope_single_position():
    print('=' * 70)
    print('Test: Single Position RoPE (llama_rope kernel)')
    print('=' * 70)
    heads_num = 8
    head_dim = 512
    pos = 123
    max_seq_len = 8192
    (freqs_cos, freqs_sin) = precompute_freqs_cis(head_dim, max_seq_len)
    x = torch.randn(heads_num, head_dim, generator=input_generator(), dtype=torch.float16)
    y_cpu = apply_rope_single_cpu(x.float(), freqs_cos.float(), freqs_sin.float(), pos, head_dim).half()
    q = x.unsqueeze(0).unsqueeze(0)
    k = x.unsqueeze(0).unsqueeze(0)
    position_ids = torch.tensor([[pos]])
    q_rpu = q.to('rpu')
    k_rpu = k.to('rpu')
    freqs_cos_rpu = freqs_cos.half().to('rpu')
    freqs_sin_rpu = freqs_sin.half().to('rpu')
    position_ids_rpu = position_ids.to('rpu')
    (q_out_rpu, k_out_rpu) = torch.rpu.apply_rotary_pos_emb(q_rpu, k_rpu, freqs_cos_rpu, freqs_sin_rpu, position_ids_rpu)
    y_rpu = q_out_rpu[0, 0].to('cpu')
    diff = (y_cpu - y_rpu).abs().max().item()
    mean_diff = (y_cpu - y_rpu).abs().mean().item()
    print(f'  Input shape: [{heads_num}, {head_dim}]')
    print(f'  freqs_cos/sin shape: [{max_seq_len}, {head_dim // 2}]')
    print(f'  Position: {pos}')
    print(f'  max_diff: {diff:.6f}, mean_diff: {mean_diff:.6f}')
    passed = diff < 0.1
    print(f"  Result: {('PASSED' if passed else 'FAILED')}")
    if not passed:
        print(f'  CPU output sample: {y_cpu.flatten()[:5]}')
        print(f'  device output sample: {y_rpu.flatten()[:5]}')
    assert passed

def test_rope_small():
    print('\n' + '=' * 70)
    print('Test: Small Shape RoPE')
    print('=' * 70)
    heads_num = 4
    head_dim = 64
    pos = 0
    max_seq_len = 128
    (freqs_cos, freqs_sin) = precompute_freqs_cis(head_dim, max_seq_len)
    x = torch.randn(heads_num, head_dim, generator=input_generator(), dtype=torch.float16)
    y_cpu = apply_rope_single_cpu(x.float(), freqs_cos.float(), freqs_sin.float(), pos, head_dim).half()
    q = x.unsqueeze(0).unsqueeze(0)
    k = x.unsqueeze(0).unsqueeze(0)
    position_ids = torch.tensor([[pos]])
    q_rpu = q.to('rpu')
    k_rpu = k.to('rpu')
    freqs_cos_rpu = freqs_cos.half().to('rpu')
    freqs_sin_rpu = freqs_sin.half().to('rpu')
    position_ids_rpu = position_ids.to('rpu')
    (q_out_rpu, _) = torch.rpu.apply_rotary_pos_emb(q_rpu, k_rpu, freqs_cos_rpu, freqs_sin_rpu, position_ids_rpu)
    y_rpu = q_out_rpu[0, 0].to('cpu')
    diff = (y_cpu - y_rpu).abs().max().item()
    print(f'  Input shape: [{heads_num}, {head_dim}]')
    print(f'  freqs_cos/sin shape: [{max_seq_len}, {head_dim // 2}]')
    print(f'  Position: {pos}')
    print(f'  max_diff: {diff:.6f}')
    passed = diff < 0.1
    print(f"  Result: {('PASSED' if passed else 'FAILED')}")
    if not passed:
        print(f'  CPU output sample: {y_cpu.flatten()[:8]}')
        print(f'  device output sample: {y_rpu.flatten()[:8]}')
        print(f'  Diff sample: {(y_cpu - y_rpu).abs().flatten()[:8]}')
    assert passed

def test_rope_sequence():
    print('\n' + '=' * 70)
    print('Test: Sequence RoPE (multiple positions)')
    print('=' * 70)
    batch = 1
    seq_len = 4
    num_heads = 4
    head_dim = 64
    max_seq_len = 128
    (freqs_cos, freqs_sin) = precompute_freqs_cis(head_dim, max_seq_len)
    _g = input_generator()
    q = torch.randn(batch, seq_len, num_heads, head_dim, generator=_g, dtype=torch.float16)
    k = torch.randn(batch, seq_len, num_heads, head_dim, generator=_g, dtype=torch.float16)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    q_cpu_out = torch.zeros_like(q)
    k_cpu_out = torch.zeros_like(k)
    for b in range(batch):
        for s in range(seq_len):
            pos = position_ids[b, s].item()
            q_cpu_out[b, s] = apply_rope_single_cpu(q[b, s].float(), freqs_cos.float(), freqs_sin.float(), pos, head_dim).half()
            k_cpu_out[b, s] = apply_rope_single_cpu(k[b, s].float(), freqs_cos.float(), freqs_sin.float(), pos, head_dim).half()
    q_rpu = q.to('rpu')
    k_rpu = k.to('rpu')
    freqs_cos_rpu = freqs_cos.half().to('rpu')
    freqs_sin_rpu = freqs_sin.half().to('rpu')
    position_ids_rpu = position_ids.to('rpu')
    (q_out_rpu, k_out_rpu) = torch.rpu.apply_rotary_pos_emb(q_rpu, k_rpu, freqs_cos_rpu, freqs_sin_rpu, position_ids_rpu)
    q_out = q_out_rpu.to('cpu')
    k_out = k_out_rpu.to('cpu')
    q_diff = (q_cpu_out - q_out).abs().max().item()
    k_diff = (k_cpu_out - k_out).abs().max().item()
    print(f'  Shape: [{batch}, {seq_len}, {num_heads}, {head_dim}]')
    print(f'  freqs_cos/sin shape: [{max_seq_len}, {head_dim // 2}]')
    print(f'  position_ids: {position_ids.tolist()}')
    print(f'  q max_diff: {q_diff:.6f}')
    print(f'  k max_diff: {k_diff:.6f}')
    passed = q_diff < 0.1 and k_diff < 0.1
    print(f"  Result: {('PASSED' if passed else 'FAILED')}")
    assert passed

def test_rope_qwen3_shape(record_property):
    print('\n' + '=' * 70)
    print('Test: Qwen3-0.6B Shape')
    print('=' * 70)
    batch = 1
    seq_len = 17
    num_heads = 16
    num_kv_heads = 8
    head_dim = 64
    max_seq_len = 128
    (freqs_cos, freqs_sin) = precompute_freqs_cis(head_dim, max_seq_len)
    _g = input_generator()
    q = torch.randn(batch, seq_len, num_heads, head_dim, generator=_g, dtype=torch.float16)
    k = torch.randn(batch, seq_len, num_kv_heads, head_dim, generator=_g, dtype=torch.float16)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    q_cpu_out = torch.zeros_like(q)
    k_cpu_out = torch.zeros_like(k)
    for b in range(batch):
        for s in range(seq_len):
            pos = position_ids[b, s].item()
            q_cpu_out[b, s] = apply_rope_single_cpu(q[b, s].float(), freqs_cos.float(), freqs_sin.float(), pos, head_dim).half()
            k_cpu_out[b, s] = apply_rope_single_cpu(k[b, s].float(), freqs_cos.float(), freqs_sin.float(), pos, head_dim).half()
    q_rpu = q.to('rpu')
    k_rpu = k.to('rpu')
    freqs_cos_rpu = freqs_cos.half().to('rpu')
    freqs_sin_rpu = freqs_sin.half().to('rpu')
    position_ids_rpu = position_ids.to('rpu')
    (q_out_rpu, k_out_rpu) = torch.rpu.apply_rotary_pos_emb(q_rpu, k_rpu, freqs_cos_rpu, freqs_sin_rpu, position_ids_rpu)
    q_out = q_out_rpu.to('cpu')
    k_out = k_out_rpu.to('cpu')
    for (actual, expected) in ((q_out, q_cpu_out), (k_out, k_cpu_out)):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert actual.numel() and torch.isfinite(actual).all() and torch.isfinite(expected).all()
    q_diff = (q_cpu_out - q_out).abs().max().item()
    k_diff = (k_cpu_out - k_out).abs().max().item()
    print(f'  Query shape: [{batch}, {seq_len}, {num_heads}, {head_dim}]')
    print(f'  Key shape: [{batch}, {seq_len}, {num_kv_heads}, {head_dim}]')
    print(f'  freqs_cos/sin shape: [{max_seq_len}, {head_dim // 2}]')
    print(f'  q max_diff: {q_diff:.6f}')
    print(f'  k max_diff: {k_diff:.6f}')
    passed = q_diff < 0.1 and k_diff < 0.1
    print(f"  Result: {('PASSED' if passed else 'FAILED')}")
    assert passed
if __name__ == '__main__':
    import sys
    results = []
    results.append(('small', test_rope_small()))
    results.append(('single_position', test_rope_single_position()))
    results.append(('sequence', test_rope_sequence()))
    results.append(('qwen3_shape', test_rope_qwen3_shape(lambda *_: None)))
    print('\n' + '=' * 70)
    print('Final Summary')
    print('=' * 70)
    passed_count = sum((1 for (_, passed) in results if passed))
    for (name, passed) in results:
        print(f"  {name}: {('PASS' if passed else 'FAIL')}")
    print(f'\nTotal: {passed_count}/{len(results)} passed')
    sys.exit(0 if passed_count == len(results) else 1)
