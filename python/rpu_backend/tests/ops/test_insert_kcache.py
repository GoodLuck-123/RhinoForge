import torch
import numpy as np
import rpu_backend
from rpu_backend.tests._seed import input_generator

def transform_key_cpu(k_input, num_cores=8):
    (batch, seq_len, num_kv_heads, head_dim) = k_input.shape
    CHUNK = 16
    nkv_head_chunk = min(num_cores, num_kv_heads)
    nkv_head_vx = (num_kv_heads + num_cores - 1) // num_cores
    head_dim_vx = head_dim // CHUNK
    seq_len_pad = (seq_len + CHUNK - 1) // CHUNK * CHUNK
    seq_k_vx = seq_len_pad // CHUNK
    output = torch.zeros(batch, seq_k_vx, nkv_head_vx, head_dim_vx, nkv_head_chunk, CHUNK, CHUNK, dtype=k_input.dtype, device='cpu')
    k_cpu = k_input.cpu()
    for b in range(batch):
        for kv_h in range(num_kv_heads):
            nkv_vx_idx = kv_h % nkv_head_vx
            nkv_chunk_idx = kv_h // nkv_head_vx
            for s in range(seq_len):
                seq_vx_idx = s // CHUNK
                seq_chunk_idx = s % CHUNK
                for d in range(head_dim):
                    dim_vx_idx = d // CHUNK
                    dim_chunk_idx = d % CHUNK
                    val = k_cpu[b, s, kv_h, d]
                    output[b, seq_vx_idx, nkv_vx_idx, dim_vx_idx, nkv_chunk_idx, seq_chunk_idx, dim_chunk_idx] = val
    return output

def test_insert_kcache(batch=1, seq_len=19, num_kv_heads=8, head_dim=128, num_cores=8):
    print(f"\n{'=' * 60}")
    print(f'Testing K-Cache Insert: batch={batch}, seq={seq_len}, kv_heads={num_kv_heads}, dim={head_dim}')
    print(f"{'=' * 60}")
    k_input = torch.randn(batch, seq_len, num_kv_heads, head_dim, generator=input_generator(), dtype=torch.float16)
    print('Running CPU reference transform...')
    k_cache_cpu = transform_key_cpu(k_input, num_cores)
    print(f'  CPU output shape: {k_cache_cpu.shape}')
    k_input_rpu = k_input.to('rpu')
    CHUNK = 16
    nkv_head_chunk = min(num_cores, num_kv_heads)
    nkv_head_vx = (num_kv_heads + num_cores - 1) // num_cores
    head_dim_vx = head_dim // CHUNK
    seq_len_pad = (seq_len + CHUNK - 1) // CHUNK * CHUNK
    seq_k_vx = seq_len_pad // CHUNK
    k_cache_shape = (batch, seq_k_vx, nkv_head_vx, head_dim_vx, nkv_head_chunk, CHUNK, CHUNK)
    k_cache_rpu = torch.zeros(k_cache_shape, dtype=torch.float16, device='rpu')
    print(f'Running device kernel...')
    print(f'  Input shape: {k_input_rpu.shape}')
    print(f'  Output shape: {k_cache_rpu.shape}')
    torch.rpu.insert_kcache(k_input_rpu, k_cache_rpu, 0)
    k_cache_result = k_cache_rpu.cpu()
    print('\nComparing results...')
    assert not torch.isnan(k_cache_result).any(), 'device output contains NaN!'
    assert not torch.isinf(k_cache_result).any(), 'device output contains Inf!'
    diff = (k_cache_result.float() - k_cache_cpu.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    max_idx = diff.argmax()
    max_idx_tuple = np.unravel_index(max_idx.item(), diff.shape)
    print(f'  Max diff: {max_diff:.6f} at {max_idx_tuple}')
    print(f'  Mean diff: {mean_diff:.6f}')
    print(f'  CPU value at max diff: {k_cache_cpu[max_idx_tuple].item():.6f}')
    print(f'  device value at max diff: {k_cache_result[max_idx_tuple].item():.6f}')
    print('\nSpot check (first few elements):')
    for idx in [(0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 0, 1), (0, 0, 0, 0, 0, 1, 0), (0, 0, 0, 0, 1, 0, 0)]:
        try:
            cpu_val = k_cache_cpu[idx].item()
            rpu_val = k_cache_result[idx].item()
            print(f'  {idx}: CPU={cpu_val:.4f}, device={rpu_val:.4f}, diff={abs(cpu_val - rpu_val):.6f}')
        except IndexError:
            pass
    tolerance = 0.001
    if max_diff >= tolerance:
        print(f'\n  FAIL: Max diff {max_diff:.6f} >= tolerance {tolerance}')
        print('\nDebug: Checking data pattern...')
        if (k_cache_result == 0).all():
            print('  WARNING: device output is all zeros!')
        nonzero_cpu = (k_cache_cpu != 0).sum().item()
        nonzero_rpu = (k_cache_result != 0).sum().item()
        print(f'  Non-zero elements: CPU={nonzero_cpu}, device={nonzero_rpu}')
    assert max_diff < tolerance, f'Max diff {max_diff:.6f} >= tolerance {tolerance}'
    print(f'\n  PASS: Max diff {max_diff:.6f} < tolerance {tolerance}')

def test_simple_pattern():
    print('\n' + '=' * 60)
    print('Testing with simple sequential pattern')
    print('=' * 60)
    (batch, seq_len, num_kv_heads, head_dim) = (1, 16, 8, 128)
    k_input = torch.arange(batch * seq_len * num_kv_heads * head_dim, dtype=torch.float16)
    k_input = k_input.reshape(batch, seq_len, num_kv_heads, head_dim)
    k_cache_cpu = transform_key_cpu(k_input, num_cores=8)
    k_input_rpu = k_input.to('rpu')
    k_cache_rpu = torch.zeros_like(k_cache_cpu, device='rpu')
    print(f'Input shape: {k_input.shape}')
    print(f'Output shape: {k_cache_rpu.shape}')
    torch.rpu.insert_kcache(k_input_rpu, k_cache_rpu, 0)
    k_cache_result = k_cache_rpu.cpu()
    diff = (k_cache_result.float() - k_cache_cpu.float()).abs()
    max_diff = diff.max().item()
    print(f'Max diff: {max_diff:.6f}')
    if max_diff > 0.001:
        print('\nFirst 10 mismatches:')
        mismatch_count = 0
        for idx in np.ndindex(k_cache_cpu.shape):
            cpu_val = k_cache_cpu[idx].item()
            rpu_val = k_cache_result[idx].item()
            if abs(cpu_val - rpu_val) > 0.001:
                print(f'  {idx}: CPU={cpu_val:.1f}, device={rpu_val:.1f}')
                mismatch_count += 1
                if mismatch_count >= 10:
                    break
    assert max_diff <= 0.001, f'Max diff {max_diff:.6f} > 1e-3'
    print('PASS')
if __name__ == '__main__':
    import sys
    print('Testing rpu_launch_llama_insert_kcache kernel')
    print('=' * 60)
    _cases = [('simple sequential pattern', test_simple_pattern, {}), ('seq_len=19 (v2 insert path)', test_insert_kcache, dict(batch=1, seq_len=19, num_kv_heads=8, head_dim=128)), ('seq_len=32 (v16 insert path)', test_insert_kcache, dict(batch=1, seq_len=32, num_kv_heads=8, head_dim=128))]
    _failed = []
    for (_name, _fn, _kwargs) in _cases:
        try:
            _fn(**_kwargs)
        except AssertionError as _exc:
            print(f'\n  FAILED [{_name}]: {_exc}')
            _failed.append(_name)
    sys.exit(1 if _failed else 0)
