import torch
import rpu_backend
from rpu_backend.tests._seed import input_generator
import math

def ceil_div(a, b):
    return (a + b - 1) // b

def align(x, alignment):
    return (x + alignment - 1) // alignment * alignment

def transform_value_cpu_reference(src, num_cores=8):
    CHUNK = 16
    (batch, seq_len, num_kv_heads, head_dim) = src.shape
    nkv_head_chunk = min(num_cores, num_kv_heads)
    nkv_head_vx = ceil_div(num_kv_heads, nkv_head_chunk)
    head_dim_vx = head_dim // CHUNK
    seq_len_pad = align(seq_len, CHUNK)
    seq_v_vx = seq_len_pad // CHUNK
    dst = torch.zeros(batch, seq_v_vx, nkv_head_vx, head_dim_vx, nkv_head_chunk, CHUNK, CHUNK, dtype=src.dtype)
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
                    dst[b, seq_vx_idx, nkv_vx_idx, dim_vx_idx, nkv_chunk_idx, dim_chunk_idx, seq_chunk_idx] = src[b, s, kv_h, d]
    return dst

def test_vcache_layout(batch, seq_len, num_kv_heads, head_dim, position=0, verbose=True):
    CHUNK = 16
    num_cores = 8
    nkv_head_chunk = min(num_cores, num_kv_heads)
    nkv_head_vx = ceil_div(num_kv_heads, nkv_head_chunk)
    head_dim_vx = head_dim // CHUNK
    seq_len_pad = align(seq_len, CHUNK)
    seq_v_vx = seq_len_pad // CHUNK
    if verbose:
        print(f"\n{'=' * 60}")
        print(f'Test: batch={batch}, seq_len={seq_len}, num_kv_heads={num_kv_heads}, head_dim={head_dim}')
        print(f'  -> seq_len_pad={seq_len_pad}, seq_v_vx={seq_v_vx}')
        print(f'  -> nkv_head_chunk={nkv_head_chunk}, nkv_head_vx={nkv_head_vx}')
        print(f'  -> head_dim_vx={head_dim_vx}')
        print(f'  -> Output shape: [{batch}, {seq_v_vx}, {nkv_head_vx}, {head_dim_vx}, {nkv_head_chunk}, {CHUNK}, {CHUNK}]')
    if num_kv_heads % num_cores != 0:
        if verbose:
            print(f'  SKIP: num_kv_heads ({num_kv_heads}) must be divisible by num_cores ({num_cores})')
        return (None, None, None)
    if seq_len % CHUNK != 0:
        if verbose:
            print(f'  SKIP: seq_len ({seq_len}) must be divisible by CHUNK ({CHUNK}) for current kernel')
        return (None, None, None)
    v_input_cpu = torch.randn(batch, seq_len, num_kv_heads, head_dim, generator=input_generator(), dtype=torch.float32)
    v_input = v_input_cpu.half().to('rpu')
    v_cache = torch.zeros(batch, seq_v_vx, nkv_head_vx, head_dim_vx, nkv_head_chunk, CHUNK, CHUNK, dtype=torch.float16, device='rpu')
    torch.rpu.insert_vcache(v_input, v_cache, position)
    v_cache_ref = transform_value_cpu_reference(v_input_cpu.half(), num_cores)
    v_cache_cpu = v_cache.cpu()
    diff = (v_cache_cpu.float() - v_cache_ref.float()).abs()
    max_diff = diff.max().item()
    num_mismatches = (diff > 0.01).sum().item()
    passed = max_diff < 0.01
    if verbose:
        print(f'  -> Max diff: {max_diff:.6f}')
        print(f'  -> Mismatches: {num_mismatches}')
        print(f"  -> {('PASSED' if passed else 'FAILED')}")
    return (passed, max_diff, num_mismatches)

def test_specific_indices(batch, seq_len, num_kv_heads, head_dim):
    CHUNK = 16
    num_cores = 8
    nkv_head_chunk = min(num_cores, num_kv_heads)
    nkv_head_vx = ceil_div(num_kv_heads, nkv_head_chunk)
    head_dim_vx = head_dim // CHUNK
    seq_len_pad = align(seq_len, CHUNK)
    seq_v_vx = seq_len_pad // CHUNK
    print(f'\nIndex mapping test for shape [{batch}, {seq_len}, {num_kv_heads}, {head_dim}]:')
    print(f'  nkv_head_vx={nkv_head_vx}, nkv_head_chunk={nkv_head_chunk}')
    v_input_cpu = torch.zeros(batch, seq_len, num_kv_heads, head_dim, dtype=torch.float16)
    for s in range(min(seq_len, 4)):
        for h in range(min(num_kv_heads, 8)):
            for d in range(min(head_dim, 4)):
                v_input_cpu[0, s, h, d] = s * 1000 + h * 10 + d * 0.1
    v_input = v_input_cpu.to('rpu')
    v_cache = torch.zeros(batch, seq_v_vx, nkv_head_vx, head_dim_vx, nkv_head_chunk, CHUNK, CHUNK, dtype=torch.float16, device='rpu')
    torch.rpu.insert_vcache(v_input, v_cache, 0)
    v_cache_cpu = v_cache.cpu()
    test_positions = [(0, 0, 0), (0, 1, 0), (0, 4, 0), (1, 0, 0), (0, 0, 16) if head_dim >= 32 else (0, 0, 0)]
    for (s, h, d) in test_positions:
        if s >= seq_len or h >= num_kv_heads or d >= head_dim:
            continue
        nkv_vx_idx = h % nkv_head_vx
        nkv_chunk_idx = h // nkv_head_vx
        seq_vx_idx = s // CHUNK
        seq_chunk_idx = s % CHUNK
        dim_vx_idx = d // CHUNK
        dim_chunk_idx = d % CHUNK
        input_val = v_input_cpu[0, s, h, d].item()
        output_val = v_cache_cpu[0, seq_vx_idx, nkv_vx_idx, dim_vx_idx, nkv_chunk_idx, dim_chunk_idx, seq_chunk_idx].item()
        match = abs(input_val - output_val) < 0.1
        print(f"  input[0,{s},{h},{d}]={input_val:.1f} -> output[0,{seq_vx_idx},{nkv_vx_idx},{dim_vx_idx},{nkv_chunk_idx},{dim_chunk_idx},{seq_chunk_idx}]={output_val:.1f} {('OK' if match else 'MISMATCH')}")

def main():
    print('=' * 60)
    print('Comprehensive V-Cache Layout Verification')
    print('=' * 60)
    test_cases = [(1, 32, 32, 64), (1, 16, 32, 64), (1, 32, 8, 64), (1, 32, 16, 64), (1, 32, 32, 32), (1, 32, 32, 128), (1, 64, 32, 64), (1, 128, 32, 64), (1, 48, 32, 64), (1, 32, 64, 64)]
    results = []
    for (batch, seq_len, num_kv_heads, head_dim) in test_cases:
        (passed, max_diff, num_mismatches) = test_vcache_layout(batch, seq_len, num_kv_heads, head_dim)
        if passed is not None:
            results.append((batch, seq_len, num_kv_heads, head_dim, passed, max_diff))
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    passed_count = sum((1 for r in results if r[4]))
    total_count = len(results)
    for (batch, seq_len, num_kv_heads, head_dim, passed, max_diff) in results:
        status = 'PASS' if passed else 'FAIL'
        print(f'  [{status}] ({batch}, {seq_len}, {num_kv_heads}, {head_dim}) max_diff={max_diff:.6f}')
    print(f'\nTotal: {passed_count}/{total_count} passed')
    print('\n' + '=' * 60)
    print('Detailed Index Mapping Test')
    print('=' * 60)
    test_specific_indices(1, 32, 32, 64)
    return passed_count == total_count
if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
