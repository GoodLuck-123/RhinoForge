import torch
import rpu_backend
from rpu_backend.tests._seed import input_generator

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

def test_padding_case(seq_len, num_kv_heads=32, head_dim=64):
    CHUNK = 16
    num_cores = 8
    batch = 1
    seq_len_pad = align(seq_len, CHUNK)
    seq_v_vx = seq_len_pad // CHUNK
    nkv_head_chunk = min(num_cores, num_kv_heads)
    nkv_head_vx = ceil_div(num_kv_heads, nkv_head_chunk)
    head_dim_vx = head_dim // CHUNK
    print(f"\n{'=' * 60}")
    print(f'Testing seq_len={seq_len} (padded to {seq_len_pad})')
    print(f'  Output shape: [{batch}, {seq_v_vx}, {nkv_head_vx}, {head_dim_vx}, {nkv_head_chunk}, {CHUNK}, {CHUNK}]')
    if num_kv_heads % num_cores != 0:
        print(f'  SKIP: num_kv_heads must be divisible by {num_cores}')
        return None
    v_input_cpu = torch.randn(batch, seq_len, num_kv_heads, head_dim, generator=input_generator(), dtype=torch.float32)
    if seq_len % CHUNK != 0:
        pad_size = seq_len_pad - seq_len
        v_input_padded = torch.zeros(batch, seq_len_pad, num_kv_heads, head_dim, dtype=torch.float32)
        v_input_padded[:, :seq_len, :, :] = v_input_cpu
        v_input = v_input_padded.half().to('rpu')
        print(f'  Input padded from {seq_len} to {seq_len_pad}')
    else:
        v_input = v_input_cpu.half().to('rpu')
    v_cache = torch.zeros(batch, seq_v_vx, nkv_head_vx, head_dim_vx, nkv_head_chunk, CHUNK, CHUNK, dtype=torch.float16, device='rpu')
    try:
        torch.rpu.insert_vcache(v_input, v_cache, 0)
        if seq_len % CHUNK != 0:
            v_cache_ref = transform_value_cpu_reference(v_input_padded.half(), num_cores)
        else:
            v_cache_ref = transform_value_cpu_reference(v_input_cpu.half(), num_cores)
        v_cache_cpu = v_cache.cpu()
        diff = (v_cache_cpu.float() - v_cache_ref.float()).abs()
        max_diff = diff.max().item()
        num_mismatches = (diff > 0.01).sum().item()
        passed = max_diff < 0.01
        print(f'  Max diff: {max_diff:.6f}')
        print(f'  Mismatches: {num_mismatches}')
        print(f"  {('PASSED' if passed else 'FAILED')}")
        if seq_len % CHUNK != 0:
            last_seq_vx = (seq_len - 1) // CHUNK
            start_chunk = seq_len % CHUNK
            if start_chunk != 0:
                padding_vals = v_cache_cpu[0, last_seq_vx, :, :, :, :, start_chunk:].abs().sum().item()
                print(f'  Padding area sum: {padding_vals} (should be 0 or very small)')
        return passed
    except Exception as e:
        print(f'  ERROR: {e}')
        return False

def main():
    print('=' * 60)
    print('V-Cache Padding Test')
    print('=' * 60)
    test_cases = [16, 32, 48, 64, 128, 20, 30, 33, 47, 63, 100]
    results = []
    for seq_len in test_cases:
        result = test_padding_case(seq_len)
        if result is not None:
            results.append((seq_len, result))
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    for (seq_len, passed) in results:
        status = 'PASS' if passed else 'FAIL'
        print(f'  [{status}] seq_len={seq_len}')
    passed_count = sum((1 for (_, p) in results if p))
    print(f'\nTotal: {passed_count}/{len(results)} passed')
    return passed_count == len(results)
if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
