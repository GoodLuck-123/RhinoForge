import torch
import rpu_backend
from rpu_backend.tests._seed import input_generator

def test_insert_vcache():
    batch = 1
    seq_len = 32
    num_cores = 8
    num_kv_percore_heads = 4
    total_kv_heads = num_cores * num_kv_percore_heads
    head_dim = 64
    position = 0
    print(f'Creating input tensor: [{batch}, {seq_len}, {total_kv_heads}, {head_dim}]')
    v_input = torch.randn(batch, seq_len, total_kv_heads, head_dim, generator=input_generator(), dtype=torch.float16).to('rpu')
    sValChunk = 16
    headDimChunk = 16
    sValVx = seq_len // sValChunk
    nKVHeadVx = num_kv_percore_heads
    headDimVx = head_dim // headDimChunk
    nKVHeadChunk = num_cores
    print(f'Creating V-Cache tensor: [{batch}, {sValVx}, {nKVHeadVx}, {headDimVx}, {nKVHeadChunk}, {headDimChunk}, {sValChunk}]')
    v_cache = torch.zeros(batch, sValVx, nKVHeadVx, headDimVx, nKVHeadChunk, headDimChunk, sValChunk, dtype=torch.float16, device='rpu')
    print(f'\nv_input shape: {v_input.shape}')
    print(f'v_cache shape: {v_cache.shape}')
    print(f'position: {position}')
    assert v_input.shape == (1, 32, 32, 64), f'Input shape mismatch: {v_input.shape}'
    assert v_cache.shape == (1, 2, 4, 4, 8, 16, 16), f'Cache shape mismatch: {v_cache.shape}'
    print('\nShape validation passed!')
    print('\nCalling rpu_launch_llama_insert_vcache...')
    try:
        torch.rpu.insert_vcache(v_input, v_cache, position)
    except Exception as e:
        print(f'ERROR: Kernel execution failed: {e}')
        raise
    print('Kernel execution completed!')
    cache_sum = v_cache.abs().sum().item()
    print(f'\nV-Cache abs sum: {cache_sum}')
    assert cache_sum > 0, 'V-Cache is still all zeros - kernel wrote no data'
    print('SUCCESS: V-Cache contains non-zero values after insert')
if __name__ == '__main__':
    test_insert_vcache()
