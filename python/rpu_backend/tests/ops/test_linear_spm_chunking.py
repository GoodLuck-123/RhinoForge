import torch
import torch.nn as nn
import rpu_backend
from rpu_backend.runtime.weights import get_linear_partition, transform_linear_weight
torch.manual_seed(7)
SPM_USABLE = 8 * 1024 * 1024 - 1024

def check(M, K, N, expect_partition):
    pid = get_linear_partition(K, N)
    assert pid == expect_partition, f'K={K} N={N}: partition {pid} != {expect_partition}'
    local_k = K // 8 if pid == 0 else K
    local_n = N if pid == 0 else N // 8
    chunk_rows = (SPM_USABLE - 512) // ((local_k + local_n) * 2)
    n_chunks = -(-M // chunk_rows)
    assert n_chunks > 1, f'K={K} N={N} M={M}: fits in one chunk ({chunk_rows} rows) — this test would not exercise the loop'
    x = torch.randn(M, K, dtype=torch.float16)
    lin = nn.Linear(K, N, bias=False).half()
    with torch.no_grad():
        y_cpu = lin(x)
    lin.weight.data = transform_linear_weight(lin.weight.data, pid)
    with torch.no_grad():
        y = lin.to('rpu')(x.to('rpu')).to('cpu')
    max_diff = (y - y_cpu).abs().max().item()
    ok = max_diff < 0.01
    print(f"M={M} K={K} N={N} partition={('row' if pid == 0 else 'col')} chunks={n_chunks} (<={chunk_rows} rows each) max_diff={max_diff:.6f} {('[PASS]' if ok else '[FAIL]')}")
    return ok
results = [check(300, 128, 32752, expect_partition=0), check(300, 16, 262016, expect_partition=1)]
import sys
print(f'\n[summary] {sum(results)}/{len(results)} cases passed')
sys.exit(0 if all(results) else 1)
