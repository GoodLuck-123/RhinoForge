import torch
import torch.nn as nn
import rpu_backend
from rpu_backend.runtime.weights import get_linear_partition, transform_linear_weight
torch.manual_seed(42)
_RESULTS = []

def test_linear(batch, in_feat, out_feat):
    x = torch.randn(batch, in_feat, dtype=torch.float16)
    linear = nn.Linear(in_feat, out_feat, bias=False).half()
    with torch.no_grad():
        y_cpu = linear(x)
    pid = get_linear_partition(in_feat, out_feat)
    linear.weight.data = transform_linear_weight(linear.weight.data, pid)
    x_rpu = x.to('rpu')
    linear_rpu = linear.to('rpu')
    with torch.no_grad():
        y_rpu = linear_rpu(x_rpu)
    y_rpu_cpu = y_rpu.to('cpu')
    max_diff = (y_rpu_cpu - y_cpu).abs().max().item()
    _ok = max_diff < 0.01
    _RESULTS.append(_ok)
    return (_ok, max_diff)
print('=' * 70)
print('Finding the boundary varying K (in_features)')
print('=' * 70)
print('\nFixed M=8, N=512, varying K:')
for k in [64, 128, 160, 192, 224, 256, 288, 320, 384, 512]:
    (passed, diff) = test_linear(8, k, 512)
    status = 'PASS' if passed else 'FAIL'
    print(f'  K={k:4d}: {status}, max_diff={diff:.6f}')
print('\nFixed M=4, N=256, varying K:')
for k in [64, 128, 160, 192, 224, 256, 288, 320, 384, 512]:
    (passed, diff) = test_linear(4, k, 256)
    status = 'PASS' if passed else 'FAIL'
    print(f'  K={k:4d}: {status}, max_diff={diff:.6f}')
print('\nFiner test M=8, N=512, K from 128 to 256 (step=16):')
for k in range(128, 272, 16):
    (passed, diff) = test_linear(8, k, 512)
    status = 'PASS' if passed else 'FAIL'
    print(f'  K={k:4d}: {status}, max_diff={diff:.6f}')
print('\n' + '=' * 70)
import sys as _sys
print(f'\n[summary] {sum(_RESULTS)}/{len(_RESULTS)} cases passed')
_sys.exit(0 if _RESULTS and all(_RESULTS) else 1)
