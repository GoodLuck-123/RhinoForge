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
    match = max_diff < 0.1
    _RESULTS.append(bool(match))
    return (match, max_diff)
print('=' * 70)
print('Testing out_features boundary (batch=16, in_features=2048)')
print('=' * 70)
batch = 16
in_feat = 2048
print('\nAround 8192:')
for out_feat in range(8000, 8400, 16):
    (match, diff) = test_linear(batch, in_feat, out_feat)
    status = 'PASS' if match else 'FAIL'
    if not match or out_feat % 128 == 0:
        print(f'  out_features={out_feat:5d}: {status}, max_diff={diff:.4f}')
print('\nMultiples of 128 around failure point:')
for out_feat in [7936, 8064, 8192, 8320, 8448]:
    (match, diff) = test_linear(batch, in_feat, out_feat)
    status = 'PASS' if match else 'FAIL'
    print(f'  out_features={out_feat:5d} (128*{out_feat // 128}): {status}, max_diff={diff:.4f}')
print('\nPower of 2 tests:')
for out_feat in [4096, 6144, 8192, 10240, 12288, 14336, 16384]:
    (match, diff) = test_linear(batch, in_feat, out_feat)
    status = 'PASS' if match else 'FAIL'
    print(f'  out_features={out_feat:5d}: {status}, max_diff={diff:.4f}')
print('\nWith smaller batch (batch=1):')
batch = 1
for out_feat in [8192, 16384, 32768]:
    (match, diff) = test_linear(batch, in_feat, out_feat)
    status = 'PASS' if match else 'FAIL'
    print(f'  (1, 2048) x ({out_feat}, 2048): {status}, max_diff={diff:.4f}')
print('\n3D input shapes (flattened):')
test_cases = [(4 * 64, 256, 512), (1 * 512, 768, 768)]
for (batch, in_feat, out_feat) in test_cases:
    (match, diff) = test_linear(batch, in_feat, out_feat)
    status = 'PASS' if match else 'FAIL'
    print(f'  ({batch}, {in_feat}) x ({out_feat}, {in_feat}): {status}, max_diff={diff:.4f}')
print('\n' + '=' * 70)
import sys as _sys
print(f'\n[summary] {sum(_RESULTS)}/{len(_RESULTS)} cases passed')
_sys.exit(0 if _RESULTS and all(_RESULTS) else 1)
