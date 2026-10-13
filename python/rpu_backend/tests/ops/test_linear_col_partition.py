import torch
import torch.nn as nn
import rpu_backend
from rpu_backend.runtime.weights import get_linear_partition, transform_linear_weight
torch.manual_seed(42)
NUM_CORES = 8

def check_partition(M, K, N):
    can_row = K % 128 == 0 and N % 16 == 0
    can_col = N % 128 == 0 and K % 16 == 0
    if can_row and can_col:
        return 'col' if N > K else 'row'
    elif can_row:
        return 'row'
    elif can_col:
        return 'col'
    else:
        return 'cpu'

def is_col_partition(M, K, N):
    return check_partition(M, K, N) == 'col'

def test_linear(M, K, N, has_bias=False, rtol=0.1):
    partition = check_partition(M, K, N)
    if partition != 'col':
        print(f'\n[SKIP] M={M}, K={K}, N={N} -> {partition} (不是 col partition)')
        return None
    print(f"\n{'=' * 60}")
    print(f'M={M}, K={K}, N={N}, bias={has_bias}, partition={partition}')
    print(f'  local_n={N // NUM_CORES}, output_row_stride={N * 2}')
    x = torch.randn(1, M, K, dtype=torch.float16)
    linear = nn.Linear(K, N, bias=has_bias).half()
    with torch.no_grad():
        y_cpu = linear(x)
    partition_id = get_linear_partition(K, N)
    with torch.no_grad():
        transformed_weight = transform_linear_weight(linear.weight.data, partition_id)
        linear.weight.data = transformed_weight
    x_rpu = x.to('rpu')
    linear_rpu = linear.to('rpu')
    with torch.no_grad():
        y_rpu = linear_rpu(x_rpu)
    y_rpu_cpu = y_rpu.to('cpu')
    diff = (y_rpu_cpu - y_cpu).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    passed = max_diff < rtol
    status = 'PASS' if passed else 'FAIL'
    print(f'  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f} [{status}]')
    if not passed:
        diff_2d = diff.squeeze(0)
        local_n = N // NUM_CORES
        print(f'  错误分布:')
        for core_id in range(NUM_CORES):
            col_start = core_id * local_n
            col_end = col_start + local_n
            core_diff = diff_2d[:, col_start:col_end]
            core_max = core_diff.max().item()
            core_errors = (core_diff > rtol).sum().item()
            if core_errors > 0:
                print(f'    Core {core_id}: max={core_max:.4f}, errors={core_errors}')
    return passed
print('=' * 60)
print('Linear Col Partition 测试')
print('=' * 60)
print('\nCol partition 条件: N % 128 == 0 且 K % 16 == 0\n强制 col partition: K % 128 != 0 (禁用 row) 或 N > K\n')
test_cases = [(19, 1024, 151936, False), (1, 1024, 151936, False), (16, 1024, 151936, False), (32, 1024, 151936, False), (64, 1024, 151936, False), (128, 1024, 151936, False), (192, 1024, 151936, False), (19, 16, 1024, False), (19, 32, 1024, False), (19, 48, 2048, False), (19, 64, 4096, False), (19, 80, 8192, False), (19, 96, 16384, False), (19, 112, 32768, False), (19, 512, 151936, False), (19, 2048, 151936, False), (19, 4096, 151936, False), (19, 64, 128, False), (19, 64, 256, False), (19, 64, 512, False), (19, 64, 1024, False), (19, 64, 2048, False), (19, 64, 4096, False), (19, 64, 8192, False), (19, 1024, 32768, False), (19, 1024, 151936, True), (32, 64, 8192, True), (1, 16, 128, False), (15, 16, 128, False), (16, 16, 128, False), (17, 16, 128, False), (31, 16, 128, False), (32, 16, 128, False), (200, 64, 8192, False), (384, 64, 8192, False)]
passed_count = 0
failed_count = 0
skipped_count = 0
for (M, K, N, has_bias) in test_cases:
    try:
        result = test_linear(M, K, N, has_bias)
        if result is None:
            skipped_count += 1
        elif result:
            passed_count += 1
        else:
            failed_count += 1
    except Exception as e:
        print(f'\n[ERROR] M={M}, K={K}, N={N}: {e}')
        failed_count += 1
print('\n' + '=' * 60)
print(f'测试结果: {passed_count} passed, {failed_count} failed, {skipped_count} skipped')
print('=' * 60)
import sys as _sys
_sys.exit(1 if failed_count else 0)
