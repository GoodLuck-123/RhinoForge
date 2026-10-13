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
        return 'row' if K >= N else 'col'
    elif can_row:
        return 'row'
    elif can_col:
        return 'col'
    else:
        return 'cpu'

def test_linear(M, K, N, has_bias=False, rtol=0.1):
    partition = check_partition(M, K, N)
    if partition != 'row':
        print(f'\n[SKIP] M={M}, K={K}, N={N} -> {partition} (不是 row partition)')
        return None
    local_k = K // NUM_CORES
    print(f"\n{'=' * 60}")
    print(f'M={M}, K={K}, N={N}, bias={has_bias}, partition={partition}')
    print(f'  local_k={local_k}, input_row_stride={K * 2}, output_row_stride={N * 2}')
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
        print(f'  错误分布 (按行):')
        for row in range(min(M, 10)):
            row_diff = diff_2d[row, :]
            max_err = row_diff.max().item()
            num_err = (row_diff > rtol).sum().item()
            if num_err > 0:
                print(f'    Row {row}: max={max_err:.4f}, errors={num_err}/{N}')
        print(f'  按 Core 分析:')
        cols_per_section = N // 8 if N >= 8 else N
        for sec in range(min(8, N // cols_per_section if cols_per_section > 0 else 1)):
            col_start = sec * cols_per_section
            col_end = min(col_start + cols_per_section, N)
            sec_diff = diff_2d[:, col_start:col_end]
            sec_max = sec_diff.max().item()
            sec_errors = (sec_diff > rtol).sum().item()
            if sec_errors > 0:
                print(f'    Cols {col_start}-{col_end - 1}: max={sec_max:.4f}, errors={sec_errors}')
    return passed
print('=' * 60)
print('Linear Row Partition 测试')
print('=' * 60)
print('\nRow partition 条件: K % 128 == 0 且 N % 16 == 0\n强制 row partition: N % 128 != 0 (禁用 col) 或 K >= N\n')
test_cases = [(19, 1024, 512, False), (19, 1024, 1024, False), (19, 1024, 16, False), (19, 1024, 32, False), (19, 1024, 48, False), (19, 1024, 64, False), (19, 1024, 80, False), (19, 1024, 96, False), (19, 1024, 112, False), (1, 1024, 64, False), (16, 1024, 64, False), (32, 1024, 64, False), (64, 1024, 64, False), (128, 1024, 64, False), (192, 1024, 64, False), (19, 128, 64, False), (19, 256, 64, False), (19, 512, 64, False), (19, 2048, 64, False), (19, 4096, 64, False), (19, 2048, 1024, False), (19, 4096, 2048, False), (19, 1024, 64, True), (32, 2048, 512, True), (1, 128, 16, False), (15, 128, 16, False), (16, 128, 16, False), (17, 128, 16, False), (31, 128, 16, False), (32, 128, 16, False), (200, 1024, 64, False), (384, 1024, 64, False)]
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
        import traceback
        traceback.print_exc()
        failed_count += 1
print('\n' + '=' * 60)
print(f'测试结果: {passed_count} passed, {failed_count} failed, {skipped_count} skipped')
print('=' * 60)
import sys as _sys
_sys.exit(1 if failed_count else 0)
