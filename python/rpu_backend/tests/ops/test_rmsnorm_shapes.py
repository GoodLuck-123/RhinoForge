import torch
import torch.nn as nn
import rpu_backend
from functools import wraps
rmsnorm_shapes = []

def patch_rmsnorm_for_profiling(rmsnorm_class):
    original_forward = rmsnorm_class.forward

    @wraps(original_forward)
    def profiling_forward(self, hidden_states):
        shape_info = {'input_shape': tuple(hidden_states.shape), 'normalized_dim': hidden_states.shape[-1], 'weight_shape': tuple(self.weight.shape), 'dtype': str(hidden_states.dtype)}
        rmsnorm_shapes.append(shape_info)
        return original_forward(self, hidden_states)
    rmsnorm_class.forward = profiling_forward

def collect_shapes_from_qwen3():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
    patch_rmsnorm_for_profiling(Qwen3RMSNorm)
    from rpu_backend.model_registry import model_path as _model_path
    model_path = str(_model_path('qwen3-0.6b'))
    print(f'Loading model from {model_path}...')
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map='cpu', local_files_only=True)
    model.eval()
    text = 'Hello from Hugging Face!'
    inputs = tokenizer(text, return_tensors='pt')
    print('Running forward pass to collect shapes...')
    with torch.no_grad():
        _ = model(**inputs)
    unique_shapes = {}
    for info in rmsnorm_shapes:
        key = (info['input_shape'], info['normalized_dim'])
        if key not in unique_shapes:
            unique_shapes[key] = info
    return list(unique_shapes.values())

def print_shape_summary(shapes):
    print('\n' + '=' * 70)
    print('RMSNorm Shape Summary from Qwen3')
    print('=' * 70)
    print(f"{'Input Shape':<30} {'C (normalized)':<15} {'Weight':<15}")
    print('-' * 70)
    for info in shapes:
        print(f"{str(info['input_shape']):<30} {info['normalized_dim']:<15} {str(info['weight_shape']):<15}")
    print('=' * 70)
    print(f'Total unique shapes: {len(shapes)}')

def test_rmsnorm_shape(input_shape, normalized_dim, eps=1e-06):
    print(f'\nTesting shape: {input_shape}, C={normalized_dim}')
    x = torch.randn(input_shape, dtype=torch.float16)
    weight = torch.randn(normalized_dim, dtype=torch.float16)
    x_f32 = x.float()
    variance = x_f32.pow(2).mean(-1, keepdim=True)
    x_normed = x_f32 * torch.rsqrt(variance + eps)
    y_cpu = (weight.float() * x_normed).half()
    x_rpu = x.to('rpu')
    weight_rpu = weight.to('rpu')
    y_rpu = torch.rpu.rms_norm(x_rpu, [normalized_dim], weight_rpu, eps)
    y_rpu_cpu = y_rpu.to('cpu')
    max_diff = (y_cpu - y_rpu_cpu).abs().max().item()
    mean_diff = (y_cpu - y_rpu_cpu).abs().mean().item()
    nonzero_mask = y_cpu.abs() > 1e-06
    if nonzero_mask.any():
        rel_error = ((y_cpu - y_rpu_cpu).abs() / (y_cpu.abs() + 1e-06))[nonzero_mask].max().item()
    else:
        rel_error = 0.0
    passed = max_diff < 0.1
    status = 'PASSED' if passed else 'FAILED'
    print(f'  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, rel_error={rel_error:.4f} [{status}]')
    if not passed:
        print(f'  CPU output sample: {y_cpu.flatten()[:5]}')
        print(f'  device output sample: {y_rpu_cpu.flatten()[:5]}')
    return (passed, max_diff)

def test_all_qwen3_shapes():
    print('\n' + '=' * 70)
    print('Testing All Qwen3 RMSNorm Shapes on device')
    print('=' * 70)
    test_shapes = [((1, 17, 1024), 1024, 'input_layernorm'), ((1, 17, 16, 64), 64, 'q_norm/k_norm'), ((2, 17, 1024), 1024, 'batch=2 layernorm'), ((4, 17, 1024), 1024, 'batch=4 layernorm'), ((1, 1, 1024), 1024, 'seq=1 layernorm'), ((1, 128, 1024), 1024, 'seq=128 layernorm'), ((1, 256, 1024), 1024, 'seq=256 layernorm'), ((1, 17, 256), 256, 'C=256 boundary'), ((1, 17, 512), 512, 'C=512'), ((1, 17, 128), 128, 'C=128 (< 256)'), ((1, 17, 64), 64, 'C=64 (head_dim)'), ((1, 17, 32), 32, 'C=32 (small)')]
    results = []
    for (shape, c, name) in test_shapes:
        print(f'\n--- {name} ---')
        try:
            (passed, max_diff) = test_rmsnorm_shape(shape, c)
            results.append((name, shape, c, passed, max_diff))
        except Exception as e:
            print(f'  ERROR: {e}')
            results.append((name, shape, c, False, float('inf')))
    print('\n' + '=' * 70)
    print('Test Summary')
    print('=' * 70)
    print(f"{'Name':<25} {'Shape':<20} {'C':<6} {'Status':<8} {'MaxDiff':<10}")
    print('-' * 70)
    passed_count = 0
    for (name, shape, c, passed, max_diff) in results:
        status = 'PASS' if passed else 'FAIL'
        if passed:
            passed_count += 1
        diff_str = f'{max_diff:.6f}' if max_diff != float('inf') else 'ERROR'
        print(f'{name:<25} {str(shape):<20} {c:<6} {status:<8} {diff_str:<10}')
    print('-' * 70)
    print(f'Passed: {passed_count}/{len(results)}')
    assert passed_count == len(results), f'{len(results) - passed_count}/{len(results)} RMSNorm shapes FAILED'

def test_simple():
    print('=' * 70)
    print('Simple RMSNorm Tests')
    print('=' * 70)
    test_all_qwen3_shapes()
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--collect':
        shapes = collect_shapes_from_qwen3()
        print_shape_summary(shapes)
    else:
        test_simple()
