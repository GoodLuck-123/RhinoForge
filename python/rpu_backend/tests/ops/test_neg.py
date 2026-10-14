import torch
import rpu_backend
from rpu_backend.tests._seed import input_generator

def test_eager_unary_fp16_matrix(record_property):
    from rpu_backend.tests.conformance._numeric import require_same_semantic
    generator = input_generator()
    for name in ('cos', 'exp', 'log', 'rsqrt', 'silu', 'sin', 'sqrt', 'tanh'):
        op = torch.nn.functional.silu if name == 'silu' else getattr(torch, name)
        for shape in ((16,), (17, 64), (101, 256)):
            x = torch.rand(shape, generator=generator, dtype=torch.float32)
            x = (0.25 + 3.75 * x if name in ('log', 'rsqrt', 'sqrt') else 6 * x - 3).half()
            ref = op(x.float())
            got = op(x.to('rpu')).cpu()
            require_same_semantic(got, ref, actual_dtype=torch.float16, cosine_min=0.999, relative_l2_max=0.01, row_relative_l2_max=0.02, label=f'eager {name} FP16 {shape}')

def test_neg_basic():
    print('=' * 60)
    print('Test: basic neg')
    x = torch.tensor([1.0, -2.0, 3.0, -4.0, 0.0], dtype=torch.float16)
    x_rpu = x.to('rpu')
    y_cpu = torch.neg(x)
    y_rpu = torch.neg(x_rpu)
    y_rpu_cpu = y_rpu.to('cpu')
    print(f'Input:      {x}')
    print(f'CPU output: {y_cpu}')
    print(f'device output: {y_rpu_cpu}')
    assert torch.allclose(y_cpu, y_rpu_cpu, atol=0.001), 'Basic neg test failed'
    print('PASSED\n')

def test_neg_shapes():
    print('=' * 60)
    print('Test: various shapes')
    shapes = [(16,), (128,), (256,), (1024,), (4, 16), (8, 32), (16, 64), (2, 4, 8), (4, 8, 16), (2, 3, 4, 5), (1, 1, 1, 128), (32, 32, 32)]
    g = input_generator()
    for shape in shapes:
        x = torch.randn(shape, generator=g, dtype=torch.float16)
        x_rpu = x.to('rpu')
        y_cpu = torch.neg(x)
        y_rpu = torch.neg(x_rpu).to('cpu')
        max_diff = (y_cpu - y_rpu).abs().max().item()
        passed = torch.allclose(y_cpu, y_rpu, atol=0.001)
        status = 'PASSED' if passed else 'FAILED'
        print(f'  Shape {str(shape):20s} max_diff={max_diff:.6f} {status}')
        assert passed, f'Shape {shape} failed'
    print('All shape tests PASSED\n')

def test_neg_special_values():
    print('=' * 60)
    print('Test: special values')
    x = torch.tensor([0.0, -0.0, 1.0, -1.0, float('inf'), float('-inf'), 0.0001, -0.0001, 65504.0, -65504.0, 6.1e-05, -6.1e-05], dtype=torch.float16)
    x_rpu = x.to('rpu')
    y_cpu = torch.neg(x)
    y_rpu = torch.neg(x_rpu).to('cpu')
    print(f'Input:      {x}')
    print(f'CPU output: {y_cpu}')
    print(f'device output: {y_rpu}')
    mismatches = []
    for i in range(len(x)):
        cpu_val = y_cpu[i].item()
        rpu_val = y_rpu[i].item()
        if torch.isinf(y_cpu[i]) or torch.isnan(y_cpu[i]):
            match = torch.isinf(y_cpu[i]) == torch.isinf(y_rpu[i]) and torch.isnan(y_cpu[i]) == torch.isnan(y_rpu[i])
            if torch.isinf(y_cpu[i]) and torch.isinf(y_rpu[i]):
                match = match and (y_cpu[i] > 0) == (y_rpu[i] > 0)
        elif cpu_val == 0.0:
            match = rpu_val == 0.0 and torch.signbit(y_cpu[i]).item() == torch.signbit(y_rpu[i]).item()
        else:
            match = abs(cpu_val - rpu_val) < 0.001
        status = 'OK' if match else 'MISMATCH'
        print(f'  [{i}] input={x[i].item():12.6g} cpu={cpu_val:12.6g} device={rpu_val:12.6g} {status}')
        if not match:
            mismatches.append((i, x[i].item(), cpu_val, rpu_val))
    assert not mismatches, 'device neg disagrees with CPU on special values (index, input, cpu, device): ' + repr(mismatches)
    print('Special values test PASSED\n')

def test_neg_contiguous():
    print('=' * 60)
    print('Test: non-contiguous tensors')
    x = torch.randn(16, 32, generator=input_generator(), dtype=torch.float16)
    x_t = x.t()
    assert not x_t.is_contiguous(), 'Tensor should be non-contiguous'
    x_t_rpu = x_t.to('rpu')
    y_cpu = torch.neg(x_t)
    y_rpu = torch.neg(x_t_rpu).to('cpu')
    max_diff = (y_cpu - y_rpu).abs().max().item()
    print(f'  Transposed tensor max_diff={max_diff:.6f}')
    assert torch.allclose(y_cpu, y_rpu, atol=0.001), 'Non-contiguous test failed'
    x = torch.randn(64, 64, generator=input_generator(), dtype=torch.float16)
    x_slice = x[::2, ::2]
    assert not x_slice.is_contiguous(), 'Sliced tensor should be non-contiguous'
    x_slice_rpu = x_slice.to('rpu')
    y_cpu = torch.neg(x_slice)
    y_rpu = torch.neg(x_slice_rpu).to('cpu')
    max_diff = (y_cpu - y_rpu).abs().max().item()
    print(f'  Sliced tensor max_diff={max_diff:.6f}')
    assert torch.allclose(y_cpu, y_rpu, atol=0.001), 'Sliced tensor test failed'
    print('Non-contiguous tests PASSED\n')

def test_neg_inplace():
    print('=' * 60)
    print('Test: unary minus operator (-x)')
    x = torch.randn(128, generator=input_generator(), dtype=torch.float16)
    x_rpu = x.clone().to('rpu')
    y_cpu = -x
    y_rpu = (-x_rpu).to('cpu')
    max_diff = (y_cpu - y_rpu).abs().max().item()
    print(f'  Unary minus max_diff={max_diff:.6f}')
    assert torch.allclose(y_cpu, y_rpu, atol=0.001), 'Unary minus test failed'
    print('Unary minus test PASSED\n')

def test_neg_large():
    print('=' * 60)
    print('Test: large tensors')
    sizes = [1024 * 1024, 4 * 1024 * 1024]
    g = input_generator()
    for size in sizes:
        x = torch.randn(size, generator=g, dtype=torch.float16)
        x_rpu = x.to('rpu')
        y_cpu = torch.neg(x)
        y_rpu = torch.neg(x_rpu).to('cpu')
        max_diff = (y_cpu - y_rpu).abs().max().item()
        passed = torch.allclose(y_cpu, y_rpu, atol=0.001)
        status = 'PASSED' if passed else 'FAILED'
        print(f'  Size {size:>10,} elements, max_diff={max_diff:.6f} {status}')
        assert passed, f'Large tensor test failed for size {size}'
    print('Large tensor tests PASSED\n')

def test_neg_double_neg():
    print('=' * 60)
    print('Test: double negation (neg(neg(x)) == x)')
    x = torch.randn(256, generator=input_generator(), dtype=torch.float16)
    x_rpu = x.to('rpu')
    y_rpu = torch.neg(torch.neg(x_rpu)).to('cpu')
    max_diff = (x - y_rpu).abs().max().item()
    print(f'  max_diff from original={max_diff:.6f}')
    assert torch.allclose(x, y_rpu, atol=0.001), 'Double negation test failed'
    print('Double negation test PASSED\n')

def test_neg_zero_equivalence():
    print('=' * 60)
    print('Test: neg(x) == 0 - x equivalence')
    x = torch.randn(256, generator=input_generator(), dtype=torch.float16)
    x_rpu = x.to('rpu')
    y1 = torch.neg(x_rpu)
    zero = torch.zeros_like(x_rpu)
    y2 = zero - x_rpu
    y1_cpu = y1.to('cpu')
    y2_cpu = y2.to('cpu')
    max_diff = (y1_cpu - y2_cpu).abs().max().item()
    print(f'  max_diff between neg(x) and 0-x: {max_diff:.6f}')
    assert torch.allclose(y1_cpu, y2_cpu, atol=0.001), 'Equivalence test failed'
    print('Equivalence test PASSED\n')
if __name__ == '__main__':
    print('Testing neg operator on device\n')
    test_neg_basic()
    test_neg_shapes()
    test_neg_special_values()
    test_neg_contiguous()
    test_neg_inplace()
    test_neg_large()
    test_neg_double_neg()
    test_neg_zero_equivalence()
    print('=' * 60)
    print('ALL TESTS PASSED!')
