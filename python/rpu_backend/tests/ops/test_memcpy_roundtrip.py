import sys
import torch
import numpy as np
import rpu_backend
from rpu_backend.runtime.weights import transform_linear_weight
torch.manual_seed(42)

def setup_rpu():
    torch.rpu.set_profile(False)
    torch.rpu.set_caching_allocator(True)
    torch.rpu.set_ddr_flush(True)
    torch.rpu.set_debug_export(False)
    torch.rpu.set_spm_debug(False)

def check(name, ref, got, atol=0.001, rtol=0.001):
    ref_f = ref.float().cpu()
    got_f = got.float().cpu()
    diff = (ref_f - got_f).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    rpu_is_zero = got_f.abs().max().item() == 0.0
    passed = torch.allclose(ref_f, got_f, atol=atol, rtol=rtol) and (not rpu_is_zero)
    status = 'PASS' if passed else 'FAIL'
    extra = ''
    if rpu_is_zero:
        extra = ' [device ALL ZEROS!]'
    elif not passed:
        max_idx = diff.argmax().item()
        extra = f' ref={ref_f.view(-1)[max_idx]:.4f} device={got_f.view(-1)[max_idx]:.4f}'
    print(f'  [{status}] {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}{extra}')
    return passed

def test_add_zero_roundtrip():
    print('\n' + '=' * 60)
    print('Test 1: torch.add(x, 0) roundtrip (eltwise DDR<->SPM)')
    print('=' * 60)
    all_passed = True
    test_cases = [((1, 64, 2560), 'seq=64,  hidden=2560'), ((1, 100, 2560), 'seq=100, hidden=2560'), ((1, 112, 2560), 'seq=112, hidden=2560'), ((1, 120, 2560), 'seq=120, hidden=2560'), ((1, 126, 2560), 'seq=126, hidden=2560'), ((1, 127, 2560), 'seq=127, hidden=2560'), ((1, 128, 2560), 'seq=128, hidden=2560'), ((1, 129, 2560), 'seq=129, hidden=2560'), ((1, 256, 2560), 'seq=256, hidden=2560'), ((1 * 100 * 2560,), 'flat 256000 elems (100*2560)'), ((1 * 128 * 2560,), 'flat 327680 elems (128*2560)')]
    for (shape, desc) in test_cases:
        x_cpu = torch.randn(shape, dtype=torch.float16)
        zero_cpu = torch.zeros(shape, dtype=torch.float16)
        x_rpu = x_cpu.to('rpu')
        zero_rpu = zero_cpu.to('rpu')
        result_rpu = torch.add(x_rpu, zero_rpu)
        result_cpu = result_rpu.cpu()
        passed = check(desc, x_cpu, result_cpu)
        if not passed:
            all_passed = False
    assert all_passed, 'see the [FAIL] lines above'

def test_mul_one_roundtrip():
    print('\n' + '=' * 60)
    print('Test 2: torch.mul(x, 1) roundtrip (eltwise DDR<->SPM)')
    print('=' * 60)
    all_passed = True
    test_cases = [((1, 100, 2560), 'seq=100, hidden=2560'), ((1, 128, 2560), 'seq=128, hidden=2560'), ((1, 256, 2560), 'seq=256, hidden=2560')]
    for (shape, desc) in test_cases:
        x_cpu = torch.randn(shape, dtype=torch.float16)
        one_cpu = torch.ones(shape, dtype=torch.float16)
        x_rpu = x_cpu.to('rpu')
        one_rpu = one_cpu.to('rpu')
        result_rpu = torch.mul(x_rpu, one_rpu)
        result_cpu = result_rpu.cpu()
        passed = check(desc, x_cpu, result_cpu)
        if not passed:
            all_passed = False
    assert all_passed, 'see the [FAIL] lines above'

def test_ddr_copy_roundtrip():
    print('\n' + '=' * 60)
    print("Test 3: DDR copy roundtrip (.to('rpu').cpu(), no SPM)")
    print('=' * 60)
    all_passed = True
    test_cases = [((1, 100, 2560), 'seq=100, hidden=2560'), ((1, 128, 2560), 'seq=128, hidden=2560'), ((1, 256, 2560), 'seq=256, hidden=2560'), ((327680,), 'flat 327680 elems')]
    for (shape, desc) in test_cases:
        x_cpu = torch.randn(shape, dtype=torch.float16)
        x_rpu = x_cpu.to('rpu')
        x_back = x_rpu.cpu()
        passed = check(desc, x_cpu, x_back, atol=0, rtol=0)
        if not passed:
            all_passed = False
    assert all_passed, 'see the [FAIL] lines above'

def test_rmsnorm_roundtrip():
    print('\n' + '=' * 60)
    print('Test 4: RMSNorm DDR<->SPM roundtrip')
    print('=' * 60)
    all_passed = True
    hidden_size = 2560
    eps = 1e-06
    test_cases = [(100, 'seq=100'), (112, 'seq=112'), (120, 'seq=120'), (128, 'seq=128'), (256, 'seq=256')]
    for (seq_len, desc) in test_cases:
        x_cpu = torch.randn(1, seq_len, hidden_size, dtype=torch.float16)
        w_cpu = torch.ones(hidden_size, dtype=torch.float16)
        x_f32 = x_cpu.float()
        rms = x_f32.pow(2).mean(-1, keepdim=True).add(eps).rsqrt()
        ref = (x_f32 * rms).half() * w_cpu
        x_rpu = x_cpu.to('rpu')
        w_rpu = w_cpu.to('rpu')
        rpu_out = torch.ops.rpu.rms_norm(x_rpu, [hidden_size], w_rpu, eps)
        rpu_out_cpu = rpu_out.cpu()
        passed = check(desc, ref, rpu_out_cpu, atol=0.05, rtol=0.02)
        if not passed:
            all_passed = False
    assert all_passed, 'see the [FAIL] lines above'

def test_linear_ddr():
    print('\n' + '=' * 60)
    print('Test 5: Linear (DDR) at boundary sizes')
    print('=' * 60)
    all_passed = True
    hidden_size = 2560
    out_size = 512
    test_cases = [(100, 'M=100'), (128, 'M=128'), (256, 'M=256')]
    for (M, desc) in test_cases:
        x_cpu = torch.randn(M, hidden_size, dtype=torch.float16)
        w_cpu = torch.randn(out_size, hidden_size, dtype=torch.float16)
        ref = (x_cpu.float() @ w_cpu.float().T).half()
        x_rpu = x_cpu.to('rpu')
        w_rpu = transform_linear_weight(w_cpu, 1).to('rpu')
        try:
            rpu_out = rpu_backend._cpp_ext.linear_with_partition(x_rpu, w_rpu, None, 1)
            rpu_out_cpu = rpu_out.cpu()
            rpu_is_zero = rpu_out_cpu.float().abs().max().item() == 0.0
            if rpu_is_zero:
                print(f'  [FAIL] {desc}: device output ALL ZEROS!')
                all_passed = False
            else:
                passed = check(desc, ref, rpu_out_cpu, atol=2.0, rtol=0.1)
                if not passed:
                    all_passed = False
        except Exception as e:
            print(f'  [FAIL] {desc}: {type(e).__name__}: {e}')
            all_passed = False
    assert all_passed, 'linear_ddr: see the [FAIL] lines above'

def main():
    setup_rpu()
    results = {}

    def run(label, fn):
        try:
            fn()
        except AssertionError as exc:
            print(f'  [FAIL] {label}: {exc}')
            results[label] = False
        else:
            results[label] = True
    run('ddr_copy', test_ddr_copy_roundtrip)
    run('add_zero', test_add_zero_roundtrip)
    run('mul_one', test_mul_one_roundtrip)
    run('rmsnorm', test_rmsnorm_roundtrip)
    run('linear', test_linear_ddr)
    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)
    all_ok = True
    for (name, passed) in results.items():
        status = 'PASS' if passed else 'FAIL'
        print(f'  {name}: {status}')
        if not passed:
            all_ok = False
    if all_ok:
        print('\nAll tests passed — DDR<->SPM memcpy is likely NOT the issue.')
        print("The bug is probably in the fused decoder layer's compute kernels or orchestration.")
    else:
        print('\nSome tests FAILED — DDR<->SPM memcpy may be broken at specific sizes.')
    return 0 if all_ok else 1
if __name__ == '__main__':
    sys.exit(main())
