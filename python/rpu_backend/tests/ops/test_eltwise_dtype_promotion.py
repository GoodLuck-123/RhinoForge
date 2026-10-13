import itertools
import pytest
import torch
import rpu_backend
OPS = {'mul': lambda x, y: x * y, 'add': lambda x, y: x + y, 'sub': lambda x, y: x - y, 'div': lambda x, y: x / y}

def _pair():
    torch.manual_seed(0)
    small = torch.randn(1024, dtype=torch.float16).abs() + 0.5
    big = torch.randn(1, 128, 1024, dtype=torch.float32).abs() + 0.5
    return (small, big)

@pytest.mark.parametrize('op_name,fp16_first', list(itertools.product(sorted(OPS), [True, False])))
def test_mixed_dtype_matches_cpu_in_both_orders(op_name, fp16_first):
    op = OPS[op_name]
    (small, big) = _pair()
    (lhs, rhs) = (small, big) if fp16_first else (big, small)
    ref = op(lhs, rhs)
    got = op(lhs.to('rpu'), rhs.to('rpu')).cpu()
    assert got.dtype == ref.dtype, f'{op_name} fp16_first={fp16_first}: promotion says {ref.dtype}, got {got.dtype}'
    assert got.shape == ref.shape
    cos = torch.nn.functional.cosine_similarity(ref.float().flatten(), got.float().flatten(), dim=0).item()
    assert cos > 0.999, f'{op_name} fp16_first={fp16_first}: cos={cos:.6f} max_abs={(ref.float() - got.float()).abs().max().item():.6f}'

def test_same_dtype_fp16_still_takes_the_device_path():
    torch.manual_seed(0)
    a = torch.randn(1, 128, 1024, dtype=torch.float16)
    b = torch.randn(1, 128, 1024, dtype=torch.float16).abs() + 0.5
    got = a.to('rpu') * b.to('rpu')
    assert got.dtype == torch.float16
    assert got.device.type == 'rpu'
    cos = torch.nn.functional.cosine_similarity((a * b).float().flatten(), got.cpu().float().flatten(), dim=0).item()
    assert cos > 0.999, f'fp16*fp16 regressed onto a wrong path: cos={cos:.6f}'

@pytest.mark.parametrize('op_name,a_dtype', list(itertools.product(sorted(OPS), [torch.int8, torch.bool])))
def test_non_fp16_left_operand_with_fp16_right(op_name, a_dtype):
    op = OPS[op_name]
    if a_dtype is torch.bool:
        a = torch.tensor([True, False, True, True])
    else:
        a = torch.arange(4, dtype=a_dtype)
    b = torch.arange(1, 5, dtype=torch.float16)
    if a_dtype is torch.bool and op_name == 'sub':
        with pytest.raises(RuntimeError, match='bool tensor is not supported'):
            op(a, b)
        with pytest.raises(RuntimeError, match='bool tensor is not supported'):
            op(a.to('rpu'), b.to('rpu'))
        return
    ref = op(a, b)
    got = op(a.to('rpu'), b.to('rpu')).cpu()
    assert got.dtype == ref.dtype, f'{a_dtype} {op_name} fp16: promotion says {ref.dtype}, got {got.dtype}'
    assert got.shape == ref.shape
    assert torch.equal(got.float(), ref.float()), f'{a_dtype} {op_name}: {got} != {ref}'

@pytest.mark.parametrize('b_dtype', [torch.int8, torch.bool])
def test_fp16_left_with_non_fp16_right_keeps_the_device_path(b_dtype):
    a = torch.arange(1, 5, dtype=torch.float16)
    b = torch.tensor([True, False, True, True]) if b_dtype is torch.bool else torch.arange(1, 5, dtype=b_dtype)
    ref = a * b
    got = a.to('rpu') * b.to('rpu')
    assert got.dtype == torch.float16 == ref.dtype
    assert got.device.type == 'rpu'
    assert torch.equal(got.cpu().float(), ref.float()), f'{got.cpu()} != {ref}'

def test_inplace_keeps_the_left_dtype():
    torch.manual_seed(0)
    a = torch.randn(1, 128, 1024, dtype=torch.float16)
    b = torch.randn(1024, dtype=torch.float32).abs() + 0.5
    got = a.clone().to('rpu')
    got *= b.to('rpu')
    assert got.dtype == torch.float16, f'in-place promoted to {got.dtype}'
    ref = a * b.half()
    cos = torch.nn.functional.cosine_similarity(ref.float().flatten(), got.cpu().float().flatten(), dim=0).item()
    assert cos > 0.999, f'in-place fp16 *= fp32 value: cos={cos:.6f}'

@pytest.mark.parametrize('op_name,a_dtype', list(itertools.product(sorted(OPS), [torch.int8, torch.bool])))
def test_non_fp16_left_operand_with_a_broadcast(op_name, a_dtype):
    op = OPS[op_name]
    if a_dtype is torch.bool:
        a = torch.tensor([True, False, True, True]).expand(1, 8, 4).contiguous()
    else:
        a = torch.arange(4, dtype=a_dtype).expand(1, 8, 4).contiguous()
    b = torch.arange(1, 5, dtype=torch.float16)
    if a_dtype is torch.bool and op_name == 'sub':
        with pytest.raises(RuntimeError, match='bool tensor is not supported'):
            op(a, b)
        with pytest.raises(RuntimeError, match='bool tensor is not supported'):
            op(a.to('rpu'), b.to('rpu'))
        return
    ref = op(a, b)
    got = op(a.to('rpu'), b.to('rpu')).cpu()
    assert got.dtype == ref.dtype
    assert got.shape == ref.shape
    assert torch.equal(got.float(), ref.float()), f'{a_dtype} {op_name}: {got} != {ref}'

@pytest.mark.parametrize('op_name', sorted(OPS))
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16], ids=['fp32', 'bf16'])
def test_non_fp16_on_both_sides_falls_back(op_name, dtype):
    op = OPS[op_name]
    torch.manual_seed(0)
    a = torch.randn(1, 8, 64, dtype=dtype)
    b = torch.randn(64, dtype=dtype).abs() + 0.5
    ref = op(a, b)
    got = op(a.to('rpu'), b.to('rpu')).cpu()
    assert got.dtype == dtype == ref.dtype, f'{dtype} {op_name}: got {got.dtype}, promotion says {ref.dtype}'
    assert got.shape == ref.shape
    assert torch.equal(got.float(), ref.float()), f'{dtype} {op_name} value drifted'

def test_fp16_scalar_and_pow_native_overloads(record_property):
    from rpu_backend.tests.conformance._numeric import format_metrics, require_same_semantic
    gen = torch.Generator().manual_seed(20260908)
    for shape in ((17, 64), (101, 256)):
        x = (torch.rand(shape, generator=gen) * 1.75 + 0.25).half()
        exponent = (torch.rand(shape, generator=gen) * 2 - 0.5).half()
        cases = ((torch.ops.aten.mul.Scalar, (1.75,), x.float() * 1.75, False), (torch.ops.aten.div.Scalar, (2.5,), x.float() / 2.5, False), (torch.ops.aten.mul_.Scalar, (-0.75,), x.float() * -0.75, True), (torch.ops.aten.sub_.Scalar, (0.5, 1.5), x.float() - 0.75, True), (torch.ops.aten.pow.Tensor_Scalar, (1.5,), x.float().pow(1.5), False), (torch.ops.aten.pow.Tensor_Tensor, (exponent.to('rpu'),), x.float().pow(exponent.float()), False))
        for (op, args, expected, inplace) in cases:
            operand = x.to('rpu')
            pointer = operand.data_ptr()
            actual = op(operand, *args)
            if inplace:
                assert actual.data_ptr() == operand.data_ptr() == pointer
            metrics = require_same_semantic(actual.cpu(), expected, actual_dtype=torch.float16, cosine_min=0.999, relative_l2_max=0.01, row_relative_l2_max=0.02, label=str(op))
            print(f'{op} shape={shape}: {format_metrics(metrics)}')
        actual = torch.ops.aten.pow.Scalar(1.75, exponent.to('rpu'))
        metrics = require_same_semantic(actual.cpu(), torch.pow(1.75, exponent.float()), actual_dtype=torch.float16, cosine_min=0.999, relative_l2_max=0.01, row_relative_l2_max=0.02, label='aten.pow.Scalar')
        print(f'aten.pow.Scalar shape={shape}: {format_metrics(metrics)}')
