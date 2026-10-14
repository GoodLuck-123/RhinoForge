import torch
import rpu_backend
import numpy as np
from rpu_backend.tests._seed import input_generator

def _assert_arg_reduce(x, dim, idx_ref, idx_got, mode, label, keepdim=False):
    assert idx_got.shape == idx_ref.shape, f'{label}: index shape mismatch'
    assert idx_got.dtype in (torch.int32, torch.int64), f'{label}: non-integer index dtype'
    assert x.numel() and torch.isfinite(x).all(), f'{label}: invalid input'
    xf = x.float()
    if dim is None:
        (xf, d) = (xf.flatten(), 0)
        (i_ref, i_got) = (idx_ref.reshape(1), idx_got.reshape(1))
    else:
        d = dim
        i_ref = idx_ref if keepdim else idx_ref.unsqueeze(d)
        i_got = idx_got if keepdim else idx_got.unsqueeze(d)
    (i_ref, i_got) = (i_ref.long(), i_got.long())
    ext = xf.amax(dim=d, keepdim=True) if mode == 'max' else xf.amin(dim=d, keepdim=True)
    sel = xf.gather(d, i_got)
    wrong = (sel != ext).nonzero()
    assert wrong.numel() == 0, f'{label}: device picked a NON-EXTREMAL element at {wrong.tolist()[:5]} — selected {sel[sel != ext][:5].tolist()}, extremum {ext.expand_as(sel)[sel != ext][:5].tolist()}'
    diff = (i_got != i_ref).nonzero()
    if diff.numel():
        mult = (xf == ext).sum(dim=d, keepdim=True)
        assert False, f'{label}: first-index tie-break broken at {diff.tolist()[:5]} — cpu {i_ref.flatten()[:5].tolist()} vs device {i_got.flatten()[:5].tolist()}; both attain the extremum (multiplicity {mult.flatten()[:5].tolist()}), so this is a tie-break change, not a wrong answer'

def test_argmax_basic():
    print('=' * 60)
    print('Testing argmax on device backend')
    print('=' * 60)
    print('\n[Test 1] 2D tensor [N, C], reduce over dim=-1 (C)')
    x_cpu = torch.randn(10, 128, generator=input_generator(), dtype=torch.float16)
    x_rpu = x_cpu.to('rpu')
    result_cpu = torch.argmax(x_cpu.float(), dim=-1)
    result_rpu = torch.argmax(x_rpu, dim=-1)
    result_rpu_cpu = result_rpu.to('cpu')
    print(f'  Input shape: {x_cpu.shape}')
    print(f'  Output shape: {result_cpu.shape}')
    print(f'  CPU result: {result_cpu[:5]}')
    print(f'  device result: {result_rpu_cpu[:5]}')
    _assert_arg_reduce(x_cpu, -1, result_cpu, result_rpu_cpu, 'max', 'argmax [10,128] dim=-1')
    print('  Match: True')
    print('\n[Test 2] 2D tensor [N, C], reduce over dim=0 (N)')
    result_cpu = torch.argmax(x_cpu.float(), dim=0)
    result_rpu = torch.argmax(x_rpu, dim=0)
    result_rpu_cpu = result_rpu.to('cpu')
    print(f'  Output shape: {result_cpu.shape}')
    print(f'  CPU result[:5]: {result_cpu[:5]}')
    print(f'  device result[:5]: {result_rpu_cpu[:5]}')
    _assert_arg_reduce(x_cpu, 0, result_cpu, result_rpu_cpu, 'max', 'argmax [10,128] dim=0')
    print('  Match: True')
    print('\n[Test 3] 3D tensor [B, N, C], reduce over dim=-1 (C)')
    x_cpu = torch.randn(2, 8, 256, generator=input_generator(), dtype=torch.float16)
    x_rpu = x_cpu.to('rpu')
    result_cpu = torch.argmax(x_cpu.float(), dim=-1)
    result_rpu = torch.argmax(x_rpu, dim=-1)
    result_rpu_cpu = result_rpu.to('cpu')
    print(f'  Input shape: {x_cpu.shape}')
    print(f'  Output shape: {result_cpu.shape}')
    print(f'  CPU result[0, :5]: {result_cpu[0, :5]}')
    print(f'  device result[0, :5]: {result_rpu_cpu[0, :5]}')
    _assert_arg_reduce(x_cpu, -1, result_cpu, result_rpu_cpu, 'max', 'argmax [2,8,256] dim=-1')
    print('  Match: True')
    print('\n[Test 4] LLM-like shape [1, 32000] (vocab size)')
    x_cpu = torch.randn(1, 32000, generator=input_generator(), dtype=torch.float16)
    x_rpu = x_cpu.to('rpu')
    result_cpu = torch.argmax(x_cpu.float(), dim=-1)
    result_rpu = torch.argmax(x_rpu, dim=-1)
    result_rpu_cpu = result_rpu.to('cpu')
    print(f'  Input shape: {x_cpu.shape}')
    print(f'  CPU argmax: {result_cpu.item()}')
    print(f'  device argmax: {result_rpu_cpu.item()}')
    _assert_arg_reduce(x_cpu, -1, result_cpu, result_rpu_cpu, 'max', 'argmax [1,32000] dim=-1')
    print('  Match: True')
    print('\n[Test 5] keepdim=True')
    x_cpu = torch.randn(4, 64, generator=input_generator(), dtype=torch.float16)
    x_rpu = x_cpu.to('rpu')
    result_cpu = torch.argmax(x_cpu.float(), dim=-1, keepdim=True)
    result_rpu = torch.argmax(x_rpu, dim=-1, keepdim=True)
    result_rpu_cpu = result_rpu.to('cpu')
    print(f'  Input shape: {x_cpu.shape}')
    print(f'  Output shape (keepdim=True): {result_cpu.shape}')
    print(f'  CPU result[:3]: {result_cpu[:3].squeeze()}')
    print(f'  device result[:3]: {result_rpu_cpu[:3].squeeze()}')
    _assert_arg_reduce(x_cpu, -1, result_cpu, result_rpu_cpu, 'max', 'argmax [4,64] keepdim', keepdim=True)
    print('  Match: True')

def test_argmin_basic(record_property):
    print('\n' + '=' * 60)
    print('Testing argmin on device backend')
    print('=' * 60)
    print('\n[Test 1] 2D tensor [N, C], reduce over dim=-1')
    x_cpu = torch.randn(10, 128, generator=input_generator(), dtype=torch.float16)
    x_rpu = x_cpu.to('rpu')
    result_cpu = torch.argmin(x_cpu.float(), dim=-1)
    result_rpu = torch.argmin(x_rpu, dim=-1)
    result_rpu_cpu = result_rpu.to('cpu')
    print(f'  Input shape: {x_cpu.shape}')
    print(f'  CPU result: {result_cpu[:5]}')
    print(f'  device result: {result_rpu_cpu[:5]}')
    assert result_rpu_cpu.dtype == torch.int32
    _assert_arg_reduce(x_cpu, -1, result_cpu, result_rpu_cpu, 'min', 'argmin [10,128] dim=-1')
    print('  Match: True')
    print('\n[Test 2] Large C dimension [1, 32000]')
    x_cpu = torch.randn(1, 32000, generator=input_generator(), dtype=torch.float16)
    x_rpu = x_cpu.to('rpu')
    result_cpu = torch.argmin(x_cpu.float(), dim=-1)
    result_rpu = torch.argmin(x_rpu, dim=-1)
    result_rpu_cpu = result_rpu.to('cpu')
    print(f'  CPU argmin: {result_cpu.item()}')
    print(f'  device argmin: {result_rpu_cpu.item()}')
    assert result_rpu_cpu.dtype == torch.int32
    _assert_arg_reduce(x_cpu, -1, result_cpu, result_rpu_cpu, 'min', 'argmin [1,32000] dim=-1')
    print('  Match: True')
    x_cpu = torch.randn(10, 128, generator=input_generator(), dtype=torch.float16)
    x_rpu = x_cpu.to('rpu')
    for (dim, label) in ((0, 'argmin [10,128] dim=0'), (None, 'argmin [10,128] global')):
        result_cpu = torch.argmin(x_cpu.float(), dim=dim)
        result_rpu_cpu = torch.argmin(x_rpu, dim=dim).to('cpu')
        assert result_rpu_cpu.dtype == torch.int32
        _assert_arg_reduce(x_cpu, dim, result_cpu, result_rpu_cpu, 'min', label)

def test_argmin_3d_dim0():
    print('\n' + '=' * 60)
    print('Testing argmin on 3D [B,N,C] with dim=0 (regression: bug_003)')
    print('=' * 60)
    x_cpu = torch.randn(2, 3, 4, generator=input_generator(), dtype=torch.float16)
    x_rpu = x_cpu.to('rpu')
    expected = torch.argmin(x_cpu.float(), dim=0)
    got = torch.argmin(x_rpu, dim=0).to('cpu')
    print(f'  Input shape: {x_cpu.shape}')
    print(f'  Expected shape: {expected.shape}, got shape: {got.shape}')
    assert got.shape == expected.shape, f'Shape mismatch: expected {expected.shape}, got {got.shape}'
    _assert_arg_reduce(x_cpu, 0, expected, got, 'min', 'argmin [2,3,4] dim=0')
    print(f'  Match: True')

def test_argmax_no_dim():
    print('\n' + '=' * 60)
    print('Testing argmax without dim (global)')
    print('=' * 60)
    x_cpu = torch.randn(4, 8, 16, generator=input_generator(), dtype=torch.float16)
    x_rpu = x_cpu.to('rpu')
    result_cpu = torch.argmax(x_cpu.float())
    result_rpu = torch.argmax(x_rpu)
    result_rpu_cpu = result_rpu.to('cpu')
    print(f'  Input shape: {x_cpu.shape}')
    print(f'  CPU global argmax: {result_cpu.item()}')
    print(f'  device global argmax: {result_rpu_cpu.item()}')
    _assert_arg_reduce(x_cpu, None, result_cpu, result_rpu_cpu, 'max', 'argmax [4,8,16] global')
    print('  Match: True')

def test_arg_reduce_tie_is_not_decided_by_index():
    print('\n' + '=' * 60)
    print('Testing arg-reduce with DELIBERATE duplicate extrema')
    print('=' * 60)
    x_cpu = torch.zeros(1, 32000, dtype=torch.float16)
    x_cpu[0, 100] = -1.0
    x_cpu[0, 20000] = -1.0
    assert int((x_cpu.float() == x_cpu.float().min()).sum()) == 2, 'input must be tied'
    ref = torch.argmin(x_cpu.float(), dim=-1)
    got = torch.argmin(x_cpu.to('rpu'), dim=-1).to('cpu')
    print(f'  CPU argmin={ref.item()}  device argmin={got.item()}')
    _assert_arg_reduce(x_cpu, -1, ref, got, 'min', 'argmin tied [1,32000]')
    alt = torch.tensor([20000])
    assert not torch.equal(alt, ref), 'index-only comparison rejects this correct answer'
    _assert_arg_reduce(x_cpu, -1, alt, alt, 'min', 'argmin tied [1,32000] last-index')
    print('  Match: True (value invariant accepts both extremal indices)')
def test_arg_reduce_after_shared_spm_pool_initialization(capfd):
    """Eager reductions reuse the shared pool without consuming its scratch."""
    import ctypes
    import re

    flush = ctypes.CDLL(None).fflush
    flush.argtypes = [ctypes.c_void_p]
    flush.restype = ctypes.c_int

    def shared_usage():
        torch.ops.rpu.spm_alloc_dump("arg_reduce_watermark")
        assert flush(None) == 0
        output = capfd.readouterr().out
        match = re.search(
            r"^\[SpmAllocator\] arg_reduce_watermark: (.+)$", output, re.MULTILINE
        )
        assert match is not None, output
        return match.group(1)

    torch.ops.rpu.spm_alloc_init()
    before = shared_usage()
    # Argmin uses native reduceC/C128/reduceN; argmax retains its CPU fallback.
    for shape, dim in [((1, 32000), -1), ((4, 64), -1), ((4, 64), 0)]:
        values = torch.arange(shape[0] * shape[1], dtype=torch.float32)
        cpu = (values.remainder(257) - 128).reshape(shape).half()
        device = cpu.to("rpu")
        for _ in range(3):
            for mode, reduce in [("min", torch.argmin), ("max", torch.argmax)]:
                expected = reduce(cpu.float(), dim=dim)
                actual = reduce(device, dim=dim).cpu()
                _assert_arg_reduce(
                    cpu, dim, expected, actual, mode,
                    f"shared-pool {mode} {shape} dim={dim}",
                )
                assert shared_usage() == before

def test_native_argmin_rejects_graph_capture():
    import pytest

    cpu = torch.arange(64, dtype=torch.float16).reshape(1, 64)
    device = cpu.to("rpu")
    cache = rpu_backend.graph.GraphCache()
    sig = rpu_backend.graph.GraphSignature(
        op_id="argmin_requires_passthrough", shapes=[1, 64], dtypes=[torch.float16]
    )
    try:
        with pytest.raises(RuntimeError, match="argmin requires PASSTHROUGH outside Graph"):
            with cache.capture(sig):
                torch.argmin(device, dim=-1)
        assert cache.cache_invariant_ok()
        assert all(entry.kernel_count == 0 for entry in cache.snapshot())
    finally:
        cache.clear()
    assert cache.size() == 0 and cache.cache_invariant_ok()
    actual = torch.argmin(device, dim=-1).cpu()
    assert torch.equal(actual.long(), torch.argmin(cpu.float(), dim=-1))

if __name__ == '__main__':
    torch.rpu.set_debug(False)
    test_argmax_basic()
    test_argmin_basic(lambda *_: None)
    test_argmin_3d_dim0()
    test_argmax_no_dim()
    test_arg_reduce_tie_is_not_decided_by_index()
    print('\n' + '=' * 60)
    print('All tests completed!')
    print('=' * 60)
