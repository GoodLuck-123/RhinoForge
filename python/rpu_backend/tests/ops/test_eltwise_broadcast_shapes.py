import pytest
import torch
import rpu_backend

def _check(shape_a, shape_b, op=torch.mul):
    torch.manual_seed(0)
    a = torch.randn(*shape_a, dtype=torch.float16)
    b = torch.randn(*shape_b, dtype=torch.float16).abs() + 0.5
    ref = op(a, b)
    got = op(a.to('rpu'), b.to('rpu')).cpu()
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert got.numel() and torch.isfinite(got).all() and torch.isfinite(ref).all()
    cos = torch.nn.functional.cosine_similarity(ref.float().flatten(), got.float().flatten(), dim=0).item()
    assert cos > 0.999, f'fp16 {tuple(shape_a)} {op.__name__} {tuple(shape_b)}: cos={cos:.6f} max_abs={(ref.float() - got.float()).abs().max().item():.6f}'
    torch.testing.assert_close(got, ref, rtol=0.001, atol=0.001)

def test_no_broadcast_is_correct():
    _check((1, 128, 1024), (1, 128, 1024))

def test_middle_axis_broadcast():
    _check((1, 128, 1024), (1, 1, 1024))

def test_scm_independent_u16_tail():
    _check((1, 33, 64), (1, 1, 64))

def test_rank_mismatch_broadcast():
    _check((1, 128, 1024), (1024,))

def test_trailing_axis_broadcast():
    _check((1, 128, 1024), (1, 128, 1))

@pytest.mark.parametrize('n', [256, 1024])
def test_many_blocks(n):
    _check((1, n, 1024), (1, 1, 1024))
    _check((1, n, 1024), (1, n, 1))

def test_partial_overlap_stride():
    _check((1, 704, 256), (1, 1, 256))

@pytest.mark.parametrize('n', [352, 512, 2048])
def test_shapes_that_were_already_right_stay_right(n):
    _check((1, n, 256), (1, 1, 256))
    _check((1, n, 16), (1, 1, 16))

@pytest.mark.parametrize('c', [24, 100])
def test_c_not_v16_aligned(c):
    _check((1, 8, c), (1, 1, c))
    _check((1, 8, c), (1, 8, 1))

@pytest.mark.parametrize('c', [3072, 4304, 6144])
def test_c_exhausts_the_vlm_budget(c):
    _check((1, 100, c), (1, 1, c))
    _check((1, 100, c), (1, 100, 1))

def test_vision_intermediate_rank_mismatch():
    _check((1, 100, 4304), (4304,))

@pytest.mark.parametrize('op', [torch.mul, torch.add, torch.sub, torch.div])
def test_every_binary_op_shares_the_wrapper(op):
    _check((1, 128, 1024), (1, 1, 1024), op)
    _check((1, 128, 1024), (1, 128, 1), op)
PATTERN_MATRIX = [('1xC_NxC-launch', (1, 1, 256), (1, 512, 256), True), ('1xC_NxC-demote', (1, 1, 1024), (1, 128, 1024), False), ('NxC_1xC-launch', (1, 512, 256), (1, 1, 256), True), ('NxC_1xC-demote', (1, 128, 1024), (1, 1, 1024), False), ('Nx1_NxC-launch', (1, 64, 1), (1, 64, 1024), True), ('Nx1_NxC-demote', (1, 128, 1), (1, 128, 1024), False), ('NxC_Nx1-launch', (1, 64, 1024), (1, 64, 1), True), ('NxC_Nx1-demote', (1, 128, 1024), (1, 128, 1), False)]

@pytest.mark.parametrize('op', [torch.mul, torch.add, torch.sub, torch.div], ids=lambda o: o.__name__)
@pytest.mark.parametrize('label,sa,sb,launchable', PATTERN_MATRIX, ids=[r[0] for r in PATTERN_MATRIX])
def test_pattern_matrix(label, sa, sb, launchable, op):
    _check(sa, sb, op)

def test_mid_axis_broadcast_with_non_unit_prefix(record_property):
    for op in (torch.mul, torch.sub):
        _check((1, 1, 8, 64), (3, 1, 8, 64), op)
        _check((3, 1, 8, 64), (1, 1, 8, 64), op)
        _check((1, 1, 8, 1024), (128, 1, 8, 1024), op)

@pytest.mark.parametrize('op', [torch.mul, torch.div], ids=lambda o: o.__name__)
def test_rank_2_and_rank_4(op):
    _check((512, 256), (1, 256), op)
    _check((128, 1024), (1, 1024), op)
    _check((64, 1024), (64, 1), op)
    _check((1, 8, 64, 64), (1, 1, 1, 64), op)
    _check((2, 4, 16, 128), (2, 4, 16, 1), op)
    _check((2, 4, 16, 1024), (1024,), op)

def test_nx1_c256_special_branch(record_property):
    _check((1, 700, 256), (1, 700, 1))
    _check((1, 700, 1), (1, 700, 256))

def test_scalar_shaped_operand_never_reaches_pattern_matching(record_property):
    for op in (torch.mul, torch.add, torch.sub, torch.div):
        _check((1, 128, 1024), (1, 1, 1), op)

def test_unmatchable_broadcast_materialises():
    for op in (torch.mul, torch.sub):
        _check((1, 8, 1), (1, 1, 8), op)
_INPLACE_ROWS = []
for _op in ('mul_', 'add_', 'sub_', 'div_'):
    for (_lbl, _sa, _sb, _launch) in (('NxC_1xC-launch', (1, 512, 256), (1, 1, 256), True), ('NxC_1xC-demote', (1, 128, 1024), (1, 1, 1024), False), ('NxC_Nx1-launch', (1, 64, 1024), (1, 64, 1), True), ('NxC_Nx1-demote', (1, 128, 1024), (1, 128, 1), False)):
        _INPLACE_ROWS.append(pytest.param(_op, _sa, _sb, _launch, id=f'{_op}-{_lbl}'))

@pytest.mark.parametrize('op,sa,sb,launchable', _INPLACE_ROWS)
def test_inplace_broadcast(op, sa, sb, launchable):
    torch.manual_seed(0)
    a = torch.randn(*sa, dtype=torch.float16).abs() + 0.5
    b = torch.randn(*sb, dtype=torch.float16).abs() + 0.5
    ref = getattr(a.clone(), op)(b)
    got = getattr(a.clone().to('rpu'), op)(b.to('rpu')).cpu()
    assert got.shape == ref.shape
    cos = torch.nn.functional.cosine_similarity(ref.float().flatten(), got.float().flatten(), dim=0).item()
    assert cos > 0.999, f'fp16 {tuple(sa)}.{op}({tuple(sb)}) launchable={launchable}: cos={cos:.6f} max_abs={(ref.float() - got.float()).abs().max().item():.6f}'
_ALPHA_ROWS = []
for (_opname, _lbl, _sa, _sb, _bad) in (('add', '1xC_NxC-launch', (1, 1, 256), (1, 512, 256), False), ('add', 'NxC_1xC-launch', (1, 512, 256), (1, 1, 256), True), ('add', 'Nx1_NxC-launch', (1, 64, 1), (1, 64, 1024), False), ('add', 'NxC_Nx1-launch', (1, 64, 1024), (1, 64, 1), True), ('add', 'NxC_1xC-demote', (1, 128, 1024), (1, 1, 1024), False), ('sub', '1xC_NxC-launch', (1, 1, 256), (1, 512, 256), False), ('sub', 'NxC_1xC-launch', (1, 512, 256), (1, 1, 256), False), ('sub', 'Nx1_NxC-launch', (1, 64, 1), (1, 64, 1024), False), ('sub', 'NxC_Nx1-launch', (1, 64, 1024), (1, 64, 1), False), ('sub', 'NxC_1xC-demote', (1, 128, 1024), (1, 1, 1024), False)):
    _ALPHA_ROWS.append(pytest.param(_opname, _sa, _sb, id=f'{_opname}-{_lbl}'))

@pytest.mark.parametrize('opname,sa,sb', _ALPHA_ROWS)
def test_alpha_reaches_both_kernels(opname, sa, sb):
    op = torch.add if opname == 'add' else torch.sub
    torch.manual_seed(0)
    a = torch.randn(*sa, dtype=torch.float16)
    b = torch.randn(*sb, dtype=torch.float16).abs() + 0.5
    ref = op(a, b, alpha=2.5)
    got = op(a.to('rpu'), b.to('rpu'), alpha=2.5).cpu()
    cos = torch.nn.functional.cosine_similarity(ref.float().flatten(), got.float().flatten(), dim=0).item()
    assert cos > 0.999, f'{opname}(alpha=2.5) {tuple(sa)} {tuple(sb)}: cos={cos:.6f} max_abs={(ref.float() - got.float()).abs().max().item():.6f}'
