import pytest
import rpu_backend
pytestmark = pytest.mark.skipif(not getattr(rpu_backend, '_cpp_loaded', False), reason='shape_plan is a C++ predicate; needs the native backend')
OK = dict(position=0, seq_len=256, num_cores=8, num_kv_heads=8, head_dim=64)

def plan(**kw):
    from rpu_backend._cpp_ext import kv_helpers
    return kv_helpers.shape_plan(**{**OK, **kw})

def test_aligned_shape_is_v16_eligible():
    p = plan()
    assert p['v16_shape_ok'] is True
    assert p['tail_seq'] == 0
    assert p['hybrid_shape_ok'] is False, 'no tail => nothing for the v2 tail to do'

@pytest.mark.parametrize('position', [0, 16, 240, 8176])
def test_position_must_be_16_aligned(position):
    assert plan(position=position)['v16_shape_ok'] is True

@pytest.mark.parametrize('position', [1, 15, 17, 31, 255])
def test_unaligned_position_disqualifies_v16_entirely(position):
    p = plan(position=position)
    assert p['v16_shape_ok'] is False
    assert p['hybrid_shape_ok'] is False

@pytest.mark.parametrize('seq_len', [1, 15])
def test_seq_below_16_has_no_v16_and_no_bulk(seq_len):
    p = plan(seq_len=seq_len)
    assert p['v16_shape_ok'] is False
    assert p['bulk_seq'] == 0
    assert p['hybrid_shape_ok'] is False, 'bulk_seq >= 16 is required'

def test_seq_exactly_16_is_the_lower_edge():
    p = plan(seq_len=16)
    assert p['v16_shape_ok'] is True
    assert (p['bulk_seq'], p['tail_seq']) == (16, 0)

@pytest.mark.parametrize('num_cores', [1, 2, 4, 7])
def test_v16_requires_exactly_8_cores(num_cores):
    assert plan(num_cores=num_cores)['v16_shape_ok'] is False

@pytest.mark.parametrize('num_kv_heads', [2, 4, 7])
def test_kv_heads_must_be_divisible_by_cores(num_kv_heads):
    assert plan(num_kv_heads=num_kv_heads)['v16_shape_ok'] is False

@pytest.mark.parametrize('num_kv_heads', [8, 16, 32])
def test_kv_head_multiples_of_8_are_fine(num_kv_heads):
    assert plan(num_kv_heads=num_kv_heads)['v16_shape_ok'] is True

@pytest.mark.parametrize('head_dim,ok', [(64, True), (128, True), (256, True), (72, False), (80, True), (24, False)])
def test_head_dim_must_be_16_aligned(head_dim, ok):
    assert plan(head_dim=head_dim)['v16_shape_ok'] is ok

@pytest.mark.parametrize('seq_len,bulk,tail', [(255, 240, 15), (17, 16, 1), (31, 16, 15), (33, 32, 1), (4095, 4080, 15)])
def test_hybrid_splits_bulk_and_tail(seq_len, bulk, tail):
    p = plan(seq_len=seq_len)
    assert (p['bulk_seq'], p['tail_seq']) == (bulk, tail)
    assert p['v16_shape_ok'] is False, 'the whole sequence is not v16-aligned'
    assert p['hybrid_shape_ok'] is True, 'but the aligned bulk still is'

@pytest.mark.parametrize('seq_len', [16, 32, 256, 4096])
def test_aligned_lengths_need_no_hybrid(seq_len):
    p = plan(seq_len=seq_len)
    assert p['tail_seq'] == 0
    assert p['v16_shape_ok'] is True
    assert p['hybrid_shape_ok'] is False

def test_bulk_and_tail_always_reconstruct_seq_len():
    for seq_len in range(1, 300):
        p = plan(seq_len=seq_len)
        assert p['bulk_seq'] + p['tail_seq'] == seq_len
        assert p['bulk_seq'] % 16 == 0
        assert 0 <= p['tail_seq'] < 16

def test_hybrid_never_claims_a_shape_v16_would_reject():
    for seq_len in (17, 31, 255, 4095):
        for position in (0, 16, 1, 15):
            for num_cores in (8, 4):
                p = plan(seq_len=seq_len, position=position, num_cores=num_cores)
                if not p['hybrid_shape_ok']:
                    continue
                bulk = plan(seq_len=p['bulk_seq'], position=position, num_cores=num_cores)
                assert bulk['v16_shape_ok'] is True
