import pytest
from rpu_backend._cpp_ext import sdpa_helpers as h
from rpu_backend.adapters.wall_oss import llm
GEOM = dict(num_q_heads=16, num_kv_heads=8, num_cores=8, head_dim=128, attn_mask_type=1)
CAP = llm._PREFILL_CHUNK_SIZE_CAP_TP8_PAD16

def _plan(real_len):
    return llm._prefill_execution_plan(real_len, CAP, pad16=True, equal_two=True, geom=GEOM)

@pytest.mark.parametrize('real_len', list(range(545, 641)))
def test_release_envelope_plans_are_executable(real_len):
    (execution_len, chunk_size) = _plan(real_len)
    assert (execution_len, chunk_size) == (640, 320)
    assert h.is_valid_chunk_size(**GEOM, cs=chunk_size, seq_len=execution_len, position=0), f'host planned ({execution_len}, {chunk_size}) for real_len={real_len}, which the C++ chunk predicate rejects'

@pytest.mark.parametrize('real_len', [1, 16, 100, 257, 383, 385, 449, 512, 544])
def test_shorter_prefixes_are_executable_too(real_len):
    (execution_len, chunk_size) = _plan(real_len)
    assert execution_len >= real_len
    assert h.is_valid_chunk_size(**GEOM, cs=chunk_size, seq_len=execution_len, position=0), f'host planned ({execution_len}, {chunk_size}) for real_len={real_len}, which the C++ chunk predicate rejects'

@pytest.mark.parametrize('real_len', [641, 700, 737, 768, 769])
def test_beyond_the_envelope_refuses_instead_of_certifying(real_len):
    with pytest.raises(ValueError, match='no safe two-chunk execution profile'):
        _plan(real_len)

def test_the_predicate_the_host_asks_is_the_one_that_moved():
    assert h.is_valid_chunk_size(**GEOM, cs=384, seq_len=384, position=0)
    assert not h.is_valid_chunk_size(**GEOM, cs=384, seq_len=768, position=0)
    assert not llm._equal_two_chunk_supported(384, 768, GEOM)
if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-q']))
