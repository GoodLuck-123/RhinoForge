import pytest
import rpu_backend
from rpu_backend.graph import GraphCache, GraphSignature
pytestmark = pytest.mark.skipif(not getattr(rpu_backend, '_cpp_loaded', False), reason='GraphCache capacity is enforced in C++; needs the native backend')
DEFAULT = 32

def _sig(i):
    return GraphSignature(op_id='cap_probe', shapes=(i,), dtypes=(5,))

def _fill(cache, n, start=0):
    for i in range(start, start + n):
        assert cache.get_or_create(_sig(i)) is not None
    return cache

def test_default_max_entries_is_32():
    assert GraphCache.DEFAULT_MAX_ENTRIES == DEFAULT
    assert GraphCache().max_entries() == DEFAULT

def test_31_and_32_entries_fit():
    cache = _fill(GraphCache(), DEFAULT - 1)
    assert cache.size() == DEFAULT - 1
    cache.get_or_create(_sig(DEFAULT - 1))
    assert cache.size() == DEFAULT

def test_entry_33_raises_and_cache_stays_full():
    cache = _fill(GraphCache(), DEFAULT)
    assert cache.size() == DEFAULT
    with pytest.raises(RuntimeError, match='capacity reached'):
        cache.get_or_create(_sig(DEFAULT))
    assert cache.size() == DEFAULT
    assert cache.lookup(_sig(DEFAULT)) is None

def test_existing_signature_still_resolves_at_capacity():
    cache = _fill(GraphCache(), DEFAULT)
    assert cache.get_or_create(_sig(0)) is not None
    assert cache.lookup(_sig(DEFAULT - 1)) is not None
    assert cache.size() == DEFAULT

def test_evict_is_the_documented_recovery():
    cache = _fill(GraphCache(), DEFAULT)
    cache.evict(_sig(0))
    assert cache.size() == DEFAULT - 1
    assert cache.lookup(_sig(0)) is None
    cache.get_or_create(_sig(DEFAULT))
    assert cache.size() == DEFAULT

def test_clear_frees_every_slot():
    cache = _fill(GraphCache(), DEFAULT)
    cache.clear()
    assert cache.size() == 0
    _fill(cache, DEFAULT)
    assert cache.size() == DEFAULT

@pytest.mark.parametrize('cap', [1, 2, 5])
def test_bound_is_the_configured_value_not_a_hardcoded_32(cap):
    cache = GraphCache(max_entries=cap)
    assert cache.max_entries() == cap
    _fill(cache, cap)
    assert cache.size() == cap
    with pytest.raises(RuntimeError, match='capacity reached'):
        cache.get_or_create(_sig(cap))

def test_repeated_same_signature_consumes_one_slot():
    cache = GraphCache(max_entries=2)
    for _ in range(10):
        cache.get_or_create(_sig(0))
    assert cache.size() == 1
