"""Board-free contracts for generic caching-allocator lifecycle and stats."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOCATOR_CPP = (ROOT / "src/core/rpu_caching_allocator.cpp").read_text(
    encoding="utf-8"
)
ALLOCATOR_H = (ROOT / "src/core/rpu_caching_allocator.h").read_text(
    encoding="utf-8"
)
BACKEND = (ROOT / "src/core/rpu_backend.cpp").read_text(encoding="utf-8")
PYBIND = (ROOT / "src/core/rpu_pybind.inc").read_text(encoding="utf-8")
PROFILE = (ROOT / "src/core/rpu_profile.h").read_text(encoding="utf-8")
GRAPH_EXECUTE = (ROOT / "src/graph/graph_runtime_execute.cpp").read_text(
    encoding="utf-8"
)


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def test_allocator_closes_before_the_sdk_ddr_manager():
    shutdown = _between(BACKEND, "void rpu_shutdown()", "// KernelCache remains")
    assert shutdown.index("mark_shutdown()") < shutdown.index("SPM_POOL.clear()")
    assert shutdown.index("SPM_POOL.clear()") < shutdown.index("rpu::empty_cache()")
    assert shutdown.index("rpu::empty_cache()") < shutdown.index("RpuDdrShutdown()")
    assert "if (g_rpu_caching_allocator_enabled)" not in shutdown
    assert "void RPUCachingAllocator::mark_shutdown()" in ALLOCATOR_CPP


def test_allocator_memory_stats_remain_exposed():
    for binding in (
        "empty_cache",
        "get_memory_stats",
        "reset_peak_memory_stats",
        "reset_accumulated_memory_stats",
    ):
        assert f'm.def("{binding}"' in PYBIND
    assert 'd["num_device_alloc"] = stats.num_device_alloc' in BACKEND


def test_sdk07b_live_map_workaround_is_not_in_the_backend():
    source = "\n".join(
        (ALLOCATOR_CPP, ALLOCATOR_H, BACKEND, PYBIND, PROFILE, GRAPH_EXECUTE)
    )
    for token in (
        "prime_unified_arena",
        "prime_unified_ddr_arena",
        "flush_ddr_bounded",
        "unified_arena_",
        "active_blocks_by_address_",
        "_DDR_ARENA_BYTES",
    ):
        assert token not in source
