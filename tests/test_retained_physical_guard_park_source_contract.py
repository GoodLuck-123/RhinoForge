"""Board-free contract for retained physical-Graph park/rebind."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src/graph/graph_runtime.h"
RUNTIME = ROOT / "src/graph/graph_runtime.cpp"
PIPELINE = ROOT / "src/core/rpu_spm_pipeline.cpp"


def _function(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start : index + 1]
    raise AssertionError(f"unterminated function: {signature}")


def test_public_state_is_explicit_and_move_only():
    header = HEADER.read_text(encoding="utf-8")
    assert "RpuRetainedPhysicalArenaGuard(\n        const RpuRetainedPhysicalArenaGuard&) = delete;" in header
    assert "bool bound() const noexcept" in header
    assert "void park(const RpuPhysicalArenaAuthority& authority);" in header
    assert "void rebind(const RpuPhysicalArenaAuthority& authority);" in header
    assert "dropping it early leaves the process registry" in header
    assert "fail-closed" in header


def test_park_drops_only_volatile_lease_identity():
    source = RUNTIME.read_text(encoding="utf-8")
    park = _function(source, "void RpuRetainedPhysicalArenaGuard::park(")
    assert "require_graph_quiescent=*/true" in park
    assert "record.graph_state == RetainedGraphLifetimeState::Live" in park
    for immutable in (
        "record.plan_hash == plan_hash_",
        "record.persistent_generation == persistent_generation_",
        "record.arena_base == arena_base_",
        "record.arena_end == arena_end_",
        "record.reserved_top_bytes == reserved_top_bytes_",
    ):
        assert immutable in park
    assert "record.binding_state = RetainedPhysicalArenaBindingState::Parked" in park
    for volatile in (
        "record.physical_execution_token = 0",
        "record.lease_epoch = 0",
        "record.allocator_generation = 0",
        "record.lease_generation.reset()",
        "record.expected_lease_generation = 0",
    ):
        assert volatile in park
    assert "record.plan_hash = 0" not in park
    assert "record.persistent_generation = 0" not in park
    assert "record.arena_base = 0" not in park
    assert "record.arena_end = 0" not in park
    assert "record.reserved_top_bytes = 0" not in park


def test_rebind_requires_exact_immutable_plan_and_fresh_authority():
    source = RUNTIME.read_text(encoding="utf-8")
    rebind = _function(source, "void RpuRetainedPhysicalArenaGuard::rebind(")
    assert "require_graph_quiescent=*/true" in rebind
    assert "record.binding_state == RetainedPhysicalArenaBindingState::Parked" in rebind
    assert "record.graph_state == RetainedGraphLifetimeState::Live" in rebind
    for immutable in (
        "authority.plan_hash_ == plan_hash_",
        "authority.persistent_generation_ == persistent_generation_",
        "authority.arena_base_ == arena_base_",
        "authority.arena_end_ == arena_end_",
        "authority.reserved_top_bytes_ == reserved_top_bytes_",
    ):
        assert immutable in rebind
    assert "authority.lease_epoch_ != 0" in rebind
    assert "authority.allocator_generation_ != 0" in rebind
    assert "*authority_generation == authority.expected_lease_generation_" in rebind
    assert "record.binding_state = RetainedPhysicalArenaBindingState::Bound" in rebind
    assert "record.physical_execution_token = live_physical_token" in rebind
    assert "record.lease_epoch = authority.lease_epoch_" in rebind
    assert "record.allocator_generation = authority.allocator_generation_" in rebind


def test_parked_graph_cannot_replay_and_does_not_block_lease_release():
    source = RUNTIME.read_text(encoding="utf-8")
    replay = _function(source, "void validate_retained_physical_graph_replay(")
    token = _function(source, "uint64_t bound_physical_execution_token_for_graph(")
    release = _function(
        source,
        "void RpuRetainedPhysicalArenaGuard::require_release_allowed(",
    )
    active = _function(source, "bool has_active_retained_physical_arena_guards()")
    assert "record.binding_state == RetainedPhysicalArenaBindingState::Bound" in replay
    assert "RetainedPhysicalArenaBindingState::Bound" in token
    assert ": 0;" in token
    assert "RetainedPhysicalArenaBindingState::Parked" in release
    assert "continue;" in release
    assert "registry.provisional_by_graph_lifetime.empty()" in release
    assert "RetainedPhysicalArenaBindingState::Bound" in active


def test_parked_retirement_requires_graph_invalidation_and_cleanup_claim():
    source = RUNTIME.read_text(encoding="utf-8")
    retire = _function(source, "void RpuRetainedPhysicalArenaGuard::retire()")
    assert "RpuExecutionCleanupGuard" in retire
    assert "retire parked" in retire
    assert "record.graph_state != RetainedGraphLifetimeState::Live" in retire
    assert "registry.token_by_graph_lifetime.erase" in retire
    assert "registry.records.erase" in retire
    assert "clear();" in retire


def test_current_contract_is_thread_affine_and_release_gates_before_mutation():
    runtime = RUNTIME.read_text(encoding="utf-8")
    pipeline = PIPELINE.read_text(encoding="utf-8")
    rebind = _function(runtime, "void RpuRetainedPhysicalArenaGuard::rebind(")
    retire = _function(runtime, "void RpuRetainedPhysicalArenaGuard::retire()")
    release = _function(pipeline, "void SpmPipelineLease::release()")
    assert "owner_thread_ == std::this_thread::get_id()" in rebind
    assert "authority.owner_thread_ == owner_thread_" in rebind
    assert "owner_thread_ == std::this_thread::get_id()" in retire
    gate = release.index("RpuExecutionCoordinator::require_graph_quiescent(")
    unlock = release.index("SPM_ALLOC.pipeline_arena_locked_ = false")
    reset = release.index("SPM_ALLOC.reset_temporary()")
    exit_claim = release.index("RpuExecutionCoordinator::exit_physical(")
    assert gate < unlock < reset < exit_claim
