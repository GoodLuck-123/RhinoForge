"""Board-free source gate for the multiview Vision outer-fast facade."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp").read_text()
HEADER = (ROOT / "src/fused/rpu_qwen3vl_pooler_spm_z1.h").read_text()
FMB_HEADER = (ROOT / "src/core/fused_model_base.h").read_text()
FMB_SOURCE = (ROOT / "src/core/fused_model_base.cpp").read_text()


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def test_multiview_vision_component_is_staged_after_exact_lifecycle_checks():
    stage = _between(
        SOURCE,
        "void stage_multiview_vision_outer_fast_component(",
        "void bind_multiview_vision_outer_fast_input(",
    )
    publish = stage.index("stage_spm_outer_fast_component(")

    for marker in (
        "validate_multiview_composite_profile();",
        "multiview_composite_prepared_",
        "!multiview_composite_dispatch_",
        "multiview_primed_position_ptr_",
        "position_idx_keepalive_.data_ptr<int16_t>()",
        "multiview_primed_position_owner_",
        "position_idx_keepalive_.unsafeGetTensorImpl()",
    ):
        assert marker in stage[:publish]
    assert "guard, k_caches, v_caches" in stage[publish:]


def test_multiview_vision_input_binds_one_of_three_exact_composite_slots():
    bind = _between(
        SOURCE,
        "void bind_multiview_vision_outer_fast_input(",
        "void unprepare_multiview_composite_layout(",
    )
    publish = bind.index("bind_spm_outer_fast_composite_input(")

    for marker in (
        "ordinal < 3",
        "input.dim() == 3",
        "input.size(0) == 1",
        "input.size(1) == QWEN3VL_POOLER_Z1_NUM_PATCHES",
        "input.size(2) == QWEN3VL_POOLER_Z1_VISION_HIDDEN",
        "input.scalar_type() == at::kHalf",
        "input.device().type() == at::kPrivateUse1",
        "input.is_contiguous()",
    ):
        assert marker in bind[:publish]
    assert "guard, input, ordinal" in bind[publish:]


def test_multiview_vision_outer_fast_facade_is_internal_only():
    wrappers = SOURCE[SOURCE.index("namespace v3::qwen3vl_pooler_z1_internal {") :]
    for name in (
        "stage_multiview_vision_outer_fast_component",
        "bind_multiview_vision_outer_fast_input",
    ):
        assert name in HEADER
        assert name in wrappers
    assert "->stage_multiview_vision_outer_fast_component(" in wrappers
    assert "->bind_multiview_vision_outer_fast_input(guard, input, ordinal);" in wrappers


def test_generic_three_slot_binding_is_capability_gated_and_owner_bound():
    bind = _between(
        FMB_SOURCE,
        "void FusedModelBase::bind_spm_outer_fast_composite_input(",
        "// =============================================================================",
    )

    assert "bind_spm_outer_fast_composite_input(" in FMB_HEADER
    for marker in (
        "pipeline_lease_epoch_ != 0",
        "pipeline_plan_hash_ != 0",
        "pipeline_manifest_snapshot_.valid",
        "StableThreeInputSlotsSameOwnerPreload",
        "producer_ordinal <",
        "composite_hidden_in_src_bases_[producer_ordinal]",
        "slot != 0",
        "guard.bind_fast_mutable_dma(",
        "GraphOuterFastDmaSide::Source",
    ):
        assert marker in bind
