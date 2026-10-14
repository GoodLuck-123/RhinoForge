"""Board-free source gate for the multiview Text outer-fast facade."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/fused/rpu_qwen3_model.cpp").read_text(encoding="utf-8")
HEADER = (ROOT / "src/fused/rpu_qwen3_model.h").read_text(encoding="utf-8")
COORDINATOR_HEADER = (
    ROOT / "src/fused/rpu_qwen3vl_pooler_spm_z1.h"
).read_text(encoding="utf-8")
FMB_SOURCE = (ROOT / "src/core/fused_model_base.cpp").read_text(encoding="utf-8")
FMB_HEADER = (ROOT / "src/core/fused_model_base.h").read_text(encoding="utf-8")


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def test_text_outer_fast_owns_four_stable_complement_slots():
    state = _between(
        HEADER,
        "QWEN3VL_MULTIVIEW_TEXT_COMPLEMENT_RUNS",
        "// ── Kernel selection",
    )

    for run in (
        "std::array<int64_t, 2>{0, 1}",
        "std::array<int64_t, 2>{65, 2}",
        "std::array<int64_t, 2>{131, 2}",
        "std::array<int64_t, 2>{197, 28}",
    ):
        assert run in state
    assert "std::array<uint64_t, 4>" in state
    assert "qwen3vl_multiview_text_complement_src_bases_{}" in state
    assert "at::Tensor qwen3vl_multiview_text_output_owner_;" in state


def test_row_run_dma_has_an_explicit_live_base_and_each_run_uses_its_slot():
    declarations = _between(
        FMB_HEADER,
        "void emit_layer_input_row_run_dma(",
        "void emit_spm_pipeline_composite_run_add(",
    )
    implementations = _between(
        FMB_SOURCE,
        "void FusedModelBase::emit_layer_input_row_run_dma(",
        "void FusedModelBase::\nrequire_spm_pipeline_composite_consumer_physical_authority() const",
    )
    ingress = _between(
        SOURCE,
        "void CausalDecoderModel::emit_qwen3vl_multiview_text_complement_ingress(",
        "void CausalDecoderModel::emit_qwen3vl_multiview_text_retained_deepstack_add(",
    )

    assert "uint64_t& live_src_base" in declarations
    assert "uint64_t& live_src_base" in implementations
    assert "rpu_launch_ddr_broadcast_spm_dma_mutable(" in implementations
    assert "live_src_base, *pimpl_->ctx_.hidden_states" in implementations
    assert "for (size_t ordinal = 0;" in ingress
    assert "QWEN3VL_MULTIVIEW_TEXT_COMPLEMENT_RUNS.size()" in ingress
    assert "qwen3vl_multiview_text_complement_src_bases_[ordinal]" in ingress
    assert "run[0], run[1], \"residual1\"" in ingress


def test_text_component_staging_requires_the_exact_primed_idle_lifecycle():
    stage = _between(
        SOURCE,
        "void CausalDecoderModel::stage_qwen3vl_multiview_text_outer_fast_component(",
        "void CausalDecoderModel::bind_qwen3vl_multiview_text_outer_fast_input(",
    )
    publish = stage.index("stage_spm_outer_fast_component(")

    for marker in (
        "validate_qwen3vl_multiview_text_contract();",
        "qwen3vl_multiview_text_composite_inputs_primed_",
        "!qwen3vl_multiview_text_composite_dispatch_",
        "qwen3vl_multiview_text_composite_position_ids_addr_ != 0",
        "qwen3vl_multiview_text_composite_rope_cos_il_addr_ != 0",
        "qwen3vl_multiview_text_composite_rope_sin_il_addr_ != 0",
        "ka_last_position_ == 0",
        "ka_last_seq_ ==",
        "ka_last_pos_src_ != nullptr",
        "ka_last_cos_src_ != nullptr",
        "ka_last_sin_src_ != nullptr",
        "partial_mrope_active_",
    ):
        assert marker in stage[:publish]
    assert "guard, k_caches, v_caches" in stage[publish:]


def test_text_input_binds_all_four_exact_mutable_source_ranges():
    bind = _between(
        SOURCE,
        "void CausalDecoderModel::bind_qwen3vl_multiview_text_outer_fast_input(",
        "at::Tensor CausalDecoderModel::qwen3vl_multiview_text_outer_fast_output()",
    )
    publish = bind.index("guard.bind_fast_mutable_dma(")

    for marker in (
        "validate_qwen3vl_multiview_text_contract();",
        "qwen3vl_multiview_text_composite_inputs_primed_",
        "!qwen3vl_multiview_text_composite_dispatch_",
        "hidden_states.dim() == 3",
        "hidden_states.size(0) == 1",
        "hidden_states.size(1) == QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN",
        "hidden_states.size(2) == QWEN3VL_MULTIVIEW_TEXT_HIDDEN",
        "hidden_states.scalar_type() == at::kHalf",
        "hidden_states.device().type() == at::kPrivateUse1",
        "hidden_states.is_contiguous()",
        "qwen3vl_multiview_text_complement_src_bases_",
    ):
        assert marker in bind[:publish]
    assert "for (size_t ordinal = 0;" in bind
    assert "QWEN3VL_MULTIVIEW_TEXT_COMPLEMENT_RUNS.size()" in bind
    assert "qwen3vl_multiview_text_complement_src_bases_[ordinal]" in bind
    assert "GraphOuterFastDmaSide::Source" in bind[publish:]
    assert "static_cast<size_t>(run[0]) * row_bytes" in bind[publish:]
    assert "static_cast<size_t>(run[1]) * row_bytes" in bind[publish:]


def test_outer_fast_output_is_replay_only_exact_and_owner_bound():
    forward = _between(
        SOURCE,
        "at::Tensor CausalDecoderModel::forward_qwen3vl_multiview_text_composite(",
        "void CausalDecoderModel::emit_qwen3vl_multiview_text_complement_ingress(",
    )
    output = _between(
        SOURCE,
        "at::Tensor CausalDecoderModel::qwen3vl_multiview_text_outer_fast_output()",
        "void CausalDecoderModel::emit_qwen3vl_multiview_text_complement_ingress(",
    )

    assert "const at::Tensor& registry_output = output_tensor();" in forward
    assert "result.is_same(registry_output)" in forward
    assert "qwen3vl_multiview_text_output_owner_ = result;" in forward
    for marker in (
        "RpuKernelGraph::has_active()",
        "RpuKernelGraph::State::REPLAYING",
        "validate_qwen3vl_multiview_text_contract();",
        "qwen3vl_multiview_text_composite_inputs_primed_",
        "!qwen3vl_multiview_text_composite_dispatch_",
        "const at::Tensor& current = output_tensor();",
        "current.size(0) == 1",
        "current.size(1) == QWEN3VL_MULTIVIEW_TEXT_EXECUTION_LEN",
        "current.size(2) == QWEN3VL_MULTIVIEW_TEXT_HIDDEN",
        "current.scalar_type() == at::kHalf",
        "current.device().type() == at::kPrivateUse1",
        "current.is_contiguous()",
        "qwen3vl_multiview_text_output_owner_.is_same(current)",
        "unsafeGetStorageImpl()",
        "current.data_ptr()",
        "return current;",
    ):
        assert marker in output


def test_prepare_and_teardown_fail_closed_on_stale_slots_or_output_owner():
    prepare = _between(
        SOURCE,
        "CausalDecoderModel::prepare_qwen3vl_multiview_text_composite_layout(",
        "CausalDecoderModel::resolve_qwen3vl_multiview_text_composite_profile()",
    )
    unprepare = _between(
        SOURCE,
        "void CausalDecoderModel::unprepare_qwen3vl_multiview_text_composite_layout()",
        "at::Tensor CausalDecoderModel::forward_qwen3vl_multiview_text_composite(",
    )
    destroy = _between(
        SOURCE,
        "void CausalDecoderModel::check_qwen3vl_pooler_z1_destroy_allowed() const",
        "}  // namespace v3",
    )

    assert "qwen3vl_multiview_text_complement_src_bases_" in prepare
    assert "slot == 0" in prepare
    assert "!qwen3vl_multiview_text_output_owner_.defined()" in prepare
    assert "qwen3vl_multiview_text_complement_src_bases_.fill(0);" in unprepare
    assert "qwen3vl_multiview_text_output_owner_ = at::Tensor();" in unprepare
    assert "qwen3vl_multiview_text_complement_src_bases_" in destroy
    assert "slot == 0" in destroy
    assert "!qwen3vl_multiview_text_output_owner_.defined()" in destroy


def test_internal_stage_bind_output_facade_is_declared_and_forwarded():
    wrappers = _between(
        SOURCE,
        "namespace v3::qwen3vl_pooler_z1_internal {",
        "}  // namespace v3::qwen3vl_pooler_z1_internal",
    )
    for name in (
        "stage_multiview_text_outer_fast_component(",
        "bind_multiview_text_outer_fast_input(",
        "multiview_text_outer_fast_output(",
    ):
        assert name in HEADER
        assert name in COORDINATOR_HEADER
        assert name in wrappers
    assert "->stage_qwen3vl_multiview_text_outer_fast_component(" in wrappers
    assert "->bind_qwen3vl_multiview_text_outer_fast_input(" in wrappers
    assert "->qwen3vl_multiview_text_outer_fast_output();" in wrappers
