"""Board-free source gates for the LingBot2 N256/S225 dry-layout probe."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COORDINATOR = ROOT / "src/fused/rpu_qwen3vl_pooler_spm_z1.cpp"
COORDINATOR_HEADER = ROOT / "src/fused/rpu_qwen3vl_pooler_spm_z1.h"
TEXT_MODEL = ROOT / "src/fused/rpu_qwen3_model.cpp"
TEXT_HEADER = ROOT / "src/fused/rpu_qwen3_model.h"
VISION_MODEL = ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp"
FMB = ROOT / "src/core/fused_model_base.cpp"
DECLS = ROOT / "src/core/rpu_kernel_decls.h"
REGISTRATIONS = ROOT / "src/core/rpu_dispatch_registrations.inc"


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def test_probe_is_one_default_only_control_plane_op():
    schema = (
        "qwen3vl_multiview_spm_dry_probe(int vision_handle, int text_handle, "
        "int num_patches, int execution_len) -> int[]"
    )
    registrations = REGISTRATIONS.read_text(encoding="utf-8")
    assert registrations.count(schema) == 1
    assert 'm.impl("qwen3vl_multiview_spm_dry_probe"' not in registrations

    for path in (COORDINATOR_HEADER, DECLS):
        assert "std::vector<int64_t> rpu_qwen3vl_multiview_spm_dry_probe(" in (
            path.read_text(encoding="utf-8")
        )
    for split_name in (
        "prepare_multiview_vision_dry",
        "cancel_multiview_vision_dry",
        "prepare_multiview_text_dry",
        "cancel_multiview_text_dry",
    ):
        assert f'm.def("{split_name}' not in registrations


def test_probe_is_quiescent_reverse_transaction_without_execution():
    source = COORDINATOR.read_text(encoding="utf-8")
    probe = _between(
        source,
        "std::vector<int64_t> rpu_qwen3vl_multiview_spm_dry_probe(",
        "int64_t rpu_qwen3vl_pooler_spm_z1_prepare(",
    )
    guard = probe.index("RpuExecutionCleanupGuard execution_guard(")
    vision_prepare = probe.index("prepare_multiview_vision_dry(")
    text_prepare = probe.index("prepare_multiview_text_dry(")
    collect = probe.index("result = {")
    text_cancel = probe.index("cancel_multiview_text_dry(")
    vision_cancel = probe.index("cancel_multiview_vision_dry(")
    final_allocator_gate = probe.index(
        'check_allocator_unchanged(before, "after reverse dry cancel")'
    )
    rethrow = probe.index("std::rethrow_exception(failure)")
    assert guard < vision_prepare < text_prepare < collect
    assert collect < text_cancel < vision_cancel < final_allocator_gate < rethrow
    assert "SpmPipelineLease::acquire_physical" not in probe
    assert "rpu_launch_" not in probe
    assert "RpuKernelGraph::begin" not in probe
    assert "SPM_ALLOC.init" not in probe
    assert "SPM_ALLOC.reset" not in probe
    assert "SPM_ALLOC.alloc" not in probe


def test_probe_freezes_exact_lingbot2_shape_and_budget_contract():
    source = COORDINATOR.read_text(encoding="utf-8")
    probe = _between(
        source,
        "std::vector<int64_t> rpu_qwen3vl_multiview_spm_dry_probe(",
        "int64_t rpu_qwen3vl_pooler_spm_z1_prepare(",
    )
    for token in (
        "num_patches == 256 && execution_len == 225",
        "shape.chunks.size() == 1",
        "shape.chunks.front().offset == 0",
        "shape.chunks.front().len == execution_len",
        "shape.kv_insert_chunks.size() == 1",
        "shape.kv_insert_chunks.front().len == execution_len",
        "shape.chunk_mode == v3::ChunkMode::SEQUENTIAL",
        "shape.inter_layer_io == v3::InterLayerIO::SPM_RESIDENT",
        "!shape.chunk_outer_within_group",
        "SpmAllocator::SPM_PLANNING_BUDGET",
        "SpmAllocator::ALIGN - 1",
    ):
        assert token in probe
    assert "kProbeSchema = 3" in probe
    assert "vision_stage_descriptor(" in probe
    assert "vision_stage_descriptor.size()" in probe
    assert "text.stage_descriptor.size()" in probe
    assert "text.stage_descriptor.begin()" in probe
    assert "7.5" not in probe


def test_split_dry_modes_are_symmetric_with_physical_and_forward_paths():
    text = TEXT_MODEL.read_text(encoding="utf-8")
    text_prepare = _between(
        text,
        "CausalDecoderModel::prepare_qwen3vl_multiview_text_dry_layout(",
        "void CausalDecoderModel::cancel_qwen3vl_multiview_text_dry_layout()",
    )
    assert "get_chunk_size_override() == 0" in text_prepare
    assert "set_chunk_size_override(0)" not in text_prepare
    assert "qwen3vl_pooler_z1_retained_deepstack_count_ == 0" in text_prepare
    assert "qwen3vl_pooler_z1_retained_deepstack_bindings_empty()" in text_prepare

    text_header = TEXT_HEADER.read_text(encoding="utf-8")
    ordinary_text_forward = _between(
        text_header,
        "    at::Tensor forward(\n",
        "        // ─────────────────────────────────────────────────────────────────────\n"
        "        // Model must be configured",
    )
    assert "!qwen3vl_multiview_text_dry_prepared_" in ordinary_text_forward
    assert text.count("cannot replace a prepared multiview dry") >= 2

    vision = VISION_MODEL.read_text(encoding="utf-8")
    ordinary_vision_forward = _between(
        vision,
        "    at::Tensor forward(\n",
        "    at::Tensor forward_impl(",
    )
    physical_vision_prepare = _between(
        vision,
        "SpmPipelineComponentLayout prepare_pooler_z1_layout(",
        "SpmPipelineComponentLayout prepare_multiview_dry_layout(",
    )
    assert "!multiview_dry_prepared_" in ordinary_vision_forward
    assert "!multiview_dry_prepared_" in physical_vision_prepare
    dry_vision_prepare = _between(
        vision,
        "SpmPipelineComponentLayout prepare_multiview_dry_layout(",
        "void cancel_multiview_dry_layout()",
    )
    assert "max_hw_ == 48" in dry_vision_prepare


def test_framework_dry_manifest_cannot_replace_live_or_physical_authority():
    source = FMB.read_text(encoding="utf-8")
    live_prepare = _between(
        source,
        "SpmPipelineComponentLayout FusedModelBase::prepare_spm_pipeline_component(",
        "FusedModelBase::prepare_spm_pipeline_component_for_cpu_contract(",
    )
    dry_prepare = _between(
        source,
        "FusedModelBase::prepare_spm_pipeline_component_for_cpu_contract(",
        "void FusedModelBase::cancel_spm_pipeline_component_for_cpu_contract()",
    )
    assert "cannot replace a CPU-dry manifest" in live_prepare
    assert "pimpl_->pipeline_lease_epoch_ == 0" in dry_prepare
    assert '"physical component"' in dry_prepare
    assert '"live allocation manifest"' in dry_prepare
    assert dry_prepare.index("temporary_names.reserve(") < dry_prepare.index(
        "retire_pipeline_manifest(*pimpl_)"
    )
    assert dry_prepare.index("retire_pipeline_manifest(*pimpl_)") < dry_prepare.index(
        "pipeline_temporary_names_.swap(temporary_names)"
    )
