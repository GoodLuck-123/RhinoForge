"""Board-free source gate for the Qwen3-VL pooler Z1 Text outer-fast facade."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src/fused/rpu_qwen3_model.h"
SOURCE = ROOT / "src/fused/rpu_qwen3_model.cpp"
COORDINATOR_HEADER = ROOT / "src/fused/rpu_qwen3vl_pooler_spm_z1.h"


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def test_text_outer_fast_stages_only_a_current_bound_z1_component():
    source = SOURCE.read_text(encoding="utf-8")
    stage = _between(
        source,
        "void CausalDecoderModel::stage_qwen3vl_pooler_z1_outer_fast_component(",
        "at::Tensor CausalDecoderModel::qwen3vl_pooler_z1_outer_fast_output()",
    )

    assert "validate_qwen3vl_pooler_z1_contract();" in stage
    for marker in (
        "qwen3vl_pooler_z1_active_",
        "qwen3vl_pooler_z1_bound_",
        "qwen3vl_pooler_z1_inputs_primed_",
        "qwen3vl_pooler_z1_position_ids_addr_ != 0",
        "ka_last_position_ == 0",
        "ka_last_seq_ ==",
        "ka_last_pos_src_ != nullptr",
        "qwen3vl_pooler_z1_partial_mrope_ ==",
        "partial_mrope_active_",
        "cos_il_keepalive_.data_ptr<c10::Half>()",
        "sin_il_keepalive_.data_ptr<c10::Half>()",
    ):
        assert marker in stage
    assert "stage_spm_outer_fast_component(guard, k_caches, v_caches);" in stage
    assert "bind_spm_outer_fast_input" not in stage


def test_outer_fast_output_is_the_sealed_registry_tensor():
    source = SOURCE.read_text(encoding="utf-8")
    forward = _between(
        source,
        "at::Tensor CausalDecoderModel::forward_qwen3vl_pooler_z1(",
        "void CausalDecoderModel::stage_qwen3vl_pooler_z1_outer_fast_component(",
    )
    output = _between(
        source,
        "at::Tensor CausalDecoderModel::qwen3vl_pooler_z1_outer_fast_output()",
        "uint32_t CausalDecoderModel::qwen3vl_pooler_z1_layer_input_residual_addr(",
    )

    assert "const at::Tensor& registry_output = output_tensor();" in forward
    assert "qwen3vl_pooler_z1_output_owner_ = result;" in forward
    assert "RpuKernelGraph::has_active()" in output
    assert "qwen3vl_pooler_z1_active_" in output
    assert "qwen3vl_pooler_z1_bound_" in output
    assert "const at::Tensor& current = output_tensor();" in output
    for marker in (
        "current.size(0) == 1",
        "QWEN3VL_POOLER_Z1_CANARY_EXECUTION_LEN",
        "QWEN3VL_POOLER_Z1_CANARY_HIDDEN",
        "current.scalar_type() == at::kHalf",
        "current.device().type() == at::kPrivateUse1",
        "current.is_contiguous()",
        "qwen3vl_pooler_z1_output_owner_.is_same(current)",
        "unsafeGetStorageImpl()",
        "current.data_ptr()",
    ):
        assert marker in output
    assert "return current;" in output


def test_internal_facade_is_declared_in_both_component_headers():
    source = SOURCE.read_text(encoding="utf-8")
    model_header = HEADER.read_text(encoding="utf-8")
    coordinator_header = COORDINATOR_HEADER.read_text(encoding="utf-8")
    wrappers = _between(
        source,
        "namespace v3::qwen3vl_pooler_z1_internal {",
        "}  // namespace v3::qwen3vl_pooler_z1_internal",
    )

    for marker in (
        "stage_text_outer_fast_component(",
        "text_outer_fast_output(",
    ):
        assert marker in wrappers
        assert marker in model_header
        assert marker in coordinator_header
