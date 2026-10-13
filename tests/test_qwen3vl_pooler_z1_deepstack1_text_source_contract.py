"""Board-free source contract for Qwen3-VL Text retained DeepStack."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "src/fused/rpu_qwen3_model.h"
SOURCE = ROOT / "src/fused/rpu_qwen3_model.cpp"
COORDINATOR_HEADER = ROOT / "src/fused/rpu_qwen3vl_pooler_spm_z1.h"


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def test_retained_prepare_accepts_only_zero_one_or_three_and_omits_dense_scratch():
    header = HEADER.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    profile = _between(
        source,
        "void CausalDecoderModel::validate_qwen3vl_pooler_z1_model_profile(",
        "CausalDecoderModel::prepare_qwen3vl_pooler_z1_layout(",
    )
    prepare = _between(
        source,
        "CausalDecoderModel::prepare_qwen3vl_pooler_z1_layout(",
        "SpmDense2DSpec CausalDecoderModel::qwen3vl_pooler_z1_destination_spec(",
    )

    assert "int64_t retained_deepstack_count = 0" in header
    assert "retained_deepstack_count == 0" in profile
    assert "retained_deepstack_count == 1" in profile
    assert "retained_deepstack_count == 3" in profile
    assert "deepstack_lang_layers_.size()) ==" in profile
    assert "QWEN3VL_POOLER_Z1_DEEPSTACK_LAYERS" in profile
    assert "QWEN3VL_POOLER_Z1_DEEPSTACK_LAYERS = {0, 1, 2}" in header
    assert "QWEN3VL_POOLER_Z1_DS1_LAYER = 0" in header
    assert prepare.index(
        "qwen3vl_pooler_z1_retained_deepstack_count_ = retained_deepstack_count"
    ) < prepare.index("compose_fmb_default_three_stage_chunk_plan(")
    assert "ChunkMode::SEQUENTIAL" in prepare
    assert "prepare_spm_pipeline_component(" in prepare
    assert "QWEN3VL_POOLER_Z1_CANARY_TEMPORARY_BYTES" in prepare

    declarations = _between(
        header,
        "// ★ R-Phase 2 (Qwen3-VL): DeepStack scratch buffer.",
        "// Norm-weight preloads",
    )
    assert "qwen3vl_pooler_z1_retained_deepstack_count_ == 0" in declarations
    assert 'decls.push_back({"deepstack_scratch"' in declarations


def test_groot_profile_is_exactly_16_layers_s82_ds3_fp16_with_same_temp():
    header = HEADER.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    profile = _between(
        source,
        "void CausalDecoderModel::validate_qwen3vl_pooler_z1_model_profile(",
        "CausalDecoderModel::prepare_qwen3vl_pooler_z1_layout(",
    )
    prepare = _between(
        source,
        "CausalDecoderModel::prepare_qwen3vl_pooler_z1_layout(",
        "SpmDense2DSpec CausalDecoderModel::qwen3vl_pooler_z1_destination_spec(",
    )
    contract = _between(
        source,
        "void CausalDecoderModel::validate_qwen3vl_pooler_z1_contract() const",
        "void CausalDecoderModel::prime_qwen3vl_pooler_z1_inputs(",
    )

    for token in (
        "QWEN3VL_POOLER_Z1_CANARY_LAYERS = 28",
        "QWEN3VL_POOLER_Z1_GROOT_LAYERS = 16",
        "QWEN3VL_POOLER_Z1_CANARY_REAL_LEN = 72",
        "QWEN3VL_POOLER_Z1_GROOT_REAL_LEN = 82",
        "QWEN3VL_POOLER_Z1_DEEPSTACK_LAYERS = {0, 1, 2}",
        "QWEN3VL_POOLER_Z1_CANARY_TEMPORARY_BYTES =",
        "1966080",
    ):
        assert token in header
    for token in (
        "num_layers() == QWEN3VL_POOLER_Z1_CANARY_LAYERS",
        "real_len == QWEN3VL_POOLER_Z1_CANARY_REAL_LEN",
        "num_layers() == QWEN3VL_POOLER_Z1_GROOT_LAYERS",
        "real_len == QWEN3VL_POOLER_Z1_GROOT_REAL_LEN",
        "retained_deepstack_count == 3",
        "qwen3vl_profile || groot_profile",
        "QWEN3VL_POOLER_Z1_DEEPSTACK_LAYERS",
        "!nvfp4_ && !has_qkv_bias_ && has_qk_norm_ && !adarms_",
        "tensor.scalar_type() == at::kHalf",
        "!weights.q_ws.defined()",
        "layer < num_layers()",
    ):
        assert token in profile
    assert "QWEN3VL_POOLER_Z1_GROOT_REAL_LEN" in prepare
    assert "retained_deepstack_count, real_len" in prepare
    assert "QWEN3VL_POOLER_Z1_CANARY_TEMPORARY_BYTES" in prepare
    assert "QWEN3VL_POOLER_Z1_GROOT_REAL_LEN" in contract
    assert "validate_qwen3vl_pooler_z1_model_profile();" in contract


def test_retained_ports_use_exact_spec_and_prefix_binding():
    header = HEADER.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    spec = _between(
        source,
        "SpmDense2DSpec CausalDecoderModel::qwen3vl_pooler_z1_deepstack_spec(",
        "SpmDense2DSpec CausalDecoderModel::qwen3vl_pooler_z1_deepstack1_spec()",
    )
    bind = _between(
        source,
        "void CausalDecoderModel::bind_qwen3vl_pooler_z1_deepstack(",
        "void CausalDecoderModel::bind_qwen3vl_pooler_z1_deepstack1(",
    )
    complete = _between(
        source,
        "qwen3vl_pooler_z1_retained_deepstack_bindings_complete() const",
        "CausalDecoderModel::prepare_qwen3vl_pooler_z1_layout(",
    )

    assert "QWEN3VL_POOLER_Z1_DS1_ROWS = 64" in header
    assert "ordinal >= 0" in spec
    assert "ordinal < qwen3vl_pooler_z1_retained_deepstack_count_" in spec
    assert "spec.rows = QWEN3VL_POOLER_Z1_DS1_ROWS" in spec
    assert "spec.cols = QWEN3VL_POOLER_Z1_CANARY_HIDDEN" in spec
    assert "ordinal == qwen3vl_pooler_z1_deepstack_bound_count_" in bind
    assert "deepstack.spec() == expected" in bind
    assert "deepstack.resolve_physical_addr(/*core=*/0, lease)" in bind
    assert "retained DeepStack ports alias" in bind
    assert "qwen3vl_pooler_z1_deepstack_addrs_" in bind
    assert "++qwen3vl_pooler_z1_deepstack_bound_count_" in bind
    assert "count != 0 && count != 1 && count != 3" in complete
    assert "requested != (addr != 0)" in complete
    assert "addr == qwen3vl_pooler_z1_deepstack_addrs_[prior]" in complete


def test_retained_lifecycle_and_outerfast_are_fail_closed():
    header = HEADER.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    clear = _between(
        source,
        "void CausalDecoderModel::clear_qwen3vl_pooler_z1_layout(",
        "at::Tensor CausalDecoderModel::forward_qwen3vl_pooler_z1(",
    )
    unprepare = _between(
        source,
        "void CausalDecoderModel::unprepare_qwen3vl_pooler_z1_layout()",
        "void CausalDecoderModel::adopt_qwen3vl_pooler_z1_layout(",
    )
    destroy = _between(
        source,
        "void CausalDecoderModel::check_qwen3vl_pooler_z1_destroy_allowed()",
        "}  // namespace v3",
    )

    for marker in (
        "std::array<uint32_t, 3> qwen3vl_pooler_z1_deepstack_addrs_{}",
        "qwen3vl_pooler_z1_retained_deepstack_count_ = 0",
        "qwen3vl_pooler_z1_deepstack_bound_count_ = 0",
    ):
        assert marker in header
    assert "qwen3vl_pooler_z1_deepstack_addrs_.fill(0)" in clear
    assert "qwen3vl_pooler_z1_deepstack_bound_count_ = 0" in clear
    assert "qwen3vl_pooler_z1_deepstack_addrs_.fill(0)" in unprepare
    assert "qwen3vl_pooler_z1_retained_deepstack_count_ = 0" in unprepare
    assert "qwen3vl_pooler_z1_deepstack_bound_count_ = 0" in unprepare
    assert "qwen3vl_pooler_z1_retained_deepstack_count_ == 0" in destroy
    assert "qwen3vl_pooler_z1_retained_deepstack_bindings_empty()" in destroy
    assert source.count(
        "qwen3vl_pooler_z1_retained_deepstack_bindings_complete()"
    ) >= 5
    assert source.count(
        "qwen3vl_pooler_z1_retained_deepstack_bindings_empty()"
    ) >= 5


def test_ds1_and_ds3_dispatch_reject_dense_ddr_and_add_only_visual_rows():
    header = HEADER.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    forward_entry = _between(
        header,
        "const bool qwen3vl_pooler_z1_retained_deepstack_dispatch =",
        "const bool decode_fused_lm_head",
    )
    add = _between(
        source,
        "void CausalDecoderModel::emit_qwen3vl_pooler_z1_retained_deepstack_add(",
        "uint32_t CausalDecoderModel::qwen3vl_pooler_z1_layer_input_residual_addr(",
    )

    assert "rejects dense DDR embeds" in forward_entry
    assert "deepstack_dense_refs_.clear()" in forward_entry
    assert "deepstack_dense_src_base_.fill(0)" in forward_entry
    assert header.count("emit_qwen3vl_pooler_z1_retained_deepstack_add(") == 3
    assert "QWEN3VL_POOLER_Z1_DS1_ROW_BEGIN = 4" in header
    assert "QWEN3VL_POOLER_Z1_DEEPSTACK_LAYERS" in add
    assert "candidate < qwen3vl_pooler_z1_retained_deepstack_count_" in add
    assert "chunk.idx == 0 && chunk.offset == 0" in add
    assert "QWEN3VL_POOLER_Z1_DS1_ROWS *" in add
    assert "qwen3vl_pooler_z1_deepstack_addrs_" in add
    assert "rpu_launch_eltwise_binary_spm_kernel(" in add
    assert "rpu_launch_ddr_broadcast_spm_dma_mutable" not in add
    assert 'addr(0, "deepstack_scratch")' not in add

    # Both layer bodies retain the ordinary mutable-DMA fallback. Retained DS1
    # and DS3 select the direct SPM-port branch before it.
    assert header.count("qwen3vl_pooler_z1_retained_deepstack_count_ > 0") >= 3
    assert header.count("&deepstack_dense_src_base_[idx_in_list]") == 2


def test_generic_facades_keep_ds1_compatibility_and_add_ds3_prepare():
    header = HEADER.read_text(encoding="utf-8")
    coordinator_header = COORDINATOR_HEADER.read_text(encoding="utf-8")
    source = SOURCE.read_text(encoding="utf-8")
    wrappers = _between(
        source,
        "namespace v3::qwen3vl_pooler_z1_internal {",
        "}  // namespace v3::qwen3vl_pooler_z1_internal",
    )

    for marker in (
        "prepare_text_deepstack1(",
        "prepare_text_deepstack3(",
        "text_deepstack_spec(",
        "text_deepstack1_spec(",
        "bind_text_deepstack(",
        "bind_text_deepstack1(",
    ):
        assert marker in header
        assert marker in wrappers
    # The parent coordinator owns its header update; the original DS1 wrappers
    # remain available throughout that transition.
    for marker in ("prepare_text_deepstack1(", "text_deepstack1_spec(", "bind_text_deepstack1("):
        assert marker in coordinator_header

    assert "/*retained_deepstack_count=*/0" in wrappers
    assert "/*retained_deepstack_count=*/1" in wrappers
    assert "/*retained_deepstack_count=*/3" in wrappers
    assert "return text_deepstack_spec(handle, /*ordinal=*/0)" in wrappers
    assert "handle, lease, deepstack1, /*ordinal=*/0" in wrappers

    wall_declarations = _between(
        header,
        "namespace v3::wall_oss_z1_internal {",
        "}  // namespace v3::wall_oss_z1_internal",
    )
    assert "deepstack1" not in wall_declarations
