from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text()


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _between(text: str, begin: str, end: str) -> str:
    start = text.index(begin)
    return text[start : text.index(end, start)]


def test_fixed_spm_scatter_has_tensor_owned_destination_census() -> None:
    source = _read("src/ops/rpu_memcpy.cpp")
    owned = _between(
        source,
        "void rpu_launch_spm_scatter_ddr_dma(\n        uint32_t spm_addr_unified,\n        const at::Tensor& ddr_tensor",
        "void rpu_launch_spm_scatter_ddr_dma_mutable",
    )
    compact = _compact(owned)

    assert "validate_fp16_dma_owner_slice(" in owned
    assert (
        "graph_dma::stage_census_fixed_burst(ddr_tensor,"
        "GraphOuterFastDmaSide::Destination,byte_begin,byte_count,"
        "static_cast<size_t>(num_cores));"
    ) in compact
    assert (
        "rpu_launch_spm_scatter_ddr_dma(spm_addr_unified,"
        "ddr_tensor.data_ptr<c10::Half>()+dst_offset_elements,"
        "elements_per_core,core_stride_bytes,num_cores);"
    ) in compact


def test_rope_2d_spm_owned_position_is_the_only_ddr_operand() -> None:
    source = _read("src/ops/rpu_rope_2d.cpp")
    compact = _compact(source)

    assert (
        "typedDDR-registercensusforbidsrawrope_2d_spmpositioninput"
        in compact
    )
    assert "KernelId::ROPE_2D_SPM" in source
    assert "GraphDdrRegisterRole::SemanticInput" in source
    assert "GraphDdrRegisterAccess::Read" in source
    assert re.search(r"SemanticInput,[\s\S]*?Read,[\s\S]*?0, 1", source)
    assert "register_writer->write(*kernel);" in source
    assert "position_idx.data_ptr<int16_t>()" in source


def test_mrope_owned_register_abi_covers_weights_and_position() -> None:
    source = _read("src/ops/rpu_mrope.cpp")
    compact = _compact(source)

    assert "typedDDR-registercensusforbidsrawmrope_spmDDRoperands" in compact
    assert re.search(r"ModelWeight,[\s\S]*?Read,[\s\S]*?6, 7", source)
    assert re.search(r"ModelWeight,[\s\S]*?Read,[\s\S]*?8, 9", source)
    assert re.search(r"SemanticInput,[\s\S]*?Read,[\s\S]*?12, 13", source)
    assert "register_writer->write(*kernel);" in source
    assert "cos.data_ptr<c10::Half>()" in source
    assert "sin.data_ptr<c10::Half>()" in source
    assert "position_ids.data_ptr<int32_t>()" in source


def test_qwen3vl_z1_vision_owns_transports_only_on_the_z1_branch() -> None:
    source = _read("src/fused/rpu_qwen3vl_vision_model.cpp")
    preload = _between(source, "void emit_preload_weights()", "void emit_kv_first_body")
    kv_first = _between(source, "void emit_kv_first_body", "void build_layer_subgraph")
    layer = _between(source, "void build_layer_subgraph", "bool merger_active() const")
    merger = _between(
        source, "void emit_one_merger_prefix", "void emit_one_merger("
    )
    rope_spm = _between(
        kv_first,
        "if (rope_spm_enabled_)",
        "} else {\n            consume_manifest_route(\n"
        "                FmbRouteFamily::ROPE, QWEN3VL_Q_ROPE_DDR_SITE",
    )
    q_store = _between(
        kv_first,
        "const at::Tensor& q_slot = q_ddr_slot();",
        "return;",
    )
    q_load = _between(
        layer,
        "const at::Tensor& q_slot = q_ddr_slot();",
        "auto& k_cache",
    )

    for owner in (
        "bn.ln1_w", "bn.ln1_b", "bn.ln2_w", "bn.ln2_b",
        "bn.q_b", "bn.k_b", "bn.v_b", "bn.fc1_b",
        "bn.o_b", "bn.fc2_b",
    ):
        assert f"{owner}.data_ptr<c10::Half>()" in preload
    assert preload.count("if (pipeline_dispatch_active())") >= 3
    assert "bn.ln1_w, /*src_offset_elements=*/0" in preload
    assert "bn.q_b, /*src_offset_elements=*/0" in preload
    assert "freq_cos_, /*src_offset_elements=*/0" in preload
    assert "freq_sin_, /*src_offset_elements=*/0" in preload
    assert "freq_cos_.data_ptr<c10::Half>()" in preload
    assert "freq_sin_.data_ptr<c10::Half>()" in preload

    assert "if (pipeline_dispatch_active())" in rope_spm
    assert rope_spm.count("position_idx_keepalive_,") == 2
    assert rope_spm.count("position_idx_keepalive_.data_ptr<int16_t>()") == 2

    assert "if (pipeline_dispatch_active())" in q_store
    assert "addr(0, \"q\"), q_slot," in q_store
    assert "dst_offset_elements" in q_store
    assert "q_slot.data_ptr<c10::Half>() + chunk.offset * q_row_stride" in q_store

    assert "if (pipeline_dispatch_active())" in q_load
    assert "q_slot, /*src_offset_elements=*/chunk.offset * q_row_stride" in q_load
    assert "q_slot.data_ptr<c10::Half>() + chunk.offset * q_row_stride" in q_load

    for owner in ("ln_q_w", "ln_q_b", "m0_b", "m2_b"):
        assert f"{owner}.data_ptr<c10::Half>()" in merger
    assert merger.count("if (pipeline_dispatch_active())") == 2
    assert "ln_q_w, /*src_offset_elements=*/0" in merger
    assert "m0_b, /*src_offset_elements=*/0" in merger


def test_qwen3vl_z1_profiles_preflight_tensor_owners_before_capture() -> None:
    vision_source = _read("src/fused/rpu_qwen3vl_vision_model.cpp")
    text_source = _read("src/fused/rpu_qwen3_model.cpp")
    vision = _between(
        vision_source,
        "void validate_pooler_z1_profile() const",
        "void validate_pooler_z1_contract() const",
    )
    text = _between(
        text_source,
        "void CausalDecoderModel::validate_qwen3vl_pooler_z1_model_profile() const",
        "CausalDecoderModel::prepare_qwen3vl_pooler_z1_layout",
    )
    vision_prepare = _between(
        vision_source,
        "SpmPipelineComponentLayout prepare_pooler_z1_layout(",
        "void prime_pooler_z1_inputs(",
    )
    text_prepare = _between(
        text_source,
        "CausalDecoderModel::prepare_qwen3vl_pooler_z1_layout(",
        "CausalDecoderModel::qwen3vl_pooler_z1_destination_spec(",
    )

    for profile in (vision, text):
        assert "tensor.device().type() == at::kPrivateUse1" in profile
        assert "tensor.layout() == c10::Layout::Strided" in profile
        assert "tensor.is_contiguous()" in profile
        assert "tensor.numel() >= min_numel" in profile
        assert "require_fp16_matrix_min" in profile
        assert "require_fp16_vector" in profile

    assert "position_idx_keepalive_.scalar_type() == at::kShort" in vision
    assert "position_idx_keepalive_.size(1) == 2" in vision
    assert "freq_cos_.size(1) == rope_cols" in vision
    assert "position_ids_keepalive_.scalar_type() == at::kInt" in text
    assert "position_ids_keepalive_.size(1) == 3" in text
    assert "cos_.size(1) == rope_cols" in text
    assert vision_prepare.index("!RpuKernelGraph::has_active()") < (
        vision_prepare.index("validate_pooler_z1_profile();")
    )
    assert text_prepare.index("!RpuKernelGraph::has_active()") < (
        text_prepare.index(
            "validate_qwen3vl_pooler_z1_model_profile(\n"
            "        retained_deepstack_count, real_len);"
        )
    )


def test_owned_overloads_are_declared_without_removing_raw_abi() -> None:
    header = _read("src/core/rpu_kernel_decls.h")

    assert "int16_t *position_idx_ddr_ptr" in header
    assert "const at::Tensor& position_idx" in header
    assert "int32_t *position_ids_ddr_ptr" in header
    assert "const at::Tensor& position_ids" in header
    assert "const at::Tensor& ddr_tensor" in _between(
        header,
        "void rpu_launch_spm_scatter_ddr_dma(\n    uint32_t          spm_addr_unified",
        "// Mutable counterpart of rpu_launch_spm_scatter_ddr_dma",
    )
