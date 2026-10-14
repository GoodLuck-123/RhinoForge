"""Board-free contracts for Qwen3.5/Gemma raw-SPM attention adoption."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
Q35_TEXT = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text(
    encoding="utf-8"
)
Q35_TEXT_HEADER = (ROOT / "src/fused/rpu_qwen3_5_model.h").read_text(
    encoding="utf-8"
)
Q35_VISION = (ROOT / "src/fused/rpu_qwen3_5_vision_model.cpp").read_text(
    encoding="utf-8"
)
GEMMA = (ROOT / "src/fused/rpu_gemma_model.cpp").read_text(encoding="utf-8")
HELPERS = (ROOT / "src/core/rpu_helpers.h").read_text(encoding="utf-8")


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def test_shared_raw_spm_pair_predicate_matches_both_launcher_abis():
    predicate = _between(
        HELPERS,
        "inline bool sdpa_by_mha_spm_is_valid(",
        "// Keeper-side runtime validator.",
    )
    for contract in (
        "attn_mask_type != 0",
        "attn_mask_type != 1",
        "attn_mask_type != 4",
        "seq_k > kU16Max",
        "num_q_heads % num_cores != 0",
        "num_kv_heads % num_cores != 0",
        "batch > 1 && seq_k % kV16 != 0",
        "kv_heads_per_core > 32",
        "kMaxTileElementsV16",
        "sync_product > 8 && sync_product % 8 != 0",
    ):
        assert contract in predicate


def test_qwen35_text_enumerates_exact_raw_and_reasoned_ddr_routes():
    eligibility = _between(
        Q35_TEXT,
        "bool Qwen3_5Model::subclass_spm_kv_by_mha_eligible(",
        "std::vector<FmbPhysicalExecutionManifest>\n"
        "Qwen3_5Model::physical_manifest_domain_for_candidate(",
    )
    for contract in (
        "z2_bound_ || action_mode_",
        "position != 0",
        "layout.batch_size != 1",
        "plan.chunk_mode != ChunkMode::SEQUENTIAL",
        "plan.input.chunks.size() != 1",
        "plan.qkv.chunks.size() != 1",
        "plan.compute.chunks.size() != 1",
        "layout.max_kv_seq_len != seq",
        "sdpa_by_mha_spm_is_valid(",
    ):
        assert contract in eligibility

    manifest = _between(
        Q35_TEXT,
        "FmbPhysicalExecutionManifest Qwen3_5Model::physical_manifest_for_candidate(",
        "FmbPhysicalManifestForwardCapability\n"
        "Qwen3_5Model::physical_manifest_forward_capability(",
    )
    assert "5148991311835582171" in Q35_TEXT_HEADER
    assert "QWEN35_TEXT_ATTN_PREFIX_HISTORY_DDR_REQUIRED" in manifest
    assert "QWEN35_TEXT_ATTN_MULTI_CHUNK_DDR_REQUIRED" in manifest
    assert "QWEN35_TEXT_ATTN_ACTION_PREFIX_DDR_REQUIRED" in manifest
    assert "QWEN35_TEXT_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED" in manifest
    assert "raw_spm ? kFullAttentionRawSpmSite : kFullAttentionSite" in manifest

    forward = _between(
        Q35_TEXT,
        "void Qwen3_5Model::build_full_attention(",
        "void Qwen3_5Model::build_gdn(",
    )
    assert "ctx().attention_policy" in forward
    assert "ctx().find_physical_route(" in forward
    assert "ctx().consume_physical_route(" in forward
    assert forward.index("rpu_launch_v_transpose_spm(") < forward.index(
        "rpu_launch_sdpa_by_mha_spm("
    )
    assert "physical_manifest_domain_for_candidate(" not in forward


def test_qwen35_vision_keeps_temporal_ddr_and_adopts_exact_spatial_raw():
    assert "7600841761525222001" in Q35_VISION
    assert "809625674292115789" in Q35_VISION
    manifest = _between(
        Q35_VISION,
        "Qwen3_5VisionModel::physical_manifest_for_candidate(",
        "FmbPhysicalManifestForwardCapability\n"
        "Qwen3_5VisionModel::physical_manifest_forward_capability(",
    )
    assert "QWEN35_VISION_ATTN_TEMPORAL_PACKED_KV_DDR_REQUIRED" in manifest
    assert "QWEN35_VISION_TEMPORAL_ATTN_SITE" in manifest
    assert "QWEN35_VISION_SPATIAL_RAW_SPM_ATTN_SITE" in manifest
    assert "QWEN35_VISION_ATTN_MULTI_CHUNK_DDR_REQUIRED" in manifest
    assert "QWEN35_VISION_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED" in manifest
    assert "/*batch=*/3, temporal_patches_per_frame_" in manifest

    spatial = _between(
        Q35_VISION,
        "void Qwen3_5VisionModel::emit_spatial_group(",
        "void Qwen3_5VisionModel::build_layer_subgraph(",
    )
    assert "ctx().attention_policy" in spatial
    assert "QWEN35_VISION_SPATIAL_RAW_SPM_ATTN_SITE" in spatial
    assert spatial.index("rpu_launch_v_transpose_spm(") < spatial.index(
        "rpu_launch_sdpa_by_mha_spm("
    )

    resolver = _between(
        Q35_VISION,
        "std::vector<std::vector<int64_t>> Qwen3_5VisionModel::resolve_prefill_domain(",
        "ChunkPlan Qwen3_5VisionModel::plan_kv_first_chunks(",
    )
    assert "resolve_prefill_stage_domain_for_shape(" in resolver
    assert "for (const auto& candidate : physical_candidates)" in resolver
    assert "spm_pipeline_component_layout_fits_for_cpu_contract(" in resolver
    assert "finish_kvinsert_cost_domain_oracle(returned_candidates)" in resolver
    assert "detail::prefer_balanced_chunk" in resolver  # compatibility-only projection


def test_gemma_adopts_only_masked_p0_single_chunk_raw_route():
    assert "4569660003355245327" in GEMMA
    eligibility = _between(
        GEMMA,
        "bool subclass_spm_kv_by_mha_eligible(",
        "std::vector<FmbPhysicalExecutionManifest>\n"
        "    physical_manifest_domain_for_candidate(",
    )
    for contract in (
        "position != 0",
        "!layout.use_attn_mask",
        "layout.is_causal",
        "plan.chunk_mode != ChunkMode::KV_FIRST",
        "plan.input.chunks.size() != 1",
        "plan.qkv.chunks.size() != 1",
        "plan.compute.chunks.size() != 1",
        "layout.max_kv_seq_len != seq",
        "/*MASK_2D=*/4",
    ):
        assert contract in eligibility

    forward = _between(
        GEMMA,
        "void build_layer_subgraph(int layer_idx, const ChunkInfo& chunk) override",
        "};\n\n}  // namespace v3",
    )
    assert "GEMMA_ATTN_PREFIX_HISTORY_DDR_REQUIRED" in GEMMA
    assert "GEMMA_ATTN_MULTI_CHUNK_DDR_REQUIRED" in GEMMA
    assert "GEMMA_ATTN_CAPABILITY_FALLBACK_DDR_REQUIRED" in GEMMA
    assert "ctx().attention_policy" in forward
    assert "ctx().find_physical_route(" in forward
    assert "ctx().consume_physical_route(" in forward
    assert forward.index("rpu_launch_v_transpose_spm(") < forward.index(
        "rpu_launch_sdpa_by_mha_spm("
    )
    assert "physical_manifest_domain_for_candidate(" not in forward
