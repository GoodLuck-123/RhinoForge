"""Board-free closure contracts for the remaining SPM-attention sites."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start:source.index(end, start)]


def test_qwen25_generic_sites_share_one_exact_ddr_raw_domain() -> None:
    source = _read("src/fused/rpu_qwen25vl_vision_model.cpp")
    for site_id in (
        5587652333716339060,  # MISC raw
        2545818117373449458,  # MISC DDR
        5718369161792163142,  # per-window raw
        913880989367815128,   # per-window DDR
        8716771559110690244,  # standard raw
        4186875649763190810,  # standard DDR
    ):
        assert str(site_id) in source
    for contract in (
        "physical_manifest_domain_for_candidate(",
        "ddr_layout.attention_policy = AttentionExecutionPolicy::DDR_KV;",
        "raw_layout.attention_policy =",
        "generic_attention_site_raw_valid(",
        "sdpa_by_mha_spm_is_valid(",
        "QWEN25_ATTN_DDR_CAPABILITY_FALLBACK",
        "QWEN25_ATTN_DDR_RAW_KERNEL_INCOMPATIBLE",
        "QWEN25_KV_FLAG_DDR_MIRROR",
    ):
        assert contract in source

    buffers = _between(
        source, "std::vector<BufferDecl> declare_buffers(",
        "void emit_preload_weights()",
    )
    assert "raw_generic_layout" in buffers
    assert "Align(cs, int64_t{16}) * nkv_per_core * hd * DWIDTH" in buffers

    forward = _between(
        source, "void build_layer_subgraph(", "void emit_merger_chunk("
    )
    # New exact schedules may add consumers; admission still belongs to the
    # COMPLETE route rather than ad hoc feasibility checks inside emission.
    assert "consume_generic_attention_route(" in forward
    assert "rpu_launch_v_transpose_spm(" in forward
    assert "rpu_launch_sdpa_by_mha_spm(" in forward
    assert "generic_attention_raw_valid(" not in forward
    assert "physical_attention_policy_for_site(" not in forward


def test_internvla_and_navdp_keep_only_self_raw_and_cross_ddr() -> None:
    nextdit = _read("src/fused/rpu_internvla_nextdit_model.cpp")
    assert "ddr_layout.attention_policy = AttentionExecutionPolicy::DDR_KV;" in nextdit
    assert "kNextDitAttentionDdrPrefixHistory" in nextdit
    for site in (
        "kNextDitSelfAttentionPackedRawSite",
        "kNextDitSelfAttentionPackedDdrSite",
        "kNextDitSelfAttentionSingleRawSite",
        "kNextDitSelfAttentionSingleDdrSite",
    ):
        assert site in nextdit
    cross = _between(nextdit, "// Cross attention:", "// Per-head tanh gate")
    assert "AttentionExecutionPolicy::DDR_KV" in cross
    assert "rpu_launch_sdpa_by_mha_spm(" not in cross

    navdp = _read("src/fused/rpu_navdp_model.cpp")
    assert "kNavdpCrossAttentionMemoryPrefixDdrRequired" in navdp
    assert "physical_manifest_domain_for_candidate(" in navdp
    assert "ddr_layout.attention_policy = AttentionExecutionPolicy::DDR_KV;" in navdp
    eligibility = _between(
        navdp, "bool subclass_spm_kv_by_mha_eligible(",
        "std::vector<BufferDecl> declare_buffers",
    )
    assert "sdpa_by_mha_spm_is_valid(" in eligibility
    cross = _between(navdp, "// ---- sub-block 2:", "// ---- sub-block 3:")
    assert "kNavdpCrossAttentionMemoryPrefixDdrRequired" in cross
    assert "rpu_launch_sdpa_by_mha_spm(" not in cross


def test_prefix_history_negative_site_census_is_explicit() -> None:
    native_reasons = {
        "src/fused/rpu_halo_action_expert_model.cpp":
            "HALO_ACTION_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED",
        "src/fused/rpu_halo_image_flow_model.cpp":
            "HALO_IMAGE_FLOW_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED",
        "src/fused/rpu_hyvla_expert_model.cpp":
            "HYVLA_EXPERT_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED",
        "src/fused/rpu_hyvla_vlm_model.cpp":
            "HYVLA_VLM_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED",
        "src/fused/rpu_lingbot_v2_moe_model.cpp":
            "CAUSAL_DECODER_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED",
        "src/fused/rpu_rhino_vla_model.cpp":
            "RHINO_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED",
    }
    for path, reason in native_reasons.items():
        source = _read(path)
        assert "AttentionExecutionPolicy::DDR_KV" in source
        assert reason in source

    base = _read("src/fused/rpu_qwen3_model.h")
    assert "CAUSAL_DECODER_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED" in base
    for path in (
        "src/fused/rpu_lingbot_denoise_model.h",
        "src/fused/rpu_wall_oss_action_step_model.h",
    ):
        assert "public CausalDecoderModel" in _read(path)

    adapter_reasons = {
        "python/rpu_backend/adapters/pi05/adarms.py":
            "PREFIX_HISTORY_DDR_REQUIRED_NO_RAW_RESIDENCY_ABI",
        "python/rpu_backend/adapters/pi05/runtime.py":
            "PREFIX_HISTORY_DDR_REQUIRED_NO_RAW_RESIDENCY_ABI",
        "python/rpu_backend/adapters/qwenpi05/fused.py":
            "PREFIX_HISTORY_DDR_REQUIRED_NO_RAW_RESIDENCY_ABI",
        "python/rpu_backend/adapters/wall_oss/action.py":
            "PREFIX_HISTORY_DDR_REQUIRED_SHARED_RPUCACHE_ABI",
    }
    for path, reason in adapter_reasons.items():
        source = _read(path)
        assert '"attention_policy": "DDR_REQUIRED"' in source
        assert reason in source
