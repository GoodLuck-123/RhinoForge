"""Host-only source contract for shared KV-insert segment authority."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _function(source: str, name: str) -> str:
    definition = re.search(
        re.escape(name) + r"\([^;{}]*\)\s*(?:const\s*)?\{", source
    )
    assert definition is not None, name
    brace = definition.end() - 1
    depth = 0
    for index in range(brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1 : index]
    raise AssertionError(f"unclosed function {name}")


def test_plan_is_fixed_explicit_and_fmb_encodable():
    header = _read("src/ops/rpu_kvinsert_segment_plan.h")
    implementation = _read("src/ops/rpu_kvinsert_segment_plan.cpp")
    for route in (
        "V2",
        "ALIGNED_V16",
        "PAD16_V16",
        "HYBRID2",
        "HYBRID3",
    ):
        assert route in header
        assert f"KvInsertRoute::{route}" in implementation
    assert "class KvInsertSegmentPlan final" in header
    assert "std::array<KvInsertSegment, 3> segments_" in header
    assert "uint32_t certified_capabilities_" in header
    assert "KvInsertSegmentPlan(KvInsertRoute route" in header
    assert header.index("KvInsertSegmentPlan(KvInsertRoute route") > header.index("private:")
    assert "kKvInsertRouteTopologyWords = 17" in header
    assert "kKvInsertRouteArgumentWords = 22" in header
    assert "std::array<int64_t, kKvInsertRouteArgumentWords>" in header
    assert "std::vector" not in implementation
    assert "std::getenv" not in implementation
    validator = _function(implementation, "rpu_validate_kvinsert_segment_plan")
    assert "plan.certified_capabilities()" in validator
    assert "uses_v16 && num_cores != 8" in validator
    assert "KV_INSERT_CAP_NON8_TP_V16" in validator


def test_auto_enumerates_and_ranks_the_full_feasible_route_domain():
    header = _read("src/ops/rpu_kvinsert_segment_plan.h")
    implementation = _read("src/ops/rpu_kvinsert_segment_plan.cpp")
    auto = _function(implementation, "rpu_resolve_kvinsert_segment_plan_auto")
    for token in (
        "feasible_routes",
        "KvInsertRoute::V2",
        "KvInsertRoute::ALIGNED_V16",
        "KvInsertRoute::PAD16_V16",
        "KvInsertRoute::HYBRID2",
        "KvInsertRoute::HYBRID3",
        "route_cost(candidate, certificate)",
        "winner_score",
        "KvInsertSelectionState::CAPABILITY_RANKED",
        "KvInsertSelectionState::HARDWARE_COST_CERTIFIED",
    ):
        assert token in auto
    assert "return rpu_resolve_kvinsert_segment_plan(" not in auto
    assert "KvInsertCostCertificate" in header
    assert "feasible_route_mask" in header
    assert "cost_score" in header
    assert "dependency_identity" in header
    assert "cost_certificate_identity" in header


def test_causal_owner_binds_one_catalog_for_auto_and_descriptor_restore():
    base = _read("src/core/fused_model_base.cpp")
    state = _read("src/core/fused_model_base_impl.h")
    causal = _read("src/fused/rpu_qwen3_model.h")
    dispatch = _read("src/core/rpu_dispatch_registrations.inc")
    bind = _function(base, "FusedModelBase::bind_kvinsert_costs")
    assert "get_last_resolved_chunk_size() == 0" in bind
    assert "!pimpl_->kvinsert_cost_catalog_.bound()" in bind
    assert bind.index("KvInsertCostCatalog::from_arguments(") < bind.index(
        "pimpl_->kvinsert_cost_catalog_ ="
    )
    assert "KvInsertCostCatalog kvinsert_cost_catalog_;" in state
    for function in ("resolve_kvinsert_plan_auto", "restore_kvinsert_plan"):
        body = _function(base, f"FusedModelBase::{function}")
        assert "pimpl_->kvinsert_cost_catalog_" in body
        assert "catalog.scope(site_id, static_cast<int64_t>(graph_lifecycle))" in body
        assert "catalog.certificates()" in body
        assert function + "(" in causal
    assert "rpu_resolve_kvinsert_segment_plan_auto(" not in causal
    assert "rpu_kvinsert_segment_plan_from_route_arguments(" not in causal
    restore = _function(base, "FusedModelBase::restore_kvinsert_plan")
    assert "ctx().has_complete_physical_manifest()" in restore
    assert "ctx().physical_manifest().graph_lifecycle" in restore
    assert "site_id, manifest.graph_lifecycle" in causal
    assert "causal_decoder_bind_kvinsert_costs(int handle, int[] identity," in dispatch
    assert "causal_decoder_kvinsert_cost_catalog_sha256(int handle) -> str" in dispatch




def test_all_competing_fmb_leaves_bind_their_own_native_catalog():
    # Two CausalDecoder subclasses inherit AUTO/restore but have their own
    # registries: binding the ordinary decoder cannot configure these leaves.
    owners = (
        ("adarms", "adarms", "AdaRMSRegistry"),
        ("dinov3_vision", "dinov3_vision", "DINOv3VisionRegistry"),
        ("gr00t_dit", "gr00t_dit", "Gr00tDiTRegistry"),
        ("gr00t_vl_encoder", "gr00t_vl_encoder", "Gr00tVlEncoderRegistry"),
        ("halo_action_expert", "halo_action_expert", "HaloActionExpertRegistry"),
        ("halo_image_flow", "halo_image_flow", "HaloImageFlowRegistry"),
        ("hyvit2_vision", "hyvit2", "HYViT2Registry"),
        ("hyvla_expert", "hyvla_expert", "HyVlaExpertRegistry"),
        ("hyvla_vlm", "hyvla_vlm", "HyVlaVlmRegistry"),
        ("internvla_nextdit", "internvla_nextdit", "InternVLANextDiTRegistry"),
        ("lingbot_v2_moe", "lingbot_v2_moe", "LingbotV2MoeRegistry"),
        ("navdp", "navdp", "NavdpRegistry"),
        ("pi05_denoise_step", "pi05_denoise_step", "Pi05DenoiseStepRegistry"),
        ("qwen25vl_vision", "qwen25vl_vision", "Qwen25VLVisionRegistry"),
        ("qwen3_5", "qwen3_5", "Qwen3_5Registry"),
        ("qwen3_5_moe", "qwen3_5_moe", "Qwen3_5MoeRegistry"),
        ("qwen3_5_vision", "qwen3_5_vision", "Qwen3_5VisionRegistry"),
        ("qwen3vl_vision", "qwen3vl_vision", "Qwen3VLVisionRegistry"),
        ("qwenpi05", "qwenpi05", "QwenPI05Registry"),
        ("rhino_vla", "rhino_vla", "RhinoVLARegistry"),
        ("siglip", "siglip", "SigLIPRegistry"),
        ("wall_oss_action_step", "wall_oss_action_step", "WallOssActionStepRegistry"),
        ("lingbot_denoise", "lingbot_denoise", "LingbotDenoiseRegistry"),
    )
    header = _read("src/core/rpu_kernel_decls.h")
    dispatch = _read("src/core/rpu_dispatch_registrations.inc")
    expected_sources = {f"rpu_{file}_model.cpp" for file, _, _ in owners}
    direct_auto_sources = {
        path.name for path in (ROOT / "src/fused").glob("rpu_*_model.cpp")
        if "resolve_kvinsert_plan_auto(" in path.read_text(encoding="utf-8")
    }
    assert expected_sources == direct_auto_sources | {
        "rpu_wall_oss_action_step_model.cpp", "rpu_lingbot_denoise_model.cpp",
    }
    for file, prefix, registry in owners:
        source = _read(f"src/fused/rpu_{file}_model.cpp")
        bind_name = f"rpu_{prefix}_bind_kvinsert_costs"
        getter_name = f"rpu_{prefix}_kvinsert_cost_catalog_sha256"
        create = _function(source, f"rpu_{prefix}_create")
        assert f"{registry}::create(" in create, prefix
        bind = _function(source, bind_name)
        getter = _function(source, getter_name)
        assert f'{registry}::get(handle, "{bind_name}")' in bind, prefix
        assert "->bind_kvinsert_costs(identity, catalog_sha256, certificate_rows)" in bind
        assert f'{registry}::get(handle,"{getter_name}")' in "".join(getter.split()), prefix
        assert "->kvinsert_cost_catalog_sha256()" in getter
        for name, body in ((bind_name, bind), (getter_name, getter)):
            assert body.count("::get(") == 1
            assert "CausalDecoderRegistry" not in body
            assert header.count(name + "(") == 1
            assert dispatch.count(f"TORCH_FN({name})") == 1
        assert dispatch.count(
            f'{prefix}_bind_kvinsert_costs(int handle, int[] identity, '
            'str catalog_sha256, int[] certificate_rows) -> ()'
        ) == 1
        assert dispatch.count(
            f'{prefix}_kvinsert_cost_catalog_sha256(int handle) -> str'
        ) == 1
    for file in ("wall_oss_action_step", "lingbot_denoise"):
        assert "public CausalDecoderModel" in _read(f"src/fused/rpu_{file}_model.h")
    # Fixed-V2 owners have no measured route competition. Composite Pi05 has
    # no leaf FMB of its own and must not pretend to propagate a parent catalog.
    for prefix in ("gemma", "gemma2", "gemma4", "pi05"):
        assert f'"{prefix}_bind_kvinsert_costs(' not in dispatch




def test_competing_fused_owners_propagate_site_and_actual_lifecycle():
    sites = {
        "adarms": ("ADARMS_KV_INSERT_SITE",),
        "dinov3_vision": ("DINO_KV_INSERT_SITE",),
        "gr00t_dit": ("site_id",),
        "gr00t_vl_encoder": ("GR00T_VL_KV_INSERT_SITE",),
        "halo_action_expert": ("HALO_ACTION_KV_SITE",),
        "halo_image_flow": ("HALO_IMAGE_FLOW_KV_SITE",),
        "hyvit2_vision": ("HYVIT2_KV_SITE",),
        "hyvla_expert": ("HYVLA_EXPERT_KV_SITE",),
        "hyvla_vlm": ("HYVLA_VLM_KV_SITE",),
        "internvla_nextdit": ("self_kv_site", "kNextDitCrossKvPopulateSite"),
        "lingbot_v2_moe": ("L2_KV_INSERT",),
        "navdp": ("kNavdpSelfKvSite", "kNavdpCrossKvSite"),
        "pi05_denoise_step": ("PI05_DENOISE_KV_INSERT_SITE",),
        "qwen25vl_vision": ("QWEN25_KV_INSERT_SITE",),
        "qwen3_5": ("kFullAttentionKvInsertSite",),
        "qwen3_5_moe": ("QWEN35_MOE_KV_INSERT_SITE",),
        "qwen3_5_vision": ("site_id",),
        "qwen3vl_vision": ("QWEN3VL_KV_INSERT_SITE",),
        "qwenpi05": ("QWENPI_KV_INSERT_SITE",),
        "rhino_vla": ("RHINO_KV_INSERT_SITE",),
        "siglip": ("SIGLIP_KV_INSERT_SITE",),
    }
    for owner, expected in sites.items():
        source = _read(f"src/fused/rpu_{owner}_model.cpp")
        for site in expected:
            assert re.search(
                rf"\bresolve_kvinsert_plan_auto\(\s*{site},\s*manifest.graph_lifecycle,",
                source,
            ), (owner, site)
        assert "restore_kvinsert_plan(" in source, owner
        assert "rpu_kvinsert_segment_plan_from_route_arguments(" not in source, owner


def test_unbound_fused_kv_calls_are_only_fixed_v2_or_explicit_legacy_fallbacks():
    # No COMPLETE production candidate may bypass its owner catalog. Legacy
    # no-manifest / native-Z2 fallback calls intentionally remain unsealed;
    # they cannot borrow a lifecycle from an unrelated retained descriptor.
    fixed_v2 = {"gemma", "gemma2", "gemma4"}
    legacy_counts = {
        "adarms": 1, "halo_action_expert": 1, "halo_image_flow": 1,
        "pi05_denoise_step": 1, "qwen25vl_vision": 1,
        "qwen3vl_vision": 1, "qwen3_5": 1, "qwen3_5_vision": 3,
    }
    for path in (ROOT / "src/fused").glob("rpu_*_model.cpp"):
        owner = path.stem.removeprefix("rpu_").removesuffix("_model")
        source = path.read_text(encoding="utf-8")
        calls = re.findall(
            r"\brpu_resolve_kvinsert_segment_plan_auto\([\s\S]*?\);", source
        )
        if owner in fixed_v2:
            assert len(calls) == 1, owner
            assert calls[0].rstrip().endswith("KV_INSERT_CAP_V2);"), owner
        elif owner in legacy_counts:
            blocks = re.findall(
                r"if \(!ctx\(\).has_complete_physical_manifest\(\)\) \{[^{}]*\}",
                source,
            )
            unbound = sum("return rpu_resolve_kvinsert_segment_plan_auto(" in block
                          for block in blocks)
            assert len(calls) == unbound == legacy_counts[owner], owner
        else:
            assert not calls, owner


def test_cross_attention_restores_preserve_auto_cost_authority():
    for owner in ("navdp", "internvla_nextdit"):
        source = _read(f"src/fused/rpu_{owner}_model.cpp")
        body = _function(source, "resolve_kv_insert_plan")
        assert "const KvInsertSegmentPlan plan = restore_kvinsert_plan(" in body
        assert "const KvInsertSegmentPlan shape_plan =" in body
        assert "canonical.begin() + kKvInsertRouteTopologyWords" in body
        assert "return plan;" in body


def test_pi05_direct_cache_uses_exact_abi_without_expanding_gemma_auto_domain():
    source = _read("src/fused/rpu_gemma_model.cpp")
    declaration = source.split("const KvInsertSegmentPlan kv_plan =", 1)[1].split(
        "const KvInsertRouteArguments arguments", 1
    )[0]
    exact, ordinary = declaration.split(": rpu_resolve_kvinsert_segment_plan_auto(")
    assert "pi05_prefill_kv1_pair_direct_" in exact
    assert "? rpu_resolve_kvinsert_segment_plan(" in exact
    assert "KV_INSERT_CAP_V16, KvInsertRoute::ALIGNED_V16)" in exact
    assert ordinary.rstrip().endswith("KV_INSERT_CAP_V2);")
    assert "KV_INSERT_CAP_V16" not in ordinary


def test_qwen35_dynamic_decode_never_borrows_position_zero_costs():
    source = _read("src/fused/rpu_qwen3_5_model.cpp")
    branch = source.split(
        "const KvInsertSegmentPlan kv_plan = dynamic_decode_position", 1
    )[1].split("const KvInsertRouteArguments arguments", 1)[0]
    dynamic, ordinary = branch.split(": resolve_kvinsert_plan_auto(", 1)
    assert "rpu_resolve_kvinsert_segment_plan(" in dynamic
    assert "KV_INSERT_CAP_V2," in dynamic
    assert "KvInsertRoute::V2)" in dynamic
    assert "kFullAttentionKvInsertSite, manifest.graph_lifecycle," in ordinary


def test_plan_launchers_share_const_authority_and_never_resolve_or_read_env():
    header = _read("src/core/rpu_kernel_decls.h")
    source = _read("src/ops/rpu_llama_kvcache.cpp")
    for kind in ("k", "v"):
        name = f"rpu_launch_insert_{kind}cache_spm_unified_with_plan"
        assert name in header
        declaration = header[header.index(name) : header.index(");", header.index(name))]
        assert "const KvInsertSegmentPlan& plan" in declaration
        body = _function(source, name)
        assert "rpu_validate_kvinsert_segment_plan(" in body
        assert "for (size_t i = 0; i < plan.segment_count(); ++i)" in body
        assert "stage_kernel_ddr_registers(" in body
        assert "rpu_resolve_kvinsert_segment_plan(" not in body
        assert "rpu_resolve_legacy_kvinsert_segment_plan(" not in body
        assert "std::getenv(" not in body
        assert "single-launch" not in body


def test_only_legacy_translator_reads_kvinsert_environment():
    source = _read("src/ops/rpu_llama_kvcache.cpp")
    translator = _function(source, "legacy_kvinsert_settings_from_env")
    assert source.count("std::getenv(") == translator.count("std::getenv(")
    for name in (
        "RPU_KVINSERT_V16",
        "RPU_KVINSERT_V16_ANY_TP",
        "RPU_KVINSERT_V16_TRACE",
        "RPU_KVINSERT_HYBRID_V16",
        "RPU_RHINOVLA_KVINSERT_HYBRID_V16",
        "RPU_KVINSERT_HYBRID3_V16",
    ):
        assert name in translator


def test_legacy_abi_is_a_thin_compatibility_wrapper():
    source = _read("src/ops/rpu_llama_kvcache.cpp")
    for kind in ("k", "v"):
        name = f"rpu_launch_insert_{kind}cache_spm_unified"
        body = _function(source, name)
        assert "legacy_kvinsert_settings_from_env()" in body
        assert "rpu_resolve_legacy_kvinsert_segment_plan(" in body
        assert f"rpu_launch_insert_{kind}cache_spm_unified_with_plan(" in body
        assert "stage_kernel_ddr_registers(" not in body


def test_production_kv_pairs_resolve_once_and_share_one_plan():
    header = _read("src/core/rpu_kernel_decls.h")
    source = _read("src/ops/rpu_llama_kvcache.cpp")
    paired = "rpu_launch_insert_kvcache_spm_unified"
    paired_with_plan = paired + "_with_plan"
    assert paired in header
    declaration = header[
        header.index(paired_with_plan) : header.index(");", header.index(paired_with_plan))
    ]
    assert "const KvInsertSegmentPlan& plan" in declaration

    with_plan = _function(source, paired_with_plan)
    assert with_plan.index("rpu_launch_insert_kcache_spm_unified_with_plan(") < with_plan.index(
        "rpu_launch_insert_vcache_spm_unified_with_plan("
    )
    assert with_plan.count("plan, trace") == 2
    assert "legacy_kvinsert_settings_from_env()" not in with_plan

    compatibility = _function(source, paired)
    assert compatibility.count("legacy_kvinsert_settings_from_env()") == 1
    assert compatibility.count("rpu_resolve_legacy_kvinsert_segment_plan(") == 1
    assert compatibility.count(paired_with_plan + "(") == 1

    production = _read("src/ops/rpu_sdpa.cpp") + "".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "src/fused").glob("*.[ch]pp")
    ) + "".join(
        path.read_text(encoding="utf-8")
        for path in (ROOT / "src/fused").glob("*.h")
    )
    assert "rpu_launch_insert_kcache_spm_unified(" not in production
    assert "rpu_launch_insert_vcache_spm_unified(" not in production
    assert paired + "(" not in production
    assert paired_with_plan + "(" in production

    arena = _function(source, "rpu_launch_llama_insert_kvcache_arena")
    assert "k_input.sizes() == v_input.sizes()" in arena
    assert paired + "(" not in arena
    assert arena.count(paired_with_plan + "(") == 1
    assert "rpu_resolve_kvinsert_segment_plan_auto(" not in arena
    assert "kv_arena_plan_from_arguments(" in arena
    assert arena.index("kv_arena_plan_from_arguments(") < arena.index(
        "stage_kv_to_arena("
    )
    resolver = _function(source, "rpu_resolve_llama_kvcache_arena_plan")
    assert resolver.count("rpu_resolve_kvinsert_segment_plan_auto(") == 1
    assert "rpu_kvinsert_route_arguments(" in resolver
    rhinovla = _read("python/rpu_backend/adapters/rhinovla/convert.py")
    assert "torch.rpu.resolve_kvcache_arena_plan(" in rhinovla
    assert "list(planned_route_arguments)" in rhinovla
    assert "torch.rpu.insert_kcache_arena(" not in rhinovla
    assert "torch.rpu.insert_vcache_arena(" not in rhinovla


def test_arena_launchers_require_exact_route_arguments_before_staging():
    source = _read("src/ops/rpu_llama_kvcache.cpp")
    header = _read("src/core/rpu_kernel_decls.h")
    pybind = _read("src/core/rpu_pybind.inc")
    for kind in ("kcache", "vcache", "kvcache"):
        name = f"rpu_launch_llama_insert_{kind}_arena"
        body = _function(source, name)
        assert "rpu_resolve_kvinsert_segment_plan_auto(" not in body
        assert body.index("kv_arena_plan_from_arguments(") < body.index(
            "stage_kv_to_arena("
        )
        declaration = header[
            header.index(name) : header.index(");", header.index(name))
        ]
        assert "at::IntArrayRef planned_route_arguments" in declaration
    assert pybind.count('py::arg("planned_route_arguments")') == 3
