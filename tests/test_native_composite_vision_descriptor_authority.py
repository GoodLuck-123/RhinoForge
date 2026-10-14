"""Source contracts for Qwen3-VL Vision authority in native composites."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp"
POOLER = ROOT / "src/fused/rpu_qwen3vl_pooler_spm_z1.cpp"
Z2 = ROOT / "src/fused/rpu_qwen3vl_multiview_spm_z2.cpp"
REGISTRATIONS = ROOT / "src/core/rpu_dispatch_registrations.inc"
LINGBOT = ROOT / "python/rpu_backend/adapters/lingbot_vla_v2/runtime.py"
GROOT = ROOT / "python/rpu_backend/adapters/gr00t/runtime.py"


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start:source.index(end, start)]


def test_vision_prepare_retains_complete_descriptor_and_every_forward_uses_it():
    source = VISION.read_text(encoding="utf-8")
    helper = _between(
        source,
        "std::vector<int64_t> build_native_composite_stage_descriptor(",
        "bool retained_deepstack_addresses_empty() const",
    )
    assert "resolve_prefill_stage_domain_for_shape(" in helper
    assert "FmbPhysicalManifestState::COMPLETE" in helper
    assert "FmbGraphLifecycle::COMPOSITE_CHILD" in helper
    assert "encode_fmb_prefill_stage_candidate(candidates.front())" in helper

    for begin, end in (
        ("SpmPipelineComponentLayout prepare_pooler_z1_layout(",
         "SpmPipelineComponentLayout prepare_multiview_dry_layout("),
        ("SpmPipelineComponentLayout prepare_multiview_dry_layout(",
         "void cancel_multiview_dry_layout()"),
        ("SpmPipelineComponentLayout prepare_multiview_composite_layout(",
         "SpmFmbResolvedExecutionProfile"),
    ):
        prepare = _between(source, begin, end)
        assert "build_native_composite_stage_descriptor(" in prepare
        assert "native_composite_stage_descriptor_ = std::move(" in prepare

    forward = _between(
        source, "at::Tensor forward_impl(",
        "SpmPipelineComponentLayout prepare_pooler_z1_layout(",
    )
    assert "TORCH_CHECK(!planned_stage_descriptor.empty()" in forward
    assert "run_all_layers(" in forward
    assert "planned_stage_descriptor);" in forward
    assert "planned_stage_descriptor.empty() ?" not in forward

    for begin, end in (
        ("void forward_pooler_z1(", "void forward_multiview_composite("),
        ("void forward_multiview_composite(",
         "void check_pooler_z1_destroy_allowed() const"),
    ):
        occurrence = _between(source, begin, end)
        assert "native_composite_stage_descriptor_" in occurrence
        assert "/*planned_stage_descriptor=*/{}" not in occurrence


def test_pipeline_manifest_declares_the_actual_merger_route_family():
    source = VISION.read_text(encoding="utf-8")
    manifest = _between(
        source,
        "FmbPhysicalExecutionManifest make_physical_manifest_for_candidate(",
        "FmbPhysicalManifestForwardCapability",
    )
    assert "z1_dispatch_" in manifest
    assert "z1_retained_deepstack_count_" in manifest
    assert "multiview_composite_dispatch_" in manifest
    assert "QWEN3VL_MERGER_YIELD_ALL_REDUCE_SITE" in manifest
    assert "QWEN3VL_MERGER_ALL_REDUCE_SITE" in manifest


def test_lingbot_dry_and_prepare_preserve_the_same_vision_descriptor():
    pooler = POOLER.read_text(encoding="utf-8")
    dry = _between(
        pooler,
        "std::vector<int64_t> rpu_qwen3vl_multiview_spm_dry_probe(",
        "int64_t rpu_qwen3vl_pooler_spm_z1_prepare(",
    )
    assert "kProbeSchema = 3" in dry
    assert "vision_stage_descriptor(" in dry
    assert "result.insert(" in dry

    z2 = Z2.read_text(encoding="utf-8")
    validation = _between(
        z2, "void validate_lingbot2_planner_descriptor(",
        "void validate_planner_identity(",
    )
    assert "prepare_stage_candidate(" in validation
    assert "FmbPhysicalManifestState::COMPLETE" in validation
    assert "FmbGraphLifecycle::COMPOSITE_CHILD" in validation
    prepare = _between(z2, "int64_t prepare_impl(", "}  // namespace")
    assert "prepared_vision_descriptor" in prepare
    assert "dry Vision descriptor disagrees with physical prepare" in prepare
    assert "lingbot2_child_stage_descriptor(planner_descriptor, false, operation)" in prepare
    physical = _between(VISION.read_text(), "SpmPipelineComponentLayout prepare_multiview_composite_layout(", "SpmFmbResolvedExecutionProfile")
    assert 'kvinsert_cost_domain("qwen3vl_vision", selected_descriptor)' in physical
    assert "kvinsert_cost_owner_prefix(selected_descriptor)" in physical
    assert "stage_descriptor.assign(selected_descriptor.begin(), selected_descriptor.end())" in physical
    assert "bind_descriptor_layout(" in physical


def test_native_composite_a6_and_graph_identity_include_vision_authority():
    lingbot = LINGBOT.read_text(encoding="utf-8")
    build = _between(
        lingbot, "def _build_z2_composite_plan(",
        "def _validate_prepared_request(",
    )
    assert build.count("_plan_z2_native_child(") == 2
    assert "child.selected.stage_tuple.physical_descriptor for child in children[:2]" in build
    assert build.count("parent_handle=handles[2]") == 2
    assert build.count("queue_owner_id=handles[2]") == 2
    assert build.count("lease_owner_id=handles[2]") == 2
    assert "parent_metadata" in build

    groot = GROOT.read_text(encoding="utf-8")
    children = _between(
        groot, "def _qwen_z1_component_plans(self):",
        "@torch.no_grad()",
    )
    assert "vision_descriptor, self._vision_execution" in children
    assert "self._qwen_z1_stage_descriptors()" in children
    assert "self._refresh_qwen_z1_cost_plan(selected=selected)" in children
    assert "plan_bounded_prefill_execution(" in children
    assert 'execution_native=(prefix, int(handle))' in children
    assert "queue_owner = id(self._qwen_z1_gc)" in children
    assert "lease_owner = id(self)" in children
    graph = _between(
        groot, "signature = rpu_backend.graph.GraphSignature(",
        "_assert_qwen3vl_component_caches_empty",
    )
    assert "*vision_plan.graph_key_words()" in graph
    assert "*text_plan.graph_key_words()" in graph

    registration = REGISTRATIONS.read_text(encoding="utf-8")
    assert (
        "qwen3vl_pooler_spm_z1_vision_stage_descriptor(int vision_handle, "
        "int text_handle, int plan_hash) -> int[]"
    ) in registration
