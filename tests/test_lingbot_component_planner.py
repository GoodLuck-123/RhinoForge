"""Board-free contracts for the LingBot V1/V2 planner owners."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_lingbot_v1_action_descriptor_controls_the_fused_unroll():
    native = _read("src/fused/rpu_lingbot_denoise_model.cpp")
    header = _read("src/fused/rpu_lingbot_denoise_model.h")
    runtime = _read("python/rpu_backend/adapters/lingbot_vla/runtime.py")
    dispatch = _read("src/core/rpu_dispatch_registrations.inc")

    assert "FIXED_KERNEL_BASIS" in native
    assert "resolve_action_stage_domain(" in header
    assert "physical_manifest_for_candidate(" in header
    assert "CausalDecoderModel::physical_manifest_for_candidate(" in native
    assert "FmbGraphLifecycle::COMPOSITE_CHILD" in native
    assert "ctx().body_iter" in native
    assert "{}, bit);" in native
    assert "planned_stage_descriptor" in header
    assert "planned_stage_descriptor);" in native
    assert "lingbot_denoise_resolve_action_stage_domain" in runtime
    assert "*action_plan.graph_key_words()" in runtime
    assert "stage_tuple.physical_descriptor" in runtime
    assert '"physical_descriptor": tuple(' in runtime
    assert "lingbot_denoise_resolve_action_stage_domain" in dispatch
    assert "int[] planned_stage_descriptor=[]" in dispatch

    # The V1 checkpoint's CPU Vision path owns no device launcher or graph.
    assert '"status": "NOT_APPLICABLE"' in runtime
    assert '"authority": "CPU_FP32"' in runtime
    assert "CPU Vision unexpectedly resolved physical controls" in runtime


def test_lingbot_v2_action_uses_complete_typed_ddr_descriptor():
    native = _read("src/fused/rpu_lingbot_v2_moe_model.cpp")
    header = _read("src/fused/rpu_lingbot_v2_moe_model.h")
    runtime = _read("python/rpu_backend/adapters/lingbot_vla_v2/runtime.py")
    dispatch = _read("src/core/rpu_dispatch_registrations.inc")

    assert "FIXED_KERNEL_BASIS" in native
    assert "resolve_action_stage_domain(" in header
    assert "physical_manifest_for_candidate(" in header
    assert "FmbPhysicalManifestState::COMPLETE" in native
    assert "FmbGraphLifecycle::COMPOSITE_CHILD" in native
    assert "CAUSAL_DECODER_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED" in native
    assert "AttentionExecutionPolicy::DDR_KV" in native
    assert "restore_kvinsert_plan(" in native
    assert "L2_KV_INSERT, manifest.graph_lifecycle," in native
    assert "rpu_kvinsert_segment_plan_from_route_arguments(" not in native
    assert "rpu_launch_insert_kvcache_spm_unified_with_plan(" in native
    assert "rpu_launch_insert_kvcache_spm_unified(" not in native
    assert "5154633245405693922LL" in native
    assert "planned_stage_descriptor" in header
    assert "planned_stage_descriptor);" in native

    assert "lingbot_v2_moe_resolve_action_stage_domain" in runtime
    assert "qwen3vl_vision_resolve_stage_domain" in runtime
    vision = next(node for node in ast.walk(ast.parse(runtime))
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_vision_execution_plans")
    planner = next(node for node in ast.walk(vision)
                   if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name)
                   and node.func.id == "plan_bounded_prefill_execution")
    metadata = next(item.value for item in planner.keywords
                    if item.arg == "physical_metadata")
    # Group geometry remains in the actual native plan alongside its new
    # component/generation identity, rather than being replaced by it.
    assert isinstance(metadata, ast.Tuple)
    assert any(isinstance(item, ast.Starred)
               and isinstance(item.value, ast.Name)
               and item.value.id == "group_metadata" for item in metadata.elts)
    assert "LINGBOT2_VISION_COMPONENT" in ast.unparse(metadata)
    assert "execution_generation" in ast.unparse(metadata)
    assert "*action_plan.graph_key_words()" in runtime
    assert "GRAPH_NATIVE_COMPOSITE1" in runtime
    assert "bind_execution_session(" in runtime
    assert "native_execution_reconfigure(" in runtime
    assert "LingBot2 action suffix has one fixed C64 tile" in runtime
    assert "lingbot_v2_moe_resolve_action_stage_domain" in dispatch


def test_lingbot_v2_repeated_routes_use_v3_invocations():
    native = _read("src/fused/rpu_lingbot_v2_moe_model.cpp")

    assert "for (int64_t body = 0; body < num_steps_; ++body)" in native
    assert "acc32(L2_PRE_WC_LINEAR, body);" in native
    assert "acc32(L2_POST_OP_LINEAR, body);" in native
    assert "for (int64_t expert = 0; expert < num_experts_; ++expert)" in native
    assert "acc16(L2_DENSE_GATE, expert);" in native
    assert "acc16(L2_DENSE_DOWN, expert);" in native
    assert "const int64_t invocation = chunk.idx;" in native
    assert "route.arguments, chunk.idx" in native


def test_lingbot_v2_descriptor_owns_runtime_topology_and_router_scaling():
    native = _read("src/fused/rpu_lingbot_v2_moe_model.cpp")

    for token in (
        "L2_DENOISE_SCHEDULE",
        "L2_ADARMS_SCHEDULE",
        "L2_ROUTER_SCHEDULE",
        "L2_MOE_EXECUTION_TOPOLOGY",
        "L2_ROUTED_SCALING",
        "FmbRouteFamily::GRAPH_SCHEDULE",
        "FmbRouteFamily::ACTIVATION",
        "static_cast<int64_t>(routed_scaling_.x)",
        "cold_schunk_, cold_rchunk_, cold_rchunk_requested_",
        "debug_routed_only_ ? 1 : 0",
        "debug_down_acc16_ ? 1 : 0",
    ):
        assert token in native
    assert native.count("FmbRouteFamily::GRAPH_SCHEDULE") >= 8
    scaling = native[native.index("if (!fused_router_ && routed_scaling_ != c10::Half(1.0f)) {") :]
    assert scaling.index("ctx().consume_physical_route(") < scaling.index(
        "rpu_launch_eltwise_binary_scalar_spm_kernel("
    )
