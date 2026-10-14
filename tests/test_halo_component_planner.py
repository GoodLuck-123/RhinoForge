import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text()


def test_halo_native_fmb_children_consume_the_planned_descriptor():
    action_adapter = _read("python/rpu_backend/adapters/halo/runtime.py")
    image_adapter = _read("python/rpu_backend/adapters/halo/image_flow.py")
    action_h = _read("src/fused/rpu_halo_action_expert_model.h")
    action_cpp = _read("src/fused/rpu_halo_action_expert_model.cpp")
    image_h = _read("src/fused/rpu_halo_image_flow_model.h")
    image_cpp = _read("src/fused/rpu_halo_image_flow_model.cpp")

    for header, source in ((action_h, action_cpp), (image_h, image_cpp)):
        assert "resolve_stage_domain(" in header
        assert "planned_stage_descriptor" in header
        assert "physical_manifest_for_candidate(" in header
        assert "physical_manifest_forward_capability(" in header
        assert "encode_fmb_prefill_stage_domain(" in source
        assert "resolve_prefill_stage_domain_for_shape(" in source
        assert "planned_stage_descriptor" in source
        assert "FmbPhysicalManifestState::COMPLETE" in source
        assert "rpu_launch_insert_kvcache_spm_unified_with_plan(" in source
        assert "rpu_launch_insert_kvcache_spm_unified(" not in source

    assert "planned_stage_descriptor);" in action_cpp
    assert "/*planned_chunk_size=*/0, planned_stage_descriptor" in image_cpp
    assert "self._attn_mask, cfg.prefix_len, descriptor)" in action_adapter
    assert "pb.cos, pb.sin, pb.prefix_len, descriptor)" in image_adapter
    assert "8691412429184154460LL" in action_cpp
    assert "1056254339788262076LL" in action_cpp
    assert "1091791700429545120LL" in image_cpp
    assert "6173290503646854925LL" in image_cpp
    assert "route.arguments, chunk.idx" in image_cpp
    assert "AttentionExecutionPolicy::DDR_KV, chunk.idx" in image_cpp

    production_calls = (
        (action_adapter, "halo_action_expert_step_forward", 2),
        (image_adapter, "halo_image_flow_step_forward", 1),
    )
    for source, op_name, expected_count in production_calls:
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == op_name
        ]
        assert len(calls) == expected_count
        assert all(
            call.args
            and isinstance(call.args[-1], ast.Name)
            and call.args[-1].id == "descriptor"
            for call in calls
        )

    for source in (action_cpp, image_cpp):
        assert "if (!ctx().has_complete_physical_manifest())" in source


def test_halo_image_flow_envelope_is_builder_owned_and_not_8192():
    adapter = _read("python/rpu_backend/adapters/halo/image_flow.py")
    native = _read("src/fused/rpu_halo_image_flow_model.cpp")

    assert "max_seq_len: int = 10976" in adapter
    assert "halo_image_flow_set_chunk_envelope(" in adapter
    assert "handle, int(max_seq_len), 0" in adapter
    assert "set_chunk_envelope(/*max_kv_len=*/8192" not in native
    assert "set_chunk_envelope(max_kv_len" in native
    assert "before the first forward" in _read("src/core/fused_model_base.cpp")


def test_all_halo_physical_children_use_one_component_controller_contract():
    runtime = _read("python/rpu_backend/adapters/halo/runtime.py")
    package = _read("python/rpu_backend/adapters/halo/__init__.py")
    siblings = "\n".join(
        _read(path)
        for path in (
            "python/rpu_backend/adapters/halo/text_cache.py",
            "python/rpu_backend/adapters/halo/text_decode.py",
            "python/rpu_backend/adapters/halo/image_flow.py",
            "python/rpu_backend/adapters/halo/vit_cache.py",
        )
    )

    for component in (
        '"action_expert"',
        '"image_flow"',
        '"text_cache"',
        '"text_decode"',
        '"vision_encoder"',
        '"vit_language_model"',
    ):
        assert component in runtime
    assert "class HaloExecutionController" in runtime
    assert "bind_execution_session(" in runtime
    assert "resolve_component_rpu_execution(" in runtime
    assert "GRAPH_COMPOSITE_CHILD" in runtime
    assert "HaloExecutionController" in package
    assert siblings.count("execution_controller: HaloExecutionController") == 4
    assert "execution_controller: HaloExecutionController" in runtime


def test_halo_prefix_history_routes_remain_ddr_required():
    files = "\n".join(
        _read(path)
        for path in (
            "python/rpu_backend/adapters/halo/runtime.py",
            "python/rpu_backend/adapters/halo/text_cache.py",
            "python/rpu_backend/adapters/halo/text_decode.py",
            "python/rpu_backend/adapters/halo/image_flow.py",
            "python/rpu_backend/adapters/halo/vit_cache.py",
        )
    )
    assert files.count('kv_route="DDR_REQUIRED"') >= 6
    assert '"kv_route:DDR_REQUIRED"' in files
    assert "PREFIX_HISTORY_DDR_REQUIRED_NO_RAW_RESIDENCY_ABI" in files
    runtime = _read("python/rpu_backend/adapters/halo/runtime.py")
    assert 'receipt["kv_route_reason"]' in runtime
    assert 'receipt["attention_reason"]' in runtime
    assert "route_reason:{_PREFIX_HISTORY_DDR_REASON}" in runtime


def test_halo_schemas_keep_full_descriptor_as_optional_abi_tail():
    registrations = _read("src/core/rpu_dispatch_registrations.inc")
    declarations = _read("src/core/rpu_kernel_decls.h")
    for op in (
        "halo_action_expert_resolve_stage_domain",
        "halo_image_flow_resolve_stage_domain",
        "halo_action_expert_set_chunk_envelope",
        "halo_image_flow_set_chunk_envelope",
    ):
        assert op in registrations
        assert op in declarations
    assert registrations.count('int[] planned_stage_descriptor=[]') >= 4
