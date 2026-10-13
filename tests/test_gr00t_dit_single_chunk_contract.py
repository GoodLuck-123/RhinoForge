"""GR00T action DiT must never execute a multi-chunk self-attention plan."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIT = (ROOT / "src/fused/rpu_gr00t_dit_model.cpp").read_text()
VL = (ROOT / "src/fused/rpu_gr00t_vl_encoder_model.cpp").read_text()
BASE = (ROOT / "src/core/fused_model_base.cpp").read_text()
RUNTIME = (ROOT / "python/rpu_backend/adapters/gr00t/runtime.py").read_text()
DECLARATIONS = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
REGISTRATIONS = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()


def _between(source, start, end):
    begin = source.index(start)
    return source[begin:source.index(end, begin)]


def test_action_auto_is_one_chunk_and_multichunk_fails_before_build():
    # The shipped action geometry is 40 actions plus one state token.  Native
    # auto starts at ceil16(41)=48, so the current action body is one chunk.
    assert "_AH, _AD, _AD_PAD = 40, 132, 144" in RUNTIME
    assert "_ACTION_QUERY_ROWS = _AH + 1" in RUNTIME
    assert (
        "_ACTION_CHUNK_SIZE = ((_ACTION_QUERY_ROWS + 15) // 16) * 16"
        in RUNTIME
    )
    action_rows = 40 + 1
    auto_chunk = ((action_rows + 15) // 16) * 16
    assert (
        action_rows,
        auto_chunk,
        (action_rows + auto_chunk - 1) // auto_chunk,
    ) == (41, 48, 1)

    dynamic = _between(
        DIT,
        "ModelDynamicConfig dynamic_config(const ChunkPlan& plan)",
        "// ---- SPM manifest",
    )
    assert "TORCH_CHECK(plan.num_chunks == 1" in dynamic
    assert "overwrite self-attention K/V at position 0" in dynamic

    self_attn = _between(
        DIT, "void build_self_attn_block", "void build_cross_attn_block"
    )
    assert "restore_kvinsert_plan(" in DIT
    assert "site_id, manifest.graph_lifecycle," in DIT
    assert "rpu_kvinsert_segment_plan_from_route_arguments(" not in DIT
    assert "rpu_launch_insert_kvcache_spm_unified_with_plan(" in self_attn
    assert "rpu_launch_insert_kvcache_spm_unified(" not in self_attn

    # FusedModelBase invokes the virtual guard before SPM allocation and before
    # the layer loop, so a forced 16-row plan (3 chunks for seq=41) cannot emit.
    forward = _between(
        BASE, "// 5. chunk computation", "// 13. post (D-501 post_fn)"
    )
    guard = forward.index("dynamic_config(plan)")
    assert guard < forward.index("ensure_allocated_impl")
    assert guard < forward.index("build_layer_subgraph")
    assert (action_rows + 16 - 1) // 16 == 3


def test_vl_and_action_forward_consume_full_native_stage_descriptors():
    for source, resolver, getter in (
        (
            VL,
            "rpu_gr00t_vl_encoder_resolve_stage_domain",
            "rpu_gr00t_vl_encoder_get_resolved_chunk_size",
        ),
        (
            DIT,
            "rpu_gr00t_denoise_resolve_stage_domain",
            "rpu_gr00t_denoise_get_resolved_chunk_size",
        ),
    ):
        assert resolver in source
        assert getter in source
        assert "resolve_prefill_stage_domain_for_shape(" in source
        assert "encode_fmb_prefill_stage_domain(" in source
        assert "at::IntArrayRef planned_stage_descriptor" in source
        assert "/*planned_chunk_size=*/0, planned_stage_descriptor" in source
        assert resolver in DECLARATIONS
        assert getter in DECLARATIONS
        assert resolver in REGISTRATIONS
        assert getter in REGISTRATIONS

    vl_schema = _between(
        REGISTRATIONS,
        'm.def("gr00t_vl_encoder_forward(',
        'm.def("gr00t_vl_encoder_forward_z2',
    )
    action_schema = _between(
        REGISTRATIONS,
        'm.def("gr00t_denoise_unroll_forward(',
        'm.def("gr00t_denoise_unroll_forward_z2',
    )
    assert "int[] planned_stage_descriptor=[]" in vl_schema
    assert "int[] planned_stage_descriptor=[]" in action_schema


def test_runtime_keys_graphs_and_receipts_on_each_component_plan():
    for component in (
        '"text_backbone"',
        '"vision_encoder"',
        '"vl_encoder"',
        '"action_dit"',
    ):
        assert component in RUNTIME
    assert RUNTIME.count("*plan.graph_key_words()") >= 1
    assert "*action_plan.graph_key_words()" in RUNTIME
    assert "*prefill_plan_result.graph_key_words()" in RUNTIME
    assert "physical_plan_digest" in RUNTIME
    assert "dry_forward_agreement" in RUNTIME
    for method, next_method, plan_name in (
        ("    def _vl_encode(", "    def _action_execution_plan(", "plan"),
        ("    def _denoise(", "    # ---- full image", "action_plan"),
    ):
        body = _between(RUNTIME, method, next_method)
        assert f"{plan_name}.selected.stage_tuple.physical_descriptor" in body
        assert "stage_descriptor = tuple(" in body
        assert "                stage_descriptor," in body
        assert "physical_descriptor=stage_descriptor" in body
    assert "@execution_serialized" in _between(
        RUNTIME, "    @execution_serialized", "# ============================================================================="
    )


def test_plain_and_step_entries_share_descriptor_authority_without_unroll_credit():
    for entry, end in (("at::Tensor forward(", "// step_forward"),
                       ("at::Tensor step_forward(", "// unroll_forward")):
        body = _between(DIT, entry, end)
        assert "at::IntArrayRef planned_stage_descriptor" in body
        assert "/*planned_chunk_size=*/0, planned_stage_descriptor" in body
    plain = _between(DIT, "at::Tensor forward(", "// step_forward")
    assert "denoise_mode_ = false" in plain and "loop_mode_ = false" in plain
    for name in ("gr00t_dit_forward", "gr00t_denoise_step_forward"):
        schema = REGISTRATIONS.split(f'm.def("{name}(', 1)[1].split("-> Tensor", 1)[0]
        assert "int[] planned_stage_descriptor=[]" in schema
    assert "rpu_gr00t_dit_resolve_stage_domain" in DECLARATIONS
    assert "rpu_gr00t_dit_resolve_stage_domain" in REGISTRATIONS
    # The ordinary -1 unroll keeps its original two-word arguments; each direct
    # step keys its baked cond/tau offsets, and the cold oracle restores them.
    assert "int step_idx=-1" in REGISTRATIONS
    assert "step_idx_ = saved_step" in DIT
    assert DIT.count("adaln_route_arguments(denoise_mode_, loop_mode_, vl_seq_, step_idx_)") == 2
