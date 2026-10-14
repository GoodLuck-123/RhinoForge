from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
RUNTIME = (ROOT / "src/core/rpu_runtime_extras.cpp").read_text()


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def test_qwen3vl_live_norm_preloads_are_owned_on_both_spm_pipeline_branches():
    declarations = _between(
        HEADER,
        "std::vector<BufferDecl> declare_buffers(const LayoutContext& ctx) override",
        "ModelStaticConfig static_config() override",
    )
    owned_layer_weights = (
        "layer_weights_[L].input_norm_w",
        "layer_weights_[L].post_norm_w",
        "layer_weights_[L].q_norm_w",
        "layer_weights_[L].k_norm_w",
    )
    for marker in owned_layer_weights:
        block = declarations[declarations.index(marker) :]
        block = block[: block.index("decls.push_back(")]
        assert "if (qwen3vl_pooler_z1_dispatch_ ||" in block
        assert "qwen3vl_multiview_text_composite_dispatch_)" in block
        assert "weight, 0," in block
        assert "weight.data_ptr<c10::Half>()" in block

    final_norm = _between(
        declarations,
        'final_norm.name     = "final_norm_w"',
        "decls.push_back(final_norm);",
    )
    assert "if (qwen3vl_pooler_z1_dispatch_ ||" in final_norm
    assert "qwen3vl_multiview_text_composite_dispatch_)" in final_norm
    assert "final_norm_w_, 0, hidden_size(), core0_addr" in final_norm
    assert "final_norm_w_.data_ptr<c10::Half>()" in final_norm

    # This canary does not widen ownership to the excluded AdaRMS, qkv-bias,
    # or NVFP4 branches.
    excluded = _between(
        declarations,
        "if (adarms_) {",
        "// ★ final_norm fuse:",
    )
    for marker in ("input_shift", "post_shift", "has_qkv_bias_", "nvfp4_"):
        assert marker in excluded
    assert excluded.count(
        "if (qwen3vl_pooler_z1_dispatch_ ||"
    ) == 2  # q/k norm callbacks only


def test_qwen3vl_default_mrope_is_owned_on_both_spm_pipeline_branches():
    rope = _between(
        HEADER,
        "void emit_qk_rope(",
        "// Replay-safe AdaRMS",
    )
    partial = rope.index("if (has_mrope_ && partial_mrope_active_)")
    owned = rope.index("else if (has_mrope_ &&")
    legacy = rope.index("else if (has_mrope_)", owned + 1)
    assert partial < owned < legacy
    assert "qwen3vl_pooler_z1_dispatch_ ||" in rope[owned:legacy]
    assert "qwen3vl_multiview_text_composite_dispatch_" in rope[owned:legacy]
    assert "cos_, sin_, position_ids_keepalive_, cos_sin_start" in rope[owned:legacy]
    assert "cos_.data_ptr<c10::Half>(), sin_.data_ptr<c10::Half>()" in rope[legacy:]
    assert "position_ids_keepalive_.data_ptr<int32_t>()" in rope[legacy:]


def test_ring_allreduce_has_an_explicit_spm_only_census_proof():
    launcher = _between(
        RUNTIME,
        "void launch_ring_all_reduce_sum_residual(",
        "void rpu_launch_all_reduce_sum_residual_impl(",
    )
    lookup = launcher.index("Kernel_t* kernel = GET_KERNEL(kernel_id)")
    proof = launcher.index(
        "GraphKernelNoDdrProof::RingAllReduceSpmResidual"
    )
    census_guard = launcher.index("if (graph.kernel_register_census_active())")
    guarded = launcher[census_guard:lookup]

    assert "residual_ddr == nullptr" in guarded
    assert "stage_kernel_no_ddr(" in guarded
    assert census_guard < proof < lookup

    admission = _between(
        RUNTIME,
        "bool ring_all_reduce_sum_residual_admitted(",
        "void launch_ring_all_reduce_sum_residual(",
    )
    for address in ("input_spm", "residual_spm", "output_spm"):
        assert address in admission
    assert "SPM_ALLOC.is_initialized()" in admission
    assert "SPM_ALLOC.addr(0, 0)" in admission
    assert "TWOSTAGE" not in RUNTIME
