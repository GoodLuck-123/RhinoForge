"""Board-free contract for sized DDR flushes on live tensor storage."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TENSOR_OPS = (ROOT / "src/core/rpu_tensor_ops.inc").read_text(encoding="utf-8")
VISION = (ROOT / "src/fused/rpu_qwen25vl_vision_model.cpp").read_text(
    encoding="utf-8"
)
DISPATCH = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text(
    encoding="utf-8"
)
LINGBOT2_MOE = (ROOT / "src/fused/rpu_lingbot_v2_moe_model.cpp").read_text(
    encoding="utf-8"
)
QWEN3VL_VISION = (ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp").read_text(
    encoding="utf-8"
)
CAUSAL_DECODER = (ROOT / "src/fused/rpu_qwen3_model.h").read_text(encoding="utf-8")


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def test_tensor_bridge_bounds_contiguous_and_strided_flushes():
    zero_copy = _between(
        TENSOR_OPS,
        "static at::Tensor rpu_to_cpu_zerocopy_impl(",
        "// ------------------------ copy_ (CPU <-> device, device <-> device)",
    )
    copy = _between(TENSOR_OPS, "at::Tensor &rpu_copy_impl(", "// rpu_copy_: in-place")

    assert zero_copy.count(
        "rpu_ddr_flush_force_sized(data_ptr, original_storage.nbytes());"
    ) == 2
    assert "rpu_ddr_flush_force(data_ptr);" not in zero_copy
    assert copy.count("rpu_ddr_flush_force_sized(src.data_ptr(), src.nbytes());") == 3
    assert copy.count("rpu_ddr_flush_force_sized(self.data_ptr(), self.nbytes());") == 3
    assert copy.count("storage.data_ptr().get(), storage.nbytes()") == 3
    assert "rpu_ddr_flush_force(const_cast<void*>(src.data_ptr()))" not in copy
    assert (
        "rpu_elementwise_copy_impl(self, src);\n"
        "        const auto storage = self.storage();"
    ) in copy


def test_qwen25vl_flushes_live_position_without_publishing_fresh_merger_output():
    ordinary = _between(
        VISION,
        "at::Tensor forward_impl(",
        "SpmPipelineComponentLayout prepare_z1_layout(",
    )
    prime = _between(VISION, "void prime_z1_inputs(", "void rollback_z1_inputs(")
    merger = _between(VISION, "void merger_post_fn()", "private:")

    assert "num_patches_in * 2 * sizeof(int16_t)" in ordinary
    assert "num_patches * 2 * sizeof(int16_t)" in prime
    assert "rpu_ddr_flush_force_sized" not in merger
    for body in (ordinary, prime, merger):
        assert "rpu_ddr_flush_force(" not in body


def test_deepstack_handoff_validates_and_retains_device_inputs():
    body = _between(CAUSAL_DECODER, "// R-Phase 2 (Qwen3-VL): DeepStack forward-entry handling",
                    "const bool decode_fused_lm_head")
    retain = body.index("deepstack_dense_refs_.push_back(d);")
    address = body.index("::rhino_lkn::RpuGetDevAddr(d.data_ptr<c10::Half>());")
    for guard in ("d.size(0) >= seq_len_in", "d.size(1) == h_local",
                  "d.scalar_type() == at::kHalf", "d.is_contiguous()",
                  "d.device().type() == at::kPrivateUse1"):
        assert body.index(guard) < retain < address
    # Producers already publish CPU writes; a validated device handoff must
    # retain storage without repeating the tensor bridge's cache maintenance.
    assert "rpu_ddr_flush_force" not in body


def test_host_fill_ops_flush_the_backing_storage():
    fill = _between(DISPATCH, 'm.impl("fill_.Scalar"', '// cos/sin - device native')
    zero = _between(DISPATCH, 'm.impl("zero_"', '// gt / lt / ge / le / eq')

    assert fill.count("storage.data_ptr().get(), storage.nbytes()") == 2
    assert "rpu_ddr_flush_force(self.data_ptr())" not in fill
    assert "storage.data_ptr().get(), storage.nbytes()" in zero
    assert "rpu_ddr_flush_force(self.data_ptr())" not in zero


def test_lingbot2_exact_z2_flushes_only_live_ranges():
    build = _between(
        LINGBOT2_MOE,
        "ones_pad_ = at::ones",
        "// ── DEBUG dense-soft router opt-in",
    )
    denoise = _between(
        LINGBOT2_MOE,
        "void LingbotV2MoeExpertModel::denoise_unroll_forward(",
        "}  // namespace v3",
    )
    prime = _between(
        QWEN3VL_VISION,
        "void prime_multiview_composite_position()",
        "void clear_multiview_composite_position()",
    )

    for tensor in ("ones_pad_", "expert_scatter_ids_"):
        assert f"{tensor}.data_ptr<c10::Half>()" in build
        assert f"{tensor}.nbytes()" in build
    for tensor in ("x0_rpu", "state_rpu"):
        assert f"{tensor}.data_ptr<c10::Half>(), {tensor}.nbytes()" in denoise
    reentry = denoise.index("TORCH_CHECK(!denoise_unroll_active_")
    enter = denoise.index("denoise_unroll_active_ = true;", reentry)
    cleanup = denoise.index("auto clear_denoise_scope = c10::make_scope_exit", enter)
    forward = denoise.index("(void)CausalDecoderModel::forward(", cleanup)
    assert reentry < enter < cleanup < forward
    assert "denoise_unroll_active_ = false;" in denoise[cleanup:forward]
    assert "ChunkOverrideGuard" not in denoise
    assert "g_chunk_size_override" not in denoise
    assert "QWEN3VL_POOLER_Z1_NUM_PATCHES * 2 * sizeof(int16_t)" in prime
    for body in (build, denoise, prime):
        assert "rpu_ddr_flush_force(" not in body
