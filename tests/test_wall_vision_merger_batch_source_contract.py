"""Board-free contract for exact Wall Vision M432 merger batching."""

from pathlib import Path


VISION = (
    Path(__file__).resolve().parents[1]
    / "src/fused/rpu_qwen25vl_vision_model.cpp"
).read_text()


def test_exact_w8_merger_batches_after_in_place_image_norms() -> None:
    decl = VISION[
        VISION.index("if (is_exact_mimc()) {", VISION.index("declare_buffers(")) :
        VISION.index("return decls;", VISION.index("declare_buffers("))
    ]
    assert "if (exact_mimc_w8_qkv_enabled()) {" in decl
    assert "const int64_t mrows = packed_seq / 4;" in decl
    assert '{"merger_mid", mid_sz, 11, 13,' in decl
    assert '{"merger_zero", out_sz, 14, 14,' in decl
    assert "2 * out_sz <= packed_res" in decl

    post = VISION[
        VISION.index("void merger_post_fn()") : VISION.index(
            "private:", VISION.index("void merger_post_fn()")
        )
    ]
    exact, fallback = post.split(
        "if (exact_mimc_w8_qkv_enabled()) {", 1
    )[1].split(
        "\n        const int64_t seq = total_seq / image_batch_count_;", 1
    )
    assert "image_addr, image_addr" in exact
    assert "total_seq, /*dst_offset_elems=*/0, packed_addr" in exact
    assert "/*input_is_normalized=*/true" in exact
    assert fallback.count("emit_merger_chunk(") == 1


def test_batched_partial_output_and_mutable_dma_are_disjoint() -> None:
    emit = VISION[
        VISION.index("void emit_merger_chunk(") : VISION.index(
            "void merger_post_fn()", VISION.index("void emit_merger_chunk(")
        )
    ]
    assert "const uint32_t partial_addr = input_is_normalized" in emit
    assert "? input_addr : addr(0, \"merger_normed\")" in emit
    assert "partial_addr, addr(0, \"merger_zero\"), merger_out_addr" in emit
    assert "rpu_launch_spm_copy_ddr_dma_mutable(" in emit
