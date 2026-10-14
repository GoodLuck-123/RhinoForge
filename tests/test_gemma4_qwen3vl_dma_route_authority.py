"""Board-free contracts for the residual Gemma4/Qwen3-VL DMA routes."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / "src/fused" / name).read_text(encoding="utf-8")


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def test_gemma4_ple_norm_preload_consumes_its_complete_route() -> None:
    source = _read("rpu_gemma4_model.cpp")
    manifest = _between(
        source, "physical_manifest_for_candidate(", "physical_manifest_forward_capability("
    )
    callback = _between(
        source, 'bn.name = "ple_post_norm_w"', "d.push_back(bn);"
    )

    assert "7312830814832177843LL" in source
    assert source.count("GEMMA4_PLE_NORM_PRELOAD_SITE") == 3
    assert "if (has_ple_)" in manifest
    assert "{layer, weight.numel(), has_ple_ ? 1 : 0}, layer" in manifest
    assert "{L, w.numel(), has_ple_ ? 1 : 0}, L" in callback
    assert callback.index("consume_physical_route(") < callback.index(
        "rpu_launch_ddr_broadcast_spm_dma("
    )


def test_qwen3vl_snapshot_and_merger_copies_use_exact_dma_receipts() -> None:
    source = _read("rpu_qwen3vl_vision_model.cpp")
    manifest = _between(
        source, "physical_manifest_for_candidate(", "physical_manifest_forward_capability("
    )
    kv_first = _between(source, "void emit_kv_first_body", "void build_layer_subgraph")
    layer = _between(source, "void build_layer_subgraph", "bool merger_active() const")
    merger = _between(source, "void merger_post_fn()", "void multiview_composite_post_fn()")

    sites = {
        "QWEN3VL_DEEPSTACK_SNAPSHOT_SITE": "559423039313734461LL",
        "QWEN3VL_POOLER_MERGED_COPY_SITE": "2748564172848473782LL",
        "QWEN3VL_DEEPSTACK_MERGED_COPY_SITE": "8851919130771715685LL",
    }
    for name, site_id in sites.items():
        assert site_id in source
        assert source.count(name) == 3

    assert "for (int64_t layer_idx : deepstack_visual_indexes_)" in manifest
    assert "for (const ChunkInfo& chunk : plan.compute.chunks)" in manifest
    assert "deepstack_visual_indexes_.size())" in manifest
    assert "layer_idx, chunk.offset, chunk.len, hidden_size()}" in manifest
    invocation = "layer_idx * num_compute_chunks + chunk.idx"
    assert invocation in manifest
    assert invocation in layer

    snapshot = _between(layer, "const auto deepstack_it", "const bool pipeline_final")
    assert "deepstack_visual_indexes_.size())" in snapshot
    assert "layer_idx, chunk.offset, seq_len, h}" in snapshot
    assert snapshot.index("consume_physical_route(") < snapshot.index(
        "rpu_launch_spm_copy_ddr_dma("
    )

    assert "!pipeline_dispatch_active() && !prefix_scatter_active_" in manifest
    assert "{mrows, oh, prefix_scatter_active_ ? 1 : 0}" in merger
    assert merger.count("consume_physical_route(") == 2
    assert merger.count("rpu_launch_spm_copy_ddr_dma(") == 2
    first_consume = merger.index("consume_physical_route(")
    first_launch = merger.index("rpu_launch_spm_copy_ddr_dma(")
    second_consume = merger.index("consume_physical_route(", first_consume + 1)
    second_launch = merger.index("rpu_launch_spm_copy_ddr_dma(", first_launch + 1)
    assert first_consume < first_launch < second_consume < second_launch

    assert "rope_spm ? 1 : 0" in manifest
    assert "FMB_ROUTE_FLAG_ROPE_TABLE_SPM" in manifest
    assert "FMB_ROUTE_FLAG_ROPE_TABLE_DDR" in manifest
    assert "QWEN3VL_Q_ROPE_SPM_PIPELINE_SITE" in manifest
    assert "QWEN3VL_K_ROPE_SPM_PIPELINE_SITE" in manifest
    assert "const std::vector<int64_t> rope_route_arguments" in kv_first
    assert "vision_attention_arguments(logical_len / image_batch_count_, chunk.len)" in manifest
    attention_args = _between(source, "std::vector<int64_t> vision_attention_arguments(",
                              "bool qwen3vl_2b_delivery_profile_active()")
    assert "{image_batch_count_, image_rows, chunk_rows," in attention_args
    assert "num_q_heads(), head_dim(), num_cores()}" in attention_args
    assert merger.count("prefix_scatter_active_ ? 1 : 0") >= 2
