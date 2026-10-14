from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _function(source: str, name: str) -> str:
    return source.split(name, 1)[1].split("\n}\n", 1)[0]


def test_hyvla_native_handles_declare_closed_chunk_envelopes() -> None:
    vlm = (ROOT / "src/fused/rpu_hyvla_vlm_model.cpp").read_text(encoding="utf-8")
    expert = (ROOT / "src/fused/rpu_hyvla_expert_model.cpp").read_text(
        encoding="utf-8"
    )

    vlm_set = _function(vlm, "void HyVlaVlmModel::set_moe_weights(")
    assert "chunk_size >= 16 && chunk_size <= 240 && chunk_size % 16 == 0" in vlm_set
    assert (
        "set_chunk_envelope(/*max_kv_len=*/chunk_size, /*chunk=*/chunk_size);"
        in vlm_set
    )
    assert vlm_set.index("set_chunk_envelope(") < vlm_set.index("moe_layers_.clear()")

    expert_set = _function(expert, "void HyVlaExpertModel::set_weights_expert(")
    assert "chunk_size == 51" in expert_set
    assert "set_chunk_envelope(/*max_kv_len=*/291, /*chunk=*/64);" in expert_set
    assert expert_set.index("set_chunk_envelope(") < expert_set.index(
        "CausalDecoderModel::set_weights("
    )

    for source, name in (
        (vlm, "bool HyVlaVlmModel::subclass_chunk_size_valid("),
        (expert, "bool HyVlaExpertModel::subclass_chunk_size_valid("),
    ):
        validity = _function(source, name)
        assert "cs >= seq_len && chunk_within_envelope(cs)" in validity

    declarations = vlm.split("HyVlaVlmModel::declare_buffers", 1)[1].split(
        "HyVlaVlmModel::build_layer_subgraph", 1
    )[0]
    assert "lctx.chunk_size >= moe_chunk_size_" not in declarations
    assert "const int64_t cs      = lctx.chunk_size" in declarations
    assert "chunk.len == moe_chunk_size_" in vlm
