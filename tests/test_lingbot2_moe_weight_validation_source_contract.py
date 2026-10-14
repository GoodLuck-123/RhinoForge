"""Board-free contracts for LingBot2 MoE weight-boundary validation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/fused/rpu_lingbot_v2_moe_model.cpp").read_text()


def _section(start: str, end: str) -> str:
    begin = SOURCE.index(start)
    return SOURCE[begin : SOURCE.index(end, begin)]


def test_unpacked_weights_reject_shape_extent_or_layout_drift():
    setter = _section(
        "void LingbotV2MoeExpertModel::set_moe_weights(",
        "// declare_buffers",
    )
    compact = " ".join(setter.split())

    for token in (
        "t.is_contiguous()",
        "t.sizes() == expected_shape",
        "t.numel() == expected_numel",
        "check_exact(router_gate_w[i], {num_experts, h}",
        "check_exact(router_bias[i], {num_experts}",
        "check_exact(expert_gate_w[i], {routed_inter, h}",
        "check_exact(expert_up_w[i], {routed_inter, h}",
        "check_exact(expert_down_w[i], {h, routed_inter}",
    ):
        assert token in compact

    assert setter.index("check_exact(router_gate_w[i]") < setter.index(
        "router_gate_.assign"
    )
    assert setter.index("check_exact(expert_down_w[i]") < setter.index(
        "expert_down_.assign"
    )


def test_packed_weights_reject_noncontiguous_views_before_retention():
    setter = _section(
        "void LingbotV2MoeExpertModel::set_packed_expert_weights(",
        "// build_layer_subgraph",
    )

    assert "t.is_contiguous()" in setter
    assert "must be contiguous fp16/int8" in setter
    assert setter.index("t.is_contiguous()") < setter.index("packed_gate_.assign")
