from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src" / "fused" / "rpu_lingbot_v2_moe_model.cpp"
SELECT = ROOT / "src" / "ops" / "rpu_moe_select.cpp"


def test_lingbot2_uses_strict_oplib_topk_and_dense_spm_scatter():
    model = MODEL.read_text()
    select = SELECT.read_text()

    assert "rpu_launch_backend_moe_select_fp16_spm(" in model
    assert "topk_by_select_fp16" in select
    assert "unary_cast_uint16_int32" in select
    assert "scatter_elements_v2_spm_smallC" in select
    assert "ValuOpType::GTE" not in model
    assert "exactly top_k" in select


def test_correction_bias_is_selection_only():
    select = SELECT.read_text()

    bias_add = select.index("correction_bias_addr, scores_token_major_addr")
    strict_topk = select.index("launch_topk_by_select_fp16(", bias_add)
    unbiased_gather = select.index(
        "scores_expert_major_addr, dense_mask_addr, dense_weights_addr",
        strict_topk,
    )
    assert bias_add < strict_topk < unbiased_gather


def test_expert_id_lookup_is_persistent_spm_state():
    model = MODEL.read_text()

    assert 'ids.name      = "moe_expert_scatter_ids"' in model
    assert "ids.storage   = StorageClass::Persistent" in model
    assert "expert_scatter_ids_.data_ptr<c10::Half>()" in model


def test_fp16_zero_sum_has_a_representable_denominator_guard():
    select = SELECT.read_text()

    reduction = select.index("for (rows /= 2; rows >= 1; rows /= 2)")
    guard = select.index("c10::Half(0x0001, c10::Half::from_bits())", reduction)
    divide = select.index("ValuOpType::DIV", guard)
    assert reduction < guard < divide
