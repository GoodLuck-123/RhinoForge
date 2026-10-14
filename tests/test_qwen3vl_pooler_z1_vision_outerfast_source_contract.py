from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp").read_text()
HEADER = (ROOT / "src/fused/rpu_qwen3vl_pooler_spm_z1.h").read_text()


def _between(source: str, begin: str, end: str) -> str:
    start = source.index(begin)
    return source[start : source.index(end, start)]


def test_vision_z1_outer_fast_component_publishes_only_after_full_validation():
    stage = _between(
        SOURCE,
        "void stage_pooler_z1_outer_fast_component(",
        "void bind_pooler_z1_outer_fast_input(",
    )
    publish = stage.index("stage_spm_outer_fast_component(")

    assert stage.index("validate_pooler_z1_contract();") < publish
    for token in (
        "z1_adopted_",
        "z1_bound_",
        "!z1_dispatch_",
        "z1_inputs_primed_",
        "z1_source_addr_ != 0",
        "retained_deepstack_bindings_complete()",
        "z1_epoch_ != 0",
        "z1_plan_hash_ != 0",
        "z1_primed_position_ptr_",
        "position_idx_keepalive_.data_ptr<int16_t>()",
        "z1_primed_position_owner_",
        "position_idx_keepalive_.unsafeGetTensorImpl()",
    ):
        assert token in stage[:publish]
    assert "guard, k_caches, v_caches" in stage[publish:]


def test_vision_z1_outer_fast_input_is_exact_before_mutable_binding():
    bind = _between(
        SOURCE,
        "void bind_pooler_z1_outer_fast_input(",
        "void forward_pooler_z1(",
    )
    publish = bind.index("bind_spm_outer_fast_input(")

    assert bind.index("validate_pooler_z1_contract();") < publish
    for token in (
        "z1_adopted_",
        "z1_bound_",
        "retained_deepstack_bindings_complete()",
        "!z1_dispatch_",
        "input.dim() == 3",
        "input.size(0) == 1",
        "input.size(1) == QWEN3VL_POOLER_Z1_NUM_PATCHES",
        "input.size(2) == QWEN3VL_POOLER_Z1_VISION_HIDDEN",
        "input.scalar_type() == at::kHalf",
        "input.device().type() == at::kPrivateUse1",
        "input.is_contiguous()",
    ):
        assert token in bind[:publish]


def test_vision_z1_outer_fast_internal_facade_is_declared_and_forwarded():
    for name in (
        "stage_vision_outer_fast_component",
        "bind_vision_outer_fast_input",
    ):
        assert name in HEADER
        assert name in SOURCE
    wrappers = SOURCE[SOURCE.index("namespace v3::qwen3vl_pooler_z1_internal {") :]
    assert "->stage_pooler_z1_outer_fast_component(" in wrappers
    assert "->bind_pooler_z1_outer_fast_input(guard, input);" in wrappers


def test_vision_retained_deepstack_facade_is_counted_typed_and_keeps_ds1_wrappers():
    assert "int64_t retained_deepstack_count = 0" in HEADER
    for name in (
        "vision_deepstack_source_spec",
        "bind_vision_deepstack_source",
        "vision_deepstack1_source_spec",
        "bind_vision_deepstack1_source",
    ):
        assert name in HEADER
        assert name in SOURCE
    wrappers = SOURCE[SOURCE.index("namespace v3::qwen3vl_pooler_z1_internal {") :]
    assert "->prepare_pooler_z1_layout(num_patches, retained_count, selected_descriptor);" in wrappers
    assert "->pooler_z1_deepstack_source_spec(num_patches, ordinal);" in wrappers
    assert "->bind_pooler_z1_deepstack_source(lease, source, ordinal);" in wrappers
    assert "return vision_deepstack_source_spec(" in wrappers
    assert "handle, num_patches, /*ordinal=*/0" in wrappers
    assert "bind_vision_deepstack_source(" in wrappers
    assert "handle, lease, source, /*ordinal=*/0" in wrappers
