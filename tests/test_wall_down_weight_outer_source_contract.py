"""Board-free contract for exact Wall P640 Down weight reuse."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CACHE_H = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
CACHE_INC = (ROOT / "src/core/rpu_kernel_cache.inc").read_text()
DECLS = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
LINEAR = (ROOT / "src/ops/rpu_linear.cpp").read_text()
DECODER = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
RUNTIME = (ROOT / "src/core/rpu_runtime_extras.cpp").read_text()
SYMBOL = "wall_down_weight_outer_w8a16_c320x2_n64k128"
LAUNCHER = "rpu_launch_wall_prefill_down_w8a16_c320x2_spm_kernel("


def test_fixed_down_launcher_preserves_typed_ddr_and_multicore_abi():
    assert "WALL_DOWN_WEIGHT_OUTER_W8A16_C320X2_N64" in CACHE_H
    assert "KernelId::WALL_DOWN_WEIGHT_OUTER_W8A16_C320X2_N64" in CACHE_INC
    assert f'"{SYMBOL}"' in CACHE_H
    assert f'"{SYMBOL}"' in CACHE_INC
    assert LAUNCHER in DECLS

    start = LINEAR.index(f"void {LAUNCHER}")
    launcher = LINEAR[
        start : LINEAR.index(
            "void rpu_launch_wall_prefill_oproj_w8a16_c320x2_spm_kernel(",
            start,
        )
    ]
    for geometry in (
        "constexpr int64_t kM = 640;",
        "constexpr int64_t kN = 2048;",
        "constexpr int64_t kK = 11008;",
        "constexpr int64_t kLocalK = kK / NUM_CORES;",
    ):
        assert geometry in launcher
    assert launcher.count("GraphDdrRegisterOperandSpec{") == 2
    assert launcher.count("GraphDdrRegisterRole::Int8ModelWeight") == 1
    assert launcher.count("GraphDdrRegisterRole::PerChannelScale") == 1
    assert "DevAddrShift8LoHi, 2, 3" in launcher
    assert "DevAddrShift8LoHi, 20, 21" in launcher
    assert "kernel->set_regs(24, static_cast<uint16_t>(output1_spm_addr));" in launcher
    assert "kernel->set_regs(25, static_cast<uint16_t>(output1_spm_addr >> 16));" in launcher
    assert "queue->set_broadcast_mode(true);" in launcher
    assert "queue->enqueu_kernel(*kernel, {32, 1, 1}, cores);" in launcher
    assert launcher.index("register_writer.write(*kernel);") < launcher.index(
        "queue->enqueu_kernel("
    )


def test_staged_schedule_reuses_down_only_after_the_bf16_transition():
    start = DECODER.index("void build_wall_p640_staged_layer_subgraph(")
    staged = DECODER[start : DECODER.index("void build_layer_subgraph(", start)]
    assert "const bool down_weight_outer = layer_idx >= 3;" in staged
    assert (
        "const bool paired_down_postnorm =\n"
        "            down_weight_outer && residual_is_bf16;"
    ) in staged
    assert staged.count(LAUNCHER) == 1
    assert (
        'paired_down_postnorm\n'
        '                    ? addr(0, "residual1") : addr(0, "down")'
    ) in staged
    assert 'addr(0, "residual1") + kHalfHiddenBytes, lw.down_ws, wall_sorting_);' in staged
    paired = staged[
        staged.index("if (paired_down_postnorm) {") :
        staged.index("} else {", staged.index("if (paired_down_postnorm) {"))
    ]
    assert paired.count(
        "rpu_launch_wall_ring_postnorm_pair_fp16_partial_bf16_residual_local_spm_kernel("
    ) == 1
    assert (
        'addr(0, "residual1"), addr(0, "residual_local"),\n'
        '                addr(0, "input_norm"), next_norm_w, eps_, wall_sorting_.paired_postnorm_tile40);'
    ) in paired
    assert "if (!down_weight_outer)" in staged
    assert "down_weight_outer && half == 1" in staged
    assert "partial_spm, raw_residual_local_spm," in staged
    assert "const bool transition_to_bf16 = layer_idx == 2;" in staged
    assert "const bool residual_is_bf16 = layer_idx > 2;" in staged
    assert staged.index("if (transition_to_bf16) {") < staged.index(
        "const bool down_weight_outer = layer_idx >= 3;"
    )


def test_text_carrier_alias_is_narrow_and_range_proven():
    start = RUNTIME.index("void rpu_launch_wall_oss_mixed_ring_allreduce(")
    mixed = RUNTIME[start : RUNTIME.index("void launch_wall_down_postnorm_m576(", start)]
    exact = mixed[
        mixed.index("const bool exact_m320_local_postnorm_inplace") :
        mixed.index("TORCH_CHECK(\n        exact_m576_postnorm_id", mixed.index(
            "const bool exact_m320_local_postnorm_inplace"
        ))
    ]
    for guard in (
        "exact_m576_postnorm_id == KernelId::_COUNT",
        "fused_postnorm",
        "M == 320 && N == 2048",
        "!partial_is_bf16",
        "residual_ddr == nullptr",
        "residual_local_shard",
        "input_spm == output_spm",
        "postnorm->raw_residual_local_spm == residual_spm",
    ):
        assert guard in exact
    range_gate = mixed[
        mixed.index("} else if (exact_m320_local_postnorm_inplace) {") :
        mixed.index("} else if (residual_local_shard) {", mixed.index(
            "} else if (exact_m320_local_postnorm_inplace) {"
        ))
    ]
    assert range_gate.count("in_current_spm(") == 3
    assert range_gate.count("disjoint(") == 3
    assert "input_spm != output_spm ||\n                     exact_m320_local_postnorm_inplace" in mixed
    assert "exact_m320_local_postnorm_inplace ||\n                ring_all_reduce_sum_residual_admitted(" in mixed
    assert (
        "!exact_m320_local_postnorm_inplace ||\n"
        "        id == KernelId::WALL_RING_POSTNORM_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_INOUT_M320N2048"
        in mixed
    )
