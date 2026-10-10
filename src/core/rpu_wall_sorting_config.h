#pragma once

#include <cstdint>

// Cold per-handle selection for the exact WRC v728 parcel-sorting profile.
// These values are copied into the owning model before weights/graphs exist;
// launchers receive the same immutable selection on BUILD and REPLAY.
struct WallSortingConfig {
    uint32_t fp16_transfer_mask = 0;  // Vision=1, Prefill=2, Action=4.
    bool fp16_mlp_m224 = false;
    bool silu_mul = false;
    bool action_activation_reuse = false;
    bool prefill_row_reuse = false;
    bool prefill_mlp_retile = false;
    bool paired_postnorm_tile40 = false;
    bool vision_row_reuse = false;
};

// The original sorting horizon remains admitted. Base's M32 transfer is cold
// opt-in through the existing activation-fusion selector; other horizons retain
// generated Linear. The row count is also part of the owning descriptor.
inline bool wall_action_fused_rows_supported(
    int64_t rows, const WallSortingConfig& sorting) {
    return rows == 20 || (rows == 32 && sorting.silu_mul);
}

// Row-aligned reduce-scatter: M20 -> 3/3/3/3/2/2/2/2, M32 -> 4/core.
inline constexpr uint32_t wall_action_residual_rows(int64_t rows, uint32_t core) {
    return static_cast<uint32_t>(rows / 8 + (core < rows % 8 ? 1 : 0));
}

inline constexpr uint32_t wall_action_residual_row_offset(
    int64_t rows, uint32_t core) {
    return static_cast<uint32_t>(core * (rows / 8) +
        (core < rows % 8 ? core : rows % 8));
}

// Exact base H32 NVFP4 buckets. Projection/ring geometry stays M32; the
// rotary launcher separately proves that its owned tables cover position+32.
// This does not widen the distinct paired-KV insertion capability.
inline constexpr bool wall_base_nvfp4_fused_prefix_supported(int64_t position) {
    return position == 336 || position == 384 || position == 448 ||
        position == 512 || position == 640;
}
