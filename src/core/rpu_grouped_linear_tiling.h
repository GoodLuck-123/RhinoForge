#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <tuple>

// Mirrors rhino-ops ca2d1c84 generator/grouped_parallel_linear_tiling.py.
// M is the total routed rows, not the runtime-only row count of one expert.
namespace rpu_grouped_tiling {

// rhino-ops 135b93fc grouped UINT4 metadata:
// [expert,Qblock4,Nblock64,core,qslot4,n64], FP16 S and negative zero.
// Weight nibble packing and the measured tile selector are unchanged.
struct Uint4MetadataLayout {
    static constexpr int64_t version = 1;
    int64_t q_blocks, n_blocks, alignment_bytes;
    int64_t group_block_bytes, expert_bytes;
};

inline Uint4MetadataLayout uint4_metadata_layout(
    int64_t local_n, int64_t local_k, int64_t group_size, int64_t cores) {
    if (local_n <= 0 || local_k <= 0 || cores < 1 || cores > 8 ||
        (group_size != 32 && group_size != 64 && group_size != 128) ||
        local_k % group_size)
        throw std::invalid_argument("invalid grouped UINT4 metadata geometry");
    const int64_t q_blocks = (local_k / group_size + 3) / 4;
    const int64_t n_blocks = (local_n + 63) / 64;
    const int64_t alignment = cores * 512;
    const int64_t group_block_bytes = n_blocks * alignment;
    return {q_blocks, n_blocks, alignment, group_block_bytes,
            q_blocks * group_block_bytes};
}

struct Tile {
    int64_t n;
    int64_t m;
};

inline int64_t acc16_m_tile(int64_t n) {
    switch (n) {
        case 32: return 288;
        case 48: return 256;
        case 64: return 224;
        case 80: return 192;
        case 96: return 160;
        case 112: return 144;
        case 128: return 128;
        default: throw std::invalid_argument("unsupported grouped UINT4 N tile");
    }
}

inline Tile select_uint4_acc16(int64_t m, int64_t experts, int64_t n, int64_t k) {
    if (m <= 0 || experts <= 0 || n <= 0 || k <= 0)
        throw std::invalid_argument("GroupedParallelLinear dimensions must be positive");
    struct Measurement { int64_t m, experts, n, k, tile; };
    static constexpr std::array<Measurement, 11> measurements{{
        {8, 128, 96, 2048, 32}, {8, 128, 128, 2048, 32},
        {8, 128, 2048, 128, 128}, {512, 128, 96, 2048, 32},
        {512, 128, 128, 2048, 48}, {512, 128, 2048, 128, 128},
        {1024, 128, 96, 2048, 32}, {1024, 128, 128, 2048, 48},
        {1024, 128, 2048, 128, 128}, {204, 32, 64, 768, 32},
        {204, 32, 768, 64, 64},
    }};
    const int64_t active = std::min(m, experts);
    const double rows_per_expert = static_cast<double>(m) / active;
    std::array<std::pair<double, int64_t>, measurements.size()> neighbours;
    for (size_t i = 0; i < measurements.size(); ++i) {
        const auto& measured = measurements[i];
        if (std::tie(m, experts, n, k) ==
            std::tie(measured.m, measured.experts, measured.n, measured.k))
            return {measured.tile, acc16_m_tile(measured.tile)};
        const int64_t measured_active = std::min(measured.m, measured.experts);
        const double measured_rows = static_cast<double>(measured.m) / measured_active;
        const double dm = std::log2(rows_per_expert / measured_rows);
        const double dg = 0.5 * std::log2(static_cast<double>(active) / measured_active);
        const double dn = 4.0 * std::log2(static_cast<double>(n) / measured.n);
        const double dk = 0.5 * std::log2(static_cast<double>(k) / measured.k);
        neighbours[i] = {dm * dm + dg * dg + dn * dn + dk * dk, measured.tile};
    }
    std::sort(neighbours.begin(), neighbours.end());
    static constexpr std::array<int64_t, 4> candidates{{32, 48, 64, 128}};
    std::array<double, candidates.size()> votes{};
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < candidates.size(); ++j)
            if (candidates[j] == neighbours[i].second)
                votes[j] += 1.0 / (neighbours[i].first + 0.01);
    const auto key = [&](size_t i) {
        const int64_t tile = candidates[i];
        return std::make_tuple(
            -votes[i], std::ceil(std::ceil(rows_per_expert) / acc16_m_tile(tile)),
            std::ceil(static_cast<double>(n) / tile) * tile, tile);
    };
    size_t best = 0;
    for (size_t i = 1; i < candidates.size(); ++i)
        if (key(i) < key(best)) best = i;
    return {candidates[best], acc16_m_tile(candidates[best])};
}

}  // namespace rpu_grouped_tiling
