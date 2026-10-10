#pragma once
#include "rpu_helpers.h"
#include "rpu_kernel_decls.h"
#include <array>
#include <cmath>
#include <cstring>

// Exact two-camera owner49 chain. Every root retains its original full M392
// capacity; compact raw uses only its prefix. The only legal root alias is
// partial == normalized, after the original reduce-scatter store/barrier.
inline void launch_hyvit2_vision_owner_spm(
    uint32_t partial, uint32_t residual, uint32_t raw, uint32_t normalized,
    uint32_t gamma, uint32_t beta, double eps, int64_t rows, bool compact) {
    TORCH_CHECK(rows == 392 && (compact || (std::isfinite(eps) && eps == 1e-6)) &&
                    SPM_ALLOC.is_initialized(),
                "HyViT2 owner LayerNorm requires M392/N1152/TP8 and epsilon 1e-6");
    constexpr uint64_t full = 392 * 1152 * 2, shard = full / 8, affine = 1152 * 2;
    const std::array<std::pair<uint32_t, uint64_t>, 6> ranges{{
        {partial, full}, {residual, full}, {raw, full},
        {normalized, full}, {gamma, affine}, {beta, affine}}};
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    for (size_t i = 0; i < (compact ? 3u : ranges.size()); ++i) {
        const uint64_t address = ranges[i].first, bytes = ranges[i].second;
        TORCH_CHECK(address >= base && address % 256 == 0 &&
                        address - base + bytes <= SpmAllocator::SPM_USABLE,
                    "HyViT2 owner LayerNorm requires complete aligned SPM roots");
        for (int core = 0; core < 8; ++core)
            TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core, 0)) + address - base + bytes <=
                            uint64_t(UINT32_MAX) + 1,
                        "HyViT2 owner LayerNorm SPM range overflows on a core");
        for (size_t j = 0; j < i; ++j) {
            const bool allowed = !compact && i == 3 && j == 0 && address == ranges[j].first;
            TORCH_CHECK(allowed || address + bytes <= ranges[j].first ||
                            uint64_t(ranges[j].first) + ranges[j].second <= address,
                        "HyViT2 owner LayerNorm requires disjoint full roots except partial == normalized");
        }
    }
    const auto id = compact ? KernelId::HYVIT2_VISION_COMPACT_RESIDUAL_M392N1152
                            : KernelId::HYVIT2_VISION_OWNER_LAYERNORM_M392N1152;
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel, "HyViT2 owner LayerNorm payload unavailable");
    auto& graph = RpuKernelGraph::active();
    if (graph.kernel_register_census_active())
        graph.stage_kernel_no_ddr(id, GraphKernelNoDdrProof::RingAllReduceSpmResidual);
    kernel->reset_regs();
    const auto set = [&](uint32_t index, uint16_t value) { kernel->set_regs(index, value); };
    const auto pair = [&](uint32_t index, uint32_t value) {
        set(index, uint16_t(value)); set(index + 1, uint16_t(value >> 16));
    };
    set(0, 8); set(1, 1); pair(2, 63488); pair(4, residual - base); set(6, 0);
    if (!compact) {
        pair(8, raw); pair(10, normalized); pair(12, gamma); pair(14, beta);
        set(16, 49); set(17, 1152);
        const float inverse = 1.0f / 1152;
        uint32_t bits; std::memcpy(&bits, &inverse, sizeof(bits)); pair(18, bits);
        set(20, c10::Half(static_cast<float>(eps) * 1152).x);
        set(21, 0); pair(22, 0); pair(24, affine); set(26, 1); set(27, 16);
        set(28, c10::Half(std::sqrt(1152.f)).x); set(29, 1);
    }
    set(42, 0); set(64, 8); set(65, 1); set(66, 1);
    // Ring uses the original SCU epoch bank; clear both halves for every launch.
    for (uint32_t i = 128; i <= 255; ++i) set(i, 0);
    for (uint32_t core = 0; core < 8; ++core) {
        const std::array<std::pair<uint32_t, uint32_t>, 5> fields{{
            {0, uint32_t(SPM_ALLOC.addr(core, partial - base) - base)},
            {16, uint32_t(SPM_ALLOC.addr(core, raw - base) - base)},
            {32, uint32_t(core * shard)}, {48, shard / 2},
            {64, compact ? 0u : uint32_t(SPM_ALLOC.addr(core, normalized - base) - base)}}};
        for (const auto& field : fields) {
            rpu_set_legacy_scm_u16_checked(kernel, 4096 + field.first + core,
                uint16_t(field.second), "HyViT2 owner LayerNorm");
            rpu_set_legacy_scm_u16_checked(kernel, 4096 + field.first + 8 + core,
                uint16_t(field.second >> 16), "HyViT2 owner LayerNorm");
        }
    }
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {8, 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}

inline void launch_hyvit2_vision_owner_layernorm_spm(
    uint32_t partial, uint32_t residual, uint32_t compact_raw, uint32_t normalized,
    uint32_t gamma, uint32_t beta, double eps, int64_t rows) {
    launch_hyvit2_vision_owner_spm(partial, residual, compact_raw, normalized, gamma, beta, eps, rows, false);
}

inline void launch_hyvit2_vision_compact_residual_spm(
    uint32_t partial, uint32_t compact_residual, uint32_t full_raw, int64_t rows) {
    launch_hyvit2_vision_owner_spm(partial, compact_residual, full_raw, 0, 0, 0, 0, rows, true);
}
