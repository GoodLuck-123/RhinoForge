#pragma once
#include "rpu_helpers.h"

// Exact M51 ring partition: 6528 FP16 elements/core, paced shared ABI.
// Non-final layers overwrite the dead residual with the next input norm.
// The last layer normalizes the completed raw residual in place.
inline void launch_hyvla_down_nextnorm_spm(
    uint32_t partial, uint32_t residual_norm, uint32_t raw, uint32_t gamma,
    int64_t rows, int64_t cols, int64_t norm_rows, double eps, int cores, bool final_norm) {
    TORCH_CHECK(rows == 51 && cols == 1024 && norm_rows == 64 &&
                    eps == 1e-5 && cores == 8 && SPM_ALLOC.is_initialized(),
                "HyVLA Down nextnorm requires M51/H1024/R64/eps1e-5/TP8");
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint32_t, uint64_t>, 4> regions{{
        {partial, 51 * 1024 * 2}, {residual_norm, 64 * 1024 * 2},
        {raw, 64 * 1024 * 2}, {gamma, 1024 * 2}}};
    for (size_t i = 0; i < regions.size(); ++i) {
        const auto [address, bytes] = regions[i];
        TORCH_CHECK(address % 256 == 0 && address >= base &&
                        uint64_t(address) - base + bytes <= SpmAllocator::SPM_USABLE,
                    "HyVLA Down nextnorm requires complete aligned absolute SPM ranges");
        for (int core = 0; core < 8; ++core)
            TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core, 0)) + address - base + bytes <=
                            uint64_t(UINT32_MAX) + 1,
                        "HyVLA Down nextnorm SPM address overflows on a core");
        for (size_t j = 0; j < i; ++j)
            TORCH_CHECK(uint64_t(address) + bytes <= regions[j].first ||
                            uint64_t(regions[j].first) + regions[j].second <= address,
                        "HyVLA Down nextnorm requires four disjoint owners");
    }
    auto* kernel = RpuKernelGraph::active().get_kernel_reset("hyvla_expert_down_nextnorm_s51");
    TORCH_CHECK(kernel, "HyVLA Down nextnorm expansion kernel unavailable");
    kernel->reset_regs();
    auto u16 = [&](uint32_t r, uint32_t v) { kernel->set_regs(r, uint16_t(v)); };
    auto u32 = [&](uint32_t r, uint32_t v) { u16(r, v); u16(r + 1, v >> 16); };
    u16(0, 8); u16(1, 1); u32(2, 31 * 256 * 8);
    u32(4, residual_norm - base); u16(6, 0);
    u16(8, 1024); u16(9, c10::Half(1.f / 1024).x);
    u16(10, 2048); u16(11, 8);
    u32(12, raw); u32(14, final_norm ? raw : residual_norm); u32(16, gamma);
    u16(18, c10::Half(eps).x);
    u16(64, 8); u16(65, 1); u16(66, 1);
    kernel->set_regs(128, uint32_t{0});
    for (int core = 0; core < 8; ++core) {
        const uint32_t in = SPM_ALLOC.addr(core, uint32_t(partial - base)) - base;
        const uint32_t out = SPM_ALLOC.addr(core, uint32_t(raw - base)) - base;
        const std::array<uint32_t, 4> values{{in, out, uint32_t(core * 6528 * 2), 6528}};
        for (int i = 0; i < 4; ++i) {
            rpu_set_legacy_scm_u16_checked(kernel, 4096 + i * 16 + core,
                                          uint16_t(values[i]), "HyVLA Down nextnorm");
            rpu_set_legacy_scm_u16_checked(kernel, 4096 + i * 16 + 8 + core,
                                          uint16_t(values[i] >> 16), "HyVLA Down nextnorm");
        }
    }
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {uint16_t(8), uint16_t(1), uint16_t(1)},
                        {0, 1, 2, 3, 4, 5, 6, 7});
}
