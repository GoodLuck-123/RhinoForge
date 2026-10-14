#pragma once

// Exact task-private ABI: original selected reduction and two independent gammas.
inline void launch_hyvla_paired_rmsnorm_spm(
    uint32_t input, uint32_t text_out, uint32_t act_out,
    uint32_t text_gamma, uint32_t act_gamma,
    int64_t rows, int64_t cols, double eps, int cores,
    RpuRmsNormSpmRoute route = RpuRmsNormSpmRoute::V16_GRID8) {
    TORCH_CHECK((rows >= 128 && rows <= 240 && rows % 16 == 0) &&
                    cols == 2048 && eps == 1e-5 && cores == 8 &&
                    SPM_ALLOC.is_initialized(),
                "HyVLA paired RMSNorm requires S128..240 aligned16/H2048/eps1e-5/TP8 and live SPM");
    TORCH_CHECK(route == RpuRmsNormSpmRoute::V16_GRID8 ||
                    (route == RpuRmsNormSpmRoute::V32_GRID8 && rows % 32 == 0),
                "HyVLA paired RMSNorm requires the original V16 or 32-aligned V32 route");
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint32_t, uint64_t>, 5> regions{{
        {input, rows * cols * 2}, {text_out, rows * cols * 2},
        {act_out, rows * cols * 2}, {text_gamma, cols * 2}, {act_gamma, cols * 2}}};
    for (size_t i = 0; i < regions.size(); ++i) {
        const auto [address, bytes] = regions[i];
        TORCH_CHECK(address % 256 == 0 && address >= base &&
                        uint64_t(address) - base + bytes <= SpmAllocator::SPM_USABLE,
                    "HyVLA paired RMSNorm requires complete aligned absolute SPM ranges");
        for (int core = 0; core < 8; ++core) {
            TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core, 0)) + address - base + bytes <=
                            uint64_t(UINT32_MAX) + 1,
                        "HyVLA paired RMSNorm SPM address overflows on a core");
        }
        for (size_t j = 0; j < i; ++j) {
            TORCH_CHECK(uint64_t(address) + bytes <= regions[j].first ||
                            uint64_t(regions[j].first) + regions[j].second <= address,
                        "HyVLA paired RMSNorm requires five disjoint owners");
        }
    }
    auto* kernel = RpuKernelGraph::active().get_kernel_reset(
        route == RpuRmsNormSpmRoute::V32_GRID8
            ? "hyvla_mot_paired_rmsnorm_v32" : "hyvla_mot_paired_rmsnorm_s240");
    TORCH_CHECK(kernel, "HyVLA paired RMSNorm expansion kernel unavailable");
    kernel->reset_regs();
    kernel->set_regs(0, uint16_t(cols));
    kernel->set_regs(1, c10::Half(1.0f / cols).x);
    kernel->set_regs(2, uint16_t(cols * 2));
    kernel->set_regs(3, uint16_t(rows / 8));
    kernel->set_regs(10, c10::Half(eps).x);
    for (auto pair : {std::pair<uint32_t, uint32_t>{4, input}, {6, text_out},
                      {8, text_gamma}, {12, act_out}, {14, act_gamma}}) {
        kernel->set_regs(pair.first, uint16_t(pair.second));
        kernel->set_regs(pair.first + 1, uint16_t(pair.second >> 16));
    }
    auto* queue = GET_QUEUE(cores);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {uint16_t(8), uint16_t(1), uint16_t(1)},
                        {0, 1, 2, 3, 4, 5, 6, 7});
}
