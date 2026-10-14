#pragma once
#include "rpu_helpers.h"

// Exact original Q2/K1 partial-MRoPE arithmetic; share only cos/sin VLM data.
inline void launch_hyvla_qk_rope_spm(
    uint32_t q, uint32_t q_out, uint32_t k, uint32_t k_out,
    const at::Tensor& cos, const at::Tensor& sin,
    int64_t rows, int64_t position, int cores) {
    TORCH_CHECK(rows == 51 && (position >= 128 && position <= 240 && position % 16 == 0) &&
                    cores == 8 && SPM_ALLOC.is_initialized(),
                "HyVLA QK RoPE requires S51/P128..240 aligned16/Q2/K1/D128/TP8");
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint32_t, uint64_t>, 4> regions{{
        {q, 51 * 256 * 2}, {q_out, 51 * 256 * 2},
        {k, 51 * 128 * 2}, {k_out, 51 * 128 * 2}}};
    for (size_t i = 0; i < regions.size(); ++i) {
        const auto [address, bytes] = regions[i];
        TORCH_CHECK(address % 256 == 0 && address >= base &&
                        uint64_t(address) - base + bytes <= SpmAllocator::SPM_USABLE,
                    "HyVLA QK RoPE requires complete aligned absolute SPM ranges");
        for (int core = 0; core < 8; ++core)
            TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core, 0)) + address - base + bytes <=
                            uint64_t(UINT32_MAX) + 1,
                        "HyVLA QK RoPE SPM address overflows on a core");
        for (size_t j = 0; j < i; ++j)
            TORCH_CHECK(uint64_t(address) + bytes <= regions[j].first ||
                            uint64_t(regions[j].first) + regions[j].second <= address,
                        "HyVLA QK RoPE requires four disjoint owners");
    }
    for (const auto* t : {&cos, &sin}) {
        TORCH_CHECK(t->defined() && t->is_contiguous() &&
                        t->device().type() == c10::DeviceType::PrivateUse1 &&
                        t->scalar_type() == at::kHalf && t->dim() == 2 &&
                        t->size(1) == 64 && t->size(0) >= position + rows,
                    "HyVLA QK RoPE requires FP16 device cos/sin [rows>=position+51,64]");
        const uint64_t address = rhino_lkn::RpuGetDevAddr(t->data_ptr());
        constexpr uint64_t limit = UINT64_C(1) << 40;
        TORCH_CHECK(address % 256 == 0 && address < limit && t->nbytes() <= limit - address,
                    "HyVLA QK RoPE requires complete encodable DDR ranges");
    }
    TORCH_CHECK(cos.size(0) == sin.size(0) && cos.device() == sin.device(),
                "HyVLA QK RoPE requires matching table owners");
    constexpr auto id = KernelId::HYVLA_EXPERT_QK_ROPE_S51;
    auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id, {
        {GraphDdrRegisterAbi{GraphDdrRegisterRole::SemanticInput,
            GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi, 0, 1}, cos},
        {GraphDdrRegisterAbi{GraphDdrRegisterRole::SemanticInput,
            GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, sin}});
    rpu_ddr_flush(cos.data_ptr()); rpu_ddr_flush(sin.data_ptr());
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel, "HyVLA QK RoPE expansion kernel unavailable");
    kernel->reset_regs();
    auto u16 = [&](uint32_t r, uint32_t v) { kernel->set_regs(r, uint16_t(v)); };
    auto u32 = [&](uint32_t r, uint32_t v) { u16(r, v); u16(r + 1, v >> 16); };
    writer.write(*kernel);
    u32(4, q); u32(6, q_out); u32(8, position);
    u16(10, 128); u16(11, 2); u16(12, rows); u16(13, 128); u16(14, 1);
    u16(15, 1); u32(16, k); u32(18, k_out);
    u16(64, 1); u16(65, 1); u16(66, 1);
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {uint16_t(1), uint16_t(1), uint16_t(1)},
                        {0, 1, 2, 3, 4, 5, 6, 7});
}
