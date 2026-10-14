#pragma once
#include "rpu_helpers.h"
#include "ops/rpu_kvinsert_segment_plan.h"

// Exact R15 Q2/K1 RoPE, then original BF16-bitcopy V16 head norms.
// Q/K storage is overwritten only after both RoPE stores finish; padding is normalized.
// A third phase barrier precedes four original S64 writers; their mutex is counter3.
inline void launch_hyvla_qk_rope_norm_kv_spm(
    uint32_t q, uint32_t q_out, uint32_t k, uint32_t k_out,
    uint32_t q_gamma, uint32_t k_gamma, uint32_t v,
    const at::Tensor& cos, const at::Tensor& sin,
    at::Tensor& k_cache, at::Tensor& v_cache,
    int64_t rows, int64_t position, int64_t q_norm_rows, int64_t k_norm_rows,
    double eps, int cores, const KvInsertSegmentPlan& plan) {
    TORCH_CHECK(rows == 51 && (position >= 128 && position <= 240 && position % 16 == 0) &&
                    q_norm_rows == 112 && k_norm_rows == 64 &&
                    eps == 1e-5 && cores == 8 && SPM_ALLOC.is_initialized(),
                "HyVLA QK RoPE requires S51/P128..240 aligned16/Q2/K1/D128/Q112/K64/eps1e-5/TP8");
    rpu_validate_kvinsert_segment_plan(plan, 8, 8, 128);
    TORCH_CHECK(plan.route() == KvInsertRoute::PAD16_V16 &&
                    plan.logical_rows() == rows && plan.physical_rows() == k_norm_rows &&
                    plan.segment_count() == 1,
                "HyVLA QK norm+KV requires the actual S51/S64 PAD16_V16 plan");
    const auto& segment = plan.segment(0);
    TORCH_CHECK(segment.position == position && segment.rows == 64 &&
                    segment.token_offset == 0 && segment.kernel == KvInsertKernel::V16,
                "HyVLA QK norm+KV requires an actual-position full64-row writer");
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint32_t, uint64_t>, 7> regions{{
        {q, 112 * 128 * 2}, {q_out, 112 * 128 * 2},
        {k, 64 * 128 * 2}, {k_out, 64 * 128 * 2},
        {q_gamma, 128 * 2}, {k_gamma, 128 * 2}, {v, 64 * 128 * 2}}};
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
                        "HyVLA QK RoPE requires seven disjoint owners");
    }
    for (const auto* t : {&cos, &sin}) {
        TORCH_CHECK(t->defined() && t->is_contiguous() &&
                        t->device().type() == c10::DeviceType::PrivateUse1 &&
                        t->scalar_type() == at::kHalf && t->dim() == 2 &&
                        t->size(1) == 64 && t->size(0) >= position + rows,
                    "HyVLA QK RoPE requires FP16 device cos/sin [rows>=position+51,64]");
        const uint64_t address = rhino_lkn::RpuGetDevAddr(t->data_ptr());
        constexpr uint64_t limit = UINT64_C(1) << 40;
        TORCH_CHECK(address != 0 && address % 256 == 0 && address < limit && t->nbytes() <= limit - address,
                    "HyVLA QK RoPE requires complete encodable DDR ranges");
    }
    TORCH_CHECK(cos.size(0) == sin.size(0) && cos.device() == sin.device(),
                "HyVLA QK RoPE requires matching table owners");
    for (const auto* cache : {&k_cache, &v_cache}) {
        TORCH_CHECK(cache->defined() && cache->is_contiguous() &&
                        cache->device() == cos.device() && cache->scalar_type() == at::kHalf &&
                        cache->dim() == 7 && cache->size(0) == 1 &&
                        cache->size(1) >= (position + 64 + 15) / 16 &&
                        cache->size(2) == 1 && cache->size(3) == 8 && cache->size(4) == 8 &&
                        cache->size(5) == 16 && cache->size(6) == 16,
                    "HyVLA QK norm+KV requires owned FP16 [1,ceil((position+64)/16),1,8,8,16,16] caches or larger");
        constexpr uint64_t limit = UINT64_C(1) << 40;
        const uint64_t address = rhino_lkn::RpuGetDevAddr(cache->data_ptr());
        TORCH_CHECK(address != 0 && address % 256 == 0 && address < limit &&
                        cache->nbytes() <= limit - address,
                    "HyVLA QK norm+KV requires complete encodable cache DDR ranges");
    }
    TORCH_CHECK(k_cache.size(1) == v_cache.size(1), "HyVLA QK norm+KV cache capacities differ");
    const std::array<const at::Tensor*, 4> ddr{&cos, &sin, &k_cache, &v_cache};
    // Read-only tables may alias each other; no writable cache may overlap any owner.
    for (size_t i = 2; i < ddr.size(); ++i) for (size_t j = 0; j < i; ++j) {
        const uint64_t a = rhino_lkn::RpuGetDevAddr(ddr[i]->data_ptr());
        const uint64_t b = rhino_lkn::RpuGetDevAddr(ddr[j]->data_ptr());
        TORCH_CHECK(a + ddr[i]->nbytes() <= b || b + ddr[j]->nbytes() <= a,
                    "HyVLA QK norm+KV requires independent cache/table DDR owners");
    }
    constexpr auto id = KernelId::HYVLA_EXPERT_QK_ROPE_NORM_KV_S51;
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel, "HyVLA QK RoPE+norm expansion kernel unavailable");
    auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id, {
        {GraphDdrRegisterAbi{GraphDdrRegisterRole::SemanticInput,
            GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi, 0, 1}, cos},
        {GraphDdrRegisterAbi{GraphDdrRegisterRole::SemanticInput,
            GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, sin},
        {GraphDdrRegisterAbi{GraphDdrRegisterRole::KeyCache,
            GraphDdrRegisterAccess::Write, GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, k_cache},
        {GraphDdrRegisterAbi{GraphDdrRegisterRole::ValueCache,
            GraphDdrRegisterAccess::Write, GraphDdrRegisterEncoding::DevAddrShift8LoHi, 24, 25}, v_cache}});
    for (auto* tensor : ddr) {
        rpu_ddr_flush(tensor->data_ptr());
        RpuKernelGraph::active().keep_alive(*tensor);
    }
    kernel->reset_regs();
    auto u16 = [&](uint32_t r, uint32_t v) { kernel->set_regs(r, uint16_t(v)); };
    auto u32 = [&](uint32_t r, uint32_t v) { u16(r, v); u16(r + 1, v >> 16); };
    writer.write(*kernel);
    u32(4, q); u32(6, q_out); u32(8, position);
    u16(10, 128); u16(11, 2); u16(12, rows); u16(13, 128); u16(14, 1);
    u16(15, 1); u32(16, k); u32(18, k_out);
    auto norm = [&](uint32_t first, uint32_t in, uint32_t out, uint32_t gamma, uint16_t per_block) {
        u16(first, 128); u16(first + 1, c10::Half(1.f / 128).x);
        u16(first + 2, 256); u16(first + 3, per_block);
        u32(first + 4, in); u32(first + 6, out); u32(first + 8, gamma);
        u16(first + 10, c10::Half(eps).x);
    };
    norm(32, q_out, q, q_gamma, 14);
    norm(44, k_out, k, k_gamma, 8);
    // Writer reuses position8/9, heads14, dimension10 and normalized K16/17.
    // Typed cache20/21 and24/25 must remain inside the SDK launch ABI0..63.
    u32(22, v); u16(26, 8);
    kernel->set_regs(132, uint32_t{0});
    kernel->set_regs(134, uint32_t{0});
    kernel->set_regs(128, uint32_t{0});
    kernel->set_regs(130, uint32_t{0});
    u16(64, 8); u16(65, 1); u16(66, 1);
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {uint16_t(8), uint16_t(1), uint16_t(1)},
                        {0, 1, 2, 3, 4, 5, 6, 7});
    rpu_ddr_flush(k_cache.data_ptr());
    rpu_ddr_flush(v_cache.data_ptr());
}
