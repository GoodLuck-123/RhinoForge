#pragma once
#include "rpu_ops.h"
#include "rpu_kernel_decls.h"
#include "rpu_spm_allocator.h"
#include "rpu_linear_tiling.h"

// Exact TP8 M51, K1024, globalN2048, uniform ACC16. Resident weight bytes are
// deinterleaved per-core FP4, never the W8 resident packing. Block scales stay DDR.
inline void launch_hyvla_gateup_nvfp4(
    uint32_t input, uint32_t gate, uint32_t up,
    uint32_t gate_spm, uint32_t up_spm, int64_t rows,
    const at::Tensor& gate_weight, const at::Tensor& up_weight,
    const at::Tensor& gate_scale, const at::Tensor& up_scale,
    uint32_t gate_ts, uint32_t up_ts, uint16_t layer, bool resident,
    bool input_reuse = false, bool overlap = false) {
    TORCH_CHECK(!overlap || (!resident && !input_reuse),
                "HyVLA Gate/Up raw-load overlap requires the retained DDR SwiGLU producer");
    TORCH_CHECK(rows == 51 && layer < 32 && SPM_ALLOC.is_initialized(),
                "HyVLA NVFP4 Gate/Up requires M51/L32 and initialized SPM");
    TORCH_CHECK(!resident || !input_reuse,
                "HyVLA NVFP4 input reuse is a DDR-only cold capability");
    for (auto* w : {&gate_weight, &up_weight})
        TORCH_CHECK(w->defined() && w->device().type() == c10::DeviceType::PrivateUse1 &&
                    w->scalar_type() == at::kByte && w->is_contiguous() &&
                    w->dim() == 2 && w->size(0) == 2048 && w->size(1) == 512,
                    "HyVLA NVFP4 requires packed Byte [2048,512] weights");
    for (auto* sc : {&gate_scale, &up_scale})
        TORCH_CHECK(sc->defined() && sc->device() == gate_weight.device() &&
                    sc->scalar_type() == at::kByte && sc->is_contiguous() &&
                    sc->dim() == 2 && sc->size(0) == 64 && sc->size(1) == 2048,
                    "HyVLA NVFP4 requires striped FP8 Byte [64,2048] scales");
    TORCH_CHECK(up_weight.device() == gate_weight.device(), "HyVLA NVFP4 device mismatch");
    std::vector<std::pair<uint32_t, uint64_t>> regions{
        {input, 51*1024*2}, {gate, 51*256*2}, {gate_ts, 32*sizeof(float)}, {up_ts, 32*sizeof(float)}};
    if (resident) {
        regions.insert(regions.end(), {{up, 51*256*2}, {gate_spm, 256*1024/2}, {up_spm, 256*1024/2}});
    } else {
        TORCH_CHECK(up == 0 && gate_spm == 0 && up_spm == 0,
                    "HyVLA NVFP4 DDR fusion has no Up/SPM-weight operand");
    }
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    for (size_t i = 0; i < regions.size(); ++i) {
        const uint64_t start = regions[i].first, bytes = regions[i].second;
        TORCH_CHECK(start >= base && start % 256 == 0 &&
                        start - base <= SpmAllocator::SPM_USABLE - bytes,
                    "HyVLA NVFP4 requires complete aligned absolute SPM regions");
        for (int core = 0; core < 8; ++core)
            TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core, 0)) + start - base + bytes - 1 <= UINT32_MAX,
                        "HyVLA NVFP4 absolute SPM range exceeds uint32");
        for (size_t j = 0; j < i; ++j)
            TORCH_CHECK(start + bytes <= regions[j].first ||
                            uint64_t(regions[j].first) + regions[j].second <= start,
                        "HyVLA NVFP4 requires disjoint SPM operands");
    }
    const KernelId id = overlap ? KernelId::HYVLA_NVFP4_GATEUP_SWIGLU_S51_OVERLAP
        : resident ? KernelId::HYVLA_NVFP4_GATEUP_RESIDENT_S51
        : (input_reuse ? KernelId::HYVLA_NVFP4_GATEUP_SWIGLU_INPUT_REUSE_S51
                       : KernelId::HYVLA_NVFP4_GATEUP_SWIGLU_S51);
    std::vector<GraphDdrRegisterOperandSpec> operands;
    auto add = [&](const at::Tensor& t, GraphDdrRegisterRole role, int lo) {
        const uint64_t address = rhino_lkn::RpuGetDevAddr(t.data_ptr());
        TORCH_CHECK(address != 0 && address % 256 == 0 && (address >> 8) <= UINT32_MAX &&
                        uint64_t(t.nbytes()) <= (uint64_t(1) << 40) - address,
                    "HyVLA NVFP4 requires aligned shift8 DDR operands");
        operands.push_back({GraphDdrRegisterAbi{role, GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi, uint16_t(lo), uint16_t(lo+1)}, t});
    };
    if (!resident) add(gate_weight, GraphDdrRegisterRole::PackedFp4ModelWeight, 2);
    add(gate_scale, GraphDdrRegisterRole::Fp8BlockScale, 20);
    if (!resident) add(up_weight, GraphDdrRegisterRole::PackedFp4ModelWeight, 32);
    add(up_scale, GraphDdrRegisterRole::Fp8BlockScale, 34);
    auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id, operands);
    for (auto* t : {&gate_weight, &up_weight, &gate_scale, &up_scale}) {
        rpu_ddr_flush(t->data_ptr());
        RpuKernelGraph::active().keep_alive(*t);
    }
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel, "HyVLA NVFP4 Gate/Up payload missing");
    kernel->reset_regs();
    rpu_pl_tiling::set_parallel_linear_nvfp4_regs(*kernel, {
        {input, 0, gate, 0, 51, 256, 1024, 0, 8, 2048, 512, 0, 1, 0},
        8*256*2, gate_ts, layer});
    rpu_pl_tiling::set_u32_reg_pair(*kernel, 36, up_ts);
    if (resident) {
        rpu_pl_tiling::set_u32_reg_pair(*kernel, 2, gate_spm);
        rpu_pl_tiling::set_u32_reg_pair(*kernel, 32, up_spm);
        rpu_pl_tiling::set_u32_reg_pair(*kernel, 38, up);
    } else {
        kernel->set_regs(uint32_t(128), uint32_t(0));
        kernel->set_regs(uint32_t(130), uint32_t(0));
    }
    kernel->set_regs(64, uint16_t(8));
    kernel->set_regs(65, uint16_t(1));
    kernel->set_regs(66, uint16_t(1));
    writer.write(*kernel);
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {8, 1, 1}, {0,1,2,3,4,5,6,7});
}
