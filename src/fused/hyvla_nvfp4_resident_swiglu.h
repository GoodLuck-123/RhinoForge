#pragma once
#include "rpu_helpers.h"
#include "rpu_kernel_decls.h"
#include "rpu_linear_tiling.h"

// Packed FP4 resident weights use the existing per-core [N16,K64,512B] ABI.
// The complete paired producer retires both FP16 outputs before grid1 SwiGLU.
inline void launch_hyvla_nvfp4_resident_swiglu_spm(
    uint32_t input, uint32_t gate, uint32_t up,
    uint32_t gate_weight, uint32_t up_weight,
    const at::Tensor& gate_scale, const at::Tensor& up_scale,
    uint32_t gate_ts, uint32_t up_ts, uint16_t layer,
    int64_t rows = 51, int cores = 8, bool overlap = false) {
    TORCH_CHECK(rows == 51 && cores == 8 && layer < 32 && SPM_ALLOC.is_initialized(),
                "HyVLA resident NVFP4 SwiGLU requires M51/K1024/N256/TP8/L32");
    const std::array<std::pair<uint32_t,uint64_t>,7> regions{{
        {input,51*1024*2}, {gate,51*256*2}, {up,51*256*2},
        {gate_weight,256*1024/2}, {up_weight,256*1024/2},
        {gate_ts,32*sizeof(float)}, {up_ts,32*sizeof(float)}}};
    const uint64_t base = SPM_ALLOC.addr(0,0);
    for (size_t i=0; i<regions.size(); ++i) {
        const uint64_t address=regions[i].first, bytes=regions[i].second;
        TORCH_CHECK(address >= base && address % 256 == 0 &&
                        address-base+bytes <= SpmAllocator::SPM_USABLE,
                    "HyVLA resident NVFP4 SwiGLU requires complete aligned SPM owners");
        for (int core=0; core<8; ++core)
            TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core,0))+address-base+bytes <= uint64_t(UINT32_MAX)+1,
                        "HyVLA resident NVFP4 SwiGLU SPM range overflows on a core");
        for (size_t j=0; j<i; ++j)
            TORCH_CHECK(address+bytes <= regions[j].first ||
                            uint64_t(regions[j].first)+regions[j].second <= address,
                        "HyVLA resident NVFP4 SwiGLU requires seven disjoint SPM owners");
    }
    for (const auto* scale : {&gate_scale,&up_scale}) {
        TORCH_CHECK(scale->defined() && scale->is_contiguous() &&
                        scale->device().type() == c10::DeviceType::PrivateUse1 &&
                        scale->scalar_type() == at::kByte && scale->dim() == 2 &&
                        scale->size(0) == 64 && scale->size(1) == 2048 &&
                        scale->device() == gate_scale.device(),
                    "HyVLA resident NVFP4 SwiGLU requires striped Byte FP8 [64,2048] scales");
        constexpr uint64_t limit=UINT64_C(1)<<40;
        const uint64_t address=rhino_lkn::RpuGetDevAddr(scale->data_ptr());
        TORCH_CHECK(address && address%256==0 && address<limit && scale->nbytes()<=limit-address,
                    "HyVLA resident NVFP4 SwiGLU requires complete aligned shift8 DDR scales");
    }
    const auto id = overlap ? KernelId::HYVLA_NVFP4_RESIDENT_SWIGLU_S51_OVERLAP
                           : KernelId::HYVLA_NVFP4_RESIDENT_SWIGLU_S51;
    auto* kernel=GET_KERNEL(id);
    TORCH_CHECK(kernel,"HyVLA resident NVFP4 SwiGLU payload unavailable");
    auto writer=RpuKernelGraph::active().stage_kernel_ddr_registers(id,{
        {GraphDdrRegisterAbi{GraphDdrRegisterRole::Fp8BlockScale,GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi,20,21},gate_scale},
        {GraphDdrRegisterAbi{GraphDdrRegisterRole::Fp8BlockScale,GraphDdrRegisterAccess::Read,
            GraphDdrRegisterEncoding::DevAddrShift8LoHi,34,35},up_scale}});
    for (const auto* scale : {&gate_scale,&up_scale}) {
        rpu_ddr_flush(scale->data_ptr());
        RpuKernelGraph::active().keep_alive(*scale);
    }
    kernel->reset_regs();
    rpu_pl_tiling::set_parallel_linear_nvfp4_regs(*kernel,{
        {input,0,gate,0,51,256,1024,0,8,2048,512,0,1,0},4096,gate_ts,layer});
    rpu_pl_tiling::set_u32_reg_pair(*kernel,2,gate_weight);
    rpu_pl_tiling::set_u32_reg_pair(*kernel,32,up_weight);
    rpu_pl_tiling::set_u32_reg_pair(*kernel,36,up_ts);
    rpu_pl_tiling::set_u32_reg_pair(*kernel,38,up);
    kernel->set_regs(48,uint16_t(32512));
    kernel->set_regs(49,uint16_t(13056));
    rpu_pl_tiling::set_u32_reg_pair(*kernel,52,gate);
    rpu_pl_tiling::set_u32_reg_pair(*kernel,54,up);
    rpu_pl_tiling::set_u32_reg_pair(*kernel,56,gate);
    kernel->set_regs(128,uint32_t{0});
    writer.write(*kernel);
    kernel->set_regs(64,uint16_t(8));
    kernel->set_regs(65,uint16_t(1));
    kernel->set_regs(66,uint16_t(1));
    auto* queue=GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel,{uint16_t(8),uint16_t(1),uint16_t(1)},
                        {0,1,2,3,4,5,6,7});
}
