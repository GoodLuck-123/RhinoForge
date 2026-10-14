#pragma once
#include "rpu_helpers.h"
#include "rpu_kernel_decls.h"
#include "rpu_linear_tiling.h"

// GU4/GU8 scale bundle [core,layer,Gate/Up,K32,N16,2,16], 32KiB/layer/core.
// Caller owns and preloads current layer slices; FP4 weights remain unchanged.
inline void launch_hyvla_nvfp4_resident_scales_spm(
    uint32_t input,uint32_t gate,uint32_t up,uint32_t gate_weight,uint32_t up_weight,
    uint32_t gate_scale,uint32_t up_scale,uint32_t gate_ts,uint32_t up_ts,
    uint16_t layer,bool swiglu=false,int64_t rows=51,int cores=8,int scale_layers=8) {
    TORCH_CHECK(rows==51 && cores==8 && (scale_layers==4 || scale_layers==8) &&
                    layer<scale_layers && SPM_ALLOC.is_initialized(),
                "HyVLA resident FP8 scales require M51/K1024/N256/TP8 and the first explicit four/eight layers");
    const std::array<std::pair<uint32_t,uint64_t>,9> regions{{
        {input,51*1024*2},{gate,51*256*2},{up,51*256*2},
        {gate_weight,131072},{up_weight,131072},{gate_scale,16384},{up_scale,16384},
        {gate_ts,32*sizeof(float)},{up_ts,32*sizeof(float)}}};
    const uint64_t base=SPM_ALLOC.addr(0,0);
    for(size_t i=0;i<regions.size();++i){
        const uint64_t address=regions[i].first,bytes=regions[i].second;
        TORCH_CHECK(address>=base && address%256==0 && address-base+bytes<=SpmAllocator::SPM_USABLE,
                    "HyVLA resident scales require complete aligned SPM owners");
        for(int core=0;core<8;++core)
            TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core,0))+address-base+bytes<=uint64_t(UINT32_MAX)+1,
                        "HyVLA resident scale SPM range overflows on a core");
        for(size_t j=0;j<i;++j)
            TORCH_CHECK(address+bytes<=regions[j].first ||
                            uint64_t(regions[j].first)+regions[j].second<=address,
                        "HyVLA resident scales require nine disjoint SPM owners");
    }
    const auto id=swiglu ? KernelId::HYVLA_NVFP4_RESIDENT_SCALES_SWIGLU_S51
                        : KernelId::HYVLA_NVFP4_RESIDENT_SCALES_S51;
    auto* kernel=GET_KERNEL(id);
    TORCH_CHECK(kernel,"HyVLA resident scale payload unavailable");
    kernel->reset_regs();
    rpu_pl_tiling::set_parallel_linear_nvfp4_regs(*kernel,{
        {input,0,gate,0,51,256,1024,0,8,2048,512,0,1,0},512,gate_ts,layer});
    rpu_pl_tiling::set_u32_reg_pair(*kernel,2,gate_weight);
    rpu_pl_tiling::set_u32_reg_pair(*kernel,32,up_weight);
    rpu_pl_tiling::set_u32_reg_pair(*kernel,20,gate_scale);
    rpu_pl_tiling::set_u32_reg_pair(*kernel,34,up_scale);
    rpu_pl_tiling::set_u32_reg_pair(*kernel,36,up_ts);
    rpu_pl_tiling::set_u32_reg_pair(*kernel,38,up);
    if(swiglu){
        kernel->set_regs(48,uint16_t(32512));
        kernel->set_regs(49,uint16_t(13056));
        rpu_pl_tiling::set_u32_reg_pair(*kernel,52,gate);
        rpu_pl_tiling::set_u32_reg_pair(*kernel,54,up);
        rpu_pl_tiling::set_u32_reg_pair(*kernel,56,gate);
        kernel->set_regs(128,uint32_t{0});
    }
    kernel->set_regs(64,uint16_t(8));
    kernel->set_regs(65,uint16_t(1));
    kernel->set_regs(66,uint16_t(1));
    auto* queue=GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel,{uint16_t(8),uint16_t(1),uint16_t(1)},
                        {0,1,2,3,4,5,6,7});
}
