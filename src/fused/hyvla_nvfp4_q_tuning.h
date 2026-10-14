#pragma once
#include "rpu_helpers.h"
#include "rpu_kernel_decls.h"
#include "rpu_linear_tiling.h"

// Q-only private experiment. false: canonical N32/grid8; true: N64/grid4
// with only the K-loop tail VMAT fence removed. Tensor-scale ABI is unchanged.
inline void launch_hyvla_nvfp4_q_tuning_spm(
    uint32_t input, const at::Tensor& weight, const at::Tensor& block_scale,
    uint32_t tensor_scale, uint32_t output, uint16_t layer, bool overlap,
    int64_t rows = 51, int cores = 8) {
    TORCH_CHECK(rows == 51 && cores == 8 && layer < 32 && SPM_ALLOC.is_initialized(),
                "HyVLA Q tuning requires M51/N2048/K1024/TP8/L32 and live SPM");
    TORCH_CHECK(weight.defined() && weight.device().type() == c10::DeviceType::PrivateUse1 &&
                    weight.scalar_type() == at::kByte && weight.is_contiguous() &&
                    weight.dim() == 2 && weight.size(0) == 2048 && weight.size(1) == 512,
                "HyVLA Q tuning requires packed FP4 Byte [2048,512]");
    TORCH_CHECK(block_scale.defined() && block_scale.device() == weight.device() &&
                    block_scale.scalar_type() == at::kByte && block_scale.is_contiguous() &&
                    block_scale.dim() == 2 && block_scale.size(0) == 64 && block_scale.size(1) == 2048,
                "HyVLA Q tuning requires striped FP8 Byte [64,2048] on the same device");
    const std::array<std::pair<uint32_t, uint64_t>, 3> regions{{
        {input, 51 * 1024 * 2}, {output, 51 * 256 * 2}, {tensor_scale, 32 * sizeof(float)}}};
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    for (size_t i = 0; i < regions.size(); ++i) {
        const uint64_t address = regions[i].first, bytes = regions[i].second;
        TORCH_CHECK(address >= base && address % 256 == 0 &&
                        address - base + bytes <= SpmAllocator::SPM_USABLE,
                    "HyVLA Q tuning requires complete aligned absolute SPM owners");
        for (int core = 0; core < 8; ++core)
            TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core, 0)) + address - base + bytes <=
                            uint64_t(UINT32_MAX) + 1,
                        "HyVLA Q tuning SPM address overflows on a core");
        for (size_t j = 0; j < i; ++j)
            TORCH_CHECK(address + bytes <= regions[j].first ||
                            uint64_t(regions[j].first) + regions[j].second <= address,
                        "HyVLA Q tuning requires disjoint input/output/tensor-scale owners");
    }
    const KernelId id = overlap ? KernelId::HYVLA_EXPERT_Q_NVFP4_S51_OVERLAP
                                : KernelId::HYVLA_EXPERT_Q_NVFP4_S51_N32;
    std::vector<GraphDdrRegisterOperandSpec> operands;
    for (const bool scale : {false, true}) {
        const auto& tensor = scale ? block_scale : weight;
        const uint64_t address = rhino_lkn::RpuGetDevAddr(tensor.data_ptr());
        constexpr uint64_t limit = UINT64_C(1) << 40;
        TORCH_CHECK(address != 0 && address % 256 == 0 && address < limit &&
                        uint64_t(tensor.nbytes()) <= limit - address,
                    "HyVLA Q tuning requires complete aligned shift8 DDR owners");
        const uint16_t lo = scale ? 20 : 2;
        operands.push_back({GraphDdrRegisterAbi{
            scale ? GraphDdrRegisterRole::Fp8BlockScale : GraphDdrRegisterRole::PackedFp4ModelWeight,
            GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,
            lo, uint16_t(lo + 1)}, tensor});
    }
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel, "HyVLA Q tuning payload unavailable");
    auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id, operands);
    for (const auto* tensor : {&weight, &block_scale}) {
        rpu_ddr_flush(tensor->data_ptr());
        RpuKernelGraph::active().keep_alive(*tensor);
    }
    kernel->reset_regs();
    rpu_pl_tiling::set_parallel_linear_nvfp4_regs(*kernel, {
        {input, 0, output, 0, 51, 256, 1024, 0, 8, 2048, 512, 0, 1, 0},
        4096, tensor_scale, layer});
    writer.write(*kernel);
    const uint16_t grid = overlap ? 4 : 8;
    kernel->set_regs(64, grid);
    kernel->set_regs(65, uint16_t(1));
    kernel->set_regs(66, uint16_t(1));
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {grid, uint16_t(1), uint16_t(1)},
                        {0, 1, 2, 3, 4, 5, 6, 7});
}
