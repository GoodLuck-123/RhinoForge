#pragma once
#include "rpu_helpers.h"
#include "ops/rpu_linear_tiling.h"

// Exact Action Down M51/N1024/globalK2048/TP8, uniform NVFP4 ACC16.
// The cold native owner admits this optional N128 payload and owns/preloads
// the complete FP32[32] tensor-scale table. No collective or norm is fused.
inline void launch_hyvla_nvfp4_down_n128_spm(
    uint32_t input, const at::Tensor& weight, const at::Tensor& scale,
    uint32_t tensor_scale_spm, uint16_t layer, uint32_t output,
    int64_t rows, int64_t cols, int cores) {
    TORCH_CHECK(rows == 51 && cols == 1024 && cores == 8 && layer < 32 &&
                    SPM_ALLOC.is_initialized(),
                "HyVLA NVFP4 Down N128 requires M51/N1024/TP8/L32 and initialized SPM");
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    const std::array<std::pair<uint32_t, uint64_t>, 3> regions{{
        {input, 51 * 256 * 2}, {output, 51 * 1024 * 2},
        {tensor_scale_spm, 32 * sizeof(float)}}};
    for (size_t i = 0; i < regions.size(); ++i) {
        const auto [address, bytes] = regions[i];
        TORCH_CHECK(address % 256 == 0 && address >= base &&
                        uint64_t(address) - base + bytes <= SpmAllocator::SPM_USABLE,
                    "HyVLA NVFP4 Down requires complete aligned absolute SPM ranges");
        for (int core = 0; core < 8; ++core)
            TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core, 0)) + address - base + bytes <=
                            uint64_t(UINT32_MAX) + 1,
                        "HyVLA NVFP4 Down SPM address overflows on a core");
        for (size_t j = 0; j < i; ++j)
            TORCH_CHECK(uint64_t(address) + bytes <= regions[j].first ||
                            uint64_t(regions[j].first) + regions[j].second <= address,
                        "HyVLA NVFP4 Down requires three disjoint SPM owners");
    }
    TORCH_CHECK(weight.defined() && weight.is_contiguous() &&
                    weight.device().type() == c10::DeviceType::PrivateUse1 &&
                    weight.scalar_type() == at::kByte && weight.dim() == 2 &&
                    weight.size(0) == 1024 && weight.size(1) == 1024,
                "HyVLA NVFP4 Down requires packed device Byte weight [1024,1024]");
    TORCH_CHECK(scale.defined() && scale.is_contiguous() &&
                    scale.device() == weight.device() && scale.scalar_type() == at::kByte &&
                    scale.dim() == 2 && scale.size(0) == 128 && scale.size(1) == 1024,
                "HyVLA NVFP4 Down requires striped FP8 Byte scale [128,1024]");
    constexpr uint64_t limit = UINT64_C(1) << 40;
    for (auto* tensor : {&weight, &scale}) {
        const uint64_t address = rhino_lkn::RpuGetDevAddr(tensor->data_ptr());
        TORCH_CHECK(address != 0 && address % 256 == 0 && address < limit &&
                        tensor->nbytes() <= limit - address,
                    "HyVLA NVFP4 Down requires complete encodable shift8 DDR ranges");
    }
    constexpr auto id = KernelId::HYVLA_EXPERT_DOWN_NVFP4_S51_N128;
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel, "HyVLA NVFP4 Down N128 payload unavailable");
    auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id, {
        {GraphDdrRegisterAbi{GraphDdrRegisterRole::PackedFp4ModelWeight,
            GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi, 2, 3}, weight},
        {GraphDdrRegisterAbi{GraphDdrRegisterRole::Fp8BlockScale,
            GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi, 20, 21}, scale}});
    for (auto* tensor : {&weight, &scale}) {
        rpu_ddr_flush(tensor->data_ptr());
        RpuKernelGraph::active().keep_alive(*tensor);
    }
    kernel->reset_regs();
    rpu_pl_tiling::set_parallel_linear_nvfp4_regs(*kernel, {
        {input, 0, output, 0, 51, 1024, 256, 0, 8, 512, 2048, 0, 0, 0},
        8 * 1024 * 2, tensor_scale_spm, layer});
    writer.write(*kernel);
    kernel->set_regs(64, uint16_t(8));
    kernel->set_regs(65, uint16_t(1));
    kernel->set_regs(66, uint16_t(1));
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {uint16_t(8), uint16_t(1), uint16_t(1)},
                        {0, 1, 2, 3, 4, 5, 6, 7});
}
