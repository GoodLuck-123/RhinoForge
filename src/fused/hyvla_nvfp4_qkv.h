#pragma once
#include "rpu_helpers.h"
#include "rpu_kernel_decls.h"
#include "rpu_linear_tiling.h"

// qkv=false requires vectors [K,V]; true requires [Q,K,V]. Each projection
// owns distinct FP8 block scales, FP32 tensor-scale table and FP16 output.
inline void launch_hyvla_nvfp4_qkv_spm(
    uint32_t input, const std::vector<at::Tensor>& weights,
    const std::vector<at::Tensor>& block_scales,
    const std::vector<uint32_t>& tensor_scales,
    const std::vector<uint32_t>& outputs, uint16_t layer, bool qkv,
    int64_t rows = 51, int cores = 8) {
    const size_t count = qkv ? 3 : 2;
    TORCH_CHECK(rows == 51 && cores == 8 && layer < 32 && SPM_ALLOC.is_initialized() &&
                    weights.size() == count && block_scales.size() == count &&
                    tensor_scales.size() == count && outputs.size() == count,
                "HyVLA NVFP4 KV/QKV requires M51/K1024/TP8/L32 and complete operand vectors");
    std::vector<std::pair<uint32_t, uint64_t>> regions{{input, 51 * 1024 * 2}};
    for (size_t i = 0; i < count; ++i) {
        const int64_t global_n = qkv && i == 0 ? 2048 : 1024;
        const auto& w = weights[i];
        const auto& s = block_scales[i];
        TORCH_CHECK(w.defined() && w.device().type() == c10::DeviceType::PrivateUse1 &&
                        w.scalar_type() == at::kByte && w.is_contiguous() &&
                        w.dim() == 2 && w.size(0) == global_n && w.size(1) == 512,
                    "HyVLA NVFP4 projection requires packed Byte [globalN,512]");
        TORCH_CHECK(s.defined() && s.device() == w.device() &&
                        s.scalar_type() == at::kByte && s.is_contiguous() &&
                        s.dim() == 2 && s.size(0) == 64 && s.size(1) == global_n &&
                        w.device() == weights[0].device(),
                    "HyVLA NVFP4 projection requires striped FP8 Byte [64,globalN] on the same device");
        regions.push_back({outputs[i], 51 * uint64_t(global_n / 8) * 2});
        regions.push_back({tensor_scales[i], 32 * sizeof(float)});
    }
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    for (size_t i = 0; i < regions.size(); ++i) {
        const uint64_t address = regions[i].first, bytes = regions[i].second;
        TORCH_CHECK(address >= base && address % 256 == 0 &&
                        address - base + bytes <= SpmAllocator::SPM_USABLE,
                    "HyVLA NVFP4 KV/QKV requires complete aligned absolute SPM owners");
        for (int core = 0; core < 8; ++core)
            TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core, 0)) + address - base + bytes <=
                            uint64_t(UINT32_MAX) + 1,
                        "HyVLA NVFP4 KV/QKV SPM range overflows on a core");
        for (size_t j = 0; j < i; ++j)
            TORCH_CHECK(address + bytes <= regions[j].first ||
                            uint64_t(regions[j].first) + regions[j].second <= address,
                        "HyVLA NVFP4 KV/QKV requires disjoint SPM owners");
    }
    const KernelId id = qkv ? KernelId::HYVLA_EXPERT_QKV_NVFP4_S51
                            : KernelId::HYVLA_EXPERT_KV_NVFP4_S51;
    const std::array<uint16_t, 3> weight_slots{2, 32, 40};
    const std::array<uint16_t, 3> scale_slots{20, 34, 42};
    const std::array<uint16_t, 3> table_slots{24, 36, 44};
    const std::array<uint16_t, 3> output_slots{4, 38, 46};
    std::vector<GraphDdrRegisterOperandSpec> operands;
    for (size_t i = 0; i < count; ++i) {
        for (int scale = 0; scale < 2; ++scale) {
            const auto& tensor = scale ? block_scales[i] : weights[i];
            const uint64_t address = rhino_lkn::RpuGetDevAddr(tensor.data_ptr());
            constexpr uint64_t limit = UINT64_C(1) << 40;
            TORCH_CHECK(address != 0 && address % 256 == 0 && address < limit &&
                            uint64_t(tensor.nbytes()) <= limit - address,
                        "HyVLA NVFP4 KV/QKV requires complete aligned shift8 DDR owners");
            const uint16_t lo = scale ? scale_slots[i] : weight_slots[i];
            operands.push_back({GraphDdrRegisterAbi{
                scale ? GraphDdrRegisterRole::Fp8BlockScale : GraphDdrRegisterRole::PackedFp4ModelWeight,
                GraphDdrRegisterAccess::Read, GraphDdrRegisterEncoding::DevAddrShift8LoHi,
                lo, uint16_t(lo + 1)}, tensor});
        }
    }
    auto* kernel = GET_KERNEL(id);
    TORCH_CHECK(kernel, "HyVLA NVFP4 KV/QKV payload unavailable");
    auto writer = RpuKernelGraph::active().stage_kernel_ddr_registers(id, operands);
    for (size_t i = 0; i < count; ++i)
        for (const auto* tensor : {&weights[i], &block_scales[i]}) {
            rpu_ddr_flush(tensor->data_ptr());
            RpuKernelGraph::active().keep_alive(*tensor);
        }
    kernel->reset_regs();
    const uint32_t n = qkv ? 256 : 128;
    rpu_pl_tiling::set_parallel_linear_nvfp4_regs(*kernel, {
        {input, 0, outputs[0], 0, 51, n, 1024, 0, 8, 2048, n * 2, 0, 1, 0},
        8 * n * 2, tensor_scales[0], layer});
    for (size_t i = 1; i < count; ++i) {
        rpu_pl_tiling::set_u32_reg_pair(*kernel, table_slots[i], tensor_scales[i]);
        rpu_pl_tiling::set_u32_reg_pair(*kernel, output_slots[i], outputs[i]);
    }
    writer.write(*kernel);
    kernel->set_regs(64, uint16_t(8));
    kernel->set_regs(65, uint16_t(1));
    kernel->set_regs(66, uint16_t(1));
    auto* queue = GET_QUEUE(8);
    queue->set_broadcast_mode(true);
    queue->enqueu_kernel(*kernel, {uint16_t(8), uint16_t(1), uint16_t(1)},
                        {0, 1, 2, 3, 4, 5, 6, 7});
}
