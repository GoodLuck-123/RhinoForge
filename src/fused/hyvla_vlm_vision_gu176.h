#pragma once
#include "rpu_ops.h"
#include "rpu_kernel_decls.h"
#include "rpu_spm_allocator.h"

// Actual installation mask proof, not a camera-count hint. Two leading text
// tokens, then three [start, 7*(7 vision + split), end] blocks; remaining rows
// are text. The caller freezes this result with its independently copied masks.
inline bool hyvla_standard_three_camera_mask(const float* text, int64_t rows) {
    if (!text || (rows != 192 && rows != 208 && rows != 224 && rows != 240)) return false;
    for (int64_t row = 0; row < rows; ++row) {
        bool vision = false;
        if (row >= 2 && row < 176) {
            const int64_t slot = (row - 2) % 58;
            vision = slot >= 1 && slot <= 56 && (slot - 1) % 8 < 7;
        }
        if (text[row] != (vision ? 0.0f : 1.0f)) return false;
    }
    return true;
}

// Vision-only composition. Keep the installed W8 ACC16 auto-tile Linear and
// separate FP16 SiLU/Mul arithmetic. Full-S storage stays owned; zero gate tail
// before the caller's unchanged full-S Down. Finite masked outputs are the
// experiment's equivalence domain: replacing an inactive NaN with zero is not
// claimed to preserve IEEE 0*NaN propagation.
inline void launch_hyvla_vlm_vision_gu176_w8(
    uint32_t input, uint32_t gate, uint32_t up, int64_t rows,
    const at::Tensor& gate_weight, const at::Tensor& up_weight,
    const at::Tensor& gate_scale, const at::Tensor& up_scale) {
    TORCH_CHECK((rows == 192 || rows == 208 || rows == 224 || rows == 240) &&
                    SPM_ALLOC.is_initialized(), "HyVLA vision GU176 requires an admitted prefix");
    for (const auto* weight : {&gate_weight, &up_weight})
        TORCH_CHECK(weight->defined() && weight->device().type() == c10::DeviceType::PrivateUse1 &&
                    weight->device() == gate_weight.device() && weight->scalar_type() == at::kChar &&
                    weight->is_contiguous() && weight->dim() == 2 &&
                    weight->size(0) == 6144 && weight->size(1) == 2048,
                    "HyVLA vision GU176 requires installed W8 [6144,2048]");
    for (const auto* scale : {&gate_scale, &up_scale})
        TORCH_CHECK(scale->defined() && scale->device() == gate_weight.device() &&
                    scale->scalar_type() == at::kHalf && scale->is_contiguous() &&
                    scale->dim() == 1 && scale->size(0) == 6144,
                    "HyVLA vision GU176 requires FP16 [6144] scales");
    for (const auto* tensor : {&gate_weight, &up_weight, &gate_scale, &up_scale}) {
        const uint64_t address = rhino_lkn::RpuGetDevAddr(tensor->data_ptr());
        TORCH_CHECK(address % 256 == 0 && address < (uint64_t(1) << 40) &&
                        tensor->nbytes() <= (uint64_t(1) << 40) - address,
                    "HyVLA vision GU176 DDR range/alignment");
    }
    const std::array<std::pair<uint32_t, uint64_t>, 3> regions{{
        {input, uint64_t(rows)*2048*2}, {gate, uint64_t(rows)*768*2},
        {up, uint64_t(rows)*768*2}}};
    const uint64_t base = SPM_ALLOC.addr(0, 0);
    for (size_t i = 0; i < regions.size(); ++i) {
        const uint64_t start = regions[i].first, bytes = regions[i].second;
        TORCH_CHECK(start >= base && start % 256 == 0 &&
                        start - base <= SpmAllocator::SPM_USABLE - bytes,
                    "HyVLA vision GU176 requires full-S aligned SPM storage");
        for (int core = 0; core < 8; ++core)
            TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core, 0)) + start - base + bytes - 1 <= UINT32_MAX,
                        "HyVLA vision GU176 absolute SPM overflow");
        for (size_t j = 0; j < i; ++j)
            TORCH_CHECK(start + bytes <= regions[j].first ||
                            uint64_t(regions[j].first) + regions[j].second <= start,
                        "HyVLA vision GU176 requires disjoint SPM operands");
    }
    rpu_launch_linear_spm_to_spm_acc16_kernel(input, gate_weight, gate,
        176, 6144, 2048, 1, 8, 0u, gate_scale);
    rpu_launch_eltwise_unary_spm_kernel(gate, gate, 176*768, ValuOpType::SILU, false, 8);
    rpu_launch_linear_spm_to_spm_acc16_kernel(input, up_weight, up,
        176, 6144, 2048, 1, 8, 0u, up_scale);
    rpu_launch_eltwise_binary_spm_kernel(gate, up, gate, 176*768,
        ValuOpType::MUL, c10::Half(1.0f), 8);
    rpu_launch_fill_spm_kernel(gate + 176*768*2, (rows-176)*768, c10::Half(0.0f), 8);
}
