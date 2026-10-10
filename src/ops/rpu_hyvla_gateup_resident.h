#pragma once
#include "rpu_linear_tiling.h"

// Same exact M51 bounds for the legacy pair and private fused producer.
// Grid8 fusion resets its core phase counter in every launch/replay packet.
inline void launch_hyvla_gateup_resident_spm(
    uint32_t input_spm, uint32_t gate_output_spm, uint32_t up_output_spm,
    uint32_t gate_weight_spm, uint32_t up_weight_spm, int64_t rows, bool fused_swiglu) {
  TORCH_CHECK(rows == 51 && SPM_ALLOC.is_initialized(),
              "HyVLA resident Gate/Up requires M51 and initialized SPM");
  constexpr uint64_t kWeightBytes = 256 * 1024 * sizeof(c10::Half);
  const std::array<std::pair<uint32_t, uint64_t>, 5> regions{{
      {input_spm, 51 * 1024 * sizeof(c10::Half)},
      {gate_output_spm, 51 * 256 * sizeof(c10::Half)},
      {up_output_spm, 51 * 256 * sizeof(c10::Half)},
      {gate_weight_spm, kWeightBytes}, {up_weight_spm, kWeightBytes}}};
  const uint64_t base0 = SPM_ALLOC.addr(0, 0);
  for (size_t i = 0; i < regions.size(); ++i) {
    const uint64_t start = regions[i].first, bytes = regions[i].second;
    TORCH_CHECK(start >= base0 && start % 256 == 0 &&
                    start - base0 <= SpmAllocator::SPM_USABLE - bytes,
                "HyVLA resident Gate/Up requires complete aligned SPM regions");
    for (int core = 0; core < 8; ++core) {
      TORCH_CHECK(uint64_t(SPM_ALLOC.addr(core, 0)) + start - base0 + bytes - 1 <= UINT32_MAX,
                  "HyVLA resident Gate/Up absolute SPM range exceeds uint32");
    }
    for (size_t j = 0; j < i; ++j) {
      const uint64_t other = regions[j].first, other_bytes = regions[j].second;
      TORCH_CHECK(start + bytes <= other || other + other_bytes <= start,
                  "HyVLA resident Gate/Up requires disjoint SPM operands");
    }
  }
  auto* kernel = fused_swiglu
      ? RpuKernelGraph::active().get_kernel_reset("hyvla_resident_swiglu_s51")
      : GET_KERNEL(KernelId::WALL_ACTION_GATE_UP_FP16_SPM_WEIGHT_M592N32K128);
  TORCH_CHECK(kernel != nullptr, "HyVLA resident Gate/Up FP16 payload unavailable");
  kernel->reset_regs();
  rpu_pl_tiling::set_parallel_linear_regs(*kernel,
      rpu_pl_tiling::ParallelLinearRegisterArgs{
          input_spm, 0, gate_output_spm, 0, 51, 256, 1024, 0, 8,
          1024 * sizeof(c10::Half), 256 * sizeof(c10::Half), 0, 1, 1});
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 2, gate_weight_spm);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 24, up_weight_spm);
  rpu_pl_tiling::set_u32_reg_pair(*kernel, 30, up_output_spm);
  if (fused_swiglu) kernel->set_regs(128, uint32_t{0});
  auto* queue = GET_QUEUE(8);
  queue->set_broadcast_mode(true);
  queue->enqueu_kernel(*kernel, {uint16_t(fused_swiglu ? 8 : 16), 1, 1}, {0, 1, 2, 3, 4, 5, 6, 7});
}
