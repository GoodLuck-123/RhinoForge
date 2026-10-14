#pragma once

#include "fused_model_base.h"
#include "rpu_kernel_decls.h"
#include <array>
#include <cmath>

namespace v3 {

// Cold G05-only striped_v2 owners. The kernel loads eight FP32 scales at a
// time; 24 layers occupy exactly three groups, inside a 256-byte SPM slot.
struct G05Nvfp4Tables {
    std::vector<at::Tensor> values;
    inline static constexpr std::array<const char*, 5> names{
        "g05_nvfp4_q_ts", "g05_nvfp4_o_ts", "g05_nvfp4_gate_ts",
        "g05_nvfp4_up_ts", "g05_nvfp4_down_ts"};
    inline static constexpr std::array<int64_t, 5> output_sizes{
        2048, 1024, 4096, 4096, 1024};
    inline static constexpr std::array<int64_t, 5> input_sizes{
        1024, 2048, 1024, 1024, 4096};
    inline static constexpr std::array<int, 5> partitions{1, 0, 1, 1, 0};

    bool enabled() const { return !values.empty(); }
    void install(at::TensorList tables) {
        TORCH_CHECK(tables.empty() || tables.size() == 5,
                    "G05 NVFP4 requires exactly five tensor-scale tables");
        for (const auto& table : tables) {
            TORCH_CHECK(table.defined() && table.scalar_type() == at::kFloat &&
                            table.device().type() == at::kPrivateUse1 &&
                            table.is_contiguous() && table.dim() == 1 &&
                            table.numel() == 24,
                        "G05 NVFP4 tensor scales must be contiguous FP32 device [24]");
            const auto cpu = table.cpu();
            const auto* p = cpu.data_ptr<float>();
            for (int layer = 0; layer < 24; ++layer) {
                const float retired = static_cast<float>(c10::Half(p[layer]));
                TORCH_CHECK(std::isfinite(p[layer]) && p[layer] > 0 &&
                                std::isfinite(retired) && retired > 0,
                            "G05 NVFP4 tensor scale must remain finite and positive in ACC16");
            }
        }
        values.assign(tables.begin(), tables.end());
    }
    void declare(std::vector<BufferDecl>& decls, int cores) const {
        for (size_t family = 0; family < values.size(); ++family) {
            const auto table = values[family];
            decls.push_back({names[family], 256, 0, 0, StorageClass::Persistent,
                0, nullptr, BufferScope::LayerWide,
                [table, cores](FusedModelBase&, int, uint32_t destination) {
                    rpu_launch_ddr_broadcast_spm_dma(
                        reinterpret_cast<c10::Half*>(table.data_ptr<float>()),
                        table.numel() * 2, destination, cores);
                }});
        }
    }
};

} // namespace v3
