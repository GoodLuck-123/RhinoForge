"""Action-mask lifetimes using production declarations and the host allocator."""
from pathlib import Path


from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]


def _program():
    fmb = (ROOT / "src/core/fused_model_base.h").read_text()
    helpers = (ROOT / "src/core/rpu_helpers.h").read_text()
    allocator = (ROOT / "src/core/rpu_spm_allocator.cpp").read_text()
    allocator_h = (ROOT / "src/core/rpu_spm_allocator.h").read_text()
    p = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define TORCH_CHECK(c, ...) do {if(!(c))throw std::runtime_error("guard");}while(0)
#define DWIDTH 2
#define NUM_CORES 8
#define SDPA_MAX_VLM_SIZE 400
#define RPU_SDPA_TILE_K_CONSERVATIVE_MODE 1
#define QWEN3_FUSED_LM_HEAD_MAX_VOCAB 200000
template<class T> T CeilDiv(T a,T b) {return (a+b-1)/b;}
template<class T,class U> T Align(T a,U b) {return CeilDiv(a,static_cast<T>(b))*b;}
'''
    for marker in ("struct SdpaTiling", "enum class SdpaKernelType", "struct SdpaConfig"):
        p += _definition(helpers, marker) + ";\n"
    for marker in (
        "inline int64_t sdpa_tile_m_v16_initial(",
        "inline uint32_t GetMhaAttnUnivVlmNeed(",
        "inline std::optional<int64_t> choose_tile_k_aggressive(",
        "inline std::optional<int64_t> choose_tile_k_conservative(",
        "inline std::optional<int64_t> choose_tile_k(",
        "inline bool sdpa_try_compute_tiling(",
        "inline SdpaTiling sdpa_compute_tiling(",
        "inline int64_t sdpa_compute_tmp_v16_size(",
    ):
        p += _definition(helpers, marker) + "\n"
    p += "class SpmAllocator {public: static constexpr size_t ALIGN=256;\n"
    p += "static size_t align_up(size_t n){return (n+255)&~size_t{255};}\n"
    for marker in ("struct AllocRequest", "struct AliasedPlan"):
        p += _definition(allocator_h, marker) + ";\n"
    p += "static AliasedPlan plan_temporary_aliased(const std::vector<AllocRequest>&);};\n"
    p += _definition(allocator, "SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(")
    p += "namespace v3 { class FusedModelBase {};\n"
    for marker in ("enum class StorageClass", "enum class BufferScope", "struct BufferDecl", "enum class FmbForwardOperandResidency", "struct LayoutContext"):
        if marker == "struct LayoutContext":
            p += "enum class AttentionExecutionPolicy {DDR_KV,SPM_KV_BY_MHA};\n"
            p += "enum class FmbRopeTableResidency {UNSPECIFIED};\n"
        p += _definition(fmb, marker) + ";\n"
    p += "namespace detail {int64_t estimate_temporary_total(const std::vector<BufferDecl>&);}\n"
    p += _definition((ROOT / "src/core/fused_model_base.cpp").read_text(),
                     "int64_t detail::estimate_temporary_total(")
    p += "}\n"
    residency = (ROOT / "src/core/rpu_spm_residency.h").read_text()
    p += residency[residency.index("namespace v3::detail {"):]
    p += "\nnamespace v3 {\n"
    return p
