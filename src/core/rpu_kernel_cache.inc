// Kernel/program discovery and process-wide cache implementation fragment.
#include "rpu_ops.h"
#include "rpu_kernel_manifest.h"
#include "rpu_linear_tiling.h"
#include "rpu_sha256.h"
#include "rpu_graph_program_contracts.h"

#include <dlfcn.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// KernelCache 实现 - 预加载所有 kernel 以减少运行时开销
// =============================================================================

// Kernel library path: env RPU_KERNEL_LIB_PATH > rhinoOpLib_current.ref next
// to the .so (production install) > <so_dir>/../../src/rhinoOpLib_current.ref
// (dev tree, .so under build/rpu_backend/) > cwd-relative fallback.
//
// rhinoOpLib_current.ref is the canonical published main payload in src/.
// Publish operators in rhino-ops first, then update this file and its manifest
// identity together. No model-specific or historical REF is a runtime input.
static std::string get_kernel_lib_path() {
    const char* env = std::getenv("RPU_KERNEL_LIB_PATH");
    if (env && *env) return std::string(env);

    constexpr const char* REF_NAME = "rhinoOpLib_current.ref";

    auto try_dir = [](const std::string& dir) -> std::string {
        std::string p = dir + "/" + REF_NAME;
        return access(p.c_str(), F_OK) == 0 ? p : std::string{};
    };

    Dl_info info;
    if (dladdr((void*)get_kernel_lib_path, &info) && info.dli_fname) {
        std::string so_dir(info.dli_fname);
        auto pos = so_dir.rfind('/');
        if (pos != std::string::npos) {
            so_dir = so_dir.substr(0, pos);
        }
        // Co-located next to the .so (production install layout).
        if (auto p = try_dir(so_dir); !p.empty()) return p;
        // Dev tree: build/rpu_backend/*.so → ../../src/<versioned asset>.
        if (auto p = try_dir(so_dir + "/../../src"); !p.empty()) return p;
    }

    // Last-ditch fallback: project-relative path from cwd. Caller (KernelCache)
    // will fail loudly with a clear "cannot open ..." if neither dladdr nor
    // cwd resolves to a real file — better than a hardcoded absolute path
    // tied to one developer's machine.
    return std::string("src/") + REF_NAME;
}

static const std::string& KERNEL_LIB_PATH() {
    static const std::string path = get_kernel_lib_path();
    return path;
}

static std::string get_kernel_lib_path_expansion() {
    const char* env = std::getenv("RPU_KERNEL_LIB_PATH_EXPANSION");
    if (env && *env) return std::string(env);

    constexpr const char* EXP_REF_NAME = "rhinoExpansionOpLib_current.ref";
    const std::string& current = KERNEL_LIB_PATH();
    auto pos = current.rfind('/');
    if (pos != std::string::npos) {
        return current.substr(0, pos) + "/" + EXP_REF_NAME;
    }
    return EXP_REF_NAME;  // Main was explicitly selected relative to cwd.
}

static const std::string& KERNEL_LIB_PATH_EXPANSION() {
    static const std::string path = get_kernel_lib_path_expansion();
    return path;
}

#define RHINO_OP_LIB_PATH           KERNEL_LIB_PATH().c_str()
#define RHINO_EXPANSION_OP_LIB_PATH KERNEL_LIB_PATH_EXPANSION().c_str()
#define GEMM_LIB_PATH               KERNEL_LIB_PATH().c_str()
#define SOFTMAX_LIB_PATH            KERNEL_LIB_PATH().c_str()


// 需要预加载的 kernel 列表
static const std::vector<std::pair<std::string, std::string>> KERNEL_LIST = {
    // Optional fused kernels supplied by the matching runtime asset.
    {"partial_mrope_qk_fused", RHINO_EXPANSION_OP_LIB_PATH},
    {"llama_insert_kv_cache_multiwarp_v16", RHINO_EXPANSION_OP_LIB_PATH},
    {"llama_insert_kv_cache_multiwarp_decode", RHINO_EXPANSION_OP_LIB_PATH},
    {"llama_rms_norm_multiwarp_n_v64", RHINO_EXPANSION_OP_LIB_PATH},
    {"parallel_linear_w8a16_m448n64k128_qkv_planar", RHINO_EXPANSION_OP_LIB_PATH},
    {"llama_silu_mul_exact_strided", RHINO_EXPANSION_OP_LIB_PATH},
    // Qwen3.5 decode fused kernels shipped by the current main op-lib ref.
    {"qwen3_5_rms_norm_gated", RHINO_OP_LIB_PATH},
    {"qwen3_5_rank1_fma", RHINO_OP_LIB_PATH},
    {"qwen3_5_mul_reduce_rows", RHINO_OP_LIB_PATH},
    // Qwen3.5 35B sparse-MoE grouped expert and routing kernels. These are
    // required main-ref symbols: preload them so KernelId fast-cache misses
    // fail during initialization rather than at the first model launch.
    {"grouped_parallel_linear_acc16", RHINO_OP_LIB_PATH},
    {"grouped_parallel_linear_w8a16_acc16", RHINO_OP_LIB_PATH},
    {"grouped_parallel_linear_acc32", RHINO_EXPANSION_OP_LIB_PATH},
    {"grouped_parallel_linear_w8a16_acc32", RHINO_EXPANSION_OP_LIB_PATH},
    {"grouped_parallel_linear_acc16_expert_y4", RHINO_EXPANSION_OP_LIB_PATH},
    {"grouped_parallel_linear_w8a16_acc16_expert_y4", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_route_compact_e32_t51_tp64", RHINO_EXPANSION_OP_LIB_PATH},
    {"topk_norm_scalar_fp16", RHINO_OP_LIB_PATH},
    {"routed_col_parallel_linear_m608n32k128", RHINO_OP_LIB_PATH},
    {"routed_col_parallel_linear_w8a16_m592n32k128", RHINO_OP_LIB_PATH},
    {"lingbot2_topk_dense_route_safe", RHINO_EXPANSION_OP_LIB_PATH},
    {"llama_rms_norm_nogamma", RHINO_OP_LIB_PATH},
    {"llama_rms_norm_nogamma_v16", RHINO_OP_LIB_PATH},
    {"llama_rms_norm_nogamma_v32", RHINO_OP_LIB_PATH},
    {"lingbot2_combine_t51_r204_h768_scan", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_topk_e32_k4_lane2_safe", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_swiglu_route_r204_d64", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_combine_t51_r204_h768_inverse", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_combine_shared_t51_r204_h768_inverse", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_grouped_gateup_routed_acc16_expert_y4", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_grouped_gateup_routed_w8a16_acc16_expert_y4", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_grouped_gateup_routed_fp16_swiglu_route", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_grouped_gateup_routed_w8_swiglu_route", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_grouped_down_w8a16_acc16_scale_shuffle", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_insert_kv_t51", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_shared_silu_t51_d96_vector", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_qkv_multibase_fp16_t51", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_qkv_multibase_w8_t51", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_sdpa_t51_k275_tail", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_vision_fc1_gelu_fp16_m384n96k128", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_vision_fc1_gelu_w8_m400n80k128", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_prefill_qkv_multibase_fp16_m224", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_prefill_qkv_multibase_w8_m224", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_vision_sdpa_spm_b3s256h16d64_32b", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_grouped_down_fp16_ordered_combine", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_grouped_down_w8_ordered_combine", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_topk_lane2_compact_inverse_e32_t51_tp64", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_adarms_norm_shift_h768_diagonal", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_vision_qkv_joint_fp16_m768", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_vision_qkv_joint_w8_m768", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_qk_rope_kv_t51", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_prefill_gateup_shared_a_w8_m224", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_router_sigmoid_pad_t51_h768_e32", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_shared_gateup_resident_silu_w8_t51_h768_i96", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_shared_gateup_resident_silu_fp16_t51_h768_i96", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_ring_adarms_norm_shift_t51_h768", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_prefill_qk_norm_rope_m224", RHINO_EXPANSION_OP_LIB_PATH},
    {"lingbot2_route_compact_inverse_e32_t51_tp64", RHINO_EXPANSION_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc16_m176n128k128", RHINO_EXPANSION_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc16_m208n112k128", RHINO_EXPANSION_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc16_m240n96k128", RHINO_EXPANSION_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc16_m272n80k128", RHINO_EXPANSION_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc16_m320n64k128", RHINO_EXPANSION_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc16_m384n48k128", RHINO_EXPANSION_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc16_m480n32k128", RHINO_EXPANSION_OP_LIB_PATH},
    // Optional at startup for older refs; UINT4 profile consumption fails closed.
    {"grouped_parallel_linear_wuint4a16_pgrp_m288n32k256", RHINO_OP_LIB_PATH},
    {"grouped_parallel_linear_wuint4a16_pgrp_m256n48k256", RHINO_OP_LIB_PATH},
    {"grouped_parallel_linear_wuint4a16_pgrp_m224n64k256", RHINO_OP_LIB_PATH},
    {"grouped_parallel_linear_wuint4a16_pgrp_m192n80k256", RHINO_OP_LIB_PATH},
    {"grouped_parallel_linear_wuint4a16_pgrp_m160n96k256", RHINO_OP_LIB_PATH},
    {"grouped_parallel_linear_wuint4a16_pgrp_m144n112k256", RHINO_OP_LIB_PATH},
    {"grouped_parallel_linear_wuint4a16_pgrp_m128n128k256", RHINO_OP_LIB_PATH},
    {"wall_base_nvfp4_qkv_bias_barrier_n64", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_nvfp4_qkv_bias_barrier_n32", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_nvfp4_o_ring_norm_h32", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_nvfp4_down_ring_norm_h32", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_partial_mrope_qk_h32", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_nvfp4_gate_up_parallel_h32", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_nvfp4_kv_parallel_bias_h32", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_nvfp4_qkv_parallel_bias_h32", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_nvfp4_qkv_bias_rope_h32", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_nvfp4_gate_up_parallel_silu_h32", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_nvfp4_gate_up_shared_silu_h32", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_prefill_gate_up_shared_a_m384n48k128", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_prefill_gate_up_shared_silu_m384n48k128", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_prefill_gate_up_shared_silu_noup_m384n48k128", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_vision_owner_ring_norm_m672_h1280", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_vision_down_owner_ring_norm_m672_h1280", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_vision_qk_rope_2d_spm_m672_m324", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_vision_ragged_sdpa_b1_s324_s336_h16d80", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_prefill_o_owner_ring_norm_m224_h2048", RHINO_EXPANSION_OP_LIB_PATH},
    {"wall_base_prefill_gate_up_shared_silu_fp16_m320n80k128", RHINO_EXPANSION_OP_LIB_PATH},
    {"topk_scalar_fp16", RHINO_OP_LIB_PATH},
    {"topk_by_kvsort_fp16", RHINO_OP_LIB_PATH},
    {"bincount_scalar", RHINO_OP_LIB_PATH},
    {"llm_topk_prob_norm", RHINO_OP_LIB_PATH},
    {"unary_cast_uint16_int32", RHINO_OP_LIB_PATH},
    {"gather_scalar", RHINO_OP_LIB_PATH},
    {"gather", RHINO_OP_LIB_PATH},
    {"llm_gather_mul_v2", RHINO_OP_LIB_PATH},
    {"scatternd_int16_reduction", RHINO_OP_LIB_PATH},
    {"llama_gemv", RHINO_OP_LIB_PATH},
    {"llama_gemv_wint8", RHINO_OP_LIB_PATH},
    {"qwen3vl_gateup_swiglu_gemv", RHINO_EXPANSION_OP_LIB_PATH},
    {"qwen3vl_qk_norm_mrope_d128", RHINO_EXPANSION_OP_LIB_PATH},
    {"qwen3vl_qk_norm_mrope_kv_insert_d128", RHINO_EXPANSION_OP_LIB_PATH},
    {"qwen3vl_o_ring_norm_m1_h2560_w8", RHINO_EXPANSION_OP_LIB_PATH},
    // rhinoOpLib kernels
    {"parallel_linear_wnvfp4a16_acc32_m176n128k128", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc32_m208n112k128", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc32_m240n96k128", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc32_m272n80k128", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc32_m320n64k128", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc32_m384n48k128", RHINO_OP_LIB_PATH},
    {"parallel_linear_wnvfp4a16_acc32_m480n32k128", RHINO_OP_LIB_PATH},
    {"binary_sameshape", RHINO_OP_LIB_PATH},
    {"binary_sameshape_ddr", RHINO_EXPANSION_OP_LIB_PATH},
    {"binary_scalar", RHINO_OP_LIB_PATH},
    {"binary_scalar_ddr", RHINO_EXPANSION_OP_LIB_PATH},
    {"binary_scalar_power", RHINO_OP_LIB_PATH},
    {"binary_scalar_power_ddr", RHINO_EXPANSION_OP_LIB_PATH},
    {"binary_Nx1_NxC256_batch", RHINO_OP_LIB_PATH},
    {"binary_Nx1_NxC256_batch_ddr", RHINO_EXPANSION_OP_LIB_PATH},
    {"binary_Nx1_NxC256_batch_bopa", RHINO_OP_LIB_PATH},
    {"binary_Nx1_NxC256_batch_bopa_ddr", RHINO_EXPANSION_OP_LIB_PATH},
    {"binary_Nx1_NxC_v16_batch", RHINO_OP_LIB_PATH},
    {"binary_Nx1_NxC_v16_batch_ddr", RHINO_EXPANSION_OP_LIB_PATH},
    {"binary_Nx1_NxC_v16_batch_bopa", RHINO_OP_LIB_PATH},
    {"binary_Nx1_NxC_v16_batch_bopa_ddr", RHINO_EXPANSION_OP_LIB_PATH},
    // V2 tile-based Nx1_NxC (ConvertEltwiseBinary_Nx1_NxC_V2): runtime routes c%16!=0 || c>1024
    // || n>8192 here (e.g. GDN recurrent state *= exp(g_last), c=Dk*Dv=16384).
    {"binary_Nx1_NxC_tileN", RHINO_OP_LIB_PATH},
    {"binary_Nx1_NxC_tileC", RHINO_OP_LIB_PATH},
    {"binary_Nx1_NxC_tileN_bopa", RHINO_OP_LIB_PATH},
    {"binary_Nx1_NxC_tileC_bopa", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_v256_batch", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_v256_batch_ddr", RHINO_EXPANSION_OP_LIB_PATH},
    {"binary_1xC_NxC_v256_batch_bopa", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_v256_batch_bopa_ddr", RHINO_EXPANSION_OP_LIB_PATH},
    {"binary_1xC_NxC_v16_batch", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_v16_batch_ddr", RHINO_EXPANSION_OP_LIB_PATH},
    {"binary_1xC_NxC_v16_batch_bopa", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_v16_batch_bopa_ddr", RHINO_EXPANSION_OP_LIB_PATH},
    // V2 tile-based 1xC_NxC (ConvertEltwiseBinary_1xC_NxC_V2): the runtime routes
    // c%16!=0 || c>1024 || n>8192 here (e.g. GDN prep dt_bias/neg_exp_A, c=vg_c=4).
    {"binary_1xC_NxC_tileN", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_tileC", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_tileN_bopa", RHINO_OP_LIB_PATH},
    {"binary_1xC_NxC_tileC_bopa", RHINO_OP_LIB_PATH},
    {"unary", RHINO_OP_LIB_PATH},
    {"unary_gelu_tanh", RHINO_OP_LIB_PATH},
    {"unary_gelu_erf", RHINO_OP_LIB_PATH},
    {"unary_gelu_erf_ultra_precision", RHINO_OP_LIB_PATH},
    {"unary_softplus", RHINO_OP_LIB_PATH},
    {"unary_ddr", RHINO_EXPANSION_OP_LIB_PATH},
    {"layer_norm", RHINO_OP_LIB_PATH},
    {"layer_norm_simple", RHINO_OP_LIB_PATH},
    {"layer_norm_bf16", RHINO_EXPANSION_OP_LIB_PATH},
    // gemm kernels (常用尺寸) - bmm 动态生成名称，这里预加载常见的组合
    // 格式: gemm_fp16_spm_16b_wMxN_kK_1core_buf1_nt_lpaddr_{peak|univ}
    {"gemm_fp16_spm_16b_w64x64_k64_1core_buf1_nt_lpaddr_peak", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w64x64_k64_1core_buf1_nt_lpaddr_univ", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w64x64_k128_1core_buf1_nt_lpaddr_peak", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w64x64_k128_1core_buf1_nt_lpaddr_univ", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w128x128_k64_1core_buf1_nt_lpaddr_peak", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w128x128_k64_1core_buf1_nt_lpaddr_univ", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w128x128_k128_1core_buf1_nt_lpaddr_peak", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w128x128_k128_1core_buf1_nt_lpaddr_univ", GEMM_LIB_PATH},
    // GDN chunk A@B (nt_) with N=Dv/Dk=128, M=C=64: value_new/k_cumdecay (K=C=64),
    // out=kcd@S / q@S (K=dk=128), recur_out=attn@v (K=C=64).
    {"gemm_fp16_spm_16b_w64x128_k64_1core_buf1_nt_lpaddr_peak", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w64x128_k64_1core_buf1_nt_lpaddr_univ", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w64x128_k128_1core_buf1_nt_lpaddr_peak", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w64x128_k128_1core_buf1_nt_lpaddr_univ", GEMM_LIB_PATH},
    // GDN chunk transposing bmm: rowByRow (nn_) for attn = q@kᵀ (M=N=C=64, K=dk=128),
    // colByCol (tt_) for state += kᵀ@v (M=N=dk/dv=128, K=C=64). Kernels already in oplib.
    {"gemm_fp16_spm_16b_w64x64_k128_1core_buf1_nn_lpaddr_peak", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w64x64_k128_1core_buf1_nn_lpaddr_univ", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w128x128_k64_1core_buf1_tt_lpaddr_peak", GEMM_LIB_PATH},
    {"gemm_fp16_spm_16b_w128x128_k64_1core_buf1_tt_lpaddr_univ", GEMM_LIB_PATH},
    // softmax kernel
    {"softmax_c16_gauto", SOFTMAX_LIB_PATH},
    // flash attention kernel
    {"llm_fp16_32b_prefill_flash_attn_univ_dp", RHINO_OP_LIB_PATH},
    {"llm_fp16_16b_prefill_flash_attn_univ_dp", RHINO_OP_LIB_PATH},
    {"llm_fp16_32b_prefill_flash_attn_univ_vctxlen", RHINO_OP_LIB_PATH},  // vctxlen (dynamic context length)
    {"llm_fp16_32b_prefill_flash_attn_univ_vctxlen_minibatch", RHINO_OP_LIB_PATH},  // per-image K/V slicing (multi-image vision packing)
    {"llm_fp16_16b_prefill_flash_attn_univ_vctxlen_minibatch", RHINO_OP_LIB_PATH},
    {"llm_fp16_32b_prefill_trans_prjv_dp_vctxlen", RHINO_OP_LIB_PATH},
    {"llm_fp16_16b_prefill_flash_attn_by_mha_univ_vctxlen", RHINO_OP_LIB_PATH},
    // RoPE kernels
    {"llama_rope", RHINO_OP_LIB_PATH},
    {"all_gather_multi_core", RHINO_OP_LIB_PATH},
    {"all_gather_multi_core_little_chunk", RHINO_OP_LIB_PATH},
    {"llm_all_reduce_residual", RHINO_OP_LIB_PATH},
    {"llm_all_reduce_residual_nopace", RHINO_OP_LIB_PATH},
    // tile kernels (for expand/repeat optimization)
    {"tile_Nx1_NxC", RHINO_OP_LIB_PATH},
    {"tile_Nx1_NxC_v16", RHINO_OP_LIB_PATH},
    {"tile_general_largeC", RHINO_OP_LIB_PATH},
    {"tile_general_smallC", RHINO_OP_LIB_PATH},
    // reduce mean kernels
    {"reduce_mean_last_dim", RHINO_OP_LIB_PATH},
    {"reduce_mean_last_dim_ddr", RHINO_EXPANSION_OP_LIB_PATH},
    // RMSNorm kernels (llama_rms_norm_fp16 dropped, llama_rms_norm_bf16 renamed
    // to llama_rms_norm in rhinoOpLib v0.1.1_38977cf7)
    {"llama_rms_norm_bf16_ddr", RHINO_EXPANSION_OP_LIB_PATH},
    {"llama_rms_norm", RHINO_OP_LIB_PATH},  // SPM版本
    // R48b —— 多行/warp 变体（**optional**：ref 里没有就跳过，不影响启动）
    {"llama_rms_norm_v16", RHINO_OP_LIB_PATH},
    {"llama_rms_norm_v32", RHINO_OP_LIB_PATH},
    // Base Newton is registered below; all three Newton kernels are consumed
    // only by the isolated conformance probe.
    {"llama_rms_norm_newton_v16", RHINO_OP_LIB_PATH},
    {"llama_rms_norm_newton_v32", RHINO_OP_LIB_PATH},
    // R50 —— mrope 的 SPM 表孪生（optional；只被隔离探针 / 打开开关的路径取用）
    {"llama_mrope_interleave_local_spm_tbl", RHINO_OP_LIB_PATH},
    {"llama_rms_norm_newton", RHINO_OP_LIB_PATH},  // conformance-only
    {"llama_rope_ddr", RHINO_EXPANSION_OP_LIB_PATH},
    // M-RoPE kernel (Qwen3-VL R-Phase 1)
    {"llama_mrope_interleave", RHINO_OP_LIB_PATH},
    // Partial RoPE kernels (Qwen3.5 partial rotary):
    // partial_mrope = prefill, partial_rope_1d = decode.
    {"partial_mrope", RHINO_OP_LIB_PATH},
    {"partial_rope_1d", RHINO_OP_LIB_PATH},
    // Qwen3.5 GDN decode 核心:l2norm + causal conv1d-update。
    {"l2norm", RHINO_OP_LIB_PATH},
    {"llm_fla_conv1d_w16a16_acc16", RHINO_OP_LIB_PATH},
    {"llm_fla_conv1d_noinplace_w16a16_acc16", RHINO_OP_LIB_PATH},
    // Qwen3.5 GDN prefill chunk: cumulative sum (in default ref).
    {"cumsum_reduceC_tileN", RHINO_OP_LIB_PATH},
    // Qwen3.5 GDN prefill chunk: strict-lower-tri (I - M)^-1 (in default ref).
    {"unit_tril_inv", RHINO_OP_LIB_PATH},
    // Qwen3.5 GDN gated-norm fused silu(x)*y (in default ref).
    {"llama_silu_mul", RHINO_OP_LIB_PATH},
    // Qwen3.5 GDN: sum over rows (axis 0), replaces tree-reduce (in default ref).
    {"reduce_sum_non_last_dim_out_seg", RHINO_OP_LIB_PATH},
    // Qwen3.5 GDN prefill chunk: slice/insert_slice (per-chunk PCS/PCI + conv halo) +
    // Bx1xC broadcast eltwise (decay = g_cumsum[row] - g_cumsum[col]). These are
    // lazy-loadable in passthrough but the recorded fused graph needs them preloaded.
    {"slice_last_dim_axis", RHINO_OP_LIB_PATH},
    {"slice_single_axis", RHINO_OP_LIB_PATH},
    {"binary_Bx1xC_BxNxC_tileN", RHINO_OP_LIB_PATH},
    {"binary_Bx1xC_BxNxC_tileC", RHINO_OP_LIB_PATH},
    {"binary_Bx1xC_BxNxC_tileN_bopa", RHINO_OP_LIB_PATH},
    {"binary_Bx1xC_BxNxC_tileC_bopa", RHINO_OP_LIB_PATH},
    // 2D RoPE kernel (Qwen3-VL R-Phase 3 vision encoder)
    {"rope_2d_ddr", RHINO_OP_LIB_PATH},
    {"rope_2d_spm", RHINO_OP_LIB_PATH},
    // Fast bilinear pos-embed interpolation (Qwen3-VL / Qwen3.5 vision STEP 0)
    {"fast_pos_embed_interpolate", RHINO_OP_LIB_PATH},
    // LLaMA KV-Cache kernels
    {"llama_insert_vcache_multiwarp", RHINO_OP_LIB_PATH},
    {"llama_insert_kcache_multiwarp", RHINO_OP_LIB_PATH},
    {"llama_insert_vcache_multiwarp_v16", RHINO_OP_LIB_PATH},
    {"llama_insert_kcache_multiwarp_v16", RHINO_OP_LIB_PATH},
    // DDR to SPM memcpy (multi-core)
    // SPM to DDR memcpy (multi-core)
    // Argmax/Argmin kernels
    {"argmax_reduceC_tileN", RHINO_OP_LIB_PATH},
    {"argmax_reduceC_tileN_C128", RHINO_OP_LIB_PATH},
    {"argmax_reduceC_tileN_sli", RHINO_OP_LIB_PATH},
    {"argmax_reduceC_tileN_C128_sli", RHINO_OP_LIB_PATH},
    {"argmax_reduceN_tileC", RHINO_OP_LIB_PATH},
    {"argmax_reduceN_tileC_sli", RHINO_OP_LIB_PATH},
    {"argmin_reduceC_tileN", RHINO_OP_LIB_PATH},
    {"argmin_reduceC_tileN_C128", RHINO_OP_LIB_PATH},
    {"argmin_reduceC_tileN_sli", RHINO_OP_LIB_PATH},
    {"argmin_reduceC_tileN_C128_sli", RHINO_OP_LIB_PATH},
    {"argmin_reduceN_tileC", RHINO_OP_LIB_PATH},
    {"argmin_reduceN_tileC_sli", RHINO_OP_LIB_PATH},
    // memset lives in the expansion ref when absent from the main ref.
    {"memset_spm_multi_core_v2", RHINO_EXPANSION_OP_LIB_PATH},
    // DDR to SPM memcpy, scatter read DDR, broadcast write spm
    // GDN prefill chunk zeroing: `fill` (1D) replaces the absent memset_v2 (rhino's
    // CreateFill drives this same kernel). Used by rpu_launch_fill_spm_kernel.
    {"fill", RHINO_OP_LIB_PATH},
    // Conv2d SPM kernels (1core, tile_m=128, tile_n=128, univ)
    {"conv_fp16_f1_spm_16b_w128x128_k48_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f1_spm_16b_w128x128_k64_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f1_spm_16b_w128x128_k96_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f1_spm_16b_w128x128_k128_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_16b_w128x128_k48_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_16b_w128x128_k64_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_16b_w128x128_k96_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_16b_w128x128_k128_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_16b_w128x128_k48_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_16b_w128x128_k64_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_16b_w128x128_k96_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_16b_w128x128_k128_1core_buf1_lpaddr_univ", RHINO_OP_LIB_PATH},
    // Conv2d multi-core column kernels (tile_k=32, tile_n=32/64/128)
    {"conv_fp16_f1_spm_column_w128x32_k32_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_column_w128x32_k32_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_column_w128x32_k32_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f1_spm_column_w128x64_k32_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_column_w128x64_k32_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_column_w128x64_k32_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f1_spm_column_w128x128_k32_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_column_w128x128_k32_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_column_w128x128_k32_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    // Conv2d multi-core column kernels (tile_k=64, tile_n=32/64/128)
    {"conv_fp16_f1_spm_column_w128x32_k64_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_column_w128x32_k64_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_column_w128x32_k64_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f1_spm_column_w128x64_k64_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_column_w128x64_k64_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_column_w128x64_k64_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f1_spm_column_w128x128_k64_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_f3_spm_column_w128x128_k64_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    {"conv_fp16_fn_spm_column_w128x128_k64_coren_buf1_sIsWsO_univ", RHINO_OP_LIB_PATH},
    // Im2col kernels (multi-core, SPM→SPM, for patch embedding fusion)
    {"im2col_coren_m64xn32_buf1_sIsO", RHINO_OP_LIB_PATH},
    {"im2col_coren_m128xn32_buf1_sIsO", RHINO_OP_LIB_PATH},
    {"im2col_coren_m64xn64_buf1_sIsO", RHINO_OP_LIB_PATH},
    {"im2col_coren_m128xn64_buf1_sIsO", RHINO_OP_LIB_PATH},
    // Pad kernel (constant fill, SPM→SPM)
    {"pad_const_NxC_NxCP", RHINO_OP_LIB_PATH},
    // Transpose kernel (3D, SPM→SPM)
    {"transpose_ncb_c16", RHINO_OP_LIB_PATH},
    {"transpose_nbc_c16", RHINO_OP_LIB_PATH},
    // transpose_nbc_c16 above also serves the GDN/NextDiT perm{1,0,2} path.
    // perm{0,2,1} = order 021 for the chunk bmm rowByRow/colByCol pre-transpose.
    {"transpose_bcn_c16", RHINO_OP_LIB_PATH},
    // 4D perm{0,2,1,3} used by G0.5 compact STEP0 Graph capture.
    {"transpose_qnbc_c16", RHINO_OP_LIB_PATH},
    // SPM-gather-DDR (empirical test for paligemma2 attn_tp scatter fix).
    // DDR scatter → per-core SPM (address-tabular). Preloaded for future callers.
    {"pi05_denoise_gate_up_geglu_nvfp4_acc16_m320n64k128", RHINO_EXPANSION_OP_LIB_PATH},
    {"unary_silu_high_precision", RHINO_OP_LIB_PATH},
    {"unary_tanh_high_precision", RHINO_OP_LIB_PATH},
    {"unary_gelu_erf_high_precision", RHINO_OP_LIB_PATH},
    {"rhinovla_newton_norm_shift_h1024", RHINO_EXPANSION_OP_LIB_PATH},
    {"rhinovla_silu_high_mul", RHINO_EXPANSION_OP_LIB_PATH},
    {"unary_gelu_tanh_high_precision", RHINO_OP_LIB_PATH},
};

static void print_tensor_info(const at::Tensor &t, const std::string &name) {
    std::cout << "[" << name << "] "
              << "device=" << t.device()
              << ", dtype=" << t.scalar_type()
              << ", layout=" << t.layout()
              << ", sizes=[";
    for (int i = 0; i < t.dim(); i++) {
        std::cout << t.size(i);
        if (i != t.dim() - 1) std::cout << ", ";
    }
    std::cout << "], strides=[";
    for (int i = 0; i < t.dim(); i++) {
        std::cout << t.stride(i);
        if (i != t.dim() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}
// Optional startup entries may be absent from their designated role. A model
// that requires one still fails before weight loading via require_names().
static bool is_optional_kernel(const std::string& name) {
    static const std::unordered_set<std::string> optional = {
        "grouped_parallel_linear_acc32",
        "grouped_parallel_linear_w8a16_acc32",
        "grouped_parallel_linear_wuint4a16_pgrp_m128n128k256",
        "grouped_parallel_linear_wuint4a16_pgrp_m144n112k256",
        "grouped_parallel_linear_wuint4a16_pgrp_m160n96k256",
        "grouped_parallel_linear_wuint4a16_pgrp_m192n80k256",
        "grouped_parallel_linear_wuint4a16_pgrp_m224n64k256",
        "grouped_parallel_linear_wuint4a16_pgrp_m256n48k256",
        "grouped_parallel_linear_wuint4a16_pgrp_m288n32k256",
        "grouped_parallel_linear_acc16_expert_y4",
        "grouped_parallel_linear_w8a16_acc16_expert_y4",
        "lingbot2_route_compact_e32_t51_tp64",
        "topk_norm_scalar_fp16",
        "routed_col_parallel_linear_m608n32k128",
        "routed_col_parallel_linear_w8a16_m592n32k128",
        "lingbot2_topk_dense_route_safe",
        "llama_rms_norm_nogamma",
        "llama_rms_norm_nogamma_v16",
        "llama_rms_norm_nogamma_v32",
        "lingbot2_combine_t51_r204_h768_scan",
        "lingbot2_topk_e32_k4_lane2_safe",
        "lingbot2_swiglu_route_r204_d64",
        "lingbot2_combine_t51_r204_h768_inverse",
        "lingbot2_combine_shared_t51_r204_h768_inverse",
        "lingbot2_grouped_gateup_routed_acc16_expert_y4",
        "lingbot2_grouped_gateup_routed_w8a16_acc16_expert_y4",
        "lingbot2_grouped_gateup_routed_fp16_swiglu_route",
        "lingbot2_grouped_gateup_routed_w8_swiglu_route",
        "lingbot2_grouped_down_w8a16_acc16_scale_shuffle",
        "lingbot2_insert_kv_t51",
        "lingbot2_shared_silu_t51_d96_vector",
        "lingbot2_qkv_multibase_fp16_t51",
        "lingbot2_qkv_multibase_w8_t51",
        "lingbot2_sdpa_t51_k275_tail",
        "lingbot2_vision_fc1_gelu_fp16_m384n96k128",
        "lingbot2_vision_fc1_gelu_w8_m400n80k128",
        "lingbot2_prefill_qkv_multibase_fp16_m224",
        "lingbot2_prefill_qkv_multibase_w8_m224",
        "lingbot2_vision_sdpa_spm_b3s256h16d64_32b",
        "lingbot2_grouped_down_fp16_ordered_combine",
        "lingbot2_grouped_down_w8_ordered_combine",
        "lingbot2_topk_lane2_compact_inverse_e32_t51_tp64",
        "lingbot2_adarms_norm_shift_h768_diagonal",
        "lingbot2_vision_qkv_joint_fp16_m768",
        "lingbot2_vision_qkv_joint_w8_m768",
        "lingbot2_qk_rope_kv_t51",
        "lingbot2_prefill_gateup_shared_a_w8_m224",
        "lingbot2_router_sigmoid_pad_t51_h768_e32",
        "lingbot2_shared_gateup_resident_silu_w8_t51_h768_i96",
        "lingbot2_shared_gateup_resident_silu_fp16_t51_h768_i96",
        "lingbot2_ring_adarms_norm_shift_t51_h768",
        "lingbot2_prefill_qk_norm_rope_m224",
        "lingbot2_route_compact_inverse_e32_t51_tp64",
        "parallel_linear_wnvfp4a16_acc16_m176n128k128",
        "parallel_linear_wnvfp4a16_acc16_m208n112k128",
        "parallel_linear_wnvfp4a16_acc16_m240n96k128",
        "parallel_linear_wnvfp4a16_acc16_m272n80k128",
        "parallel_linear_wnvfp4a16_acc16_m320n64k128",
        "parallel_linear_wnvfp4a16_acc16_m384n48k128",
        "parallel_linear_wnvfp4a16_acc16_m480n32k128",
        "pi05_denoise_gate_up_geglu_nvfp4_acc16_m320n64k128",
        "rhinovla_newton_norm_shift_h1024",
        "rhinovla_silu_high_mul",
        "unary_gelu_erf_high_precision",
        "unary_gelu_tanh_high_precision",
        "unary_silu_high_precision",
        "unary_tanh_high_precision",
        "wall_base_nvfp4_down_ring_norm_h32",
        "wall_base_nvfp4_gate_up_parallel_h32",
        "wall_base_nvfp4_gate_up_parallel_silu_h32",
        "wall_base_nvfp4_gate_up_shared_silu_h32",
        "wall_base_nvfp4_kv_parallel_bias_h32",
        "wall_base_nvfp4_o_ring_norm_h32",
        "wall_base_nvfp4_qkv_bias_barrier_n32",
        "wall_base_nvfp4_qkv_bias_barrier_n64",
        "wall_base_nvfp4_qkv_bias_rope_h32",
        "wall_base_nvfp4_qkv_parallel_bias_h32",
        "wall_base_partial_mrope_qk_h32",
        "wall_base_prefill_gate_up_shared_a_m384n48k128",
        "wall_base_prefill_gate_up_shared_silu_fp16_m320n80k128",
        "wall_base_prefill_gate_up_shared_silu_m384n48k128",
        "wall_base_prefill_gate_up_shared_silu_noup_m384n48k128",
        "wall_base_prefill_o_owner_ring_norm_m224_h2048",
        "wall_base_vision_down_owner_ring_norm_m672_h1280",
        "wall_base_vision_owner_ring_norm_m672_h1280",
        "wall_base_vision_qk_rope_2d_spm_m672_m324",
        "wall_base_vision_ragged_sdpa_b1_s324_s336_h16d80",
        // RMSNorm vector kernels are optional at process startup. An explicit
        // production route still fails at its launcher if V16/V32 is absent.
        "llama_rms_norm_newton",
        "llama_rms_norm_v16", "llama_rms_norm_v32",
        "llama_rms_norm_newton_v16", "llama_rms_norm_newton_v32",
        "qwen3vl_gateup_swiglu_gemv",
        "qwen3vl_qk_norm_mrope_d128",
        "qwen3vl_qk_norm_mrope_kv_insert_d128",
        "qwen3vl_o_ring_norm_m1_h2560_w8",
        // R50 —— mrope SPM 表孪生：同上，默认路径不碰。
        "llama_mrope_interleave_local_spm_tbl",
    };
    return optional.count(name) != 0;
}

// The sidecar is public metadata; operator payloads remain opaque. Cache the
// allowlist before constructing any Program, because Launch aborts on a miss.
const std::unordered_set<std::string>& KernelCache::kernel_manifest_names(
    const std::string& lib_path) {
    std::lock_guard<std::mutex> lock(loaded_oplib_paths_mutex_);
    auto found = oplib_symbols_.find(lib_path);
    if (found != oplib_symbols_.end()) return found->second;
    return oplib_symbols_.emplace(
        lib_path, rpu_kernel_manifest::load(lib_path)).first->second;
}

static std::string canonical_payload_path(const std::string& path) {
    std::error_code error;
    const auto canonical = std::filesystem::canonical(path, error);
    TORCH_CHECK(
        !error, "cannot canonicalize runtime payload '", path,
        "': ", error.message());
    return canonical.string();
}

static const std::string& FROZEN_KERNEL_LIB_PATH() {
    static const std::string path = canonical_payload_path(KERNEL_LIB_PATH());
    return path;
}

static const std::string& FROZEN_KERNEL_LIB_PATH_EXPANSION() {
    static const std::string path =
        canonical_payload_path(KERNEL_LIB_PATH_EXPANSION());
    return path;
}

std::shared_ptr<OplibSnapshot> KernelCache::oplib_snapshot(const std::string& path) {
    std::lock_guard<std::mutex> lock(loaded_oplib_paths_mutex_);
    auto found = oplib_snapshots_.find(path);
    if (found != oplib_snapshots_.end()) return found->second;
    auto snapshot = std::make_shared<OplibSnapshot>(path);
    // Hash the sealed image consumed by Program_t, never the mutable role path.
    const auto digest = rpu_payload_identity::sha256(snapshot->bytes());
    const auto inserted = oplib_snapshots_.emplace(path, snapshot);
    try {
        oplib_sha256_.emplace(path, digest);
    } catch (...) {
        oplib_snapshots_.erase(inserted.first);
        throw;
    }
    return snapshot;
}

CachedKernel KernelCache::load_kernel_from_oplib(
    const std::string& path, const std::string& kernel_name) {
    // Callers pass the frozen canonical role path. Never resolve it afresh:
    // replacement/deletion must not change which snapshot a later Program uses.
    TORCH_CHECK(kernel_manifest_names(path).count(kernel_name) != 0,
                "kernel not listed in the operator manifest: ", kernel_name);
    const auto snapshot = oplib_snapshot(path);
    CachedKernel cached;
    cached.program = std::make_unique<::rhino_lkn::Program_t>();
    const int32_t rc = cached.program->create_with_binary_file(
        snapshot->path(), kernel_name.c_str());
    TORCH_CHECK(
        rc == 0, "failed to load kernel '", kernel_name, "' from '",
        path, "': rc=", rc);
    cached.kernel = std::make_unique<::rhino_lkn::Kernel_t>(
        *cached.program, kernel_name.c_str());
    {
        std::lock_guard<std::mutex> lock(loaded_oplib_paths_mutex_);
        cached.graph_snapshot_contracts =
            rpu_graph_program_contracts::audited_contracts(
                oplib_sha256_.at(path), kernel_name);
        loaded_oplib_paths_.insert(path);
    }
    return cached;
}

std::vector<std::string> KernelCache::loaded_oplib_paths() const {
    std::lock_guard<std::mutex> lock(loaded_oplib_paths_mutex_);
    return {loaded_oplib_paths_.begin(), loaded_oplib_paths_.end()};
}

std::vector<std::pair<std::string, std::string>> KernelCache::loaded_oplib_bytes() const {
    std::lock_guard<std::mutex> lock(loaded_oplib_paths_mutex_);
    TORCH_CHECK(initialized_, "runtime oplib identity requires completed kernel initialization");
    std::vector<std::pair<std::string, std::string>> result;
    for (const auto& path : loaded_oplib_paths_) {
        result.emplace_back(path, oplib_snapshots_.at(path)->bytes());
    }
    return result;
}

uint32_t KernelCache::graph_snapshot_program_contracts(
    const std::string& kernel_name) const {
    const auto contracts = std::atomic_load_explicit(
        &graph_snapshot_contracts_, std::memory_order_acquire);
    if (!contracts) return 0;
    const auto found = contracts->find(kernel_name);
    return found == contracts->end() ? 0 : found->second;
}

std::shared_ptr<const std::unordered_map<std::string, uint32_t>>
KernelCache::graph_snapshot_contracts_snapshot() const {
    auto result = std::make_shared<std::unordered_map<std::string, uint32_t>>();
    for (const auto& item : kernels_) {
        if (item.second.program && item.second.kernel && item.second.graph_snapshot_contracts) {
            result->emplace(item.first, item.second.graph_snapshot_contracts);
        }
    }
    return result;
}

void KernelCache::initialize() {
    if (initialized_) return;

    if (log_at(3)) {
        std::cout << "[KernelCache] Initializing kernel cache..." << std::endl;
    }
    auto start = std::chrono::high_resolution_clock::now();

    const std::string& main_path = FROZEN_KERNEL_LIB_PATH();
    const std::string& expansion_path =
        FROZEN_KERNEL_LIB_PATH_EXPANSION();
    TORCH_CHECK(
        main_path != expansion_path,
        "[KernelCache] main and expansion op-lib paths resolve to the same "
        "payload: '", main_path, "'");
    const auto& main_symbols = kernel_manifest_names(main_path);
    const auto& expansion_symbols = kernel_manifest_names(expansion_path);
    static constexpr const char* kMainRoleCanaries[] = {
        "qwen3_5_rms_norm_gated",
        "qwen3_5_rank1_fma",
        "qwen3_5_mul_reduce_rows",
    };
    for (const char* name : kMainRoleCanaries) {
        TORCH_CHECK(
            main_symbols.count(name) != 0 && expansion_symbols.count(name) == 0,
            "[KernelCache] op-lib role mismatch for main-only kernel '", name,
            "' (main='", main_path, "' expansion='", expansion_path, "')");
    }

    std::vector<std::string> missing_required;
    for (const auto& [kernel_name, lib_path] : KERNEL_LIST) {
        const bool expansion_role =
            lib_path == KERNEL_LIB_PATH_EXPANSION();
        const auto& selected_symbols =
            expansion_role ? expansion_symbols : main_symbols;
        if (expansion_role) {
            TORCH_CHECK(
                main_symbols.count(kernel_name) == 0,
                "[KernelCache] expansion-only kernel '", kernel_name,
                "' is also present in main op-lib '", main_path, "'");
        }
        if (selected_symbols.count(kernel_name) == 0) {
            // Probe before create_with_binary_file: the SDK aborts on a missing
            // symbol, so required kernels fail through an actionable exception.
            if (is_optional_kernel(kernel_name)) {
                if (log_at(3)) {
                    std::cout << "[KernelCache] optional kernel '" << kernel_name
                              << "' absent from its designated "
                              << (expansion_role ? "expansion" : "main")
                              << " ref — skipping (tolerated)" << std::endl;
                }
            } else {
                missing_required.push_back(
                    std::string(expansion_role ? "expansion:" : "main:") +
                    kernel_name);
            }
            continue;
        }
        try {
            CachedKernel cached = load_kernel_from_oplib(
                expansion_role ? expansion_path : main_path, kernel_name);
            kernels_.erase(kernel_name);
            kernels_.emplace(kernel_name, std::move(cached));
        } catch (const std::exception& e) {
            if (log_at(2)) {
                std::cerr << "[KernelCache] Warning: Failed to load kernel '"
                          << kernel_name << "': " << e.what() << std::endl;
            }
            // Probe said the ref ships this symbol but the load still threw ⇒ corrupt
            // or ABI-incompatible ref. A REQUIRED kernel must abort via the same
            // fail-fast as the absent-from-both-refs path, not leave a nullptr behind.
            if (!is_optional_kernel(kernel_name)) {
                missing_required.push_back(
                    std::string(expansion_role ? "expansion:" : "main:") +
                    kernel_name);
            }
        }
    }

    // Fail-fast instead of letting one ref silently satisfy the other role.
    TORCH_CHECK(missing_required.empty(),
                "[KernelCache] ", missing_required.size(),
                " required kernel(s) absent from their designated op-lib refs "
                "(main='", main_path, "' expansion='", expansion_path,
                "'). First missing: '",
                (missing_required.empty() ? std::string() : missing_required.front()),
                "'. Check the .ref install and RPU_KERNEL_LIB_PATH[_EXPANSION].");

    // All eight 7-tile families are required: FP16/W8/UINT4 x ACC16/ACC32
    // belong to main; the unchanged signed W4-pgrp ABI belongs to expansion.
    // Never satisfy a missing tile from the other role.
    {
        auto at_names = rpu_pl_tiling::autotile_kernel_names();
        const auto autotile_library = [&](const std::string& name)
            -> const std::string& {
            return name.rfind("parallel_linear_wint4a16_pgrp_", 0) == 0
                ? expansion_path : main_path;
        };
        std::vector<std::string> missing_at;
        for (const auto& kname : at_names) {
            const std::string& at_lib = autotile_library(kname);
            const auto& at_symbols = kernel_manifest_names(at_lib);
            if (!at_symbols.count(kname)) {
                missing_at.push_back(
                    std::string(at_lib == expansion_path ? "expansion:" : "main:")
                    + kname);
            }
        }
        // Program_t may terminate the process instead of throwing when a symbol
        // is absent. Inspect both roles first so an old/wrong artifact fails
        // through a normal, actionable TORCH_CHECK.
        TORCH_CHECK(
            missing_at.empty(),
            "[KernelCache] autotile ref mismatch: ", missing_at.size(), "/",
            at_names.size(), " required tiled linear kernels are absent from their "
            "designated refs (main='", main_path, "' expansion='", expansion_path,
            "'). First missing: '",
            (missing_at.empty() ? std::string() : missing_at.front()),
            "'. Install the op ref matching the selected runtime profile.");
        int at_loaded = 0;
        for (const auto& kname : at_names) {
            const std::string& at_lib = autotile_library(kname);
            try {
                CachedKernel cached = load_kernel_from_oplib(at_lib, kname);
                // Some tiled names are also part of the fixed preload list.
                // Destroy a replaced Kernel before its owning Program.
                kernels_.erase(kname);
                kernels_.emplace(kname, std::move(cached));
                ++at_loaded;
            } catch (const std::exception& e) {
                if (log_at(2)) {
                    std::cerr << "[KernelCache] autotile: failed to load '" << kname
                              << "' from " << at_lib << ": " << e.what() << std::endl;
                }
            }
        }
        TORCH_CHECK(
            at_loaded == static_cast<int>(at_names.size()),
            "[KernelCache] autotile ref mismatch: loaded ", at_loaded, "/",
            at_names.size(), " required tiled linear kernels from their designated "
            "refs (main='", main_path, "' expansion='", expansion_path,
            "'). Install the op refs matching the selected runtime profile.");
        if (log_at(3)) {
            std::cout << "[KernelCache] autotile: loaded " << at_loaded
                      << "/" << at_names.size()
                      << " tiled linear variants from main/expansion roles"
                      << std::endl;
        }
    }

    // Preload optional fused and compact-residual operators from expansion.
    // Missing symbols are tolerated so unrelated installs remain unchanged;
    // an exact-profile caller fails at lookup with a targeted error.
    {
        const std::string& mixed_lib = expansion_path;
        int mixed_loaded = 0;
        static constexpr const char* kMixedNames[] = {
            "llm_all_reduce_residual_local_spm",
            "fused_gate_up_swiglu_w8a16_m320n112k128",
            "pi05_gate_up_geglu_w8a16_m400n80k128",
            "pi05_adarms_norm_shift_h1024",
            "pi05_gated_residual_h1024",
            "pi05_all_reduce_residual_xor3_m768n1152",
            "pi05_all_reduce_residual_xor3_m400n2048",
            "pi05_all_reduce_residual_xor3_m50n1024",
            "pi05_k_rope_insert_m50d256p64",
            "pi05_denoise_gateup_resident_w8a16_m512n48k128",
            "gr00t_ff1_resident_w8a16_m512n48k128",
            "gr00t_qkv_multibase_w8a16_m592n32k128",
            "gr00t_all_reduce_residual_xor3_m41n1536",
            "g05_linear_fp16_acc32_spm_weight_m304n80k128",
            "g05_kv_owner_fp16_acc32_m32n256k1024",
            "g05_kv_owner_fanout_m32d256",
            "g05_vision_owner_reduce_layernorm_m256n1024",
            "g05_vision_xor3_compact_residual_raw_full_m256n1024",
            "g05_q_gate_fp16_acc32_m32n256k1024",
            "g05_gate_up_fp16_acc32_m32n512k1024",
            "g05_silu_round_mul_m32n512",
            "g05_vision_qkv_fp16_acc32_m256n128k1024",
            "g05_text_gate_up_fp16_acc32_m256n768k2048",
            "g05_gateup_resident_w8a16_acc32_m240n96k128",
            "g05_kv_owner_w8a16_acc32_m32n256k1024",
            "g05_q_gate_w8a16_acc32_m32n256k1024",
            "g05_gate_up_w8a16_acc32_m32n512k1024",
            "g05_vision_qkv_w8a16_acc32_m256n128k1024",
            "g05_text_gate_up_w8a16_acc32_m256n768k2048",
            "g05_qkv_gate_owner_w8a16_acc32_m32n256k1024",
            "g05_k_owner_rms_norm_m32d256",
            "g05_k_owner_partial_mrope_m32d256r64",
            "g05_owner_insert_kcache_multiwarp",
            "g05_owner_insert_kcache_multiwarp_v16",
            "g05_owner_insert_vcache_multiwarp",
            "g05_owner_insert_vcache_multiwarp_v16",
            "g05_gateup_resident_w8a16_acc16_m352n96k128",
            "g05_kv_owner_w8a16_acc16_m32n256k1024",
            "g05_q_gate_w8a16_acc16_m32n256k1024",
            "g05_gate_up_w8a16_acc16_m32n512k1024",
            "g05_vision_qkv_w8a16_acc16_m256n128k1024",
            "g05_text_gate_up_w8a16_acc16_m256n768k2048",
            "g05_qkv_gate_owner_w8a16_acc16_m32n256k1024",
            "g05_sub_gated_residual_h1024",
            "g05_rtc_adarms_norm_shift_m32h1024",
            "g05_o_resident_w8a16_acc16_m32n1024k256",
            "g05_rtc_sub_gated_residual_m32h1024",
    "g05_o_resident_w8a16_acc16_m32n1024k256_n128",
    "g05_ring_rtc_sub_gated_residual_m32h1024",
    "g05_down_partial672_w8a16_acc16_m32n1024k512",
    "g05_vision_fc2_weight_outer_w8a16_acc16_m768n1024k512",
    "g05_gate_up_paired_nvfp4_acc16_m32n512k1024_n128",
    "g05_o_resident_nvfp4_acc16_m32n1024k256_n80",
    "g05_down_resident_nvfp4_acc16_m32n1024k512_n64",
    "g05_gate_up_swiglu_nvfp4_acc16_m32n512k1024_n128",
    "g05_gate_kv_owner_w8a16_acc16_m32n256k1024_n128",
    "g05_q_norm_partial_rope_m32d256r64_group4",
    "g05_k_owner_norm_partial_rope_m32d256r64_group4",
    "g05_text_ring_owner_rms_m256h2048",
    "g05_owner_insert_kv_cache_m32d256_p0",
    "g05_owner_insert_kv_cache_m32d256_p245",
    "g05_attn_sigmoid_round_mul_m32d256",
    "g05_ring_rtc_then_norm_shift_m32h1024",
    "g05_gate_up_fp16_acc16_m32n512k1024",
    "g05_text_gate_up_fp16_acc16_m256n768k2048",
    "g05_gate_up_paired_nvfp4_acc32_m32n512k1024_n128",
    "g05_o_resident_nvfp4_acc32_m32n1024k256_n80",
    "g05_down_resident_nvfp4_acc32_m32n1024k512_n64",
    "g05_gate_up_swiglu_nvfp4_acc32_m32n512k1024_n128",
    "hyvla_expert_rmsnorm_v16_bitcopy",
            "pi05_prefill_o_linear_xor3_rmsnorm_m400n2048k256",
            "pi05_down_weight_outer_w8a16_c400x2_m160n80k128",
            "pi05_owner_norm_full_residual_m400n2048",
            "pi05_owner_norm_compact_residual_m400n2048",
            "pi05_xor3_compact_residual_raw_full_m400n2048",
            "pi05_prefill_qkv_weight_outer_w8a16_c400x2_m128n64k128",
            "pi05_prefill_o_weight_outer_w8a16_c400x2_m128n80k128",
            "pi05_vision_fc_weight_outer_w8a16_m768n80k128",
            "pi05_denoise_q_pair16_rope_m50n256k1024",
            "pi05_xor3_gated_residual_m50n1024",
            "pi05_denoise_kv1_direct_cache_m50d256p64",
            "pi05_denoise_kv1_pair_owner_w8a16_m50n32x2k1024",
            "pi05_prefill_kv1_pair_owner_w8a16_m400n32x2k2048",
            "pi05_prefill_kv1_direct_cache_m400d256p400",
            "pi05_prefill_gate_up_geglu_weight_outer_w8a16_c400x2_m160n80k128",
            "pi05_denoise_gate_up_geglu_w8a16_m512n48k128",
            "pi05_denoise_gate_up_geglu_wint4a16_pgrp_m512n48k128",
            "pi05_denoise_gate_up_geglu_wint4a16_pgrp_m464n64k128",
            "parallel_linear_wnvfp4a16_acc32_m320n64k128",
            "parallel_linear_wnvfp4a16_acc32_m384n48k128",
            "pi05_denoise_gate_up_geglu_nvfp4_acc32_m320n64k128",
            "pi05_owner_norm_compact_a8_m400n2048",
            "pi05_prefill_gate_up_geglu_w8a8_split_m48n32k2048",
            "pi05_prefill_gate_up_geglu_online_w8a8_p544_c272_m48n32k2048",
            "pi05_vision_owner_reduce_layernorm_m768n1152",
            "pi05_vision_xor3_compact_residual_raw_full_m768n1152_epoch",
            "pi05_denoise_kv1_pair_owner_fp16_m50n32x2k1024",
            "pi05_denoise_gate_up_geglu_fp16_m608n32k128",
            "pi05_vision_owner_reduce_layernorm_m512n1152",
            "pi05_vision_xor3_compact_residual_raw_full_m512n1152_epoch",
            "pi05_all_reduce_residual_xor3_m512n1152",
            "pi05_all_reduce_residual_xor3_m272n2048",
            "pi05_owner_norm_full_residual_m272n2048",
            "pi05_owner_norm_compact_residual_m272n2048",
            "pi05_xor3_compact_residual_raw_full_m272n2048",
            "pi05_owner_norm_compact_a8_m272n2048",
            "pi05_prefill_kv1_direct_cache_m272d256p272",
            "pi05_prefill_o_weight_outer_fp16_c400x2_m128n80k128",
            "pi05_prefill_o_weight_outer_fp16_c272x2_m128n80k128",
            "pi05_prefill_o_weight_outer_w8a16_c272x2_m128n80k128",
            "pi05_down_weight_outer_fp16_c400x2_m160n80k128",
            "pi05_down_weight_outer_fp16_c272x2_m160n80k128",
            "pi05_down_weight_outer_w8a16_c272x2_m160n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_fp16_c400x2_m160n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_fp16_c272x2_m160n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_w8a16_c272x2_m160n80k128",
            "pi05_prefill_kv1_pair_owner_fp16_m400n32x2k2048",
            "pi05_prefill_kv1_pair_owner_fp16_m272n32x2k2048",
            "pi05_prefill_kv1_pair_owner_w8a16_m272n32x2k2048",
            "pi05_prefill_gate_up_geglu_w8a8_split_c272_m48n32k2048",
            "pi05_all_reduce_residual_xor3_m416n2048",
            "pi05_owner_norm_full_residual_m416n2048",
            "pi05_owner_norm_compact_residual_m416n2048",
            "pi05_xor3_compact_residual_raw_full_m416n2048",
            "pi05_owner_norm_compact_a8_m416n2048",
            "pi05_prefill_kv1_direct_cache_m416d256p416",
            "pi05_prefill_o_weight_outer_fp16_c416x2_m128n80k128",
            "pi05_prefill_o_weight_outer_w8a16_c416x2_m128n80k128",
            "pi05_down_weight_outer_fp16_c416x2_m128n80k128",
            "pi05_down_weight_outer_w8a16_c416x2_m128n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_fp16_c416x2_m128n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_w8a16_c416x2_m128n80k128",
            "pi05_prefill_kv1_pair_owner_fp16_m416n32x2k2048",
            "pi05_prefill_kv1_pair_owner_w8a16_m416n32x2k2048",
            "pi05_prefill_gate_up_geglu_w8a8_split_c416_m48n32k2048",
            "pi05_all_reduce_residual_xor3_m288n2048",
            "pi05_owner_norm_full_residual_m288n2048",
            "pi05_owner_norm_compact_residual_m288n2048",
            "pi05_xor3_compact_residual_raw_full_m288n2048",
            "pi05_owner_norm_compact_a8_m288n2048",
            "pi05_prefill_kv1_direct_cache_m288d256p288",
            "pi05_prefill_o_weight_outer_fp16_c288x2_m128n80k128",
            "pi05_prefill_o_weight_outer_w8a16_c288x2_m128n80k128",
            "pi05_down_weight_outer_fp16_c288x2_m160n80k128",
            "pi05_down_weight_outer_w8a16_c288x2_m160n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_fp16_c288x2_m160n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_w8a16_c288x2_m160n80k128",
            "pi05_prefill_kv1_pair_owner_fp16_m288n32x2k2048",
            "pi05_prefill_kv1_pair_owner_w8a16_m288n32x2k2048",
            "pi05_prefill_gate_up_geglu_w8a8_split_c288_m48n32k2048",
            "pi05_all_reduce_residual_xor3_m448n2048",
            "pi05_all_reduce_residual_xor3_m464n2048",
            "pi05_owner_norm_full_residual_m448n2048",
            "pi05_owner_norm_full_residual_m464n2048",
            "pi05_owner_norm_compact_residual_m448n2048",
            "pi05_owner_norm_compact_residual_m464n2048",
            "pi05_xor3_compact_residual_raw_full_m448n2048",
            "pi05_xor3_compact_residual_raw_full_m464n2048",
            "pi05_owner_norm_compact_a8_m448n2048",
            "pi05_owner_norm_compact_a8_m464n2048",
            "pi05_prefill_kv1_direct_cache_m448d256p448",
            "pi05_prefill_kv1_direct_cache_m464d256p464",
            "pi05_prefill_o_weight_outer_fp16_c448x2_m96n80k128",
            "pi05_prefill_o_weight_outer_fp16_c464x2_m80n80k128",
            "pi05_prefill_o_weight_outer_w8a16_c448x2_m96n80k128",
            "pi05_prefill_o_weight_outer_w8a16_c464x2_m80n80k128",
            "pi05_down_weight_outer_fp16_c448x2_m96n80k128",
            "pi05_down_weight_outer_fp16_c464x2_m80n80k128",
            "pi05_down_weight_outer_w8a16_c448x2_m96n80k128",
            "pi05_down_weight_outer_w8a16_c464x2_m80n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_fp16_c448x2_m96n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_fp16_c464x2_m80n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_w8a16_c448x2_m96n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_w8a16_c464x2_m80n80k128",
            "pi05_prefill_kv1_pair_owner_fp16_m448n32x2k2048",
            "pi05_prefill_kv1_pair_owner_fp16_m464n32x2k2048",
            "pi05_prefill_kv1_pair_owner_w8a16_m448n32x2k2048",
            "pi05_prefill_kv1_pair_owner_w8a16_m464n32x2k2048",
            "pi05_prefill_gate_up_geglu_w8a8_split_c448_m48n32k2048",
            "pi05_prefill_gate_up_geglu_w8a8_split_c464_m48n32k2048",
            "pi05_all_reduce_residual_xor3_m320n2048",
            "pi05_all_reduce_residual_xor3_m336n2048",
            "pi05_owner_norm_full_residual_m320n2048",
            "pi05_owner_norm_full_residual_m336n2048",
            "pi05_owner_norm_compact_residual_m320n2048",
            "pi05_owner_norm_compact_residual_m336n2048",
            "pi05_xor3_compact_residual_raw_full_m320n2048",
            "pi05_xor3_compact_residual_raw_full_m336n2048",
            "pi05_owner_norm_compact_a8_m320n2048",
            "pi05_owner_norm_compact_a8_m336n2048",
            "pi05_prefill_kv1_direct_cache_m320d256p320",
            "pi05_prefill_kv1_direct_cache_m336d256p336",
            "pi05_prefill_o_weight_outer_fp16_c320x2_m128n80k128",
            "pi05_prefill_o_weight_outer_fp16_c336x2_m128n80k128",
            "pi05_prefill_o_weight_outer_w8a16_c320x2_m128n80k128",
            "pi05_prefill_o_weight_outer_w8a16_c336x2_m128n80k128",
            "pi05_down_weight_outer_fp16_c320x2_m160n80k128",
            "pi05_down_weight_outer_fp16_c336x2_m160n80k128",
            "pi05_down_weight_outer_w8a16_c320x2_m160n80k128",
            "pi05_down_weight_outer_w8a16_c336x2_m160n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_fp16_c320x2_m160n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_fp16_c336x2_m160n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_w8a16_c320x2_m160n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_w8a16_c336x2_m160n80k128",
            "pi05_prefill_kv1_pair_owner_fp16_m320n32x2k2048",
            "pi05_prefill_kv1_pair_owner_fp16_m336n32x2k2048",
            "pi05_prefill_kv1_pair_owner_w8a16_m320n32x2k2048",
            "pi05_prefill_kv1_pair_owner_w8a16_m336n32x2k2048",
            "pi05_prefill_gate_up_geglu_w8a8_split_c320_m48n32k2048",
            "pi05_prefill_gate_up_geglu_w8a8_split_c336_m48n32k2048",
            "pi05_all_reduce_residual_xor3_m432n2048",
            "pi05_owner_norm_full_residual_m432n2048",
            "pi05_owner_norm_compact_residual_m432n2048",
            "pi05_xor3_compact_residual_raw_full_m432n2048",
            "pi05_owner_norm_compact_a8_m432n2048",
            "pi05_prefill_kv1_direct_cache_m432d256p432",
            "pi05_prefill_o_weight_outer_fp16_c432x2_m112n80k128",
            "pi05_prefill_o_weight_outer_w8a16_c432x2_m112n80k128",
            "pi05_down_weight_outer_fp16_c432x2_m112n80k128",
            "pi05_down_weight_outer_w8a16_c432x2_m112n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_fp16_c432x2_m112n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_w8a16_c432x2_m112n80k128",
            "pi05_prefill_kv1_pair_owner_fp16_m432n32x2k2048",
            "pi05_prefill_kv1_pair_owner_w8a16_m432n32x2k2048",
            "pi05_prefill_gate_up_geglu_w8a8_split_c432_m48n32k2048",
            "pi05_all_reduce_residual_xor3_m304n2048",
            "pi05_owner_norm_full_residual_m304n2048",
            "pi05_owner_norm_compact_residual_m304n2048",
            "pi05_xor3_compact_residual_raw_full_m304n2048",
            "pi05_owner_norm_compact_a8_m304n2048",
            "pi05_prefill_kv1_direct_cache_m304d256p304",
            "pi05_prefill_o_weight_outer_fp16_c304x2_m128n80k128",
            "pi05_prefill_o_weight_outer_w8a16_c304x2_m128n80k128",
            "pi05_down_weight_outer_fp16_c304x2_m160n80k128",
            "pi05_down_weight_outer_w8a16_c304x2_m160n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_fp16_c304x2_m160n80k128",
            "pi05_prefill_gate_up_geglu_weight_outer_w8a16_c304x2_m160n80k128",
            "pi05_prefill_kv1_pair_owner_fp16_m304n32x2k2048",
            "pi05_prefill_kv1_pair_owner_w8a16_m304n32x2k2048",
            "pi05_prefill_gate_up_geglu_w8a8_split_c304_m48n32k2048",
            "wall_oproj_weight_outer_w8a16_c320x2_n64k128",
            "fused_gate_up_swiglu_w8a16_multibase_m192x3n96k128",
            "wall_prefill_qkv_fp16_m448n64k128",
            "wall_prefill_qkv_fp16_m320x2n64k128",
            "wall_prefill_qkv_fp16_tile_owner_m320x2n64k128",
            "wall_prefill_gate_up_swiglu_fp16_m320n112k128",
            "wall_prefill_gate_up_swiglu_fp16_m160x4n96k128",
            "wall_prefill_gate_up_swiglu_fp16_m224x3n96k128",
            "wall_prefill_down_fp16_c320x2_n64k128",
            "wall_prefill_oproj_fp16_c320x2_n64k128",
            "wall_prefill_row_fp16_m160x4n96k128",
            "wall_action_gate_up_fp16_m592n32k128",
            "wall_action_gate_up_fp16_spm_weight_m592n32k128",
            "wall_action_gate_up_fp16_a16resident_spm_weight_m20n32k1024",
            "wall_action_qkv_fp16_partial_mrope_m20n64k128",
            "wall_action_qkv_fp16_owner_partial_mrope_m20n64k128",
            "wall_action_qkv_fp16_partial_mrope_spm_weight_m20n64k128",
            "hyvla_expert_o_ring_postnorm_s51",
            "hyvla_expert_qk_rope_s51",
            "hyvla_expert_qk_rope_norm_s51",
            "hyvla_expert_kv_nvfp4_s51",
            "hyvla_expert_qkv_nvfp4_s51",
            "hyvla_expert_qk_rope_norm_kv_s51",
            "hyvla_nvfp4_resident_swiglu_s51",
            "hyvla_expert_q_nvfp4_s51_n32",
            "hyvla_expert_q_nvfp4_s51_overlap",
            "hyvla_nvfp4_resident_scales_s51",
            "hyvla_nvfp4_resident_scales_swiglu_s51",
            "hyvla_expert_qkv_nvfp4_s51_overlap",
            "hyvla_expert_sdpa_m16_s51",
            "hyvla_expert_o_ring_postnorm_nvfp4_s51_n128_overlap",
            "hyvla_expert_down_nextnorm_nvfp4_s51_overlap",
            "hyvla_nvfp4_gateup_swiglu_s51_overlap",
            "hyvla_nvfp4_resident_swiglu_s51_overlap",
            "hyvla_expert_sdpa_k128_s51",
            "hyvla_vlm_vision_gu176_w8",
            "hyvla_mot_paired_rmsnorm_v32",
            "hyvla_expert_kv_fp16_s51",
            "hyvla_expert_kv_w8_s51",
            "hyvit2_vision_owner_reduce_layernorm_m392n1152",
            "hyvit2_vision_xor3_compact_residual_raw_full_m392n1152",
            "hyvla_expert_gateup_w8_s51_m512n48k128",
            "hyvla_w8_resident_swiglu_s51",
            "hyvla_expert_o_ring_postnorm_w8_s51",
            "hyvla_expert_o_resident_w8_s51",
            "hyvla_nvfp4_gateup_swiglu_s51",
            "hyvla_nvfp4_gateup_resident_s51",
            "hyvla_expert_o_ring_postnorm_nvfp4_s51",
            "hyvla_expert_o_ring_postnorm_nvfp4_s51_n128",
            "hyvla_nvfp4_gateup_swiglu_input_reuse_s51",
            "hyvla_expert_down_nvfp4_s51_n128",
            "hyvla_expert_down_nextnorm_nvfp4_s51",
            "hyvla_expert_down_nextnorm_nvfp4_s51_n128",
            "hyvla_expert_xor3_norm_s51",
            "hyvla_vlm_gateup_swiglu_w8_s240",
            "wall_action_fp16_linear_ring_postnorm_m20n1024k256",
            "wall_action_fp16_linear_ring_postnorm_spm_weight_m20n1024k256",
            "qkv_parallel_w8a16_multibase_m192x3n96k128",
            // Wall fixed routes are supplied by the expansion asset.
            "qkv_parallel_w8a16_multibase_m448n64k128",
            "qkv_parallel_w8a16_multibase_m400n80k128",
            "qkv_parallel_w8a16_multibase_m592n32k128",
            "gate_up_parallel_w8a16_multibase_m592n32k128",
            "wall_ring_postnorm_fp16_partial_bf16_residual_m320n2048",
            "wall_ring_postnorm_fp16_partial_bf16_residual_local_inout_m320n2048",
            "fused_gate_up_swiglu_w8a16_m352n96k128",
            "wall_down_postnorm_m576_fp16_partial_fp16_residual",
            "wall_down_postnorm_m576_bf16_partial_fp16_residual",
            "wall_down_postnorm_m576_fp16_partial_bf16_residual",
            "llm_fp16_16b_prefill_flash_attn_by_mha_wall_window_b3s576h16d80",
            "wall_ring_postnorm_m576_fp16_partial_fp16_residual_local_inout_xor3",
            "wall_ring_postnorm_m576_bf16_partial_fp16_residual_local_inout_xor3",
            "wall_ring_postnorm_m576_fp16_partial_bf16_residual_local_inout_xor3",
            "qkv_parallel_w8a16_multibase_m320x2n64k128",
            "wall_down_weight_outer_w8a16_c320x2_n64k128",
            "fused_gate_up_swiglu_w8a16_m128x5n96k128",
            "qkv_parallel_w8a16_multibase_partial_mrope_m20n64k128_spm_weight",
            "gate_up_parallel_w8a16_multibase_spm_weight_m592n32k128",
            "wall_action_w8a16_n128_linear_ring_postnorm_spm_weight_m20n1024k256",
            "qkv_parallel_w8a16_multibase_owner_partial_mrope_m20n64k128",
            "kv_owner_fanout_m20",
            "qkv_parallel_w8a16_multibase_tile_owner_m320x2n64k128",
            "owner_direct_kv_m640",
            "wall_vision_row_weight_outer_w8a16_m256n80k448",
            "wall_vision_row_weight_outer_w8a16_m352n80k160",
            "wall_action_gate_up_w8a16_a16resident_spm_weight_m20n32k1024",
            "wall_prefill_row_w8a16_m160x4n96k128",
            "wall_prefill_gate_up_swiglu_w8a16_m160x4n96k128",
            "wall_ring_postnorm_pair_tile40_fp16_partial_bf16_residual_local_inout_m320x2n2048",
            "wall_vision_qkv_fp16_multibase_m192x3n96k128",
            "wall_vision_gate_up_swiglu_fp16_multibase_m192x3n96k128",
            "wall_vision_row_weight_outer_fp16_m352n80k160",
            "wall_vision_row_weight_outer_fp16_m256n80k448",
            "qkv_parallel_w8a16_multibase_partial_mrope_m20n64k128",
            "wall_action_w8a16_n128_linear_ring_postnorm_m20n1024k256",
            "wall_ring_postnorm_pair_fp16_partial_bf16_residual_local_inout_m320x2n2048",
            "w8a16_row_weight_reuse_m352n80k256",
        };
        if (access(mixed_lib.c_str(), F_OK) == 0) {
            const auto& mixed_symbols = kernel_manifest_names(mixed_lib);
            for (const char* name : kMixedNames) {
                const std::string kname(name);
                if (mixed_symbols.count(kname) == 0) continue;
                try {
                    CachedKernel cached = load_kernel_from_oplib(
                        mixed_lib, kname);
                    kernels_.erase(kname);
                    kernels_.emplace(kname, std::move(cached));
                    ++mixed_loaded;
                } catch (const std::exception& e) {
                    if (log_at(2)) {
                        std::cerr << "[KernelCache] mixed: failed to load '"
                                  << kname << "' from " << mixed_lib << ": "
                                  << e.what() << std::endl;
                    }
                }
            }
        }
        if (log_at(3) && mixed_loaded != 0) {
            std::cout << "[KernelCache] mixed: loaded " << mixed_loaded
                      << "/" << sizeof(kMixedNames) / sizeof(kMixedNames[0])
                      << " kernels from "
                      << mixed_lib << std::endl;
        }
    }

    {
        const std::set<std::string> expected_paths = {
            main_path, expansion_path};
        std::lock_guard<std::mutex> lock(loaded_oplib_paths_mutex_);
        TORCH_CHECK(
            loaded_oplib_paths_ == expected_paths,
            "[KernelCache] loaded op-lib receipt does not match the frozen "
            "main/expansion roles");
    }

    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    if (log_at(3)) {
        std::cout << "[KernelCache] Loaded " << kernels_.size() << " kernels in "
                  << elapsed << "ms" << std::endl;
    }

    // 填充快速缓存数组 (O(1) 访问)
    populate_fast_cache();

    // Publish only after the complete pointer table is ready. Capability
    // readers acquire this immutable summary instead of racing the raw cache.
    // Commit initialized_ after publication succeeds: constructing the immutable
    // snapshot may throw, and a failed initialization must remain retryable.
    KernelPayloadAvailability::ExtendedPayloads pi05_extended_payloads;
    const size_t extended_begin = static_cast<size_t>(
        KernelId::PI05_DENOISE_KV1_PAIR_OWNER_FP16_M50N32X2K1024);
    for (size_t i = extended_begin; i < static_cast<size_t>(KernelId::_COUNT); ++i) {
        if (fast_cache_[i]) pi05_extended_payloads.set(i - extended_begin);
    }
    // Allocate both immutable summaries before publishing either capability.
    const auto graph_contracts = graph_snapshot_contracts_snapshot();
    payload_availability_.publish(
        fast_cache_[static_cast<size_t>(KernelId::RMS_NORM_SPM_V16)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::RMS_NORM_SPM_V32)] != nullptr,
        fast_cache_[static_cast<size_t>(
            KernelId::W8A16_ROW_WEIGHT_REUSE_M352N80K256)] != nullptr,
        fast_cache_[static_cast<size_t>(
            KernelId::PI05_VISION_OWNER_REDUCE_LAYERNORM_M768N1152)] != nullptr,
        fast_cache_[static_cast<size_t>(
            KernelId::PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M768N1152)] != nullptr,
        fast_cache_[static_cast<size_t>(
            KernelId::PI05_PREFILL_GATE_UP_GEGLU_ONLINE_W8A8_P544_C272)] != nullptr,
        pi05_extended_payloads,
        fast_cache_[static_cast<size_t>(KernelId::QWEN3VL_GATEUP_SWIGLU_GEMV)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::QWEN3VL_QK_NORM_MROPE_D128)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::QWEN3VL_QK_NORM_MROPE_KV_INSERT_D128)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::QWEN3VL_O_RING_NORM_M1_H2560_W8)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::RMS_NORM_NEWTON_SPM)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::RMS_NORM_SPM_NEWTON_V16)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::RMS_NORM_SPM_NEWTON_V32)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::WALL_QKV_W8A16_MULTIBASE_M448N64)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::PI05_ADARMS_NORM_SHIFT_H1024)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::PI05_GATED_RESIDUAL_H1024)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::PARALLEL_LINEAR_NVFP4_V2_M320N64)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::PARALLEL_LINEAR_NVFP4_V2_M384N48)] != nullptr,
        fast_cache_[static_cast<size_t>(KernelId::QWEN3VL_SILU_MUL_EXACT_STRIDED)] != nullptr);
    std::atomic_store_explicit(
        &graph_snapshot_contracts_, graph_contracts, std::memory_order_release);
    initialized_ = true;
}

void KernelCache::require_names(const std::vector<std::string>& names) {
    initialize();
    const auto& main_names = kernel_manifest_names(FROZEN_KERNEL_LIB_PATH());
    const auto& expansion_names = kernel_manifest_names(
        FROZEN_KERNEL_LIB_PATH_EXPANSION());
    for (const auto& name : names) {
        TORCH_CHECK(!name.empty() &&
                        (main_names.count(name) || expansion_names.count(name)),
                    "Selected model profile requires operator '", name,
                    "'; install the matching RhinoForge main/expansion assets before loading weights");
        const auto loaded = kernels_.find(name);
        TORCH_CHECK(loaded != kernels_.end() && loaded->second.kernel != nullptr,
                    "Selected model profile operator '", name,
                    "' was listed but could not be loaded; install a valid matching "
                    "RhinoForge main/expansion assets before loading weights");
    }
}

void KernelCache::populate_fast_cache() {
    // 根据 KernelId 枚举和 KERNEL_ID_NAMES 填充 fast_cache_ 数组
    for (size_t i = 0; i < static_cast<size_t>(KernelId::_COUNT); ++i) {
        const char* name = KERNEL_ID_NAMES[i];
        auto it = kernels_.find(name);
        if (it != kernels_.end()) {
            fast_cache_[i] = it->second.kernel.get();
        } else {
            fast_cache_[i] = nullptr;
            bool optional_id =
                // Includes appended optional high-precision unary IDs.
                (i >= static_cast<size_t>(KernelId::PI05_DENOISE_KV1_PAIR_OWNER_FP16_M50N32X2K1024)) ||
                i == static_cast<size_t>(KernelId::PI05_OWNER_NORM_COMPACT_A8_M400N2048)
                 || i == static_cast<size_t>(KernelId::PI05_PREFILL_GATE_UP_GEGLU_W8A8_SPLIT_M48N32K2048)
                 || i == static_cast<size_t>(KernelId::PI05_PREFILL_GATE_UP_GEGLU_ONLINE_W8A8_P544_C272)
                 || i == static_cast<size_t>(KernelId::PI05_VISION_OWNER_REDUCE_LAYERNORM_M768N1152)
                 || i == static_cast<size_t>(KernelId::PI05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M768N1152)
                 ||
                i == static_cast<size_t>(KernelId::PI05_DENOISE_GATE_UP_GEGLU_NVFP4_M320N64K128)
                 || i == static_cast<size_t>(KernelId::PARALLEL_LINEAR_NVFP4_V2_M320N64)
                 || i == static_cast<size_t>(KernelId::PARALLEL_LINEAR_NVFP4_V2_M384N48)
                 || i == static_cast<size_t>(KernelId::RMS_NORM_NEWTON_SPM)
                 || i == static_cast<size_t>(KernelId::QWEN3VL_GATEUP_SWIGLU_GEMV)
                 || i == static_cast<size_t>(KernelId::QWEN3VL_QK_NORM_MROPE_D128)
                 || i == static_cast<size_t>(KernelId::QWEN3VL_QK_NORM_MROPE_KV_INSERT_D128)
                 || i == static_cast<size_t>(KernelId::QWEN3VL_O_RING_NORM_M1_H2560_W8)
                 || (i >= static_cast<size_t>(
                         KernelId::PL_ACC32_OUT_BF16_M416N48)
                     && i <= static_cast<size_t>(
                         KernelId::RMS_NORM_BF16IN_FP16OUT_V32))
                 || (i >= static_cast<size_t>(
                         KernelId::LLM_ALL_REDUCE_RESIDUAL_RING_LOCAL_SPM)
                     && i <= static_cast<size_t>(
                         KernelId::LLM_ALL_REDUCE_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_SPM))
                 || (i >= static_cast<size_t>(KernelId::RMS_NORM_SPM_V16)
                     && i <= static_cast<size_t>(KernelId::MROPE_SPM_TBL))
                 || i == static_cast<size_t>(
                         KernelId::WALL_QKV_W8A16_MULTIBASE_M448N64)
                 || i == static_cast<size_t>(
                         KernelId::WALL_QKV_W8A16_MULTIBASE_M320X2N64)
                 || i == static_cast<size_t>(
                         KernelId::WALL_DOWN_WEIGHT_OUTER_W8A16_C320X2_N64)
                 || i == static_cast<size_t>(
                         KernelId::WALL_PREFILL_GATE_UP_SWIGLU_W8A16_M128X5N96K128)
                 || i == static_cast<size_t>(
                         KernelId::PI05_GATE_UP_GEGLU_W8A16_M400N80)
                 || i == static_cast<size_t>(
                         KernelId::PI05_ADARMS_NORM_SHIFT_H1024)
                 || i == static_cast<size_t>(
                         KernelId::PI05_GATED_RESIDUAL_H1024)
                 || (i >= static_cast<size_t>(KernelId::PI05_RING_XOR3_M768N1152)
                     && i <= static_cast<size_t>(KernelId::PI05_RING_XOR3_M50N1024))
                 || i == static_cast<size_t>(
                         KernelId::PI05_K_ROPE_INSERT_M50D256P64)
                 || i == static_cast<size_t>(
                         KernelId::PI05_DENOISE_GATEUP_RESIDENT_W8A16_M512N48K128)
                 || i == static_cast<size_t>(
                         KernelId::GR00T_FF1_RESIDENT_W8A16_M512N48K128)
                 || i == static_cast<size_t>(
                         KernelId::GR00T_QKV_MULTIBASE_W8A16_M592N32K128)
                 || i == static_cast<size_t>(
                         KernelId::GR00T_ALL_REDUCE_RESIDUAL_XOR3_M41N1536)
                 || i == static_cast<size_t>(
                         KernelId::PI05_PREFILL_O_LINEAR_XOR3_RMSNORM_M400N2048K256)
                 || i == static_cast<size_t>(
                         KernelId::PI05_DOWN_WEIGHT_OUTER_W8A16_C400X2_M160N80K128)
                 || (i >= static_cast<size_t>(KernelId::PI05_OWNER_NORM_FULL_RESIDUAL_M400N2048)
                     && i <= static_cast<size_t>(KernelId::PI05_XOR3_COMPACT_RESIDUAL_RAW_FULL_M400N2048))
                 || i == static_cast<size_t>(
                         KernelId::PI05_PREFILL_QKV_WEIGHT_OUTER_W8A16_C400X2_M128N64K128)
                 || i == static_cast<size_t>(
                         KernelId::PI05_PREFILL_O_WEIGHT_OUTER_W8A16_C400X2_M128N80K128)
                 || i == static_cast<size_t>(
                         KernelId::PI05_VISION_FC_WEIGHT_OUTER_W8A16_M768N80K128)
                 || i == static_cast<size_t>(
                         KernelId::PI05_DENOISE_Q_PAIR16_ROPE_M50N256K1024)
                 || i == static_cast<size_t>(
                         KernelId::PI05_XOR3_GATED_RESIDUAL_M50N1024)
                 || i == static_cast<size_t>(
                         KernelId::PI05_DENOISE_KV1_DIRECT_CACHE_M50D256P64)
                 || i == static_cast<size_t>(
                         KernelId::PI05_DENOISE_KV1_PAIR_OWNER_W8A16_M50N32X2K1024)
                 || i == static_cast<size_t>(
                         KernelId::PI05_PREFILL_KV1_PAIR_OWNER_W8A16_M400N32X2K2048)
                 || i == static_cast<size_t>(
                         KernelId::PI05_PREFILL_KV1_DIRECT_CACHE_M400D256P400)
                 || i == static_cast<size_t>(
                         KernelId::PI05_PREFILL_GATE_UP_GEGLU_WEIGHT_OUTER_W8A16_C400X2_M160N80K128)
                 || i == static_cast<size_t>(
                         KernelId::PI05_DENOISE_GATE_UP_GEGLU_W8A16_M512N48K128)
                 || i == static_cast<size_t>(
                         KernelId::WALL_OPROJ_WEIGHT_OUTER_W8A16_C320X2_N64)
                 || i == static_cast<size_t>(
                         KernelId::WALL_ACTION_QKV_PARTIAL_MROPE_W8A16_M20N64)
                 || i == static_cast<size_t>(
                         KernelId::WALL_ACTION_LINEAR_RING_POSTNORM_W8A16_M20N1024K256)
                 || i == static_cast<size_t>(
                         KernelId::WALL_RING_POSTNORM_PAIR_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_INOUT_M320X2N2048)
                 || i == static_cast<size_t>(
                         KernelId::WALL_VISION_GATE_UP_SWIGLU_W8A16_MULTIBASE_M192X3N96)
                 || i == static_cast<size_t>(
                         KernelId::WALL_ACTION_QKV_W8A16_MULTIBASE_M592N32)
                 || i == static_cast<size_t>(
                         KernelId::WALL_ACTION_GATE_UP_W8A16_MULTIBASE_M592N32)
                 || i == static_cast<size_t>(
                         KernelId::WALL_ACTION_QKV_PARTIAL_MROPE_W8A16_M20N64_SPM_WEIGHT)
                 || i == static_cast<size_t>(
                         KernelId::WALL_ACTION_GATE_UP_W8A16_MULTIBASE_M592N32_SPM_WEIGHT)
                 || i == static_cast<size_t>(
                         KernelId::WALL_ACTION_LINEAR_RING_POSTNORM_W8A16_M20N1024K256_SPM_WEIGHT)
                 || i == static_cast<size_t>(
                         KernelId::WALL_ACTION_QKV_OWNER_PARTIAL_MROPE_W8A16_M20N64)
                 || i == static_cast<size_t>(
                         KernelId::WALL_ACTION_KV_OWNER_FANOUT_M20)
                 || i == static_cast<size_t>(
                         KernelId::WALL_QKV_W8A16_MULTIBASE_TILE_OWNER_M320X2N64)
                 || i == static_cast<size_t>(
                         KernelId::WALL_PREFILL_OWNER_DIRECT_KV_M640)
                 || i == static_cast<size_t>(
                         KernelId::WALL_VISION_ROW_WEIGHT_OUTER_W8A16_M256N80K448)
                 || i == static_cast<size_t>(
                         KernelId::WALL_VISION_ROW_WEIGHT_OUTER_W8A16_M352N80K160)
                 || i == static_cast<size_t>(
                         KernelId::WALL_ACTION_GATE_UP_W8A16_A16RESIDENT_SPM_WEIGHT_M20N32K1024)
                 || i == static_cast<size_t>(KernelId::WALL_PREFILL_ROW_W8A16_M160X4N96K128)
                 || i == static_cast<size_t>(KernelId::WALL_VISION_QKV_FP16_MULTIBASE_M192X3N96K128)
                 || i == static_cast<size_t>(KernelId::WALL_VISION_GATE_UP_SWIGLU_FP16_MULTIBASE_M192X3N96K128)
                 || i == static_cast<size_t>(KernelId::WALL_VISION_ROW_WEIGHT_OUTER_FP16_M352N80K160)
                 || i == static_cast<size_t>(KernelId::WALL_VISION_ROW_WEIGHT_OUTER_FP16_M256N80K448)
                 || i == static_cast<size_t>(KernelId::WALL_PREFILL_QKV_FP16_M448N64K128)
                 || i == static_cast<size_t>(KernelId::WALL_PREFILL_QKV_FP16_M320X2N64K128)
                 || i == static_cast<size_t>(KernelId::WALL_PREFILL_QKV_FP16_TILE_OWNER_M320X2N64K128)
                 || i == static_cast<size_t>(KernelId::WALL_PREFILL_GATE_UP_SWIGLU_FP16_M320N112K128)
                 || i == static_cast<size_t>(KernelId::WALL_PREFILL_GATE_UP_SWIGLU_FP16_M160X4N96K128)
                 || i == static_cast<size_t>(KernelId::WALL_PREFILL_GATE_UP_SWIGLU_FP16_M224X3N96K128)
                 || i == static_cast<size_t>(KernelId::WALL_PREFILL_DOWN_FP16_C320X2_N64K128)
                 || i == static_cast<size_t>(KernelId::WALL_PREFILL_OPROJ_FP16_C320X2_N64K128)
                 || i == static_cast<size_t>(KernelId::WALL_PREFILL_ROW_FP16_M160X4N96K128)
                 || i == static_cast<size_t>(KernelId::WALL_ACTION_GATE_UP_FP16_M592N32K128)
                 || i == static_cast<size_t>(KernelId::WALL_ACTION_GATE_UP_FP16_SPM_WEIGHT_M592N32K128)
                 || i == static_cast<size_t>(KernelId::WALL_ACTION_GATE_UP_FP16_A16RESIDENT_SPM_WEIGHT_M20N32K1024)
                 || i == static_cast<size_t>(KernelId::WALL_ACTION_QKV_FP16_PARTIAL_MROPE_M20N64K128)
                 || i == static_cast<size_t>(KernelId::WALL_ACTION_QKV_FP16_OWNER_PARTIAL_MROPE_M20N64K128)
                 || i == static_cast<size_t>(KernelId::WALL_ACTION_QKV_FP16_PARTIAL_MROPE_SPM_WEIGHT_M20N64K128)
                 || i == static_cast<size_t>(KernelId::WALL_ACTION_FP16_LINEAR_RING_POSTNORM_M20N1024K256)
                 || i == static_cast<size_t>(KernelId::WALL_ACTION_FP16_LINEAR_RING_POSTNORM_SPM_WEIGHT_M20N1024K256)
                 || i == static_cast<size_t>(KernelId::WALL_PREFILL_GATE_UP_SWIGLU_W8A16_M160X4N96K128)
                 || i == static_cast<size_t>(KernelId::WALL_RING_POSTNORM_PAIR_TILE40_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_INOUT_M320X2N2048)
                 || i == static_cast<size_t>(
                         KernelId::WALL_PREFILL_GATE_UP_SWIGLU_W8A16_M320N112)
                 || i == static_cast<size_t>(KernelId::HYVLA_EXPERT_GATEUP_W8_S51_N48)
                 || i == static_cast<size_t>(KernelId::HYVLA_W8_RESIDENT_SWIGLU_S51)
                 || i == static_cast<size_t>(KernelId::HYVLA_EXPERT_O_RING_POSTNORM_W8_S51)
                 || i == static_cast<size_t>(KernelId::HYVLA_EXPERT_O_RESIDENT_W8_S51)
                 || i == static_cast<size_t>(KernelId::HYVLA_NVFP4_GATEUP_SWIGLU_S51)
                 || i == static_cast<size_t>(KernelId::HYVLA_NVFP4_GATEUP_RESIDENT_S51)
                 || i == static_cast<size_t>(KernelId::HYVLA_EXPERT_O_RING_POSTNORM_NVFP4_S51)
                 || i == static_cast<size_t>(KernelId::HYVLA_EXPERT_O_RING_POSTNORM_NVFP4_S51_N128)
                 || i == static_cast<size_t>(KernelId::HYVLA_NVFP4_GATEUP_SWIGLU_INPUT_REUSE_S51)
                 || i == static_cast<size_t>(KernelId::HYVLA_EXPERT_DOWN_NVFP4_S51_N128)
                 || i == static_cast<size_t>(KernelId::HYVLA_EXPERT_DOWN_NEXTNORM_NVFP4_S51)
                 || i == static_cast<size_t>(KernelId::HYVLA_EXPERT_DOWN_NEXTNORM_NVFP4_S51_N128)
                 || i == static_cast<size_t>(
                         KernelId::WALL_RING_POSTNORM_FP16_PARTIAL_BF16_RESIDUAL_M320N2048)
                 || i == static_cast<size_t>(
                         KernelId::WALL_RING_POSTNORM_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_INOUT_M320N2048)
                 || i == static_cast<size_t>(
                         KernelId::WALL_VISION_GATE_UP_SWIGLU_W8A16_M352N96)
                 || (i >= static_cast<size_t>(
                         KernelId::WALL_DOWN_POSTNORM_M576_FP16_PARTIAL_FP16_RESIDUAL)
                     && i <= static_cast<size_t>(
                         KernelId::WALL_WINDOW_SDPA_BY_MHA_B3S576H16D80))
                 || (i >= static_cast<size_t>(
                         KernelId::WALL_RING_POSTNORM_M576_FP16_PARTIAL_FP16_RESIDUAL_LOCAL_INOUT)
                     && i <= static_cast<size_t>(
                         KernelId::WALL_RING_POSTNORM_M576_FP16_PARTIAL_BF16_RESIDUAL_LOCAL_INOUT))
                 || i == static_cast<size_t>(
                         KernelId::W8A16_ROW_WEIGHT_REUSE_M352N80K256);
            // 这些是热点 kernel，如果找不到则警告
            if (log_at(2) && !optional_id) {
                std::cerr << "[KernelCache] Warning: Fast cache kernel '"
                          << name << "' not found" << std::endl;
            }
        }
    }
    if (log_at(3)) {
        std::cout << "[KernelCache] Populated fast cache with "
                  << static_cast<size_t>(KernelId::_COUNT) << " kernel pointers" << std::endl;
    }
}

::rhino_lkn::Kernel_t* KernelCache::get_kernel(const std::string& kernel_name) {
    auto it = kernels_.find(kernel_name);
    if (it != kernels_.end()) {
        return it->second.kernel.get();
    }

    // 如果没找到，尝试动态加载 (fallback)
    if (log_at(2)) {
        std::cerr << "[KernelCache] Warning: Kernel '" << kernel_name
                  << "' not in cache, loading dynamically..." << std::endl;
    }

    const std::string& main_path = FROZEN_KERNEL_LIB_PATH();
    const std::string& expansion_path = FROZEN_KERNEL_LIB_PATH_EXPANSION();
    const std::string* selected_path = nullptr;
    if (kernel_name.rfind("parallel_linear_wint4a16_pgrp_", 0) == 0) {
        // Main's W4 ABI differs from the signed packed weights used here.
        if (kernel_manifest_names(expansion_path).count(kernel_name)) {
            selected_path = &expansion_path;
        }
    } else if (kernel_manifest_names(main_path).count(kernel_name)) {
        selected_path = &main_path;
    } else if (kernel_manifest_names(expansion_path).count(kernel_name)) {
        selected_path = &expansion_path;
    }
    if (selected_path != nullptr) {
        try {
            CachedKernel cached = load_kernel_from_oplib(
                *selected_path, kernel_name);
            auto* ptr = cached.kernel.get();
            kernels_.erase(kernel_name);
            kernels_.emplace(kernel_name, std::move(cached));
            if (initialized_) {
                const auto contracts = graph_snapshot_contracts_snapshot();
                std::atomic_store_explicit(
                    &graph_snapshot_contracts_, contracts, std::memory_order_release);
            }
            return ptr;
        } catch (...) {
        }
    }

    return nullptr;
}

::rhino_lkn::Program_t* KernelCache::get_program(const std::string& kernel_name) {
    auto it = kernels_.find(kernel_name);
    if (it != kernels_.end()) {
        return it->second.program.get();
    }
    return nullptr;
}
