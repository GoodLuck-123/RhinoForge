#pragma once
#include "rpu_helpers.h"

// Four query tiles, one local KV head: 8 N-vectors * 256 K * 4 * 32 B.
constexpr int64_t HYVLA_S51_M16_SDPA_TMP_BYTES = 262144;
constexpr int64_t HYVLA_S51_K128_SDPA_TMP_BYTES = 32768;

inline bool rpu_hyvla_s51_key_rows_supported(int64_t key_rows) {
    return key_rows == 243 || key_rows == 259 || key_rows == 275 || key_rows == 291;
}

inline std::optional<SdpaTiling> rpu_hyvla_s51_m16_tiling(
    int64_t query_rows, int64_t key_rows, int64_t declared_tmp_bytes) {
    if (query_rows != 51 || !rpu_hyvla_s51_key_rows_supported(key_rows) ||
        declared_tmp_bytes < HYVLA_S51_M16_SDPA_TMP_BYTES) return std::nullopt;
    const SdpaTiling t{1, 8, 16, 16, 128, 256};
    const auto grid_x = CeilDiv(query_rows, t.tile_m);
    const int64_t required_tmp = t.tile_n_v16 * t.tile_k * grid_x * 32;
    if (grid_x != 4 || grid_x * 2 != 8 ||
        t.tile_m_v16 * t.tile_n_v16 * t.tile_k_v16 > 1024 ||
        GetMhaAttnUnivVlmNeed(1, 8, 16, 8, 8, 8, 4) > SDPA_MAX_VLM_SIZE ||
        required_tmp > declared_tmp_bytes) return std::nullopt;
    return t;
}

// Keep one M64 query tile and two local Q heads. Smaller online-softmax K
// blocks change rounding, so this is an explicit profile candidate, not a
// replacement for the generic selector or an arithmetic-equivalence claim.
inline std::optional<SdpaTiling> rpu_hyvla_s51_k128_tiling(
    int64_t query_rows, int64_t key_rows, int64_t declared_tmp_bytes) {
    if (query_rows != 51 || !rpu_hyvla_s51_key_rows_supported(key_rows) ||
        declared_tmp_bytes < HYVLA_S51_K128_SDPA_TMP_BYTES) return std::nullopt;
    const SdpaTiling t{4, 8, 8, 64, 128, 128};
    const auto grid_x = CeilDiv(query_rows, t.tile_m);
    const int64_t required_tmp = t.tile_n_v16 * t.tile_k * grid_x * 32;
    if (grid_x != 1 || grid_x * 2 != 2 ||
        t.tile_m_v16 * t.tile_n_v16 * t.tile_k_v16 > 1024 ||
        GetMhaAttnUnivVlmNeed(4, 8, 8, 8, 8, 8, 4) > SDPA_MAX_VLM_SIZE ||
        required_tmp > declared_tmp_bytes) return std::nullopt;
    return t;
}

void rpu_launch_sdpa_hyvla_s51_m16(
    const at::Tensor& key, const at::Tensor& value,
    uint32_t q_off, uint32_t output_off, uint32_t tmp_off, uint32_t mask_off,
    int64_t query_rows, int64_t key_rows, int64_t declared_tmp_bytes);

void rpu_launch_sdpa_hyvla_s51_k128(
    const at::Tensor& key, const at::Tensor& value,
    uint32_t q_off, uint32_t output_off, uint32_t tmp_off, uint32_t mask_off,
    int64_t query_rows, int64_t key_rows, int64_t declared_tmp_bytes);
