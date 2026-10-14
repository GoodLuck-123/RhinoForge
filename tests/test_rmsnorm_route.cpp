#include "core/rpu_kernel_decls.h"

#include <c10/util/Exception.h>

#include <iostream>

namespace {

int failed = 0;

#define EXPECT(condition, message)                                            \
    do {                                                                      \
        if (!(condition)) {                                                   \
            std::cerr << "FAIL: " << (message) << '\n';                     \
            ++failed;                                                        \
        }                                                                     \
    } while (0)

template <typename Fn>
bool throws_c10(Fn&& fn) {
    try {
        fn();
    } catch (const c10::Error&) {
        return true;
    }
    return false;
}

}  // namespace

int main() {
    constexpr uint32_t base = RPU_RMSNORM_CAP_BASE;
    constexpr uint32_t fixed16 = base | RPU_RMSNORM_CAP_V16;
    constexpr uint32_t fixed32 = base | RPU_RMSNORM_CAP_V32;
    constexpr uint32_t automatic = fixed16 | RPU_RMSNORM_CAP_V32;

    static_assert(static_cast<int64_t>(RpuRmsNormSpmRoute::V16) == 2);
    static_assert(static_cast<int64_t>(RpuRmsNormSpmRoute::V32) == 3);

    EXPECT(rpu_resolve_rmsnorm_spm_route(base, 32, 2048) ==
               RpuRmsNormSpmRoute::BASE,
           "BASE never selects a vector kernel");
    EXPECT(rpu_resolve_rmsnorm_spm_route(fixed16, 32, 2048) ==
               RpuRmsNormSpmRoute::V16_GRID8,
           "FIXED16 selects V16_GRID8 even when M is divisible by 32");
    EXPECT(rpu_resolve_rmsnorm_spm_route(fixed32, 320, 2048) ==
               RpuRmsNormSpmRoute::V32_GRID8,
           "FIXED32 selects V32_GRID8 for M320");
    EXPECT(rpu_resolve_rmsnorm_spm_route(fixed32, 576, 1280) ==
               RpuRmsNormSpmRoute::V32_GRID8,
           "FIXED32 selects V32_GRID8 for M576");
    EXPECT(rpu_resolve_rmsnorm_spm_route(fixed32, 336, 2048) ==
               RpuRmsNormSpmRoute::BASE,
           "FIXED32 must not silently select V16 for M336");
    EXPECT(rpu_resolve_rmsnorm_spm_route(automatic, 480, 128) ==
               RpuRmsNormSpmRoute::V32_GRID8,
           "AUTO prefers V32_GRID8 for M480");
    EXPECT(rpu_resolve_rmsnorm_spm_route(automatic, 240, 2048) ==
               RpuRmsNormSpmRoute::V16_GRID8,
           "AUTO selects V16_GRID8 for M240");
    EXPECT(rpu_resolve_rmsnorm_spm_route(automatic, 51, 2048) ==
               RpuRmsNormSpmRoute::BASE,
           "AUTO selects BASE when no vector width divides M");
    EXPECT(rpu_resolve_rmsnorm_spm_route(automatic, 32, 8193) ==
               RpuRmsNormSpmRoute::BASE,
           "AUTO selects BASE when columns exceed the vector width limit");
    EXPECT(rpu_resolve_mixed_rmsnorm_spm_route(336, 2048) ==
               RpuMixedRmsNormSpmRoute::V16,
           "mixed M336 selects its certified V16 kernel");
    EXPECT(rpu_resolve_mixed_rmsnorm_spm_route(320, 2048) ==
               RpuMixedRmsNormSpmRoute::V32,
           "mixed M320 selects its certified V32 kernel");
    EXPECT(rpu_resolve_mixed_rmsnorm_spm_route(576, 1280) ==
               RpuMixedRmsNormSpmRoute::V32,
           "mixed Vision M576 selects its certified V32 kernel");

    EXPECT(throws_c10([] {
               (void)rpu_resolve_rmsnorm_spm_route(/*capability=*/0, 32, 2048);
           }),
           "a capability without BASE fails closed");
    EXPECT(throws_c10([] {
               (void)rpu_resolve_rmsnorm_spm_route(
                   RPU_RMSNORM_CAP_BASE | (1u << 7), 32, 2048);
           }),
           "unknown capability bits fail closed");
    EXPECT(throws_c10([] {
               (void)rpu_resolve_rmsnorm_spm_route(
                   RPU_RMSNORM_CAP_BASE, 0, 2048);
           }),
           "non-positive rows fail closed");
    EXPECT(throws_c10([] {
               (void)rpu_resolve_rmsnorm_spm_route(
                   RPU_RMSNORM_CAP_ALL, 32, 0);
           }),
           "non-positive columns fail closed");
    EXPECT(throws_c10([] {
               (void)rpu_resolve_rmsnorm_spm_route(
                   RPU_RMSNORM_CAP_ALL, 32, 32768);
           }),
           "an FP16 row stride exceeding uint16 fails closed");
    EXPECT(throws_c10([] {
               (void)rpu_resolve_mixed_rmsnorm_spm_route(336, 1280);
           }),
           "unknown mixed RMSNorm geometry fails closed");

    if (failed == 0) {
        std::cout << "RMSNorm route tests passed\n";
    }
    return failed == 0 ? 0 : 1;
}
