// test_lingbot_v2_moe_cos_sin_offset_static.cpp — COMPILE-TIME contract test.
//
// Device-free by construction: this TU is built as an OBJECT library (compiled,
// never linked, never run), so it cannot touch /dev/rpu. It proves two things
// that only the compiler can prove, and that a runtime test could not without
// an device:
//
//   [1] BACK-COMPAT: every pre-existing 6-argument call site of
//       rpu_lingbot_v2_moe_forward still compiles unchanged, because
//       cos_sin_offset defaults to -1 (= "RoPE base same as position"). This is
//       the mechanical guarantee behind "V1 default behaviour byte-identical".
//   [2] The 7-argument form (explicit RoPE base) compiles and returns Tensor.
//   [3] The 8-argument LingBot2-private form additionally carries an AdaRMS schedule step.
//
// Both checks run in UNEVALUATED contexts (decltype), so no code is emitted and
// nothing is invoked.
#include "rpu_kernel_decls.h"

#include <ATen/ATen.h>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

// [1] 6-arg legacy shape must still compile => the -1 default is truly optional.
//     If someone ever drops the default, THIS static_assert breaks the build
//     rather than silently changing every existing caller's RoPE base.
static_assert(
    std::is_same_v<
        decltype(rpu_lingbot_v2_moe_forward(
            std::declval<int64_t>(),
            std::declval<const at::Tensor&>(),
            std::declval<std::vector<at::Tensor>>(),
            std::declval<std::vector<at::Tensor>>(),
            std::declval<const std::optional<at::Tensor>&>(),
            std::declval<int64_t>())),
        at::Tensor>,
    "rpu_lingbot_v2_moe_forward must remain callable with 6 args "
    "(cos_sin_offset defaults to -1 = same as position); a missing default "
    "would silently repoint the RoPE base of every existing call site.");

// [2] 7-arg form: explicit RoPE position base, decoupled from the KV offset.
static_assert(
    std::is_same_v<
        decltype(rpu_lingbot_v2_moe_forward(
            std::declval<int64_t>(),
            std::declval<const at::Tensor&>(),
            std::declval<std::vector<at::Tensor>>(),
            std::declval<std::vector<at::Tensor>>(),
            std::declval<const std::optional<at::Tensor>&>(),
            std::declval<int64_t>(),
            std::declval<int64_t>())),
        at::Tensor>,
    "rpu_lingbot_v2_moe_forward must accept a trailing cos_sin_offset "
    "(RoPE position base) while `position` keeps driving the KV-insert offset.");

// [3] 8-arg form: controlled direct-schedule selector, with both older forms intact.
static_assert(
    std::is_same_v<
        decltype(rpu_lingbot_v2_moe_forward(
            std::declval<int64_t>(),
            std::declval<const at::Tensor&>(),
            std::declval<std::vector<at::Tensor>>(),
            std::declval<std::vector<at::Tensor>>(),
            std::declval<const std::optional<at::Tensor>&>(),
            std::declval<int64_t>(),
            std::declval<int64_t>(),
            std::declval<int64_t>())),
        at::Tensor>,
    "rpu_lingbot_v2_moe_forward must accept a trailing AdaRMS schedule step.");

// [4] Parity with the precedent it mirrors: causal_decoder_forward's trailing
//     cos_sin_offset. If that precedent's shape moves, this flags it.
static_assert(
    std::is_same_v<
        decltype(rpu_causal_decoder_forward(
            std::declval<int64_t>(),
            std::declval<const at::Tensor&>(),
            std::declval<at::TensorList>(),
            std::declval<at::TensorList>(),
            std::declval<const std::optional<at::Tensor>&>(),
            std::declval<int64_t>(),
            std::declval<bool>())),
        at::Tensor>,
    "causal_decoder_forward precedent: trailing optionals incl. cos_sin_offset.");

}  // namespace
