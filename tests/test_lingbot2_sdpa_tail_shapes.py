"""Run actual bounded LingBot SDPA tail admission and wrapper owner checks."""
from pathlib import Path
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition


ROOT = Path(__file__).resolve().parents[1]


def test_actual_tail_shape_predicate_and_complete_wrapper_bounds(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    ops = (ROOT / "src/ops/rpu_sdpa.cpp").read_text()
    model = (ROOT / "src/fused/rpu_lingbot_v2_moe_model.cpp").read_text()
    actual = "\n".join(_definition(ops, name) for name in (
        "bool rpu_lingbot2_sdpa_tail_t51_key_rows_supported(",
        "void rpu_launch_lingbot2_sdpa_tail_t51_spm_kernel(",
        "void rpu_launch_lingbot2_sdpa_t51_k275_tail_spm_kernel(",
    ))
    predicate = _definition(
        model, "bool LingbotV2MoeExpertModel::use_sdpa_tail_t51(")
    source = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
template<class T> T CeilDiv(T value, T divisor) { return (value + divisor - 1) / divisor; }
namespace c10 { enum class DeviceType { CPU, PrivateUse1 }; enum class Layout { Strided, Sparse };
struct Half { uint16_t storage; }; }
static_assert(sizeof(c10::Half) == 2);
namespace at {
constexpr int kHalf = 1;
struct Device {
  c10::DeviceType kind = c10::DeviceType::PrivateUse1;
  int index = 0;
  c10::DeviceType type() const { return kind; }
  bool operator==(const Device& other) const { return kind == other.kind && index == other.index; }
};
struct Storage { uint64_t bytes; uint64_t nbytes() const { return bytes; } };
struct Tensor {
  bool live = true, contiguous = true;
  int dtype = kHalf, ndim = 7;
  c10::Layout kind = c10::Layout::Strided;
  Device dev;
  std::array<int64_t, 7> shape {1, 64, 1, 8, 8, 16, 16};
  int64_t offset = 0;
  uint64_t backing = 64 * 32768, address = 0x100000;
  bool defined() const { return live; }
  int scalar_type() const { return dtype; }
  Device device() const { return dev; }
  c10::Layout layout() const { return kind; }
  bool is_contiguous() const { return contiguous; }
  int dim() const { return ndim; }
  int64_t size(int n) const { return shape[n]; }
  const auto& sizes() const { return shape; }
  int64_t storage_offset() const { return offset; }
  Storage storage() const { return {backing}; }
  uint64_t nbytes() const { return uint64_t(shape[1]) * 32768; }
  void* data_ptr() const { return reinterpret_cast<void*>(address); }
};
}
uint64_t RpuGetDevAddr(void* pointer) { return reinterpret_cast<uintptr_t>(pointer); }
struct SpmAllocator { static constexpr uint64_t SPM_USABLE = 8387584; };
struct Allocator {
  bool live = true, bad_core7 = false;
  bool is_initialized() const { return live; }
  uint32_t addr(int core, int) const {
    return bad_core7 && core == 7 ? 0xfffff000u : 0x10000000u + uint32_t(core) * 0x1000000u;
  }
} allocator;
#define SPM_ALLOC allocator
enum class SdpaKernelType { FLASH_ATTN_SPM };
struct SdpaConfig { SdpaKernelType type; int h, q, kv, tp, mask; };
struct SdpaTiling { int tile_m_v16, tile_n_v16, tile_k_v16; };
bool wrong_tile = false;
SdpaTiling sdpa_compute_tiling(const SdpaConfig& cfg, int64_t rows) {
  assert(cfg.type == SdpaKernelType::FLASH_ATTN_SPM && cfg.h == 128 &&
         cfg.q == 32 && cfg.kv == 8 && cfg.tp == 8 && cfg.mask == 4 && rows == 51);
  return {wrong_tile ? 3 : 4, 8, 16};
}
enum class KernelId { UNUSED };
constexpr int MASK_2D = 4;
int launches = 0;
int64_t last_keys = 0;
void launch_sdpa_spm_unified_impl(
  const at::Tensor&, const at::Tensor&, int mask, std::optional<double> scale,
  uint32_t, uint32_t, uint32_t, uint32_t, int64_t rows, int64_t q, int64_t kv,
  int64_t h, int64_t keys, int cores, int vtp, int64_t offset, bool narrow,
  bool delivery, const SdpaTiling* tile, std::optional<KernelId> id, bool tail) {
  assert(mask == 4 && !scale && rows == 51 && q == 32 && kv == 8 && h == 128 &&
         cores == 8 && vtp == 8 && !offset && !narrow && !delivery && !tile && !id && tail);
  last_keys = keys; ++launches;
}
class LingbotV2MoeExpertModel {
public:
  bool sparse_sdpa_tail_t51_ = true;
  bool use_sdpa_tail_t51(int64_t rows, int64_t keys) const;
};
''' + actual + "\n" + predicate + r'''
template<class F> void reject(F function) {
  int before = launches; bool threw = false;
  try { function(); } catch (const std::exception&) { threw = true; }
  assert(threw && launches == before);
}
int main() {
  at::Tensor k, v; v.address = 0x400000;
  LingbotV2MoeExpertModel model;
  auto run = [&](int64_t keys, uint32_t mask = 0x40000) {
    rpu_launch_lingbot2_sdpa_tail_t51_spm_kernel(k, v, 256, 0x10000, 0x20000, mask, keys);
  };
  for (int64_t keys : {275, 291, 307, 371, 451}) {
    assert(model.use_sdpa_tail_t51(51, keys));
    assert(!model.use_sdpa_tail_t51(50, keys) && !model.use_sdpa_tail_t51(52, keys));
    k.shape[1] = v.shape[1] = CeilDiv(keys, int64_t{16});
    run(keys); assert(last_keys == keys);
    --k.shape[1]; reject([&] { run(keys); }); ++k.shape[1];
    --v.shape[1]; reject([&] { run(keys); }); ++v.shape[1];
  }
  k.shape[1] = v.shape[1] = 64;
  for (int64_t keys : {-1, 0, 51, 274, 276, 290, 292, 306, 308, 370, 372, 450, 452, 65536}) {
    assert(!model.use_sdpa_tail_t51(51, keys)); reject([&] { run(keys); });
  }
  model.sparse_sdpa_tail_t51_ = false;
  for (int64_t keys : {275, 291, 307, 371, 451}) assert(!model.use_sdpa_tail_t51(51, keys));
  rpu_launch_lingbot2_sdpa_t51_k275_tail_spm_kernel(k, v, 256, 0x10000, 0x20000, 0x40000);
  assert(last_keys == 275);
  constexpr uint32_t near_end = SpmAllocator::SPM_USABLE - 51 * 288 * 2;
  run(275, near_end); reject([&] { run(451, near_end); });
  run(451, SpmAllocator::SPM_USABLE - 51 * 464 * 2);
  reject([&] { run(451, 0x20100); });
  reject([&] { rpu_launch_lingbot2_sdpa_tail_t51_spm_kernel(k, v, 257, 0x10000, 0x20000, 0x40000, 451); });
  allocator.bad_core7 = true; reject([&] { run(451); }); allocator.bad_core7 = false;
  allocator.live = false; reject([&] { run(451); }); allocator.live = true;
  wrong_tile = true; reject([&] { run(451); }); wrong_tile = false;
  for (int fault = 0; fault != 13; ++fault) {
    auto original = k;
    switch (fault) {
      case 0: k.live = false; break;
      case 1: k.dtype = 2; break;
      case 2: k.dev.kind = c10::DeviceType::CPU; break;
      case 3: k.dev.index = 1; break;
      case 4: k.kind = c10::Layout::Sparse; break;
      case 5: k.contiguous = false; break;
      case 6: k.ndim = 6; break;
      case 7: k.shape[4] = 4; break;
      case 8: k.offset = -1; break;
      case 9: k.backing = k.nbytes() - 1; break;
      case 10: k.address = 0; break;
      case 11: k.address += 1; break;
      case 12: k.address = (uint64_t{1} << 40) - 256; break;
    }
    reject([&] { run(451); }); k = original;
  }
}
'''
    cpp = tmp_path / "tail_admission.cpp"
    cpp.write_text(source)
    binary = tmp_path / "tail_admission"
    compiled = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                               str(cpp), "-o", str(binary)], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
