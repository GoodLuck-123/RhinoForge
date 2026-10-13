"""The native tensor registry flushes reads and retains tensor ownership."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_registry_retains_live_tensors_after_clear(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = (Path(__file__).resolve().parents[1] /
              "src/core/rpu_runtime_state.cpp").read_text()
    bodies = source[source.index("void set_debug_export("):]
    program = r'''
#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
namespace c10 { enum class DeviceType { CPU, PrivateUse1 }; }
namespace at {
struct Tensor {
  std::shared_ptr<int> value;
  bool on_device = false;
  bool defined() const { return bool(value); }
  struct Device { c10::DeviceType kind; c10::DeviceType type() const { return kind; } };
  Device device() const { return {on_device ? c10::DeviceType::PrivateUse1 : c10::DeviceType::CPU}; }
  void* data_ptr() const { return value.get(); }
  size_t nbytes() const { return sizeof(int); }
};
}
int flushes = 0;
void* last_pointer = nullptr;
void rpu_ddr_flush_force_sized(void* pointer, size_t size) {
  assert(size == sizeof(int)); ++flushes; last_pointer = pointer;
}
bool g_debug_export_enabled = false;
std::unordered_map<std::string, at::Tensor> g_debug_tensors;
'''
    program += bodies + r'''
int main() {
  assert(!get_debug_export());
  set_debug_export(true);
  g_debug_tensors["activation"] = {std::make_shared<int>(7), true};
  auto retained = get_debug_tensor("activation");
  assert(flushes == 1 && last_pointer == retained.data_ptr());
  assert(*retained.value == 7 && list_debug_tensors().size() == 1);
  clear_debug_tensors();
  assert(list_debug_tensors().empty() && *retained.value == 7);
  assert(!get_debug_tensor("missing").defined() && flushes == 1);
  g_debug_tensors["cpu"] = {std::make_shared<int>(3), false};
  assert(*get_debug_tensor("cpu").value == 3 && flushes == 1);
  set_debug_export(false);
  assert(!get_debug_export() && list_debug_tensors().empty());
}
'''
    cpp, binary = tmp_path / "registry.cpp", tmp_path / "registry"
    cpp.write_text(program)
    subprocess.run([compiler, "-std=c++17", str(cpp), "-o", str(binary)], check=True,
                   capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
