"""Exercise host tracing configuration without loading the device runtime."""
from pathlib import Path
import runpy
import shutil
import subprocess

from test_graph_execution_core_budget import _function

ROOT = Path(__file__).resolve().parents[1]


def test_hwperf_accepts_caller_directories_and_checks_runtime(tmp_path):
    compiler = shutil.which("c++")
    assert compiler, "host configuration checks require a C++ compiler"
    source = (ROOT / "src/core/rpu_backend.cpp").read_text()
    setter = _function(source, "void rpu_set_hw_perf_trace(")
    config = _function((ROOT / "src/core/rpu_profile.h").read_text(), "struct HwPerfConfig {")
    code = r'''
#include <atomic>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
''' + config + r''';
std::mutex g_hw_perf_mu;
HwPerfConfig g_hw_perf_cfg;
std::atomic<int64_t> g_hw_perf_dumps_left{0};
std::atomic<bool> g_hw_perf_enabled_fast{false};
bool compatible = true;
bool rpu_hw_perf_runtime_compatible() { return compatible; }
bool log_at(int) { return false; }
''' + setter + r'''
int main(int argc, char** argv) {
    assert(argc == 2);
    namespace fs = std::filesystem;
    fs::current_path(argv[1]);
    assert(g_hw_perf_cfg.output_dir == ".");
    rpu_set_hw_perf_trace(true, ".", 7);
    assert(g_hw_perf_cfg.enabled && g_hw_perf_cfg.output_dir == ".");
    assert(g_hw_perf_dumps_left == 7 && g_hw_perf_enabled_fast);
    fs::create_directory("shared");
    fs::permissions("shared", fs::perms::owner_all | fs::perms::group_read | fs::perms::group_exec);
    fs::create_directory_symlink("shared", "alias");
    rpu_set_hw_perf_trace(true, "alias/nested", 3);
    assert(fs::is_directory("shared/nested") && g_hw_perf_cfg.output_dir == "alias/nested");
    // Failure to create the output directory is reported by the writer.
    std::ofstream("regular-file") << "caller content";
    rpu_set_hw_perf_trace(true, "regular-file/nested", 2);
    assert(g_hw_perf_cfg.output_dir == "regular-file/nested");
    compatible = false;
    bool rejected = false;
    try { rpu_set_hw_perf_trace(true, "rejected", 99); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected && !fs::exists("rejected") && g_hw_perf_dumps_left == 2);
    rpu_set_hw_perf_trace(false, "", -1);
    assert(!g_hw_perf_cfg.enabled && g_hw_perf_cfg.output_dir.empty());
    assert(g_hw_perf_dumps_left == -1 && !g_hw_perf_enabled_fast);
}
'''
    cpp = tmp_path / "config.cpp"
    cpp.write_text(code)
    binary = tmp_path / "config"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(cpp), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(binary), str(tmp_path)], check=True, capture_output=True, text=True)


def test_python_trace_context_uses_current_directory_and_disables_on_error(monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    namespace = runpy.run_path(str(ROOT / "python/rpu_backend/runtime/debug.py"))
    setter = namespace["set_hw_perf_trace"]
    trace = namespace["hw_perf_trace"]
    calls = []
    class Extension:
        def set_hw_perf_trace(self, *args):
            calls.append(args)
    setter.__globals__["_cpp_loaded"] = lambda: True
    setter.__globals__["_cpp"] = lambda: Extension()
    setter(True)
    try:
        with trace():
            raise RuntimeError("test failure")
    except RuntimeError:
        pass
    assert calls == [(True, ".", 32), (True, ".", 32), (False, ".", 32)]
