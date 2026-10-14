"""Exercise opt-in tensor diagnostics without loading the device backend."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def checksum_binary(tmp_path_factory):
    torch = pytest.importorskip("torch")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    source = (ROOT / "src/graph/graph_runtime_execute.cpp").read_text()
    begin = source.index("        static const bool hcb_checksum =")
    gate_end = source.index("        const std::string hcb_op_name", begin)
    describe = source.index("        auto hcb_checksum_str =", gate_end)
    describe_end = source.index("        const char* hcb_state", describe)
    diagnostic = source[begin:gate_end] + source[describe:describe_end]
    assert "data_ptr" not in diagnostic
    code = '''#include <ATen/ATen.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
std::string diagnostic(const at::Tensor& t) {
''' + diagnostic + '''
    return hcb_checksum ? hcb_checksum_str(t) : std::string{};
}
int main() {
    auto input = at::tensor({1.0f, 2.0f});
    auto saved = input.clone();
    std::cout << diagnostic(input);
    if (!at::equal(input, saved)) return 2;
    auto result = input + 1;
    return at::equal(result, at::tensor({2.0f, 3.0f})) ? 0 : 3;
}
'''
    folder = tmp_path_factory.mktemp("graph-checksum")
    cpp = folder / "checksum.cpp"
    cpp.write_text(code)
    binary = folder / "checksum"
    torch_root = Path(torch.__file__).resolve().parent
    lib = torch_root / "lib"
    command = [compiler, "-std=c++17", "-O0", str(cpp), "-o", str(binary),
        "-I" + str(torch_root / "include"),
        "-I" + str(torch_root / "include/torch/csrc/api/include"),
        "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)),
        "-L" + str(lib), "-Wl,-rpath," + str(lib),
        "-ltorch_cpu", "-lc10"]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return binary


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_checksum_preserves_tensor_and_omits_addresses(checksum_binary, enabled):
    result = subprocess.run([str(checksum_binary)], check=True,
        capture_output=True, text=True,
        env={**os.environ, "RPU_GRAPH_HCB_CHECKSUM": enabled})
    if enabled == "0":
        assert result.stdout == ""
    else:
        assert "shape=[2]" in result.stdout
        assert "sum=3" in result.stdout
        assert "val=" in result.stdout
        assert "dev=" not in result.stdout
        assert "0x" not in result.stdout
