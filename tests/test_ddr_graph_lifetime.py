"""Execute the real registry with ATen CPU tensors; no native device import.

The same C++ source includes actual RpuKernelGraph lifetime tests in the
separate conformance target. This host executable excludes those Graph calls.
"""
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_graph_owned_ddr_registry_lifetimes_with_real_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    compiler = shutil.which("g++")
    assert compiler, "ATen registry lifetime tests require a host C++ compiler"
    source = ROOT / "python/rpu_backend/tests/conformance/_cpp/test_ddr_graph_lifetime.cpp"
    torch_root = Path(torch.__file__).resolve().parent
    executable = tmp_path / "ddr_registry_lifetime"
    command = [compiler, "-std=c++17", "-O0", "-DRPU_DDR_REGISTRY_HOST_ONLY",
               f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
               "-I", str(ROOT / "src"), "-I", str(torch_root / "include"),
               str(source), "-L", str(torch_root / "lib"), "-ltorch_cpu", "-lc10",
               f"-Wl,-rpath,{torch_root / 'lib'}", "-o", str(executable)]
    built = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, built.stdout + built.stderr
    completed = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result["completed"] and result["scope"] == "ATen CPU registry; Graph lifecycle excluded"
