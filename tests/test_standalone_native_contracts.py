"""Run standalone native contracts without loading the device runtime."""
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("target,include", [
    ("test_pi05_execution_topology", "core"),
    ("test_lingbot_v2_moe_select", "fused"),
])
def test_standalone_native_contract(target, include, tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    output = tmp_path / target
    subprocess.run([compiler, "-std=c++17", "-I", str(ROOT / "src" / include),
                    str(ROOT / "tests" / f"{target}.cpp"), "-o", str(output)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(output)], check=True, capture_output=True, text=True)
