"""Native cost bindings follow the source release and each locked provider."""
import json
from pathlib import Path
import shutil
import subprocess

from _native_runtime_identity import ROOT, runtime_identity_include
from test_fmb_standalone_layout_rebuild import _definition


def test_each_provider_and_release_change_invalidates_native_cost_identity(tmp_path):
    source = tmp_path / "source"
    (source / "release").mkdir(parents=True)
    for name in ("VERSION", "release/version.cmake", "release/runtime-dependencies.json"):
        shutil.copyfile(ROOT / name, source / name)
    original = json.loads((source / "release/runtime-dependencies.json").read_text())
    implementation = (ROOT / "src/ops/rpu_kvinsert_segment_plan.cpp").read_text()
    functions = "\n".join(_definition(implementation, marker) for marker in (
        "uint64_t dependency_hash_append(", "int64_t current_dependency_identity("))
    values = []
    for changed in (None, "release", "launch", "main", "expansion"):
        dependencies = json.loads(json.dumps(original))
        version = (ROOT / "VERSION").read_text().strip()
        if changed == "release":
            base_version = f"{int(version.split('.', 1)[0]) + 1}.0.0"
            version = base_version + ".dev0"
            dependencies["launch_kernel"].update(
                version=base_version, cmake_version=base_version, release_version=version)
        elif changed == "launch":
            dependencies["launch_kernel"]["commit"] = "a" * 40
        elif changed:
            dependencies["artifacts"]["rhino_ops_" + changed]["sha256"] = "b" * 64
        (source / "VERSION").write_text(version + "\n")
        (source / "release/runtime-dependencies.json").write_text(json.dumps(dependencies))
        include = runtime_identity_include(tmp_path, source)
        generated = (include / "rhino_runtime_identity.h").read_text()
        assert version in generated and dependencies["launch_kernel"]["commit"] in generated
        for asset in dependencies["artifacts"].values():
            assert asset["sha256"] in generated
        cpp, binary = tmp_path / "identity.cpp", tmp_path / "identity"
        cpp.write_text('#include "rhino_runtime_identity.h"\n#include <cstdint>\n#include <limits>\n#include <iostream>\n' +
                       functions + '\nint main(){std::cout << current_dependency_identity();}\n')
        built = subprocess.run(["c++", "-std=c++17", "-I", str(include), str(cpp), "-o", str(binary)],
                               capture_output=True, text=True, timeout=30)
        assert built.returncode == 0, built.stderr
        result = subprocess.run([str(binary)], capture_output=True, text=True, check=True)
        values.append(int(result.stdout))
    assert len(set(values)) == len(values)
    assert all(value > 0 for value in values)
