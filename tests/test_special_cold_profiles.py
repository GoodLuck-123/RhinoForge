"""Multi-cache installed envelopes: actual C++ bodies, no backend/device execution."""
import ast
from pathlib import Path
import re
import subprocess

import pytest

from test_siglip_cold_profile import _body

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = (("internvla_nextdit", "InternVLANextDiTRegistry"),
            ("navdp", "NavdpRegistry"), ("gr00t_vl_encoder", "Gr00tVlEncoderRegistry"),
            ("gr00t_dit", "Gr00tDiTRegistry"))


@pytest.mark.parametrize("family,registry", FAMILIES)
def test_actual_special_owner_cold_profile(tmp_path, family, registry):
    fixture = next(node for node in ast.walk(ast.parse(
        (ROOT / "tests/test_siglip_cold_profile.py").read_text()))
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cpp"
                                               for t in node.targets))
    cpp = ast.literal_eval(fixture.value).split("int main() {", 1)[0]
    cpp = "#include <algorithm>\n" + cpp.replace("SigLIPRegistry", registry).replace("rpu_siglip_", f"rpu_{family}_")
    native = (ROOT / f"src/fused/rpu_{family}_model.cpp").read_text()
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    if family == "internvla_nextdit":
        constant, = re.findall(r"#define NEXTDIT_MAX_CROSS_SEQ \d+", native)
        cpp = constant + "\n" + cpp
        cpp = cpp.replace("int64_t handle, int64_t max_kv_len, int64_t chunk) __WRAPPER__",
                          "int64_t handle, int64_t packed_self_capacity, int64_t cross_capacity, int64_t chunk) __WRAPPER__")
        checks = """
    set(128,64,0); assert(model.installed_profile_identity()[8]==128);
    set(128,256,0); assert(model.installed_profile_identity()[8]==256);
    set(128,640,0); assert(model.installed_profile_identity()[8]==320);
    reject([&] { set(0,256,0); }); reject([&] { set(128,0,0); });
    reject([&] { set(-1,256,0); }); reject([&] { set(128,-1,0); });
    reject([&] { set(128,256,15); });
"""
        args = "int64_t self, int64_t cross, int64_t chunk"
        call = "self,cross,chunk"
        refresh = "set(128,640,0)"
    else:
        args, call, refresh = "int64_t capacity, int64_t chunk", "capacity,chunk", "set(768,0)"
        checks = "set(768,0); assert(model.installed_profile_identity()[8]==768);"
        if family == "gr00t_dit":
            method = _body(native.replace("\n    }", "\n}"), "void set_cache_envelope(")
            cpp = cpp.replace("    int64_t last=0;", "    int64_t last=0, action_horizon_=40;\n    bool denoise_ready_=true;\n"
                + "    void set_cache_envelope(int64_t max_kv_len, int64_t chunk) " + method)
            checks += """
    model.action_horizon_=1000; set(64,0); assert(model.installed_profile_identity()[8]==1001);
    model.action_horizon_=40;
    model.denoise_ready_=false; set(64,0); assert(model.installed_profile_identity()[8]==64);
    model.denoise_ready_=true;
"""
        checks += "reject([&] { set(0,0); }); reject([&] { set(-1,0); }); reject([&] { set(768,15); });"
    for marker, source, signature in (
        ("__SETTER__", base, "void FusedModelBase::set_chunk_envelope("),
        ("__INVALIDATE__", base, "void FusedModelBase::invalidate_planner_cache("),
        ("__PROFILE__", base, "std::vector<int64_t> FusedModelBase::installed_profile_identity()"),
        ("__WRAPPER__", native, f"void rpu_{family}_set_chunk_envelope("),
    ):
        cpp = cpp.replace(marker, _body(source, signature))
    cpp += f"""
int main() {{
    auto set=[]({args}) {{ rpu_{family}_set_chunk_envelope(42,{call}); }};
    {checks}
    model.impl.kv_cost_candidates_={{1}}; model.impl.kv_cost_planning_context_=3;
    model.impl.kv_cost_layout_scope_=5; model.impl.kv_cost_snapshot_reason_="READY";
    {refresh};
    auto profile=model.installed_profile_identity();
    assert(model.impl.kv_cost_candidates_.empty() && !model.impl.kv_cost_planning_context_);
    assert(model.impl.kv_cost_layout_scope_==0);
    assert(model.impl.kv_cost_snapshot_reason_=="NATIVE_STAGE_ORACLE_REQUIRED");
    model.last=16; reject([&] {{ {refresh}; }});
    assert(model.installed_profile_identity()==profile);
    for (auto word : profile) std::cout << word << ' ';
}}
"""
    source, binary = tmp_path / "profile.cpp", tmp_path / "profile"
    source.write_text(cpp)
    result = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source), "-o", str(binary)],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert f'm.def("{family}_set_chunk_envelope", TORCH_FN(rpu_{family}_set_chunk_envelope));' in (
        ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    assert f"void rpu_{family}_set_chunk_envelope(" in (ROOT / "src/core/rpu_kernel_decls.h").read_text()
