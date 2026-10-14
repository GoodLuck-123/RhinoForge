"""Four direct-FMB cold installs on CPU doubles; no model or HW evidence."""
import ast
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from test_siglip_cold_profile import _body
from rpu_backend.tests.acceptance import test_pi05_component_install_transaction as pi_install
from rpu_backend.tests.acceptance import test_gemma4_install_transaction as g4_install
from rpu_backend.tests.acceptance.test_pi05_component_install_transaction import native_install
from rpu_backend.tests.acceptance.test_gemma4_install_transaction import install_runtime

ROOT = Path(__file__).resolve().parents[1]
OWNERS = {"gemma": "Gemma", "gemma2": "Gemma2", "adarms": "AdaRMS", "gemma4": "Gemma4"}


def test_actual_four_native_wrappers_fmb_profile_and_strict_reader(tmp_path):
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <limits>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
struct FusedModelBase {
    struct ChunkEnvelope { int64_t max_kv_len=0, chunk=0; };
    struct Impl {
        ChunkEnvelope chunk_envelope_;
        std::vector<int> kv_cost_candidates_;
        std::optional<int> kv_cost_planning_context_;
        int kv_cost_layout_scope_=0;
        std::string kv_cost_snapshot_reason_;
        int64_t planner_cache_epoch_=0;
        std::vector<int> prepared_stage_candidates_, prepared_spm_costs_;
    } impl;
    Impl* pimpl_=&impl;
    int64_t last=0;
    int64_t num_layers() const { return 18; }
    int64_t hidden_size() const { return 2048; }
    int64_t intermediate_size() const { return 16384; }
    int64_t num_q_heads() const { return 8; }
    int64_t num_kv_heads() const { return 1; }
    int64_t head_dim() const { return 256; }
    int64_t attn_tp() const { return 1; }
    int64_t get_last_resolved_chunk_size() const { return last; }
    std::vector<int64_t> kvinsert_cost_weight_identity() const { return {1}; }
    const ChunkEnvelope& chunk_envelope() const { return impl.chunk_envelope_; }
    void set_chunk_envelope(int64_t max_kv_len, int64_t chunk);
    void invalidate_planner_cache();
    std::vector<int64_t> installed_profile_identity() const;
} models[4];
void FusedModelBase::set_chunk_envelope(int64_t max_kv_len, int64_t chunk) __SETTER__
void FusedModelBase::invalidate_planner_cache() __INVALIDATE__
std::vector<int64_t> FusedModelBase::installed_profile_identity() const __PROFILE__
__WRAPPERS__
template<class F> void reject(F run) {
    bool rejected=false;
    try { run(); } catch (const std::runtime_error&) { rejected=true; }
    assert(rejected);
}
int main() {
    using Setter=void(*)(int64_t,int64_t,int64_t);
    Setter setters[]={rpu_gemma_set_chunk_envelope,rpu_gemma2_set_chunk_envelope,
        rpu_adarms_set_chunk_envelope,rpu_gemma4_set_chunk_envelope};
    for (int index=0; index<4; ++index) {
        auto& model=models[index];
        auto set=setters[index];
        assert(model.installed_profile_identity()[8]==0);
        for (int64_t capacity : {31, 2048, 8192}) {
            model.impl.kv_cost_candidates_={7};
            model.impl.kv_cost_planning_context_=8;
            model.impl.kv_cost_layout_scope_=9;
            model.impl.kv_cost_snapshot_reason_="READY";
            set(index+40,capacity,0);
            const auto profile=model.installed_profile_identity();
            assert(profile[8]==capacity && profile[9]==0);
            assert(model.impl.kv_cost_candidates_.empty() && !model.impl.kv_cost_planning_context_);
            assert(model.impl.kv_cost_layout_scope_==0 && model.impl.kv_cost_snapshot_reason_=="NATIVE_STAGE_ORACLE_REQUIRED");
            std::cout << index << ' ';
            for (auto word : profile) std::cout << word << ' ';
            std::cout << '\n';
            reject([&] { set(index+41,capacity,0); });
            reject([&] { set(index+40,0,0); });
            reject([&] { set(index+40,capacity,15); });
            model.last=16;
            reject([&] { set(index+40,capacity+16,0); });
            assert(model.installed_profile_identity()==profile);
            model.last=0;
        }
    }
}
'''
    wrappers = []
    for index, (prefix, name) in enumerate(OWNERS.items()):
        source = (ROOT / f"src/fused/rpu_{prefix}_model.cpp").read_text()
        wrappers.append(f'''struct {name}Registry {{
            static FusedModelBase* get(int64_t handle, const char* caller) {{
                TORCH_CHECK(handle=={index+40});
                assert(std::string(caller)=="rpu_{prefix}_set_chunk_envelope");
                return &models[{index}];
            }}
        }};
        void rpu_{prefix}_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk)
        {_body(source, f"void rpu_{prefix}_set_chunk_envelope(")}
        ''')
    cpp = cpp.replace("__WRAPPERS__", "\n".join(wrappers))
    for marker, signature in (("__INVALIDATE__", "void FusedModelBase::invalidate_planner_cache("), ("__SETTER__", "void FusedModelBase::set_chunk_envelope("),
            ("__PROFILE__", "std::vector<int64_t> FusedModelBase::installed_profile_identity()")):
        cpp = cpp.replace(marker, _body(base, signature))
    source_path, binary = tmp_path / "language_profiles.cpp", tmp_path / "language_profiles"
    source_path.write_text(cpp)
    result = subprocess.run(["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source_path),
        "-o", str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    profiles = [[int(word) for word in line.split()] for line in result.stdout.splitlines()]
    assert len(profiles) == 12


def test_four_int_only_ops_are_separate_from_hot_weight_refresh():
    declarations = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    registrations = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    for prefix in OWNERS:
        assert f'void rpu_{prefix}_set_chunk_envelope(int64_t handle, int64_t max_kv_len, int64_t chunk);' in declarations
        assert f'm.def("{prefix}_set_chunk_envelope", TORCH_FN(rpu_{prefix}_set_chunk_envelope));' in registrations
        source = (ROOT / f"src/fused/rpu_{prefix}_model.cpp").read_text()
        assert source.count("set_chunk_envelope(") == 2
        assert "set_chunk_envelope" not in _body(source, f"void rpu_{prefix}_set_weights(")


@pytest.mark.parametrize("kind", ["gemma", "adarms"])
@pytest.mark.parametrize("failure", [False, True])
def test_real_pi05_install_binds_physical_cache_before_publication_and_cleans_failure(native_install, monkeypatch, kind, failure):
    component = pi_install._COMPONENTS[kind]
    model, state = pi_install._model(), native_install[kind]
    model._rpu_max_seq_len = 96  # Logical RoPE has 32 rows: it is not physical KV capacity.
    original_forward = model.forward
    def bind(handle, capacity, chunk):
        assert state["events"] == [("create", handle), ("set_weights", handle)]
        assert not hasattr(model, component["handle"]) and model.forward is original_forward
        assert (capacity, chunk) == (96, 0)
        state["events"].append(("envelope", handle, capacity))
        if failure:
            raise RuntimeError("cold envelope failure")
    monkeypatch.setattr(torch.ops.rpu, kind + "_set_chunk_envelope", bind)
    if failure:
        with pytest.raises(RuntimeError, match="cold envelope failure"):
            component["install"](model)
        assert not state["alive"] and state["events"][-1] == ("destroy", 1)
        assert not hasattr(model, component["handle"]) and model.forward is original_forward
    else:
        component["install"](model)
        try:
            assert model._rpu_cache.kwargs["max_seq_len"] == 96
            assert state["events"][-1] == ("envelope", 1, 96)
            assert getattr(model, component["handle"]) == 1 and model.forward is not original_forward
        finally:
            pi_install._finalize(model, component)


@pytest.mark.parametrize("kind", ["gemma", "adarms"])
def test_real_pi05_replacement_envelope_failure_preserves_old_owner(native_install, monkeypatch, kind):
    component, model = pi_install._COMPONENTS[kind], pi_install._model()
    component["install"](model)
    old = dict(vars(model))
    state = native_install[kind]
    state["events"].clear()
    def fail(handle, capacity, chunk):
        assert handle == 2 and getattr(model, component["handle"]) == 1
        raise RuntimeError("replacement envelope failure")
    monkeypatch.setattr(torch.ops.rpu, kind + "_set_chunk_envelope", fail)
    try:
        with pytest.raises(RuntimeError, match="replacement envelope failure"):
            component["install"](model)
        assert vars(model) == old and state["alive"] == {1}
        assert state["events"] == [("create",2), ("set_weights",2), ("destroy",2)]
        assert getattr(model, component["finalizer"]).alive
    finally:
        pi_install._finalize(model, component)


@pytest.mark.parametrize("cache_capacity,rope_rows,failure", [(96,32,False), (16,128,False), (96,32,True)])
def test_actual_gemma2_call_expression_and_pending_helper_keep_one_based_rope_bound(monkeypatch, cache_capacity, rope_rows, failure):
    from rpu_backend.adapters.paligemma import gemma2

    events = []
    monkeypatch.setattr(torch.ops.rpu, "gemma2_create", lambda: events.append("create") or 17, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "gemma2_set_weights", lambda *args: events.append("weights"), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "gemma2_destroy", lambda handle: events.append("destroy"), raising=False)
    def bind(handle, capacity, chunk):
        assert (handle, capacity, chunk) == (17, min(cache_capacity, rope_rows-1), 0)
        assert events == ["create", "weights"]
        events.append("envelope")
        if failure: raise RuntimeError("cold envelope failure")
    monkeypatch.setattr(torch.ops.rpu, "gemma2_set_chunk_envelope", bind, raising=False)
    path = ROOT / "python/rpu_backend/adapters/paligemma/gemma2.py"
    tree = ast.parse(path.read_text())
    installer = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "patch_gemma2_model_for_rpu_all_layers_once")
    call, = [node for node in ast.walk(installer) if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name) and node.func.id == "_configure_gemma2_handle"]
    publication = next(node for node in ast.walk(installer) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == "_rpu_gemma2_handle" for target in node.targets))
    assert call.lineno < publication.lineno
    # Evaluate the production install expression itself, not a copied min formula.
    def install():
        return eval(compile(ast.Expression(call), str(path), "eval"), {
            "_configure_gemma2_handle": gemma2._configure_gemma2_handle,
            "weight_args": (), "max_seq": cache_capacity, "cos": SimpleNamespace(shape=(rope_rows,8))})
    if failure:
        with pytest.raises(RuntimeError, match="cold envelope failure"): install()
        assert events == ["create", "weights", "envelope", "destroy"]
    else:
        assert install() == 17
        assert events == ["create", "weights", "envelope"]


@pytest.mark.parametrize("failure", [False, True])
def test_real_gemma4_install_and_original_forward_bind_before_first_query(install_runtime, monkeypatch, failure):
    events = []
    monkeypatch.setattr(g4_install._Model, "_rpu_max_seq", 640, raising=False)
    def bind(handle, capacity, chunk):
        assert install_runtime["events"] == [("create", 1), ("set_weights", 1)]
        assert (handle, capacity, chunk) == (1, 640, 0)
        events.append("envelope")
        if failure: raise RuntimeError("cold envelope failure")
    monkeypatch.setattr(torch.ops.rpu, "gemma4_set_chunk_envelope", bind)
    resolve = torch.ops.rpu.gemma4_resolve_stage_domain
    forward = torch.ops.rpu.gemma4_forward
    def query(*args):
        assert events and events[0] == "envelope"
        events.append("query")
        return resolve(*args)
    def dispatch(*args):
        assert "query" in events
        events.append("forward")
        return forward(*args)
    monkeypatch.setattr(torch.ops.rpu, "gemma4_resolve_stage_domain", query)
    monkeypatch.setattr(torch.ops.rpu, "gemma4_forward", dispatch)
    if failure:
        model = g4_install._Model()
        with pytest.raises(RuntimeError, match="cold envelope failure"):
            g4_install._adapter(model).to_rpu()
        assert not install_runtime["alive"]
        assert not getattr(model, "_rpu_gemma4_ready", False)
        assert events == ["envelope"]
        assert install_runtime["events"][-2:] == [("graph_clear", None), ("destroy", 1)]
    else:
        # Reuse the existing true Python prefill + decode test, with an extra
        # order observer. Native tensors/kernels remain its CPU boundary doubles.
        g4_install.test_prefill_chunk_config_reaches_native_descriptor_and_graph_signature(
            install_runtime, monkeypatch, "auto", 32)
        assert events[0] == "envelope" and events.count("forward") == 2
        assert not install_runtime["alive"]
