"""Exercise LingBot2's private uploads and public native publication boundary."""
import ast
from contextlib import contextmanager
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace as NS

import pytest
import torch

from test_fmb_standalone_layout_rebuild import _definition


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def denoise(monkeypatch):
    path = ROOT / "python/rpu_backend/adapters/lingbot_vla_v2/runtime.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "LingbotVlaV2Policy")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_denoise_unroll_run")
    scope = dict(torch=torch, STATE_DIM=55, ACTION_DIM=55, N_ACTION=50,
                 EXP_HID=768, EXP_LAYERS=36, NUM_STEPS=10, PREFIX_POLICY_VERSION=1,
                 rpu_backend=NS(graph=NS(GraphSignature=lambda **kw: kw)))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[method.name]


@pytest.mark.parametrize("inference", [False, True])
def test_private_uploads_refresh_both_inputs_before_capture_and_keep_outputs(
        monkeypatch, denoise, inference):
    pending = []
    calls = []

    @contextmanager
    def capture(signature):
        yield
        pending.pop()()

    policy = NS(
        _denoise_state_host=torch.zeros(1, 64, dtype=torch.float16),
        _denoise_x0_host=torch.zeros(1, 51, 64, dtype=torch.float16),
        _denoise_state=torch.zeros(1, 64, dtype=torch.float16),
        _denoise_x0=torch.zeros(1, 51, 64, dtype=torch.float16),
        _denoise_x_final=torch.empty(1, 51, 64, dtype=torch.float16),
        _prepared_graph_profile=None, _exp=7,
        _cache=NS(k_caches=[], v_caches=[], reset_to_position=lambda pos: None),
        _dgc=NS(capture=capture))
    descriptor = [1, 2, 3]
    plan = NS(graph_key_words=lambda: (4,),
              selected=NS(stage_tuple=NS(physical_descriptor=descriptor)))
    expected = {}

    def native(handle, x0, state, _k, _v, mask, out, dt, length, rope, steps,
               stage=(), inputs_published=False):
        assert handle == 7 and stage is descriptor and inputs_published is True
        assert x0 is policy._denoise_x0 and state is policy._denoise_state
        assert torch.equal(state[:, :55], expected["state"].half())
        assert torch.equal(x0[:, 1:, :55], expected["noise"].half())
        assert not torch.count_nonzero(state[:, 55:])
        assert not torch.count_nonzero(x0[:, :1])
        assert not torch.count_nonzero(x0[:, :, 55:])
        # Result is produced only when capture exits, as with the real Graph.
        value = x0 + state.unsqueeze(1)
        pending.append(lambda: out.copy_(value))
        calls.append(inputs_published)

    monkeypatch.setattr(torch.ops.rpu, "lingbot_v2_moe_denoise_unroll_forward",
                        native, raising=False)
    results = []
    with torch.inference_mode(inference):
        for delta in (0.0, 0.25, 0.0):
            # New, non-contiguous CPU inputs exercise copying and fp16 rounding.
            state = (torch.linspace(-1.01, 1.01, 110).view(1, 110) + delta)[:, ::2]
            noise = (torch.arange(5500).view(1, 50, 110).float() / 5500 + delta)[..., ::2]
            expected.update(state=state, noise=noise)
            value = denoise(policy, state, noise, None, 225, 215, 0, -0.1, 10, plan)
            assert value.shape == (1, 50, 55) and value.dtype == torch.float32
            assert torch.equal(value, (noise.half() + state.half().unsqueeze(1)).float())
            results.append(value)
        assert torch.equal(results[0], results[2])
        assert not torch.equal(results[0], results[1])
        assert results[0].data_ptr() != results[2].data_ptr()
        # A rejected staging copy cannot dispatch a promised native call.
        with pytest.raises(RuntimeError):
            denoise(policy, state, noise[:, :-1], None, 225, 215, 0, -0.1, 10, plan)
    assert calls == [True, True, True] and not pending


def test_dispatch_schema_accepts_old_calls_and_explicit_publication():
    source = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    start = source.index('m.def("lingbot_v2_moe_denoise_unroll_forward(')
    statement = source[start:source.index(");", start)]
    schema = "".join(ast.literal_eval(s) for s in re.findall(r'"(?:\\.|[^"\\])*"', statement))
    calls = []

    def forward(handle, x0, state, keys, values, mask, out, dt, prefix, rope,
                steps, planned_stage_descriptor=(), inputs_published=False):
        calls.append((tuple(planned_stage_descriptor), inputs_published))

    lib = torch.library.Library("lingbot2_publication_test", "DEF")
    try:
        lib.define(schema)
        lib.impl("lingbot_v2_moe_denoise_unroll_forward", forward, "CPU")
        op = torch.ops.lingbot2_publication_test.lingbot_v2_moe_denoise_unroll_forward
        value = torch.zeros(1)
        args = (7, value, value, [], [], None, value, -0.1, 225, 0, 10)
        op(*args)
        op(*args, [11, 12])
        op(*args, [11, 12], True)
    finally:
        lib._destroy()
    assert calls == [((), False), ((11, 12), False), ((11, 12), True)]


def test_native_default_publishes_dirty_inputs_and_opt_in_is_per_call(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    cpp = (ROOT / "src/fused/rpu_lingbot_v2_moe_model.cpp").read_text()
    header = (ROOT / "src/fused/rpu_lingbot_v2_moe_model.h").read_text()
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    def declaration(source, name):
        start = source.index(name)
        return source[start:source.index(";", start) + 1]
    program = r'''
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
namespace c10 {
using Half = float;
struct ScopeExit { std::function<void()> fn; ~ScopeExit() { fn(); } };
template<class F> ScopeExit make_scope_exit(F fn) { return {fn}; }
}
namespace at {
enum { kHalf=1, kPrivateUse1=2 };
using IntArrayRef = std::vector<int64_t>;
struct Tensor {
    std::vector<int64_t> shape;
    std::shared_ptr<std::vector<float>> storage;
    int dtype=kHalf, device_type=kPrivateUse1;
    bool contiguous=true;
    Tensor(std::vector<int64_t> s={}):shape(s),storage(std::make_shared<std::vector<float>>(numel(),0)) {}
    int64_t numel() const { return std::accumulate(shape.begin(),shape.end(),int64_t(1),std::multiplies<int64_t>()); }
    int64_t nbytes() const { return numel()*2; }
    int64_t dim() const { return shape.size(); }
    int64_t size(int i) const { return shape.at(i); }
    int scalar_type() const { return dtype; }
    bool is_contiguous() const { return contiguous; }
    struct Device { int value; int type() const { return value; } };
    Device device() const { return {device_type}; }
    template<class T=void> T* data_ptr() const { return reinterpret_cast<T*>(storage->data()); }
};
}
namespace rhino_lkn { uintptr_t RpuGetDevAddr(void* p) { return reinterpret_cast<uintptr_t>(p); } }
struct Publication { void* ptr; int64_t bytes; float first; };
std::vector<Publication> published;
void rpu_ddr_flush_force_sized(float* p, int64_t bytes) { published.push_back({p,bytes,*p}); }
int forwards=0;
struct CausalDecoderModel {
    template<class... Args> void forward(Args&&...) { ++forwards; }
};
struct LingbotV2MoeExpertModel: CausalDecoderModel {
    bool denoise_unroll_=true, denoise_dt_pinned_=false, denoise_unroll_active_=false;
    int64_t num_steps_=10, chunk_size_=51, action_dim_pad_=64, state_dim_pad_=64;
    c10::Half denoise_dt_=0;
    at::Tensor denoise_x0_ref_,denoise_state_ref_,denoise_final_ref_,denoise_stage_;
    uintptr_t denoise_x0_src_base_=0,denoise_state_src_base_=0,denoise_final_dst_base_=0;
METHOD_DECL
};
METHOD
LingbotV2MoeExpertModel model;
struct LingbotV2MoeRegistry {
    static LingbotV2MoeExpertModel* get(int64_t, const char*) { return &model; }
};
WRAPPER_DECL
WRAPPER
int main() {
    at::Tensor x({1,51,64}), state({1,64}), output({1,51,64}), mask;
    std::vector<at::Tensor> keys, values;
    auto run = [&](bool ready) {
        rpu_lingbot_v2_moe_denoise_unroll_forward(7,x,state,keys,values,mask,output,-0.1,225,0,10,{},ready);
        assert(!model.denoise_unroll_active_);
        assert(model.denoise_x0_ref_.storage==x.storage);
        assert(model.denoise_state_src_base_==reinterpret_cast<uintptr_t>(state.data_ptr()));
    };
    run(true);
    assert(published.empty() && forwards==1);
    // Direct CPU writes have no tensor-version bookkeeping in the native API.
    *x.data_ptr<float>()=3; *state.data_ptr<float>()=5;
    rpu_lingbot_v2_moe_denoise_unroll_forward(7,x,state,keys,values,mask,output,-0.1,225,0,10);
    assert(published.size()==2 && forwards==2);
    assert(published[0].ptr==x.data_ptr() && published[0].bytes==6528 && published[0].first==3);
    assert(published[1].ptr==state.data_ptr() && published[1].bytes==128 && published[1].first==5);
    // Replacing owners still refreshes mutable DMA bases on an opted-in call.
    auto previous=x.storage;
    x=at::Tensor({1,51,64}); state=at::Tensor({1,64});
    run(true);
    assert(published.size()==2 && forwards==3 && model.denoise_x0_ref_.storage!=previous);
    run(false);
    assert(published.size()==4 && published[2].ptr==x.data_ptr());
    // The promise bypasses publication only, never input admission.
    state.dtype=0;
    bool rejected=false;
    try { run(true); } catch(const std::runtime_error&) { rejected=true; }
    assert(rejected && published.size()==4 && forwards==4);
}
'''
    for key, value in {
        "METHOD_DECL": declaration(header, "void denoise_unroll_forward("),
        "METHOD": _definition(cpp, "void LingbotV2MoeExpertModel::denoise_unroll_forward("),
        "WRAPPER_DECL": declaration(decls, "void rpu_lingbot_v2_moe_denoise_unroll_forward("),
        "WRAPPER": _definition(cpp, "void rpu_lingbot_v2_moe_denoise_unroll_forward("),
    }.items():
        program = program.replace(key, value)
    source = tmp_path / "publication.cpp"
    source.write_text(program)
    binary = tmp_path / "publication"
    built = subprocess.run([compiler, "-std=c++17", str(source), "-o", str(binary)],
                           capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
