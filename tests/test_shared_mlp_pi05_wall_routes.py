"""Run the shared MLP emitter with both Wall and Pi0.5 caller contracts."""

from pathlib import Path
import re

from test_rmsnorm_capability_publication_host import _block, _compile_and_run


ROOT = Path(__file__).resolve().parents[1]


def test_shared_mlp_preserves_wall_fusion_and_pi05_residual_routes(tmp_path):
    header = (ROOT / "src/core/fused_model_base.h").read_text()
    source = (ROOT / "src/core/fused_model_base.cpp").read_text()
    start = header.index("    void emit_mlp_pipeline(")
    declaration = header[start:header.index(";", start) + 1]
    implementation = _block(source, "void FusedModelBase::emit_mlp_pipeline(")
    adarms = (ROOT / "src/fused/rpu_adarms_model.cpp").read_text()
    call_start = adarms.index("emit_mlp_pipeline(lw.gate_proj_w")
    adarms_call = adarms[call_start:adarms.index(");", call_start) + 2]
    enums = "\n".join(_block(header, marker) + ";" for marker in (
        "enum class FmbRouteFamily : int",
        "enum class FmbLinearRouteSelector : int",
        "enum class FmbSharedAllReduceRouteSelector",
    ))
    constants = "\n".join(re.findall(
        r"inline constexpr int64_t FMB_SHARED_MLP_\w+\s*=\s*\d+LL;", header
    ))
    program = (f'#include "{ROOT / "src/core/rpu_wall_sorting_config.h"}"\n'
               f'#include "{ROOT / "src/core/rpu_math_precision.h"}"\n') + r'''
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
// Host route probe has no profiler; retain the production emitter unchanged.
#define RECORD_FUNCTION(...) ((void)0)
namespace at {
enum ScalarType { kHalf, kChar };
struct Tensor {
    ScalarType dtype = kHalf;
    bool present = false;
    ScalarType scalar_type() const { return dtype; }
    bool defined() const { return present; }
};
}
namespace c10 { using Half = float; }
enum class ActivationKind { SILU, GELU, NONE };
enum class ValuOpType { SILU, ADD, MUL };
enum class GeluMode { NONE, TANH };
''' + enums + constants + r'''
struct Context { int64_t physical_route_invocation = 7; };
struct Route { FmbRouteFamily family; int64_t site, selector; };
std::vector<Route> routes;
std::vector<std::string> launches;
std::vector<bool> nvfp4_acc32;
uint32_t ring_residual = 0;
int64_t wall_rows = 0;
int64_t silu_elements = 0;
bool ring_xor3 = false;
void consume_shared_runtime_route(Context&, FmbRouteFamily family, int64_t site,
    int64_t selector, int64_t = 0, std::initializer_list<int64_t> = {}, int = 8, int = 8) {
    routes.push_back({family, site, selector});
}
int64_t fmb_ring_all_reduce_route_selector(int64_t, int64_t, bool xor3, int) {
    return xor3 ? 101 : 102;
}
template<class... Args> void rpu_launch_linear_spm_to_spm_acc16_kernel(Args...) { launches.push_back("linear"); }
void rpu_launch_pi05_nvfp4_v2_kernel(uint32_t, const at::Tensor&, uint32_t,
    int64_t, int64_t, int64_t, int, int, const at::Tensor&, uint32_t, int, bool acc32) {
    launches.push_back("nvfp4"); nvfp4_acc32.push_back(acc32);
}
void rpu_launch_wall_action_gate_up_w8a16_multibase_spm_kernel(
    uint32_t,const at::Tensor&,const at::Tensor&,uint32_t,uint32_t,
    const at::Tensor&,const at::Tensor&,uint32_t,uint32_t,
    const WallSortingConfig&,int64_t rows) { launches.push_back("wall");wall_rows=rows; }
template<class... Args> void rpu_launch_eltwise_unary_spm_kernel(Args...) { launches.push_back("unary"); }
template<class... Args> void rpu_launch_eltwise_binary_spm_kernel(Args...) { launches.push_back("mul"); }
void rpu_launch_silu_mul_spm_kernel(uint32_t,uint32_t,uint32_t,int64_t elems,int) {
    launches.push_back("silu_mul"); silu_elements=elems;
}
template<class... Args> void rpu_launch_rhinovla_silu_high_mul_spm_kernel(Args...) { throw std::runtime_error("unexpected HIGH fusion"); }
template<class... Args> void rpu_prepare_ring_all_reduce_input(Args...) { throw std::runtime_error("unexpected topology"); }
template<class... Args> void rpu_launch_linear_spm_to_spm_acc32_out_bf16_kernel(Args...) { throw std::runtime_error("unexpected BF16"); }
template<class... Args> void rpu_launch_all_reduce_bf16_partial_fp16_residual_spm_kernel(Args...) { throw std::runtime_error("unexpected BF16"); }
template<class... Args> void rpu_launch_all_reduce_fp16_partial_bf16_residual_spm_kernel(Args...) { throw std::runtime_error("unexpected BF16"); }
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t, uint32_t residual,
    uint32_t, int64_t, int64_t, int, int, bool xor3) {
    launches.push_back("ring"); ring_residual = residual; ring_xor3 = xor3;
}
class FusedModelBase {
public:
    struct Impl { int64_t hidden_size_ = 1024, intermediate_size_ = 2048; Context ctx_; };
    Impl impl;
    Impl* pimpl_ = &impl;
    int mlp_tp() const { return 8; }
    int num_cores() const { return 8; }
    uint32_t addr(int, const char* name) const {
        return std::string(name) == "residual2" ? 0x500000 : 0x100000;
    }
''' + declaration + r'''
    void run_actual_adarms_call(bool quantized, bool linear_acc32_) {
        struct Weights {
            at::Tensor gate_proj_w, up_proj_w, down_proj_w;
            at::Tensor gate_ws, up_ws, down_ws;
        } lw;
        struct Tables { bool active; bool enabled() const { return active; } };
        const Tables nvfp4_{quantized};
        const int64_t seq_len = 50;
        const int layer_idx = 3;
''' + adarms_call + r'''
    }
};
''' + implementation + r'''
int main() {
    static_assert(FmbLinearRouteSelector::ROW_WEIGHT_REUSE !=
                  FmbLinearRouteSelector::PI05_NVFP4_V2_ACC32);
    FusedModelBase model;
    const at::Tensor w8{at::kChar, true}, fp16{at::kHalf, true};
    // Main's existing Wall call supplies the two fusion booleans after skip_down.
    model.emit_mlp_pipeline(w8, w8, w8, 20, ActivationKind::SILU,
        fp16, fp16, fp16, 0, 0, 0, 0, false, false, false, true, true, true, true);
    assert((launches == std::vector<std::string>{"wall", "silu_mul"}));
    assert(routes.size() == 1 && routes[0].family == FmbRouteFamily::ACTIVATION &&
           routes[0].site == FMB_SHARED_MLP_SILU_MUL_SITE);

    assert(wall_rows==20);
    for(bool half_weights:{false,true}) for(bool enabled:{false,true}) {
        launches.clear();routes.clear();wall_rows=0;
        WallSortingConfig sorting;
        sorting.silu_mul=enabled;sorting.fp16_transfer_mask=half_weights?4:0;
        const auto weight=half_weights?fp16:w8;
        const auto scale=half_weights?at::Tensor{}:fp16;
        model.emit_mlp_pipeline(weight,weight,weight,32,ActivationKind::SILU,
            scale,scale,scale,0,0,0,0,false,false,false,true,true,true,true,
            0,false,false,0,0,sorting);
        if(enabled) {
            assert((launches==std::vector<std::string>{"wall","silu_mul"}));
            assert(wall_rows==32);
        } else {
            assert((launches==std::vector<std::string>{"linear","linear","silu_mul"}));
            assert(wall_rows==0);
        }
    }

    // Small base prefill keeps ordinary gate/up/down and binds the fused
    // activation to its real C336/C384 geometry in the COMPLETE descriptor.
    model.impl.hidden_size_ = 2048;
    model.impl.intermediate_size_ = 11008;
    for(int64_t rows:{336,384}) for(bool half_weights:{false,true}) {
        launches.clear();routes.clear();wall_rows=0;silu_elements=0;
        const auto weight=half_weights?fp16:w8;
        model.emit_mlp_pipeline(weight,weight,weight,rows,ActivationKind::SILU,
            {},{},{},0,0,0,0,false,false,false,true,false,false,true);
        assert((launches==std::vector<std::string>{"linear","linear","silu_mul","linear","ring"}));
        assert(wall_rows==0 && silu_elements==rows*1376);
        assert(routes.size()==5 && routes[2].family==FmbRouteFamily::ACTIVATION &&
            routes[2].site==FMB_SHARED_MLP_SILU_MUL_SITE);
    }
    model.impl.hidden_size_ = 1024;
    model.impl.intermediate_size_ = 4096;
    // Pi0.5's independent options retain its zero-residual address and XOR3.
    for (bool acc32 : {false, true}) {
    launches.clear(); routes.clear(); nvfp4_acc32.clear();
    model.emit_mlp_pipeline(fp16, fp16, fp16, 50, ActivationKind::GELU,
        fp16, fp16, fp16, 256, 512, 768, 3, false, false, acc32, false, false,
        false, false, 0x400000, true, true);
    assert((launches == std::vector<std::string>{"nvfp4", "unary", "nvfp4", "mul", "nvfp4", "ring"}));
    assert((nvfp4_acc32 == std::vector<bool>{acc32, acc32, acc32}));
    assert(ring_residual == 0x400000 && ring_xor3);
    assert(routes.size() == 4);
    for (size_t i = 0; i < 3; ++i)
        assert(routes[i].family == FmbRouteFamily::LINEAR &&
               routes[i].selector == static_cast<int64_t>(acc32
                   ? FmbLinearRouteSelector::PI05_NVFP4_V2_ACC32
                   : FmbLinearRouteSelector::PI05_NVFP4_V2_ACC16));
    assert(routes.back().selector == 101);
    }

    launches.clear(); routes.clear();
    model.emit_mlp_pipeline(fp16, fp16, fp16, 50, ActivationKind::GELU);
    assert((launches == std::vector<std::string>{"linear", "unary", "linear", "mul", "linear", "ring"}));
    assert(ring_residual == 0x500000 && !ring_xor3);
    assert(routes[0].selector == static_cast<int64_t>(FmbLinearRouteSelector::AUTO_TILE));

    // Exercise the production caller as well as the shared emitter. An omitted
    // boolean can otherwise shift the residual and precision arguments.
    for (bool quantized : {false, true}) for (bool acc32 : {false, true}) {
        launches.clear(); routes.clear(); nvfp4_acc32.clear();
        model.run_actual_adarms_call(quantized, acc32);
        const auto selector = quantized
            ? (acc32 ? FmbLinearRouteSelector::PI05_NVFP4_V2_ACC32
                     : FmbLinearRouteSelector::PI05_NVFP4_V2_ACC16)
            : FmbLinearRouteSelector::AUTO_TILE;
        assert(routes.size() == 4);
        for (size_t i = 0; i < 3; ++i)
            assert(routes[i].family == FmbRouteFamily::LINEAR &&
                   routes[i].selector == static_cast<int64_t>(selector));
        assert(launches.front() == (quantized ? "nvfp4" : "linear"));
        assert(ring_residual == 0x500000 && !ring_xor3);
        if (quantized)
            assert((nvfp4_acc32 == std::vector<bool>{acc32, acc32, acc32}));
        else
            assert(nvfp4_acc32.empty());
    }

    bool rejected = false;
    try {
        model.emit_mlp_pipeline(fp16, fp16, fp16, 50, ActivationKind::GELU,
            {}, {}, {}, 0, 0, 0, 0, false, false, false, true, true, false, true);
    } catch (const std::runtime_error&) { rejected = true; }
    assert(rejected);
}
'''
    _compile_and_run(tmp_path, "shared_mlp_pi05_wall", program)


def test_pi05_acc32_rejects_fixed_acc16_geglu_candidate(tmp_path):
    source = (ROOT / "src/fused/rpu_gemma_model.cpp").read_text()
    predicate = _block(source, "    bool pi05_geglu_eligible(")
    program = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
namespace at { enum ScalarType { kHalf, kChar }; }
struct Tensor { at::ScalarType dtype=at::kChar; auto scalar_type() const { return dtype; } };
struct Gemma {
    bool linear_acc32_=false;
    struct LayerWeights { Tensor gate_proj_w, up_proj_w; };
    std::vector<LayerWeights> layer_weights_=std::vector<LayerWeights>(18);
    int num_layers()const{return 18;} int hidden_size()const{return 2048;}
    int intermediate_size()const{return 16384;} int num_q_heads()const{return 8;}
    int num_kv_heads()const{return 8;} int head_dim()const{return 256;} int attn_tp()const{return 8;}
''' + predicate + r'''
};
int main() {
    Gemma model;
    assert(model.pi05_geglu_eligible(400));
    model.linear_acc32_=true;
    assert(!model.pi05_geglu_eligible(400));
    model.linear_acc32_=false;
    model.layer_weights_.back().up_proj_w.dtype=at::kHalf;
    assert(!model.pi05_geglu_eligible(400));
}
'''
    _compile_and_run(tmp_path, "pi05_accumulation_geglu", program)
