"""Run the production RTC admission and row/DMA routing bodies without a board."""
from pathlib import Path
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition


ROOT = Path(__file__).resolve().parents[1]


def test_native_rtc_admission_row_groups_and_mutable_prefix(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    source = (ROOT / "src/fused/rpu_rhino_vla_model.cpp").read_text()
    methods = "\n".join(_definition(source, prefix) for prefix in (
        "void set_denoise_loop_rtc_tables(",
        "void prepare_rtc(",
        "std::vector<int64_t> planner_cache_identity() const",
        "template <typename Emit>\n    void for_each_rtc_group(",
        "void emit_conditioned_norm_shift(",
        "void emit_rtc_prefix_clamp(",
    ))
    program = r'''
#include <cassert>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <tuple>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("rejected"); } while (0)
constexpr int DWIDTH=2;
namespace c10 { using Half = float; }
namespace at {
enum { kHalf=1, kPrivateUse1=2 };
using IntArrayRef = std::vector<int64_t>;
struct Device { int kind=kPrivateUse1; int type() const { return kind; } };
struct Tensor {
    IntArrayRef shape; int dtype=kHalf; Device dev; bool valid=true, contiguous=true;
    Tensor(IntArrayRef sizes={}): shape(sizes) {}
    bool defined() const { return valid; }
    Device device() const { return dev; }
    int scalar_type() const { return dtype; }
    bool is_contiguous() const { return contiguous; }
    const IntArrayRef& sizes() const { return shape; }
    void* data_ptr() const { return const_cast<Tensor*>(this); }
};
}
bool active_graph=false, blocked_thread=false;
struct RpuExecutionCoordinator {
    static void require_graph_quiescent(const char*) { TORCH_CHECK(!active_graph); }
    static void check_current_thread_execution_allowed(const char*) { TORCH_CHECK(!blocked_thread); }
};
namespace rhino_lkn { uint64_t RpuGetDevAddr(void* p) { return reinterpret_cast<uint64_t>(p); } }
struct FusedModelBase { std::vector<int64_t> planner_cache_identity() const { return {1, 42}; } };
enum class FmbRouteFamily { MUTABLE_DMA };
enum class RhinoMutableDmaRoute { DDR_BROADCAST_TO_SPM };
constexpr int64_t RHINO_RTC_PREFIX_DMA_SITE=91;
std::vector<int64_t> consumed;
struct Context {
    void consume_physical_route(FmbRouteFamily, int64_t site, int64_t, int,
                                std::vector<int64_t> args) {
        assert(site==RHINO_RTC_PREFIX_DMA_SITE); consumed=args;
    }
};
uint64_t* dma_base=nullptr;
int64_t dma_count=-1;
void rpu_launch_ddr_broadcast_spm_dma_mutable(uint64_t* base, int64_t offset,
        int64_t count, uint32_t target, int cores) {
    assert(offset==0 && target==4096 && cores==1);
    dma_base=base; dma_count=count;
}
struct Model : FusedModelBase {
    struct Config { bool precompute_adarms=true, skip_adarms_gemv=false,
        adarms_resident=false, expert_fusions=false, packed_qkv=false,
        precompute_time_proj=false, fold_action_time_in=false; } cold_config_;
    bool rtc_mode_=false, denoise_loop_weights_ready_=true, loop_mode_=true,
        dt_pinned_=true, direct_action_input_=true, expert_w8a16_=false,
        expert_cond_w8a16_=false, full_action_w8a16_=false, high_precision_=false,
        denoise_adarms_tables_ready_=false, fast_replay_skip_layer_loop_=true;
    int64_t rtc_delay_=0, num_steps_=10, action_horizon_=30, suffix_len_=31,
        action_dim_=96, action_dim_pad_=96, state_dim_=96, state_dim_pad_=96,
        rope_position_=75, last_chunk=0;
    float dt_=-0.1f;
    at::Tensor adarms_pair_table_, adarms_final_table_;
    uint64_t adarms_pair_table_src_base_=0, adarms_final_table_src_base_=0,
        action_prefix_src_base_=0;
    int generation=0;
    Context context;
    std::vector<std::vector<int64_t>> norms;
    int64_t num_layers() const { return 18; }
    int64_t hidden_size() const { return 1024; }
    int64_t intermediate_size() const { return 3072; }
    int64_t num_q_heads() const { return 16; }
    int64_t num_kv_heads() const { return 8; }
    int64_t head_dim() const { return 128; }
    int64_t attn_tp() const { return 8; }
    int64_t get_last_resolved_chunk_size() const { return last_chunk; }
    void invalidate_model_state() { ++generation; }
    Context& ctx() { return context; }
    uint32_t addr(int core, const char*) { assert(core==0); return 4096; }
    void emit_norm_shift(uint32_t input, uint32_t output, uint32_t scale,
                         uint32_t shift, int64_t rows) {
        norms.push_back({input, output, scale, shift, rows});
    }
METHODS
};
template<class F> void reject(F fn) {
    bool rejected=false; try { fn(); } catch (const std::runtime_error&) { rejected=true; }
    assert(rejected);
}
int main() {
    at::Tensor pair({7,10,18,3,6144}), final({7,10,3,3072});
    for (int failure=0; failure<18; ++failure) {
        Model m; auto p=pair; auto f=final;
        switch(failure) {
            case 0: m.last_chunk=31; break;
            case 1: m.rtc_mode_=true; break;
            case 2: m.action_horizon_=50; break;
            case 3: m.num_steps_=9; break;
            case 4: m.dt_=0.1f; break;
            case 5: m.cold_config_.adarms_resident=true; break;
            case 6: m.cold_config_.expert_fusions=true; break;
            case 7: m.expert_w8a16_=true; break;
            case 8: m.cold_config_.precompute_adarms=false; break;
            case 9: p.shape={7,10,3,18,6144}; break;
            case 10: f.shape={10,3,3072}; break;
            case 11: p.contiguous=false; break;
            case 12: active_graph=true; break;
            case 13: blocked_thread=true; break;
            case 14: m.cold_config_.packed_qkv=true; break;
            case 15: m.cold_config_.precompute_time_proj=true; break;
            case 16: m.cold_config_.fold_action_time_in=true; break;
            case 17: m.dt_=-0.2f; break;
        }
        reject([&]{m.set_denoise_loop_rtc_tables(p,f);});
        assert(m.generation==0 && !m.denoise_adarms_tables_ready_);
        active_graph=blocked_thread=false;
    }
    Model m;
    reject([&]{m.prepare_rtc(0);});
    m.set_denoise_loop_rtc_tables(pair,final);
    assert(m.generation==1 && m.rtc_mode_ && m.denoise_adarms_tables_ready_);
    reject([&]{m.set_denoise_loop_rtc_tables(pair,final);});
    auto identity0=m.planner_cache_identity();
    for(int delay=0;delay<=6;++delay) {
        m.prepare_rtc(delay);
        assert(m.generation==1); // A semantic selection must preserve captured owners.
        auto identity=m.planner_cache_identity(); assert(identity.back()==delay);
        m.norms.clear();
        m.emit_conditioned_norm_shift(100,200,300,400,31,12288);
        assert(m.norms.size()==(delay==0 ? 2 : 3));
        assert(m.norms.front()==std::vector<int64_t>({100,200,300,400,1}));
        if(delay) assert(m.norms[1]==std::vector<int64_t>({2148,2248,12588,12688,delay}));
        assert(m.norms.back()==std::vector<int64_t>({100+(1+delay)*2048,
            200+(1+delay)*2048,24876,24976,30-delay}));
        dma_base=nullptr; consumed.clear();
        m.action_prefix_src_base_=1000+delay;
        m.emit_rtc_prefix_clamp();
        if(delay) {
            assert(dma_base==&m.action_prefix_src_base_ && dma_count==delay*96);
            assert(consumed==std::vector<int64_t>({delay*96}));
            // Captured DMA stores a pointer to the current source address.
            m.action_prefix_src_base_=9000+delay; assert(*dma_base==9000+delay);
        } else assert(dma_base==nullptr && consumed.empty());
    }
    m.prepare_rtc(0); assert(m.planner_cache_identity()==identity0 && m.generation==1);
    reject([&]{m.prepare_rtc(-1);}); reject([&]{m.prepare_rtc(7);});
    assert(m.rtc_delay_==0 && m.generation==1);
    reject([&]{m.emit_conditioned_norm_shift(100,200,300,400,30,12288);});
    Model ordinary;
    ordinary.emit_conditioned_norm_shift(100,200,300,400,31,12288);
    assert(ordinary.norms==std::vector<std::vector<int64_t>>({{100,200,300,400,31}}));
}
'''.replace("METHODS", methods)
    path = tmp_path / "rtc.cpp"
    path.write_text(program)
    binary = tmp_path / "rtc"
    subprocess.run([compiler, "-std=c++17", "-O0", str(path), "-o", str(binary)],
                   check=True, capture_output=True, text=True)
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
