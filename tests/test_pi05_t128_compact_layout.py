"""Execute the C448 production schedule with checked SPM/DDR lifetimes."""
from pathlib import Path
import re

from test_siglip_attention_dry_live_contract import _function, _run_native

ROOT = Path(__file__).resolve().parents[1]


def test_c448_single_compact_schedule_preserves_both_residuals(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    source = (ROOT / "src/fused/rpu_gemma_model.cpp").read_text()
    signatures = (
        "int64_t pi05_pair_rows(", "int64_t pi05_prefix_rows(",
        "int64_t pi05_pair_slab_bytes(", "int64_t pi05_pair_owner_bytes(",
        "int64_t pi05_pair_owner_elements(", "int64_t pi05_pair_tile_rows(",
        "bool pi05_pair_single_compact(", "bool pi05_pair_slab_compact(",
        "int64_t pi05_pair_spill_elements(", "int64_t pi05_pair_compact_root_bytes(",
        "const char* pi05_pair_kv_norm_name(", "int64_t pi05_a8_bytes(",
        "int64_t pi05_a8_root_bytes(", "bool pi05_pair_scratch_mask(",
        "std::vector<int64_t> pi05_pair_input_compact_arguments(",
        "uint32_t pi05_pair_input_compact_address(",
        "std::vector<int64_t> pi05_pair_spill_arguments(", "int64_t pi05_pair_spill_stride(",
        "std::vector<int64_t> pi05_o_pair_arguments(",
        "std::vector<int64_t> pi05_o_pair_norm_arguments(",
        "std::vector<int64_t> pi05_o_pair_companion_arguments(",
        "std::vector<int64_t> pi05_prefill_gateup_pair_arguments(",
        "std::vector<int64_t> pi05_down_pair_arguments(",
        "std::vector<FmbRouteManifestEntry> pi05_o_pair_routes(",
        "void emit_pi05_o_pair_finish(",
    )
    methods = "\n".join(_function(source, "    " + sig) for sig in signatures)
    # The real owner validation runs on host memory in this CPU-only harness.
    methods = methods.replace("at::kPrivateUse1", "at::kCPU")
    names = set(re.findall(r"\b(?:PI05_\w+_SITE|GEMMA_FINAL_RMSNORM_SITE)\b", methods))
    constants = "\n".join(re.search(rf"^constexpr int64_t {name} = .+;$", source, re.M)[0]
                          for name in sorted(names))
    code = r'''
#include "core/fused_model_base.h"
#include "core/rpu_profile.h"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <limits>
#include <string>
#include <vector>
using namespace v3;
''' + constants + r'''
enum class Pi05PrefillPairedProjection { AttentionOutputRow };
enum class Pi05OwnerNormResidual { Compact };
constexpr int64_t C=448,F=C*4096,D=F/8;
constexpr uint32_t S0=0x100,S1=S0+F,S2=S1+1865728,S3=S2+F,C0=S3+F;
std::vector<uint8_t> spm(C0+D+256);
std::vector<int> ops;
bool a8=false;
void put(uint32_t p,int64_t n,uint8_t x){assert(p+n<=int64_t(spm.size()));std::fill_n(spm.begin()+p,n,x);}
void has(uint32_t p,int64_t n,uint8_t x){assert(p+n<=int64_t(spm.size()));for(int64_t i=0;i<n;++i)assert(spm[p+i]==x);}
void rpu_launch_pi05_prefill_pair_spm_kernel(Pi05PrefillPairedProjection,uint32_t x0,uint32_t x1,
    const at::Tensor&,uint32_t y0,uint32_t y1,const at::Tensor&,int64_t rows){
    assert(rows==C && x0==S3 && x1==S3+D && y0==S0 && y1==S2);
    has(x0,D,21);has(x1,D,22);has(S3+4*D,D,2);
    put(y0,F,10);put(y1,F,20);ops.push_back(1);
}
void norm(uint32_t partial,uint32_t residual,uint32_t raw,uint32_t normalized,bool quant){
    const bool first=partial==S0;
    has(partial,F,first?10:20);has(residual,D,first?1:2);
    assert(raw==(first?S3:C0));
    put(raw,D,first?3:5);put(normalized,quant?C*2080:F,first?4:6);
    ops.push_back(first?2:3);
}
void rpu_launch_pi05_owner_norm_spm_kernel(uint32_t p,uint32_t r,uint32_t raw,uint32_t n,
    uint32_t,Pi05OwnerNormResidual,double,int64_t rows){assert(rows==C);norm(p,r,raw,n,false);}
void rpu_launch_pi05_owner_norm_a8_spm_kernel(uint32_t p,uint32_t r,uint32_t raw,uint32_t n,
    uint32_t,uint32_t sx,double,int64_t rows){assert(rows==C);norm(p,r,raw,n,true);put(sx,C*2+32,30);}
void rpu_launch_spm_scatter_ddr_dma(uint32_t src,const at::Tensor& tensor,int64_t offset,
    int64_t count,int64_t stride,int cores){
    assert(src==S3 && offset==0 && count==D/2 && cores==8);
    assert(stride==tensor.size(1)*512);has(src,D,3);
    auto* p=reinterpret_cast<uint8_t*>(tensor.data_ptr<c10::Half>());
    for(int i=0;i<8;++i)std::memcpy(p+i*stride,spm.data()+src,D);
    ops.push_back(4);
}
void gate(uint32_t y0,uint32_t y1){assert(y0==S2 && y1==S3);put(y0,F,7);put(y1,F,8);ops.push_back(5);}
void rpu_launch_pi05_prefill_gate_up_geglu_weight_outer_w8a16_c400x2_kernel(
    uint32_t x0,uint32_t x1,const at::Tensor&,const at::Tensor&,uint32_t y0,uint32_t y1,
    const at::Tensor&,const at::Tensor&,int64_t rows){assert(rows==C);has(x0,F,4);has(x1,F,6);gate(y0,y1);}
void rpu_launch_pi05_prefill_gate_up_geglu_w8a8_split_kernel(uint32_t x,uint32_t,uint32_t stage,
    const at::Tensor&,const at::Tensor&,uint32_t y0,uint32_t y1,const at::Tensor&,const at::Tensor&,int64_t rows){
    assert(rows==C && x==S1 && stage==S0);has(x,C*2080,4);has(x+C*2080,C*2080,6);
    put(stage,384*2048*2,40);gate(y0,y1);
}
void rpu_launch_pi05_down_pair_spm_kernel(uint32_t x0,uint32_t x1,const at::Tensor&,
    uint32_t y0,uint32_t y1,const at::Tensor&,int64_t rows){assert(rows==C);has(x0,F,7);has(x1,F,8);put(y0,F,9);put(y1,F,10);ops.push_back(6);}
void rpu_launch_ddr_scatter_spm_dma(const at::Tensor& tensor,int64_t offset,int64_t count,
    int64_t stride,uint32_t dst,int cores){
    assert(dst==S3 && offset==0 && count==D/2 && cores==8);has(dst,F,8);
    assert(stride==tensor.size(1)*512);
    const auto* p=reinterpret_cast<const uint8_t*>(tensor.data_ptr<c10::Half>());
    for(int core=0;core<8;++core)for(int64_t i=0;i<D;++i)assert(p[core*stride+i]==3);
    std::memcpy(spm.data()+dst,p,D);ops.push_back(7);
}
void rpu_launch_pi05_compact_residual_spm_kernel(uint32_t partial,uint32_t residual,uint32_t out,int64_t rows){
    assert(rows==C);const bool first=partial==S0;
    assert(residual==(first?S3:C0) && out==(first?S2:S0));
    has(partial,F,first?9:10);has(residual,D,first?3:5);put(out,F,first?11:12);ops.push_back(first?8:10);
}
template<class... A> void rpu_launch_spm_local_copy_dma(A...){assert(false);}
template<class... A> void rpu_launch_pi05_gate_up_geglu_w8a16_spm_kernel(A...){assert(false);}
template<class... A> void rpu_launch_rmsnorm_spm_kernel(A...){assert(false);}
struct Context {
    std::vector<FmbRouteManifestEntry> consumed;
    void consume_physical_route(FmbRouteFamily f,int64_t site,int64_t sel,int64_t flags,
        std::vector<int64_t> args,int64_t invocation=0){consumed.push_back({site,f,sel,flags,args,invocation});}
};
struct Owner {
    int64_t pi05_pair_rows_=C;
    bool pi05_prefill_a8_=false,pi05_prefill_o_pair_=true,pi05_prefill_gateup_pair_=true;
    double eps_=1e-6;
    struct Weights {at::Tensor o_w,o_ws,gate_proj_w,up_proj_w,gate_ws,up_ws,down_proj_w,down_ws;};
    std::vector<Weights> layer_weights_=std::vector<Weights>(18);
    std::vector<std::array<at::Tensor,2>> pi05_a8_scales_=std::vector<std::array<at::Tensor,2>>(18);
    at::Tensor q_ddr_buf_; Context context;
    Context& ctx(){return context;}
    int num_layers() const{return 18;} int num_cores() const{return 8;}
    bool pi05_o_pair_owner() const{return true;}
    bool pi05_pair_fp16() const{return false;}
    void pi05_down_pair_require_graph() const{}
    bool use_pi05_prefill_gateup_pair() const{return true;}
    uint32_t addr(int,const char* name) const{
        if(std::string(name)=="residual1")return S0;
        if(std::string(name)=="pi05_pair_s1")return S1;
        if(std::string(name)=="pi05_pair_s2")return S2;
        if(std::string(name)=="pi05_pair_s3")return S3;
        if(std::string(name)=="pi05_pair_residual")return C0;
        return 0;
    }
    uint32_t layer_addr(int,int,const char*) const{return 0;}
    int consume_rmsnorm_route(int64_t,const ChunkInfo&) const{return 0;}
    void emit_layer_output_dma(int,const ChunkInfo& c,const char* name){
        has(addr(0,name),F,c.idx==0?11:12);ops.push_back(c.idx==0?9:11);
    }
''' + methods + r'''
};
template<class F> void rejects(F f){bool bad=false;try{f();}catch(const c10::Error&){bad=true;}assert(bad);}
int main(){
    at::set_num_threads(1);
    for(bool quant:{false,true})for(int capacity:{896,1024}){
        Owner o;o.pi05_prefill_a8_=quant;
        o.q_ddr_buf_=at::zeros({8,capacity,256},at::TensorOptions().dtype(at::kHalf));
        std::fill(spm.begin(),spm.end(),0);ops.clear();
        assert(o.pi05_pair_compact_root_bytes()==D && o.pi05_pair_tile_rows()==96);
        assert(o.pi05_pair_input_compact_address(0)==C0 && o.pi05_pair_input_compact_address(1)==S3+4*D);
        // A production policy regression back to S3 would overwrite input C1.
        put(C0,D,1);put(S3+4*D,D,2);
        const auto kvroot=std::string(o.pi05_pair_kv_norm_name(false));
        assert(kvroot=="input_norm_kv");put(kvroot=="input_norm_kv"?S1:S3,F,50);has(S3+4*D,D,2);
        // Include a full final M128 SDPA tile, beyond the logical second half.
        put(S3,512*512,21);put(S3+D,512*512,22);put(S3,D,21);has(S3+4*D,D,2);
        o.emit_pi05_o_pair_finish(0);
        assert(ops==std::vector<int>({1,2,3,4,5,6,7,8,9,10,11}));
        const auto expected=o.pi05_o_pair_routes();
        for(const auto& actual:o.context.consumed){
            if(actual.site_id==PI05_GATEUP_PAIR_LINEAR_SITE || actual.site_id==PI05_DOWN_PAIR_LINEAR_SITE || actual.site_id==PI05_DOWN_PAIR_OUTPUT_SITE)continue;
            int matches=0;for(const auto& e:expected)matches+=actual.site_id==e.site_id && actual.family==e.family && actual.selector==e.selector && actual.flags==e.flags && actual.invocation==e.invocation && actual.arguments==e.arguments;
            assert(matches==1);
        }
        int spills=0;for(const auto& r:expected){assert(r.site_id!=PI05_O_PAIR_ROTATE_SITE);spills+=r.site_id==PI05_O_PAIR_SPILL_SITE;}
        assert(spills==2 && o.pi05_pair_spill_stride()==capacity*512);
        rejects([&]{o.pi05_pair_input_compact_address(2);});
        o.q_ddr_buf_=at::empty({7,capacity,256},at::TensorOptions().dtype(at::kHalf));rejects([&]{o.pi05_pair_spill_stride();});
        o.q_ddr_buf_=at::empty({8,880,256},at::TensorOptions().dtype(at::kHalf));rejects([&]{o.pi05_pair_spill_stride();});
    }
    for(int rows:{272,288,304,320,400,416,432}){
        Owner o;o.pi05_pair_rows_=rows;assert(!o.pi05_pair_single_compact());
        assert(o.pi05_pair_compact_root_bytes()==rows*1024);
        assert(std::string(o.pi05_pair_kv_norm_name(false))=="input_norm");
        assert(std::string(o.pi05_pair_kv_norm_name(true))=="input_norm_kv");
        assert(o.pi05_pair_input_compact_arguments(1)==std::vector<int64_t>({rows*256,rows*512,8,1}));
        const auto routes=o.pi05_o_pair_routes();
        assert(std::none_of(routes.begin(),routes.end(),[](const auto& route){return route.site_id==PI05_O_PAIR_SPILL_SITE;}));
    }
}
'''
    _run_native(tmp_path, code, torch)
