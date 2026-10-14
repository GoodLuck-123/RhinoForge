"""Execute the exact G05 attention-gate launcher/cold/COMPLETE/fallback bodies."""
from pathlib import Path

from test_g05_q_gate_native import _compile_run
from test_g05_vision_owner_model import _block

ROOT = Path(__file__).resolve().parents[1]
CALL = 'rpu_launch_g05_attn_sigmoid_round_mul_spm_kernel'
ID = 'G05_ATTN_SIGMOID_ROUND_MUL_M32D256'


def test_attn_sigmoid_mul_native_exact_spans_and_registers(tmp_path):
    source = (ROOT/'src/ops/rpu_eltwise_unary.cpp').read_text()
    body = _block(source, 'void '+CALL+'(')
    cpp = r'''
#include <cassert>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(x,...) do{if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10{struct Half{uint16_t x;explicit Half(double v):x(v==1?0x3c00:0){}};}
struct RpuKernelGraph{enum class State{RECORDING,REPLAYING,READY};State status=State::RECORDING;static RpuKernelGraph&active(){static RpuKernelGraph g;return g;}State state()const{return status;}};
struct SpmAllocator{static constexpr uint64_t SPM_USABLE=8388608;bool live=true;bool is_initialized()const{return live;}uint32_t addr(int,int)const{return 0x20000000;}}spm;
#define SPM_ALLOC spm
constexpr int kKernelRegOk=0;enum class KernelId{G05_ATTN_SIGMOID_ROUND_MUL_M32D256};
struct KernelCache{bool loaded=true;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId){return loaded;}};
int gets=0,launches=0,reject_reg=-1;bool missing=false;
struct Kernel{std::map<uint32_t,uint16_t>regs;void reset_regs(){regs.clear();}int set_regs(uint32_t r,uint16_t v){if(int(r)==reject_reg)return 1;assert(!regs.count(r));regs[r]=v;return 0;}}kernel;
Kernel*get(KernelId){++gets;return missing?nullptr:&kernel;}
#define GET_KERNEL(id) get(id)
struct Queue{bool broadcast=false;void set_broadcast_mode(bool b){broadcast=b;}void enqueu_kernel(Kernel&k,std::vector<uint16_t>grid,std::vector<uint8_t>cores){assert(broadcast);assert(grid==std::vector<uint16_t>({8,1,1}));assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));assert(&k==&kernel);++launches;}}queue;
Queue*get_queue(int n){assert(n==8);return &queue;}
#define GET_QUEUE(n) get_queue(n)
void __CALL__(uint32_t gate,uint32_t output)__BODY__
template<class F>void reject(F f){const int before=launches;try{f();assert(false);}catch(const std::runtime_error&){}assert(launches==before);}
int main(){
 const uint32_t gate=0x20010000,output=0x20018000;
 for(auto state:{RpuKernelGraph::State::RECORDING,RpuKernelGraph::State::REPLAYING}){
  RpuKernelGraph::active().status=state;__CALL__(gate,output);
  const std::map<uint32_t,uint16_t>expected{{0,1024},{1,1024},{4,0},{5,0x2001},{6,0},{7,0x2001},{8,2048},{9,0},{12,0x8000},{13,0x2001},{56,0x3c00},{57,0x3c00},{58,0x3c00},{59,0},{60,0},{61,1},{62,0x2692},{63,0}};
  assert(kernel.regs==expected);
 }
 const int before=gets;
 for(auto bad:std::vector<uint32_t>{0,0x1fffff00,gate+1,0x207fc100,0x20800000}){reject([&]{__CALL__(bad,output);});reject([&]{__CALL__(gate,bad);});}
 for(uint32_t bad:{gate,gate+256,gate-256,gate+16128})reject([&]{__CALL__(gate,bad);});
 RpuKernelGraph::active().status=RpuKernelGraph::State::READY;reject([&]{__CALL__(gate,output);});RpuKernelGraph::active().status=RpuKernelGraph::State::RECORDING;
 spm.live=false;reject([&]{__CALL__(gate,output);});spm.live=true;
 KernelCache::instance().loaded=false;reject([&]{__CALL__(gate,output);});KernelCache::instance().loaded=true;assert(gets==before);
 missing=true;reject([&]{__CALL__(gate,output);});missing=false;
 reject_reg=13;reject([&]{__CALL__(gate,output);});reject_reg=-1;
 __CALL__(0x207fc000,gate);assert(launches==3); // full last span exactly fits.
}
'''.replace('__CALL__', CALL).replace('__BODY__', body)
    _compile_run(tmp_path, 'attn_gate_native', cpp)


def test_attn_sigmoid_mul_cold_complete_and_fallback(tmp_path):
    source=(ROOT/'src/fused/rpu_qwen3_5_model.cpp').read_text()
    header=(ROOT/'src/fused/rpu_qwen3_5_model.h').read_text()
    methods=[
        ('void prepare_action_attn_sigmoid_mul()', 'void Qwen3_5Model::prepare_action_attn_sigmoid_mul('),
        ('bool use_action_attn_sigmoid_mul(int64_t rows) const', 'bool Qwen3_5Model::use_action_attn_sigmoid_mul('),
        ('std::vector<int64_t> action_attn_sigmoid_mul_arguments() const', 'std::vector<int64_t> Qwen3_5Model::action_attn_sigmoid_mul_arguments('),
        ('void launch_action_attn_sigmoid_mul(int layer,const ChunkInfo& chunk)', 'void Qwen3_5Model::launch_action_attn_sigmoid_mul('),
        ('bool use_action_nvfp4(int64_t rows) const', 'bool Qwen3_5Model::use_action_nvfp4('),
    ]
    definitions='\n'.join(decl+_block(source,marker) for decl,marker in methods)
    start=source.index('    if (use_action_attn_sigmoid_mul(seq_len))')
    caller=source[start:source.index('    // O-proj row-partition',start)]
    manifest='if (use_action_attn_sigmoid_mul(chunk.len))'+_block(source,'if (use_action_attn_sigmoid_mul(chunk.len))')
    layout='if (action_mode_ && action_attn_sigmoid_mul_available_)'+_block(source,'if (action_mode_ && action_attn_sigmoid_mul_available_)')
    start=header.index('            identity.push_back(static_cast<int64_t>(action_attn_sigmoid_mul_available_));')
    identity=header[start:header.index('            identity.push_back(static_cast<int64_t>(action_nvfp4_swiglu_available_));',start)]
    assert 'prepare_action_attn_sigmoid_mul();' in _block(source,'void Qwen3_5Model::set_action_io_weights(')
    reset='action_attn_sigmoid_mul_available_ = false;'
    assert reset in _block(source,'void Qwen3_5Model::set_weights(')
    cpp=r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(x,...) do{if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10{struct Half{float value;explicit Half(float v):value(v){}};}
namespace detail{int64_t layout_mix(int64_t a,int64_t b){return uint64_t(a)*131+uint64_t(b);}}
enum class KernelId{G05_ATTN_SIGMOID_ROUND_MUL_M32D256};
struct KernelCache{bool available=true;int queries=0;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId){++queries;return available;}};
enum class FmbRouteFamily{ACTIVATION,LINEAR};enum class ValuOpType{SIGMOID,MUL};
constexpr int64_t QWEN35_ACTION_ATTN_SIGMOID_MUL_SITE=8546526474721653363LL;
struct ChunkInfo{int64_t len=32,idx=0,offset=0;};
struct Route{int64_t site;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t>args;int64_t invocation;};
std::vector<std::string>calls;
void rpu_launch_g05_attn_sigmoid_round_mul_spm_kernel(uint32_t gate,uint32_t out){assert(gate==0x10000&&out==0x18000);calls.push_back("fused");}
void rpu_launch_eltwise_unary_spm_kernel(uint32_t a,uint32_t b,int64_t n,ValuOpType op,bool gelu,int cores){assert(a==0x10000&&b==a&&n==8192&&op==ValuOpType::SIGMOID&&!gelu&&cores==8);calls.push_back("sigmoid");}
void rpu_launch_eltwise_binary_spm_kernel(uint32_t a,uint32_t b,uint32_t out,int64_t n,ValuOpType op,c10::Half scale,int cores){assert(a==0x10000&&b==0x18000&&out==b&&n==8192&&op==ValuOpType::MUL&&scale.value==1&&cores==8);calls.push_back("mul");}
struct Model{
 bool action_attn_sigmoid_mul_available_=false,has_gdn_=false,action_mode_=true,linear_acc32_=false,z2_bound_=false,action_step_active_=true,action_loop_active_=true,debug=false,complete=true;
 int64_t action_len_=32,action_num_steps_=10,layers=24,hidden=1024,intermediate=4096,q=8,kv=8,d=256,tp=8,mtp=8,cores=8;
 struct{bool on=true;bool enabled()const{return on;}}action_nvfp4_;
 std::vector<uint8_t>layer_is_full_=std::vector<uint8_t>(24,1);struct{std::vector<Route>routes;}manifest;
 int64_t num_layers()const{return layers;}int64_t hidden_size()const{return hidden;}int64_t intermediate_size()const{return intermediate;}int64_t num_q_heads()const{return q;}int64_t num_kv_heads()const{return kv;}int64_t head_dim()const{return d;}int64_t attn_tp()const{return tp;}int64_t mlp_tp()const{return mtp;}int64_t num_cores()const{return cores;}bool get_debug_export()const{return debug;}
 Model&ctx(){return *this;}bool has_complete_physical_manifest()const{return complete;}
 uint32_t addr(int,const char*s)const{return std::string(s)=="attn_gate"?0x10000:0x18000;}
 void consume_physical_route(FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,const std::vector<int64_t>&args,int64_t invocation){auto it=std::find_if(manifest.routes.begin(),manifest.routes.end(),[&](auto&r){return r.site==site&&r.invocation==invocation;});TORCH_CHECK(complete&&it!=manifest.routes.end()&&it->family==f&&it->selector==selector&&it->flags==flags&&it->args==args);}
 __METHODS__
 void install(){prepare_action_attn_sigmoid_mul();}
 void reset(){__RESET__}
 void plan(){manifest.routes.clear();ChunkInfo chunk;__MANIFEST__}
 void attention(int layer_idx,const ChunkInfo&chunk={}){const int64_t seq_len=chunk.len,nq=8,tp=8,hd=256,auxiliary_cores=8;__CALLER__}
 int64_t layout_hash(){int64_t hash=0;__LAYOUT__ return hash;}
 std::vector<int64_t>cost(){std::vector<int64_t>identity;__IDENTITY__ return identity;}
};
template<class F>void reject(F f){try{f();assert(false);}catch(const std::runtime_error&){};assert(calls.empty());}
int main(){
 Model m;m.install();assert(m.action_attn_sigmoid_mul_available_);m.plan();assert(m.manifest.routes.size()==1);const auto hash=m.layout_hash();const auto cost=m.cost();const int queries=KernelCache::instance().queries;
 for(int l=0;l<24;++l){calls.clear();m.attention(l);assert(calls==std::vector<std::string>{"fused"});}assert(KernelCache::instance().queries==queries);
 for(size_t i=0;i<m.manifest.routes[0].args.size();++i){auto x=m;++x.manifest.routes[0].args[i];calls.clear();reject([&]{x.attention(0);});}
 for(int f=0;f<7;++f){auto x=m;if(f==0)++x.manifest.routes[0].selector;if(f==1)++x.manifest.routes[0].flags;if(f==2)++x.manifest.routes[0].site;if(f==3)++x.manifest.routes[0].invocation;if(f==4)x.manifest.routes[0].family=FmbRouteFamily::LINEAR;if(f==5)x.manifest.routes.clear();if(f==6)x.complete=false;calls.clear();reject([&]{x.attention(0);});}
 for(auto chunk:std::vector<ChunkInfo>{{32,1,0},{32,0,32},{31,0,0}}){calls.clear();reject([&]{m.launch_action_attn_sigmoid_mul(0,chunk);});}
 for(int l:{-1,24}){calls.clear();reject([&]{m.launch_action_attn_sigmoid_mul(l,{});});}
 // The Sigmoid and FP16-retired multiply are independent of upstream Linear accumulation.
 {auto x=m;x.linear_acc32_=true;x.install();assert(x.action_attn_sigmoid_mul_available_);x.plan();calls.clear();x.attention(0);assert(calls==std::vector<std::string>({"fused"}));}
 for(int f=0;f<16;++f){if(f==2)continue;auto x=m;if(f==0)x.action_nvfp4_.on=false;if(f==1)x.action_mode_=false;if(f==2)x.linear_acc32_=true;if(f==3)x.z2_bound_=true;if(f==4)x.action_len_=31;if(f==5)x.cores=4;if(f==6)x.has_gdn_=true;if(f==7)x.layers=23;if(f==8)x.hidden=2048;if(f==9)x.intermediate=6144;if(f==10)x.q=16;if(f==11)x.kv=4;if(f==12)x.d=128;if(f==13)x.tp=4;if(f==14)x.mtp=4;if(f==15)x.layer_is_full_.pop_back();x.install();assert(!x.action_attn_sigmoid_mul_available_);x.plan();calls.clear();x.attention(0);assert(calls==std::vector<std::string>({"sigmoid","mul"}));}
 for(int l=0;l<24;++l){auto x=m;x.layer_is_full_[l]=0;x.install();assert(!x.action_attn_sigmoid_mul_available_);}
 for(int f=0;f<4;++f){auto x=m;if(f==0)x.action_step_active_=false;if(f==1)x.action_loop_active_=false;if(f==2)x.action_num_steps_=9;if(f==3)x.debug=true;assert(!x.use_action_attn_sigmoid_mul(32));x.plan();calls.clear();x.attention(0);assert(calls==std::vector<std::string>({"sigmoid","mul"}));}
 KernelCache::instance().available=false;m.install();assert(!m.action_attn_sigmoid_mul_available_&&m.layout_hash()!=hash&&m.cost()!=cost);m.plan();assert(m.manifest.routes.empty());calls.clear();m.attention(0);assert(calls==std::vector<std::string>({"sigmoid","mul"}));
 KernelCache::instance().available=true;m.install();assert(m.layout_hash()==hash&&m.cost()==cost);m.reset();assert(!m.action_attn_sigmoid_mul_available_);
}
'''
    for key,value in dict(METHODS=definitions,MANIFEST=manifest,CALLER=caller,LAYOUT=layout,IDENTITY=identity,RESET=reset).items():
        cpp=cpp.replace('__'+key+'__',value)
    _compile_run(tmp_path,'attn_gate_model',cpp)
