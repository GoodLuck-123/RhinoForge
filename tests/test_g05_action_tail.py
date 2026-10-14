"""Execute real G05 Action-tail launcher, cold/COMPLETE/caller and fallback bodies."""
from pathlib import Path
from test_g05_q_gate_native import _compile_run
from test_g05_vision_owner_model import _block
ROOT=Path(__file__).resolve().parents[1]


def test_action_tail_wrapper_full_spans_and_replay_counters(tmp_path,monkeypatch):
    import test_g05_ring_rtc_residual as old
    captured=[];monkeypatch.setattr(old,'_compile_run',lambda p,n,cpp:captured.append(cpp))
    old.test_actual_ring_wrapper_guards_shared_scm_and_no_ddr_census(tmp_path)
    cpp=captured[0].split('int main(){',1)[0]
    s=(ROOT/'src/core/rpu_runtime_extras.cpp').read_text();marker='void rpu_launch_g05_action_tail_spm_kernel(';start=s.index(marker)
    function=s[start:s.index('{',start)]+_block(s,marker)
    cpp=cpp.replace('#include <algorithm>','#include <algorithm>\n#include <cmath>')
    cpp=cpp.replace('struct RpuKernelGraph{bool census=true;','struct RpuKernelGraph{enum class State{RECORDING,REPLAYING,PASSTHROUGH};State status=State::RECORDING;State state()const{return status;}bool census=true;')
    cpp=cpp.replace('assert(id==KernelId::G05_RING_RTC_SUB_GATED_RESIDUAL_M32H1024&&','assert(id==KernelId::G05_RING_RTC_THEN_NORM_SHIFT_M32H1024&&')
    cpp=cpp.replace('assert(stages>gets);','assert(RpuKernelGraph::active().census?stages>gets:stages==0);')
    cpp+=function+r'''
int main(){
 const uint32_t base=allocator.addr(0,0),x=base+0x10000,r=base+0x30000,raw=base+0x50000;
 std::array<uint32_t,6>p{base+0x70000,base+0x71800,base+0x72000,base+0x72800,base+0x73800,base+0x74000};
 auto call=[&](int clean=5,double eps=1e-6){rpu_launch_g05_action_tail_spm_kernel(x,r,raw,p[0],p[1],p[2],p[3],p[4],p[5],clean,eps);};
 for(bool census:{true,false}){RpuKernelGraph::active().census=census;stages=gets=0;
  for(auto state:{RpuKernelGraph::State::RECORDING,RpuKernelGraph::State::REPLAYING})for(int clean:{0,5,32}){
   RpuKernelGraph::active().status=state;kernel.regs[128]=8;call(clean);const auto&s=queue.submissions.back();
   assert(s.id==KernelId::G05_RING_RTC_THEN_NORM_SHIFT_M32H1024&&s.grid==std::vector<uint16_t>({8,1,1}));
   assert(pair(s,4)==r-base&&pair(s,18)==p[0]&&pair(s,20)==r&&pair(s,22)==p[1]&&s.regs.at(24)==clean);
   assert(s.regs.at(32)==1024&&s.regs.at(33)==0x1400&&pair(s,34)==2048&&pair(s,36)==raw&&pair(s,38)==r&&pair(s,40)==p[2]&&s.regs.at(42)==0x11&&pair(s,44)==p[3]&&pair(s,46)==p[4]&&pair(s,48)==p[5]&&s.regs.at(50)==clean);
   for(int i=128;i<256;++i)assert(s.regs.at(i)==0);
   for(int c=0;c<8;++c)for(int i=0;i<4;++i){uint32_t expected[]={allocator.addr(c,x-base)-base,allocator.addr(c,raw-base)-base,uint32_t(c*8192),4096};assert((uint32_t(s.regs.at(4096+i*16+c))|(uint32_t(s.regs.at(4096+i*16+c+8))<<16))==expected[i]);}
  }
 }
 for(int clean:{-1,33})reject([&]{call(clean);});
 for(double eps:{0.,1e-5,1.,double(INFINITY),double(NAN)})reject([&]{call(5,eps);});
 for(int which=0;which<9;++which)for(auto bad:{0u,x+1,base+SpmAllocator::SPM_USABLE-256}){
  std::array<uint32_t,9>a{x,r,raw,p[0],p[1],p[2],p[3],p[4],p[5]};a[which]=bad;
  reject([&]{rpu_launch_g05_action_tail_spm_kernel(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],5,1e-6);});
 }
 for(uint32_t raw_bad:{x,r,r+256})reject([&]{rpu_launch_g05_action_tail_spm_kernel(x,r,raw_bad,p[0],p[1],p[2],p[3],p[4],p[5],5,1e-6);});
 for(int k=0;k<6;++k){auto keep=p[k];for(auto bad:{x,r,raw,p[(k+1)%6]+256}){p[k]=bad;reject([&]{call();});}p[k]=keep;}
 KernelCache::instance().loaded=false;reject([&]{call();});KernelCache::instance().loaded=true;
 allocator.initialized=false;reject([&]{call();});allocator.initialized=true;
 RpuKernelGraph::active().status=RpuKernelGraph::State::PASSTHROUGH;reject([&]{call();});
}
'''
    _compile_run(tmp_path,'action_tail_native',cpp)


def test_action_tail_cold_complete_local_flag_and_fallback(tmp_path):
    source=(ROOT/'src/fused/rpu_qwen3_5_model.cpp').read_text();header=(ROOT/'src/fused/rpu_qwen3_5_model.h').read_text()
    methods=[('void prepare_action_tail()','void Qwen3_5Model::prepare_action_tail('),('bool use_action_tail(int64_t rows)const','bool Qwen3_5Model::use_action_tail('),('std::vector<int64_t>action_tail_arguments()const','std::vector<int64_t> Qwen3_5Model::action_tail_arguments('),('void launch_action_tail(int layer,const ChunkInfo&chunk)','void Qwen3_5Model::launch_action_tail('),('bool use_action_nvfp4(int64_t rows)const','bool Qwen3_5Model::use_action_nvfp4('),('bool use_action_mod_body()const','bool Qwen3_5Model::use_action_mod_body('),('bool use_action_ring_rtc_residual(int64_t rows)const','bool Qwen3_5Model::use_action_ring_rtc_residual('),('int64_t action_norm_route(int64_t rows)const','int64_t Qwen3_5Model::action_norm_route('),('std::vector<int64_t>action_norm_arguments(int64_t rows)const','std::vector<int64_t> Qwen3_5Model::action_norm_arguments(')]
    definitions='\n'.join(decl+_block(source,marker)for decl,marker in methods)
    attention=_block(source,'void Qwen3_5Model::build_full_attention(');attention=attention[attention.index('    // Phase 5:'):attention.rindex('}')]
    tail=_block(source,'void Qwen3_5Model::emit_mlp_and_output(');tail=tail[tail.index('    // post-attention RMSNorm:'):tail.index('    // MLP (SwiGLU)')]
    layer=_block(source,'void Qwen3_5Model::build_layer_subgraph(');layer=layer[layer.index('    bool text_owner_norm_emitted = false;'):layer.rindex('}')]
    manifest='if (use_action_tail(chunk.len))'+_block(source,'if (use_action_tail(chunk.len))')
    layout='if (action_mode_ && action_tail_available_)'+_block(source,'if (action_mode_ && action_tail_available_)')
    start=header.index('            identity.push_back(static_cast<int64_t>(action_tail_available_));');identity=header[start:header.index('            identity.push_back(static_cast<int64_t>(action_attn_sigmoid_mul_available_));',start)]
    assert 'prepare_action_tail();'in _block(source,'void Qwen3_5Model::set_action_io_weights(')
    assert 'action_tail_available_ = false;'in _block(source,'void Qwen3_5Model::set_weights(')
    cpp=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define DWIDTH 2
#define TORCH_CHECK(x,...) do{if(!(x))throw std::runtime_error("rejected");}while(0)
namespace c10{struct Half{uint16_t x;explicit Half(double v):x(v==1e-6?0x11:0){}};}
namespace detail{int64_t layout_mix(int64_t a,int64_t b){return uint64_t(a)*131+uint64_t(b);}}
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8387584;uint32_t addr(int,int)const{return 0x40000000u;}}allocator;
#define SPM_ALLOC allocator
enum class KernelId{G05_RING_RTC_THEN_NORM_SHIFT_M32H1024};struct KernelCache{bool available=true;int queries=0;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId){++queries;return available;}};
enum class FmbRouteFamily{ALL_REDUCE,NORMALIZATION,GRAPH_SCHEDULE,OTHER};enum class RpuRmsNormSpmRoute{BASE};
constexpr int64_t QWEN35_ACTION_TAIL_SITE=7035949248365271364LL,QWEN35_TEXT_FULL_ALL_REDUCE_SITE=1,QWEN35_ACTION_FUSED_NORM_SHIFT_SITE=2;
struct ChunkInfo{int64_t len=32,idx=0,offset=0;};struct Route{int64_t site;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t>args;int64_t invocation;};
std::vector<std::string>calls;bool fail_producer=false;int native_clean=-1;uint32_t native_gate=0,native_scale=0;
void rpu_launch_g05_action_tail_spm_kernel(uint32_t x,uint32_t skip,uint32_t raw,uint32_t cg,uint32_t ng,uint32_t cs,uint32_t ch,uint32_t ns,uint32_t nh,int64_t clean,double eps){assert(x==0x40010000&&skip==0x40030000&&raw==0x40050000&&ng==cg+6144&&ch==cs+2048&&ns==cs+6144&&nh==cs+8192&&cs==cg+8192&&eps==1e-6);if(fail_producer)throw std::runtime_error("producer");calls.push_back("fused");native_clean=clean;native_gate=cg;native_scale=cs;}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t,uint32_t,uint32_t,int64_t,int64_t,int,int){calls.push_back("old-reduce");}
void rpu_launch_rmsnorm_spm_kernel(uint32_t,uint32_t,uint32_t,int64_t,int64_t,double,RpuRmsNormSpmRoute,int){assert(false);}
struct Model{
 static constexpr int64_t QWEN35_ACTION_MLP_TAIL_SITE=4368884220398897667LL;
 bool action_mlp_tail_available_=false;
 void prepare_action_mlp_tail(){}
 template<class...T>bool use_action_mlp_tail(T...)const{return false;}
 template<class...T>bool action_mlp_tail_handoff(T...){return false;}
 std::vector<int64_t>action_mlp_tail_arguments()const{return {};}
 template<class...T>void launch_action_mlp_tail(T...){assert(false);}
 static constexpr int64_t QWEN35_TEXT_MLP_TAIL_SITE=8475656447499929675LL;
 bool text_mlp_tail_available_=false;
 void prepare_text_mlp_tail(){}
 template<class...T>bool use_text_mlp_tail(T...)const{return false;}
 template<class...T>bool text_mlp_tail_handoff(T...){return false;}
 std::vector<int64_t>text_mlp_tail_arguments(int64_t = 245)const{return {};}
 template<class...T>void launch_text_mlp_tail(T...){assert(false);}

 bool action_tail_available_=false,action_mode_=true,linear_acc32_=false,z2_bound_=false,has_gdn_=false,use_silu_=true,action_step_active_=true,action_loop_active_=true,action_rtc_active_=true,debug=false,complete=true;
 bool action_ring_rtc_residual_available_=true,action_rtc_norm_shift_available_=true,action_norm_shift_available_=true,action_mod_body_fp16_=false,action_mod_body_w8_=false;
 int64_t action_len_=32,action_num_steps_=10,layers=24,hidden=1024,intermediate=4096,q=8,kv=8,hd=256,tp=8,mtp=8,cores=8,action_rtc_prefix_len_=5,body_iter=0,seq_len=32,position=245,batch_size=1;
 double eps_=1e-6;uint32_t adaptive_mod_row_offset_=0,mod_base=0x40090000;
 struct{bool on=true;bool enabled()const{return on;}}action_nvfp4_;
 std::vector<uint8_t>layer_is_full_=std::vector<uint8_t>(24,1);struct{std::vector<Route>routes;int64_t logical_length=32;}manifest;int stage_plan=0;
 int64_t num_layers()const{return layers;}int64_t hidden_size()const{return hidden;}int64_t intermediate_size()const{return intermediate;}int64_t num_q_heads()const{return q;}int64_t num_kv_heads()const{return kv;}int64_t head_dim()const{return hd;}int64_t attn_tp()const{return tp;}int64_t mlp_tp()const{return mtp;}int64_t num_cores()const{return cores;}bool get_debug_export()const{return debug;}
 Model&ctx(){return *this;}bool has_complete_physical_manifest()const{return complete;}const auto&physical_manifest()const{return manifest;}
 uint32_t addr(int,const char*n)const{std::string s(n);return s=="oproj"?0x40010000:s=="residual1"?0x40030000:s=="adaptive_mod"?mod_base:0x40050000;}
 uint32_t layer_addr(int,int,const char*)const{return 0x40100000;}uint32_t layer_input_residual_addr(int,const ChunkInfo&)const{return addr(0,"residual1");}
 int64_t text_ring_route(int64_t,int64_t)const{return 2;}
 void consume_physical_route(FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,const std::vector<int64_t>&args,int64_t invocation){auto it=std::find_if(manifest.routes.begin(),manifest.routes.end(),[&](auto&r){return r.site==site&&r.invocation==invocation;});TORCH_CHECK(complete&&it!=manifest.routes.end()&&it->family==f&&it->selector==selector&&it->flags==flags&&it->args==args);}
 void consume_manifest_route(FmbRouteFamily f,int64_t site,int64_t sel,int64_t invocation){consume_physical_route(f,site,sel,0,{},invocation);}
 template<class...T>bool use_text_owner_rms(T...)const{return false;}template<class...T>bool use_text_gdn_owner_rms(T...)const{return false;}template<class...T>void launch_text_owner_rms(T...){assert(false);}
 template<class...T>void build_gdn(T...){assert(false);}
 void load_adaptive_mod_row(int row){adaptive_mod_row_offset_=row*12288;}
 void launch_action_ring_rtc_residual(uint32_t,uint32_t,uint32_t,int64_t,const ChunkInfo&){calls.push_back("ring");}
 template<class...T>void apply_adaptive_residual_gate(T...){calls.push_back("gate");}
 void apply_adaptive_norm(uint32_t input,uint32_t output,const ChunkInfo&chunk){assert(input==addr(0,"input_norm")&&output==addr(0,"residual1"));if(action_norm_route(chunk.len)==3)consume_physical_route(FmbRouteFamily::NORMALIZATION,QWEN35_ACTION_FUSED_NORM_SHIFT_SITE,3,0,action_norm_arguments(chunk.len),chunk.idx);calls.push_back("norm");}
 __METHODS__
 void install(){prepare_action_tail();}void reset(){action_tail_available_=false;}
 void plan(){manifest.routes={{QWEN35_TEXT_FULL_ALL_REDUCE_SITE,FmbRouteFamily::ALL_REDUCE,2,0,{},0},{QWEN35_ACTION_FUSED_NORM_SHIFT_SITE,FmbRouteFamily::NORMALIZATION,3,0,action_norm_arguments(32),0}};ChunkInfo chunk;__MANIFEST__}
 int64_t layout_hash(){int64_t hash=0;__LAYOUT__ return hash;}
 std::vector<int64_t>cost(){std::vector<int64_t>identity;__IDENTITY__ return identity;}
 void build_full_attention(int layer_idx,const ChunkInfo&chunk,bool*text_owner_norm_emitted,bool*action_norm_emitted){int64_t seq_len=chunk.len,h=hidden_size(),tp=attn_tp();__ATTENTION__}
 void emit_mlp_and_output(int layer_idx,const ChunkInfo&chunk,bool text_owner_norm_emitted,bool action_norm_emitted){int64_t seq_len=chunk.len,h=hidden_size();__TAIL__ assert(adaptive_mod_row_offset_==uint32_t((2*layer_idx+1)*12288));}
 void layer(int layer_idx,const ChunkInfo&chunk={}){load_adaptive_mod_row(2*layer_idx);__LAYER__}
};
template<class F>void reject(F f){const auto n=calls.size();try{f();assert(false);}catch(const std::runtime_error&){}assert(calls.size()==n);}
int main(){
 Model m;m.install();m.plan();const auto hash=m.layout_hash();const auto cost=m.cost();assert(m.action_tail_available_&&m.manifest.routes.size()==3);const int queries=KernelCache::instance().queries;
 for(int body=0;body<10;++body)for(int layer=0;layer<24;++layer)for(int clean:{0,5,32}){m.body_iter=body;m.action_rtc_prefix_len_=clean;calls.clear();m.layer(layer);assert(calls==std::vector<std::string>{"fused"}&&native_clean==clean&&native_gate==m.mod_base+2*layer*12288+4096&&native_scale==m.mod_base+(2*layer+1)*12288);}
 assert(KernelCache::instance().queries==queries);
 for(size_t route=0;route<m.manifest.routes.size();++route){
  for(size_t i=0;i<m.manifest.routes[route].args.size();++i){auto x=m;++x.manifest.routes[route].args[i];reject([&]{x.layer(0);});}
  for(int f=0;f<7;++f){auto x=m;auto&r=x.manifest.routes[route];if(f==0)++r.site;if(f==1)++r.selector;if(f==2)++r.flags;if(f==3)++r.invocation;if(f==4)r.family=FmbRouteFamily::OTHER;if(f==5)x.manifest.routes.erase(x.manifest.routes.begin()+route);if(f==6)x.complete=false;reject([&]{x.layer(0);});}
 }
 for(int fault=0;fault<7;++fault){auto x=m;x.load_adaptive_mod_row(0);if(fault==0)x.body_iter=-1;if(fault==1)x.body_iter=10;if(fault==2)++x.adaptive_mod_row_offset_;if(fault==3)x.mod_base=0x407f0000;if(fault==4)x.mod_base=0;if(fault==5)++x.mod_base;if(fault==6)x.layer_is_full_[0]=0;reject([&]{x.launch_action_tail(0,{});});}
 for(auto chunk:std::vector<ChunkInfo>{{31,0,0},{32,1,0},{32,0,1}}){m.load_adaptive_mod_row(0);reject([&]{m.launch_action_tail(0,chunk);});}
 for(int layer:{-1,24})reject([&]{m.launch_action_tail(layer,{});});
 for(int fault=0;fault<15;++fault){if(fault==1)continue;auto x=m;if(fault==0)x.action_nvfp4_.on=false;if(fault==1)x.linear_acc32_=true;if(fault==2)x.has_gdn_=true;if(fault==3)x.eps_=1e-5;if(fault==4)x.layers=23;if(fault==5)x.hidden=2048;if(fault==6)x.intermediate=6144;if(fault==7)x.q=16;if(fault==8)x.kv=4;if(fault==9)x.hd=128;if(fault==10)x.tp=4;if(fault==11)x.mtp=4;if(fault==12)x.use_silu_=false;if(fault==13)x.action_ring_rtc_residual_available_=false;if(fault==14)x.action_rtc_norm_shift_available_=false;x.install();assert(!x.action_tail_available_);}
 {auto x=m;x.linear_acc32_=true;x.install();x.plan();assert(x.action_tail_available_&&x.action_tail_arguments()[4]==32&&x.layout_hash()!=hash&&x.cost()!=cost);calls.clear();x.layer(23);assert(calls==std::vector<std::string>{"fused"});x.manifest.routes.back().args[4]=16;reject([&]{x.layer(0);});}
 for(int layer=0;layer<24;++layer){auto x=m;x.layer_is_full_[layer]=0;x.install();assert(!x.action_tail_available_);}
 for(int fault=0;fault<6;++fault){auto x=m;if(fault==0)x.action_step_active_=false;if(fault==1)x.action_loop_active_=false;if(fault==2)x.action_num_steps_=1;if(fault==3)x.action_rtc_active_=false;if(fault==4)x.debug=true;if(fault==5)x.z2_bound_=true;assert(!x.use_action_tail(32));}
 KernelCache::instance().available=false;m.install();m.plan();assert(!m.action_tail_available_&&m.manifest.routes.size()==2&&m.layout_hash()!=hash&&m.cost()!=cost);calls.clear();m.layer(23);assert(calls==std::vector<std::string>({"ring","norm"}));
 KernelCache::instance().available=true;m.install();m.plan();assert(m.layout_hash()==hash&&m.cost()==cost);
 {auto x=m;x.action_nvfp4_.on=false;x.action_mod_body_w8_=true;x.install();x.plan();calls.clear();x.layer(0);assert(calls==std::vector<std::string>({"ring","norm"}));}
 {auto x=m;x.action_rtc_active_=false;x.plan();calls.clear();x.layer(0);assert(calls==std::vector<std::string>({"ring","norm"}));}
 fail_producer=true;calls.clear();reject([&]{m.layer(0);});fail_producer=false;m.layer(1);assert(calls==std::vector<std::string>{"fused"});
 {auto x=m;x.action_tail_available_=false;reject([&]{x.emit_mlp_and_output(0,{},false,true);});}
 reject([&]{m.emit_mlp_and_output(0,{},true,true);});m.reset();assert(!m.action_tail_available_);
}
'''
    for key,value in dict(METHODS=definitions,ATTENTION=attention,TAIL=tail,LAYER=layer,MANIFEST=manifest,LAYOUT=layout,IDENTITY=identity).items():cpp=cpp.replace('__'+key+'__',value)
    _compile_run(tmp_path,'action_tail_model',cpp)
