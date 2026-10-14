"""Exact Text rounded SiLU/MUL launcher and real paired-MLP cold/COMPLETE path."""
from pathlib import Path
from test_g05_q_gate_native import _compile_run
from test_g05_vision_owner_model import _block
ROOT=Path(__file__).resolve().parents[1]
CALL='rpu_launch_g05_text_silu_round_mul_spm_kernel'
SITE=7739768473980724746


def test_text_silu_round_mul_exact_spans_and_registers(tmp_path):
    source=(ROOT/'src/ops/rpu_eltwise_unary.cpp').read_text()
    cpp=r'''
#include <array>
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
constexpr int kKernelRegOk=0;enum class KernelId{G05_SILU_ROUND_MUL_M32N512};
struct KernelCache{bool loaded=true;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId){return loaded;}};
int gets=0,launches=0,reject_reg=-1;bool missing=false;
struct Kernel{std::map<uint32_t,uint16_t>regs;void reset_regs(){regs.clear();}int set_regs(uint32_t r,uint16_t v){if(int(r)==reject_reg)return 1;assert(!regs.count(r));regs[r]=v;return 0;}}kernel;
Kernel*get(KernelId){++gets;return missing?nullptr:&kernel;}
#define GET_KERNEL(id) get(id)
struct Queue{bool broadcast=false;void set_broadcast_mode(bool b){broadcast=b;}void enqueu_kernel(Kernel&k,std::vector<uint16_t>grid,std::vector<uint8_t>cores){assert(broadcast);assert(grid==std::vector<uint16_t>({8,1,1}));assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));assert(&k==&kernel);++launches;}}queue;
Queue*get_queue(int n){assert(n==8);return &queue;}
#define GET_QUEUE(n) get_queue(n)
void __CALL__(uint32_t gate,uint32_t up,uint32_t product)__BODY__
template<class F>void reject(F f){const int before=launches;try{f();assert(false);}catch(const std::runtime_error&){}assert(launches==before);}
int main(){
 const uint32_t gate=0x20010000,up=0x20070000,product=0x20100000;
 for(auto state:{RpuKernelGraph::State::RECORDING,RpuKernelGraph::State::REPLAYING}){
  RpuKernelGraph::active().status=state;__CALL__(gate,up,product);
  const std::map<uint32_t,uint16_t>expected{{0,25600},{1,17408},{4,0},{5,0x2001},{6,0},{7,0x2010},{8,51200},{9,0},{12,0},{13,0x2007},{56,0x3c00},{57,0x3c00},{58,0x3c00},{59,0},{60,0},{61,1},{62,0x3e92},{63,134}};
  assert(kernel.regs==expected);
 }
 const int before=gets;
 for(auto bad:std::vector<uint32_t>{0,0x1fffff00,gate+1,0x207a0100,0x20800000,0xffffff00}){
  reject([&]{__CALL__(bad,up,product);});reject([&]{__CALL__(gate,bad,product);});reject([&]{__CALL__(gate,up,bad);});}
 for(uint32_t bad:{gate,gate+256,gate-256,gate+393216-256})reject([&]{__CALL__(gate,bad,product);});
 // Product's complete1MiB root, not only its written384KiB prefix, must fit
 // and must not overlap Gate or Up in the otherwise-unwritten tail.
 reject([&]{__CALL__(gate,up,0x207a0000);});
 reject([&]{__CALL__(product+0x90000,up,product);});
 reject([&]{__CALL__(gate,product+0x90000,product);});
 reject([&]{__CALL__(gate,up,gate);});reject([&]{__CALL__(gate,up,up);});
 RpuKernelGraph::active().status=RpuKernelGraph::State::READY;reject([&]{__CALL__(gate,up,product);});RpuKernelGraph::active().status=RpuKernelGraph::State::RECORDING;
 spm.live=false;reject([&]{__CALL__(gate,up,product);});spm.live=true;
 KernelCache::instance().loaded=false;reject([&]{__CALL__(gate,up,product);});KernelCache::instance().loaded=true;assert(gets==before);
 missing=true;reject([&]{__CALL__(gate,up,product);});missing=false;
 for(int r:{0,1,4,5,6,7,8,9,12,13,56,57,58,59,60,61,62,63}){reject_reg=r;reject([&]{__CALL__(gate,up,product);});}reject_reg=-1;
 __CALL__(gate,up,0x20700000); // full1MiB ends exactly at usable boundary.
 __CALL__(0x207a0000,up,product); //384KiB Gate ends at same boundary.
 assert(launches==4);
}
'''.replace('__CALL__',CALL).replace('__BODY__',_block(source,'void '+CALL+'('))
    _compile_run(tmp_path,'text_round_wrapper',cpp)


def test_text_silu_round_mul_cold_complete_and_caller_fallback(tmp_path,monkeypatch):
    # Reuse the existing actual Text MLP/cold-setter fixture. Replace only its
    # explicitly disabled Text-activation facade with these real method bodies.
    import test_g05_text_gate_up_native as base
    captured=[]
    monkeypatch.setattr(base,'_compile_run',lambda path,name,cpp:captured.append(cpp))
    base.test_text_gate_up_actual_cold_install_order_manifest_and_mlp_fallback(tmp_path,True)
    cpp,=captured
    source=(ROOT/'src/fused/rpu_qwen3_5_model.cpp').read_text()
    cpp=cpp.replace('#include <algorithm>','#include <algorithm>\n#include <array>')
    cpp=cpp.replace('int64_t offset=0;int64_t storage_offset()const{return offset;}', '')
    cpp=cpp.replace('int64_t rows=6144,cols=2048,owner=0;', 'int64_t rows=6144,cols=2048,owner=0,offset=0;int device_index=0;')
    cpp=cpp.replace('struct Device{c10::DeviceType value;c10::DeviceType type()const{return value;}};Device device()const{return {location};}', 'struct Device{c10::DeviceType value;int index;c10::DeviceType type()const{return value;}bool operator!=(const Device&b)const{return value!=b.value||index!=b.index;}bool operator==(const Device&b)const{return !(*this!=b);}};Device device()const{return {location,device_index};}int64_t storage_offset()const{return offset;}')
    cpp=cpp.replace('G05_TEXT_GATE_UP_FP16_ACC32_M256N768K2048,','G05_SILU_ROUND_MUL_M32N512,G05_TEXT_GATE_UP_FP16_ACC32_M256N768K2048,',1)
    cpp=cpp.replace('bool available=true,w8_available=true,acc16_available=true,fp16_acc16_available=true;', 'bool available=true,w8_available=true,acc16_available=true,fp16_acc16_available=true,rounded_available=true;')
    cpp=cpp.replace('return available&&', 'return available&&(rounded_available||id!=KernelId::G05_SILU_ROUND_MUL_M32N512)&&',1)
    cpp=cpp.replace('struct ChunkInfo{int64_t len=256,idx=2,offset=512;};','struct ChunkInfo{int64_t len=256,idx=0,offset=0;};\nstruct FmbThreeStageChunkPlan{struct Stage{std::vector<ChunkInfo>chunks{ChunkInfo{}};};Stage qkv,compute;};')
    defs=[
      ('void prepare_text_silu_round_mul()', 'void Qwen3_5Model::prepare_text_silu_round_mul('),
      ('bool use_text_silu_round_mul(const FmbThreeStageChunkPlan& plan,int64_t physical_len,int64_t logical_len,int64_t position,int64_t batch) const','bool Qwen3_5Model::use_text_silu_round_mul(\n'),
      ('bool use_text_silu_round_mul(int64_t rows)','bool Qwen3_5Model::use_text_silu_round_mul(int64_t rows)'),
      ('std::vector<int64_t> text_silu_round_mul_arguments(int64_t logical_len=245) const','std::vector<int64_t> Qwen3_5Model::text_silu_round_mul_arguments('),
      ('void launch_text_silu_round_mul(int layer,const ChunkInfo& chunk)','void Qwen3_5Model::launch_text_silu_round_mul('),
    ]
    begin=cpp.index(' void prepare_text_silu_round_mul(){}')
    end=cpp.index('\n',cpp.index(' template<class...T>void launch_text_silu_round_mul',begin))
    cpp=cpp[:begin]+'\n'.join(decl+_block(source,marker)for decl,marker in defs)+cpp[end:]
    cpp=cpp.replace(' Model&ctx(){return *this;}',' FmbThreeStageChunkPlan stage_plan;int64_t seq_len=256,logical_length=245,position=0,batch_size=1;\n Model&physical_manifest(){return *this;}\n Model&ctx(){return *this;}',1)
    cpp=cpp.replace('if(family==FmbRouteFamily::ACTIVATION){assert(site==QWEN35_TEXT_MLP_SILU_SITE);return;}','',1)
    base_begin=source.index('        if (use_silu_ && (!use_action_silu_round_mul(chunk.len)')
    base_open=source.index('{',base_begin)
    base_rule=source[base_begin:base_open]+_block(source,source[base_begin:base_open])
    begin=cpp.index(' void plan(ChunkInfo chunk={})')
    end=cpp.index('\n void mlp(',begin)
    plan='if (use_text_silu_round_mul(plan,physical_len,logical_len,position,layout.batch_size))'+_block(source,'if (use_text_silu_round_mul(plan,physical_len,logical_len,position,layout.batch_size))')
    cpp=cpp[:begin]+''' void plan(ChunkInfo chunk={}){
 const auto&plan=stage_plan;const auto physical_len=seq_len,logical_len=logical_length;struct{int64_t batch_size;}layout{batch_size};struct{std::vector<Route>routes;}manifest;
 if(use_text_gateup_pair(chunk.len))'''+_block(source,'if (use_text_gateup_pair(chunk.len))')+'\n'+plan+'\n'+base_rule+'\n routes=manifest.routes;}'+cpp[end:]
    cpp=cpp.replace('struct Model{','void '+CALL+'(uint32_t gate,uint32_t up,uint32_t product){assert(gate==0x110000&&up==0x170000&&product==0x10000);assert(!launches.empty()&&launches.back()=="pair_w8");launches.push_back("text-round");}\nstruct Model{',1)
    cpp=cpp.replace('struct Model{','struct Model{\n int64_t gdn_nvh_=16,gdn_dk_=128,gdn_dv_=128,gdn_conv_dim_=6144,gdn_kc_=4;\n bool has_text_fp16_projection_owners()const'+_block(source,'bool Qwen3_5Model::has_text_fp16_projection_owners('),1)
    cpp=cpp[:cpp.index('int main(){')]+r'''
int main(){
 Model m;m.fast_replay_enabled_=true;m.retained_prefill_graph_=true;
 for(auto&w:m.layer_weights_){w.down_w.dtype=at::kChar;w.down_w.rows=2048;w.down_w.cols=6144;w.down_ws.rank=1;w.down_ws.rows=2048;w.down_ws.owner=w.down_w.owner+1000;}
 m.installed();assert(m.text_gateup_pair_available_&&m.text_silu_round_mul_available_&&m.use_text_silu_round_mul(256));m.plan();
 const auto hash=m.subclass_layout_hash();const auto cost=m.kvinsert_cost_weight_identity();
 assert(m.routes.size()==2&&m.routes[1].site==7739768473980724746LL&&m.routes[1].family==FmbRouteFamily::ACTIVATION);
 const auto args=m.text_silu_round_mul_arguments();assert(args==std::vector<int64_t>({2,256,2048,6144,768,8,16,196608,25600,17408,8,51200,134,1048576,393216,1,1,24,245,1,1,0,1,193,256,16,1,245}));
 assert(std::all_of(args.begin(),args.end(),[](int64_t x){return x>=0;}));
 const auto proofs=at::Tensor::proofs,queries=KernelCache::instance().queries;
 for(int l=0;l<24;++l){launches.clear();m.mlp(l);assert(launches==std::vector<std::string>({"pair_w8","text-round","down"}));}
 assert(at::Tensor::proofs==proofs&&KernelCache::instance().queries==queries);
 for(size_t a=0;a<args.size();++a){auto x=m;++x.routes[1].args[a];launches.clear();reject([&]{x.launch_text_silu_round_mul(0,{});});try{x.mlp(0);assert(false);}catch(const std::runtime_error&){}assert(launches==std::vector<std::string>({"pair_w8"}));}
 for(int f=0;f<6;++f){auto x=m;auto&r=x.routes[1];if(f==0){++r.site;}if(f==1){++r.selector;}if(f==2){++r.flags;}if(f==3){++r.invocation;}if(f==4){r.family=FmbRouteFamily::LINEAR;}if(f==5){r.args.pop_back();}launches.clear();reject([&]{x.launch_text_silu_round_mul(0,{});});}
 for(int layer:{-1,24}){launches.clear();reject([&]{m.launch_text_silu_round_mul(layer,{});});}
 for(auto chunk:std::vector<ChunkInfo>{{255,0,0},{256,1,0},{256,0,256}}){launches.clear();reject([&]{m.launch_text_silu_round_mul(0,chunk);});}
 {auto x=m;x.complete=false;launches.clear();reject([&]{x.launch_text_silu_round_mul(0,{});});}
 {auto x=m;x.routes.pop_back();launches.clear();reject([&]{x.launch_text_silu_round_mul(0,{});});}
 auto fallback=[&](auto&x){x.plan();assert(!x.use_text_silu_round_mul(256));assert(std::any_of(x.routes.begin(),x.routes.end(),[](auto&r){return r.site==QWEN35_TEXT_MLP_SILU_SITE;}));assert(std::none_of(x.routes.begin(),x.routes.end(),[](auto&r){return r.site==7739768473980724746LL;}));launches.clear();x.mlp(0);auto expected=x.use_text_gateup_pair(256)?std::vector<std::string>{"pair_w8","silu","mul","down"}:std::vector<std::string>{"gate","silu","up","mul","down"};assert(launches==expected);};
 // Each affected installed W8/scale family on every layer can withdraw only
 // this fusion, including otherwise-valid offset views and device-index drift.
 for(int layer=0;layer<24;++layer)for(int family=0;family<3;++family)for(int fault=0;fault<16;++fault){
  auto x=m;auto&w=x.layer_weights_[layer];at::Tensor*weights[]={&w.gate_w,&w.up_w,&w.down_w};at::Tensor*scales[]={&w.gate_ws,&w.up_ws,&w.down_ws};auto&v=*weights[family];auto&s=*scales[family];
  if(fault==0){v.present=false;}if(fault==1){v.dtype=at::kHalf;}if(fault==2){v.location=c10::DeviceType::CPU;}if(fault==3){v.contiguous=false;}if(fault==4){v.offset=256;}if(fault==5){v.rank=1;}if(fault==6){--v.rows;}if(fault==7){--v.cols;}
  if(fault==8){s.present=false;}if(fault==9){s.dtype=at::kChar;}if(fault==10){s.location=c10::DeviceType::CPU;}if(fault==11){s.contiguous=false;}if(fault==12){s.offset=128;}if(fault==13){s.rank=2;}if(fault==14){--s.rows;}if(fault==15){++s.device_index;}
  x.prepare_text_silu_round_mul();assert(!x.text_silu_round_mul_available_);if(fault==4||fault==12){fallback(x);}
 }
 for(int f=0;f<19;++f){auto x=m;
  if(f==0){x.action_mode_=true;}if(f==1)continue;if(f==2){x.fast_replay_enabled_=false;}if(f==3){x.use_silu_=false;}if(f==4){x.has_gdn_=false;}if(f==5){x.layers=23;}if(f==6){x.h=1024;}if(f==7){x.intermediate=4096;}if(f==8){x.cores=4;}if(f==9){x.tp=4;}if(f==10){x.heads=16;}if(f==11){x.kv=4;}if(f==12){x.hd=128;}if(f==13){x.layer_weights_.pop_back();}if(f==14){x.layer_is_full_.pop_back();}if(f==15){x.text_gateup_pair_available_=false;}if(f==16){x.text_gateup_pair_w8_=false;}if(f==17){x.layer_is_full_[0]=1;}if(f==18){++x.layer_weights_.back().down_w.device_index;}
  x.prepare_text_silu_round_mul();assert(!x.text_silu_round_mul_available_);
 }
 for(int f=0;f<13;++f){auto x=m;
  if(f==0){x.debug=true;}if(f==1){x.z2_bound_=true;}if(f==2){x.retained_prefill_graph_=false;}if(f==3){x.logical_length=244;}if(f==4){x.position=1;}if(f==5){x.batch_size=2;}if(f==6){x.stage_plan.qkv.chunks.push_back({});}if(f==7){x.stage_plan.compute.chunks.clear();}if(f==8){x.stage_plan.qkv.chunks[0].idx=1;}if(f==9){x.stage_plan.qkv.chunks[0].offset=1;}if(f==10){x.stage_plan.qkv.chunks[0].len=128;}if(f==11){x.stage_plan.compute.chunks[0].idx=1;}if(f==12){x.stage_plan.compute.chunks[0].offset=1;}fallback(x);
 }
 {auto x=m;x.seq_len=128;assert(!x.use_text_silu_round_mul(128));}
 {auto x=m;x.stage_plan.compute.chunks[0].len=128;fallback(x);}
 {auto x=m;x.debug=true;assert(x.subclass_layout_hash()!=hash&&x.kvinsert_cost_weight_identity()!=cost);}
 for(int family=0;family<6;++family){auto x=m;auto&w=x.layer_weights_.back();at::Tensor*owners[]={&w.gate_w,&w.up_w,&w.down_w,&w.gate_ws,&w.up_ws,&w.down_ws};++owners[family]->owner;assert(x.subclass_layout_hash()!=hash&&x.kvinsert_cost_weight_identity()!=cost);}
 auto&cache=KernelCache::instance();cache.rounded_available=false;m.installed();assert(m.text_gateup_pair_available_&&!m.text_silu_round_mul_available_);assert(m.subclass_layout_hash()!=hash&&m.kvinsert_cost_weight_identity()!=cost);fallback(m);
 cache.rounded_available=true;m.installed();assert(m.text_silu_round_mul_available_&&m.subclass_layout_hash()==hash&&m.kvinsert_cost_weight_identity()==cost);
 {auto x=m;x.set_linear_acc32(true);assert(x.text_silu_round_mul_available_&&x.text_silu_round_mul_arguments()[6]==32&&x.text_gateup_pair_arguments()[5]==32&&x.subclass_layout_hash()!=hash&&x.kvinsert_cost_weight_identity()!=cost);launches.clear();reject([&]{x.launch_text_silu_round_mul(0,{});});assert(launches.empty());x.plan();for(int layer=0;layer<24;++layer){linear_modes.clear();launches.clear();x.mlp(layer);assert(launches==std::vector<std::string>({"pair_w8","text-round","down"})&&linear_modes==std::vector<bool>({true}));}cache.rounded_available=false;x.installed();fallback(x);cache.rounded_available=true;}
 m.set_linear_acc32(true);assert(m.text_silu_round_mul_available_);m.set_linear_acc32(false);assert(m.text_silu_round_mul_available_);
 m.set_fast_replay(false);assert(!m.text_silu_round_mul_available_);m.set_fast_replay(true);assert(m.text_silu_round_mul_available_);
 cache.acc16_available=false;m.installed();assert(!m.text_silu_round_mul_available_&&!m.text_gateup_pair_available_);fallback(m);cache.acc16_available=true;
}
'''
    _compile_run(tmp_path,'text_round_model',cpp)
