"""Compile the actual bounded Wall NVFP4 bias route and launcher on host metadata."""
from pathlib import Path

from test_wall_base_paired_kv import COMMON, block, compile_run

ROOT = Path(__file__).resolve().parents[1]
DECODER = (ROOT / 'src/fused/rpu_qwen3_model.h').read_text()
LINEAR = (ROOT / 'src/ops/rpu_linear.cpp').read_text()
META = COMMON.replace('enum ScalarType {', 'constexpr auto kPrivateUse1=c10::DeviceType::PrivateUse1;\nenum ScalarType {')
META = META.replace('ScalarType dtype=kHalf;', 'int64_t storage_off=0; ScalarType dtype=kHalf;')
META = META.replace('size_t nbytes()const{size_t n=2;for(auto d:shape)n*=d;return n;}', '''
 int64_t numel()const{int64_t n=1;for(auto d:shape)n*=d;return n;}
 size_t nbytes()const{return numel()*(dtype==kByte||dtype==kChar?1:dtype==kFloat?4:2);}
 c10::ArrayRef<int64_t> sizes()const{return shape;}
 int64_t storage_offset()const{return storage_off;}
''')


def test_actual_bias_launcher_ranges_typed_abi_and_replay_counters(tmp_path):
    launch = block(LINEAR, 'void rpu_launch_wall_base_nvfp4_qkv_bias_spm_kernel(')
    optional = block((ROOT / 'src/core/rpu_kernel_cache.inc').read_text(),
                     'static bool is_optional_kernel(')
    source = META + r'''
#include "ops/rpu_linear_tiling.h"
#include <unordered_set>
OPTIONAL
enum class KernelId {WALL_BASE_NVFP4_QKV_BIAS_BARRIER_N64,WALL_BASE_NVFP4_QKV_BIAS_BARRIER_N32};
enum class GraphDdrRegisterRole {PackedFp4ModelWeight,Fp8BlockScale};
enum class GraphDdrRegisterAccess {Read};
enum class GraphDdrRegisterEncoding {DevAddrShift8LoHi};
struct GraphDdrRegisterAbi {GraphDdrRegisterRole role;GraphDdrRegisterAccess access;
 GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Kernel {std::array<uint16_t,132> regs{};std::vector<int> counter_writes;int reject=-1;
 void reset_regs(){regs.fill(0xffff);counter_writes.clear();}
 uint32_t set_regs(uint32_t r,uint16_t v){if(int(r)==reject)return 5;regs.at(r)=v;return 0;}
 uint32_t set_regs(uint32_t r,uint32_t v){if(int(r)==reject)return 5;assert(r%2==0);
  regs.at(r)=uint16_t(v);regs.at(r+1)=uint16_t(v>>16);if(r>=128)counter_writes.push_back(r);return 0;}
 uint32_t set_regs(uint32_t,int32_t)=delete;
} kernel;
struct KernelCache {bool loaded=true;static auto& instance(){static KernelCache c;return c;}
 bool has_loaded(KernelId)const{return loaded;}};
int staged=0,queued=0;KernelId chosen;
struct Writer {std::vector<GraphDdrRegisterOperandSpec> operands;
 void write(Kernel& k){for(const auto& o:operands){auto p=o.tensor.address>>8;
  assert(k.set_regs(o.abi.lo,uint16_t(p))==0);assert(k.set_regs(o.abi.hi,uint16_t(p>>16))==0);}}
};
struct RpuKernelGraph {static auto& active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> v){
  chosen=id;++staged;assert(v.size()==2);
  assert(v[0].abi.role==GraphDdrRegisterRole::PackedFp4ModelWeight&&v[0].abi.lo==2&&v[0].abi.hi==3);
  assert(v[1].abi.role==GraphDdrRegisterRole::Fp8BlockScale&&v[1].abi.lo==20&&v[1].abi.hi==21);
  for(auto& o:v)assert(o.abi.access==GraphDdrRegisterAccess::Read&&o.abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
  return {v};}
};
struct Queue {bool broadcast=false;void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel& k,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
  assert(broadcast&&grid==std::vector<uint16_t>({4,1,1})&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));
  assert(k.counter_writes==std::vector<int>({128,130}));
  assert(k.regs[128]==0&&k.regs[129]==0&&k.regs[130]==0&&k.regs[131]==0);++queued;
  k.regs[128]=4;k.regs[130]=9;}
} queue;
#define GET_KERNEL(id) (&::kernel)
#define GET_QUEUE(n) (&::queue)
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;bool ready=true;uint32_t base=0x20000000;
 bool is_initialized()const{return ready;}uint32_t addr(int core,uint32_t off)const{return base+core*0x1000000u+off;}
} spm;
#define SPM_ALLOC spm
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uintptr_t>(p);}
void rpu_ddr_flush(void*){}
LAUNCH
uint32_t u32(int r){return kernel.regs[r]|(uint32_t(kernel.regs[r+1])<<16);}
int main(){
 assert(is_optional_kernel("wall_base_nvfp4_qkv_bias_barrier_n64"));
 assert(is_optional_kernel("wall_base_nvfp4_qkv_bias_barrier_n32"));
 assert(!is_optional_kernel("llama_insert_kv_cache_multiwarp_v16"));
 constexpr uint32_t x=0x20000000,y=x+65536,b=y+16384,t=b+512;
 for(int n:{2048,1024}){
  at::Tensor w,s;w.dtype=s.dtype=at::kByte;w.shape={n,512};s.shape={64,n};s.address=0x18000000;
  auto run=[&](at::Tensor weight,at::Tensor scale,uint32_t input,uint32_t output,uint32_t bias,uint32_t ts,
               bool expected,int elems=40,int layer=35,int cols=0){
   int before=staged,old=queued;bool accepted=true;
   try{rpu_launch_wall_base_nvfp4_qkv_bias_spm_kernel(input,weight,output,bias,scale,ts,elems,layer,cols?cols:n);}
   catch(const std::exception&){accepted=false;}
   assert(accepted==expected&&staged==before+int(expected)&&queued==old+int(expected));
  };
  for(int repeat=0;repeat<2;++repeat){
   run(w,s,x,y,b,t,true);assert(chosen==(n==2048?KernelId::WALL_BASE_NVFP4_QKV_BIAS_BARRIER_N64:KernelId::WALL_BASE_NVFP4_QKV_BIAS_BARRIER_N32));
   assert(u32(0)==x&&u32(2)==w.address/256&&u32(4)==y&&u32(6)==b);
   assert(kernel.regs[8]==32&&u32(10)==uint32_t(n/8)&&kernel.regs[12]==1024&&kernel.regs[13]==16);
   assert(kernel.regs[14]==1&&kernel.regs[15]==8&&u32(16)==2048&&u32(18)==uint32_t(n/4));
   assert(u32(20)==s.address/256&&u32(22)==uint32_t(n*2)&&u32(24)==t&&kernel.regs[26]==1&&kernel.regs[27]==35);
   assert(kernel.regs[64]==4&&kernel.regs[65]==1&&kernel.regs[66]==1);
  }
  run(w,s,x,y,b,t,false,36);run(w,s,x,y,b,t,false,48);run(w,s,x,y,b,t,false,40,36);
  run(w,s,x,y,b,t,false,40,0,512);run(w,s,x+1,y,b,t,false);run(w,s,x,y,b,0,false);
  run(w,s,x,x+32,b,t,false);run(w,s,x,y,y,t,false);run(w,s,x,y,b,b,false);
  run(w,s,x,y,b,x+SpmAllocator::SPM_USABLE-160,true);
  run(w,s,x,y,b,x+SpmAllocator::SPM_USABLE-128,false);
  spm.ready=false;run(w,s,x,y,b,t,false);spm.ready=true;
  spm.base=0xf8fff000;run(w,s,spm.base,spm.base+65536,spm.base+81920,spm.base+82432,false);spm.base=x;
  for(int which:{0,1}){
   auto bad=which?s:w;
   auto reject=[&](at::Tensor a){run(which?w:a,which?a:s,x,y,b,t,false);};
   bad.dtype=at::kHalf;reject(bad);bad=which?s:w;bad.present=false;reject(bad);
   bad=which?s:w;bad.contiguous=false;reject(bad);bad=which?s:w;bad.storage_off=1;reject(bad);
   bad=which?s:w;bad.shape[1]++;reject(bad);bad=which?s:w;bad.where.kind=c10::DeviceType::CPU;reject(bad);
   bad=which?s:w;bad.address++;reject(bad);bad=which?s:w;bad.address=uint64_t{1}<<40;reject(bad);
  }
  {auto bad=s;bad.where.index=1;run(w,bad,x,y,b,t,false);bad=s;bad.address=w.address+256;run(w,bad,x,y,b,t,false);}
  KernelCache::instance().loaded=false;run(w,s,x,y,b,t,false);KernelCache::instance().loaded=true;
  for(int reg:{64,128,130}){kernel.reject=reg;int before=queued;bool rejected=false;
   try{rpu_launch_wall_base_nvfp4_qkv_bias_spm_kernel(x,w,y,b,s,t,40,0,n);}catch(const std::exception&){rejected=true;}
   assert(rejected&&queued==before);kernel.reject=-1;}
 }
}
'''
    compile_run(tmp_path, 'wall_bias_launch', source.replace('LAUNCH', launch).replace('OPTIONAL', optional))


def test_actual_cold_profile_complete_topology_and_qkv_dispatch(tmp_path):
    helpers = '\n'.join(block(DECODER, marker) for marker in (
        '    bool wall_base_paired_kv_profile() const',
        '    bool wall_action_fp16_transfer_enabled() const',
        '    static bool wall_action_qkv_weights_compatible(',
        '    static bool wall_action_ring_weights_compatible(',
        '    bool wall_base_nvfp4_qkv_bias_profile() const',
        '    bool wall_base_nvfp4_qkv_bias_enabled(',
        '    std::vector<int64_t> qkv_bias_topology_arguments(',
        '    uint32_t qkv_fused_bias_addr(',
        '    void emit_nvfp4_qkv_bias(',
    ))
    cold_start = DECODER.index('        wall_base_nvfp4_qkv_bias_ =')
    cold = DECODER[cold_start:DECODER.index(';', cold_start) + 1]
    start = DECODER.index('        append(FmbRouteFamily::GRAPH_SCHEDULE,\n               CAUSAL_DECODER_QKV_BIAS_TOPOLOGY_SITE')
    producer = DECODER[start:DECODER.index(';', start) + 1]
    start = DECODER.index('            ctx().consume_physical_route(\n                FmbRouteFamily::GRAPH_SCHEDULE,\n                CAUSAL_DECODER_QKV_BIAS_TOPOLOGY_SITE')
    consumer = DECODER[start:DECODER.index(';', start) + 1]
    dispatch = block(DECODER, 'if (wall_base_nvfp4_qkv_bias) {')
    start = DECODER.index(dispatch) + len(dispatch)
    dispatch += ' ' + block(DECODER[start:], 'else {')
    consume = block((ROOT / 'src/core/fmb_three_stage_chunk_plan.cpp').read_text(),
                    'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(')
    consume = consume.replace('FmbPhysicalManifestConsumer::consume_route', 'consume_physical_route')
    # The source definition omits the declaration's default invocation=0.
    consume = consume.replace('int64_t invocation) const', 'int64_t invocation = 0) const')
    find = block((ROOT / 'src/core/fmb_three_stage_chunk_plan.cpp').read_text(),
                 'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(')
    find = find.replace('FmbPhysicalManifestConsumer::find_route', 'find_route')
    source = META + r'''
enum class ChunkMode{SEQUENTIAL,KV_FIRST};
struct ChunkInfo{int64_t idx=0,offset=0,len=32,kv_seq_len=416;};
struct FmbThreeStageChunkPlan{ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;struct{std::vector<ChunkInfo> chunks{{}};}compute;};
enum class FmbRouteFamily{ATTENTION,GRAPH_SCHEDULE,LINEAR,COLLECTIVE};
struct FmbRouteManifestEntry{int64_t site_id;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t> arguments;int64_t invocation;};
constexpr int64_t CAUSAL_DECODER_QKV_BIAS_TOPOLOGY_SITE=1,CAUSAL_DECODER_Q_SITE=2,CAUSAL_DECODER_K_SITE=3,CAUSAL_DECODER_V_SITE=4;
struct Manifest{std::vector<FmbRouteManifestEntry> routes;};
struct Context{Manifest manifest_;mutable std::vector<int> consumed_;bool complete_flag=true,is_causal=false;
 int64_t seq_len=32,position=384,batch_size=1;std::optional<int> attention_mask{1};FmbThreeStageChunkPlan stage_plan;
 bool complete()const{return complete_flag;}bool has_complete_physical_manifest()const{return complete_flag;}const Manifest& manifest()const{return manifest_;}
 FIND
 CONSUME
};
struct KernelCache{bool kernel64=true,kernel32=true,program64=true,program32=true;int queries=0;
 static auto& instance(){static KernelCache c;return c;}
 void* get_kernel(std::string s){++queries;return (s.find("n64")!=std::string::npos?kernel64:kernel32)?this:nullptr;}
 void* get_program(std::string s){++queries;return (s.find("n64")!=std::string::npos?program64:program32)?this:nullptr;}
};
int fused_calls=0,linear_calls=0,adds=0;std::vector<int64_t> fused_n;
void rpu_launch_wall_base_nvfp4_qkv_bias_spm_kernel(uint32_t,const at::Tensor&,uint32_t,uint32_t bias,const at::Tensor&,uint32_t,int64_t count,uint16_t layer,int64_t n){
 assert(bias!=0&&count==40&&layer<36);++fused_calls;fused_n.push_back(n);}
template<class... T>void rpu_launch_linear_spm_to_spm_acc16_kernel(T&&...){++linear_calls;}
enum class ValuOpType{ADD};
template<class... T>void rpu_launch_eltwise_binary_1xC_NxC_spm_kernel(T&&...){++adds;}
struct Model{
 struct LayerWeights{at::Tensor q_w,k_w,v_w,o_w,gate_w,up_w,down_w,input_norm_w,post_norm_w;
  at::Tensor q_ws,k_ws,v_ws,o_ws,gate_ws,up_ws,down_ws,q_bias_w,k_bias_w,v_bias_w;};
 std::vector<LayerWeights> layer_weights_{36};WallSortingConfig wall_sorting_;at::Tensor final_norm_w_,q_nvfp4_tensor_scales_,k_nvfp4_tensor_scales_,v_nvfp4_tensor_scales_;
 bool nvfp4_=true,linear_acc32_=false,wall_action_fuse_silu_mul_=true,pre_layers_residual1_ready_=true,has_qkv_bias_=true,has_qk_norm_=false;
 bool has_mrope_=true,use_silu_=true,decoder_mixed_precision_=false,adarms_=false,wall_action_weight_residency_=false,wall_action_kv_owner_fanout_=false;
 bool wall_base_nvfp4_qkv_bias_=false,partial_mrope_active_=true;int64_t seq_len_=32;std::vector<int64_t> deepstack_lang_layers_;
 int hidden=1024;Context context;Context& ctx(){return context;}const Context& ctx()const{return context;}
 int num_layers()const{return layer_weights_.size();}int hidden_size()const{return hidden;}
 int intermediate_size()const{return 2048;}int num_q_heads()const{return 16;}int num_kv_heads()const{return 8;}
 int head_dim()const{return 128;}int num_cores()const{return 8;}int attn_tp()const{return 8;}int mlp_tp()const{return 8;}
 uint32_t addr(int,const char*)const{return 0x20000000;}uint32_t layer_addr(int,int,const char*)const{return 0x20100000;}
 uint32_t nvfp4_scale_addr(const char*)const{return 0x20200000;}int decoder_projection_route(int64_t)const{return 1;}
 bool qwen3vl_decode_gemv(int64_t)const{return false;}
 HELPERS
 void bind(){COLD}
 void install(int precision=2){nvfp4_=precision==2;wall_sorting_.silu_mul=true;wall_sorting_.fp16_transfer_mask=precision==0?4:0;
  for(auto* ts:{&q_nvfp4_tensor_scales_,&k_nvfp4_tensor_scales_,&v_nvfp4_tensor_scales_}){ts->dtype=at::kFloat;ts->shape={40};}
  for(auto& l:layer_weights_){
   for(auto* w:{&l.q_w,&l.k_w,&l.v_w,&l.o_w,&l.gate_w,&l.up_w,&l.down_w})w->dtype=precision==2?at::kByte:precision==1?at::kChar:at::kHalf;
   for(auto* s:{&l.q_ws,&l.k_ws,&l.v_ws,&l.o_ws,&l.gate_ws,&l.up_ws,&l.down_ws}){s->dtype=precision==2?at::kByte:at::kHalf;s->present=precision!=0;}
   l.q_bias_w.shape={2048};l.k_bias_w.shape=l.v_bias_w.shape={1024};
  }bind();}
 bool enabled()const{return wall_base_nvfp4_qkv_bias_enabled(context.stage_plan,seq_len_,context.seq_len,context.position,
  context.is_causal,context.attention_mask.has_value(),context.batch_size,partial_mrope_active_);}
 void build(){
  auto& plan=context.stage_plan;const int64_t physical_len=seq_len_,logical_len=context.seq_len,position=context.position,topology_selector=1;
  struct Layout{bool is_causal,use_attn_mask;int64_t batch_size;}layout{context.is_causal,context.attention_mask.has_value(),context.batch_size};
  const bool partial_mrope=partial_mrope_active_;context.manifest_.routes.clear();
  auto append=[&](FmbRouteFamily f,int64_t site,int64_t sel,int64_t flags,std::vector<int64_t> args){context.manifest_.routes.push_back({site,f,sel,flags,args,0});};
  PRODUCER
  for(auto site:{CAUSAL_DECODER_Q_SITE,CAUSAL_DECODER_K_SITE,CAUSAL_DECODER_V_SITE})append(FmbRouteFamily::LINEAR,site,1,0,{});
  context.consumed_.assign(context.manifest_.routes.size(),0);
 }
 void consume_causal_route(FmbRouteFamily family,int64_t site,int64_t selector){context.consume_physical_route(family,site,selector,0,{},0);}
 void emit(){const bool wall_base_nvfp4_qkv_bias=enabled();const int64_t topology_selector=1;
  CONSUMER
  int layer_idx=35;auto& lw=layer_weights_[layer_idx];int64_t rows=32,nq=16,nkv=8,hd=128,h=1024,tp=8;
  DISPATCH
 }
};
void clear(){fused_calls=linear_calls=adds=0;fused_n.clear();}
int main(){
 for(int p:{336,384,448,512,640})for(int precision:{0,1,2}){
  Model m;m.context.position=p;m.context.stage_plan.compute.chunks[0].kv_seq_len=p+32;m.install(precision);
  assert(m.wall_base_nvfp4_qkv_bias_==(precision==2));m.build();clear();m.emit();
  assert(fused_calls==(precision==2?3:0)&&linear_calls==(precision==2?0:3)&&adds==0);
  if(precision==2)assert(fused_n==std::vector<int64_t>({2048,1024,1024}));
  for(int count:m.context.consumed_)assert(count==1);
 }
 for(auto member:{&KernelCache::kernel64,&KernelCache::kernel32,&KernelCache::program64,&KernelCache::program32}){
  auto& cache=KernelCache::instance();cache.*member=false;Model m;m.install();assert(!m.wall_base_nvfp4_qkv_bias_);
  m.build();clear();m.emit();assert(fused_calls==0&&linear_calls==3&&adds==3);cache.*member=true;
 }
 Model m;m.install();assert(m.enabled());
 auto reject_cold=[&](){m.bind();assert(!m.wall_base_nvfp4_qkv_bias_);m.build();clear();m.emit();
  assert(fused_calls==0&&linear_calls==3&&adds==3);};
 m.linear_acc32_=true;reject_cold();m.linear_acc32_=false;
 m.wall_action_fuse_silu_mul_=false;reject_cold();m.wall_action_fuse_silu_mul_=true;
 m.hidden=2048;reject_cold();m.hidden=1024;
 m.layer_weights_[12].up_w.dtype=at::kChar;reject_cold();m.layer_weights_[12].up_w.dtype=at::kByte;
 for(auto* ts:{&m.q_nvfp4_tensor_scales_,&m.k_nvfp4_tensor_scales_,&m.v_nvfp4_tensor_scales_}){
  for(int n:{36,48}){ts->shape={n};reject_cold();}ts->shape={40};ts->dtype=at::kHalf;reject_cold();ts->dtype=at::kFloat;
 }
 m.layer_weights_[35].q_bias_w.shape={2047};reject_cold();m.layer_weights_[35].q_bias_w.shape={2048};m.bind();assert(m.enabled());
 auto fallback=[&](){assert(!m.enabled());m.build();clear();m.emit();assert(fused_calls==0&&linear_calls==3&&adds==3);};
 m.context.position=320;fallback();m.context.position=384;
 m.seq_len_=20;fallback();m.seq_len_=32;m.context.seq_len=20;fallback();m.context.seq_len=32;
 m.context.is_causal=true;fallback();m.context.is_causal=false;m.context.attention_mask.reset();fallback();m.context.attention_mask=1;
 m.context.batch_size=2;fallback();m.context.batch_size=1;m.partial_mrope_active_=false;fallback();m.partial_mrope_active_=true;
 m.context.stage_plan.chunk_mode=ChunkMode::KV_FIRST;assert(!m.enabled());m.context.stage_plan.chunk_mode=ChunkMode::SEQUENTIAL;
 m.context.stage_plan.compute.chunks[0].offset=16;fallback();m.context.stage_plan.compute.chunks[0].offset=0;
 m.context.stage_plan.compute.chunks[0].len=16;fallback();m.context.stage_plan.compute.chunks[0].len=32;
 m.context.stage_plan.compute.chunks[0].kv_seq_len=400;fallback();m.context.stage_plan.compute.chunks[0].kv_seq_len=416;
 m.context.stage_plan.compute.chunks.push_back({});fallback();m.context.stage_plan.compute.chunks.pop_back();
 m.wall_action_weight_residency_=true;fallback();m.wall_action_weight_residency_=false;
 m.wall_action_kv_owner_fanout_=true;fallback();m.wall_action_kv_owner_fanout_=false;
 for(int variant:{0,1,2,3}){m.build();
  if(variant==0)m.context.manifest_.routes[0].arguments={1};
  if(variant==1)m.context.manifest_.routes[0].arguments[3]=336;
  if(variant==2)m.context.complete_flag=false;
  if(variant==3)m.context.manifest_.routes[1].selector=99;
  clear();bool rejected=false;try{m.emit();}catch(const std::exception&){rejected=true;}
  assert(rejected&&fused_calls==0&&linear_calls==0&&adds==0);m.context.complete_flag=true;
 }
 m.build();m.wall_base_nvfp4_qkv_bias_=false;clear();bool rejected=false;
 try{m.emit();}catch(const std::exception&){rejected=true;}assert(rejected&&linear_calls==0&&adds==0);
}
'''
    for key, value in [('HELPERS', helpers), ('COLD', cold), ('PRODUCER', producer),
                       ('CONSUMER', consumer), ('CONSUME', consume), ('FIND', find), ('DISPATCH', dispatch)]:
        source = source.replace(key, value)
    compile_run(tmp_path, 'wall_bias_route', source)
