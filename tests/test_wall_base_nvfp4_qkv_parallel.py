"""Execute real Wall H32 parallel Q/K/V bias leaf and COMPLETE route."""
from pathlib import Path
from test_wall_base_paired_kv import block, compile_run
from test_wall_base_nvfp4_qkv_bias import META

ROOT=Path(__file__).resolve().parents[1]
DECODER=(ROOT/'src/fused/rpu_qwen3_model.h').read_text()


def test_actual_parallel_qkv_typed_abi_ranges_and_counters(tmp_path):
    leaf=block((ROOT/'src/ops/rpu_linear.cpp').read_text(),'void rpu_launch_wall_base_nvfp4_qkv_parallel_bias_spm_kernel(')
    source=META+r'''
#include "ops/rpu_linear_tiling.h"
enum class KernelId{WALL_BASE_NVFP4_QKV_PARALLEL_BIAS_H32};
enum class GraphDdrRegisterRole{PackedFp4ModelWeight,Fp8BlockScale};
enum class GraphDdrRegisterAccess{Read};
enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Kernel{std::array<uint16_t,132> regs{};int fail_reg=-1;void reset_regs(){regs.fill(0xffff);}
 uint32_t set_regs(int r,uint16_t v){if(r==fail_reg)return 5;regs.at(r)=v;return 0;}
 uint32_t set_regs(int r,uint32_t v){if(r==fail_reg)return 5;assert(r%2==0);regs.at(r)=uint16_t(v);regs.at(r+1)=uint16_t(v>>16);return 0;}
 uint32_t set_regs(int,int)=delete;}kernel;
int staged=0,queued=0;
struct Writer{std::vector<GraphDdrRegisterOperandSpec> operands;void write(Kernel& k){for(auto& o:operands){auto a=o.tensor.address>>8;k.set_regs(o.abi.lo,uint16_t(a));k.set_regs(o.abi.hi,uint16_t(a>>16));}}};
struct RpuKernelGraph{static auto& active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> v){
  assert(id==KernelId::WALL_BASE_NVFP4_QKV_PARALLEL_BIAS_H32&&v.size()==6);++staged;
  const int lo[]={2,20,32,34,42,44};for(int i=0;i<6;++i)assert(v[i].abi.lo==lo[i]&&v[i].abi.hi==lo[i]+1&&v[i].abi.access==GraphDdrRegisterAccess::Read&&v[i].abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi&&v[i].abi.role==(i%2==0?GraphDdrRegisterRole::PackedFp4ModelWeight:GraphDdrRegisterRole::Fp8BlockScale));return {v};}};
struct Queue{bool broadcast=false;void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel& k,std::vector<uint16_t> grid,std::vector<uint8_t> cores){assert(broadcast&&grid==std::vector<uint16_t>({8,1,1})&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));for(int r:{128,129,130,131})assert(k.regs[r]==0);assert(k.regs[64]==8&&k.regs[65]==1&&k.regs[66]==1);++queued;k.regs[128]=8;k.regs[130]=13;}}queue;
#define GET_KERNEL(id) (&::kernel)
#define GET_QUEUE(n) (&::queue)
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8387584;bool ready=true;uint32_t base=0x20000000;
 bool is_initialized()const{return ready;}uint32_t addr(int core,int off){return base+core*0x1000000u+off;}}spm;
#define SPM_ALLOC spm
struct KernelCache{bool loaded=true;static auto& instance(){static KernelCache c;return c;}bool has_loaded(KernelId){return loaded;}};
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uintptr_t>(p);}void rpu_ddr_flush(void*){}
LEAF
uint32_t word(int r){return kernel.regs[r]|(uint32_t(kernel.regs[r+1])<<16);}
template<class F>void rejects(F f){int s=staged,q=queued;bool fail=false;try{f();}catch(const std::exception&){fail=true;}assert(fail&&s==staged&&q==queued);}
int main(){
 at::Tensor qw,kw,vw,qs,ks,vs;qw.shape={2048,512};kw.shape=vw.shape={1024,512};qs.shape={64,2048};ks.shape=vs.shape={64,1024};
 int n=0;for(auto* t:{&qw,&kw,&vw,&qs,&ks,&vs}){t->dtype=at::kByte;t->address=0x10000000+(n++)*0x200000;}
 std::array<uint32_t,10> sp{0x20001000,0x20020000,0x20030000,0x20040000,0x20050000,0x20050200,0x20050300,0x20050400,0x20050500,0x20050600};int64_t count=40;uint16_t layer=35;
 auto run=[&]{rpu_launch_wall_base_nvfp4_qkv_parallel_bias_spm_kernel(sp[0],qw,kw,vw,sp[1],sp[2],sp[3],qs,ks,vs,sp[4],sp[5],sp[6],sp[7],sp[8],sp[9],count,layer);};
 for(int i=0;i<2;++i){run();const int regs[]={0,4,38,48,6,40,50,24,36,46};for(int j=0;j<10;++j)assert(word(regs[j])==sp[j]);
  const int ddr_regs[]={2,32,42,20,34,44};n=0;for(auto* t:{&qw,&kw,&vw,&qs,&ks,&vs})assert(word(ddr_regs[n++])==t->address>>8);
  assert(kernel.regs[8]==32&&word(10)==256&&kernel.regs[12]==1024&&kernel.regs[13]==16&&kernel.regs[14]==1&&kernel.regs[15]==8&&word(16)==2048&&word(18)==512&&word(22)==4096&&kernel.regs[26]==1&&kernel.regs[27]==35);
  for(int r:{11,19,23})assert(kernel.regs[r]==0);}
 layer=36;rejects(run);layer=35;count=39;rejects(run);count=40;
 for(int i=0;i<10;++i){auto old=sp[i];sp[i]=0;rejects(run);sp[i]=old+1;rejects(run);sp[i]=0x207fffe0;rejects(run);sp[i]=old;}
 for(int i=1;i<10;++i){auto old=sp[i];sp[i]=sp[0]+32;rejects(run);sp[i]=old;}
 for(auto* t:{&qw,&kw,&vw,&qs,&ks,&vs}){auto old=*t;t->shape[0]--;rejects(run);*t=old;t->dtype=at::kHalf;rejects(run);*t=old;t->storage_off=1;rejects(run);*t=old;t->contiguous=false;rejects(run);*t=old;t->address++;rejects(run);*t=old;t->where.kind=c10::DeviceType::CPU;rejects(run);*t=old;}
 auto old=kw;kw.address=qw.address;rejects(run);kw=old;kw.where.index=1;rejects(run);kw=old;
 KernelCache::instance().loaded=false;rejects(run);KernelCache::instance().loaded=true;
 spm.ready=false;rejects(run);spm.ready=true;
 for(int reg:{64,65,66,128,130}){kernel.fail_reg=reg;int before=queued;bool fail=false;try{run();}catch(const std::exception&){fail=true;}assert(fail&&queued==before);kernel.fail_reg=-1;}

}
'''
    compile_run(tmp_path,'parallel_qkv_leaf',source.replace('LEAF',leaf))


def test_actual_parallel_qkv_complete_route_and_fallback(tmp_path):
    helpers='\n'.join(block(DECODER,m) for m in [
        '    bool wall_base_nvfp4_qkv_bias_enabled(',
        '    bool wall_base_nvfp4_qk_rope_enabled(',
        '    bool wall_base_nvfp4_kv_parallel_enabled(',
        '    bool wall_base_nvfp4_qkv_parallel_enabled(',
        '    static std::vector<int64_t> wall_base_nvfp4_qkv_parallel_arguments(',
        '    void emit_wall_base_nvfp4_qkv_parallel('])
    start=DECODER.index('        wall_base_nvfp4_qkv_parallel_ =');cold=DECODER[start:DECODER.index(';',start)+1]
    producer=block(DECODER,'if (wall_base_nvfp4_qkv_parallel_enabled(plan, physical_len, logical_len,')
    emission=block(DECODER,'if (wall_base_nvfp4_qkv_parallel) {')
    planner=(ROOT/'src/core/fmb_three_stage_chunk_plan.cpp').read_text()
    consume=block(planner,'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(').replace('FmbPhysicalManifestConsumer::consume_route','consume_physical_route').replace('int64_t invocation) const','int64_t invocation = 0) const')
    find=block(planner,'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(').replace('FmbPhysicalManifestConsumer::find_route','find_route')
    source=META+r'''
enum class ChunkMode{SEQUENTIAL,KV_FIRST};
struct ChunkInfo{int64_t idx=0,offset=0,len=32,kv_seq_len=416;};
struct FmbThreeStageChunkPlan{ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;struct{std::vector<ChunkInfo> chunks{{}};}compute;};
enum class FmbRouteFamily{ATTENTION,LINEAR,COLLECTIVE};
constexpr int64_t CAUSAL_DECODER_Q_SITE=403971716841024546LL,CAUSAL_DECODER_WALL_BASE_NVFP4_QKV_PARALLEL_SITE=489263202852899179LL;
struct FmbRouteManifestEntry{int64_t site_id;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t> arguments;int64_t invocation;};
struct Manifest{std::vector<FmbRouteManifestEntry> routes;};
struct Context{Manifest manifest_;mutable std::vector<int> consumed_;bool complete_flag=true;int64_t position=384;
 bool complete()const{return complete_flag;}const Manifest& manifest()const{return manifest_;}
 FIND
 CONSUME
};
struct KernelCache{bool kernel=true,program=true;static auto& instance(){static KernelCache c;return c;}
 void* get_kernel(std::string s){assert(s=="wall_base_nvfp4_qkv_parallel_bias_h32");return kernel?this:nullptr;}
 void* get_program(std::string s){assert(s=="wall_base_nvfp4_qkv_parallel_bias_h32");return program?this:nullptr;}};
int kv=0,q=0;
void rpu_launch_wall_base_nvfp4_qkv_parallel_bias_spm_kernel(uint32_t x,const at::Tensor&,const at::Tensor&,const at::Tensor&,uint32_t qo,uint32_t k,uint32_t v,const at::Tensor&,const at::Tensor&,const at::Tensor&,uint32_t qb,uint32_t kb,uint32_t vb,uint32_t qt,uint32_t kt,uint32_t vt,int64_t count,uint16_t layer){assert(x==1&&qo==8&&k==2&&v==3&&qb==9&&kb==6&&vb==7&&qt==10&&kt==4&&vt==5&&count==40&&layer<36);++kv;}
struct Model{
 struct LayerWeights{at::Tensor q_w,k_w,v_w,q_ws,k_ws,v_ws;}lw;
 Context context;Context& ctx(){return context;}FmbThreeStageChunkPlan plan;
 bool wall_base_nvfp4_qkv_bias_=true,wall_base_nvfp4_qk_rope_=true,wall_base_nvfp4_kv_parallel_=true,wall_base_nvfp4_qkv_parallel_=false,adarms_=false,decoder_mixed_precision_=false,wall_action_weight_residency_=false,wall_action_kv_owner_fanout_=false;
 WallSortingConfig wall_sorting_;int cache_batch_slot_=0;at::Tensor k_nvfp4_tensor_scales_,q_nvfp4_tensor_scales_;
 int64_t physical_len=32,logical_len=32,seq_len_=32;bool causal=false,mask=true,partial=true;int64_t batch=1;
 uint32_t addr(int,std::string s){return s=="input_norm"?1:s=="k"?2:s=="v"?3:8;}
 uint32_t nvfp4_scale_addr(std::string s){return s=="nvfp4_k_ts"?4:s=="nvfp4_v_ts"?5:10;}
 uint32_t layer_addr(int layer,int,std::string s){assert(layer<36);return s=="k_bias"?6:s=="v_bias"?7:9;}
 int decoder_projection_route(int64_t){return 1;}
 HELPERS
 void bind(){COLD}
 Model(){for(auto* t:{&k_nvfp4_tensor_scales_,&q_nvfp4_tensor_scales_}){t->dtype=at::kFloat;t->shape={40};}bind();}
 bool enabled(){return wall_base_nvfp4_qkv_parallel_enabled(plan,physical_len,logical_len,context.position,causal,mask,batch,partial);}
 void build(){int64_t position=context.position;bool partial_mrope=partial;
  struct{bool is_causal,use_attn_mask;int64_t batch_size;}layout{causal,mask,batch};context.manifest_.routes.clear();
  auto append=[&](FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t> args,int64_t invocation){context.manifest_.routes.push_back({site,f,selector,flags,args,invocation});};
  auto append_linear=[&](int64_t site,int64_t selector){append(FmbRouteFamily::LINEAR,site,selector,0,{},0);};
  PRODUCER
  context.consumed_.assign(context.manifest_.routes.size(),0);
 }
 void consume_causal_route(FmbRouteFamily f,int64_t site,int64_t selector){context.consume_physical_route(f,site,selector,0,{},0);}
 int old_qkv=0;
 void emit(int layer_idx){auto& chunk=plan.compute.chunks.front();bool wall_base_nvfp4_qkv_parallel=enabled();int64_t nq=16,hd=128;
  EMISSION else {++old_qkv;}
 }
};
template<class F>void rejects(F f){int before=kv;bool fail=false;try{f();}catch(const std::exception&){fail=true;}assert(fail&&before==kv);}
int main(){
 for(int64_t p:{336,384,448,512,640}){
  kv=q=0;Model m;m.context.position=p;m.plan.compute.chunks[0].kv_seq_len=p+32;m.build();
  assert(m.enabled()&&m.context.manifest_.routes.size()==1);auto route=m.context.manifest_.routes[0];
  assert(route.arguments==std::vector<int64_t>({1,32,p,1,2048,1024,1024,1024,8,320,64,128,40,8,1,1,4,2,2,1,8,0,36}));
  for(int layer=0;layer<36;++layer)m.emit(layer);assert(q==0&&kv==36&&m.old_qkv==0);
  for(int visited:m.context.consumed_)assert(visited==1);
  for(int i=0;i<23;++i){m.context.manifest_.routes[0]=route;++m.context.manifest_.routes[0].arguments[i];rejects([&]{m.emit(0);});}
  m.context.manifest_.routes[0]=route;rejects([&]{m.emit_wall_base_nvfp4_qkv_parallel(m.lw,36,m.plan.compute.chunks[0]);});
  m.context.position=352;rejects([&]{m.emit_wall_base_nvfp4_qkv_parallel(m.lw,0,m.plan.compute.chunks[0]);});
 }
 for(int i=0;i<17;++i){Model x;switch(i){case 0:x.context.position=352;x.plan.compute.chunks[0].kv_seq_len=384;break;case 1:x.wall_base_nvfp4_qkv_bias_=false;break;case 2:x.physical_len=31;break;case 3:x.logical_len=31;break;case 4:x.causal=true;break;case 5:x.mask=false;break;case 6:x.partial=false;break;case 7:x.batch=2;break;case 8:x.plan.compute.chunks[0].offset=1;break;case 9:x.plan.chunk_mode=ChunkMode::KV_FIRST;break;case 10:x.wall_base_nvfp4_qk_rope_=false;break;case 11:x.cache_batch_slot_=1;break;case 12:x.context.position=336;break;case 13:x.wall_sorting_.action_activation_reuse=true;break;case 14:x.wall_action_weight_residency_=true;break;case 15:x.wall_action_kv_owner_fanout_=true;break;case 16:x.wall_base_nvfp4_kv_parallel_=false;break;}
  assert(!x.enabled());x.build();assert(x.context.manifest_.routes.empty());int before=kv;x.emit(0);assert(x.old_qkv==1&&kv==before);}
 for(int i=0;i<2;++i){auto& c=KernelCache::instance();c.kernel=i==0;c.program=i==1;Model x;assert(!x.enabled());x.build();x.emit(0);assert(x.old_qkv==1);c.kernel=c.program=true;}
}
'''
    for key,value in [('HELPERS',helpers),('COLD',cold),('PRODUCER',producer),('EMISSION',emission),('FIND',find),('CONSUME',consume)]:source=source.replace(key,value)
    compile_run(tmp_path,'parallel_qkv_route',source)
