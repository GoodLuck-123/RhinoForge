"""Execute exact Wall paired RoPE leaf and cold COMPLETE admission on host metadata."""
from pathlib import Path
from test_wall_base_paired_kv import block, compile_run
from test_wall_base_nvfp4_qkv_bias import META

ROOT = Path(__file__).resolve().parents[1]
ROPE = (ROOT / 'src/ops/rpu_partial_mrope.cpp').read_text()
DECODER = (ROOT / 'src/fused/rpu_qwen3_model.h').read_text()


def test_actual_paired_rope_owner_ranges_inplace_abi_and_qwen_boundary(tmp_path):
    leaf = '\n'.join(block(ROPE, marker) for marker in (
        'static void launch_partial_mrope_qk_spm_kernel(',
        'void rpu_launch_partial_mrope_qk_spm_kernel(',
        'void rpu_launch_wall_base_partial_mrope_qk_spm_kernel('))
    source = META + r'''
enum class KernelId{QWEN3VL_PARTIAL_MROPE_QK_FUSED,WALL_BASE_PARTIAL_MROPE_QK_H32};
enum class GraphDdrRegisterRole{SemanticInput};
enum class GraphDdrRegisterAccess{Read};
enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct GraphDdrRegisterAbi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Kernel_t{std::array<uint16_t,64> regs{};void reset_regs(){regs.fill(0xffff);}
 void set_regs(int i,uint16_t v){regs.at(i)=v;}}kernel;
KernelId chosen;int staged=0,queued=0;
struct Writer{std::vector<GraphDdrRegisterOperandSpec> operands;
 void write(Kernel_t& k){for(auto& o:operands){assert(o.tensor.address%256==0);uint64_t a=o.tensor.address>>8;
  k.set_regs(o.abi.lo,uint16_t(a));k.set_regs(o.abi.hi,uint16_t(a>>16));}}};
struct RpuKernelGraph{static auto& active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> v){
  chosen=id;++staged;assert(v.size()==2);
  for(int i=0;i<2;++i)assert(v[i].abi.role==GraphDdrRegisterRole::SemanticInput && v[i].abi.access==GraphDdrRegisterAccess::Read && v[i].abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi && v[i].abi.lo==2*i && v[i].abi.hi==2*i+1);
  return {v};}};
struct Queue{bool broadcast=false;void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel_t& k,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
  assert(broadcast && cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));
  assert(grid==std::vector<uint16_t>({1,uint16_t((k.regs[12]+63)/64),2}));++queued;}}queue;
#define GET_KERNEL(id) (&::kernel)
#define GET_QUEUE(n) (&::queue)
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8387584;bool ready=true;uint32_t base=0x20000000;
 bool is_initialized()const{return ready;}uint32_t addr(int core,int off){return base+core*0x1000000u+off;}}spm;
#define SPM_ALLOC spm
struct KernelCache{bool loaded=true;static auto& instance(){static KernelCache c;return c;}bool has_loaded(KernelId id){assert(id==KernelId::WALL_BASE_PARTIAL_MROPE_QK_H32);return loaded;}};
void rpu_ddr_flush(void*){}
LEAF
uint32_t word(int r){return kernel.regs[r]|(uint32_t(kernel.regs[r+1])<<16);}
template<class F>void rejects(F f){int s=staged,q=queued;bool failed=false;try{f();}catch(const std::exception&){failed=true;}assert(failed&&staged==s&&queued==q);}
int main(){
 at::Tensor c,s;c.shape=s.shape={416,64};s.address=0x10020000;
 auto run=[&](uint32_t q=0x20001000,uint32_t k=0x20008000,int64_t pos=384){rpu_launch_wall_base_partial_mrope_qk_spm_kernel(q,k,c,s,pos);};
 for(int pos:{0,336,384}){run(0x20001000,0x20008000,pos);
  assert(chosen==KernelId::WALL_BASE_PARTIAL_MROPE_QK_H32);
  assert(word(0)==c.address>>8&&word(2)==s.address>>8&&word(4)==0x20001000&&word(6)==word(4)&&word(8)==uint32_t(pos));
  assert(kernel.regs[10]==128&&kernel.regs[11]==2&&kernel.regs[12]==32&&kernel.regs[13]==128&&kernel.regs[14]==1);
  assert(word(15)==0x7000&&word(17)==0x7000&&kernel.regs[19]==65535);
 }
 run(0x20008000,0x20001000);assert(word(15)==uint32_t(-0x7000)&&word(17)==word(15));
 rejects([&]{run(0x20001000,0x20008000,385);});rejects([&]{run(0x20001000,0x20008000,-1);});
 c.shape=s.shape={415,64};rejects([&]{run();});c.shape=s.shape={416,64};
 for(auto* t:{&c,&s}){auto old=*t;t->dtype=at::kFloat;rejects([&]{run();});*t=old;t->contiguous=false;rejects([&]{run();});*t=old;t->shape[1]=32;rejects([&]{run();});*t=old;t->where.kind=c10::DeviceType::CPU;rejects([&]{run();});*t=old;}
 s.where.index=1;rejects([&]{run();});s.where.index=0;
 rejects([&]{run(0,0x20008000);});rejects([&]{run(0x20001002,0x20008000);});
 rejects([&]{run(0x20001000,0x20002000);});rejects([&]{run(0x20001000,0x207fff00);});
 spm.ready=false;rejects([&]{run();});spm.ready=true;
 KernelCache::instance().loaded=false;rejects([&]{run();});KernelCache::instance().loaded=true;
 for(int tokens:{1,192}){rpu_launch_partial_mrope_qk_spm_kernel(1,2,3,4,c,s,0,tokens,2,1,128,128,8);assert(chosen==KernelId::QWEN3VL_PARTIAL_MROPE_QK_FUSED&&kernel.regs[12]==tokens);}
 rejects([&]{rpu_launch_partial_mrope_qk_spm_kernel(1,2,3,4,c,s,0,32,2,1,128,128,8);});
}
'''
    compile_run(tmp_path, 'wall_qk_rope_leaf', source.replace('LEAF', leaf))


def test_actual_paired_rope_complete_and_fallback(tmp_path):
    helpers = '\n'.join(block(DECODER, marker) for marker in (
        '    bool wall_base_nvfp4_qkv_bias_enabled(',
        '    bool wall_base_nvfp4_qk_rope_enabled(',
        '    static std::vector<int64_t> wall_base_nvfp4_qk_rope_arguments(',
        '    void emit_wall_base_nvfp4_qk_rope('))
    start = DECODER.index('        wall_base_nvfp4_qk_rope_ =')
    cold = DECODER[start:DECODER.index(';',start)+1]
    producer = block(DECODER, '        if (!wall_base_nvfp4_qkv_rope && wall_base_nvfp4_qk_rope_enabled(plan, physical_len, logical_len,')
    planner = (ROOT / 'src/core/fmb_three_stage_chunk_plan.cpp').read_text()
    consume = block(planner, 'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(').replace('FmbPhysicalManifestConsumer::consume_route','consume_physical_route').replace('int64_t invocation) const','int64_t invocation = 0) const')
    find = block(planner, 'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(').replace('FmbPhysicalManifestConsumer::find_route','find_route')
    source = META + r'''
enum class ChunkMode{SEQUENTIAL,KV_FIRST};
struct ChunkInfo{int64_t idx=0,offset=0,len=32,kv_seq_len=416;};
struct FmbThreeStageChunkPlan{ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;struct{std::vector<ChunkInfo> chunks{{}};}compute;};
enum class FmbRouteFamily{ATTENTION,ROPE,COLLECTIVE};
constexpr int64_t CAUSAL_DECODER_WALL_BASE_QK_ROPE_SITE=5470596349203671821LL;
struct FmbRouteManifestEntry{int64_t site_id;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t> arguments;int64_t invocation;};
struct Manifest{std::vector<FmbRouteManifestEntry> routes;};
struct Context{Manifest manifest_;mutable std::vector<int> consumed_;bool complete_flag=true;int64_t position=384;
 bool complete()const{return complete_flag;}const Manifest& manifest()const{return manifest_;}
 FIND
 CONSUME
};
struct KernelCache{bool kernel=true,program=true;static auto& instance(){static KernelCache c;return c;}
 void* get_kernel(std::string s){assert(s=="wall_base_partial_mrope_qk_h32");return kernel?this:nullptr;}
 void* get_program(std::string s){assert(s=="wall_base_partial_mrope_qk_h32");return program?this:nullptr;}};
int calls=0;
void rpu_launch_wall_base_partial_mrope_qk_spm_kernel(uint32_t q,uint32_t k,const at::Tensor&,const at::Tensor&,int64_t p){assert(q==1&&k==2&&(p==336||p==384||p==448||p==512||p==640));++calls;}
struct Model{
 Context context;Context& ctx(){return context;}FmbThreeStageChunkPlan plan;
 bool wall_base_nvfp4_qkv_bias_=true,wall_base_nvfp4_qk_rope_=false,adarms_=false,decoder_mixed_precision_=false,wall_action_weight_residency_=false,wall_action_kv_owner_fanout_=false;
 WallSortingConfig wall_sorting_;int cache_batch_slot_=0;at::Tensor cos_il_keepalive_,sin_il_keepalive_;
 int64_t physical_len=32,logical_len=32;bool causal=false,mask=true,partial=true;int64_t batch=1;
 uint32_t addr(int,std::string s){return s=="q"?1:2;}
 HELPERS
 void bind(){COLD}
 Model(){bind();}
 bool enabled(){return wall_base_nvfp4_qk_rope_enabled(plan,physical_len,logical_len,context.position,causal,mask,batch,partial);}
 void build(){int64_t position=context.position;bool partial_mrope=partial;bool wall_base_nvfp4_qkv_rope=false;
  struct{bool is_causal,use_attn_mask;int64_t batch_size;}layout{causal,mask,batch};context.manifest_.routes.clear();
  auto append=[&](FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t> args,int64_t invocation){context.manifest_.routes.push_back({site,f,selector,flags,args,invocation});};
  PRODUCER
  context.consumed_.assign(context.manifest_.routes.size(),0);
 }
 void emit(){emit_wall_base_nvfp4_qk_rope(plan.compute.chunks.front(),context.position);}
};
template<class F>void rejects(F f){int before=calls;bool failed=false;try{f();}catch(const std::exception&){failed=true;}assert(failed&&before==calls);}
int main(){
 for(int p:{336,384,448,512,640}){Model m;m.context.position=p;m.plan.compute.chunks[0].kv_seq_len=p+32;m.build();assert(m.enabled()&&m.context.manifest_.routes.size()==1);m.emit();assert(m.context.consumed_[0]==1);
  const auto r=m.context.manifest_.routes[0];assert(r.arguments==std::vector<int64_t>({1,32,p,1,2,1,128,128,8,1,1,2}));
  for(int i=0;i<12;++i){m.context.manifest_.routes[0]=r;++m.context.manifest_.routes[0].arguments[i];rejects([&]{m.emit();});}
  m.context.manifest_.routes[0]=r;m.context.manifest_.routes[0].selector=2;rejects([&]{m.emit();});m.context.manifest_.routes.clear();rejects([&]{m.emit();});
 }
 for(int i=0;i<16;++i){Model m;switch(i){
 case 0:m.wall_base_nvfp4_qk_rope_=false;break;case 1:m.wall_base_nvfp4_qkv_bias_=false;break;
 case 2:m.physical_len=31;break;case 3:m.logical_len=31;break;case 4:m.context.position=400;break;
 case 5:m.causal=true;break;case 6:m.mask=false;break;case 7:m.batch=2;break;case 8:m.partial=false;break;
 case 9:m.plan.compute.chunks[0].offset=1;break;case 10:m.plan.compute.chunks[0].kv_seq_len=415;break;
 case 11:m.plan.chunk_mode=ChunkMode::KV_FIRST;break;case 12:m.plan.compute.chunks.push_back({});break;
 case 13:m.cache_batch_slot_=1;break;case 14:m.wall_sorting_.action_activation_reuse=true;break;case 15:m.wall_action_weight_residency_=true;break;
 }assert(!m.enabled());m.build();assert(m.context.manifest_.routes.empty());}
 for(int i=0;i<2;++i){auto& c=KernelCache::instance();c.kernel=i==0;c.program=i==1;Model m;assert(!m.enabled());c.kernel=c.program=true;}
}
'''
    for key,value in [('HELPERS',helpers),('COLD',cold),('PRODUCER',producer),('FIND',find),('CONSUME',consume)]:
        source = source.replace(key,value)
    compile_run(tmp_path, 'wall_qk_rope_route', source)
