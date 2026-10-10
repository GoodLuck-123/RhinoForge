"""Execute the exact Vision paired Q/K RoPE contract on host metadata."""
from pathlib import Path
import re

from test_wall_base_paired_kv import block, compile_run
from test_wall_base_nvfp4_qkv_bias import META


ROOT = Path(__file__).resolve().parents[1]
VISION_PATH = ROOT / 'src/fused/rpu_qwen25vl_vision_model.cpp'
ROPE_PATH = ROOT / 'src/ops/rpu_rope_2d.cpp'

TENSOR_META = META.replace('enum ScalarType {', 'enum ScalarType { kShort,')
TENSOR_META = TENSOR_META.replace('#include <c10/core/DeviceType.h>',
    '#include <c10/core/DeviceType.h>\n#include <c10/core/Layout.h>\n'
    '#include <limits>\n#include <tuple>\n#include <unordered_set>')
TENSOR_META = TENSOR_META.replace('int64_t storage_off=0;',
    'c10::Layout layout_value=c10::Layout::Strided; '
    'c10::Layout layout()const{return layout_value;} int64_t storage_off=0;')


def _replace(source, substitutions):
    for key, value in substitutions.items():
        source = re.sub(r'(?m)^[ \t]*' + re.escape(key) + r'[ \t]*$',
                        lambda match: value, source)
    return source


def test_actual_vision_qk_rope_leaf_owned_ddr_spm_and_packet(tmp_path):
    rope = ROPE_PATH.read_text()
    leaf = '\n'.join(block(rope, marker) for marker in (
        'static void launch_rope_2d_spm_kernel(',
        'void rpu_launch_wall_base_vision_qk_rope_2d_spm_kernel(',
    ))
    cache = (ROOT / 'src/core/rpu_kernel_cache.h').read_text()
    source = TENSOR_META + r'''
KERNEL_IDS
KERNEL_NAMES
OPTIONAL
enum class GraphDdrRegisterRole {SemanticInput};
enum class GraphDdrRegisterAccess {Read};
enum class GraphDdrRegisterEncoding {DevAddrShift8LoHi};
struct GraphDdrRegisterAbi {GraphDdrRegisterRole role; GraphDdrRegisterAccess access;
 GraphDdrRegisterEncoding encoding; int lo,hi;};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Kernel_t {
 std::array<uint16_t,68> regs{}; int reject=-1;
 void reset_regs(){regs.fill(0xffff);}
 uint32_t set_regs(uint32_t r,uint16_t v){if(int(r)==reject)return 5;regs.at(r)=v;return 0;}
 uint32_t set_regs(uint32_t,int32_t)=delete;
} kernel;
struct KernelCache {
 bool loaded=true,available=true;
 static auto& instance(){static KernelCache k;return k;}
 bool has_loaded(KernelId id)const{assert(id==KernelId::WALL_BASE_VISION_QK_ROPE_2D_SPM);return loaded;}
};
int staged=0,queued=0,flushes=0; uintptr_t staged_owner=0;
KernelId chosen;
struct Writer {
 std::vector<GraphDdrRegisterOperandSpec> operands;
 void write(Kernel_t& k){for(auto& o:operands){auto a=o.tensor.address>>8;
  TORCH_CHECK(k.set_regs(o.abi.lo,uint16_t(a))==0,"writer low rejected");
  TORCH_CHECK(k.set_regs(o.abi.hi,uint16_t(a>>16))==0,"writer high rejected");}}
};
using GraphKernelRegisterWriter=Writer;
struct RpuKernelGraph {
 bool census=false,reject=false;
 static auto& active(){static RpuKernelGraph g;return g;}
 bool kernel_register_census_active()const{return census;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> v){
  assert(id==KernelId::WALL_BASE_VISION_QK_ROPE_2D_SPM&&v.size()==1);
  const auto& a=v[0].abi;
  assert(a.role==GraphDdrRegisterRole::SemanticInput&&a.access==GraphDdrRegisterAccess::Read);
  assert(a.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi&&a.lo==0&&a.hi==1);
  if(reject)throw std::runtime_error("owner proof denied");
  ++staged;staged_owner=v[0].tensor.address;return {v};}
};
struct Queue {
 bool broadcast=false;
 void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel_t& k,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
  assert(broadcast&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));
  assert(grid==std::vector<uint16_t>({1,uint16_t((k.regs[14]+127)/128),1}));
  ++queued;}
} queue;
Kernel_t* get_kernel(KernelId id){chosen=id;assert(id==KernelId::WALL_BASE_VISION_QK_ROPE_2D_SPM||id==KernelId::ROPE_2D_SPM);
 return KernelCache::instance().available?&kernel:nullptr;}
Queue* get_queue(int cores){assert(cores==8);return &queue;}
#define GET_KERNEL(id) get_kernel(id)
#define GET_QUEUE(n) get_queue(n)
struct SpmAllocator {
 static constexpr uint32_t SPM_USABLE=8387584;
 bool ready=true;uint32_t base=0x20000000;
 bool is_initialized()const{return ready;}
 uint32_t addr(int core,uint32_t off)const{return base+core*0x1000000u+off;}
} spm;
#define SPM_ALLOC spm
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uintptr_t>(p);}
void rpu_ddr_flush(void*){++flushes;}
constexpr int64_t kNumEltPerWarp=32,kNumTokenPerWarp=128;
uint16_t ceil_div_u16(int64_t n,int64_t d){return uint16_t((n+d-1)/d);}
LEAF
uint32_t word(int r){return kernel.regs[r]|(uint32_t(kernel.regs[r+1])<<16);}
int main(){
 assert(is_optional_kernel("wall_base_vision_qk_rope_2d_spm_m672_m324"));
 assert(std::string(KERNEL_ID_NAMES[static_cast<size_t>(KernelId::WALL_BASE_VISION_QK_ROPE_2D_SPM)])==
        "wall_base_vision_qk_rope_2d_spm_m672_m324");
 assert(!is_optional_kernel("rope_2d_spm"));
 using Slabs=std::array<uint32_t,4>;
 constexpr uint32_t b=0x20000000;
 for(int64_t rows:{324,672}){
  const uint32_t qbytes=uint32_t(rows*2*80*2),tablebytes=128*20*2;
  const Slabs exact{b+0x1000,b+0x40000,b+0x80000,b+0x82000};
  at::Tensor pos;pos.dtype=at::kShort;pos.shape={rows==324?328:672,2};
  auto invoke=[&](Slabs s,const at::Tensor& p,int64_t offset=0,int64_t m=0,int cores=8){
   rpu_launch_wall_base_vision_qk_rope_2d_spm_kernel(s[0],s[1],s[2],s[3],p,offset,m?m:rows,cores);};
  auto rejects=[&](auto f,bool before_stage=true){int q=queued,t=staged;bool rejected=false;
   try{f();}catch(const std::exception&){rejected=true;}
   assert(rejected&&q==queued&&(!before_stage||t==staged));};
  for(bool census:{false,true}){
   const int64_t offset=0;
   RpuKernelGraph::active().census=census;queue.broadcast=false;
   const auto before=staged;invoke(exact,pos,offset);
   assert(staged==before+1&&staged_owner==pos.address);
   assert(word(0)==pos.address/256&&word(2)==exact[2]&&word(4)==exact[3]);
   assert(word(6)==exact[0]&&word(8)==exact[1]&&word(10)==uint32_t(offset));
   assert(kernel.regs[12]==80&&kernel.regs[13]==2&&kernel.regs[14]==rows&&kernel.regs[15]==80);
   for(int r=16;r<24;++r)assert(kernel.regs[r]==0);
   assert(chosen==KernelId::WALL_BASE_VISION_QK_ROPE_2D_SPM);
   assert(kernel.regs[64]==1&&kernel.regs[65]==(rows+127)/128&&kernel.regs[66]==1);
  }
  if(rows==324)for(int64_t backing_rows:{324,327}){
   auto short_owner=pos;short_owner.shape[0]=backing_rows;
   rejects([&]{invoke(exact,short_owner);});
  }
  {auto null_owner=pos;null_owner.address=0;rejects([&]{invoke(exact,null_owner);});}
  for(int i=0;i<4;++i){
   for(uint32_t address:{uint32_t{0},b-256,exact[i]+2,exact[i]+32,b+0x1000000u}){
    auto bad=exact;bad[i]=address;rejects([&]{invoke(bad,pos);});}
   const uint32_t logical_bytes=i<2?qbytes:tablebytes;
   auto edge=exact;edge[i]=b+SpmAllocator::SPM_USABLE-logical_bytes;
   // Each logical extent is 256B-aligned. Ending exactly at SPM_USABLE
   // leaves no room for the final masked vector's additional 24B read.
   assert(edge[i]%256==0);
   rejects([&]{invoke(edge,pos);});
   edge[i]-=256;invoke(edge,pos);
   for(int j=0;j<i;++j){auto bad=exact;bad[i]=bad[j];rejects([&]{invoke(bad,pos);});
    bad[i]=bad[j]+256;rejects([&]{invoke(bad,pos);});}
  }
  // Read envelopes may enter another valid slab: only logical writes must
  // be disjoint. Exercise adjacency for Q/K and both lookup tables.
  const Slabs adjacent{b+0x1000,b+0x1000+qbytes,
      b+0x1000+2*qbytes,b+0x1000+2*qbytes+tablebytes};
  invoke(adjacent,pos);
  auto reverse=exact;std::swap(reverse[0],reverse[1]);invoke(reverse,pos);
  for(int64_t m:{1,320,336,648,768})rejects([&]{invoke(exact,pos,0,m);});
  for(int cores:{0,1,4,9})rejects([&]{invoke(exact,pos,0,rows,cores);});
  for(int64_t off:{int64_t{-1},int64_t{1},int64_t{128},int64_t{UINT32_MAX}+1})
   rejects([&]{invoke(exact,pos,off);});
  for(int variant=0;variant<10;++variant){auto bad=pos;switch(variant){
   case 0:bad.present=false;break;case 1:bad.dtype=at::kHalf;break;
   case 2:bad.where.kind=c10::DeviceType::CPU;break;case 3:bad.contiguous=false;break;
   case 4:bad.layout_value=c10::Layout::Sparse;break;case 5:bad.shape={rows*2};break;
   case 6:bad.shape[1]=3;break;case 7:bad.shape[0]=rows-1;break;
   case 8:bad.address+=2;break;case 9:bad.address=uint64_t{1}<<40;break;
}
   rejects([&]{invoke(exact,bad);});}
  spm.ready=false;rejects([&]{invoke(exact,pos);});spm.ready=true;
  // Core7's last logical byte is UINT32_MAX, but its extra read overflows.
  spm.base=uint32_t((uint64_t{1}<<32)-7*0x1000000u-(exact[3]-b)-tablebytes);
  auto high=exact;for(auto& p:high)p=p-b+spm.base;
  assert(uint64_t(spm.addr(7,0))+high[3]-spm.base+tablebytes-1==UINT32_MAX);
  rejects([&]{invoke(high,pos);});
  spm.base-=256;for(auto& p:high)p-=256;invoke(high,pos);spm.base=b;
  // The shared default-ID helper keeps the legacy packet and leaves its
  // secondary registers unchanged; only the paired variant resets them.
  kernel.regs.fill(0x5a5a);int prior=staged;
  auto legacy_pos=pos;legacy_pos.shape[0]=rows+128;
  launch_rope_2d_spm_kernel(exact[0],exact[0],exact[2],exact[3],
      legacy_pos.data_ptr<int16_t>(),128,rows,2,80,80,8,nullptr);
  assert(chosen==KernelId::ROPE_2D_SPM&&staged==prior);
  assert(word(0)==pos.address/256&&word(6)==exact[0]&&word(8)==exact[0]&&word(10)==128);
  for(int r=16;r<24;++r)assert(kernel.regs[r]==0x5a5a);
  KernelCache::instance().loaded=false;rejects([&]{invoke(exact,pos);});
  KernelCache::instance().loaded=true;
  KernelCache::instance().available=false;rejects([&]{invoke(exact,pos);},false);
  KernelCache::instance().available=true;
  RpuKernelGraph::active().reject=true;rejects([&]{invoke(exact,pos);});
  RpuKernelGraph::active().reject=false;
  for(int reg:{0,1}){kernel.reject=reg;rejects([&]{invoke(exact,pos);},false);kernel.reject=-1;}
 }
}
'''
    source = _replace(source, {
        'LEAF': leaf,
        'KERNEL_IDS': block(cache, 'enum class KernelId : uint16_t') + ';',
        'KERNEL_NAMES': block(cache, 'static constexpr const char* KERNEL_ID_NAMES[]') + ';',
        'OPTIONAL': block((ROOT / 'src/core/rpu_kernel_cache.inc').read_text(),
                          'static bool is_optional_kernel('),
    })
    compile_run(tmp_path, 'wall_vision_qk_rope_leaf', source)


def test_actual_cold_admission_complete_route_and_single_chunk_boundary(tmp_path):
    vision = VISION_PATH.read_text()
    planner = (ROOT / 'src/core/fmb_three_stage_chunk_plan.cpp').read_text()
    cache = (ROOT / 'src/core/rpu_kernel_cache.h').read_text()
    helper_markers = (
        '    bool padded_pair_enabled() const',
        '    bool small_image_row_weight_outer_enabled(',
        '    bool small_image_fp16_row_profile(',
        '    bool small_image_gate_up_enabled(',
        '    bool small_image_native_window_shape() const',
        '    bool small_image_native_window_enabled() const',
        '    bool small_image_raw_kv_only_enabled() const',
        '    bool small_image_resident_mask_enabled() const',
        '    bool small_image_single_raw_query_enabled() const',
        '    bool small_image_qk_rope_enabled(',
        '    std::vector<int64_t> small_image_qk_rope_arguments(',
        '    void validate_small_image_qk_rope_plan(',
        '    void consume_small_image_qk_rope(',
        '    int64_t small_image_schedule_flags() const',
    )
    helpers = '\n'.join(block(vision, marker) for marker in helper_markers)
    cold_start = vision.index('        small_qk_rope_available_ =')
    cold = vision[cold_start:vision.index(';', cold_start) + 1]
    producer_start = vision.index('                if (small_image_qk_rope_enabled(')
    producer = block(vision[producer_start:], 'if (small_image_qk_rope_enabled(')
    tail = vision[producer_start + vision[producer_start:].index(producer) + len(producer):].lstrip()
    while tail.startswith('else'):
        arm = block(tail, 'else')
        producer += ' ' + arm
        tail = tail[len(arm):].lstrip()
    consumer = '\n'.join(block(planner, marker).replace(
        'FmbPhysicalManifestConsumer::', '') for marker in (
            'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(',
            'const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route(',
        ))
    source = TENSOR_META + r'''
#include <atomic>
#include <bitset>
#include <memory>
#undef TORCH_INTERNAL_ASSERT
#define TORCH_INTERNAL_ASSERT(x) TORCH_CHECK(x)
KERNEL_IDS
AVAILABILITY
struct KernelCache {
 KernelPayloadAvailability availability;
 static auto& instance(){static KernelCache c;return c;}
 bool has_loaded(KernelId id)const{return availability.has_loaded(id);}
};
void publish(bool present){KernelPayloadAvailability::ExtendedPayloads b;
 b.set(size_t(KernelId::WALL_BASE_VISION_QK_ROPE_2D_SPM)-KernelPayloadAvailability::kExtendedBegin,present);
 KernelCache::instance().availability.publish(false,false,false,false,false,false,b);}
ROPE_ROUTES
SITE
LEGACY_SITES
constexpr int64_t FMB_ROUTE_FLAG_ROPE_TABLE_SPM=1LL<<57;
constexpr int64_t FMB_ROUTE_FLAG_ROPE_TABLE_DDR=1LL<<56;
enum class VisionSchedule {SISC,MISC,MIMC};
enum class ChunkMode {SEQUENTIAL,KV_FIRST};
struct ChunkInfo {int64_t idx,offset,len,kv_seq_len;};
struct FmbExecutionSpan {int64_t offset,len,group_id=-1;};
struct ResolvedFmbStageChunks {
 struct {int64_t chunk_size,num_chunks;}plan;
 std::vector<ChunkInfo> chunks;
};
struct FmbThreeStageChunkPlan {
 ResolvedFmbStageChunks input,qkv,compute;
 std::vector<FmbExecutionSpan> spans;
 ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;
};
MAKE_STAGE
enum class FmbRouteFamily {ATTENTION,ROPE,COLLECTIVE};
struct FmbRouteManifestEntry {int64_t site_id;FmbRouteFamily family;
 int64_t selector,flags;std::vector<int64_t> arguments;int64_t invocation;};
struct Manifest {std::vector<FmbRouteManifestEntry> routes;};
struct Context {
 FmbThreeStageChunkPlan stage_plan;Manifest value;
 mutable std::vector<uint8_t> consumed_;
 bool ready=true;bool complete()const{return ready;}
 bool has_complete_physical_manifest()const{return complete();}
 const Manifest& manifest()const{return value;}
 CONSUMER
 auto& consume_physical_route(FmbRouteFamily f,int64_t s,int64_t r,int64_t flags,
     at::IntArrayRef args,int64_t i){return consume_route(f,s,r,flags,args,i);}
};
enum class RpuRmsNormSpmRoute {BASE};
struct Model {
 struct Sorting {bool vision_row_reuse=true,silu_mul=true;uint32_t fp16_transfer_mask=0;} wall_sorting_;
 bool generic=true,mixed_precision_=false,per_window_sdpa_enabled_=false,z1_bound_=false;
 bool debug=false,merger=true,window_mask_present_=true,rope_spm_enabled_=true,has_rope_=true;
 bool small_qk_rope_available_=false;
 int64_t layers=32,hidden=1280,heads=16,kvheads=16,dim=80,intermediate=3584;
 int64_t image_batch_count_=1,current_num_patches_=324,valid_patches_per_image_=0,max_hw_=128;
 VisionSchedule schedule_=VisionSchedule::SISC;
 std::vector<int64_t> fullatt_block_indexes_{7,15,23,31},cu_window_;
 bool is_generic()const{return generic;}bool merger_active()const{return merger;}
 bool get_debug_export()const{return debug;}int64_t num_layers()const{return layers;}
 int64_t hidden_size()const{return hidden;}int64_t num_q_heads()const{return heads;}
 int64_t num_kv_heads()const{return kvheads;}int64_t head_dim()const{return dim;}
 int64_t intermediate_size()const{return intermediate;}
 LAYER_WEIGHTS
 std::vector<LayerWeights> layer_weights_=std::vector<LayerWeights>(32);
 at::Tensor freq_cos_,freq_sin_;
 Context context;auto& ctx(){return context;}
 RpuRmsNormSpmRoute small_image_rmsnorm_route(int64_t)const{return RpuRmsNormSpmRoute::BASE;}
 bool small_image_owner_ring_norm_enabled(int64_t)const{return false;}
 bool small_image_down_owner_norm_enabled(int64_t)const{return false;}
 bool small_image_ragged_sdpa_enabled(int64_t)const{return false;}
 void bind(){COLD}
 HELPERS
 auto produce(const FmbThreeStageChunkPlan& plan){
  std::vector<FmbRouteManifestEntry> routes;
  auto append=[&](FmbRouteFamily f,int64_t site,int64_t route,int64_t invocation,
                  std::vector<int64_t> args,int64_t flags){
   routes.push_back({site,f,route,flags,std::move(args),invocation});};
  for(const auto& chunk:plan.compute.chunks){PRODUCER}
  return routes;
 }
};
FmbThreeStageChunkPlan plan_for(int64_t rows){
 FmbThreeStageChunkPlan p;
 p.input=p.qkv=p.compute=make_stage({{0,0,rows,rows}});
 if(rows==672)p.spans={{0,336,0},{336,336,1}};else p.spans={{0,324,0}};
 return p;
}
Model exact(int64_t rows,bool fp16=false){
 Model m;for(auto& w:m.layer_weights_){
  for(auto* t:{&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w})
   t->dtype=fp16?at::kHalf:at::kChar;
  for(auto* t:{&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.gate_ws,&w.up_ws,&w.down_ws})t->present=!fp16;
 }
 m.wall_sorting_.fp16_transfer_mask=fp16?1:0;
 m.freq_cos_.shape=m.freq_sin_.shape={128,20};
 m.current_num_patches_=rows;
 if(rows==672){m.image_batch_count_=2;m.valid_patches_per_image_=324;m.schedule_=VisionSchedule::MISC;}
 m.context.stage_plan=plan_for(rows);m.bind();return m;
}
template<class F>void rejects(F&& f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}assert(failed);}
int main(){
 publish(true);
 for(bool fp16:{false,true})for(int64_t rows:{324,672}){
  Model m=exact(rows,fp16);assert(m.small_image_qk_rope_enabled(rows));
  assert(m.small_image_gate_up_enabled(rows)==!fp16); // W8 MLP stays separate from attention admission.
  if(fp16){
   // Every projection and every scale in every layer is part of the new proof.
   for(int layer=0;layer<32;++layer)for(int projection=0;projection<7;++projection){
    auto bad=m;auto& w=bad.layer_weights_[layer];
    at::Tensor* weights[]{&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w};
    at::Tensor* scales[]{&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.gate_ws,&w.up_ws,&w.down_ws};
    weights[projection]->dtype=at::kChar;assert(!bad.small_image_qk_rope_enabled(rows));
    weights[projection]->dtype=at::kHalf;weights[projection]->present=false;
    assert(!bad.small_image_qk_rope_enabled(rows));
    weights[projection]->present=true;scales[projection]->present=true;
    assert(!bad.small_image_qk_rope_enabled(rows));
   }
   auto row0=m;row0.wall_sorting_.vision_row_reuse=false;
   assert(!row0.small_image_qk_rope_enabled(rows));
   auto unpadded=m;unpadded.intermediate=3456;assert(!unpadded.small_image_qk_rope_enabled(rows));
   auto no_transfer=m;no_transfer.wall_sorting_.fp16_transfer_mask=6;
   assert(!no_transfer.small_image_qk_rope_enabled(rows));
  }
  const auto flags=m.small_image_schedule_flags();
  assert(m.small_image_qk_rope_arguments(rows,0)==std::vector<int64_t>(
     {1,rows,1280,16,2,80,80,8,128,0,(rows+127)/128,128,rows==672?2:1,324,rows==672?336:324}));
  m.validate_small_image_qk_rope_plan(m.context.stage_plan,rows);
  const auto routes=m.produce(m.context.stage_plan);assert(routes.size()==1);
  assert(routes[0].site_id==QWEN25_SMALL_QK_ROPE_SPM_SITE&&routes[0].selector==3);
  m.context.value.routes=routes;m.context.consumed_.resize(routes.size());
  for(int layer=0;layer<32;++layer){m.context.consumed_[0]=0;
   m.consume_small_image_qk_rope(rows,0,0);assert(m.context.consumed_[0]==1);}
  for(int variant=0;variant<24;++variant){auto bad=m;
   switch(variant){
    case 0:bad.small_qk_rope_available_=false;break;
    case 1:bad.rope_spm_enabled_=false;break;
    case 2:bad.max_hw_=127;break;
    case 3:bad.layers=31;break;
    case 4:bad.layer_weights_.pop_back();break;
    case 5:bad.layer_weights_[17].q_w.dtype=fp16?at::kChar:at::kHalf;break;
    case 6:bad.layer_weights_[31].down_w.dtype=fp16?at::kChar:at::kHalf;break;
    case 7:bad.wall_sorting_.vision_row_reuse=false;break;
    case 8:bad.wall_sorting_.silu_mul=false;break;
    case 9:bad.debug=true;break;
    case 10:bad.generic=false;break;
    case 11:bad.mixed_precision_=true;break;
    case 12:bad.per_window_sdpa_enabled_=true;break;
    case 13:bad.z1_bound_=true;break;
    case 14:bad.merger=false;break;
    case 15:bad.hidden=1024;break;
    case 16:bad.kvheads=8;break;
    case 17:bad.schedule_=VisionSchedule::MIMC;break;
    case 18:bad.current_num_patches_=320;break;
    case 19:bad.window_mask_present_=false;break;
    case 20:bad.layer_weights_[31].gate_ws.present=fp16;break;
    case 21:bad.has_rope_=false;break;
    case 22:bad.freq_cos_.shape[1]=40;break;
    case 23:bad.freq_sin_.shape[1]=40;break;
   }
   assert(!bad.small_image_qk_rope_enabled(rows));
   rejects([&]{bad.consume_small_image_qk_rope(rows,0,0);});
  }
  for(int64_t wrong:{0,256,320,336,648,768,972})assert(!m.small_image_qk_rope_enabled(wrong));
  for(int stage=0;stage<3;++stage)for(int mutation=0;mutation<8;++mutation){
   auto p=m.context.stage_plan;auto* s=stage==0?&p.input:stage==1?&p.qkv:&p.compute;
   switch(mutation){
    case 0:s->plan.chunk_size=rows==324?336:324;break;
    case 1:s->plan.num_chunks=2;break;case 2:s->chunks.clear();break;
    case 3:s->chunks[0].idx=1;break;case 4:s->chunks[0].offset=1;break;
    case 5:s->chunks[0].len--;break;case 6:s->chunks[0].kv_seq_len++;break;
    case 7:s->chunks.push_back(s->chunks[0]);break;
   }
   rejects([&]{m.validate_small_image_qk_rope_plan(p,rows);});
  }
  {auto p=m.context.stage_plan;p.chunk_mode=ChunkMode::KV_FIRST;
   rejects([&]{m.validate_small_image_qk_rope_plan(p,rows);});}
  for(int mutation=0;mutation<4;++mutation){auto p=m.context.stage_plan;
   switch(mutation){case 0:p.spans.clear();break;case 1:p.spans[0].offset=1;break;
    case 2:p.spans[0].len--;break;case 3:p.spans[0].group_id=3;break;}
   rejects([&]{m.validate_small_image_qk_rope_plan(p,rows);});
  }
  rejects([&]{m.consume_small_image_qk_rope(rows,1,0);});
  rejects([&]{m.consume_small_image_qk_rope(rows,0,1);});
  {auto bad=m;bad.context.ready=false;rejects([&]{bad.consume_small_image_qk_rope(rows,0,0);});}
  for(int mutation=0;mutation<6;++mutation){auto bad=m;auto& r=bad.context.value.routes[0];
   switch(mutation){case 0:r.family=FmbRouteFamily::ATTENTION;break;
    case 1:r.site_id++;break;case 2:r.selector=1;break;case 3:r.flags^=1;break;
    case 4:r.invocation=1;break;case 5:r.arguments.pop_back();break;}
   rejects([&]{bad.consume_small_image_qk_rope(rows,0,0);});
   assert(bad.context.consumed_[0]==1); // The prior successful receipt survives rejection.
  }
  for(size_t i=0;i<15;++i){auto bad=m;++bad.context.value.routes[0].arguments[i];
   rejects([&]{bad.consume_small_image_qk_rope(rows,0,0);});}
  publish(false);auto fallback=exact(rows,fp16);
  assert(!fallback.small_image_qk_rope_enabled(rows));
  const auto old_routes=fallback.produce(fallback.context.stage_plan);assert(old_routes.size()==2);
  assert(old_routes[0].site_id==QWEN25_Q_ROPE_SPM_SITE&&old_routes[1].site_id==QWEN25_K_ROPE_SPM_SITE);
  for(const auto& r:old_routes)assert(r.selector==1&&r.flags==FMB_ROUTE_FLAG_ROPE_TABLE_SPM&&r.arguments==std::vector<int64_t>{1});
  assert(fallback.small_image_schedule_flags()!=flags);
  assert(m.small_image_qk_rope_enabled(rows)); // Handle snapshot is cold.
  publish(true);assert(!fallback.small_image_qk_rope_enabled(rows));fallback.bind();
  assert(fallback.small_image_qk_rope_enabled(rows));
 }
 KernelCache::instance().availability.withdraw();
 rejects([&]{Model m;m.bind();});
}
'''
    source = _replace(source, {
        'KERNEL_IDS': block(cache, 'enum class KernelId : uint16_t') + ';',
        'AVAILABILITY': block(cache, 'class KernelPayloadAvailability final') + ';',
        'ROPE_ROUTES': block(vision, 'enum class Qwen25VisionRopeRoute') + ';',
        'SITE': re.search(r'constexpr int64_t QWEN25_SMALL_QK_ROPE_SPM_SITE\s*=.*?;', vision, re.S).group(),
        'LEGACY_SITES': '\n'.join(re.search(r'constexpr int64_t ' + name + r'\s*=.*?;', vision, re.S).group() for name in (
            'QWEN25_Q_ROPE_SPM_SITE', 'QWEN25_K_ROPE_SPM_SITE',
            'QWEN25_Q_ROPE_DDR_SITE', 'QWEN25_K_ROPE_DDR_SITE')),
        'LAYER_WEIGHTS': block(vision, '    struct LayerWeights {') + ';',
        'MAKE_STAGE': block(planner, 'ResolvedFmbStageChunks make_stage('),
        'CONSUMER': consumer,
        'HELPERS': helpers,
    })
    source = source.replace('COLD', cold).replace('PRODUCER', producer)
    compile_run(tmp_path, 'wall_vision_qk_rope_route', source)
