"""Execute the native chunked-merger manifest, layout and emitter without device.

Launch doubles record shapes/owners/ranges; they are not numerical evidence.
"""
from pathlib import Path
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition
from test_qwen3vl_vision_raw_spm import _program as raw_program


ROOT = Path(__file__).resolve().parents[1]


def _method(source, marker):
    start = source.index(marker)
    # Default Tensor/IntArrayRef arguments contain braces before the body.
    body = source.index(") {", start) + 2
    depth, end = 1, body + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def _program():
    source = (ROOT / "src/fused/rpu_qwen3vl_vision_model.cpp").read_text()
    planner = (ROOT / "src/core/fmb_three_stage_chunk_plan.cpp").read_text()
    program = raw_program().split("int main(){", 1)[0]
    consumer = "\n".join(_definition(planner, marker).replace("FmbPhysicalManifestConsumer::", "")
                         for marker in (
        "bool FmbPhysicalManifestConsumer::complete() const",
        "const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::find_route(",
        "const FmbRouteManifestEntry& FmbPhysicalManifestConsumer::consume_route("))
    support = r'''
namespace at {using IntArrayRef=std::vector<int64_t>;}
struct Context {
 FmbThreeStageChunkPlan stage_plan;
 FmbPhysicalExecutionManifest manifest_;mutable std::vector<uint8_t> consumed_;
 const FmbPhysicalExecutionManifest& manifest()const{return manifest_;}
 int64_t seq_len=1200,physical_route_invocation=91;bool input_in_spm=false,output_to_spm=false;
 __CONSUMER__
 bool has_complete_physical_manifest()const{return complete();}
 const FmbRouteManifestEntry& consume_physical_route(FmbRouteFamily f,int64_t s,int64_t r,int64_t flags,at::IntArrayRef a={},int64_t i=0)const{return consume_route(f,s,r,flags,a,i);}
 void bind(FmbPhysicalExecutionManifest m){manifest_=std::move(m);consumed_.assign(manifest_.routes.size(),0);}
};
struct Event {std::string kind;std::vector<int64_t> args;};
std::vector<Event> events;
void event(const char*k,std::initializer_list<int64_t>a){events.push_back({k,a});}
struct RpuKernelGraph {static bool has_active(){return true;}static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}void keep_alive(const Tensor&t){event("keep",{t.id});}};
uint64_t RpuGetDevAddr(void*p){return reinterpret_cast<uintptr_t>(p);}
void rpu_ddr_flush(void*){}
enum class ValuOpType{ADD};enum class GeluMode{ERF};enum class RpuUnaryPrecision{BASE,HIGH};
void rpu_launch_ddr_broadcast_spm_dma(c10::Half*,int64_t,uint32_t,int){}
void rpu_launch_ddr_broadcast_spm_dma(const Tensor&t,int64_t off,int64_t n,uint32_t a,int c){
 assert(off>=0&&n>0&&off+n<=t.elements);event("read",{t.id,off,n,a,c});}
void rpu_launch_ddr_broadcast_spm_dma_mutable(uint64_t&live,const Tensor&t,int64_t off,int64_t n,uint32_t a,int c){
 assert(live==RpuGetDevAddr(t.data_ptr<c10::Half>()));assert(off%2==0&&off/2+n<=t.elements);event("mutable",{t.id,off/2,n,a,c});}
void rpu_launch_spm_copy_ddr_dma(uint32_t a,const Tensor&t,int64_t off,int64_t n){
 assert(off>=0&&n>0&&off+n<=t.elements);event("write",{t.id,off,n,a});}
void rpu_launch_spm_copy_ddr_dma(uint32_t,c10::Half*,int64_t){}
template<class...A>void rpu_launch_ddr_scatter_spm_dma(A...){}
void rpu_launch_layernorm_spm_kernel(uint32_t in,uint32_t out,uint32_t,uint32_t,int64_t m,int64_t c,double,bool,uint32_t,int){event("norm",{m,c,in,out});}
void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t,const Tensor&w,uint32_t,int64_t m,int64_t n,int64_t k,int col,int,uint32_t,const Tensor&,uint32_t,int,bool acc32){event("linear",{m,n,k,col,w.id,w.scalar_type(),acc32});}
void rpu_launch_eltwise_unary_spm_kernel(uint32_t,uint32_t,int64_t n,ValuOpType,GeluMode mode,int,RpuUnaryPrecision){assert(mode==GeluMode::ERF);event("erf",{n});}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t,uint32_t,uint32_t,int64_t m,int64_t n,int,int){event("ring",{m,n});}
void rpu_launch_spm_local_shard_copy_kernel(uint32_t src,int64_t elems,int64_t stride,uint32_t dst,int cores){
 assert(elems==153600&&stride==307200&&cores==8);
 assert(uint64_t(src)+2457600<=dst||uint64_t(dst)+307200<=src);
 event("compact_copy",{src,elems,stride,dst,cores});}
void rpu_launch_all_reduce_sum_residual_local_spm_kernel(uint32_t in,uint32_t residual,uint32_t out,int64_t rows,int64_t cols,int producers,int consumers){
 assert(rows==1200&&cols==1024&&producers==8&&consumers==8);
 assert(uint64_t(in)+2457600<=out||uint64_t(out)+2457600<=in);
 for(auto full:{in,out})assert(uint64_t(full)+2457600<=residual||uint64_t(residual)+307200<=full);
 event("compact_ring",{in,residual,out,rows,cols,producers,consumers});}
'''.replace("__CONSUMER__", consumer)
    program = program.replace("class Qwen3VLVisionModel:public FusedModelBase {", support + "\nclass Qwen3VLVisionModel:public FusedModelBase {")
    methods = "\n".join(_method(source, marker) for marker in (
        "void consume_manifest_route(", "void emit_one_merger_prefix(",
        "void emit_one_merger(", "void emit_chunked_merger_post_fn(", "void merger_post_fn()",
        "void emit_fp16_compact_residual_copy(", "void emit_fp16_compact_attention_ring(",
        "void emit_fp16_compact_mlp_ring("))
    members = r'''
 Context context;Context& ctx(){return context;}
 Tensor raw{at::kHalf,1};const Tensor& output_tensor()const{return raw;}
 uint64_t chunked_merger_raw_src_base_=0;double merger_ln_eps_=1e-6;
 Tensor merger_ln_q_w_,merger_ln_q_b_,merger_m0_b_,merger_m2_b_;
 std::vector<Tensor> ds_merger_ln_q_w_{3},ds_merger_ln_q_b_{3},ds_merger_m0_b_{3},ds_merger_m2_b_{3};
 std::map<int64_t,Tensor> pooler_merged_ddr_slots_;
 std::map<std::pair<int64_t,int64_t>,Tensor> deepstack_ddr_slots_,deepstack_merged_ddr_slots_;
 std::map<std::string,uint32_t> offsets;
 uint32_t addr(int,const char*n)const{return offsets.at(n);}
 void zero_owner_spm(uint32_t,int64_t){}
 c10::Half* prefix_pooler_ptr_=nullptr;std::vector<c10::Half*>prefix_dense_ptrs_;
 void scatter_merger_out_to_prefix(int64_t,c10::Half*){assert(false);}
'''
    program = program.replace("};\nFmbThreeStageChunkPlan plan_for", members + methods + "\n};\nFmbThreeStageChunkPlan plan_for", 1)
    return program + r'''
void prepare(Qwen3VLVisionModel&m,int64_t n=1200,int64_t cs=608){
 m.current_num_patches_=n;m.context.seq_len=n;m.context.stage_plan=plan_for(n,cs);
 auto layout=layout_for(n,cs);layout.attention_policy=m.compact_encoder_active()?AttentionExecutionPolicy::SPM_KV_BY_MHA:AttentionExecutionPolicy::DDR_KV;
 auto domain=m.physical_manifest_domain_for_candidate(m.context.stage_plan,layout,n,n,0);
 auto selected=std::find_if(domain.begin(),domain.end(),[&](const auto&f){return fmb_rope_table_residency(f)==FmbRopeTableResidency::SPM&&std::all_of(f.routes.begin(),f.routes.end(),[&](const auto&r){return r.family!=FmbRouteFamily::ATTENTION||fmb_attention_execution_policy(r)==layout.attention_policy;});});
 assert(selected!=domain.end());auto manifest=*selected;
 validate_physical_manifest_impl(manifest,n,0);m.context.bind(manifest);
 m.raw.elements=n*1024;m.pooler_merged_ddr_slots_[n]={at::kHalf,10,n/4*m.merger_out_hidden_};
 for(int i=0;i<3;++i){auto key=std::make_pair(m.deepstack_visual_indexes_[i],n);m.deepstack_ddr_slots_[key]={at::kHalf,20+i,n*1024};m.deepstack_merged_ddr_slots_[key]={at::kHalf,30+i,n/4*m.merger_out_hidden_};}
 auto decls=m.declare_buffers(layout);std::map<std::string,int64_t>sizes;
 std::vector<SpmAllocator::AllocRequest>requests;std::vector<std::string>names;
 size_t persistent=0;
 for(auto&d:decls){sizes[d.name]=d.size;if(d.alias_of)continue;if(d.storage==StorageClass::Temp){requests.push_back({d.size,d.phase_start,d.phase_end,int(d.scope)});names.push_back(d.name);}else persistent+=d.size*(d.storage==StorageClass::PersistentPerLayer?d.per_layer:1);}
 auto plan=SpmAllocator::plan_temporary_aliased(requests);
 assert(plan.peak_bytes+persistent+392192<=8303616);
 for(size_t i=0;i<names.size();++i)m.offsets[names[i]]=0x40000000+plan.offsets[i];
 for(auto&d:decls){if(d.alias_of){assert(d.size<=sizes.at(d.alias_of));m.offsets[d.name]=m.offsets.at(d.alias_of);}else if(d.storage!=StorageClass::Temp)m.offsets[d.name]=0x41000000;}
 if(m.fp16_chunked_merger_active()){
  assert(persistent==356352 && plan.peak_bytes==(m.fp16_compact_encoder_active()?6761472:4672512));
  std::vector<BufferDecl>post;
  for(auto&d:decls)if(std::string(d.name).find("merger_")==0){assert(d.storage==StorageClass::Temp&&d.scope==BufferScope::OutsideLayerLoop&&!d.alias_of);post.push_back(d);}
  assert(post.size()==9);
  for(auto&a:post)for(auto&b:post)if(std::string(a.name)!=b.name)assert(m.offsets.at(a.name)+a.size<=m.offsets.at(b.name)||m.offsets.at(b.name)+b.size<=m.offsets.at(a.name));
  assert(manifest.linear_accumulation==FmbLinearAccumulationPolicy::ACC32);
  auto cfg=m.dynamic_config(m.context.stage_plan.compute.plan);
  if(m.fp16_compact_encoder_active())assert(cfg.chunk_mode==ChunkMode::SEQUENTIAL&&cfg.inter_layer_io==InterLayerIO::SPM_RESIDENT&&cfg.attention_policy==AttentionExecutionPolicy::SPM_KV_BY_MHA);
  else assert(cfg.chunk_mode==ChunkMode::KV_FIRST&&cfg.inter_layer_io==InterLayerIO::DDR_PINGPONG&&cfg.attention_policy==AttentionExecutionPolicy::DDR_KV);
 }else if(m.w8_compact_encoder_active()){
  assert(persistent==(m.merger_out_hidden_==2048?377856:378880)&&plan.peak_bytes==6761472);
  std::vector<BufferDecl>post;int parameters=0;
  for(auto&d:decls)if(std::string(d.name).find("merger_")==0){
   assert(!d.alias_of);
   if(d.storage==StorageClass::Temp){assert(d.scope==BufferScope::OutsideLayerLoop);post.push_back(d);}
   else{assert(d.storage==StorageClass::Persistent);++parameters;}
  }
  assert(post.size()==5&&parameters==4);
  for(auto&a:post)for(auto&b:post)if(std::string(a.name)!=b.name)assert(m.offsets.at(a.name)+a.size<=m.offsets.at(b.name)||m.offsets.at(b.name)+b.size<=m.offsets.at(a.name));
  assert(manifest.linear_accumulation==FmbLinearAccumulationPolicy::ACC16);
  auto cfg=m.dynamic_config(m.context.stage_plan.compute.plan);
  assert(cfg.chunk_mode==ChunkMode::SEQUENTIAL&&cfg.inter_layer_io==InterLayerIO::SPM_RESIDENT&&cfg.attention_policy==AttentionExecutionPolicy::SPM_KV_BY_MHA);
 }else if(cs<n){assert(m.offsets.at("merger_in")==m.offsets.at("input_norm"));assert(m.offsets.at("merger_zero")==m.offsets.at("input_norm"));assert(m.offsets.at("merger_normed")==m.offsets.at("oproj"));}
}
void check_emission(Qwen3VLVisionModel&m){
 events.clear();auto oldplan=m.context.stage_plan;const int64_t old_inv=m.context.physical_route_invocation;
 m.merger_post_fn();assert(m.current_num_patches_==m.context.seq_len);assert(m.context.physical_route_invocation==old_inv);
 assert(m.context.stage_plan.compute.chunks.size()==oldplan.compute.chunks.size());
 const auto chunks=m.merger_compute_chunks(oldplan);size_t reads=0,writes=0,norms=0,linears=0,rings=0,erfs=0;
 for(const auto&e:events){
  if(e.kind=="read"||e.kind=="mutable"){const size_t ord=reads/chunks.size();const auto&c=chunks[reads%chunks.size()];assert(e.kind==(ord==0?"mutable":"read"));assert(e.args[0]==(ord==0?m.raw.id:19+int64_t(ord)));assert(e.args[1]==c.offset*1024&&e.args[2]==c.len*1024);assert(e.args[3]==m.addr(0,"merger_in")&&e.args[4]==8);++reads;}
  if(e.kind=="write"){const size_t ord=writes/chunks.size();const auto&c=chunks[writes%chunks.size()];assert(e.args[0]==(ord==0?10:29+int64_t(ord)));assert(e.args[1]==c.offset/4*m.merger_out_hidden_&&e.args[2]==c.len/4*m.merger_out_hidden_);++writes;}
  if(e.kind=="norm"){size_t ord=norms/chunks.size();auto&c=chunks[norms%chunks.size()];assert(e.args[0]==(ord==0?c.len:c.len/4));assert(e.args[1]==(ord==0?1024:4096));++norms;}
  if(e.kind=="linear"){auto&c=chunks[(linears/2)%chunks.size()];assert(e.args[0]==c.len/4&&e.args[2]==4096);assert(e.args[1]==(linears%2?m.merger_out_hidden_:4096)&&e.args[3]==(linears%2?0:1));assert(e.args[5]==(m.fp16_chunked_merger_enabled_?at::kHalf:at::kChar)&&bool(e.args[6])==m.linear_acc32_);++linears;}
  if(e.kind=="ring"){assert(e.args[0]==chunks[rings%chunks.size()].len/4&&e.args[1]==m.merger_out_hidden_);++rings;}
  if(e.kind=="erf"){assert(e.args[0]==chunks[erfs%chunks.size()].len/4*512);++erfs;}
 }
 assert(reads==4*chunks.size()&&writes==reads&&norms==reads&&rings==reads&&erfs==reads&&linears==reads*2);
 for(size_t i=0;i<m.context.manifest_.routes.size();++i){auto&r=m.context.manifest_.routes[i];if(r.site_id==QWEN3VL_MERGER_M0_LINEAR_SITE||r.site_id==QWEN3VL_MERGER_M2_LINEAR_SITE||r.site_id==QWEN3VL_MERGER_ALL_REDUCE_SITE||r.site_id==QWEN3VL_CHUNKED_MERGER_INPUT_SITE||r.site_id==QWEN3VL_CHUNKED_MERGER_OUTPUT_SITE)assert(m.context.consumed_[i]);}
}
void fp16_weights(Qwen3VLVisionModel&m){
 m.linear_acc32_=true;m.merger_m0_w_.dtype=m.merger_m2_w_.dtype=at::kHalf;
 for(auto*weights:{&m.ds_merger_m0_w_,&m.ds_merger_m2_w_})for(auto&w:*weights)w.dtype=at::kHalf;
 for(auto&w:m.layer_weights_)w.q_w.dtype=w.k_w.dtype=w.v_w.dtype=w.o_w.dtype=w.fc1_w.dtype=w.fc2_w.dtype=at::kHalf;
}
template<class F>void rejects(F f){bool raised=false;try{f();}catch(const std::runtime_error&){raised=true;}assert(raised);}
void check_compact_encoder_and_post(){
 for(bool w8:{false,true})for(int64_t output:{2048,2560}){
  Qwen3VLVisionModel m;m.merger_out_hidden_=output;
  if(w8){w8_compact_weights(m);m.set_w8_compact_encoder(true);}
  else{fp16_weights(m);m.set_fp16_chunked_merger(true);m.set_fp16_compact_encoder(true);}
  prepare(m,1200,1200);
  assert(m.context.stage_plan.compute.chunks.size()==1);
  const auto post=m.merger_compute_chunks(m.context.stage_plan);
  assert(post.size()==2&&post[0].idx==0&&post[0].offset==0&&post[0].len==608);
  assert(post[1].idx==1&&post[1].offset==608&&post[1].len==592);
  assert(m.merger_compute_chunk_size(m.context.stage_plan)==608);
  const auto native_manifest=m.context.manifest_;
  events.clear();m.emit_fp16_compact_residual_copy(0);
  m.emit_fp16_compact_attention_ring();
  m.emit_fp16_compact_residual_copy(1);
  m.emit_fp16_compact_mlp_ring();
  assert(events.size()==4);
  for(size_t i=0;i<events.size();++i){
   auto&e=events[i];assert(e.kind==(i%2?"compact_ring":"compact_copy"));
   if(i%2)assert(e.args==std::vector<int64_t>({m.addr(0,"input_norm"),m.addr(0,"compact_residual"),m.addr(0,"residual1"),1200,1024,8,8}));
   else assert(e.args==std::vector<int64_t>({m.addr(0,"residual1"),153600,307200,m.addr(0,"compact_residual"),8}));
  }
  for(size_t i=0;i<native_manifest.routes.size();++i){auto&r=native_manifest.routes[i];
   assert(r.site_id!=QWEN3VL_ATTN_ALL_REDUCE_SITE&&r.site_id!=QWEN3VL_MLP_ALL_REDUCE_SITE);
   if(r.site_id==QWEN3VL_FP16_COMPACT_COPY_SITE||r.site_id==QWEN3VL_COMPACT_ATTN_ALL_REDUCE_SITE||r.site_id==QWEN3VL_COMPACT_MLP_ALL_REDUCE_SITE){
    assert(m.context.consumed_[i]);
    std::vector<int64_t> args{1,2,1200,1024,8,153600,307200};
    if(w8)args.insert(args.end(),{8,16});assert(r.arguments==args);
    if(r.family==FmbRouteFamily::ALL_REDUCE)assert(r.selector==int64_t(FmbSharedAllReduceRouteSelector::RING_PACED));
   }
  }
  check_emission(m); // Native post consumer covers all four outputs at 608/592.
  assert(m.context.stage_plan.compute.chunks.size()==1);
  const auto owner=m.chunked_merger_raw_src_base_;
  m.raw.id=2;check_emission(m);assert(m.chunked_merger_raw_src_base_!=owner);
  m.raw.id=1;check_emission(m);assert(m.chunked_merger_raw_src_base_==owner);
  for(auto site:{QWEN3VL_FP16_COMPACT_COPY_SITE,QWEN3VL_COMPACT_ATTN_ALL_REDUCE_SITE,QWEN3VL_COMPACT_MLP_ALL_REDUCE_SITE})for(int fault=0;fault<6;++fault){
   auto broken=native_manifest;auto it=std::find_if(broken.routes.begin(),broken.routes.end(),[&](auto&r){return r.site_id==site&&r.invocation==0;});assert(it!=broken.routes.end());
   if(fault==0)broken.routes.erase(it);if(fault==1)++it->selector;if(fault==2)++it->arguments.back();
   if(fault==3){if(w8)it->arguments.resize(it->arguments.size()-2);else ++it->invocation;} // FP16 owner identity is not W8.
   if(fault==4)it->site_id=site==QWEN3VL_FP16_COMPACT_COPY_SITE?5348861830992402021LL:site==QWEN3VL_COMPACT_ATTN_ALL_REDUCE_SITE?QWEN3VL_ATTN_ALL_REDUCE_SITE:QWEN3VL_MLP_ALL_REDUCE_SITE;
   if(fault==5)it->site_id=QWEN3VL_MERGER_ALL_REDUCE_SITE;
   m.context.bind(broken);events.clear();
   rejects([&]{if(site==QWEN3VL_FP16_COMPACT_COPY_SITE)m.emit_fp16_compact_residual_copy(0);else if(site==QWEN3VL_COMPACT_ATTN_ALL_REDUCE_SITE)m.emit_fp16_compact_attention_ring();else m.emit_fp16_compact_mlp_ring();});
   assert(events.empty());
  }
  m.context.bind(native_manifest);events.clear();
  rejects([&]{m.emit_fp16_compact_residual_copy(2);});
  m.image_batch_count_=2;
  rejects([&]{m.emit_fp16_compact_attention_ring();});
  rejects([&]{m.emit_fp16_compact_mlp_ring();});assert(events.empty());
  m.image_batch_count_=1;
  // Post DMA/Linear routes have their own eight invocations despite one
  // encoder chunk. Missing or stale tail descriptors fail at actual consume.
  for(auto site:{QWEN3VL_CHUNKED_MERGER_INPUT_SITE,QWEN3VL_MERGER_M0_LINEAR_SITE,QWEN3VL_CHUNKED_MERGER_OUTPUT_SITE}){
   auto broken=native_manifest;auto it=std::find_if(broken.routes.begin(),broken.routes.end(),[&](auto&r){return r.site_id==site&&r.invocation==7;});assert(it!=broken.routes.end());broken.routes.erase(it);
   m.context.bind(broken);events.clear();rejects([&]{m.merger_post_fn();});
  }
 }
}
void check_fp16_candidate(){
 for(int64_t out:{2048,2560}){
  Qwen3VLVisionModel m;m.merger_out_hidden_=out;fp16_weights(m);
  m.set_fp16_chunked_merger(true);assert(m.invalidations==1);
  prepare(m);check_emission(m);
  assert(m.context.stage_plan.compute.chunks[0].kv_seq_len==608&&m.context.stage_plan.compute.chunks[1].kv_seq_len==1200);
  auto wrong=m.context.stage_plan;wrong.compute.chunks[0].kv_seq_len=1200;
  rejects([&]{validate_fmb_three_stage_chunk_plan(wrong);});
  assert(!m.fp16_chunked_merger_plan(wrong,1200,1200,0));
  rejects([&]{m.physical_manifest_domain_for_candidate(wrong,layout_for(1200,608),1200,1200,0);});
  const auto base=m.chunked_merger_raw_src_base_;m.raw.id=2;check_emission(m);assert(m.chunked_merger_raw_src_base_!=base);
  m.raw.id=1;check_emission(m);assert(m.chunked_merger_raw_src_base_==base);
  assert(!m.ordinary_w8_resident_profile()&&!m.subclass_spm_kv_by_mha_eligible(m.context.stage_plan,layout_for(1200,608),0));
  assert(m.subclass_chunk_size_cap(1200,0)==608&&m.subclass_chunk_size_valid(608,1200,0));
  for(int cs:{144,304,592,624,1200})assert(!m.subclass_chunk_size_valid(cs,1200,0));
  assert(!m.subclass_chunk_size_valid(608,1200,1));
  // The public staged exact value survives the dry resolver's temporary
  // override reset. Reconfiguration must restore eager routes for other C.
  for(int exact:{144,256}){
   m.configured_chunk_size_=exact;assert(!m.merger_active());
   assert(m.subclass_chunk_size_cap(1200,0)==exact&&m.subclass_chunk_size_valid(exact,1200,0));
   auto c=layout_for(1200,exact);c.attention_policy=AttentionExecutionPolicy::DDR_KV;
   auto f=m.make_physical_manifest_for_candidate(plan_for(1200,exact),c,1200,1200,0,true);
   for(auto&r:f.routes)assert(r.site_id!=QWEN3VL_MERGER_M0_LINEAR_SITE);
   for(auto&d:m.declare_buffers(c))assert(std::string(d.name).find("merger_")!=0);
  }
  m.configured_chunk_size_=0;prepare(m);check_emission(m);
  m.configured_chunk_size_=608;assert(m.merger_active());
  m.configured_chunk_size_=0;
  m.last_resolved=608;rejects([&]{m.set_fp16_chunked_merger(false);});assert(m.fp16_chunked_merger_enabled_&&m.invalidations==1);
  for(int64_t n:{256,512,784,1024,1536}){
   m.current_num_patches_=n;assert(!m.merger_active());events.clear();m.merger_post_fn();assert(events.empty());
   auto c=layout_for(n,144);c.attention_policy=AttentionExecutionPolicy::DDR_KV;
   const auto candidate=m.declare_buffers(c);
   Qwen3VLVisionModel old;fp16_weights(old);old.fused_merger_enabled_=false;old.current_num_patches_=n;
   const auto baseline=old.declare_buffers(c);assert(candidate.size()==baseline.size());
   for(size_t i=0;i<candidate.size();++i){auto&a=candidate[i];auto&b=baseline[i];assert(std::string(a.name)==b.name&&a.size==b.size&&a.storage==b.storage&&a.scope==b.scope&&a.phase_start==b.phase_start&&a.phase_end==b.phase_end);}
   auto manifest=m.make_physical_manifest_for_candidate(plan_for(n,144),c,n,n,0,true);
   for(auto&r:manifest.routes)assert(r.site_id!=QWEN3VL_MERGER_M0_LINEAR_SITE&&r.site_id!=QWEN3VL_CHUNKED_MERGER_INPUT_SITE);
  }
  m.current_num_patches_=1200;m.image_batch_count_=2;assert(!m.merger_active());
 }
 for(int exact:{144,256}){Qwen3VLVisionModel cold;fp16_weights(cold);cold.current_num_patches_=1200;cold.configured_chunk_size_=exact;cold.set_fp16_chunked_merger(true);assert(!cold.merger_active()&&cold.subclass_chunk_size_valid(exact,1200,0));}
 // Candidate admission is cold and exact, including every installed matrix.
 for(int fault=0;fault<20;++fault){Qwen3VLVisionModel v;fp16_weights(v);
  switch(fault){case 0:v.linear_acc32_=false;break;case 1:v.cores=4;break;case 2:v.layers=25;break;case 3:v.hidden=1152;break;case 4:v.intermediate=4304;break;case 5:v.heads=12;break;case 6:v.hd=80;break;case 7:v.orig_head_dim_=72;break;case 8:v.merger_out_hidden_=4096;break;case 9:v.layer_weights_.back().fc2_w.dtype=at::kChar;break;case 10:v.merger_m2_w_.dtype=at::kChar;break;case 11:v.ds_merger_m0_w_[2].dtype=at::kChar;break;case 12:v.ds_merger_m2_w_.pop_back();break;case 13:v.deepstack_visual_indexes_[2]=18;break;case 14:v.pipeline=true;break;case 15:v.delivery_compatibility_=true;break;case 16:v.prefix_scatter_active_=true;break;case 17:v.multiview_composite_prepared_=true;break;case 18:v.last_resolved=608;break;case 19:v.image_batch_count_=2;break;}
  rejects([&]{v.set_fp16_chunked_merger(true);});assert(!v.fp16_chunked_merger_enabled_&&v.invalidations==0);
 }
 for(int fault=0;fault<8;++fault){Qwen3VLVisionModel v;fp16_weights(v);v.set_fp16_chunked_merger(true);prepare(v);
  auto&p=v.context.stage_plan;auto layout=layout_for(1200,608);layout.attention_policy=AttentionExecutionPolicy::DDR_KV;
  if(fault==0)p.compute.chunks[1].len=588;if(fault==1)p.qkv.chunks[1].offset=604;if(fault==2)p.compute.plan.chunk_size=624;if(fault==3)p.chunk_mode=ChunkMode::SEQUENTIAL;
  if(fault==4)p.compute.chunks[1].kv_seq_len=1216;if(fault==5)layout.is_causal=true;if(fault==6)layout.use_attn_mask=true;if(fault==7)layout.batch_size=2;
  rejects([&]{v.make_physical_manifest_for_candidate(p,layout,1200,1200,0,true);});
  if(fault<5){events.clear();rejects([&]{v.merger_post_fn();});assert(events.empty());}
 }
 // Registered legacy FP16 fusion still executes its original single-chunk path.
 Qwen3VLVisionModel legacy;fp16_weights(legacy);prepare(legacy,512,512);events.clear();legacy.merger_post_fn();
 assert(legacy.merger_active()&&legacy.chunked_merger_raw_src_base_==0);
 for(auto&e:events){assert(e.kind!="mutable");if(e.kind=="linear")assert(e.args[5]==at::kHalf&&e.args[6]==1);}
}
int main(){
 check_compact_encoder_and_post();
 check_fp16_candidate();
 Qwen3VLVisionModel two;two.merger_out_hidden_=2048;prepare(two);check_emission(two);
 Qwen3VLVisionModel m;prepare(m);check_emission(m);
 const auto base=m.chunked_merger_raw_src_base_;m.raw.id=2;check_emission(m);assert(m.chunked_merger_raw_src_base_!=base);m.raw.id=1;check_emission(m);assert(m.chunked_merger_raw_src_base_==base);
 Qwen3VLVisionModel aligned;prepare(aligned,768,256);check_emission(aligned);
 for(int fault=0;fault<7;++fault){Qwen3VLVisionModel v;prepare(v);auto& chunks=v.context.stage_plan.compute.chunks;
  if(fault==0)chunks[1].offset+=4;if(fault==1)chunks[1].len-=4;if(fault==2)chunks[1].len-=1;if(fault==3)chunks[1].idx=0;if(fault==4)chunks[1].len=612;if(fault==5)chunks[0].len=0;if(fault==6)chunks[0].offset=-4;
  events.clear();bool rejected=false;try{v.emit_chunked_merger_post_fn();}catch(const std::runtime_error&){rejected=true;}assert(rejected&&events.empty());
  auto layout=layout_for(1200,608);layout.attention_policy=AttentionExecutionPolicy::DDR_KV;rejected=false;try{v.make_physical_manifest_for_candidate(v.context.stage_plan,layout,1200,1200,0,true);}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 }
 for(auto site:{QWEN3VL_CHUNKED_MERGER_INPUT_SITE,QWEN3VL_CHUNKED_MERGER_OUTPUT_SITE,QWEN3VL_MERGER_M0_LINEAR_SITE,QWEN3VL_MERGER_M2_LINEAR_SITE,QWEN3VL_MERGER_ALL_REDUCE_SITE})for(int fault=0;fault<3;++fault){
  Qwen3VLVisionModel v;prepare(v);auto& routes=v.context.manifest_.routes;auto it=std::find_if(routes.begin(),routes.end(),[&](auto&r){return r.site_id==site&&r.invocation==7;});assert(it!=routes.end());
  if(fault==0)routes.erase(it);if(fault==1)++it->selector;if(fault==2)it->arguments.push_back(123);
  bool rejected=false;try{v.emit_chunked_merger_post_fn();}catch(const std::runtime_error&){rejected=true;}assert(rejected);
 }
 for(int fault=0;fault<5;++fault){Qwen3VLVisionModel v;prepare(v);if(fault==0)v.linear_acc32_=true;if(fault==1)v.merger_out_hidden_=4096;if(fault==2)v.layer_weights_[23].fc2_w.dtype=at::kHalf;if(fault==3)v.image_batch_count_=2;if(fault==4)v.prefix_scatter_active_=true;events.clear();v.merger_post_fn();assert(events.empty());}
 Qwen3VLVisionModel single;prepare(single,512,512);events.clear();single.merger_post_fn();assert(single.chunked_merger_raw_src_base_==0);
 for(auto&e:events)assert(e.kind!="mutable");
}
'''


def test_native_chunked_merger_rows_dma_routes_and_layout(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    cpp, exe = tmp_path / "chunked.cpp", tmp_path / "chunked"
    cpp.write_text(_program())
    built = subprocess.run([compiler, "-std=c++17", "-O0", str(cpp), "-o", str(exe)], capture_output=True, text=True)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(exe)], capture_output=True, text=True)
    assert ran.returncode == 0, ran.stdout + ran.stderr
