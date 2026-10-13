"""Actual Text cross-layer owner RMS schedule, cold owners and caller/abort behavior."""
from pathlib import Path
import re
from test_g05_q_gate_native import _compile_run
from test_g05_vision_owner_model import _block
ROOT=Path(__file__).resolve().parents[1]
SITE=8475656447499929675


def test_text_mlp_tail_cold_complete_and_same_group_callers(tmp_path,monkeypatch):
    import test_g05_text_owner_rms as base
    captured=[]
    monkeypatch.setattr(base,'_compile_run',lambda p,n,cpp:captured.append(cpp))
    base.test_text_owner_rms_cold_complete_and_layer_local_skip(tmp_path)
    cpp=captured[0].split('int main(){',1)[0]
    source=(ROOT/'src/fused/rpu_qwen3_5_model.cpp').read_text()
    header=(ROOT/'src/fused/rpu_qwen3_5_model.h').read_text()
    fmb=(ROOT/'src/core/fused_model_base.cpp').read_text()
    traversal=fmb[fmb.index('struct FmbExecutionTraversalSpec {'):fmb.index('\nint64_t validate_fmb_resolved_chunks')]
    # Existing fake metadata is shared; actual admission/caller bodies are below.
    cpp=cpp.replace('namespace at{constexpr int','namespace at{using ScalarType=int;constexpr int',1)
    cpp=cpp.replace('namespace c10{enum class','namespace c10{struct Half{uint16_t x;explicit Half(double v):x(v==1e-6?17:0){}};enum class',1)
    cost_loop='for (const auto& weights : layer_weights_)'+_block(header,'for (const auto& weights : layer_weights_)')
    fields=sorted(set(re.findall(r'&weights\.(\w+)',cost_loop)))
    cpp=re.sub(r'struct Weights\{.*?\};','struct Weights{at::Tensor '+','.join(fields)+';};',cpp,count=1)
    cpp=cpp.replace('enum class FmbRouteFamily{ALL_REDUCE,MUTABLE_DMA}','enum class FmbRouteFamily{ALL_REDUCE,MUTABLE_DMA,GRAPH_SCHEDULE,OTHER}')
    cpp=cpp.replace('struct FmbThreeStageChunkPlan{Stage qkv,compute;};','enum class ChunkMode{SEQUENTIAL,KV_FIRST};\nstruct Span{int64_t offset=0,len=256,group_id=-1;};\nstruct FmbThreeStageChunkPlan{Stage input,qkv,compute;std::vector<Span>spans{{}};ChunkMode chunk_mode=ChunkMode::SEQUENTIAL;};')
    cpp=cpp.replace('struct Model{',f'constexpr int64_t QWEN35_TEXT_MLP_TAIL_SITE={SITE}LL;\nstruct Model{{',1)
    # Remove only the disabled Text-tail facades supplied by the old fixture.
    cpp=re.sub(r'^.*bool text_mlp_tail_available_=false;.*\n','',cpp,flags=re.M)
    cpp=re.sub(r'^.*void prepare_text_mlp_tail\(\)\{\}.*\n','',cpp,flags=re.M)
    cpp=re.sub(r'^.*template<class\.\.\.T>.*(?:use_text_mlp_tail|text_mlp_tail_handoff|launch_text_mlp_tail).*\n','',cpp,flags=re.M)
    cpp=re.sub(r'^.*std::vector<int64_t>text_mlp_tail_arguments\(int64_t=245\)const\{return \{\};\}.*\n','',cpp,flags=re.M)
    methods=[
        ('void prepare_text_mlp_tail()','void Qwen3_5Model::prepare_text_mlp_tail('),
        ('bool use_text_mlp_tail(const FmbThreeStageChunkPlan&plan,int64_t physical_len,int64_t logical_len,int64_t position,int64_t batch)const','bool Qwen3_5Model::use_text_mlp_tail('),
        ('bool text_mlp_tail_handoff(int layer,const ChunkInfo&chunk,bool producer)','bool Qwen3_5Model::text_mlp_tail_handoff('),
        ('std::vector<int64_t>text_mlp_tail_arguments(int64_t logical_len=245)const','std::vector<int64_t> Qwen3_5Model::text_mlp_tail_arguments('),
        ('void launch_text_mlp_tail(int layer,const ChunkInfo&chunk)','void Qwen3_5Model::launch_text_mlp_tail(')]
    cpp=cpp.replace('struct Model{','struct Model{\n bool text_mlp_tail_available_=false;bool input_in_spm=false,output_to_spm=false;int body_iter=0;\n'+'\n'.join(d+_block(source,m)for d,m in methods),1)
    begin=cpp.index(' Model(){');end=cpp.index('\n int64_t num_layers()',begin)
    ctor=' Model(){int64_t id=1;for(int layer=0;layer<24;++layer){layer_is_full_[layer]=(layer+1)%4==0;auto&w=layer_weights_[layer];'
    ctor+='for(auto*t:{'+','.join('&w.'+f for f in fields)+'}){t->owner=id++;}'
    ctor+='''
      w.gate_w.rows=w.up_w.rows=6144;w.down_w.cols=6144;
      for(auto*t:{&w.input_norm_w,&w.post_norm_w,&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.attn_gate_ws,&w.gate_ws,&w.up_ws,&w.down_ws,&w.gdn_q_ws,&w.gdn_k_ws,&w.gdn_v_ws,&w.gdn_in_z_ws,&w.gdn_out_ws}){t->dtype=at::kHalf;t->rank=1;t->count=2048;}
      w.gate_ws.count=w.up_ws.count=6144;
      for(auto*t:{&w.gdn_b_bg_w,&w.gdn_a_bg_w}){t->dtype=at::kHalf;t->rows=128;}
      w.gdn_b_bg_ws.present=w.gdn_a_bg_ws.present=false;
     }}
'''
    cpp=cpp[:begin]+ctor+cpp[end:]
    # Execute existing full/GDN mixer-tail call sites as well as new tail ABI.
    start=cpp.index('void rpu_launch_g05_text_owner_rms_m256h2048_kernel(');end=cpp.index('\nvoid rpu_launch_all_reduce_sum_residual_kernel',start)
    cpp=cpp[:start]+'''int expected_layer=0;
void rpu_launch_g05_text_owner_rms_m256h2048_kernel(uint32_t x,uint32_t r,uint32_t raw,uint32_t g){
 if(x==0x310000){assert(r==0x210000&&raw==0x110000&&g==uint32_t(0x400000+(expected_layer+1)*4096));if(fail_producer)throw std::runtime_error("tail producer failed");launches.push_back("tail_owner");}
 else {assert((x==0x10000||x==0x510000)&&r==0x110000&&raw==0x210000&&g>=0x400000);launches.push_back(x==0x10000?"owner":"owner_gdn");}}
'''+cpp[end:]
    start=cpp.index('void rpu_launch_rmsnorm_spm_kernel(');end=cpp.index('\nvoid append_kvinsert_cost_tensor_identity',start)
    cpp=cpp[:start]+'''void rpu_launch_rmsnorm_spm_kernel(uint32_t x,uint32_t y,uint32_t g,int64_t m,int64_t n,double eps,RpuRmsNormSpmRoute,int cores){assert(m==256&&n==2048&&eps==1e-6&&cores==8);if(x==0x110000&&y==0x210000){assert(g==uint32_t(0x400000+expected_layer*4096));launches.push_back("input_norm");}else if(x==0x110000&&y==x){assert(g==0x610000);launches.push_back("final_norm");}else{assert(x==0x210000&&y==0x110000);launches.push_back("norm");}}
'''+cpp[end:]
    cpp=cpp.replace('return s=="gdn_c_m"?', 'return s=="final_norm_w"?0x610000:s=="gdn_c_m"?',1)
    # Plan retains original ring authority and appends the actual schedule rule.
    marker='if (use_text_mlp_tail(plan,physical_len,logical_len,position,layout.batch_size))'
    rule=marker+_block(source,marker)
    at=cpp.index(' int64_t layout_hash(){')
    before=cpp[:at];close=before.rfind('}')
    cpp=before[:close]+rule+before[close:]+cpp[at:]
    begin=cpp.index(' int64_t layout_hash(){');end=cpp.index('\n std::vector<int64_t>identity()',begin)
    cpp=cpp[:begin]+' int64_t layout_hash(){int64_t hash=0;if(!action_mode_&&text_mlp_tail_available_)'+_block(source,'if (!action_mode_ && text_mlp_tail_available_)')+'return hash;}'+cpp[end:]
    begin=cpp.index(' std::vector<int64_t>identity(){');end=cpp.index('\n void build_full_attention(',begin)
    cpp=cpp[:begin]+' std::vector<int64_t>identity(){std::vector<int64_t>identity;if(!action_mode_&&text_mlp_tail_available_)'+_block(header,'if (!action_mode_ && text_mlp_tail_available_)')+cost_loop+'return identity;}'+cpp[end:]
    layer=_block(source,'void Qwen3_5Model::build_layer_subgraph(')
    entry=layer[layer.index('    if (!ctx().input_in_spm && !z2_layer0)'):layer.index('    // This layer-local')]
    extra=''' void emit_layer_input_dma(int,const ChunkInfo&){launches.push_back("input_dma");}
 void input_entry(int layer_idx,const ChunkInfo&chunk={}){const int64_t h=hidden_size();const bool z2_layer0=z2_bound_&&layer_idx==0;const uint32_t input_residual=layer_input_residual_addr(layer_idx,chunk);'''+entry+'}\n'
    begin=cpp.index(' void layer(int layer_idx,');cpp=cpp[:begin]+extra+cpp[begin:]
    tail=_block(source,'void Qwen3_5Model::emit_mlp_and_output(')
    final='if (is_last_layer)'+_block(tail,'if (is_last_layer)')
    # Existing fixture exercises the actual postnorm/tail dispatch. Extend it
    # with actual last-finalnorm logic; expensive GEMMs are the board probe.
    begin=cpp.index(' void emit_mlp_and_output(');end=cpp.index('\n void emit_layer_input_dma',begin)
    section=cpp[begin:end];section=section[:section.rfind('}')]+final+section[section.rfind('}'):]
    section=section.replace('int64_t h=hidden_size();','int64_t h=hidden_size(),seq_len=chunk.len;const bool is_last_layer=layer_idx==23;',1)
    cpp=cpp[:begin]+section+cpp[end:]
    cpp+=r'''
#define TORCH_INTERNAL_ASSERT(x) assert(x)
enum class InterLayerIO{SPM_RESIDENT,DDR_PINGPONG};enum class SpmFmbOccurrenceKind{LayerBody,KvInsertBody};enum class BufferScope{LayerWide,Compute,KvInsert};
using SpmFmbScopeMask=uint32_t;constexpr uint32_t spm_fmb_scope_bit(BufferScope x){return 1u<<int(x);}
namespace detail{bool fmb_kv_first_pair_carry_eligible(const std::vector<ChunkInfo>&,const std::vector<ChunkInfo>&,int64_t){return false;}}
'''+traversal+r'''
int main(){
 Model m;m.installed();assert(m.text_mlp_tail_available_&&m.text_owner_rms_available_&&m.text_gdn_owner_rms_available_);m.plan();const auto hash=m.layout_hash();const auto cost=m.identity();const int queries=KernelCache::instance().calls;
 assert(m.use_text_mlp_tail(m.stage_plan,256,245,0,1));for(auto v:m.text_mlp_tail_arguments())assert(v>=0);
 auto run=[&](Model&x,int group,InterLayerIO io,int fail){
  FmbExecutionTraversalSpec spec;spec.num_layers=24;spec.cross_batch=group;spec.chunks=&x.stage_plan.compute.chunks;std::vector<ChunkInfo>none;spec.kv_insert_chunks=&none;spec.inter_layer_io=io;
  int last=-1,producers=0,consumers=0,inputs=0,visited=0;bool aborted=false;
  try{for_each_fmb_execution_step(spec,0,[&](const auto&step){expected_layer=step.layer;++visited;x.input_in_spm=step.input_in_spm;x.output_to_spm=step.output_to_spm;
   bool skip=x.text_mlp_tail_handoff(step.layer,*step.chunk,false);if(skip){assert(last==step.layer-1);++consumers;}
   launches.clear();x.input_entry(step.layer,*step.chunk);const int norms=std::count(launches.begin(),launches.end(),"input_norm");assert(norms==!skip);inputs+=norms;
   fail_producer=step.layer==fail;x.layer(step.layer,*step.chunk);fail_producer=false;
   if(x.text_mlp_tail_handoff(step.layer,*step.chunk,true)){assert(launches.back()=="tail_owner");last=step.layer;++producers;}else{last=-1;assert(std::find(launches.begin(),launches.end(),"ring_mlp")!=launches.end());}
   assert(std::count(launches.begin(),launches.end(),"final_norm")==int(step.layer==23));
  });}catch(const std::runtime_error&){assert(fail>=0);aborted=true;fail_producer=false;}
  if(fail<0){int expected=io==InterLayerIO::SPM_RESIDENT&&x.text_mlp_tail_handoff(1,{},false)?24-(24+group-1)/group:0;assert(producers==expected&&consumers==expected&&inputs==24-expected&&visited==24);}
  else {assert(aborted&&visited==fail+1);}return producers;
 };
 assert(run(m,24,InterLayerIO::SPM_RESIDENT,-1)==23);
 run(m,24,InterLayerIO::DDR_PINGPONG,-1);run(m,1,InterLayerIO::SPM_RESIDENT,-1);
 for(int fail:{0,7,22}){run(m,24,InterLayerIO::SPM_RESIDENT,fail);assert(run(m,24,InterLayerIO::SPM_RESIDENT,-1)==23);}
 assert(KernelCache::instance().calls==queries);
 // Whole cold chain: input gamma faults alone disable only this new schedule.
 for(int layer=0;layer<24;++layer)for(int fault=0;fault<8;++fault){auto x=m;auto&t=x.layer_weights_[layer].input_norm_w;
  if(fault==0){t.present=false;}if(fault==1){t.dtype=at::kChar;}if(fault==2){t.contiguous=false;}if(fault==3){t.offset=128;}if(fault==4){t.rank=2;}if(fault==5){--t.count;}if(fault==6){t.location=c10::DeviceType::CPU;}if(fault==7){++t.index;}
  x.installed();assert(!x.text_mlp_tail_available_&&x.text_owner_rms_available_&&x.text_gdn_owner_rms_available_);
 }
 for(int layer=0;layer<24;++layer){auto&orig=m.layer_weights_[layer];(void)orig;
  for(int family=0;family<8;++family)for(int fault=0;fault<16;++fault){auto x=m;auto&w=x.layer_weights_[layer];bool full=x.layer_is_full_[layer];
   std::array<at::Tensor*,8>ws{&w.gate_w,&w.up_w,&w.down_w,full?&w.q_w:&w.gdn_q_w,full?&w.k_w:&w.gdn_k_w,full?&w.v_w:&w.gdn_v_w,full?&w.o_w:&w.gdn_out_w,full?&w.attn_gate_w:&w.gdn_in_z_w};
   std::array<at::Tensor*,8>ss{&w.gate_ws,&w.up_ws,&w.down_ws,full?&w.q_ws:&w.gdn_q_ws,full?&w.k_ws:&w.gdn_k_ws,full?&w.v_ws:&w.gdn_v_ws,full?&w.o_ws:&w.gdn_out_ws,full?&w.attn_gate_ws:&w.gdn_in_z_ws};
   auto&t=*ws[family];auto&s=*ss[family];if(fault==0){t.present=false;}if(fault==1){t.dtype=at::kHalf;}if(fault==2){t.contiguous=false;}if(fault==3){t.offset=256;}if(fault==4){t.rank=1;}if(fault==5){--t.rows;}if(fault==6){--t.cols;}if(fault==7){t.location=c10::DeviceType::CPU;}
   if(fault==8){s.present=false;}if(fault==9){s.dtype=at::kChar;}if(fault==10){s.contiguous=false;}if(fault==11){s.offset=128;}if(fault==12){s.rank=2;}if(fault==13){--s.count;}if(fault==14){++s.index;}if(fault==15){s.location=c10::DeviceType::CPU;}
   x.installed();assert(!x.text_mlp_tail_available_);
  }
  if(!m.layer_is_full_[layer])for(int f=0;f<4;++f){auto x=m;auto&w=x.layer_weights_[layer];if(f==0){w.gdn_b_bg_w.dtype=at::kChar;}if(f==1){--w.gdn_a_bg_w.rows;}if(f==2){w.gdn_b_bg_ws.present=true;w.gdn_b_bg_ws.count=128;}if(f==3){++w.gdn_a_bg_w.index;}x.installed();assert(!x.text_mlp_tail_available_);}
 }
 m.body_iter=0;m.input_in_spm=m.output_to_spm=true;expected_layer=0;
 auto route=std::find_if(m.manifest.routes.begin(),m.manifest.routes.end(),[](const auto&r){return r.site==QWEN35_TEXT_MLP_TAIL_SITE;});assert(route!=m.manifest.routes.end());const size_t slot=route-m.manifest.routes.begin();
 for(size_t a=0;a<route->args.size();++a){auto x=m;++x.manifest.routes[slot].args[a];launches.clear();reject([&]{x.launch_text_mlp_tail(0,{});});assert(launches.empty());expected_layer=1;reject([&]{x.input_entry(1);});assert(launches.empty());expected_layer=0;}
 for(int f=0;f<7;++f){auto x=m;auto&r=x.manifest.routes[slot];if(f==0){++r.site;}if(f==1){++r.selector;}if(f==2){++r.flags;}if(f==3){++r.invocation;}if(f==4){r.family=FmbRouteFamily::OTHER;}if(f==5){x.manifest.routes.erase(x.manifest.routes.begin()+slot);}if(f==6){x.complete=false;}reject([&]{x.launch_text_mlp_tail(0,{});});if(f!=6){expected_layer=1;reject([&]{x.input_entry(1);});expected_layer=0;}}
 // Original MLP all-reduce authority is independently required.
 {auto x=m;auto i=std::find_if(x.manifest.routes.begin(),x.manifest.routes.end(),[](const auto&r){return r.site==QWEN35_TEXT_MLP_ALL_REDUCE_SITE;});x.manifest.routes.erase(i);reject([&]{x.launch_text_mlp_tail(0,{});});}
 for(int f=0;f<17;++f){auto x=m;if(f==0){x.debug=true;}if(f==1){x.z2_bound_=true;}if(f==2){x.retained_prefill_graph_=false;}if(f==3){x.manifest.logical_length=244;}if(f==4){x.seq_len=128;}if(f==5){x.position=1;}if(f==6){x.batch_size=2;}if(f==7){x.stage_plan.input.chunks.clear();}if(f==8){x.stage_plan.input.chunks[0].offset=1;}if(f==9){x.stage_plan.spans.clear();}if(f==10){x.stage_plan.spans[0].len=128;}if(f==11){x.stage_plan.spans[0].group_id=1;}if(f==12){x.stage_plan.chunk_mode=ChunkMode::KV_FIRST;}if(f==13)x.stage_plan.qkv.chunks.push_back({});if(f==14){x.stage_plan.compute.chunks[0].idx=1;}if(f==15)continue;if(f==16){x.action_mode_=true;}assert(!x.use_text_mlp_tail(x.stage_plan,x.seq_len,x.manifest.logical_length,x.position,x.batch_size));reject([&]{x.launch_text_mlp_tail(0,{});});}
 for(int layer:{-1,23,24})reject([&]{m.launch_text_mlp_tail(layer,{});});
 for(int layer:{-1,0,24})assert(!m.text_mlp_tail_handoff(layer,{},false));
 for(int f=0;f<5;++f){auto x=m;if(f==0){x.body_iter=1;}if(f==1){x.output_to_spm=false;}ChunkInfo c;if(f==2){c.idx=1;}if(f==3){c.offset=1;}if(f==4){c.len=128;}reject([&]{x.launch_text_mlp_tail(0,c);});}
 // Changing any carried norm owner changes actual layout and existing cost identity.
 for(int layer=0;layer<24;++layer){auto x=m;++x.layer_weights_[layer].input_norm_w.owner;assert(x.layout_hash()!=hash&&x.identity()!=cost);}
 auto&cache=KernelCache::instance();cache.available=false;m.installed();m.plan();assert(!m.text_mlp_tail_available_&&m.layout_hash()!=hash&&m.identity()!=cost);run(m,24,InterLayerIO::SPM_RESIDENT,-1);
 cache.available=true;m.installed();m.plan();assert(m.text_mlp_tail_available_&&m.layout_hash()==hash&&m.identity()==cost);
 {auto x=m;x.set_linear_acc32(true);assert(x.text_mlp_tail_available_&&x.text_mlp_tail_arguments()[5]==32&&x.layout_hash()!=hash&&x.identity()!=cost);launches.clear();reject([&]{x.launch_text_mlp_tail(0,{});});assert(launches.empty());x.plan();assert(run(x,24,InterLayerIO::SPM_RESIDENT,-1)==23);for(int fail:{0,7,22}){run(x,24,InterLayerIO::SPM_RESIDENT,fail);assert(run(x,24,InterLayerIO::SPM_RESIDENT,-1)==23);}run(x,24,InterLayerIO::DDR_PINGPONG,-1);run(x,1,InterLayerIO::SPM_RESIDENT,-1);}
 m.set_linear_acc32(true);assert(m.text_mlp_tail_available_);m.set_linear_acc32(false);assert(m.text_mlp_tail_available_);
 m.set_fast_replay(false);assert(!m.text_mlp_tail_available_);m.set_fast_replay(true);assert(m.text_mlp_tail_available_);
}
'''
    _compile_run(tmp_path,'text_mlp_tail_model',cpp)
