"""Actual stateless Action MLP tail helpers/callers with the current FMB traversal."""
from pathlib import Path
from test_g05_q_gate_native import _compile_run
from test_g05_vision_owner_model import _block
ROOT=Path(__file__).resolve().parents[1]


def test_action_mlp_tail_actual_same_group_cold_complete_and_abort(tmp_path,monkeypatch):
    import test_g05_action_tail as base
    captured=[]
    monkeypatch.setattr(base,'_compile_run',lambda p,n,cpp:captured.append(cpp))
    base.test_action_tail_cold_complete_local_flag_and_fallback(tmp_path)
    cpp=captured[0].split('int main(){',1)[0]
    source=(ROOT/'src/fused/rpu_qwen3_5_model.cpp').read_text()
    header=(ROOT/'src/fused/rpu_qwen3_5_model.h').read_text()
    fmb=(ROOT/'src/core/fused_model_base.cpp').read_text()
    start=fmb.index('struct FmbExecutionTraversalSpec {');end=fmb.index('\nint64_t validate_fmb_resolved_chunks',start)
    traversal=fmb[start:end]
    cpp=cpp.replace('#include <algorithm>','#include <algorithm>\n#include <tuple>')
    cpp=cpp.replace('struct Route{','struct FmbThreeStageChunkPlan{struct Stage{std::vector<ChunkInfo>chunks{ChunkInfo{}};};Stage qkv,compute;};\nstruct Route{',1)
    methods=[
      ('void prepare_action_mlp_tail()','void Qwen3_5Model::prepare_action_mlp_tail('),
      ('bool use_action_mlp_tail(const FmbThreeStageChunkPlan&plan,int64_t rows)const','bool Qwen3_5Model::use_action_mlp_tail('),
      ('bool action_mlp_tail_handoff(int layer,const ChunkInfo&chunk,bool producer)','bool Qwen3_5Model::action_mlp_tail_handoff('),
      ('std::vector<int64_t>action_mlp_tail_arguments()const','std::vector<int64_t> Qwen3_5Model::action_mlp_tail_arguments('),
      ('void launch_action_mlp_tail(int layer,const ChunkInfo&chunk)','void Qwen3_5Model::launch_action_mlp_tail(')]
    begin=cpp.index(' void prepare_action_mlp_tail(){}');end=cpp.index('\n',cpp.index(' template<class...T>void launch_action_mlp_tail',begin))
    cpp=cpp[:begin]+'\n'.join(d+_block(source,m)for d,m in methods)+cpp[end:]
    cpp=cpp.replace('int stage_plan=0;','FmbThreeStageChunkPlan stage_plan;bool input_in_spm=false,output_to_spm=false;bool bad_alias=false;')
    cpp=cpp.replace('void install(){prepare_action_tail();}void reset(){action_tail_available_=false;}','void install(){prepare_action_tail();prepare_action_mlp_tail();}void reset(){action_tail_available_=false;action_mlp_tail_available_=false;}')
    cpp=cpp.replace('QWEN35_TEXT_FULL_ALL_REDUCE_SITE=1,','QWEN35_TEXT_MLP_ALL_REDUCE_SITE=3,QWEN35_TEXT_FULL_ALL_REDUCE_SITE=1,')
    begin=cpp.index('void rpu_launch_g05_action_tail_spm_kernel(');end=cpp.index('\nvoid rpu_launch_all_reduce_sum_residual_kernel',begin)
    cpp=cpp[:begin]+'''void rpu_launch_g05_action_tail_spm_kernel(uint32_t x,uint32_t skip,uint32_t raw,uint32_t cg,uint32_t ng,uint32_t cs,uint32_t ch,uint32_t ns,uint32_t nh,int64_t clean,double eps){assert(x==0x40070000&&skip==0x40050000&&raw==0x40030000&&ng==cg+6144&&ch==cs+2048&&ns==cs+6144&&nh==cs+8192&&cs==cg+8192&&eps==1e-6);if(fail_producer)throw std::runtime_error("producer");calls.push_back("mlp-fused");native_clean=clean;native_gate=cg;native_scale=cs;}
'''+cpp[end:]
    cpp=cpp.replace('return s=="oproj"?0x40010000:', 'return s=="down"?0x40070000:s=="input_norm"?(bad_alias?0x40060000:0x40050000):s=="oproj"?0x40010000:')
    begin=cpp.index(' void apply_adaptive_norm(');end=cpp.index('\n',begin)
    cpp=cpp[:begin]+''' void apply_adaptive_norm(uint32_t input,uint32_t output,const ChunkInfo&chunk){assert((input==addr(0,"residual1")&&output==addr(0,"input_norm"))||(input==output&&input==addr(0,"residual1")));if(action_norm_route(chunk.len)==3)consume_physical_route(FmbRouteFamily::NORMALIZATION,QWEN35_ACTION_FUSED_NORM_SHIFT_SITE,3,0,action_norm_arguments(chunk.len),chunk.idx);calls.push_back("norm");}
'''+cpp[end:]
    manifest='if (use_action_mlp_tail(plan,physical_len))'+_block(source,'if (use_action_mlp_tail(plan,physical_len))')
    begin=cpp.index(' void plan(){');end=cpp.index('\n int64_t layout_hash()',begin)
    cpp=cpp[:begin]+''' void plan(){manifest.routes={{QWEN35_TEXT_MLP_ALL_REDUCE_SITE,FmbRouteFamily::ALL_REDUCE,2,0,{},0},{QWEN35_ACTION_FUSED_NORM_SHIFT_SITE,FmbRouteFamily::NORMALIZATION,3,0,action_norm_arguments(32),0}};const auto&plan=stage_plan;int64_t physical_len=seq_len;ChunkInfo chunk;'''+manifest+'}\n'+cpp[end:]
    begin=cpp.index(' int64_t layout_hash(){');end=cpp.index('\n std::vector<int64_t>cost()',begin)
    cpp=cpp[:begin]+' int64_t layout_hash(){int64_t hash=0;if(action_mode_&&action_mlp_tail_available_)'+_block(source,'if (action_mode_ && action_mlp_tail_available_)')+'return hash;}'+cpp[end:]
    begin=cpp.index(' std::vector<int64_t>cost(){');end=cpp.index('\n void build_full_attention(',begin)
    cpp=cpp[:begin]+' std::vector<int64_t>cost(){std::vector<int64_t>identity;identity.push_back(action_mlp_tail_available_);if(action_mlp_tail_available_)'+_block(header,'if (action_mlp_tail_available_)')+'return identity;}'+cpp[end:]
    layer=_block(source,'void Qwen3_5Model::build_layer_subgraph(')
    start=layer.index('    if (action_mode_) {',layer.index('// ② input_layernorm'));end=layer.index('    // This layer-local',start)
    entry=layer[start:end]
    tail=_block(source,'void Qwen3_5Model::emit_mlp_and_output(')
    start=tail.index('    if (action_mlp_tail_handoff(layer_idx,chunk,true))');end=tail.index('    // Split the last decoder-layer output',start)
    ring=tail[start:end]
    final='if (is_last_layer)'+_block(tail,'if (is_last_layer)')
    begin=cpp.index(' void layer(int layer_idx,');end=cpp.index('\n};',begin)
    cpp=cpp[:begin]+''' void input_entry(int layer_idx,const ChunkInfo&chunk={}){const auto h=hidden_size();const uint32_t input_residual=addr(0,"residual1");'''+entry+'''}
 void mlp_end(int layer_idx,const ChunkInfo&chunk={}){int64_t seq_len=chunk.len,h=hidden_size();const bool is_last_layer=layer_idx==23;const uint32_t residual_buf=addr(0,"residual2");load_adaptive_mod_row(2*layer_idx+1);'''+ring+final+'}\n'+cpp[end:]
    cpp+=r'''
#define TORCH_INTERNAL_ASSERT(x) assert(x)
enum class ChunkMode{SEQUENTIAL,KV_FIRST};enum class InterLayerIO{SPM_RESIDENT,DDR_PINGPONG};
enum class SpmFmbOccurrenceKind{LayerBody,KvInsertBody};enum class BufferScope{LayerWide,Compute,KvInsert};
using SpmFmbScopeMask=uint32_t;constexpr uint32_t spm_fmb_scope_bit(BufferScope x){return 1u<<uint32_t(x);}
namespace detail{bool fmb_kv_first_pair_carry_eligible(const std::vector<ChunkInfo>&,const std::vector<ChunkInfo>&,int64_t){return false;}}
'''+traversal+r'''
int main(){
 Model m;m.install();m.plan();assert(m.action_mlp_tail_available_&&m.use_action_mlp_tail(m.stage_plan,32)&&m.manifest.routes.size()==3);
 auto hash=m.layout_hash();auto cost=m.cost();const int queries=KernelCache::instance().queries;
 // The runtime callback and the test use the exact same FMB traversal. Oracle
 // history verifies producer/consumer matching; production stores no ready flag.
 auto run=[&](Model&x,int group,InterLayerIO io,int fail_layer){
  FmbExecutionTraversalSpec spec;spec.num_layers=24;spec.cross_batch=group;spec.chunks=&x.stage_plan.compute.chunks;std::vector<ChunkInfo>none;spec.kv_insert_chunks=&none;spec.inter_layer_io=io;
  int producers=0,consumers=0,normalizations=0;bool aborted=false;
  for(int body=0;body<10;++body){x.body_iter=body;int previous=-1;try{for_each_fmb_execution_step(spec,body,[&](const auto&step){
   x.input_in_spm=step.input_in_spm;x.output_to_spm=step.output_to_spm;
   const bool skip=x.action_mlp_tail_handoff(step.layer,*step.chunk,false);if(skip){assert(previous==step.layer-1);++consumers;}
   calls.clear();x.input_entry(step.layer,*step.chunk);assert(x.adaptive_mod_row_offset_==uint32_t(2*step.layer*12288));assert(calls.size()==size_t(!skip));normalizations+=calls.size();
   calls.clear();fail_producer=step.layer==fail_layer;x.mlp_end(step.layer,*step.chunk);fail_producer=false;
   if(x.action_mlp_tail_handoff(step.layer,*step.chunk,true)){assert(calls==std::vector<std::string>{"mlp-fused"});assert(native_gate==x.mod_base+(2*step.layer+1)*12288+4096&&native_scale==x.mod_base+(2*step.layer+2)*12288);previous=step.layer;++producers;}
   else {assert(calls.front()=="ring");previous=-1;}
   if(step.layer==23){assert(calls==std::vector<std::string>({"ring","norm"})&&x.adaptive_mod_row_offset_==48*12288);++normalizations;}
  });}catch(const std::runtime_error&){assert(fail_layer>=0);fail_producer=false;aborted=true;}}
  if(fail_layer<0){const int expected=io==InterLayerIO::SPM_RESIDENT&&x.use_action_mlp_tail(x.stage_plan,32)?10*(24-(24+group-1)/group):0;assert(producers==expected&&consumers==expected&&normalizations==490-expected-240);}
  else assert(aborted);
  return producers;
 };
 for(int clean:{0,5,32}){m.action_rtc_prefix_len_=clean;assert(run(m,24,InterLayerIO::SPM_RESIDENT,-1)==230);assert(native_clean==clean);}
 for(int group:{1,4,6})run(m,group,InterLayerIO::SPM_RESIDENT,-1);
 run(m,24,InterLayerIO::DDR_PINGPONG,-1);
 for(int fail:{0,7,22}){run(m,24,InterLayerIO::SPM_RESIDENT,fail);assert(run(m,24,InterLayerIO::SPM_RESIDENT,-1)==230);}
 assert(KernelCache::instance().queries==queries);
 m.body_iter=0;m.input_in_spm=true;m.output_to_spm=true;m.load_adaptive_mod_row(1);calls.clear();
 for(size_t r=0;r<m.manifest.routes.size();++r){for(size_t i=0;i<m.manifest.routes[r].args.size();++i){auto x=m;++x.manifest.routes[r].args[i];reject([&]{x.launch_action_mlp_tail(0,{});});}for(int f=0;f<7;++f){auto x=m;auto&t=x.manifest.routes[r];if(f==0)++t.site;if(f==1)++t.selector;if(f==2)++t.flags;if(f==3)++t.invocation;if(f==4)t.family=FmbRouteFamily::OTHER;if(f==5)x.manifest.routes.erase(x.manifest.routes.begin()+r);if(f==6)x.complete=false;reject([&]{x.launch_action_mlp_tail(0,{});});}}
 for(int f=0;f<10;++f){auto x=m;if(f==0)x.body_iter=-1;if(f==1)x.body_iter=10;if(f==2)++x.adaptive_mod_row_offset_;if(f==3)x.mod_base=0x407f0000;if(f==4)x.mod_base=0;if(f==5)++x.mod_base;if(f==6)x.bad_alias=true;if(f==7)x.output_to_spm=false;if(f==8)x.batch_size=2;if(f==9)x.seq_len=31;reject([&]{x.launch_action_mlp_tail(0,{});});}
 for(int l:{-1,23,24})reject([&]{m.launch_action_mlp_tail(l,{});});
 for(auto c:std::vector<ChunkInfo>{{31,0,0},{32,1,0},{32,0,1}})reject([&]{m.launch_action_mlp_tail(0,c);});
 for(int f=0;f<8;++f){auto x=m;if(f==0)x.stage_plan.qkv.chunks.clear();if(f==1)x.stage_plan.compute.chunks.push_back({});if(f==2)x.stage_plan.qkv.chunks[0].idx=1;if(f==3)x.stage_plan.qkv.chunks[0].offset=1;if(f==4)x.stage_plan.qkv.chunks[0].len=31;if(f==5)x.stage_plan.compute.chunks[0].idx=1;if(f==6)x.stage_plan.compute.chunks[0].offset=1;if(f==7)x.stage_plan.compute.chunks[0].len=31;assert(!x.use_action_mlp_tail(x.stage_plan,32));reject([&]{x.launch_action_mlp_tail(0,{});});}
 for(int f=0;f<4;++f){auto x=m;if(f==0)x.action_rtc_active_=false;if(f==1){x.action_nvfp4_.on=false;x.action_mod_body_w8_=true;}if(f==2)x.action_num_steps_=1;if(f==3)x.debug=true;if(f==4)x.linear_acc32_=true;x.install();x.plan();assert(!x.use_action_mlp_tail(x.stage_plan,32));if(f<2)run(x,24,InterLayerIO::SPM_RESIDENT,-1);}
 {auto x=m;x.linear_acc32_=true;x.install();x.plan();assert(x.action_mlp_tail_arguments()[4]==32&&x.layout_hash()!=hash&&x.cost()!=cost);assert(run(x,24,InterLayerIO::SPM_RESIDENT,-1)==230);x.body_iter=0;x.output_to_spm=true;x.load_adaptive_mod_row(1);x.manifest.routes.back().args[4]=16;reject([&]{x.launch_action_mlp_tail(0,{});});}
 auto&cache=KernelCache::instance();cache.available=false;m.install();m.plan();assert(!m.action_mlp_tail_available_&&m.manifest.routes.size()==2&&m.layout_hash()!=hash&&m.cost()!=cost);run(m,24,InterLayerIO::SPM_RESIDENT,-1);
 cache.available=true;m.install();m.plan();assert(m.layout_hash()==hash&&m.cost()==cost);m.reset();assert(!m.action_mlp_tail_available_&&!m.use_action_mlp_tail(m.stage_plan,32));
}
'''
    assert 'prepare_action_mlp_tail();' in _block(source,'void Qwen3_5Model::set_action_io_weights(')
    assert 'action_mlp_tail_available_ = false;' in _block(source,'void Qwen3_5Model::set_weights(')
    _compile_run(tmp_path,'action_mlp_tail_model',cpp)
