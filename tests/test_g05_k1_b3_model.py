"""Execute exact K1 B3 scheduling with the real native descriptor codec."""
from pathlib import Path
import re
import pytest
from test_g05_vision_owner_model import _block
from test_siglip_attention_dry_live_contract import _run_native
ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize("high", [False, True])
def test_k1_b3_plan_and_camera_isolation(tmp_path,monkeypatch,high):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD","0")
    import torch
    source=(ROOT/'src/fused/rpu_qwen3_5_vision_model.cpp').read_text()
    methods=[]
    for signature in ['std::vector<int64_t> g05_high_precision_encoder_arguments(int64_t rows) const',
     'void consume_g05_high_precision_encoder_activation(int64_t rows, int64_t invocation)',
     'bool g05_k1_b3_available() const','bool is_g05_k1_b3_profile() const',
     'bool g05_k1_b3_schedule(const FmbThreeStageChunkPlan& plan) const',
     'std::vector<int64_t> g05_k1_b3_arguments() const',
     'void append_g05_k1_b3_routes(FmbPhysicalExecutionManifest& manifest) const',
     'void consume_g05_k1_b3_schedule(const ChunkInfo& chunk,bool compute)',
     'void emit_g05_k1_b3_layer(int layer,const ChunkInfo& chunk)']:
     name=re.search(r'(\w+)\(',signature)[1]
     methods.append(signature+_block(source,'Qwen3_5VisionModel::'+name+'('))
    constants='\n'.join(re.findall(r'constexpr int64_t G05_VISION_(?:K1_B3|HIGH)_\w+_SITE = \d+LL;',source))
    parentmethods='\n'.join(signature+_block(source,marker) for signature,marker in [
     ('std::vector<int64_t> g05_qkv_linear_arguments() const','Qwen3_5VisionModel::g05_qkv_linear_arguments('),
     ('std::vector<int64_t> g05_owner_ln_arguments() const','Qwen3_5VisionModel::g05_owner_ln_arguments('),
     ('int64_t input_chunk_size_for_geometry(int64_t execution_len) const','Qwen3_5VisionModel::input_chunk_size_for_geometry('),
     ('VisionExecutionPlan resolve_execution_plan(int64_t execution_len)','Qwen3_5VisionModel::resolve_execution_plan(')])
    makechunks='static std::vector<ChunkInfo> make_vision_chunks(int64_t execution_len,int64_t chunk_size)'+_block(source,'static std::vector<ChunkInfo> make_vision_chunks(')
    cpp=r'''
    #include "core/fused_model_base.h"
    #include <ATen/ATen.h>
    #include <cassert>
    #include <cmath>
    #include <iostream>
    using namespace v3;
    __CONSTANTS__
    __MAKECHUNKS__
    constexpr int64_t QWEN3_5_VISION_STEP0_CHUNK=896,G05_COMPACT_STEP0_ROWS=768,QWEN3_5_VISION_MERGER_CHUNK=512,QWEN3_5_SPATIAL_MERGE_UNIT=4;
    constexpr int64_t G05_VISION_KV_RAW_SPM_ONLY=2;
    enum class Qwen35VisionAllReduceRoute : int64_t {PREPARE_RING_INPUT=3};
    int64_t qwen35_vision_rope_route(){return 1;}
    int64_t qwen35_vision_linear_route(){return 1;}
    enum class ValuOpType{ADD};enum class GeluMode{TANH};
    std::vector<std::string> emitted;int camera=-1;int layer_value=0;bool expected_high=__HIGH__;
    constexpr uint32_t R=0x40010000,N=0x40200000,P=0x40400000,F=0x40600000,Q=0x40700000,K=0x40710000,V=0x40720000,T=0x40730000,O=0x40750000;
    void rpu_launch_layernorm_spm_kernel(uint32_t in,uint32_t out,uint32_t,uint32_t,int64_t rows,int64_t hidden,double eps,bool skip,int,int cores){
     camera=(in-R)/524288;assert(camera>=0&&camera<3&&in==R+camera*524288&&out==N+camera*524288&&rows==256&&hidden==1024&&eps==1e-6&&!skip&&cores==8);emitted.push_back("ln"+std::to_string(camera));}
    void rpu_launch_g05_vision_qkv_linear_w8a16_acc32_kernel(uint32_t in,const at::Tensor&,const at::Tensor&,const at::Tensor&,uint32_t q,uint32_t k,uint32_t v,uint32_t,uint32_t,uint32_t,const at::Tensor&,const at::Tensor&,const at::Tensor&,bool acc32){assert(in==N+camera*524288&&q==Q&&k==K&&v==V&&!acc32);emitted.push_back("qkv");}
    void rpu_launch_rope_2d_spm_kernel(uint32_t in,uint32_t out,uint32_t,uint32_t,int16_t*position,int64_t offset,int64_t rows,int64_t heads,int64_t d,int64_t padded,int cores){assert(in==out&&(in==Q||in==K)&&offset==camera*256&&position[offset*2]==camera&&rows==256&&heads==2&&d==64&&padded==64&&cores==8);emitted.push_back("rope");}
    void rpu_launch_v_transpose_spm(uint32_t v,uint32_t tmp,int64_t batch,int64_t seq,int64_t heads,int64_t dim,int cores){assert(v==V&&tmp==T&&batch==1&&seq==256&&heads==16&&dim==64&&cores==8);emitted.push_back("transpose");}
    void rpu_launch_sdpa_by_mha_spm(uint32_t q,uint32_t k,uint32_t vt,uint32_t out,uint32_t mask,int masktype,double scale,int64_t batch,int64_t seq_q,int64_t seq_k,int64_t nq,int64_t nk,int64_t d,int cores){assert(q==Q&&k==K&&vt==T&&out==O&&mask==0&&masktype==0&&scale==.125&&batch==1&&seq_q==256&&seq_k==256&&nq==16&&nk==16&&d==64&&cores==8);emitted.push_back("attention");}
    void rpu_prepare_ring_all_reduce_input(uint32_t out,int64_t rows,int64_t cols,int tp,int cores){assert(out==P+camera*524288&&rows==256&&cols==1024&&tp==8&&cores==8);emitted.push_back("prepare");}
    void rpu_launch_linear_spm_to_spm_acc16_kernel(uint32_t in,const at::Tensor&,uint32_t out,int64_t m,int64_t n,int64_t k,int part,int cores,uint32_t,const at::Tensor&,int,int,bool acc32){assert(m==256&&k==1024&&cores==8&&!acc32);if(part==0){assert(n==1024&&in==O&&out==P+camera*524288);emitted.push_back("o");}else{assert(part==1&&n==4096&&in==P+camera*524288&&out==F+camera*262144);emitted.push_back("fc1");}}
    void rpu_launch_g05_vision_owner_reduce_layernorm_m256n1024_kernel(uint32_t p,uint32_t r,uint32_t raw,uint32_t norm,uint32_t,uint32_t){assert(p==P+camera*524288&&r==R+camera*524288&&raw==N+camera*524288&&norm==p);emitted.push_back("ownerln");}
    void rpu_launch_eltwise_unary_spm_kernel(uint32_t in,uint32_t out,int64_t elements,ValuOpType,GeluMode mode,int cores,RpuUnaryPrecision precision){assert(in==F+camera*262144&&out==in&&elements==256*512&&mode==GeluMode::TANH&&cores==8&&precision==(expected_high?RpuUnaryPrecision::HIGH:RpuUnaryPrecision::BASE));emitted.push_back("gelu");}
    void rpu_launch_g05_vision_fc2_weight_outer_linear_w8a16_acc16_kernel(uint32_t in,const at::Tensor&,uint32_t out,uint32_t,const at::Tensor&){assert(in==F&&out==P&&camera==2);emitted.push_back("fc2x3");}
    void rpu_launch_g05_vision_compact_residual_m256n1024_kernel(uint32_t in,uint32_t raw,uint32_t out){int c=(in-P)/524288;assert(c>=0&&c<3&&in==P+c*524288&&raw==N+c*524288&&out==R+c*524288);emitted.push_back("compact"+std::to_string(c));}
    struct HostContext{
     FmbThreeStageChunkPlan stage_plan;FmbPhysicalManifestConsumer consumer;
     bool complete=true,is_causal=false,input_in_spm=false,output_to_spm=false;
     int64_t position=0,seq_len=768,batch_size=1;std::optional<at::Tensor>attention_mask;
     AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
     bool has_complete_physical_manifest()const{return complete;}
     const FmbRouteManifestEntry& find_physical_route(FmbRouteFamily f,int64_t s,int64_t i=0)const{return consumer.find_route(f,s,i);}
     void consume_physical_route(FmbRouteFamily f,int64_t s,int64_t sel,int64_t flags,std::vector<int64_t>a,int64_t i=0){consumer.consume_route(f,s,sel,flags,a,i);}
    };
    struct Weights{at::Tensor q_w,k_w,v_w,o_w,fc1_w,fc2_w,q_ws,k_ws,v_ws,o_ws,fc1_ws,fc2_ws;};
    struct Model{
 bool is_g05_k1_camera_group_profile()const{return false;}
 std::vector<int64_t>g05_k1_camera_group_arguments()const{assert(false);return {};}
 bool g05_k1_camera_group_available(int64_t)const{return false;}
 void consume_g05_k1_camera_group_schedule(const ChunkInfo&,bool){assert(false);}
 void emit_g05_k1_camera_group_layer(int,const ChunkInfo&){assert(false);}

     bool g05_k1_b3_available_=true,linear_acc32_=false,g05_w8a16_=true,g05_owner_ln_available_=true,g05_qkv_linear_available_=true,z2_bound_=false,debug=false,profile=true,current_compact_step0_=false;
     bool has_step0_=true,has_merger_=true;int64_t temporal_patches_per_frame_=256;
     bool high_precision_=expected_high;int num_cores()const{return 8;}int intermediate_size()const{return 4096;}
     int64_t current_temporal_num_frames_=1,current_camera_batch_count_=3,current_output_patches_=768;
     struct VisionExecutionPlan{int64_t input_chunk_size=0,qkv_chunk_size=0,compute_chunk_size=0,merger_chunk_size=0,merger_chunks_per_span=0;FmbThreeStageChunkPlan stages;};
     int64_t get_chunk_size_override()const{return 0;}
     int64_t resolve_chunk_size_for_shape(int64_t rows,int64_t,const std::optional<at::Tensor>&,bool){assert(rows==768);return 768;}
     ChunkPlan plan_kv_first_chunks(const ChunkPlan&plan){return plan;}
     double eps_=1e-6;std::vector<Weights>layer_weights_{24};
     at::Tensor position_idx_keepalive_=at::empty({768,2},at::TensorOptions().dtype(at::kShort));HostContext context;
     Model(){for(int c=0;c<3;++c)position_idx_keepalive_.narrow(0,c*256,256).fill_(c);}
     bool is_g05_k1_fast_replay_profile()const{return profile;}
     bool get_debug_export()const{return debug;}
     HostContext&ctx(){return context;}
     uint32_t addr(int,const char*name)const{std::string n(name);return n=="residual1"?R:n=="input_norm"?N:n=="oproj"?P:n=="fc1"?F:n=="q"?Q:n=="k"?K:n=="v"?V:n=="sdpa_tmp"?T:O;}
     uint32_t layer_addr(int layer,int,const char*)const{assert(layer==layer_value);return 0x40800000;}
     void emit_layer_input_dma(int,const ChunkInfo& c){assert(c.len==768&&c.offset==0);emitted.push_back("input768");}
     void emit_layer_output_dma(int,const ChunkInfo& c,const char*){assert(c.len==768&&c.offset==0);emitted.push_back("output768");}
     void consume_manifest_route(FmbRouteFamily f,int64_t s,int64_t sel,int64_t i,std::vector<int64_t>a){ctx().consume_physical_route(f,s,sel,0,a,i);}
     __PARENTS__
     __METHODS__
    };
    template<class F>void reject(F f){auto before=emitted;bool caught=false;try{f();}catch(const c10::Error&){caught=true;}assert(caught&&before==emitted);}
    int main(){
     Model m;const ChunkInfo full{0,0,768,768};
     auto execution=m.resolve_execution_plan(768);assert(execution.input_chunk_size==256&&execution.compute_chunk_size==768&&execution.qkv_chunk_size==768&&execution.merger_chunks_per_span==1);
     m.context.stage_plan=execution.stages;
     assert(m.g05_k1_b3_schedule(m.context.stage_plan));
     FmbPrefillStageCandidate c;c.input_chunk_size=256;c.qkv_chunk_size=c.compute_chunk_size=768;c.stage_plan=m.context.stage_plan;
     auto&manifest=c.physical_manifest;manifest.state=FmbPhysicalManifestState::COMPLETE;manifest.logical_length=manifest.physical_length=manifest.kv_logical_length=manifest.kv_insert_physical_rows=768;manifest.graph_lifecycle=FmbGraphLifecycle::RETAINED_CACHE;manifest.linear_accumulation=FmbLinearAccumulationPolicy::ACC16;
     m.append_g05_k1_b3_routes(manifest);std::sort(manifest.routes.begin(),manifest.routes.end(),[](const auto&a,const auto&b){return std::tie(a.family,a.site_id,a.invocation)<std::tie(b.family,b.site_id,b.invocation);});
     auto wire=encode_fmb_prefill_stage_candidate(c);auto restored=decode_fmb_prefill_stage_candidate(wire);assert(fmb_physical_manifest_fingerprint(restored.physical_manifest)==fmb_physical_manifest_fingerprint(manifest));
     m.context.consumer.reset(restored.physical_manifest);m.consume_g05_k1_b3_schedule(full,false);m.emit_g05_k1_b3_layer(0,full);m.context.consumer.require_all_consumed();
     std::vector<std::string>expected{"input768"};for(int cam=0;cam<3;++cam){expected.push_back("ln"+std::to_string(cam));expected.insert(expected.end(),{"qkv","rope","rope","transpose","attention","prepare","o","ownerln","fc1","gelu"});}expected.insert(expected.end(),{"fc2x3","compact0","compact1","compact2","output768"});assert(emitted==expected);
     // Retain all three camera slices between layers and publish only the final output.
     emitted.clear();m.context.input_in_spm=m.context.output_to_spm=true;layer_value=9;m.emit_g05_k1_b3_layer(9,full);assert(emitted.size()==expected.size()-2&&emitted.front()=="ln0"&&emitted.back()=="compact2");
     // Every packed route argument is validated before the first emitted kernel.
     for(size_t r=0;r<manifest.routes.size();++r)for(size_t a=0;a<manifest.routes[r].arguments.size();++a){auto bad=manifest;++bad.routes[r].arguments[a];m.context.consumer.reset(bad);reject([&]{m.emit_g05_k1_b3_layer(9,full);});}m.context.consumer.reset(manifest);
     for(int fault=0;fault<8;++fault){auto x=m;if(fault==0)x.context.complete=false;if(fault==1)x.context.position=1;if(fault==2)x.context.seq_len=256;if(fault==3)x.context.attention_policy=AttentionExecutionPolicy::DDR_KV;if(fault==4)x.context.stage_plan.spans[1].group_id=0;if(fault==5)x.context.stage_plan.input.chunks[1].offset=0;if(fault==6)x.current_temporal_num_frames_=6;if(fault==7)x.g05_k1_b3_available_=false;reject([&]{x.emit_g05_k1_b3_layer(9,full);});}
     reject([&]{m.emit_g05_k1_b3_layer(9,{0,0,256,256});});
     std::cout<<"K1 B3 real FMB codec/consumer + exact emitted camera geometry: PASS; descriptor words="<<wire.size()<<", routes="<<manifest.routes.size()<<"\n";
    }
    '''
    cpp=cpp.replace('__MAKECHUNKS__',makechunks).replace('__CONSTANTS__',constants).replace('__PARENTS__',parentmethods).replace('__METHODS__','\n'.join(methods)).replace('__HIGH__','true' if high else 'false')
    result=_run_native(tmp_path,cpp,torch,extra_sources=(ROOT/'src/core/fmb_three_stage_chunk_plan.cpp',))


def test_k1_b3_adapter_old_native_and_ref_fallback():
    import ast
    path=ROOT/'python/rpu_backend/adapters/qwen3_5/vision.py';source=path.read_text();tree=ast.parse(source)
    # Execute the exact optional-op installation, without masking real native failures.
    from types import SimpleNamespace
    install=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and any(
        isinstance(child,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='k1_b3_query' for t in child.targets)
        for child in n.body))
    start_install=next(i for i,n in enumerate(install.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='k1_b3_query' for t in n.targets))
    install_code=compile(ast.Module(body=install.body[start_install:start_install+2],type_ignores=[]),str(path),'exec')
    missing=object()
    def installed_capability(query=missing):
        ops=SimpleNamespace()
        if query is not missing:ops.qwen3_5_vision_k1_b3_available=query
        vision_model=SimpleNamespace()
        exec(install_code,dict(torch=SimpleNamespace(ops=SimpleNamespace(rpu=ops)),vision_model=vision_model,handle=123))
        return vision_model._rpu_vision_k1_b3_available
    assert installed_capability() is False
    assert installed_capability(lambda handle:handle!=123) is False
    assert installed_capability(lambda handle:handle==123) is True
    def bad_native(handle):raise RuntimeError('native capability failure')
    try:installed_capability(bad_native)
    except RuntimeError as exc:assert str(exc)=='native capability failure'
    else:raise AssertionError('native capability failure was swallowed')
    forward=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_rpu_vision_forward')
    start=next(i for i,n in enumerate(forward.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='requested_k1_group' for t in n.targets))
    stop=next(i for i,n in enumerate(forward.body[start:],start) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='merger_on' for t in n.targets))
    code=compile(ast.Module(body=forward.body[start:stop],type_ignores=[]),str(path),'exec')
    class Model:
     _rpu_vision_k1_b3_available=True
     _rpu_vision_chunk_size_cap=0
     _rpu_vision_exact_chunk_size=None

    def run(counts,*,available=True,frames=1,request=3,exact=None,cap=0):
     m=Model();m._rpu_vision_k1_b3_available=available;m._rpu_vision_chunk_size_cap=cap;m._rpu_vision_exact_chunk_size=exact
     ns=dict(self=m,temporal_num_frames=frames,camera_batch_count=request,compact_input=False,patches_per_image=counts,num_cores=8,total_patches=sum(counts))
     exec(code,ns);return ns['groups'],ns['camera_batch_count']
    assert run([256]*3)==([(0,768,768)],3)
    for kwargs in ({'available':False},{'exact':256},{'cap':256}):assert run([256]*3,**kwargs)==([(0,256,256),(1,256,256),(2,256,256)],1)
    assert run([256])==([(0,256,256)],1)
    assert run([256,128,256])==([(0,256,256),(1,128,128),(2,256,256)],1)
    assert run([256]*18,frames=6,request=3)==([(0,4608,768)],3)
    # Execute the actual merger source/destination row expressions for the native B3 branch.
    cpp=(ROOT/'src/fused/rpu_qwen3_5_vision_model.cpp').read_text();start=cpp.index('void Qwen3_5VisionModel::emit_merger()')
    body=cpp[start:cpp.index('// LAYER BUILD',start)]
    assert ': camera * camera_patches;' in body
    assert 'camera * (camera_patches / QWEN3_5_SPATIAL_MERGE_UNIT)' in body
    assert 'merger_wrote_fusion_target_\n                ? off / mu : merged_camera_base + off / mu' in body
