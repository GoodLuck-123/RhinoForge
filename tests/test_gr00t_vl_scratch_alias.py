"""Long-prefix VL scratch declarations, real allocator, and Z2 isolation."""
from pathlib import Path
import re
import shutil
import subprocess

from test_fmb_standalone_layout_rebuild import _definition
from test_gr00t_mask_residency import _program

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/fused/rpu_gr00t_vl_encoder_model.cpp"


def test_vl_long_prefix_alias_real_allocator_and_z2_prepare(tmp_path):
    source = SOURCE.read_text()
    # Reuse the actual tiling, BufferDecl and allocator extraction, without the
    # DiT fixture. Preserve every production VL declaration and size.
    cpp = _program(full_declarations=True).split("class Gr00tDiTModel{public:")[0]
    declaration = _definition(source, "std::vector<BufferDecl> declare_buffers(")
    fields = sorted(set(re.findall(r"&Gr00tVlEncoderModel::(\w+)", declaration)))
    layer_fields = sorted(set(re.findall(r"&LayerWeights::(\w+)", declaration)))
    while "d.preload_callback =" in declaration:
        callback = _definition(declaration, "d.preload_callback =")
        declaration = declaration.replace(callback, "d.preload_callback=nullptr", 1)
    cpp += r'''
namespace c10 {
template<class F>struct ScopeExit{F f;~ScopeExit(){f();}};
template<class F>ScopeExit<F>make_scope_exit(F f){return {f};}
}
struct SpmPipelineComponentLayout{int id=1;};
enum class ChunkMode{SEQUENTIAL};
int compose_fmb_default_three_stage_chunk_plan(const LayoutContext&,int64_t,int64_t,ChunkMode){return 1;}
class Gr00tVlEncoderModel{public:
 bool z2_layout_preparing_=false,z2_bound_=false,prepare_failure=false;
 int64_t z2_prepared_seq_len_=0,h=2048,nq=32,nkv=32,hd=64,ff=8192,layers=4,tp=8;
 int64_t hidden_size()const{return h;}int64_t num_q_heads()const{return nq;}
 int64_t num_kv_heads()const{return nkv;}int64_t head_dim()const{return hd;}
 int64_t intermediate_size()const{return ff;}int64_t num_layers()const{return layers;}
 int attn_tp()const{return tp;}
 SdpaConfig make_sdpa_config(int mask)const{return {SdpaKernelType::FLASH_ATTN_SPM,hd,nq,nq,8,mask};}
 __FIELDS__
 struct LayerWeights{__LAYER_FIELDS__};std::vector<LayerWeights>layer_weights_=std::vector<LayerWeights>(4);
 __DECLARATION__
 SpmPipelineComponentLayout prepare_spm_pipeline_component(const LayoutContext& layout,int){
  assert(z2_layout_preparing_);
  for(const auto&d:declare_buffers(layout))assert(d.alias_of==nullptr);
  TORCH_CHECK(!prepare_failure);return {};
 }
 __PREPARE__
};
bool aliased(const std::vector<BufferDecl>&d){
 for(const auto&b:d)if(std::string(b.name)=="oproj")return b.alias_of&&std::string(b.alias_of)=="norm_hidden";
 assert(false);return false;
}
int64_t fixed(const std::vector<BufferDecl>&d){int64_t n=0;for(const auto&b:d){
 if(b.storage==StorageClass::Persistent)n+=b.size;
 if(b.storage==StorageClass::PersistentPerLayer)n+=b.size*b.per_layer;}return n;}
template<class F>void rejects(F f){try{f();assert(false);}catch(const std::runtime_error&){}}
}
int main(){using namespace v3;
 Gr00tVlEncoderModel model;int64_t fixed_bytes=-1;
 constexpr int64_t four_step_available=8303616-1586688-65536;
 for(int seq:{193,225,289,238,270,334,384,512})for(bool raw:{false,true})for(bool capacity:{false,true}){
  LayoutContext layout;layout.chunk_size=capacity?Align(seq,16):seq;layout.max_kv_seq_len=seq;
  layout.batch_size=1;layout.is_causal=false;layout.physical_manifest_fingerprint=123;
  layout.attention_policy=raw?AttentionExecutionPolicy::SPM_KV_BY_MHA:AttentionExecutionPolicy::DDR_KV;
  const auto decl=model.declare_buffers(layout);const auto peak=detail::estimate_temporary_total(decl);
  const int64_t cs=layout.chunk_size,res=Align(cs*2048*2,256);
  assert(aliased(decl)==(cs>256));
  if(fixed_bytes<0)fixed_bytes=fixed(decl);assert(fixed_bytes==fixed(decl));
  auto baseline=decl;for(auto&d:baseline)d.alias_of=nullptr;
  assert(detail::estimate_temporary_total(baseline)-peak==(cs>256?res:0));
  // A cache capacity of 384/512 is not an executed VL chunk certificate.
  assert((peak<=four_step_available)==(seq<=334));
  if(seq==334)assert(detail::estimate_temporary_total(baseline)>four_step_available);
  std::vector<SpmAllocator::AllocRequest>req;std::vector<std::string>names;
  std::map<std::string,int64_t>sizes,offsets;
  for(const auto&d:decl)if(d.storage==StorageClass::Temp){sizes[d.name]=d.size;
   if(!d.alias_of){names.push_back(d.name);req.push_back({d.size,d.phase_start,d.phase_end,int(d.scope)});}}
  const auto plan=SpmAllocator::plan_temporary_aliased(req);assert(int64_t(plan.peak_bytes)==peak);
  for(size_t i=0;i<req.size();++i)offsets[names[i]]=plan.offsets[i];
  for(const auto&d:decl)if(d.alias_of){assert(sizes[d.name]<=sizes[d.alias_of]);offsets[d.name]=offsets.at(d.alias_of);}
  if(cs>256)assert(offsets["norm_hidden"]==offsets["oproj"]);
  for(auto group:{std::vector<std::string>{"oproj","residual1","residual2"},
                 std::vector<std::string>{"q","k","v","sdpa_tmp","sdpa_out"},
                 std::vector<std::string>{"norm_hidden","ff_mid","residual2"}})
   for(size_t i=0;i<group.size();++i)for(size_t j=i+1;j<group.size();++j){auto a=group[i],b=group[j];
    assert(offsets[a]+sizes[a]<=offsets[b]||offsets[b]+sizes[b]<=offsets[a]);}
  std::cout<<"vl-layout,"<<seq<<","<<cs<<","<<raw<<","<<fixed(decl)<<","<<peak<<"\n";
 }
 LayoutContext long_layout;long_layout.chunk_size=334;long_layout.batch_size=1;long_layout.is_causal=false;
 assert(aliased(model.declare_buffers(long_layout)));
 // Every unsupported mode keeps the historical independent storage.
 for(int bad=0;bad<10;++bad){auto x=model;auto l=long_layout;switch(bad){
  case 0:x.z2_layout_preparing_=true;break;case 1:x.z2_bound_=true;break;
  case 2:x.z2_prepared_seq_len_=334;break;case 3:l.batch_size=2;break;
  case 4:l.is_causal=true;break;case 5:l.use_attn_mask=true;break;
  case 6:x.layers=3;break;case 7:x.tp=4;break;case 8:x.nkv=16;break;case 9:l.chunk_size=256;break;}
  assert(!aliased(x.declare_buffers(l)));}
 for(int seq:{82,334}){auto x=model;x.prepare_z2_layout(seq);
  assert(!x.z2_layout_preparing_&&x.z2_prepared_seq_len_==seq);
  assert(!aliased(x.declare_buffers(long_layout)));}
 model.prepare_failure=true;rejects([&]{model.prepare_z2_layout(334);});
 assert(!model.z2_layout_preparing_&&model.z2_prepared_seq_len_==0);
 assert(aliased(model.declare_buffers(long_layout)));
}
'''
    cpp = (cpp.replace("__FIELDS__", "at::Tensor " + ",".join(fields) + ";")
           .replace("__LAYER_FIELDS__", "at::Tensor " + ",".join(layer_fields) + ";")
           .replace("__DECLARATION__", declaration.replace(" override", ""))
           .replace("__PREPARE__", _definition(source, "SpmPipelineComponentLayout prepare_z2_layout(")))
    path, exe = tmp_path / "vl_layout.cpp", tmp_path / "vl_layout"
    path.write_text(cpp)
    result = subprocess.run([shutil.which("g++"), "-std=c++17", "-O1", str(path), "-o", str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    print(result.stdout)


def test_vl_alias_matches_actual_projection_and_ring_lifetimes():
    block = _definition(SOURCE.read_text(), "void build_layer_subgraph(")
    # Check the actual complete layer stream, including both uses of each
    # shared slot. Future reordering must re-establish this lifetime proof.
    calls = re.findall(r"(rpu_launch_\w+)\((.*?)\);", block, re.S)
    accesses = []
    for name, args in calls:
        operands = re.findall(r'addr\(0, "(\w+)"\)', args)
        if "norm_hidden" in operands or "oproj" in operands:
            accesses.append((name, operands))
    assert accesses == [
        ("rpu_launch_eltwise_binary_1xC_NxC_spm_kernel", ["inv_w_rms", "residual1", "norm_hidden"]),
        ("rpu_launch_layernorm_spm_kernel", ["norm_hidden", "residual1", "vlln_w", "vlln_b"]),
        ("rpu_launch_layernorm_spm_kernel", ["residual1", "norm_hidden"]),
        ("rpu_launch_linear_spm_to_spm_acc16_kernel", ["norm_hidden", "q"]),
        ("rpu_launch_linear_spm_to_spm_acc16_kernel", ["norm_hidden", "k"]),
        ("rpu_launch_linear_spm_to_spm_acc16_kernel", ["norm_hidden", "v"]),
        ("rpu_launch_linear_spm_to_spm_acc16_kernel", ["sdpa_out", "oproj"]),
        ("rpu_launch_all_reduce_sum_residual_kernel", ["oproj", "residual1", "residual2"]),
        ("rpu_launch_layernorm_spm_kernel", ["residual2", "norm_hidden"]),
        ("rpu_launch_linear_spm_to_spm_acc16_kernel", ["norm_hidden", "ff_mid"]),
        ("rpu_launch_linear_spm_to_spm_acc16_kernel", ["ff_mid", "oproj"]),
        ("rpu_launch_all_reduce_sum_residual_kernel", ["oproj", "residual2", "residual1"]),
    ]
