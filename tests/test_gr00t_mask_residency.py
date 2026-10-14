"""GR00T DiT two-mask residency feasibility with production declarations/allocator."""
from pathlib import Path
import re
import shutil
import subprocess

import pytest
from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/fused/rpu_gr00t_dit_model.cpp"


def _program(*, long_profile=False, full_declarations=False):
    helpers = (ROOT / "src/core/rpu_helpers.h").read_text()
    allocator = (ROOT / "src/core/rpu_spm_allocator.cpp").read_text()
    allocator_h = (ROOT / "src/core/rpu_spm_allocator.h").read_text()
    fmb = (ROOT / "src/core/fused_model_base.h").read_text()
    source = SOURCE.read_text()
    p = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(c, ...) do {if(!(c))throw std::runtime_error("guard");}while(0)
#define DWIDTH 2
#define NUM_CORES 8
#define SDPA_MAX_VLM_SIZE 400
#define RPU_SDPA_TILE_K_CONSERVATIVE_MODE 1
template<class T>T CeilDiv(T a,T b){return (a+b-1)/b;}
template<class T,class U>T Align(T a,U b){return CeilDiv(a,static_cast<T>(b))*b;}
'''
    for marker in ("struct SdpaTiling", "enum class SdpaKernelType", "struct SdpaConfig"):
        p += _definition(helpers, marker) + ";\n"
    for marker in (
        "inline int64_t sdpa_tile_m_v16_initial(",
        "inline uint32_t GetMhaAttnUnivVlmNeed(",
        "inline std::optional<int64_t> choose_tile_k_aggressive(",
        "inline std::optional<int64_t> choose_tile_k_conservative(",
        "inline std::optional<int64_t> choose_tile_k(",
        "inline bool sdpa_try_compute_tiling(", "inline SdpaTiling sdpa_compute_tiling(",
        "inline int64_t sdpa_compute_tmp_v16_size(",
    ):
        p += _definition(helpers, marker) + "\n"
    p += "class SpmAllocator{public:static constexpr size_t ALIGN=256;\n"
    p += "static size_t align_up(size_t n){return (n+255)&~size_t{255};}\n"
    for marker in ("struct AllocRequest", "struct AliasedPlan"):
        p += _definition(allocator_h, marker) + ";\n"
    p += "static AliasedPlan plan_temporary_aliased(const std::vector<AllocRequest>&);};\n"
    p += _definition(allocator, "SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(")
    p += "namespace v3{class FusedModelBase{};\n"
    for marker in ("enum class StorageClass", "enum class BufferScope", "struct BufferDecl", "struct LayoutContext"):
        if marker == "struct LayoutContext":
            p += _definition(fmb, "enum class FmbForwardOperandResidency") + ";\n"
            p += "enum class AttentionExecutionPolicy{DDR_KV,SPM_KV_BY_MHA};\nenum class FmbRopeTableResidency{UNSPECIFIED};\n"
        p += _definition(fmb, marker) + ";\n"
    p += "namespace detail{int64_t estimate_temporary_total(const std::vector<BufferDecl>&);}\n"
    p += _definition((ROOT / "src/core/fused_model_base.cpp").read_text(), "int64_t detail::estimate_temporary_total(")
    p += "}\n"
    residency = (ROOT / "src/core/rpu_spm_residency.h").read_text()
    p += residency[residency.index("namespace v3::detail {"):]
    p += "namespace v3{\n"
    p += _definition(source, "enum class Gr00tDiTMaskSchedule") + ";\n"
    p += _definition(source, "static inline int64_t gr00t_pad16(") + "\n"
    p += r'''
class Gr00tDiTModel{public:
 int64_t h=1536,nq=32,hd=48,ff=6144,vl_seq_=82,cross_dim_=2048;
 int64_t action_horizon_=40,action_dim_pad_=144,enc_mid_=1024,dec_mid_=1024,model_out_dim_=1024;
 bool denoise_mode_=false,loop_mode_=false,z2_bound_=false;
 int64_t z2_prepared_seq_len_=0,num_steps_=4;
 struct Layer{bool is_cross;};std::vector<Layer>layer_weights_;
 Gr00tDiTModel(){for(int i=0;i<32;++i)layer_weights_.push_back({i%2==0});}
 int64_t num_layers()const{return layer_weights_.size();}
 int64_t hidden_size()const{return h;}int64_t num_q_heads()const{return nq;}
 int64_t head_dim()const{return hd;}int64_t intermediate_size()const{return ff;}
 int attn_tp()const{return 8;}
 void append_cross_kv_buffers(std::vector<BufferDecl>&,const LayoutContext&)const{}
 bool ff1_resident_profile(int64_t,int64_t)const{return false;}
 static constexpr int64_t GR00T_DIT_FF1_RESIDENT_BYTES=1179648;
 bool adaln_table_profile(int64_t,int64_t)const{return false;}
 int64_t adaln_table_body_bytes()const{return 196608;}
 SdpaConfig make_sdpa_config(int mask)const{return {SdpaKernelType::FLASH_ATTN_SPM,hd,nq,nq,8,mask};}
'''
    decl = _definition(source, "std::vector<BufferDecl> baseline_buffer_declarations(")
    split = decl.index("        // norm3 affine=False")
    head = decl[:split].replace(" override", "")
    denoise = _definition(decl, "if (denoise_mode_)")
    assert head.count("StorageClass::Temp") + denoise.count("StorageClass::Temp") == decl.count("StorageClass::Temp")
    if full_declarations:
        # Keep every production size/storage/lifetime, including Persistent
        # tables and biases. Only device preload callbacks are removed.
        fields = sorted(set(re.findall(r"&Gr00tDiTModel::(\w+)", decl)))
        layer_fields = sorted(set(re.findall(r"&LayerWeights::(\w+)", decl)))
        p = p.replace("class Gr00tDiTModel{public:",
                      "namespace at{struct Tensor{};}\nclass Gr00tDiTModel{public:\n at::Tensor " + ",".join(fields) + ";")
        p = p.replace("struct Layer{bool is_cross;};",
                      "struct Layer{bool is_cross;at::Tensor " + ",".join(layer_fields) + ";};")
        while "d.preload_callback =" in decl:
            callback = _definition(decl, "d.preload_callback =")
            decl = decl.replace(callback, "d.preload_callback=nullptr", 1)
        p += decl.replace(" override", "") + "\n"
    else:
        p += head + denoise + "\nreturn v;}\n"
    for marker in ("int first_cross_mask_layer(", "std::vector<BufferDecl> planned_buffer_declarations(",
                   "std::vector<BufferDecl> declare_buffers(", "int64_t mask_schedule_for_layout(",
                   "std::vector<int64_t> mask_schedule_arguments("):
        p += _definition(source, marker).replace(" override", "") + "\n"
    p += "};\n"
    p += r'''
void check_masks(const std::vector<BufferDecl>&d,int steps=4){
 std::vector<SpmAllocator::AllocRequest> req;
 std::vector<size_t> live;size_t text=0,image=0;
 for(auto&b:d){const std::string name=b.name;
  if(name=="sdpa_mask")text=req.size();if(name=="sdpa_mask_image")image=req.size();
  if(name=="sdpa_mask"||name=="sdpa_mask_image"||name=="cross_k_bank"||name=="cross_v_bank"||name=="adaln_body_table"||name=="ff1_resident_w8"){
   assert(b.phase_start==0&&b.phase_end==8);live.push_back(req.size());}
  req.push_back({b.size,b.phase_start,b.phase_end,int(b.scope)});}
 auto p=SpmAllocator::plan_temporary_aliased(req);
 std::vector<unsigned char>spm(p.peak_bytes,0);
 auto fill=[&](size_t i,unsigned char v){std::fill(spm.begin()+p.offsets[i],spm.begin()+p.offsets[i]+req[i].bytes,v);};
 std::vector<std::vector<unsigned char>> expected;
 for(size_t slot:live){fill(slot,0x11+slot);expected.emplace_back(req[slot].bytes,0x11+slot);
  for(size_t other=0;other<req.size();++other)if(other!=slot)
   assert(p.offsets[slot]+req[slot].bytes<=p.offsets[other]||p.offsets[other]+req[other].bytes<=p.offsets[slot]);}
 for(int body=0;body<steps;++body)for(int layer=0;layer<32;++layer){
  for(size_t i=0;i<req.size();++i)if(std::find(live.begin(),live.end(),i)==live.end())fill(i,body+layer+3);
  for(size_t i=0;i<live.size();++i)assert(std::memcmp(spm.data()+p.offsets[live[i]],expected[i].data(),expected[i].size())==0);
 }
 // One shared retained address is not sufficient for different mask contents.
 fill(text,0x11);fill(text,0x22);
 assert(spm[p.offsets[text]]!=0x11);
}
}
int main(){using namespace v3;Gr00tDiTModel model;
 for(bool denoise:{false,true})for(bool raw:{false,true})for(int seq:{82,96}){
  model.denoise_mode_=denoise;model.loop_mode_=denoise;model.vl_seq_=seq;LayoutContext l;l.chunk_size=41;l.is_causal=false;
  l.attention_policy=raw?AttentionExecutionPolicy::SPM_KV_BY_MHA:AttentionExecutionPolicy::DDR_KV;
  auto d=model.baseline_buffer_declarations(l);auto baseline=detail::estimate_temporary_total(d);
  int64_t mask=0;for(auto&b:d){assert(b.phase_start==0&&b.phase_end==0);if(std::string(b.name)=="sdpa_mask")mask=b.size;}
  auto single_slot=d;auto single=detail::plan_forward_spm_residency(single_slot,{"sdpa_mask"});
  assert(single.retained&&single.retained_peak_bytes==baseline);
  auto doubled=d;for(auto&b:d)if(std::string(b.name)=="sdpa_mask"){auto x=b;x.name="sdpa_mask_image";doubled.push_back(x);break;}
  assert(detail::estimate_temporary_total(doubled)==baseline+mask);
  bool retained=false;auto revised=model.planned_buffer_declarations(l,true,&retained);
  assert(retained);check_masks(revised);
  assert(model.mask_schedule_for_layout(l,true)==2);
  assert(model.mask_schedule_for_layout(l,false)==1);
  assert(model.declare_buffers(l).size()==d.size());
  l.physical_manifest_fingerprint=123;assert(model.declare_buffers(l).size()==d.size()+1);
  model.z2_bound_=true;assert(model.mask_schedule_for_layout(l,true)==1);model.z2_bound_=false;
  model.z2_prepared_seq_len_=96;assert(model.mask_schedule_for_layout(l,true)==1);model.z2_prepared_seq_len_=0;
  auto drift=l;drift.chunk_size=40;assert(model.mask_schedule_for_layout(drift,true)==1);
  model.cross_dim_=512;assert(model.mask_schedule_for_layout(l,true)==1);model.cross_dim_=2048;
  auto peak=detail::estimate_temporary_total(revised);
  assert(peak<=baseline);
  std::cout<<denoise<<","<<raw<<","<<seq<<","<<baseline<<","<<mask<<","<<peak<<"\n";
 }
}
'''
    if long_profile:
        cross = "\n".join(_definition(source, marker) for marker in (
            "bool cross_kv_reuse_profile(", "int64_t cross_kv_bank_stride(",
            "int64_t cross_kv_bank_bytes(", "void append_cross_kv_buffers(",
        ))
        p = p.replace(
            "void append_cross_kv_buffers(std::vector<BufferDecl>&,const LayoutContext&)const{}",
            "using LayerWeights=Layer; bool cross_kv_reuse_opt_in_=false,denoise_ready_=true;\n"
            "int64_t num_kv_heads()const{return nq;}\n" + cross)
        p = p.replace(
            "bool adaln_table_profile(int64_t,int64_t)const{return false;}",
            "struct Table{bool ready=false;bool defined()const{return ready;}}adaln_table_;\n" +
            _definition(source, "bool adaln_table_profile("))
        p = p.replace("int64_t adaln_table_body_bytes()const{return 196608;}",
                      _definition(source, "int64_t adaln_table_body_bytes("))
        p = _with_mask_consumer(p, source)
        # Keep the old <=96 cases in this same native executable. The extension
        # uses the complete production Temp declarations, bank/profile helpers,
        # residency planner and allocator, including the 196608-byte body table.
        p = p.rstrip()[:-1] + r'''
 for(int seq:{169,214})for(bool raw:{false,true})for(bool banks:{false,true})for(bool table:{false,true}){
  Gr00tDiTModel x;x.denoise_mode_=x.loop_mode_=true;x.num_steps_=10;x.vl_seq_=seq;
  const int64_t mask_bytes=seq==169?14592:18432;
  const int64_t bank_stride=seq==169?67584:86016;
  x.cross_kv_reuse_opt_in_=banks;x.adaln_table_.ready=table;
  LayoutContext l;l.chunk_size=41;l.batch_size=1;l.is_causal=false;l.physical_manifest_fingerprint=123;
  l.attention_policy=raw?AttentionExecutionPolicy::SPM_KV_BY_MHA:AttentionExecutionPolicy::DDR_KV;
  const auto baseline=x.baseline_buffer_declarations(l);const auto before=detail::estimate_temporary_total(baseline);
  auto slot=[](auto&decl,const char*name){return std::find_if(decl.begin(),decl.end(),[&](const auto&d){return std::string(d.name)==name;});};
  assert(slot(baseline,"sdpa_mask")->size==mask_bytes);
  for(auto name:{"adaln","bias_temp","gemv_partial"})assert(slot(baseline,name)->size==6144);
  assert((slot(baseline,"adaln_body_table")!=baseline.end())==table);
  if(table)assert(slot(baseline,"adaln_body_table")->size==196608);
  for(auto name:{"cross_k_bank","cross_v_bank"}){
   const auto found=slot(baseline,name);assert((found!=baseline.end())==(raw&&banks));
   if(found!=baseline.end())assert(found->size==16*bank_stride&&x.cross_kv_bank_stride()==bank_stride);
  }
  bool retained=false;const auto revised=x.planned_buffer_declarations(l,true,&retained);
  const auto after=detail::estimate_temporary_total(revised);
  std::cerr<<"long,"<<seq<<","<<raw<<","<<banks<<","<<table<<","<<before<<","<<after<<","<<retained<<"\n";
  assert(retained==table&&after<=before);
  assert(x.mask_schedule_for_layout(l,true)==(table?2:1));
  assert(x.mask_schedule_for_layout(l,false)==1);
  if(table){check_masks(revised,10);
   for(auto name:{"adaln","bias_temp","gemv_partial"}){const auto&d=*slot(revised,name);assert(d.phase_start==1&&d.phase_end==1);}
   assert(slot(revised,"sdpa_mask_image")->size==mask_bytes);
   // Omitting the third lifetime proof would exceed the original peak.
   auto insufficient=revised;auto traditional=slot(insufficient,"adaln");traditional->phase_start=0;traditional->phase_end=8;
   assert(detail::estimate_temporary_total(insufficient)>before);
  }else assert(slot(revised,"sdpa_mask_image")==revised.end()&&after==before);
  std::cout<<"long,"<<seq<<","<<raw<<","<<banks<<","<<table<<","<<before<<","<<after<<"\n";
 }
 for(int seq:{169,214}){
 Gr00tDiTModel exact;exact.denoise_mode_=exact.loop_mode_=true;exact.num_steps_=10;exact.vl_seq_=seq;
 exact.cross_kv_reuse_opt_in_=true;exact.adaln_table_.ready=true;
 LayoutContext l;l.chunk_size=41;l.batch_size=1;l.is_causal=false;l.physical_manifest_fingerprint=123;
 l.attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
 for(int bad=0;bad<13;++bad){auto x=exact;auto layout=l;switch(bad){
  case 0:x.adaln_table_.ready=false;break;case 1:x.num_steps_=5;break;case 2:x.vl_seq_=257;break;
  case 3:x.vl_seq_=0;break;case 4:x.num_steps_=3;break;case 5:x.z2_bound_=true;break;
  case 6:x.z2_prepared_seq_len_=seq;break;case 7:x.denoise_mode_=false;break;case 8:x.loop_mode_=false;break;
  case 9:layout.chunk_size=40;break;case 10:layout.batch_size=2;break;case 11:layout.is_causal=true;break;case 12:layout.use_attn_mask=true;break;}
  assert(x.mask_schedule_for_layout(layout,true)==1);bool retained=true;
  const auto revised=x.planned_buffer_declarations(layout,true,&retained);
  assert(!retained&&detail::estimate_temporary_total(revised)==detail::estimate_temporary_total(x.baseline_buffer_declarations(layout)));
  for(const auto&d:revised)assert(std::string(d.name)!="sdpa_mask_image");
 }
 // Three observations use fixed Graph positions with fresh ordinal source bytes.
 for(bool table:{false,true}){auto x=exact;x.adaln_table_.ready=table;x.record_mask(l);mask_memory.clear();
  for(int request=0;request<3;++request){x.prepared_cross_mask_.value=11+request*17;x.prepared_cross_mask_img_.value=22+request*17;
   const int before=mask_dmas;
   for(int body=0;body<10;++body)for(int layer=0;layer<32;layer+=2){x.context.body_iter=body;
    auto address=x.consume_mask(layer,{});const int value=layer%4?x.prepared_cross_mask_img_.value:x.prepared_cross_mask_.value;
    assert(mask_memory[address]==value);}
   const int uploads=table?2:x.num_steps_*16;assert(mask_dmas-before==uploads);
  }
 }
 exact.record_mask(l);exact.context.body_iter=0;
 for(size_t bad=0;bad<exact.context.arguments.size()+3;++bad){auto x=exact;
  if(bad<x.context.arguments.size())++x.context.arguments[bad];
  else if(bad==x.context.arguments.size())x.context.selector=1;
  else if(bad==x.context.arguments.size()+1)x.adaln_table_.ready=false;
  else x.vl_seq_=seq==169?214:169;
  const int before=mask_dmas;bool rejected=false;try{x.consume_mask(0,{});}catch(const std::runtime_error&){rejected=true;}
  assert(rejected&&mask_dmas==before);
 }
 }
 // Eligible long masks still need the original allocator's no-growth proof.
 // Exercise the 96->97 boundary, the last fitting mask and larger fallback.
 for(auto shape:{std::pair{4,97},std::pair{10,224},std::pair{4,225},std::pair{10,256},std::pair{4,193},std::pair{4,225},std::pair{4,289},std::pair{4,238},std::pair{4,270},std::pair{4,334}}){
  Gr00tDiTModel x;x.denoise_mode_=x.loop_mode_=true;x.num_steps_=shape.first;x.vl_seq_=shape.second;
  x.cross_kv_reuse_opt_in_=true;x.adaln_table_.ready=true;
  LayoutContext l;l.chunk_size=41;l.batch_size=1;l.is_causal=false;l.physical_manifest_fingerprint=123;
  l.attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
  const auto baseline=x.baseline_buffer_declarations(l);bool retained=false;
  const auto planned=x.planned_buffer_declarations(l,true,&retained);
  assert(detail::estimate_temporary_total(planned)<=detail::estimate_temporary_total(baseline));
  assert(retained==(shape.second<=224));
  if(retained)check_masks(planned,shape.first);
  assert(x.mask_schedule_for_layout(l,true)==(retained?2:1));
 }
}
'''
    return p


def _with_mask_consumer(program, source):
    """Add only shape state/device DMA doubles around both real consumer blocks."""
    body = _definition(source, "void build_layer_subgraph(")
    consume = _definition(body, "        if (is_first && ctx().body_iter == 0 &&")
    cross = _definition(source, "void build_cross_attn_block(")
    upload = cross[cross.index("        const bool image_mask ="):
                   cross.index("        const double attn_scale =")]
    site, = re.findall(r"constexpr int64_t GR00T_DIT_MASK_SCHEDULE_SITE = .*?;", source)
    state = site + r'''
enum class FmbRouteFamily{GRAPH_SCHEDULE};
struct ChunkInfo{int64_t len=41,idx=0;};
struct PreparedMask{int value=0;};
struct MaskContext{
 bool is_causal=false;std::optional<int> attention_mask;int64_t body_iter=0,seq_len=41,batch_size=1,selector=1;
 std::vector<int64_t>arguments;AttentionExecutionPolicy attention_policy=AttentionExecutionPolicy::SPM_KV_BY_MHA;
 struct Plan{struct Stage{std::vector<ChunkInfo>chunks{{}};}compute;}stage_plan;
 bool has_complete_physical_manifest()const{return true;}
 const MaskContext&physical_manifest()const{return *this;}
 const MaskContext&find_physical_route(FmbRouteFamily,int64_t site,int64_t invocation)const{assert(site==GR00T_DIT_MASK_SCHEDULE_SITE&&invocation==0);return *this;}
 void consume_physical_route(FmbRouteFamily,int64_t site,int64_t selected,int64_t flags,const std::vector<int64_t>&args,int64_t invocation)const{
  TORCH_CHECK(site==GR00T_DIT_MASK_SCHEDULE_SITE&&invocation==0&&flags==0&&selected==selector&&args==arguments);}
};
int mask_dmas=0;std::map<uint32_t,int>mask_memory;
void sdpa_dma_mask_to_spm(const PreparedMask&mask,uint32_t address,int64_t rows,int64_t seq,int cores){
 assert(rows==41&&(seq==169||seq==214)&&cores==8);mask_memory[address]=mask.value;++mask_dmas;}
'''
    program = program.replace("class Gr00tDiTModel{public:", state + "\nclass Gr00tDiTModel{public:", 1)
    fixture = r'''
 MaskContext context;MaskContext&ctx(){return context;}
 FmbForwardOperandResidency physical_forward_operand_residency(const MaskContext&)const{return FmbForwardOperandResidency::UNSPECIFIED;}
 PreparedMask prepared_cross_mask_,prepared_cross_mask_img_;
 struct Offset{uint32_t value;};Offset addr_offset(const char*name)const{return {std::string(name)=="sdpa_mask"?0x10000u:0x20000u};}
 void record_mask(const LayoutContext&layout){context.selector=mask_schedule_for_layout(layout,true);context.arguments=mask_schedule_arguments(41);context.attention_policy=layout.attention_policy;}
 uint32_t consume_mask(int layer_idx,const ChunkInfo&chunk){
  const bool is_first=layer_idx==0&&chunk.idx==0;const int64_t seq_len=chunk.len,S=vl_seq_;
__CONSUME__
__UPLOAD__
  return addr_offset(mask_slot).value;
 }
'''.replace("__CONSUME__", consume).replace("__UPLOAD__", upload)
    return program.replace(" SdpaConfig make_sdpa_config(", fixture + "\n SdpaConfig make_sdpa_config(", 1)


def test_public_two_mask_layout_with_real_allocator(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    cpp, binary = tmp_path / "gr00t_mask.cpp", tmp_path / "gr00t_mask"
    cpp.write_text(_program(long_profile=True))
    result = subprocess.run([compiler, "-std=c++17", "-O0", str(cpp), "-o", str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    print(result.stdout)
    rows = [line.split(",") for line in result.stdout.splitlines()]
    assert {tuple(row[:3]) for row in rows if row[0] != "long"} == {
        (denoise, raw, seq) for denoise in ("0", "1") for raw in ("0", "1") for seq in ("82", "96")}
    assert {tuple(row[1:5]) for row in rows if row[0] == "long"} == {
        (seq, raw, banks, table) for seq in ("169", "214")
        for raw in ("0", "1") for banks in ("0", "1") for table in ("0", "1")}


def test_adaln_scratch_dies_before_qk_projection_and_hooks_do_not_touch_it():
    source = SOURCE.read_text()
    modulation = _definition(source, "uint32_t adaln_layer_scale(")
    assert modulation.count('addr(0, "bias_temp")') == 1
    assert modulation.count('addr(0, "gemv_partial")') == 1
    table_branch = _definition(modulation, "        if (table)")
    assert 'addr(0, "adaln_body_table")' in table_branch
    assert 'addr(0, "adaln")' not in table_branch
    assert modulation.index("if (table)") < modulation.index("dit_adaln_gemv(")
    for name in ("build_self_attn_block", "build_cross_attn_block"):
        block = _definition(source, f"void {name}(")
        assert block.index("adaln_layer_scale(") < block.index('addr(0, "q")')
        assert 'addr(0, "bias_temp")' not in block
        assert 'addr(0, "gemv_partial")' not in block
        norm = block.index("rpu_launch_layernorm_spm_kernel(")
        norm_end = block.index(");", norm) + 2
        assert "scale, shift" in block[norm:norm_end]
        assert norm_end < block.index('addr(0, "q")')
        after_norm = re.sub(r"//[^\n]*|/\*[\s\S]*?\*/", "", block[norm_end:])
        assert re.search(r"\b(?:scale|shift)\b", after_norm) is None
        assert block.rindex('"q"') < block.index("finish_attn_and_ffn(")
        assert block.rindex('"k"') < block.index("finish_attn_and_ffn(")
    for name in ("emit_pre_layers_body", "emit_post_layers_body", "finish_attn_and_ffn"):
        block = _definition(source, f"void {name}(")
        for slot in ("q", "k", "adaln", "bias_temp", "gemv_partial"):
            assert f'"{slot}"' not in block
    cross = _definition(source, "void build_cross_attn_block(")
    assert "layer_idx % 4 != 0" in cross
    assert "prepared_cross_mask_img_" in cross
    assert cross.index("sdpa_dma_mask_to_spm(") < cross.index("rpu_launch_sdpa_by_mha_spm(")


def test_mask_schedule_is_descriptor_owned_and_initializes_both_graph_positions():
    source = SOURCE.read_text()
    declarations = _definition(source, "std::vector<BufferDecl> declare_buffers(")
    manifest = _definition(source, "FmbPhysicalExecutionManifest physical_manifest_for_attention_routes(")
    body = _definition(source, "void build_layer_subgraph(")
    cross = _definition(source, "void build_cross_attn_block(")
    assert "layout.physical_manifest_fingerprint != 0" in declarations
    assert "mask_schedule_for_layout(mask_layout, /*complete=*/true)" in manifest
    assert "mask_schedule_arguments(chunk.len)" in manifest and "mask_schedule_arguments(chunk.len)" in body
    assert body.index("GR00T_DIT_MASK_SCHEDULE_SITE") < body.index("build_cross_attn_block(")
    assert "ctx().body_iter == 0" in cross
    assert "layer_idx == first_cross_mask_layer(image_mask)" in cross
    assert '"sdpa_mask_image" : "sdpa_mask"' in cross
    assert "addr(0, mask_slot)" in cross and "addr_offset(mask_slot).value" in cross
    # Ordinal DDR ownership from batch2 remains the source of both Graph DMAs.
    prepare = _definition(source, "PreparedMask prep_mask_into(")
    assert "sdpa_stable_mask_cache(), ordinal" in prepare
    assert "slot = pm.mask_type == 4 ? pm.ddr_tensor" in prepare
    assert "copy_(" not in prepare
