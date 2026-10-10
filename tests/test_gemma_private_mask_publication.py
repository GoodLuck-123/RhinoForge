"""Gemma-private mask receipts; actual native logic with host-only SDK stores."""
from pathlib import Path
import ast

import torch

from test_siglip_attention_dry_live_contract import _function, _run_native

ROOT = Path(__file__).resolve().parents[1]


def _mask_publication_native_prefix():
    """Shared real ATen storage/copy/cache fixture; SDK visibility stays doubled."""
    sdpa = (ROOT / "src/ops/rpu_sdpa.cpp").read_text()
    declarations = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    copy_source = (ROOT / "src/core/rpu_tensor_ops.inc").read_text()
    # Reuse the existing ATen storage/device-tag transport double, not a second
    # implementation of publication or the receipt algorithm under test.
    old = ast.parse((ROOT / "tests/test_pi05_rope_positions.py").read_text())
    fn = next(n for n in old.body if isinstance(n, ast.FunctionDef)
              and n.name == "test_native_rope_cpu_dispatch_refreshes_stable_owners_and_rejects_bad_inputs")
    literal = next(n.value.value for n in fn.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "code" for t in n.targets))
    prefix = literal[:literal.index("struct Model{")]
    prefix = prefix.replace('#include <ATen/Parallel.h>', '#include <ATen/Parallel.h>\n#include <map>\n#include <mutex>\n#include <optional>')
    prefix = prefix.replace("struct HostOptions{at::TensorOptions value;c10::Device where;};", """struct HostOptions{
 at::TensorOptions value;c10::Device where;
 HostOptions device(c10::DeviceType d)const{return {value,c10::Device(d)};}
 auto dtype()const{return value.dtype();}
 struct DeviceTag{c10::Device where;auto type()const{return where.type();}};
};""")
    # The actual stable-slot key reads options.device(), so keep the real
    # TensorOptions key access and substitute only that device-tag expression.
    extra = r'''
 auto layout()const{return value.layout();}auto element_size()const{return value.element_size();}
 HostTensor select(int64_t d,int64_t i)const{auto r=*this;r.value=value.select(d,i);return r;}
 HostTensor slice(int64_t d,int64_t b,int64_t e)const{auto r=*this;r.value=value.slice(d,b,e);return r;}
 HostTensor contiguous()const{auto r=*this;r.value=value.contiguous();
  if(!r.value.is_same(value)){r.store.reset();r.override_span=false;}return r;}
 HostTensor to(at::ScalarType dtype)const{auto r=*this;r.value=value.to(dtype);
  if(!r.value.is_same(value)){r.store.reset();r.override_span=false;}return r;}
 HostTensor to(c10::DeviceType device)const{
  if(device==where.type())return *this;
  TORCH_CHECK(device==at::kCPU,"no extra device upload allowed");
  HostTensor r{at::empty(value.sizes(),value.options()),{},c10::Device(at::kCPU)};
  r.copy_(*this);return r;
 }
'''
    prefix = prefix.replace(" HostOptions options()const", extra + " HostOptions options()const")
    prefix += r'''
namespace at {
 HostTensor constant_pad_nd(const HostTensor& t,IntArrayRef pad,const Scalar& value){
  assert(t.device().is_cpu());return {at::constant_pad_nd(t.value,pad,value),{},c10::Device(at::kCPU)};
 }
}
'''
    copy = _function(copy_source, "at::Tensor &rpu_copy_impl(")
    preflight = copy[copy.index("{") + 1:copy.index("  const bool same_dtype")]
    same_device = _function(copy_source, "  if (same_device && same_dtype) {")
    start = copy_source.index("  // Case 3: Cross-device + Same dtype")
    cross = copy_source[start:copy_source.index("\n  return self;", start)]
    prefix = prefix.replace("__PREFLIGHT__", preflight).replace("__SAME_DEVICE__", same_device).replace("__CROSS_DEVICE__", cross)
    prepared = _function(declarations, "struct PreparedMask {") + ";\n"
    cache_decl = _function(declarations, "class SdpaStableMaskCache final {") + ";\n"
    start = sdpa.index("struct SdpaStableMaskCache::Impl {")
    end = sdpa.index("// CPU side: reduce dims", start)
    cache_impl = sdpa[start:end].replace("options.device().type()", "options.where.type()")
    builder = _function(sdpa, "PreparedMask sdpa_prepare_mask(")
    return prefix, prepared, cache_decl, cache_impl, builder


def test_private_chunk_mask_receipts_preserve_publication_and_alias_boundaries(tmp_path):
    """No board/full Graph claim: ATen bytes/views/versions, real native methods."""
    gemma = (ROOT / "src/fused/rpu_gemma_model.cpp").read_text()
    prefix, prepared, cache_decl, cache_impl, builder = _mask_publication_native_prefix()
    prefix = prefix.replace(" bool is_inference()const", " bool is_neg()const{return value.is_neg();}bool is_conj()const{return value.is_conj();}\n bool is_inference()const")
    normalize = _function(gemma, "static c10::optional<at::Tensor> gemma_model_prepare_chunk_mask(")
    receipt_start = gemma.index("    struct GemmaChunkMaskSpan {")
    receipt_end = gemma.index("    // ----- Model state -----", receipt_start)
    receipt_logic = gemma[receipt_start:receipt_end]
    weights = _function(gemma, "    void set_weights(")
    clear = next(line for line in weights.splitlines() if "chunk_mask_publications_.clear();" in line)
    code = prefix + prepared + cache_decl + cache_impl + normalize + builder + r'''
using GemmaMaskBuilderFn=PreparedMask (*)(const c10::optional<HostTensor>&,bool,int64_t,int64_t,SdpaStableMaskCache&,int64_t);
const GemmaMaskBuilderFn gemma_build_chunk_pm=sdpa_prepare_mask;
struct ChunkPlan{int64_t num_chunks,chunk_size;};
struct Context{int64_t position=0,seq_len=4;};
struct Model{
 Context context;uint64_t generation=1;SdpaStableMaskCache cache;
 std::vector<PreparedMask> chunk_masks_;std::optional<HostTensor> raw_mask_;
 Context& ctx(){return context;}
 uint64_t installed_model_state_generation()const{return generation;}
 SdpaStableMaskCache& sdpa_stable_mask_cache(){return cache;}
 void reinstall(bool fail=false){__CLEAR__ TORCH_CHECK(!fail,"failed weight validation");++generation;}
 __LOGIC__
};
int64_t version(const HostTensor& t){return t.value.is_inference()?-1:t.value._version();}
void raw_fill(const HostTensor& t,float a){for(int64_t i=0;i<t.numel();++i)t.data_ptr<c10::Half>()[i]=a;}
void raw_set(const HostTensor& t,int64_t row,int64_t col,float a){t.data_ptr<c10::Half>()[row*t.stride(0)+col*t.stride(1)]=a;}
void verify(const Model& model,const HostTensor& input,int64_t chunk,bool published){
 for(size_t i=0;i<model.chunk_masks_.size();++i){auto& slot=model.chunk_masks_[i].ddr_tensor;
  auto expected=gemma_model_prepare_chunk_mask(input,int64_t(i)*chunk,slot.size(0),input.size(-1));
  auto padded=at::constant_pad_nd(*expected,{0,slot.size(1)-input.size(-1)},-std::numeric_limits<float>::infinity());
  assert(model.chunk_masks_[i].mask_type==4&&std::memcmp(slot.data_ptr(),padded.data_ptr(),slot.nbytes())==0);
  if(published)assert(std::memcmp(slot.store->ddr.data(),padded.data_ptr(),slot.nbytes())==0);
 }
}
void feed(Model& m,const HostTensor& input,int64_t chunk=2){m.raw_mask_=input;
 m.context.seq_len=input.size(-2);m.context.position=input.size(-1)-input.size(-2);
 m.build_chunk_masks({(m.context.seq_len+chunk-1)/chunk,chunk});}
int main(){at::set_num_threads(1);
 // Same-shape ordinal slots stay distinct; steady hits do not copy, flush, or
 // fake an ATen version bump. Raw alias A/B/A must refresh changed values.
 for(bool inference:{false,true})for(bool normal:{false,true}){
  c10::InferenceMode mode(inference);g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=true;
  Model m;auto backing=make_tensor({6,17},at::kCPU),input=backing.narrow(0,1,4);raw_fill(backing,3);raw_set(input,2,0,4);
  const auto source_version=version(input);auto f=flushes.size(),c=copies;
  feed(m,input);assert(flushes.size()-f==2&&copies-c==2);verify(m,input,2,true);
  assert(m.chunk_masks_[0].ddr_tensor.data_ptr()!=m.chunk_masks_[1].ddr_tensor.data_ptr());
  auto owner0=m.chunk_masks_[0].ddr_tensor,owner1=m.chunk_masks_[1].ddr_tensor;
  auto v0=version(owner0),v1=version(owner1);f=flushes.size();c=copies;
  feed(m,input);assert(flushes.size()==f&&copies==c&&version(owner0)==v0&&version(owner1)==v1);
  assert(m.chunk_masks_[0].ddr_tensor.is_same(owner0)&&m.chunk_masks_[1].ddr_tensor.is_same(owner1));
  for(float value:{5.f,3.f}){raw_fill(backing,value);raw_set(input,2,0,value+1);f=flushes.size();c=copies;feed(m,input);
   assert(flushes.size()-f==2&&copies-c==2&&version(input)==source_version);verify(m,input,2,true);}
  // Full transformed-byte equality includes the -inf padding tail.
  raw_set(owner0,0,17,0);f=flushes.size();c=copies;feed(m,input);
  assert(flushes.size()-f==1&&copies-c==1);verify(m,input,2,true);
  // Signed zero is equal numerically but has different actual Half bytes.
  for(float zero:{-0.f,0.f}){raw_set(input,0,0,zero);f=flushes.size();feed(m,input);
   assert(flushes.size()-f==1);verify(m,input,2,true);}
  // A single changed first-chunk value does not republish the other ordinal.
  raw_set(input,0,0,9);f=flushes.size();c=copies;feed(m,input);
  assert(flushes.size()-f==1&&copies-c==1);verify(m,input,2,true);
  // device source always acquires and follows the original copy path, every time.
  auto resident=make_tensor({4,17},at::kPrivateUse1);raw_fill(resident,7);
  for(int j=0;j<2;++j){f=flushes.size();c=copies;feed(m,resident);
   assert(flushes.size()-f==4&&copies-c==4&&m.chunk_mask_publications_.empty());verify(m,resident,2,true);}
  f=flushes.size();c=copies;feed(m,input);assert(flushes.size()-f==2&&copies-c==2);
  // Force-off retains all copies but earns no publication receipt; next true
  // must republish even unchanged values. Normal mode has no authority.
  g_rpu_ddr_flush_force_enabled=false;f=flushes.size();c=copies;feed(m,input);
  assert(flushes.size()==f&&copies-c==2&&m.chunk_mask_publications_.empty());
  g_rpu_ddr_flush_force_enabled=true;f=flushes.size();c=copies;feed(m,input);
  assert(flushes.size()-f==2&&copies-c==2);verify(m,input,2,true);
  ++m.generation;f=flushes.size();c=copies;feed(m,input);assert(flushes.size()-f==2&&copies-c==2);
  bool caught=false;try{m.reinstall(true);}catch(const c10::Error&){caught=true;}
  assert(caught&&m.chunk_mask_publications_.empty());f=flushes.size();feed(m,input);assert(flushes.size()-f==2);
  m.reinstall();f=flushes.size();feed(m,input);assert(flushes.size()-f==2);
  // Failure after the second host write cannot leave the first receipt live.
  raw_fill(backing,11);flushes_until_failure=1;caught=false;
  try{feed(m,input);}catch(const c10::Error&){caught=true;}
  assert(caught&&m.chunk_mask_publications_.empty());f=flushes.size();feed(m,input);assert(flushes.size()-f==2);
  // Slot metadata drift rejects reuse of a previously held strong owner.
  m.chunk_mask_publications_[0].mask.ddr_tensor.value=owner0.value.narrow(1,0,16);
  f=flushes.size();feed(m,input);assert(flushes.size()-f==2);verify(m,input,2,true);
  // Different logical K sharing the padded geometry still republishes.
  auto changed=make_tensor({4,18},at::kCPU);raw_fill(changed,13);f=flushes.size();feed(m,changed);
  assert(flushes.size()-f==2);verify(m,changed,2,true);
  // Different rows/slot key gets the native cache's new owner, with no receipt
  // borrowing. Returned private masks retain original owners after refresh.
  auto retained=m.chunk_masks_[0].ddr_tensor;auto changed_rows=make_tensor({6,18},at::kCPU);raw_fill(changed_rows,15);
  f=flushes.size();feed(m,changed_rows,3);assert(flushes.size()-f==2);verify(m,changed_rows,3,true);
  assert(retained.size(0)==2&&m.chunk_masks_[0].ddr_tensor.size(0)==3&&!retained.is_same(m.chunk_masks_[0].ddr_tensor));
  // Wrong shape/slice fails through the actual normalizer and clears receipts.
  m.raw_mask_=input;m.context={0,9};caught=false;try{m.build_chunk_masks({2,5});}catch(const c10::Error&){caught=true;}
  assert(caught&&m.chunk_mask_publications_.empty());
 }
 // Full source backing, not just its active logical range, controls eligibility.
 // Expose a CPU alias of one private slot with ATen from_blob; changing it must
 // never preserve a stale published-value receipt for any ordinal.
 {Model m;g_rpu_ddr_flush_enabled=false;g_rpu_ddr_flush_force_enabled=true;
  auto input=make_tensor({4,16},at::kCPU);raw_fill(input,2);feed(m,input);
  auto slot=m.chunk_masks_[0].ddr_tensor;auto alias=HostTensor{at::from_blob(slot.data_ptr(),{2,16},at::kHalf),{},c10::Device(at::kCPU)};
  raw_fill(alias,21);auto f=flushes.size();feed(m,alias);assert(flushes.size()-f==1&&m.chunk_mask_publications_.empty());verify(m,alias,2,true);
  // Actual ATen from_blob backing spans the valid active CPU input and a
  // private slot in its unused range. Only active input bytes are accessed;
  // full backing still must reject a publication receipt.
  feed(m,input);auto current_slot=m.chunk_masks_[0].ddr_tensor;
  const auto input_ptr=reinterpret_cast<uintptr_t>(input.data_ptr());
  const auto slot_ptr=reinterpret_cast<uintptr_t>(current_slot.data_ptr());
  const auto begin=std::min(input_ptr,slot_ptr);
  const auto end=std::max(input_ptr+input.nbytes(),slot_ptr+current_slot.nbytes());
  auto full=at::from_blob(reinterpret_cast<void*>(begin),{int64_t((end-begin)/2)},at::kHalf);
  auto view=full.narrow(0,int64_t((input_ptr-begin)/2),input.numel()).reshape(input.value.sizes());
  auto wide=HostTensor{view,{},c10::Device(at::kCPU)};
  f=flushes.size();feed(m,wide);assert(flushes.size()-f==2&&m.chunk_mask_publications_.empty());
  // The uintptr-overflow metadata guard uses the existing storage-tag double;
  // source values and the original copy transport remain real ATen.
  auto overflow=input;overflow.override_span=true;overflow.storage_base=std::numeric_limits<uintptr_t>::max()-1;overflow.storage_bytes=64;
  f=flushes.size();feed(m,overflow);assert(flushes.size()-f==2&&m.chunk_mask_publications_.empty());
  m.raw_mask_.reset();m.build_chunk_masks({0,2});assert(m.chunk_mask_publications_.empty()&&m.chunk_masks_.empty());
 }
 // Original conversion and padded-byte behavior covers FP32 and a strided
 // public CPU mask. Both semantic values and the FP16 padding bits matter.
 {Model m;auto input=HostTensor{at::arange(4*34,at::kFloat).reshape({4,34}).slice(1,0,34,2),{},c10::Device(at::kCPU)};
  auto f=flushes.size();feed(m,input);assert(flushes.size()-f==2);verify(m,input,2,true);
  f=flushes.size();auto c=copies;feed(m,input);assert(flushes.size()==f&&copies==c);
  input.value.data_ptr<float>()[0]=99;f=flushes.size();feed(m,input);assert(flushes.size()-f==1);verify(m,input,2,true);
 }
 // The cache hit must not suppress ATen's original inference-owner copy_ error
 // on returning to ordinary mode. Re-entering inference can publish and hit.
 {auto m=std::make_unique<Model>();auto input=make_tensor({4,17},at::kCPU);raw_fill(input,3);
  {c10::InferenceMode mode(true);feed(*m,input);auto f=flushes.size();feed(*m,input);assert(flushes.size()==f);}
  bool caught=false;try{feed(*m,input);}catch(const c10::Error&){caught=true;}
  assert(caught&&m->chunk_mask_publications_.empty());
  {c10::InferenceMode mode(true);auto f=flushes.size();feed(*m,input);assert(flushes.size()-f==2);
   f=flushes.size();feed(*m,input);assert(flushes.size()==f);}
 }
 // A lazy negative view preserves the original Half storage bytes but changes
 // logical values. K17 uses the original ATen CPU pad to materialize those
 // values and its -inf tail before the actual transport publication.
 for(bool inference:{false,true}){
  c10::InferenceMode mode(inference);Model m;
  g_rpu_ddr_flush_enabled=false;g_rpu_ddr_flush_force_enabled=true;
  auto a=make_tensor({4,17},at::kCPU);raw_fill(a,0);raw_set(a,0,0,-0.5f);raw_set(a,2,0,-0.25f);
  std::vector<char> bytes(a.nbytes());std::memcpy(bytes.data(),a.data_ptr(),a.nbytes());
  const auto source_version=version(a);feed(m,a);verify(m,a,2,true);
  auto negative=HostTensor{at::_neg_view(a.value),{},c10::Device(at::kCPU)};
  assert(negative.is_neg()&&!negative.is_conj()&&negative.is_contiguous());
  assert(negative.data_ptr()==a.data_ptr()&&version(negative)==source_version);
  auto normalized=gemma_model_prepare_chunk_mask(negative,0,2,17);
  assert(normalized->is_neg()&&normalized->data_ptr()==a.data_ptr());
  auto semantic=at::constant_pad_nd(*normalized,{0,15},-std::numeric_limits<float>::infinity());
  assert(!semantic.is_neg()&&float(semantic.data_ptr<c10::Half>()[0])==0.5f);
  for(int repeat=0;repeat<2;++repeat){auto f=flushes.size(),c=copies;feed(m,negative);
   assert(flushes.size()-f==2&&copies-c==2&&m.chunk_mask_publications_.empty());
   verify(m,negative,2,true);assert(float(m.chunk_masks_[0].ddr_tensor.data_ptr<c10::Half>()[0])==0.5f);
   assert(std::memcmp(bytes.data(),a.data_ptr(),a.nbytes())==0&&version(a)==source_version);}
  // A->negative(A)->A restores through the original writer, then ordinary A
  // retains its steady hit. Force-off negative input grants no authority.
  auto f=flushes.size(),c=copies;feed(m,a);assert(flushes.size()-f==2&&copies-c==2);verify(m,a,2,true);
  f=flushes.size();c=copies;feed(m,a);assert(flushes.size()==f&&copies==c);
  g_rpu_ddr_flush_force_enabled=false;f=flushes.size();c=copies;feed(m,negative);
  assert(flushes.size()==f&&copies-c==2&&m.chunk_mask_publications_.empty());verify(m,negative,2,false);
  g_rpu_ddr_flush_force_enabled=true;f=flushes.size();c=copies;feed(m,a);
  assert(flushes.size()-f==2&&copies-c==2);verify(m,a,2,true);
 }
 std::cout<<"private Gemma mask authority/ordinal/alias/publication controls passed\n";
}
'''
    tail = code[len(prefix + prepared + cache_decl + cache_impl + normalize + builder):]
    tail = tail.replace("__CLEAR__", clear).replace("__LOGIC__", receipt_logic)
    # Device/options substitutions cover only the extracted native blocks and
    # model wrapper. Real ATen storage/allocation remains the existing double.
    native = prepared + cache_decl + cache_impl + normalize + builder + tail
    native = native.replace("at::TensorOptions", "HostOptions").replace("at::Tensor", "HostTensor")
    code = prefix + native
    _run_native(tmp_path, code, torch)
