"""Exercise the actual Action admission/publication block with host SDK doubles."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_action_allocation_tail_publication_and_deferred_live_inputs(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    from test_siglip_attention_dry_live_contract import _function, _run_native

    source = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    body = _function(source, "at::Tensor Qwen3_5Model::forward_action_step_impl(")
    reference_header = (ROOT / "src/fused/rpu_qwen3_5_model.h").read_text()
    trace_struct = _function(reference_header, "    struct ActionRtcTraceOutputs {") + ";"
    # ATen CPU storage substitutes for device allocations; every actual shape,
    # dtype, contiguity, pointer, cache-capacity and modulation guard remains.
    # Device and allocation boundaries alone are host doubles. Public device
    # cases keep CPU storage labeled device; marked CPU inputs execute the real
    # copy/publication/keep-alive code instead of bypassing it.
    body = body.replace("action.device().is_cpu()", "host_cpu_input(action)")
    body = body.replace("action_prefix->device().is_cpu()", "host_cpu_input(*action_prefix)")
    body = body.replace("at::kPrivateUse1", "at::kCPU")
    body = body.replace("at::empty(action.sizes(),\n"
                        "            at::TensorOptions().dtype(at::kHalf).device(action_out.device()))",
                        "private_bank(action.sizes())")
    body = body.replace("at::empty(action_prefix->sizes(),\n"
                        "            at::TensorOptions().dtype(at::kHalf).device(action_out.device()))",
                        "private_bank(action_prefix->sizes())")
    body = body.replace("at::Tensor Qwen3_5Model::forward_action_step_impl(",
                        "at::Tensor forward_action_step_impl(")
    # A host logical-completion double executes the actual AArch64 branch.
    # This does not test the physical SDK barrier or any device kernel math.
    body = body.replace("#if defined(__aarch64__)", "#if 1")
    body = body.replace('__asm__ volatile("dsb sy" ::: "memory");',
                        "system_barrier();")
    cpp = r'''
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include "core/rpu_profile.h"
#include <array>
#include <cassert>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <vector>
bool g_rpu_ddr_flush_enabled=false,g_rpu_ddr_flush_force_enabled=true;
struct Owner {
 at::Tensor backing;std::vector<uint8_t> ddr;
 c10::weak_intrusive_ptr<c10::TensorImpl,c10::UndefinedTensorImpl> view;
 uintptr_t reported_base;size_t reported_size;bool lookup=true,private_copy=false;size_t logical_bytes=0;
 Owner(at::Tensor b,const at::Tensor& v):backing(std::move(b)),ddr(backing.nbytes(),0xa7),
  view(v.getIntrusivePtr()),reported_base(reinterpret_cast<uintptr_t>(backing.data_ptr())),
  reported_size(backing.nbytes()){}
 void* get_cpu_ptr(){return reinterpret_cast<void*>(reported_base);}
 size_t get_memory_size(){return reported_size;}
};
std::vector<std::unique_ptr<Owner>> owners;
struct Range {void* ptr;size_t bytes;};
std::vector<std::vector<Range>> batches;std::vector<Range> scalars,sized_scalars;
std::vector<c10::weak_intrusive_ptr<c10::TensorImpl,c10::UndefinedTensorImpl>> cpu_inputs;
bool host_cpu_input(const at::Tensor& t){
 for(auto& weak:cpu_inputs){auto live=weak.lock();if(live&&live.get()==t.unsafeGetTensorImpl())return true;}return false;
}
void mark_cpu(const at::Tensor& t){cpu_inputs.emplace_back(t.getIntrusivePtr());}
int private_fault=-1,private_fault_target=0,private_allocations=0;
bool large_private_pool=false;
struct RpuKernelGraph {
 std::vector<at::Tensor> leases;
 static bool has_active(){return true;}
 static RpuKernelGraph& active(){static RpuKernelGraph graph;return graph;}
 void keep_alive(const at::Tensor& t){leases.push_back(t);}
};
int batch_completions=0,system_completions=0,emits=0;
Owner& locate(void* pointer,size_t& offset){
 const uintptr_t a=reinterpret_cast<uintptr_t>(pointer);
 for(auto i=owners.rbegin();i!=owners.rend();++i){auto& o=**i;
  const uintptr_t b=reinterpret_cast<uintptr_t>(o.backing.data_ptr());
  if(a>=b&&a-b<o.ddr.size()&&!o.view.expired()){offset=a-b;return o;}}
 TORCH_CHECK(false,"lost endpoint owner");
}
at::Tensor bank(at::IntArrayRef shape,int offset=32,bool pool=false){
 int64_t count=1;for(auto n:shape)count*=n;
 const int64_t storage=pool?1024*1024:((count+offset+64+31)/32)*32;
 auto flat=at::zeros({storage},at::kHalf);
 auto t=flat.narrow(0,offset,count).view(shape);
 owners.emplace_back(std::make_unique<Owner>(flat,t));return t;
}
at::Tensor private_bank(at::IntArrayRef shape){
 const int fault=private_allocations++==private_fault_target?private_fault:-1;
 auto t=bank(shape,(fault==6||fault==7)?33:32,large_private_pool);
 size_t off;auto& o=locate(t.data_ptr(),off);o.private_copy=true;o.logical_bytes=t.nbytes();
 if(fault==0)o.lookup=false;
 if(fault==1)o.reported_size=0;
 if(fault==2)o.reported_base=reinterpret_cast<uintptr_t>(t.data_ptr())+64;
 if(fault==3)o.reported_size=std::numeric_limits<size_t>::max();
 if(fault==4)o.reported_size=off+t.nbytes()-1;
 if(fault==5){o.reported_base=0;o.reported_size=reinterpret_cast<uintptr_t>(t.data_ptr())+t.nbytes()+64;}
 if(fault==6)o.reported_size=off+t.nbytes(); // logical fits, rounded end does not
 if(fault==7){o.reported_base=reinterpret_cast<uintptr_t>(t.data_ptr());o.reported_size=t.nbytes()+64;}
 if(fault==8){o.reported_base=std::numeric_limits<uintptr_t>::max()-31;o.reported_size=64;}
 return t;
}
void system_barrier(){++system_completions;}
namespace rhino_lkn {
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uint64_t>(p);}
Owner* RpuGetHostddr(void* p){size_t off;auto& o=locate(p,off);return o.lookup?&o:nullptr;}
void publish(void* p,size_t bytes){size_t off;auto& o=locate(p,off);
 assert(bytes==(o.private_copy?o.logical_bytes:o.ddr.size()-off));
 std::memcpy(o.ddr.data()+off,p,bytes);
}
void RpuDdrFlush(void* p){size_t off;auto& o=locate(p,off);
 scalars.push_back({p,o.ddr.size()-off});publish(p,o.ddr.size()-off);
}
void RpuDdrFlush(void* p,size_t bytes){sized_scalars.push_back({p,bytes});publish(p,bytes);}
void RpuDdrFlushBatch(void** pointers,size_t* sizes,size_t count){
 assert(count>0&&count<=10);std::vector<Range> packet;
 for(size_t i=0;i<count;++i){packet.push_back({pointers[i],sizes[i]});publish(pointers[i],sizes[i]);}
 batches.push_back(packet);++batch_completions;
}
}
__TRACE_STRUCT__
std::vector<std::function<void()>> nodes;
std::vector<std::vector<uint8_t>> seen;
bool graph_built=false;
struct Qwen3_5Model {
 bool action_mode_=true,action_step_active_=false,action_loop_active_=false;
 bool action_rtc_active_=false,action_trace_active_=false,action_euler_scale_pinned_=false;
 bool fast_replay_enabled_=true,fast_replay_active_=false;
 int64_t action_num_steps_=1,action_rtc_prefix_len_=0,action_len_=32,action_dim_pad_=16;
 c10::Half action_euler_scale_=c10::Half(0);
 std::vector<int64_t> action_prefix_lens_;
 at::Tensor action_input_w_=at::ones({1},at::kHalf),prefill_cos_=at::ones({64,16},at::kHalf),prefill_sin_=at::zeros({64,16},at::kHalf);
 at::Tensor action_hidden_stage_,action_velocity_stage_;
 at::Tensor adaptive_mod_ref_,action_input_ref_,action_keep_mask_ref_,action_output_ref_,action_rtc_prefix_ref_;
 at::Tensor action_trace_pre_action_ref_,action_trace_action_embed_ref_,action_trace_final_hidden_ref_;
 at::Tensor action_trace_velocity_ref_,action_trace_post_action_ref_;
 uint64_t adaptive_mod_live_base_=0,action_input_live_base_=0,action_keep_mask_live_base_=0,action_output_live_base_=0;
 uint64_t action_rtc_prefix_live_base_=0,action_trace_pre_action_live_base_=0,action_trace_action_embed_live_base_=0;
 uint64_t action_trace_final_hidden_live_base_=0,action_trace_velocity_live_base_=0,action_trace_post_action_live_base_=0;
 int64_t hidden_size()const{return 32;}int64_t num_layers()const{return 2;}
 void invalidate_model_state(bool){graph_built=false;nodes.clear();}
 at::Tensor run_all_layers(at::Tensor&,std::vector<at::Tensor>&,std::vector<at::Tensor>&,
  std::nullopt_t,int64_t,bool,int64_t,at::IntArrayRef){
  if(!graph_built){
   std::vector<uint64_t*> bases={&adaptive_mod_live_base_,&action_input_live_base_,&action_keep_mask_live_base_,&action_output_live_base_};
   if(action_rtc_active_)bases.push_back(&action_rtc_prefix_live_base_);
   if(action_trace_active_){for(auto b:{&action_trace_pre_action_live_base_,&action_trace_action_embed_live_base_,
    &action_trace_final_hidden_live_base_,&action_trace_velocity_live_base_,&action_trace_post_action_live_base_})bases.push_back(b);}
   for(auto base:bases)nodes.push_back([base]{size_t off;auto& o=locate(reinterpret_cast<void*>(*base),off);
    const size_t bytes=o.private_copy?o.logical_bytes:o.ddr.size()-off;
    seen.emplace_back(o.ddr.begin()+off,o.ddr.begin()+off+bytes);});
   graph_built=true;++emits;
  }
  return action_velocity_stage_;
 }
 __BODY__
};
struct Inputs {
 at::Tensor action,mask,out,mod,prefix;
 at::Tensor trace_pre_action,trace_action_embed,trace_final_hidden,trace_velocity,trace_post_action;
 ActionRtcTraceOutputs trace{trace_pre_action,trace_action_embed,trace_final_hidden,trace_velocity,trace_post_action};
 Inputs(bool rtc,bool traced,int steps,int width=16,int prefix_len=7){
  action=bank({1,32,width});mask=bank({width});out=bank({1,32,width});
  mod=rtc?bank({steps,5,2,96}):(steps==1?bank({5,96}):bank({steps,5,96}));
  if(rtc)prefix=bank({1,prefix_len,width});
  if(traced){trace_pre_action=bank({10,1,32,16});trace_action_embed=bank({10,1,32,32});
   trace_final_hidden=bank({10,1,32,32});trace_velocity=bank({10,1,32,16});trace_post_action=bank({10,1,32,16});}
 }
 void drop(){action=at::Tensor{};mask=at::Tensor{};out=at::Tensor{};mod=at::Tensor{};prefix=at::Tensor{};
  trace_pre_action=at::Tensor{};trace_action_embed=at::Tensor{};trace_final_hidden=at::Tensor{};
  trace_velocity=at::Tensor{};trace_post_action=at::Tensor{};}
 std::vector<at::Tensor> all(bool rtc,bool traced){
  std::vector<at::Tensor> result{mod,action,mask,out};if(rtc)result.push_back(prefix);
  if(traced)for(auto t:{trace.pre_action,trace.action_embed,trace.final_hidden,trace.velocity,trace.post_action})result.push_back(t);
  return result;
 }
};
void call(Qwen3_5Model& model,Inputs& in,bool rtc,bool traced,int steps,bool short_cache=false,int prefix_len=7,bool empty_descriptor=false){
 std::vector<at::Tensor> k(2,at::zeros({1,short_cache?1:8,1,1,1,1,1},at::kHalf)),v=k;
 const std::array<int64_t,2> prefix_lens={1,1};
 const std::vector<int64_t> descriptor=empty_descriptor?std::vector<int64_t>{}:std::vector<int64_t>{1};
 model.forward_action_step_impl(in.action,in.mask,in.out,rtc?&in.prefix:nullptr,rtc?prefix_len:0,
  traced?&in.trace:nullptr,0.1,in.mod,k,v,prefix_lens,steps,descriptor);
}
void reset_observed(){batches.clear();scalars.clear();sized_scalars.clear();seen.clear();batch_completions=system_completions=0;}
void dirty(const at::Tensor& tensor,uint8_t bits){size_t off;auto& o=locate(tensor.data_ptr(),off);
 std::memset(tensor.data_ptr(),bits,o.ddr.size()-off);
}
void execute(const std::vector<at::Tensor>& all,bool force){
 for(auto& node:nodes)node();assert(seen.size()==all.size());
 for(size_t i=0;i<all.size();++i){size_t off;auto& o=locate(all[i].data_ptr(),off);
  const size_t bytes=o.private_copy?o.logical_bytes:o.ddr.size()-off;
  assert(seen[i].size()==bytes);
  assert(std::vector<uint8_t>(o.ddr.begin(),o.ddr.begin()+off)==std::vector<uint8_t>(off,0xa7));
  if(o.private_copy)assert(std::vector<uint8_t>(o.ddr.begin()+off+bytes,o.ddr.end())==std::vector<uint8_t>(o.ddr.size()-off-bytes,0xa7));
  if(force)assert(std::memcmp(seen[i].data(),all[i].data_ptr(),seen[i].size())==0);
  else assert(seen[i]==std::vector<uint8_t>(seen[i].size(),0xa7));
 }
}
template<class F>void rejects(F fn){bool bad=false;try{fn();}catch(const c10::Error&){bad=true;}assert(bad);}
int main(){
 int cases=0;
 for(bool inference:{false,true}){c10::InferenceMode inference_mode(inference);
  for(auto flags:std::initializer_list<std::array<int,3>>{{0,0,1},{0,0,10},{1,0,10},{1,1,10}}){
   const bool rtc=flags[0],traced=flags[1];const int steps=flags[2];Qwen3_5Model model;
   Inputs in(rtc,traced,steps);const int before=emits;
   for(uint8_t bits:{uint8_t(1),uint8_t(2),uint8_t(1)}){
    auto all=in.all(rtc,traced);std::vector<int64_t> versions;
    for(auto& t:all){versions.push_back(t.is_inference()?-1:t._version());dirty(t,bits);}
    reset_observed();call(model,in,rtc,traced,steps);
    assert(batches.size()==1&&scalars.empty()&&batches[0].size()==all.size());
    assert(batch_completions==1&&system_completions==1);
    for(size_t i=0;i<all.size();++i){size_t off;auto& o=locate(all[i].data_ptr(),off);
     assert(batches[0][i].ptr==all[i].data_ptr()&&batches[0][i].bytes==o.ddr.size()-off);
     assert(batches[0][i].bytes>all[i].nbytes());
     assert((all[i].is_inference()?-1:all[i]._version())==versions[i]);}
    execute(all,true);
   }
   assert(emits==before+1);++cases;
  }
 }
 // Force-off preserves the original no-publication policy and old DDR bytes.
 {Qwen3_5Model model;Inputs in(true,true,10);auto all=in.all(true,true);
  for(auto& t:all)dirty(t,3);g_rpu_ddr_flush_force_enabled=false;reset_observed();
  call(model,in,true,true,10);assert(batches.empty()&&scalars.empty()&&system_completions==0);
  execute(all,false);g_rpu_ddr_flush_force_enabled=true;++cases;}
 // One failed owner proof must retain unsized fallback; good owners still batch.
 for(int bad=0;bad<6;++bad){Qwen3_5Model model;Inputs in(false,false,10);auto all=in.all(false,false);
  for(auto& t:all)dirty(t,4);size_t off;auto& broken=locate(in.mod.data_ptr(),off);
  if(bad==0)broken.lookup=false;
  if(bad==1)broken.reported_size=0;
  if(bad==2)broken.reported_base=reinterpret_cast<uintptr_t>(in.mod.data_ptr())+64;
  if(bad==3)broken.reported_size=std::numeric_limits<size_t>::max();
  if(bad==4)broken.reported_size=off;
  if(bad==5)broken.reported_size=std::numeric_limits<uintptr_t>::max()-broken.reported_base-31;
  reset_observed();call(model,in,false,false,10);
  assert(scalars.size()==1&&scalars[0].ptr==in.mod.data_ptr());
  assert(batches.size()==1&&batches[0].size()==3&&system_completions==1);execute(all,true);++cases;
 }
 {Qwen3_5Model model;Inputs in(false,false,10);auto all=in.all(false,false);
  for(auto& t:all){dirty(t,5);size_t off;locate(t.data_ptr(),off).lookup=false;}
  reset_observed();call(model,in,false,false,10);assert(batches.empty()&&scalars.size()==4&&system_completions==0);
  execute(all,true);++cases;}
 // Public offset views stay legal. Noncontiguous/wrong-dtype/aliased/short-cache
 // requests fail in the real admission block before any publication or DMA.
 for(int bad=0;bad<5;++bad){Qwen3_5Model model;Inputs in(false,false,10);reset_observed();
  if(bad==0)in.action=at::empty_strided({1,32,16},{512,1,32},at::kHalf);
  if(bad==1)in.mask=at::zeros({16},at::kFloat);
  if(bad==2)in.out=in.action;
  if(bad==3)in.mod=at::empty_strided({10,5,96},{480,1,5},at::kHalf);
  rejects([&]{call(model,in,false,false,10,bad==4);});assert(batches.empty()&&scalars.empty()&&nodes.empty());++cases;
 }
 // RTC/trace invalid admission must also fail before publishing any input.
 for(int bad=0;bad<2;++bad){Qwen3_5Model model;Inputs in(true,true,10);reset_observed();
  if(bad==0)in.prefix=at::zeros({1,6,16},at::kHalf);
  else in.trace.final_hidden=in.trace.action_embed;
  rejects([&]{call(model,in,true,true,10);});assert(batches.empty()&&scalars.empty()&&nodes.empty());++cases;
 }
 // Each forward can replace source addresses. The old saved output and caller
 // owners remain intact; model refs keep fresh arguments alive through END.
 {Qwen3_5Model model;const int before=emits;std::vector<at::Tensor> saved;
  for(uint8_t bits:{uint8_t(6),uint8_t(7),uint8_t(6)}){
   Inputs in(false,false,10);auto all=in.all(false,false);for(auto& t:all)dirty(t,bits);
   reset_observed();call(model,in,false,false,10);saved.push_back(in.out);
   auto* p=in.action.data_ptr();size_t off;auto& live=locate(p,off);
   in.drop();all.clear();assert(!live.view.expired());
   for(auto& node:nodes)node();assert(seen.size()==4);
   for(auto& row:seen)assert(row==std::vector<uint8_t>(row.size(),bits));
  }
  assert(emits==before+1&&saved[0].data_ptr()!=saved[1].data_ptr());
  assert(std::memcmp(saved[0].data_ptr(),saved[2].data_ptr(),saved[0].nbytes())==0);
  c10::weak_intrusive_ptr<c10::TensorImpl,c10::UndefinedTensorImpl> lease{model.action_input_ref_.getIntrusivePtr()};
  assert(!lease.expired());model=Qwen3_5Model{};nodes.clear();assert(lease.expired());
  ++cases;
 }

 // The new private route runs the same extracted function. Explicit CPU labels
 // distinguish CPU sources from ATen-backed public-device device doubles.
 auto bound=[&](Qwen3_5Model& model,bool rtc){
  std::vector<at::Tensor> result{model.adaptive_mod_ref_,model.action_input_ref_,
   model.action_keep_mask_ref_,model.action_output_ref_};
  if(rtc)result.push_back(model.action_rtc_prefix_ref_);return result;
 };
 auto fresh=[&](Inputs& in,bool cpu_action,bool cpu_prefix,uint8_t bits){
  cpu_inputs.clear();
  for(auto& t:in.all(cpu_prefix||in.prefix.defined(),false))dirty(t,bits);
  if(cpu_action){in.action=at::full({1,32,32},-0.5f*bits,at::kHalf);mark_cpu(in.action);}
  if(cpu_prefix){in.prefix=at::full(in.prefix.sizes(),0.25f*bits,at::kHalf);mark_cpu(in.prefix);}
 };
 // CPU action only/non-RTC, both RTC mixed directions, and both CPU sources.
 // Exercise actual half bytes, fresh addresses, 2MiB allocator tails, A/B/A
 // delayed reads and retention of each private source after model-ref overwrite.
 for(bool inference:{false,true}){c10::InferenceMode inference_mode(inference);
  for(int mode=0;mode<4;++mode){
   const bool rtc=mode!=0,cpu_action=mode!=2,cpu_prefix=mode>=2;
   Qwen3_5Model model;model.action_dim_pad_=32;const int before=emits;
   RpuKernelGraph::active().leases.clear();cpu_inputs.clear();large_private_pool=true;
   std::vector<at::Tensor> saved;std::vector<c10::weak_intrusive_ptr<c10::TensorImpl,c10::UndefinedTensorImpl>> previous;
   for(uint8_t bits:{uint8_t(6),uint8_t(7),uint8_t(6)}){
    Inputs in(rtc,false,10,32,5);fresh(in,cpu_action,cpu_prefix,bits);
    const auto action_bytes=in.action.clone(),prefix_bytes=rtc?in.prefix.clone():at::Tensor{};
    reset_observed();private_allocations=0;call(model,in,rtc,false,10,false,5);
    assert(batches.size()==1&&scalars.empty()&&sized_scalars.empty()&&system_completions==1);
    auto all=bound(model,rtc);assert(batches[0].size()==all.size());
    for(size_t i=0;i<all.size();++i){size_t off;auto& o=locate(all[i].data_ptr(),off);
     const size_t bytes=o.private_copy?all[i].nbytes():o.ddr.size()-off;
     assert(batches[0][i].ptr==all[i].data_ptr()&&batches[0][i].bytes==bytes);
     if(o.private_copy){assert(o.ddr.size()==2*1024*1024);assert(bytes==(i==1?2048:320));}}
    assert(at::equal(model.action_input_ref_,action_bytes));
    if(!cpu_action)assert(model.action_input_ref_.data_ptr()==in.action.data_ptr());
    if(rtc){assert(at::equal(model.action_rtc_prefix_ref_,prefix_bytes));
     if(!cpu_prefix)assert(model.action_rtc_prefix_ref_.data_ptr()==in.prefix.data_ptr());}
    if(cpu_action){assert(model.action_input_ref_.data_ptr()!=in.action.data_ptr());previous.emplace_back(model.action_input_ref_.getIntrusivePtr());}
    if(cpu_prefix){assert(model.action_rtc_prefix_ref_.data_ptr()!=in.prefix.data_ptr());previous.emplace_back(model.action_rtc_prefix_ref_.getIntrusivePtr());}
    saved.push_back(in.out);in.drop();execute(all,true);
    for(auto& lease:previous)assert(!lease.expired());
   }
   assert(emits==before+1&&saved[0].data_ptr()!=saved[1].data_ptr());
   assert(at::equal(saved[0],saved[2]));
   model=Qwen3_5Model{};nodes.clear();for(auto& lease:previous)assert(!lease.expired());
   RpuKernelGraph::active().leases.clear();for(auto& lease:previous)assert(lease.expired());
   ++cases;
  }
 }
 large_private_pool=false;cpu_inputs.clear();
 // Every private owner-bound failure uses the logical sized fallback, while
 // all other public full tails and the other private range remain in Batch.
 for(int target=0;target<2;++target)for(int bad=0;bad<9;++bad){
  Qwen3_5Model model;model.action_dim_pad_=32;Inputs in(true,false,10,32,5);
  fresh(in,true,true,4);RpuKernelGraph::active().leases.clear();reset_observed();
  private_allocations=0;private_fault=bad;private_fault_target=target;
  call(model,in,true,false,10,false,5);private_fault=-1;
  auto all=bound(model,true);assert(scalars.empty()&&sized_scalars.size()==1&&batches.size()==1&&batches[0].size()==4);
  assert(sized_scalars[0].ptr==all[target?4:1].data_ptr());
  assert(sized_scalars[0].bytes==(target?320:2048)&&system_completions==1);execute(all,true);++cases;
 }
 // Force OFF copies all current CPU bytes but keeps DDR publication disabled;
 // the CPU-copy system barrier and leases still protect eventual graph END.
 {Qwen3_5Model model;model.action_dim_pad_=32;Inputs in(true,false,10,32,5);
  fresh(in,true,true,5);reset_observed();RpuKernelGraph::active().leases.clear();
  g_rpu_ddr_flush_force_enabled=false;private_allocations=0;call(model,in,true,false,10,false,5);
  assert(batches.empty()&&scalars.empty()&&sized_scalars.empty()&&system_completions==1);
  assert(at::equal(model.action_input_ref_,in.action)&&at::equal(model.action_rtc_prefix_ref_,in.prefix));
  execute(bound(model,true),false);g_rpu_ddr_flush_force_enabled=true;++cases;}
 // Real ATen lazy neg views, grad, stride/dtype/shape and unsupported CPU mode
 // are rejected before allocation, publication, graph capture or new leases.
 for(int target=0;target<2;++target)for(int bad=0;bad<8;++bad){
  cpu_inputs.clear();Qwen3_5Model model;model.action_dim_pad_=32;
  Inputs in(true,false,10,32,5);fresh(in,true,true,2);
  at::Tensor& t=target?in.prefix:in.action;
  if(bad==0)t=at::_neg_view(t);
  if(bad==1)t.set_requires_grad(true);
  if(bad==2)t=at::empty_strided(t.sizes(),target?at::IntArrayRef({160,1,5}):at::IntArrayRef({1024,1,32}),at::kHalf);
  if(bad==3)t=t.to(at::kFloat);
  if(bad==4)t=at::zeros(target?at::IntArrayRef({1,4,32}):at::IntArrayRef({1,31,32}),at::kHalf);
  if(bad==5)model.action_dim_pad_=16;
  if(bad==6)t=at::_conj(at::ones(t.sizes(),at::kComplexFloat));
  mark_cpu(t);reset_observed();nodes.clear();RpuKernelGraph::active().leases.clear();private_allocations=0;
  rejects([&]{call(model,in,true,bad==7,10,false,5);});
  assert(private_allocations==0&&batches.empty()&&scalars.empty()&&sized_scalars.empty()&&nodes.empty()&&RpuKernelGraph::active().leases.empty());++cases;
 }
 {Qwen3_5Model model;model.action_dim_pad_=32;Inputs in(false,false,1,32);cpu_inputs.clear();fresh(in,true,false,3);
  reset_observed();nodes.clear();private_allocations=0;rejects([&]{call(model,in,false,false,1);});
  assert(private_allocations==0&&batches.empty()&&scalars.empty()&&sized_scalars.empty()&&nodes.empty());++cases;}
 // CPU requests must also reject output aliases, missing descriptors and
 // short KV storage before fresh allocation or publication.
 for(int bad=0;bad<3;++bad){Qwen3_5Model model;model.action_dim_pad_=32;
  Inputs in(true,false,10,32,5);fresh(in,true,true,3);if(bad==0)in.out=in.action;
  reset_observed();nodes.clear();RpuKernelGraph::active().leases.clear();private_allocations=0;
  rejects([&]{call(model,in,true,false,10,bad==1,5,bad==2);});
  assert(private_allocations==0&&batches.empty()&&scalars.empty()&&sized_scalars.empty()&&nodes.empty()&&RpuKernelGraph::active().leases.empty());++cases;
 }
 // Existing RTC legal prefix extremes use full current bytes, without fixing
 // CPU admission to the measured five-row recipe or rounding into pool tails.
 for(int length:{1,31}){Qwen3_5Model model;model.action_dim_pad_=32;
  Inputs in(true,false,10,32,length);cpu_inputs.clear();fresh(in,true,true,3);reset_observed();private_allocations=0;
  call(model,in,true,false,10,false,length);assert(batches.size()==1&&batches[0].back().bytes==size_t(length*64));
  execute(bound(model,true),true);++cases;}
 RpuKernelGraph::active().leases.clear();
 std::cout<<"actual Action block host cases="<<cases<<"\n";
}
'''
    _run_native(tmp_path, cpp.replace("__TRACE_STRUCT__", trace_struct).replace("__BODY__", body), torch)
