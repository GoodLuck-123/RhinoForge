"""Execute FP16 Text owner and tail candidates through real model/traversal bodies."""
import test_g05_text_owner_rms as owner
import test_g05_text_mlp_tail as tail
import test_g05_text_silu_round_mul as rounded
from test_g05_q_gate_native import _compile_run

TAIL_ENABLED=True

def test_fp16_owner_and_tail_actual_source(tmp_path,monkeypatch):
    stage="tail" if TAIL_ENABLED else "owner"
    captured=[];monkeypatch.setattr(tail,'_compile_run',lambda p,n,cpp:captured.append(cpp))
    tail.test_text_mlp_tail_cold_complete_and_same_group_callers(tmp_path,monkeypatch)
    cpp=captured[0]
    traversal_run=cpp[cpp.index(' auto run=[&](Model&x'):cpp.index(' assert(run(m,24,InterLayerIO::SPM_RESIDENT,-1)==23);')]
    cpp=cpp.split('int main(){',1)[0]
    cpp+=r'''
constexpr bool TAIL=__TAIL__;
void fp16(Model&m){
 for(auto&w:m.layer_weights_){
  for(auto*t:{&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.attn_gate_w,&w.gate_w,&w.up_w,&w.down_w,&w.gdn_q_w,&w.gdn_k_w,&w.gdn_v_w,&w.gdn_in_z_w,&w.gdn_out_w,&w.gdn_b_bg_w,&w.gdn_a_bg_w})t->dtype=at::kHalf;
  for(auto*s:{&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.attn_gate_ws,&w.gate_ws,&w.up_ws,&w.down_ws,&w.gdn_q_ws,&w.gdn_k_ws,&w.gdn_v_ws,&w.gdn_in_z_ws,&w.gdn_out_ws,&w.gdn_b_bg_ws,&w.gdn_a_bg_ws})s->present=false;
 }
}
void denied(Model&m){m.installed();assert(!m.has_text_fp16_projection_owners()&&!m.text_owner_rms_available_&&!m.text_gdn_owner_rms_available_&&!m.text_mlp_tail_available_);}
int main(){
 for(bool acc32:{false,true}){
  Model m;fp16(m);m.linear_acc32_=acc32;m.installed();assert(m.has_text_fp16_projection_owners()&&m.text_owner_rms_available_&&m.text_gdn_owner_rms_available_&&m.text_mlp_tail_available_==TAIL);m.plan();
  assert(m.text_owner_rms_arguments()[11]==(acc32?32:16)&&m.text_gdn_owner_rms_arguments()[11]==(acc32?32:16));
  if(TAIL)assert(m.text_mlp_tail_arguments()[5]==(acc32?32:16));
  const auto hash=m.layout_hash();const auto query=KernelCache::instance().calls;const auto identity=m.identity();
  __TRAVERSAL__
  assert(run(m,24,InterLayerIO::SPM_RESIDENT,-1)==(TAIL?23:0));
  run(m,24,InterLayerIO::DDR_PINGPONG,-1);run(m,1,InterLayerIO::SPM_RESIDENT,-1);
  if(TAIL)for(int fail:{0,7,22}){run(m,24,InterLayerIO::SPM_RESIDENT,fail);assert(run(m,24,InterLayerIO::SPM_RESIDENT,-1)==23);}
  assert(KernelCache::instance().calls==query);
  // Every full/GDN projection and scale, on every layer, independently withdraws the FP16 chain.
  for(int layer=0;layer<24;++layer)for(int family=0;family<8;++family)for(int fault=0;fault<10;++fault){
   auto x=m;auto&w=x.layer_weights_[layer];const bool full=x.layer_is_full_[layer];
   std::array<at::Tensor*,8>ws{&w.gate_w,&w.up_w,&w.down_w,full?&w.q_w:&w.gdn_q_w,full?&w.k_w:&w.gdn_k_w,full?&w.v_w:&w.gdn_v_w,full?&w.o_w:&w.gdn_out_w,full?&w.attn_gate_w:&w.gdn_in_z_w};
   std::array<at::Tensor*,8>ss{&w.gate_ws,&w.up_ws,&w.down_ws,full?&w.q_ws:&w.gdn_q_ws,full?&w.k_ws:&w.gdn_k_ws,full?&w.v_ws:&w.gdn_v_ws,full?&w.o_ws:&w.gdn_out_ws,full?&w.attn_gate_ws:&w.gdn_in_z_ws};
   auto&t=*ws[family];if(fault==0)t.present=false;if(fault==1)t.dtype=at::kChar;if(fault==2)t.contiguous=false;if(fault==3)t.offset=256;if(fault==4)t.rank=1;if(fault==5)--t.rows;if(fault==6)--t.cols;if(fault==7)t.location=c10::DeviceType::CPU;if(fault==8)++t.index;if(fault==9){ss[family]->present=true;ss[family]->count=1;}denied(x);
  }
  for(int layer=0;layer<24;++layer)if(!m.layer_is_full_[layer])for(int family=0;family<2;++family)for(int fault=0;fault<10;++fault){auto x=m;auto&w=x.layer_weights_[layer];auto&t=family?w.gdn_b_bg_w:w.gdn_a_bg_w;auto&s=family?w.gdn_b_bg_ws:w.gdn_a_bg_ws;
   if(fault==0)t.present=false;if(fault==1)t.dtype=at::kChar;if(fault==2)t.contiguous=false;if(fault==3)t.offset=256;if(fault==4)t.rank=1;if(fault==5)--t.rows;if(fault==6)--t.cols;if(fault==7)t.location=c10::DeviceType::CPU;if(fault==8)++t.index;if(fault==9){s.present=true;s.count=1;}denied(x);
  }
  for(int layer=0;layer<24;++layer)for(int family=0;family<2;++family)for(int fault=0;fault<8;++fault){auto x=m;auto&t=family?x.layer_weights_[layer].post_norm_w:x.layer_weights_[layer].input_norm_w;
   if(fault==0)t.present=false;if(fault==1)t.dtype=at::kChar;if(fault==2)t.contiguous=false;if(fault==3)t.offset=128;if(fault==4)t.rank=2;if(fault==5)--t.count;if(fault==6)t.location=c10::DeviceType::CPU;if(fault==7)++t.index;denied(x);
  }
  for(int layer=0;layer<24;++layer){auto x=m;x.layer_is_full_[layer]=!x.layer_is_full_[layer];denied(x);}
  for(int fault=0;fault<9;++fault){auto x=m;if(fault==0)x.action_mode_=true;if(fault==1)x.fast_replay_enabled_=false;if(fault==2)x.use_silu_=false;if(fault==3)x.has_gdn_=false;if(fault==4)x.eps_=1e-5;if(fault==5)++x.gdn_dk_;if(fault==6)x.layer_weights_.pop_back();if(fault==7)x.tp=4;if(fault==8)x.heads=16;denied(x);}
  for(int fault=0;fault<10;++fault){auto x=m;if(fault==0)x.manifest.logical_length=244;if(fault==1)x.seq_len=128;if(fault==2)x.position=1;if(fault==3)x.batch_size=2;if(fault==4)x.debug=true;if(fault==5)x.z2_bound_=true;if(fault==6)x.retained_prefill_graph_=false;if(fault==7)x.stage_plan.qkv.chunks.push_back({});if(fault==8)x.stage_plan.compute.chunks[0].offset=256;if(fault==9)x.stage_plan.compute.chunks[0].idx=1;
   assert(!x.use_text_owner_rms(x.stage_plan,x.seq_len,x.manifest.logical_length,x.position,x.batch_size)&&!x.use_text_gdn_owner_rms(x.stage_plan,x.seq_len,x.manifest.logical_length,x.position,x.batch_size));
  }
  // Old REF fallback, stable repeated calls, and actual mode mismatch fail before fused launch.
  auto&cache=KernelCache::instance();cache.available=false;m.installed();assert(!m.text_owner_rms_available_&&!m.text_gdn_owner_rms_available_&&!m.text_mlp_tail_available_);m.plan();run(m,24,InterLayerIO::SPM_RESIDENT,-1);cache.available=true;m.installed();m.plan();
  {auto x=m;x.set_linear_acc32(!acc32);assert(x.text_owner_rms_arguments()[11]==(acc32?16:32));launches.clear();reject([&]{x.launch_text_owner_rms(3,{});});assert(launches.empty());x.plan();assert(run(x,24,InterLayerIO::SPM_RESIDENT,-1)==(TAIL?23:0));if(TAIL)assert(x.layout_hash()!=hash&&x.identity()!=identity);}
  if(TAIL){m.input_in_spm=m.output_to_spm=true;for(int fault=0;fault<8;++fault){auto x=m;if(fault==0)x.stage_plan.chunk_mode=ChunkMode::KV_FIRST;if(fault==1)x.stage_plan.input.chunks.clear();if(fault==2)x.stage_plan.spans.clear();if(fault==3)x.stage_plan.spans[0].group_id=1;if(fault==4)x.body_iter=1;if(fault==5)x.output_to_spm=false;if(fault==6)x.complete=false;ChunkInfo c;if(fault==7)c.offset=256;assert(!x.text_mlp_tail_handoff(0,c,true));}}
 }
}
'''.replace('__TAIL__','true' if stage=='tail' else 'false').replace('__TRAVERSAL__',traversal_run)
    # This legacy fixture uses compact one-line if statements; preserve compiler warnings elsewhere.
    cpp=cpp.replace('#include <algorithm>','#include <algorithm>\n#pragma GCC diagnostic ignored "-Wmisleading-indentation"',1)
    _compile_run(tmp_path,'fp16_'+stage,cpp)


def test_fp16_rounded_actual_source(tmp_path,monkeypatch):
    captured=[];monkeypatch.setattr(rounded,'_compile_run',lambda p,n,cpp:captured.append(cpp))
    rounded.test_text_silu_round_mul_cold_complete_and_caller_fallback(tmp_path,monkeypatch)
    cpp=captured[0].split('int main(){',1)[0]
    cpp=cpp.replace('assert(!launches.empty()&&launches.back()=="pair_w8");','assert(!launches.empty()&&(launches.back()=="pair"||launches.back()=="pair_fp16_acc16"));')
    cpp+=r'''

void fp16(Model&m){
 for(auto&w:m.layer_weights_){
  for(auto*t:{&w.q_w,&w.k_w,&w.v_w,&w.o_w,&w.attn_gate_w,&w.gate_w,&w.up_w,&w.down_w,&w.gdn_q_w,&w.gdn_k_w,&w.gdn_v_w,&w.gdn_in_z_w,&w.gdn_out_w,&w.gdn_b_bg_w,&w.gdn_a_bg_w}){t->dtype=at::kHalf;t->rows=2048;t->cols=2048;}
  w.gate_w.rows=w.up_w.rows=6144;w.down_w.cols=6144;w.gdn_b_bg_w.rows=w.gdn_a_bg_w.rows=128;
  for(auto*t:{&w.input_norm_w,&w.post_norm_w}){t->dtype=at::kHalf;t->rows=2048;t->rank=1;}
  for(auto*s:{&w.q_ws,&w.k_ws,&w.v_ws,&w.o_ws,&w.attn_gate_ws,&w.gate_ws,&w.up_ws,&w.down_ws,&w.gdn_q_ws,&w.gdn_k_ws,&w.gdn_v_ws,&w.gdn_in_z_ws,&w.gdn_out_ws,&w.gdn_b_bg_ws,&w.gdn_a_bg_ws})s->present=false;
 }
 m.fast_replay_enabled_=m.retained_prefill_graph_=true;
}
int main(){
 for(bool acc32:{false,true}){
  Model m;fp16(m);m.linear_acc32_=acc32;m.installed();m.plan();
  assert(m.has_text_fp16_projection_owners()&&m.text_gateup_pair_available_&&!m.text_gateup_pair_w8_&&m.text_silu_round_mul_available_&&m.use_text_silu_round_mul(256));
  assert(m.text_silu_round_mul_arguments()[6]==(acc32?32:16));
  const auto hash=m.subclass_layout_hash();const auto identity=m.kvinsert_cost_weight_identity();const int queries=KernelCache::instance().queries,proofs=at::Tensor::proofs;
  const std::string pair_name=acc32?"pair":"pair_fp16_acc16";
  for(int layer=0;layer<24;++layer){launches.clear();m.mlp(layer);assert(launches==std::vector<std::string>({pair_name,"text-round","down"}));}
  assert(KernelCache::instance().queries==queries&&at::Tensor::proofs==proofs);
  auto&cache=KernelCache::instance();cache.rounded_available=false;m.installed();m.plan();assert(m.text_gateup_pair_available_&&!m.text_silu_round_mul_available_);launches.clear();m.mlp(23);assert(launches==std::vector<std::string>({pair_name,"silu","mul","down"}));cache.rounded_available=true;m.installed();m.plan();
  cache.available=false;m.installed();m.plan();assert(!m.text_gateup_pair_available_&&!m.text_silu_round_mul_available_);launches.clear();m.mlp(23);assert(launches==std::vector<std::string>({"gate","silu","up","mul","down"}));cache.available=true;m.installed();m.plan();
  for(int layer=0;layer<24;++layer){auto x=m;auto&w=x.layer_weights_[layer];(x.layer_is_full_[layer]?w.q_w:w.gdn_q_w).dtype=at::kChar;x.installed();assert(x.text_gateup_pair_available_&&!x.text_silu_round_mul_available_);}
  for(int layer=0;layer<24;++layer)for(int side=0;side<3;++side){auto x=m;auto&w=x.layer_weights_[layer];std::array<at::Tensor*,3>ss{&w.gate_ws,&w.up_ws,&w.down_ws};ss[side]->present=true;x.installed();assert(!x.text_silu_round_mul_available_);}
  for(int fault=0;fault<8;++fault){auto x=m;if(fault==0)x.logical_length=244;if(fault==1)x.position=1;if(fault==2)x.batch_size=2;if(fault==3)x.seq_len=128;if(fault==4)x.stage_plan.compute.chunks[0].offset=256;if(fault==5)x.debug=true;if(fault==6)x.z2_bound_=true;if(fault==7)x.retained_prefill_graph_=false;assert(!x.use_text_silu_round_mul(x.seq_len));}
  for(size_t arg=0;arg<m.routes[1].args.size();++arg){auto x=m;++x.routes[1].args[arg];launches.clear();reject([&]{x.launch_text_silu_round_mul(0,{});});assert(launches.empty());}
  {auto x=m;x.set_linear_acc32(!acc32);assert(x.text_silu_round_mul_available_&&x.subclass_layout_hash()!=hash&&x.kvinsert_cost_weight_identity()!=identity);launches.clear();reject([&]{x.launch_text_silu_round_mul(0,{});});assert(launches.empty());x.plan();for(int layer=0;layer<24;++layer){launches.clear();x.mlp(layer);assert(launches==std::vector<std::string>({acc32?"pair_fp16_acc16":"pair","text-round","down"}));}}
 }
}
'''
    _compile_run(tmp_path,'fp16_rounded',cpp)
