"""New lengths execute production admission, COMPLETE dispatch and typed writers."""
from pathlib import Path

from test_g05_q_gate_native import _compile_run
from test_g05_vision_owner_model import _block

ROOT = Path(__file__).resolve().parents[1]


def test_text_single_e256_new_prefix_actual_owner_tail(tmp_path, monkeypatch):
    import test_g05_text_fp16_neutral as fp16

    captured = []
    monkeypatch.setattr(fp16, '_compile_run', lambda p, n, cpp: captured.append(cpp))
    fp16.test_fp16_owner_and_tail_actual_source(tmp_path, monkeypatch)
    cpp = captured[0].split('int main(){', 1)[0]
    cpp += r'''
int main(){
 for(bool half:{false,true})for(bool acc32:{false,true}){
  Model m;if(half)fp16(m);m.linear_acc32_=acc32;m.installed();
  assert(m.text_owner_rms_available_&&m.text_gdn_owner_rms_available_&&m.text_mlp_tail_available_);
  const auto hash=m.layout_hash();const auto identity=m.identity();const auto queries=KernelCache::instance().calls;
  for(int64_t p:{193,209,225,241,245,193}){
   m.manifest.logical_length=p;m.plan();
   assert(m.use_text_owner_rms(m.stage_plan,256,p,0,1)&&m.use_text_gdn_owner_rms(m.stage_plan,256,p,0,1)&&m.use_text_mlp_tail(m.stage_plan,256,p,0,1));
   const auto&routes=m.manifest.routes;
   for(const auto&r:routes){if(r.site==QWEN35_TEXT_OWNER_RMS_SITE||r.site==QWEN35_TEXT_GDN_OWNER_RMS_SITE)assert(r.args[9]==p&&r.args[0]==2);if(r.site==QWEN35_TEXT_MLP_TAIL_SITE)assert(r.args[6]==p&&r.args[0]==2);}
   for(int layer=0;layer<24;++layer){expected_layer=layer;m.input_in_spm=layer>0;m.output_to_spm=layer<23;launches.clear();m.input_entry(layer);m.layer(layer);
    assert(std::count(launches.begin(),launches.end(),"input_norm")==int(layer==0));
    assert(std::count(launches.begin(),launches.end(),"tail_owner")==int(layer<23));
    assert(std::count(launches.begin(),launches.end(),"owner")+std::count(launches.begin(),launches.end(),"owner_gdn")==1);
   }
   assert(m.layout_hash()==hash&&m.identity()==identity&&KernelCache::instance().calls==queries);
   // A frozen descriptor for P must reject a changed logical prefix before its own launch.
   auto x=m;x.manifest.logical_length=p==193?225:193;launches.clear();reject([&]{x.launch_text_owner_rms(3,{});});assert(launches.empty());
  }
  for(int64_t p:{0,161,181,192,194,224,244,256,257,321,449})assert(!m.use_text_owner_rms(m.stage_plan,256,p,0,1)&&!m.use_text_gdn_owner_rms(m.stage_plan,256,p,0,1)&&!m.use_text_mlp_tail(m.stage_plan,256,p,0,1));
  for(int fault=0;fault<8;++fault){auto x=m;x.manifest.logical_length=225;if(fault==0)x.seq_len=192;if(fault==1)x.seq_len=320;if(fault==2)x.stage_plan.qkv.chunks.push_back({});if(fault==3)x.stage_plan.compute.chunks[0].len=192;if(fault==4)x.stage_plan.chunk_mode=ChunkMode::KV_FIRST;if(fault==5)x.stage_plan.input.chunks.push_back({});if(fault==6)x.stage_plan.spans[0].group_id=1;if(fault==7)x.retained_prefill_graph_=false;assert(!x.use_text_mlp_tail(x.stage_plan,x.seq_len,225,0,1));}
 }
}
'''
    _compile_run(tmp_path, 'g05_text_single_e256_multi_prefix', cpp)


def test_text_single_e256_new_prefix_actual_rounded(tmp_path, monkeypatch):
    import test_g05_text_fp16_neutral as fp16

    captured = []
    monkeypatch.setattr(fp16, '_compile_run', lambda p, n, cpp: captured.append(cpp))
    fp16.test_fp16_rounded_actual_source(tmp_path, monkeypatch)
    cpp = captured[0].split('int main(){', 1)[0]
    cpp += r'''
int main(){
 for(bool acc32:{false,true}){
  Model m;fp16(m);m.linear_acc32_=acc32;m.installed();const auto hash=m.subclass_layout_hash();const auto identity=m.kvinsert_cost_weight_identity();
  for(int64_t p:{193,209,225,241,245,193}){
   m.logical_length=p;m.plan();assert(m.use_text_silu_round_mul(256));assert(m.routes[1].args[18]==p&&m.routes[1].args[0]==2);
   for(int layer=0;layer<24;++layer){launches.clear();m.mlp(layer);assert(launches==std::vector<std::string>({acc32?"pair":"pair_fp16_acc16","text-round","down"}));}
   assert(m.subclass_layout_hash()==hash&&m.kvinsert_cost_weight_identity()==identity);
   auto x=m;x.logical_length=p==193?225:193;launches.clear();reject([&]{x.launch_text_silu_round_mul(0,{});});assert(launches.empty());
  }
  for(int64_t p:{0,161,181,192,194,224,244,256,257,321,449}){auto x=m;x.logical_length=p;assert(!x.use_text_silu_round_mul(256));}
  auto x=m;x.logical_length=225;x.stage_plan.compute.chunks.push_back({});assert(!x.use_text_silu_round_mul(256));
 }
}
'''
    _compile_run(tmp_path, 'g05_text_single_e256_multi_prefix_rounded', cpp)


def test_action_new_prefix_actual_owner_plan_and_joint_fallback(tmp_path, monkeypatch):
    import test_g05_owner_cache_joint as owner

    captured = []
    monkeypatch.setattr(owner, '_compile_run', lambda p, n, cpp: captured.append(cpp))
    owner.test_joint_cache_cold_complete_kv22_identity_and_fallbacks(tmp_path, monkeypatch)
    cpp = captured[0].split('int main(){', 1)[0]
    # Feed the production dispatcher the exact existing KV22 plans for the new domain.
    cpp = cpp.replace('else{parts={{KvInsertKernel::V2,p,0,11}',
        'else if(p%16==1){parts={{KvInsertKernel::V2,p,0,15},{KvInsertKernel::V16,p+15,15,16},{KvInsertKernel::V2,p+31,31,1}};route_id=KvInsertRoute::HYBRID3;}else{parts={{KvInsertKernel::V2,p,0,11}')
    cpp = cpp.replace('(p.segment(0).position==0||p.segment(0).position==245)',
        '(p.segment(0).position==0||p.segment(0).position==245||p.segment(0).position==181||p.segment(0).position%16==1)')
    cpp += r'''
int main(){
 const int64_t linear=558172469151339648LL,norm=2641642310566033775LL,insert=8281105153491195213LL;
 for(bool mixed:{false,true})for(bool acc32:{false,true}){
  Model m;m.linear_acc32_=acc32;m.owner_fixture();if(mixed){m.action_nvfp4_.values={1,2,3,4,5};for(auto&w:m.layer_weights_)for(auto*t:{&w.q_w,&w.o_w,&w.gate_w,&w.up_w,&w.down_w})t->dtype=at::kByte;}m.install();m.loop();assert(m.action_owner_cache_available_);
  m.prepare_action_owner_cache_joint();
  const auto hash=m.subclass_layout_hash();const auto identity=m.kvinsert_cost_weight_identity();
  for(int64_t p:{161,181,193,225,257,321,385,449,181,161}){
   m.position=999;for(int layer=0;layer<24;++layer)m.action_prefix_lens_[layer]=(layer%4==3)?p:0;
   m.full_plan(p);assert(m.use_action_owner_cache(32,p)&&!m.use_action_owner_cache_joint(32,p));m.position=p;
   for(int layer=0;layer<24;++layer){launches.clear();consumed.clear();m.chain(layer);const bool joint=mixed&&layer%4!=3;assert(std::find(launches.begin(),launches.end(),joint?"joint-cache":"owner-cache")!=launches.end());assert(std::count(consumed.begin(),consumed.end(),joint?QWEN35_ACTION_OWNER_CACHE_JOINT_SITE:insert)==1);assert(std::count(consumed.begin(),consumed.end(),norm)==1);if(!mixed)assert(std::count(consumed.begin(),consumed.end(),linear)==1);}
   FmbPrefillStageCandidate candidate;std::vector<FusedModelBase::Impl::KvCostObservation>observations;for(const auto&r:m.routes){candidate.physical_manifest.routes.push_back({r.site,r.family,r.selector,r.flags,r.arguments,r.invocation});if(r.family==FmbRouteFamily::KV_INSERT)observations.push_back({r.site,4,8,8,256,27,r.arguments});}assert(m.physical_manifest_forward_capability(candidate.physical_manifest).supported);const auto cost=make_kvinsert_cost_candidate(candidate,observations);assert(cost.reason=="READY"&&cost.rows.size()==2);
   const auto good=KvInsertSegmentPlan(p);const auto args=m.action_owner_cache_arguments(good);assert(args.size()==38&&args[22]==p);
   if(p==181){assert(good.parts[0].rows==11&&good.parts[1].position==192&&good.parts[2].position==208&&good.parts[2].rows==5);auto wrong=good;wrong.parts={{KvInsertKernel::V2,181,0,15},{KvInsertKernel::V16,196,15,16},{KvInsertKernel::V2,212,31,1}};reject([&]{m.action_owner_cache_arguments(wrong);});}

   for(size_t part=0;part<3;++part)for(int fault=0;fault<4;++fault){auto bad=good;auto&s=bad.parts[part];if(fault==0)++s.position;if(fault==1)++s.token_offset;if(fault==2)++s.rows;if(fault==3)s.kernel=s.kernel==KvInsertKernel::V2?KvInsertKernel::V16:KvInsertKernel::V2;reject([&]{m.action_owner_cache_arguments(bad);});}
   auto x=m;++x.action_prefix_lens_[23];reject([&]{x.insert_stage(23);});assert(m.subclass_layout_hash()==hash&&m.kvinsert_cost_weight_identity()==identity);
  }
  for(int64_t p:{-1,1,129,160,162,165,180,182,197,213,229,244,246,261,450,465,497})assert(!m.use_action_owner_cache(32,p));
  assert(m.use_action_owner_cache(32,0)&&m.use_action_owner_cache(32,245));
  if(mixed)assert(m.use_action_owner_cache_joint(32,0)&&m.use_action_owner_cache_joint(32,245));
 }
}
'''
    _compile_run(tmp_path, 'g05_action_bounded_p1_owner', cpp)


def test_action_new_prefix_actual_typed_separate_writer(tmp_path, monkeypatch):
    import test_g05_owner_cache_native as owner

    captured = []
    monkeypatch.setattr(owner, '_compile_run', lambda p, n, cpp: captured.append(cpp))
    owner.test_owner_norm_rope_and_segmented_cache_actual_wrappers(tmp_path)
    cpp = captured[0].split('int main(){', 1)[0]
    cpp += r'''
int main(){
 auto&g=RpuKernelGraph::active();at::Tensor kc{0x1240000000,{1,136,1,16,8,16,16}},vc{0x1250000000,{1,136,1,16,8,16,16}};
 for(int64_t p:{161,181,193,225,257,321,385,449,181,161}){
  const int64_t head=p==181?11:15,tail=p==181?5:1;
  const KvInsertSegmentPlan plan{KvInsertRoute::HYBRID3,{{KvInsertKernel::V2,p,0,head},{KvInsertKernel::V16,p+head,head,16},{KvInsertKernel::V2,p+head+16,head+16,tail}}};
  kc.address+=256;vc.address+=512;auto before=g.nodes.size();rpu_launch_g05_owner_insert_kvcache_spm_with_plan(kc,vc,0x10000,0x20000,plan);assert(g.nodes.size()==before+6);
  for(int kind=0;kind<2;++kind)for(int part=0;part<3;++part){const auto&s=g.nodes[before+kind*3+part];const auto&t=plan.segment(part);const bool v16=part==1;
   assert(s.id==(kind?(v16?V16:V2):(v16?K16:K2)));assert(pair(s,0)==uint32_t(t.position)&&s.regs.at(4)==t.rows);assert(pair(s,6)==uint32_t((kind?0x20000:0x10000)+t.token_offset*512));assert(pair(s,8)==uint32_t((kind?vc.address:kc.address)>>8));assert(s.grid==std::vector<uint16_t>({uint16_t(v16?1:t.rows),1,1}));assert(s.operands.size()==1&&s.operands[0].abi.access==GraphDdrRegisterAccess::Write);
  }
  reject([&]{rpu_launch_g05_owner_insert_kvcache_joint_spm_with_plan(kc,vc,0x10000,0x20000,plan);});
  for(int part=0;part<3;++part)for(int fault=0;fault<4;++fault){auto bad=plan;auto&s=bad.parts[part];if(fault==0)++s.position;if(fault==1)++s.token_offset;if(fault==2)++s.rows;if(fault==3)s.kernel=s.kernel==KvInsertKernel::V2?KvInsertKernel::V16:KvInsertKernel::V2;reject([&]{rpu_launch_g05_owner_insert_kvcache_spm_with_plan(kc,vc,0x10000,0x20000,bad);});}
  if(p==181){const KvInsertSegmentPlan wrong{KvInsertRoute::HYBRID3,{{KvInsertKernel::V2,181,0,15},{KvInsertKernel::V16,196,15,16},{KvInsertKernel::V2,212,31,1}}};reject([&]{rpu_launch_g05_owner_insert_kvcache_spm_with_plan(kc,vc,0x10000,0x20000,wrong);});}
  auto small_k=kc,small_v=vc;small_k.shape[1]=small_v.shape[1]=(p+31)/16;reject([&]{rpu_launch_g05_owner_insert_kvcache_spm_with_plan(small_k,small_v,0x10000,0x20000,plan);});
 }
 for(int64_t p:{129,160,162,165,180,182,197,213,229,261,450,465}){const KvInsertSegmentPlan plan{KvInsertRoute::HYBRID3,{{KvInsertKernel::V2,p,0,15},{KvInsertKernel::V16,p+15,15,16},{KvInsertKernel::V2,p+31,31,1}}};reject([&]{rpu_launch_g05_owner_insert_kvcache_spm_with_plan(kc,vc,0x10000,0x20000,plan);});}
}
'''
    _compile_run(tmp_path, 'g05_action_bounded_p1_typed', cpp)
