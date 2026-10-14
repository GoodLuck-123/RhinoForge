"""Actual NVFP4 SwiGLU launcher and model contracts."""
from pathlib import Path

import test_g05_nvfp4_pair_native as pair
from test_g05_q_gate_native import _compile_run
from test_g05_vision_owner_model import _block

ROOT = Path(__file__).resolve().parents[1]
SWIGLU = "G05_GATE_UP_SWIGLU_NVFP4_ACC16_M32N512K1024_N128"
PAIR_CALL = "rpu_launch_g05_gate_up_paired_nvfp4_acc16_kernel"
FUSED_CALL = "rpu_launch_g05_gate_up_swiglu_nvfp4_acc16_kernel"


def _function(source, marker):
    start = source.index(marker)
    return source[start:source.index("{", start)] + _block(source, marker)


def test_swiglu_typed_owners_spm_and_replay_counters(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(pair, "_compile_run", lambda _p, _n, cpp: captured.append(cpp))
    pair.test_pair_typed_four_owners_and_absolute_spm_boundaries(tmp_path)
    cpp = captured[0]
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    old = _function(source, "void " + PAIR_CALL + "(")
    new = _function(source, "void " + FUSED_CALL + "(")
    assert cpp.count(old) == 1
    cpp = cpp.replace(old, new).replace(PAIR_CALL, FUSED_CALL)
    cpp = cpp.replace("stage_kernel_ddr_registers(KernelId,std::vector",
                      "stage_kernel_ddr_registers(KernelId id,std::vector")
    cpp = cpp.replace("assert(!pending&&!written&&v.size()==4);",
                      f"assert(id==KernelId::{SWIGLU});assert(!pending&&!written&&v.size()==4);")
    # A reset is not a GP write. Require every packet to explicitly encode both
    # complete uint32 counter pairs, including zero high halves.
    cpp = cpp.replace("++launches;g.pending=g.written=false;",
                      "for(int r:{128,129,130,131})assert(g.kernel.regs.at(r)==0);"
                      "g.kernel.regs[128]=8;g.kernel.regs[130]=8;"
                      "++launches;g.pending=g.written=false;")
    _compile_run(tmp_path, "g05_nvfp4_swiglu_owners", cpp)


def test_swiglu_cold_complete_rounding_fallback_and_layout(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(pair, "_compile_run", lambda _p, _n, cpp: captured.append(cpp))
    pair.test_pair_cold_scope_complete_mlp_rounding_and_old_ref_fallback(tmp_path, monkeypatch)
    cpp = captured[0].split("int main(){", 1)[0]
    source = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    stubs = [
        ("void prepare_action_nvfp4_swiglu(){}", "void prepare_action_nvfp4_swiglu()", "void Qwen3_5Model::prepare_action_nvfp4_swiglu("),
        ("bool use_action_nvfp4_swiglu(int64_t) const{return false;}", "bool use_action_nvfp4_swiglu(int64_t rows) const", "bool Qwen3_5Model::use_action_nvfp4_swiglu("),
        ("std::vector<int64_t> action_nvfp4_swiglu_arguments() const{return {};}", "std::vector<int64_t> action_nvfp4_swiglu_arguments() const", "std::vector<int64_t> Qwen3_5Model::action_nvfp4_swiglu_arguments("),
        ("template<class...T>void launch_action_nvfp4_swiglu(T...){assert(false);}", "void launch_action_nvfp4_swiglu(int layer,const ChunkInfo& chunk)", "void Qwen3_5Model::launch_action_nvfp4_swiglu("),
    ]
    for stub, signature, marker in stubs:
        assert cpp.count(stub) == 1
        cpp = cpp.replace(stub, signature + _block(source, marker), 1)
    cpp = cpp.replace("enum class KernelId{", "enum class KernelId{" + SWIGLU + "," + SWIGLU.replace("ACC16", "ACC32") + ",", 1)
    cpp = cpp.replace("bool pair=true,fusion=true,nv32=true;", "bool pair=true,fusion=true,nv32=true,swiglu=true;")
    cpp = cpp.replace("++queries;", f"++queries;if(id==KernelId::{SWIGLU})return swiglu;if(id==KernelId::{SWIGLU.replace('ACC16', 'ACC32')})return swiglu&&nv32;",1)
    signature = ("void " + FUSED_CALL + "(uint32_t,const at::Tensor&,const at::Tensor&,"
                 "uint32_t,uint32_t,const at::Tensor&,const at::Tensor&,uint32_t,uint32_t,uint16_t);\n")
    cpp = cpp.replace("struct Model{", signature + signature.replace("_acc16_kernel", "_acc32_kernel") + "struct Model{", 1)
    fake = _function(cpp, "void " + PAIR_CALL + "(uint32_t in,")
    cpp += fake.replace(PAIR_CALL, FUSED_CALL).replace('"nv-pair"', '"nv-swiglu"')
    cpp += fake.replace(PAIR_CALL, FUSED_CALL.replace("_acc16_kernel", "_acc32_kernel")).replace('"nv-pair"', '"nv-swiglu-acc32"')
    header = (ROOT / "src/fused/rpu_qwen3_5_model.h").read_text()
    begin = header.index("identity.push_back(static_cast<int64_t>(action_nvfp4_swiglu_available_));")
    end = header.index("            for (const auto& owner : action_gateup_resident_)", begin)
    cpp = cpp.replace(" void mlp(int layer_idx,", " std::vector<int64_t>cost(){std::vector<int64_t>identity;" + header[begin:end] + "return identity;}\n void mlp(int layer_idx,", 1)
    cpp += r'''
int main(){
 Model m;m.action_nvfp4_.active=true;m.action_nvfp4_.values={1,2,3,4,5};m.linear_acc32_=false;m.resident=false;
 for(auto&w:m.layer_weights_){w.gate_w.dtype=w.up_w.dtype=at::kByte;w.gate_w.cols=w.up_w.cols=512;w.gate_ws.dtype=w.up_ws.dtype=at::kByte;w.gate_ws.rows=w.up_ws.rows=64;w.gate_ws.cols=w.up_ws.cols=4096;w.gate_ws.owner=w.gate_w.owner+100;w.up_ws.owner=w.up_w.owner+200;}
 auto&cache=KernelCache::instance();auto prepare=[&]{m.prepare_action_gateup_pair();m.prepare_action_silu_round_mul();m.prepare_action_nvfp4_swiglu();m.plan();};
 prepare();assert(m.action_nvfp4_swiglu_available_&&!m.use_action_nvfp4_swiglu(32));
 m.action_step_active_=m.action_loop_active_=true;m.action_num_steps_=10;m.plan();
 assert(m.use_action_nvfp4_swiglu(32)&&!m.use_action_nvfp4_swiglu(31));
 assert(m.routes.size()==1&&m.routes[0].site==m.action_nvfp4_swiglu_site()&&m.routes[0].selector==11);
 assert(m.routes[0].args==std::vector<int64_t>({1,32,512,1024,8,16,176,128,128,4,8,2048,1024,4,24,96,256,2,20,32,34,24,36,4,38,134,0x3e92,0x0492,1,8,4,128,130,0,0,0,1}));
 auto nv=m.nv_plan();assert(nv.size()==3&&nv[0].args==std::vector<int64_t>({0})&&nv[1].args==std::vector<int64_t>({1})&&nv[2].args==std::vector<int64_t>({4}));
 const auto valid=m.routes;const auto hash=m.subclass_layout_hash();const auto cost=m.cost();const auto queries=cache.queries;const auto proofs=tensor_proofs;
 for(int layer=0;layer<24;++layer){launches.clear();m.mlp(layer);assert(launches==std::vector<std::string>({"nv-swiglu","down"}));}
 assert(cache.queries==queries&&tensor_proofs==proofs);
 for(size_t i=0;i<m.routes[0].args.size();++i){m.routes=valid;++m.routes[0].args[i];reject([&]{m.mlp(0);});}m.routes=valid;
 m.routes[0].selector=9;reject([&]{m.mlp(0);});m.routes=valid;m.complete=false;reject([&]{m.mlp(0);});m.complete=true;
 reject([&]{m.launch_action_nvfp4_swiglu(-1,{});});reject([&]{m.launch_action_nvfp4_swiglu(24,{});});reject([&]{m.launch_action_nvfp4_swiglu(0,{31,2});});reject([&]{m.launch_action_nvfp4_swiglu(0,{32,3});});
 // Capability changes only at cold preparation; hot dispatch has no lookup.
 cache.swiglu=false;launches.clear();m.mlp(23);assert(launches==std::vector<std::string>({"nv-swiglu","down"})&&cache.queries==queries);
 prepare();assert(!m.action_nvfp4_swiglu_available_&&m.subclass_layout_hash()!=hash&&m.cost()!=cost);
 assert(m.routes.size()==2&&m.routes[0].selector==9);launches.clear();m.mlp(23);assert(launches==std::vector<std::string>({"nv-pair","round-mul","down"}));
 cache.swiglu=true;cache.fusion=false;prepare();assert(!m.action_nvfp4_swiglu_available_);launches.clear();m.mlp(23);assert(launches==std::vector<std::string>({"nv-pair","silu","mul","down"}));
 cache.pair=false;prepare();assert(!m.action_nvfp4_swiglu_available_);launches.clear();m.mlp(23);assert(launches==std::vector<std::string>({"gate","silu","up","mul","down"}));assert(m.nv_plan().size()==5);
 cache.pair=cache.fusion=true;prepare();assert(m.subclass_layout_hash()==hash&&m.cost()==cost);
 for(int fault=0;fault<12;++fault){auto x=m;if(fault==0)continue;if(fault==1)x.action_mode_=false;if(fault==2)x.layer_weights_.back().up_ws.dtype=at::kHalf;if(fault==3)x.layer_weights_.back().up_ws.rank=1;if(fault==4)x.layer_weights_.back().up_w.cols=1024;if(fault==5)x.layer_weights_.back().gate_w.location=c10::DeviceType::CPU;if(fault==6)x.layer_is_full_.back()=0;if(fault==7)x.use_silu_=false;if(fault==8)x.cores=4;if(fault==9)x.layer_weights_.back().up_w.storage_offset_=256;if(fault==10)x.layer_weights_.back().gate_ws.storage_offset_=256;if(fault==11)x.action_len_=31;x.prepare_action_gateup_pair();x.prepare_action_silu_round_mul();x.prepare_action_nvfp4_swiglu();assert(!x.action_nvfp4_swiglu_available_);}
 for(int fault=0;fault<3;++fault){auto x=m;if(fault==0)x.action_num_steps_=1;if(fault==1)x.action_loop_active_=false;if(fault==2)x.action_step_active_=false;assert(!x.use_action_nvfp4_swiglu(32));}
}
'''
    _compile_run(tmp_path, "g05_nvfp4_swiglu_model", cpp)
    acc32 = cpp.replace('m.linear_acc32_=false;', 'm.linear_acc32_=true;', 1)
    head, main = acc32.split('int main(){', 1)
    main = main.replace('"nv-swiglu"', '"nv-swiglu-acc32"').replace('"nv-pair"', '"nv-pair-acc32"')
    main = main.replace('selector==11', 'selector==15').replace('selector==9', 'selector==13')
    main = main.replace('{1,32,512,1024,8,16,', '{1,32,512,1024,8,32,')
    main = main.replace(' const auto valid=', ' auto alias=m;alias.routes[0].site=1871355692309183287LL;reject([&]{alias.mlp(0);});auto stale=m;stale.routes[0].args[5]=16;reject([&]{stale.mlp(0);}); const auto valid=',1)
    _compile_run(tmp_path, "g05_nvfp4_swiglu_acc32_model", head+'int main(){'+main)
