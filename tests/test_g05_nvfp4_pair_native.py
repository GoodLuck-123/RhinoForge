"""Exercise actual paired NVFP4 typed operands, cold routes and MLP caller."""
from pathlib import Path

from test_g05_q_gate_native import _compile_run
from test_siglip_cold_profile import _body

ROOT = Path(__file__).resolve().parents[1]


def test_pair_typed_four_owners_and_absolute_spm_boundaries(tmp_path):
    source = (ROOT / "src/ops/rpu_linear.cpp").read_text()
    functions = []
    for marker in ("void check_pi05_resident_spm(",
                   "static void launch_g05_gate_up_nvfp4_typed(",
                   "void rpu_launch_g05_gate_up_paired_nvfp4_acc16_kernel("):
        start = source.index(marker)
        functions.append(source[start:source.index("{", start)] + _body(source, marker))
    cpp = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>
#define TORCH_CHECK(ok,...) do{if(!(ok))throw std::runtime_error("rejected");}while(0)
constexpr int NUM_CORES=8;
namespace c10{using Half=uint16_t;}
namespace at{
constexpr int kByte=1,kHalf=2,kPrivateUse1=3;
using IntArrayRef=std::vector<int64_t>;
struct Device{int kind=kPrivateUse1,index=0;int type()const{return kind;}bool operator==(Device b)const{return kind==b.kind&&index==b.index;}};
struct Tensor{uintptr_t address=0x1234567800ULL;int dtype=kByte;bool present=true,contiguous=true;Device dev;int64_t offset=0;IntArrayRef shape{4096,512};
 bool defined()const{return present;}int scalar_type()const{return dtype;}Device device()const{return dev;}bool is_contiguous()const{return contiguous;}
 int64_t storage_offset()const{return offset;}IntArrayRef sizes()const{return shape;}size_t nbytes()const{return shape[0]*shape[1];}void*data_ptr()const{return reinterpret_cast<void*>(address);}};
}
struct SpmAllocator{static constexpr size_t SPM_USABLE=8387584;bool initialized=true;bool is_initialized()const{return initialized;}uint32_t addr(int,int)const{return 0x20000000;}}allocator;
#define SPM_ALLOC allocator
int flushes=0;void rpu_ddr_flush(void*){++flushes;}uint64_t RpuGetDevAddr(void*p){return reinterpret_cast<uintptr_t>(p);}
enum class KernelId{G05_GATE_UP_PAIRED_NVFP4_ACC16_M32N512K1024_N128,G05_GATE_UP_SWIGLU_NVFP4_ACC16_M32N512K1024_N128,G05_GATE_UP_SWIGLU_NVFP4_ACC32_M32N512K1024_N128};
enum class GraphDdrRegisterRole{PackedFp4ModelWeight,Fp8BlockScale};enum class GraphDdrRegisterAccess{Read};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct Abi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{Abi abi;at::Tensor owner;};
struct Kernel{std::map<int,uint16_t>regs;void reset_regs(){regs.clear();}void set_regs(int r,uint16_t v){regs[r]=v;}void set_regs(int r,uint32_t v){assert((r==128||r==130)&&v==0);regs[r]=uint16_t(v);regs[r+1]=uint16_t(v>>16);}};
struct Writer{void write(Kernel&);};
struct RpuKernelGraph{int stages=0;bool pending=false,written=false;Kernel kernel;std::vector<GraphDdrRegisterOperandSpec>owners;
 static RpuKernelGraph&active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId,std::vector<GraphDdrRegisterOperandSpec>v){assert(!pending&&!written&&v.size()==4);int lows[]={2,20,32,34};for(int i=0;i<4;++i){assert(v[i].abi.lo==lows[i]&&v[i].abi.hi==lows[i]+1&&v[i].abi.role==(i%2?GraphDdrRegisterRole::Fp8BlockScale:GraphDdrRegisterRole::PackedFp4ModelWeight));}owners=v;++stages;return {};}
 Kernel*get(KernelId){assert(!pending);pending=true;return &kernel;}};
void Writer::write(Kernel&k){auto&g=RpuKernelGraph::active();assert(g.pending&&!g.written);for(auto&o:g.owners){auto a=o.owner.address>>8;k.set_regs(o.abi.lo,uint16_t(a));k.set_regs(o.abi.hi,uint16_t(a>>16));}g.written=true;}
#define GET_KERNEL(id) RpuKernelGraph::active().get(id)
struct Queue{bool broadcast=false;int launches=0;void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel&,std::vector<uint16_t>grid,std::vector<uint8_t>cores){auto&g=RpuKernelGraph::active();assert(broadcast&&g.pending&&g.written&&grid==std::vector<uint16_t>({8,1,1})&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));++launches;g.pending=g.written=false;}}queue;
Queue*get_queue(int n){assert(n==8);return &queue;}
#define GET_QUEUE(n) get_queue(n)
__FUNCTIONS__
template<class F>void rejects(F f){auto&g=RpuKernelGraph::active();int n=queue.launches,s=g.stages,fl=flushes;try{f();assert(false);}catch(const std::runtime_error&){}assert(queue.launches==n&&g.stages==s&&flushes==fl&&!g.pending&&!g.written);}
uint32_t gp(int r){auto&v=RpuKernelGraph::active().kernel.regs;return uint32_t(v.at(r))|(uint32_t(v.at(r+1))<<16);}
int main(){
 std::array<at::Tensor,4>owners;for(int i=0;i<4;++i){owners[i].address+=i*0x10000000ULL;if(i%2)owners[i].shape={64,4096};}
 std::array<uint32_t,5>spm={0x20010000,0x20030000,0x20040000,0x20050000,0x20050100};
 auto launch=[&](uint16_t layer=23){rpu_launch_g05_gate_up_paired_nvfp4_acc16_kernel(spm[0],owners[0],owners[2],spm[1],spm[2],owners[1],owners[3],spm[3],spm[4],layer);};
 for(int layer:{0,18,23}){launch(layer);assert(gp(0)==spm[0]&&gp(4)==spm[1]&&gp(38)==spm[2]&&gp(24)==spm[3]&&gp(36)==spm[4]);int regs[]={2,20,32,34};for(int i=0;i<4;++i)assert(gp(regs[i])==owners[i].address>>8);auto&r=RpuKernelGraph::active().kernel.regs;assert(r.at(27)==layer&&r.at(8)==32&&gp(10)==512&&r.at(12)==1024&&r.at(13)==16&&r.at(15)==8&&gp(16)==2048&&gp(18)==1024&&gp(22)==8192&&r.at(26)==1);}
 rejects([&]{launch(24);});rejects([&]{launch(uint16_t(-1));});
 for(int i=0;i<5;++i){auto saved=spm[i];spm[i]=0;rejects([&]{launch();});spm[i]=saved+1;rejects([&]{launch();});spm[i]=0x207fff00;rejects([&]{launch();});spm[i]=saved;}
 for(int i=0;i<5;++i)for(int j=0;j<i;++j){auto saved=spm[i];spm[i]=spm[j];rejects([&]{launch();});spm[i]=saved;}
 for(int i=0;i<4;++i)for(int fault=0;fault<11;++fault){auto saved=owners[i];auto&t=owners[i];if(fault==0)t.present=false;if(fault==1)t.dtype=at::kHalf;if(fault==2)t.contiguous=false;if(fault==3)t.dev.kind=0;if(fault==4)t.dev.index=1;if(fault==5)t.offset=1;if(fault==6)--t.shape[0];if(fault==7)t.address=0;if(fault==8)++t.address;if(fault==9)t.address=1ULL<<40;if(fault==10)t.address=(1ULL<<40)-256;rejects([&]{launch();});owners[i]=saved;}
 // Fresh owners reach all four ABI pairs; they are not replaced with captured addresses.
 for(auto&t:owners){t.address+=0x80000000ULL;}launch();auto&g=RpuKernelGraph::active();for(int i=0;i<4;++i)assert(g.owners[i].owner.address==owners[i].address);
 allocator.initialized=false;rejects([&]{launch();});
}
'''
    _compile_run(tmp_path, "g05_nvfp4_pair_owners", cpp.replace("__FUNCTIONS__", "\n".join(functions)))


def test_pair_cold_scope_complete_mlp_rounding_and_old_ref_fallback(tmp_path, monkeypatch):
    import test_g05_silu_round_mul_native as fixture
    from test_g05_vision_owner_model import _block
    captured = []
    monkeypatch.setattr(fixture, "_compile_run", lambda _p, _n, cpp: captured.append(cpp))
    fixture.test_silu_round_mul_cold_manifest_resident_priority_and_fallback(tmp_path)
    cpp = captured[0].split("int main(){", 1)[0]
    cpp = cpp.replace("bool enabled()const{return false;}", "bool active=false;bool enabled()const{return active;}")
    cpp = cpp.replace("bool use_action_nvfp4(int64_t)const{return false;}",
                      "bool use_action_nvfp4(int64_t rows)const{return action_nvfp4_.enabled()&&action_mode_&&rows==32;}")
    cpp = cpp.replace("id==KernelId::G05_GATE_UP_FP16_ACC32_M32N512K1024||",
                      "id==KernelId::G05_GATE_UP_PAIRED_NVFP4_ACC16_M32N512K1024_N128||id==KernelId::G05_GATE_UP_FP16_ACC32_M32N512K1024||")
    cpp = cpp.replace("bool pair=true,fusion=true;", "bool pair=true,fusion=true,nv32=true;")
    cpp = cpp.replace("++queries;return (", "++queries;if(id==KernelId::G05_GATE_UP_PAIRED_NVFP4_ACC32_M32N512K1024_N128)return pair&&nv32;return (")
    # The shared fixture now retains both scalar-table and tensor owner identities.
    cpp = cpp.replace("std::vector<int64_t>action_nvfp4_arguments(int)const{assert(false);return {};}",
                      "std::vector<int64_t>action_nvfp4_arguments(int f)const{return {f};}")
    cpp = cpp.replace('name=="up"?0x28000:0x30000',
                      'name=="up"?0x28000:name=="gts"?0x40000:name=="uts"?0x40100:0x30000')
    cpp = cpp.replace("template<class...T>void rpu_launch_g05_gate_up_paired_nvfp4_acc16_kernel(T...){assert(false);}",
                      "void rpu_launch_g05_gate_up_paired_nvfp4_acc16_kernel(uint32_t,const at::Tensor&,const at::Tensor&,uint32_t,uint32_t,const at::Tensor&,const at::Tensor&,uint32_t,uint32_t,uint16_t);")
    cpp = cpp.replace("template<class...T>void rpu_launch_g05_gate_up_paired_nvfp4_acc32_kernel(T...){assert(false);}",
                      "void rpu_launch_g05_gate_up_paired_nvfp4_acc32_kernel(uint32_t,const at::Tensor&,const at::Tensor&,uint32_t,uint32_t,const at::Tensor&,const at::Tensor&,uint32_t,uint32_t,uint16_t);")
    source = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    marker = "if (use_action_nvfp4(chunk.len))"
    nv_manifest = _block(source[source.rindex(marker):], marker)
    cpp = cpp.replace("G05_NVFP4_PAIRED_ACC16=9",
                      "G05_NVFP4_V2_ACC16=8,G05_NVFP4_V2_ACC32=12,G05_NVFP4_PAIRED_ACC16=9")
    cpp = cpp.replace(" void mlp(int layer_idx,", " std::vector<Route> nv_plan(){ChunkInfo chunk;struct{std::vector<Route>routes;}manifest;if(use_action_nvfp4(chunk.len))" + nv_manifest + "return manifest.routes;}\n void mlp(int layer_idx,")
    cpp += r'''
void rpu_launch_g05_gate_up_paired_nvfp4_acc16_kernel(uint32_t in,const at::Tensor&g,const at::Tensor&u,uint32_t go,uint32_t uo,const at::Tensor&gs,const at::Tensor&us,uint32_t gt,uint32_t ut,uint16_t layer){assert(in==0x10000&&go==0x20000&&uo==0x28000&&gt==0x40000&&ut==0x40100&&layer<24&&g.owner==uint64_t(2*layer+1)&&u.owner==uint64_t(2*layer+2)&&gs.owner==g.owner+100&&us.owner==u.owner+200);launches.push_back("nv-pair");}
void rpu_launch_g05_gate_up_paired_nvfp4_acc32_kernel(uint32_t in,const at::Tensor&g,const at::Tensor&u,uint32_t go,uint32_t uo,const at::Tensor&gs,const at::Tensor&us,uint32_t gt,uint32_t ut,uint16_t layer){rpu_launch_g05_gate_up_paired_nvfp4_acc16_kernel(in,g,u,go,uo,gs,us,gt,ut,layer);launches.back()="nv-pair-acc32";}
int main(){
 Model m;m.action_nvfp4_.active=true;m.action_nvfp4_.values={1,2,3,4,5};m.linear_acc32_=false;m.resident=false;m.action_step_active_=m.action_loop_active_=true;m.action_num_steps_=10;
 for(auto&w:m.layer_weights_){w.gate_w.dtype=w.up_w.dtype=at::kByte;w.gate_w.cols=w.up_w.cols=512;w.gate_ws.dtype=w.up_ws.dtype=at::kByte;w.gate_ws.rows=w.up_ws.rows=64;w.gate_ws.cols=w.up_ws.cols=4096;w.gate_ws.owner=w.gate_w.owner+100;w.up_ws.owner=w.up_w.owner+200;}
 auto&cache=KernelCache::instance();m.prepare_action_gateup_pair();m.prepare_action_silu_round_mul();m.plan();
 assert(m.action_gateup_pair_available_&&m.action_gateup_pair_nvfp4_&&!m.action_gateup_pair_w8_&&m.use_action_silu_round_mul(32));
 assert(m.routes.size()==2&&m.routes[0].site==m.action_gateup_pair_site()&&m.routes[0].selector==9&&m.routes[0].args[6]==176&&m.routes[0].args[7]==128);
 auto nv=m.nv_plan();assert(nv.size()==3&&nv[0].args==std::vector<int64_t>({0})&&nv[1].args==std::vector<int64_t>({1})&&nv[2].args==std::vector<int64_t>({4}));
 auto valid=m.routes;auto hash=m.subclass_layout_hash();for(int layer=0;layer<24;++layer){launches.clear();m.mlp(layer);assert(launches==std::vector<std::string>({"nv-pair","round-mul","down"}));}
 {auto changed=m;++changed.action_nvfp4_.values[4];assert(changed.subclass_layout_hash()!=hash);}
 for(size_t i=0;i<m.routes[0].args.size();++i){m.routes=valid;++m.routes[0].args[i];reject([&]{m.mlp(0);});}m.routes=valid;
 m.routes[0].selector=8;reject([&]{m.mlp(0);});m.routes=valid;m.complete=false;reject([&]{m.mlp(0);});m.complete=true;
 reject([&]{m.launch_action_gateup_pair(-1,{});});reject([&]{m.launch_action_gateup_pair(24,{});});reject([&]{m.launch_action_gateup_pair(0,{31,2});});
 cache.fusion=false;m.prepare_action_silu_round_mul();m.plan();launches.clear();m.mlp(23);assert(launches==std::vector<std::string>({"nv-pair","silu","mul","down"}));
 auto queries=cache.queries;cache.pair=false;launches.clear();m.mlp(23);assert(launches==std::vector<std::string>({"nv-pair","silu","mul","down"})&&cache.queries==queries);m.prepare_action_gateup_pair();m.prepare_action_silu_round_mul();m.plan();assert(!m.action_gateup_pair_nvfp4_&&!m.action_gateup_pair_available_&&!m.action_silu_round_mul_available_&&m.subclass_layout_hash()!=hash);launches.clear();m.mlp(23);assert(launches==std::vector<std::string>({"gate","silu","up","mul","down"}));
 assert(m.nv_plan().size()==5);
 cache.pair=cache.fusion=true;m.prepare_action_gateup_pair();m.prepare_action_silu_round_mul();m.plan();assert(m.subclass_layout_hash()==hash);
 for(int fault=0;fault<11;++fault){auto x=m;if(fault==0)continue;if(fault==1)x.action_mode_=false;if(fault==2)x.layer_weights_.back().up_ws.dtype=at::kHalf;if(fault==3)x.layer_weights_.back().up_ws.rank=1;if(fault==4)x.layer_weights_.back().up_w.cols=1024;if(fault==5)x.layer_weights_.back().gate_w.location=c10::DeviceType::CPU;if(fault==6)x.layer_is_full_.back()=0;if(fault==7)x.use_silu_=false;if(fault==8)x.cores=4;if(fault==9)x.layer_weights_.back().up_w.storage_offset_=256;if(fault==10)x.layer_weights_.back().gate_ws.storage_offset_=256;x.prepare_action_gateup_pair();assert(!x.action_gateup_pair_nvfp4_&&!x.action_gateup_pair_available_);}
 // HIGH fuses only activation/MUL after the original mixed-NV paired kernel.
 // It keeps both actual Linear accumulation modes and independent owners.
 for(bool acc32:{false,true}){
  auto x=m;x.high_precision_=true;x.linear_acc32_=acc32;
  x.prepare_action_gateup_pair();x.prepare_action_silu_round_mul();x.prepare_action_silu_high_mul();x.plan();
  assert(x.action_gateup_pair_nvfp4_&&x.use_action_silu_high_mul(32)&&!x.use_action_silu_round_mul(32));
  assert(x.routes.size()==2&&x.routes[0].selector==(acc32?13:9)&&x.routes[1].site==QWEN35_ACTION_SILU_HIGH_MUL_SITE&&x.routes[1].args[16]==(acc32?32:16));
  for(int layer=0;layer<24;++layer){launches.clear();x.mlp(layer);assert(launches==std::vector<std::string>({acc32?"nv-pair-acc32":"nv-pair","high-mul","down"}));}
  cache.high_fusion=false;x.prepare_action_silu_high_mul();x.plan();launches.clear();x.mlp(23);assert(launches==std::vector<std::string>({acc32?"nv-pair-acc32":"nv-pair","high-silu","mul","down"}));cache.high_fusion=true;
  auto bad=x;bad.layer_weights_.back().up_ws.dtype=at::kHalf;bad.prepare_action_gateup_pair();bad.prepare_action_silu_high_mul();assert(!bad.action_gateup_pair_available_&&!bad.action_silu_high_mul_available_);
 }
}
'''
    _compile_run(tmp_path, "g05_nvfp4_pair_model", cpp)
    acc32 = cpp.replace('m.linear_acc32_=false;', 'm.linear_acc32_=true;', 1)
    head, main = acc32.split('int main(){', 1)
    main, high = main.split(' // HIGH fuses only activation/MUL after', 1)
    main = main.replace('"nv-pair"', '"nv-pair-acc32"').replace('selector==9', 'selector==13')
    main = main.replace('{3,32,512,1024,8,16,', '{3,32,512,1024,8,32,')
    # Old REF still containing ACC16 cannot qualify a true ACC32 pair.
    insert = "auto alias=m;alias.routes[0].site=4571887477213075828LL;reject([&]{alias.mlp(23);});cache.nv32=false;auto old=m;old.prepare_action_gateup_pair();assert(!old.action_gateup_pair_available_);cache.nv32=true;auto stale=m;stale.routes[0].args[5]=16;reject([&]{stale.mlp(23);});"
    assert ' auto valid=m.routes;' in main
    main = main.replace(' auto valid=m.routes;', 'assert(m.routes[0].args[5]==32);'+insert+' auto valid=m.routes;', 1)
    _compile_run(tmp_path, "g05_nvfp4_pair_acc32_model", head+'int main(){'+main+' // HIGH fuses only activation/MUL after'+high)
