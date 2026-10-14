"""Execute the actual typed NVFP4 wrappers and registers, including Pi limits."""
from pathlib import Path
from test_siglip_cold_profile import _body
from test_g05_q_gate_native import _compile_run

ROOT = Path(__file__).resolve().parents[1]


def test_nvfp4_typed_launch_preserves_precision_owners_and_model_boundaries(tmp_path):
    source = (ROOT / 'src/ops/rpu_linear.cpp').read_text()
    functions = []
    for marker in ('void check_pi05_resident_spm(', 'static void launch_action_nvfp4_v2_kernel(',
                   'void rpu_launch_pi05_nvfp4_v2_kernel(', 'void rpu_launch_g05_nvfp4_v2_kernel('):
        start = source.index(marker)
        functions.append(source[start:source.index('{', start)] + _body(source, marker))
    cpp = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <vector>
#include "rpu_linear_tiling.h"
#define TORCH_CHECK(ok,...) do{if(!(ok))throw std::runtime_error("rejected");}while(0)
namespace c10 {using Half=uint16_t;}
namespace at {
constexpr int kByte=1,kHalf=2;
enum class kPrivateUse1Type{device,CPU}; constexpr auto kPrivateUse1=kPrivateUse1Type::device;
using IntArrayRef=std::vector<int64_t>;
struct Device{ kPrivateUse1Type value=kPrivateUse1; int index=0;
 kPrivateUse1Type type()const{return value;} bool operator==(Device b)const{return value==b.value&&index==b.index;}};
struct Tensor{int dtype=kByte;bool contig=true;Device dev{};IntArrayRef shape{2048,512};uintptr_t address=0x1234567800ULL;
 int scalar_type()const{return dtype;}Device device()const{return dev;}bool is_contiguous()const{return contig;}
 IntArrayRef sizes()const{return shape;}void*data_ptr()const{return reinterpret_cast<void*>(address);}};
}
struct SpmAllocator{static constexpr size_t SPM_USABLE=8387584;bool initialized=true;
 bool is_initialized()const{return initialized;}uint32_t addr(int,int)const{return 0x20000000;}}allocator;
#define SPM_ALLOC allocator
int flushes=0;
void rpu_ddr_flush(void*){++flushes;}
enum class KernelId {ACC16=0,PARALLEL_LINEAR_NVFP4_V2_M384N48=100,PARALLEL_LINEAR_NVFP4_V2_M320N64=101};
KernelId autotile_nvfp4_acc16_kernel_id(int n){assert(n>=32&&n<=128&&n%16==0);return KernelId((128-n)/16);}
enum class GraphDdrRegisterRole{PackedFp4ModelWeight,Fp8BlockScale};
enum class GraphDdrRegisterAccess{Read};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct Abi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec{Abi abi;at::Tensor owner;};
struct Kernel{std::map<int,uint16_t> regs;void reset_regs(){regs.clear();}void set_regs(int r,uint16_t v){regs[r]=v;}void set_regs(int,uint32_t)=delete;};
struct Writer{void write(Kernel&);};
struct RpuKernelGraph{std::vector<GraphDdrRegisterOperandSpec> owners;Kernel kernel;KernelId id{};int stages=0;bool pending=false,written=false;
 static RpuKernelGraph&active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId k,std::vector<GraphDdrRegisterOperandSpec> values){assert(!pending&&!written&&values.size()==2);assert(values[0].abi.role==GraphDdrRegisterRole::PackedFp4ModelWeight&&values[0].abi.lo==2&&values[0].abi.hi==3);assert(values[1].abi.role==GraphDdrRegisterRole::Fp8BlockScale&&values[1].abi.lo==20&&values[1].abi.hi==21);id=k;owners=values;++stages;return {};}
 Kernel*get(KernelId k){assert(k==id&&!pending);pending=true;return &kernel;}};
void Writer::write(Kernel&k){auto&g=RpuKernelGraph::active();assert(g.pending&&!g.written);for(auto&o:g.owners){auto value=o.owner.address>>8;k.set_regs(o.abi.lo,uint16_t(value));k.set_regs(o.abi.hi,uint16_t(value>>16));}g.written=true;}
#define GET_KERNEL(id) RpuKernelGraph::active().get(id)
struct Queue{bool broadcast=false;int launches=0;std::vector<uint16_t> grid;
 void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> gr,std::vector<uint8_t> cores){auto&g=RpuKernelGraph::active();assert(g.pending&&g.written&&broadcast&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));grid=gr;++launches;g.pending=g.written=false;}}queue;
Queue*get_queue(int n){assert(n==8);return &queue;}
#define GET_QUEUE(n) get_queue(n)
__FUNCTIONS__
template<class F>void reject(F f){auto&g=RpuKernelGraph::active();const int launches=queue.launches,stages=g.stages,fl=flushes;try{f();assert(false);}catch(const std::runtime_error&){}assert(queue.launches==launches&&g.stages==stages&&flushes==fl&&!g.pending);}
uint32_t pair(int r){auto&v=RpuKernelGraph::active().kernel.regs;return uint32_t(v.at(r))|(uint32_t(v.at(r+1))<<16);}
int main(){
 constexpr uint32_t x=0x20010000,y=0x20030000,ts=0x20050000;
 auto&g=RpuKernelGraph::active();
 for(auto geometry:{std::array<int,3>{2048,1024,1},{4096,1024,1},{1024,2048,0},{1024,4096,0}}){
  auto n=geometry[0],k=geometry[1],part=geometry[2];at::Tensor w,s;s.address=0x3456789a00ULL;
  w.shape={n,k/2};s.shape={k/16,n};int ln=part?n/8:n,lk=part?k:k/8;
  for(int layer:{0,17,18,23}){
   rpu_launch_g05_nvfp4_v2_kernel(x,w,y,32,n,k,part,8,s,ts,layer,false);
   auto tile=rpu_pl_tiling::select_tile_nvfp4_acc16(32,ln,lk);
   assert(int(g.id)==(128-tile.n_tile)/16);
   assert(pair(0)==x&&pair(2)==w.address>>8&&pair(4)==y&&pair(6)==0&&pair(20)==s.address>>8&&pair(24)==ts);
   assert(g.kernel.regs.at(8)==32&&pair(10)==unsigned(ln)&&g.kernel.regs.at(12)==lk&&g.kernel.regs.at(13)==lk/64&&g.kernel.regs.at(27)==layer);
   assert(pair(22)==unsigned(8*ln*2)&&queue.grid==std::vector<uint16_t>({uint16_t((ln+tile.n_tile-1)/tile.n_tile),1,1}));
   rpu_launch_g05_nvfp4_v2_kernel(x,w,y,32,n,k,part,8,s,ts,layer,true);
   const bool gate_up=part==1&&n==4096;const int nt=gate_up?48:64;
   assert(g.id==(gate_up?KernelId::PARALLEL_LINEAR_NVFP4_V2_M384N48:KernelId::PARALLEL_LINEAR_NVFP4_V2_M320N64));
   assert(pair(2)==w.address>>8&&pair(20)==s.address>>8&&pair(24)==ts&&g.kernel.regs.at(27)==layer);
   assert(queue.grid==std::vector<uint16_t>({uint16_t((ln+nt-1)/nt),1,1}));
  }
  reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,y,50,n,k,part,8,s,ts,0,false);});
  reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,y,32,n,k,part,8,s,ts,24,false);});
  reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,y,32,n,k,part,4,s,ts,0,false);});
  reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,x,32,n,k,part,8,s,ts,0,false);});
  reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,y,32,n,k,part,8,s,x,0,false);});
  reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x+1,w,y,32,n,k,part,8,s,ts,0,false);});
  reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,0x207fff00,32,n,k,part,8,s,ts,0,false);});
  for(int fault=0;fault<5;++fault){auto bad=s;if(fault==0)bad.dtype=at::kHalf;if(fault==1)bad.contig=false;if(fault==2)--bad.shape[0];if(fault==3)bad.dev.value=at::kPrivateUse1Type::CPU;if(fault==4)bad.dev.index=1;
   reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,y,32,n,k,part,8,bad,ts,0,false);});}
  rpu_launch_pi05_nvfp4_v2_kernel(x,w,y,50,n,k,part,8,s,ts,17,false);assert(int(g.id)<100);
  rpu_launch_pi05_nvfp4_v2_kernel(x,w,y,50,n,k,part,8,s,ts,17,true);assert(int(g.id)>=100);
  reject([&]{rpu_launch_pi05_nvfp4_v2_kernel(x,w,y,32,n,k,part,8,s,ts,18,false);});
 }
 at::Tensor w,s;s.shape={64,2048};reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,y,32,1024,1024,1,8,s,ts,0,false);});
 allocator.initialized=false;reject([&]{rpu_launch_g05_nvfp4_v2_kernel(x,w,y,32,2048,1024,1,8,s,ts,0,false);});
}
'''
    _compile_run(tmp_path, 'g05_nvfp4_native', cpp.replace('__FUNCTIONS__', '\n'.join(functions)))
