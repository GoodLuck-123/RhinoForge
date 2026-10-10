"""Exercise the actual optional Down launcher's memory, owner and register contract."""
from test_hyvla_paired_kv import COMMON, ROOT, block, compile_run


def test_actual_down_n128_ranges_aliases_typed_owners_and_layer(tmp_path):
    launch = block((ROOT / 'src/fused/hyvla_nvfp4_down_n128.h').read_text(),
                   'inline void launch_hyvla_nvfp4_down_n128_spm(')
    source = COMMON + r'''
#include "ops/rpu_linear_tiling.h"
enum class KernelId {HYVLA_EXPERT_DOWN_NVFP4_S51_N128};
enum class GraphDdrRegisterRole {PackedFp4ModelWeight,Fp8BlockScale};
enum class GraphDdrRegisterAccess {Read};
enum class GraphDdrRegisterEncoding {DevAddrShift8LoHi};
struct GraphDdrRegisterAbi {GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;
 bool ready=true;uint32_t base=0x20000000;
 bool is_initialized()const{return ready;}
 uint32_t addr(int c,uint32_t o)const{return base+uint32_t(c)*0x1000000u+o;}
} spm;
#define SPM_ALLOC spm
struct Kernel {std::array<uint16_t,68> regs{};
 void reset_regs(){regs.fill(0);}
 void set_regs(uint32_t i,uint16_t v){regs.at(i)=v;}
 void set_regs(uint32_t,uint32_t)=delete;
 void set_regs(uint32_t,int)=delete;
} kernel;
int fetched=0,queued=0,staged=0,flushed=0,kept=0;
bool available=true;
Kernel* get_kernel(KernelId id){assert(id==KernelId::HYVLA_EXPERT_DOWN_NVFP4_S51_N128);++fetched;return available?&kernel:nullptr;}
#define GET_KERNEL(id) get_kernel(id)
struct Writer {std::vector<GraphDdrRegisterOperandSpec> ops;
 void write(Kernel& k){for(const auto& o:ops){const uint64_t a=o.tensor.address>>8;
  k.set_regs(o.abi.lo,uint16_t(a));k.set_regs(o.abi.hi,uint16_t(a>>16));}}};
struct RpuKernelGraph {static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 void keep_alive(const at::Tensor& t){assert(t.defined());++kept;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> v){
  assert(id==KernelId::HYVLA_EXPERT_DOWN_NVFP4_S51_N128 && v.size()==2);++staged;
  assert(v[0].abi.role==GraphDdrRegisterRole::PackedFp4ModelWeight && v[0].abi.access==GraphDdrRegisterAccess::Read);
  assert(v[0].abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi && v[0].abi.lo==2 && v[0].abi.hi==3);
  assert(v[0].tensor.nbytes()==1024*1024);
  assert(v[1].abi.role==GraphDdrRegisterRole::Fp8BlockScale && v[1].abi.access==GraphDdrRegisterAccess::Read);
  assert(v[1].abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi && v[1].abi.lo==20 && v[1].abi.hi==21);
  assert(v[1].tensor.nbytes()==128*1024);return {v};}};
void rpu_ddr_flush(void*){++flushed;}
namespace rhino_lkn {uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uintptr_t>(p);}}
struct Queue {bool broadcast=false;
 void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
  assert(broadcast && grid==std::vector<uint16_t>({8,1,1}));
  assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));++queued;}
} queue;
#define GET_QUEUE(n) (&::queue)
LAUNCH
int main(){
 const uint32_t base=spm.base;
 const std::array<uint32_t,3> good{base+0x10000,base+0x200000,base+0x700000};
 at::Tensor weight;weight.shape={1024,1024};weight.dtype=at::kByte;
 at::Tensor scale;scale.shape={128,1024};scale.dtype=at::kByte;scale.address=weight.address+1024*1024;
 auto run=[&](std::array<uint32_t,3> a,bool pass,int rows=51,int cols=1024,int cores=8,int layer=0){
  int before=fetched,previous=queued;bool ok=true;
  try{launch_hyvla_nvfp4_down_n128_spm(a[0],weight,scale,a[2],uint16_t(layer),a[1],rows,cols,cores);}
  catch(const std::exception&){ok=false;}
  assert(ok==pass && fetched==before+int(pass) && queued==previous+int(pass));
  assert(staged==queued && flushed==2*queued && kept==2*queued);
 };
 auto pair=[&](int i){return uint32_t(kernel.regs[i])|(uint32_t(kernel.regs[i+1])<<16);};
 for(int li:{0,1,7,8,31}){
  run(good,true,51,1024,8,li);
  assert(pair(0)==good[0] && pair(4)==good[1] && pair(2)==weight.address>>8 && pair(6)==0);
  assert(pair(8)==51 && pair(10)==1024 && kernel.regs[12]==256 && kernel.regs[13]==4);
  assert(pair(16)==512 && pair(18)==2048 && pair(22)==16384 && kernel.regs[26]==0 && kernel.regs[27]==li);
  assert(pair(20)==scale.address>>8 && pair(24)==good[2]);
  assert(kernel.regs[14]==0 && kernel.regs[15]==8 && kernel.regs[64]==8 && kernel.regs[65]==1 && kernel.regs[66]==1);
 }
 const std::array<uint32_t,3> bytes{51*256*2,51*1024*2,128};
 for(int i=0;i<3;++i){
  for(auto b:{base-256,good[i]+1}){auto a=good;a[i]=b;run(a,false);}
  for(int j=0;j<3;++j)if(i!=j){auto a=good;a[i]=good[j];run(a,false);
   if(bytes[j]>256){a[i]=good[j]+256;run(a,false);}}
  auto a=good;a[i]=base+((SpmAllocator::SPM_USABLE-bytes[i])&~255u);run(a,true);
  a[i]+=256;run(a,false);
 }
 for(int li:{32,65535})run(good,false,51,1024,8,li);
 for(int r:{0,50,52,64})run(good,false,r);
 for(int c:{0,128,1023,1025,2048})run(good,false,51,c);
 for(int n:{0,1,4,9})run(good,false,51,1024,n);
 const auto good_weight=weight;
 for(int failure=0;failure<10;++failure){weight=good_weight;
  switch(failure){case 0:weight.present=false;break;case 1:weight.contiguous=false;break;
   case 2:weight.dtype=at::kHalf;break;case 3:weight.where.kind=c10::DeviceType::CPU;break;
   case 4:weight.shape={1024,1023};break;case 5:weight.shape={1023,1024};break;
   case 6:weight.shape={1,1024,1024};break;case 7:weight.address+=128;break;
   case 8:weight.address=(uint64_t{1}<<40)-256;break;case 9:weight.address=0;break;}
  run(good,false);
 }
 weight=good_weight;
 const auto good_scale=scale;
 for(int failure=0;failure<11;++failure){scale=good_scale;
  switch(failure){case 0:scale.present=false;break;case 1:scale.contiguous=false;break;
   case 2:scale.dtype=at::kHalf;break;case 3:scale.where.kind=c10::DeviceType::CPU;break;
   case 4:scale.shape={127,1024};break;case 5:scale.shape={128,1023};break;
   case 6:scale.shape={1,128,1024};break;case 7:scale.address+=128;break;
   case 8:scale.address=(uint64_t{1}<<40)-256;break;case 9:scale.where.index=1;break;
   case 10:scale.address=0;break;}
  run(good,false);
 }
 scale=good_scale;
 for(auto* t:{&weight,&scale}){auto address=t->address;
  t->address=(uint64_t{1}<<40)-t->nbytes();run(good,true);
  t->address=uint64_t{1}<<40;run(good,false);t->address=address;}
 {available=false;const int f=fetched,s=staged,q=queued,k=kept,flush=flushed;bool rejected=false;
  try{launch_hyvla_nvfp4_down_n128_spm(good[0],weight,scale,good[2],0,good[1],51,1024,8);}
  catch(const std::exception&){rejected=true;}
  assert(rejected && fetched==f+1 && staged==s && queued==q && kept==k && flushed==flush);available=true;}
 spm.ready=false;run(good,false);spm.ready=true;
 spm.base=0xf8ff0000;auto high=good;for(auto& a:high)a+=spm.base-base;run(high,false);
}
'''
    compile_run(tmp_path, 'nvfp4_down_n128', source.replace('LAUNCH', launch))
