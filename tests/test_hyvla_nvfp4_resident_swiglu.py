"""Execute actual resident-fusion helper guards and ABI, including virtual-grid counts."""
import pytest

from test_hyvla_paired_kv import COMMON, ROOT, block, compile_run


@pytest.mark.parametrize('overlap', [False, True])
def test_actual_resident_swiglu_bounds_tables_counter_and_activation_packet(tmp_path, overlap):
    launch=block((ROOT/'src/fused/hyvla_nvfp4_resident_swiglu.h').read_text(),
                 'inline void launch_hyvla_nvfp4_resident_swiglu_spm(')
    source=COMMON+r'''
#include "ops/rpu_linear_tiling.h"
enum class KernelId {HYVLA_NVFP4_RESIDENT_SWIGLU_S51,HYVLA_NVFP4_RESIDENT_SWIGLU_S51_OVERLAP};
constexpr auto expected_id=KernelId::EXPECTED_ID;
enum class GraphDdrRegisterRole {Fp8BlockScale};
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
struct Kernel {std::array<uint16_t,130> regs{};int counters=0;
 void reset_regs(){regs.fill(0);counters=0;}
 void set_regs(uint32_t i,uint16_t v){assert(i<=66 || (i>=4096 && i<4160));regs.at(i)=v;}
 void set_regs(uint32_t i,uint32_t v){assert(i==128 && v==0);++counters;regs.at(i)=v;}
 void set_regs(uint32_t,int)=delete;
} kernel;
int fetched=0,queued=0,staged=0,flushed=0,kept=0;
bool available=true;
Kernel* get_kernel(KernelId id){assert(id==expected_id);++fetched;return available?&kernel:nullptr;}
#define GET_KERNEL(id) get_kernel(id)
struct Writer {std::vector<GraphDdrRegisterOperandSpec> ops;
 void write(Kernel& k){for(const auto& o:ops){assert(o.abi.lo>=0 && o.abi.hi<64);const uint64_t a=o.tensor.address>>8;
  k.set_regs(o.abi.lo,uint16_t(a));k.set_regs(o.abi.hi,uint16_t(a>>16));}}};
struct RpuKernelGraph {static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 void keep_alive(const at::Tensor& t){assert(t.defined());++kept;}
 Writer stage_kernel_ddr_registers(KernelId,std::vector<GraphDdrRegisterOperandSpec> v){
  assert(v.size()==2);++staged;
  for(size_t i=0;i<2;++i){const auto& o=v[i];
   assert(o.abi.role==GraphDdrRegisterRole::Fp8BlockScale && o.abi.access==GraphDdrRegisterAccess::Read);
   assert(o.abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi && o.abi.lo==(i?34:20) && o.abi.hi==o.abi.lo+1);
   assert(o.tensor.nbytes()==64*2048);}
  return {v};}};
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
 std::array<uint32_t,7> a{base+0x10000,base+0x100000,base+0x110000,
   base+0x200000,base+0x220000,base+0x700000,base+0x700100};
 const std::array<uint32_t,7> bytes{51*1024*2,51*256*2,51*256*2,131072,131072,128,128};
 at::Tensor g,u;g.dtype=u.dtype=at::kByte;g.shape=u.shape={64,2048};u.address+=0x400000;
 auto run=[&](bool pass,int rows=51,int cores=8,int layer=0){
  const int f=fetched,q=queued;bool ok=true;
  try{launch_hyvla_nvfp4_resident_swiglu_spm(a[0],a[1],a[2],a[3],a[4],g,u,a[5],a[6],uint16_t(layer),rows,cores EXTRA_ARG);}
  catch(const std::exception&){ok=false;}
  assert(ok==pass && fetched==f+int(pass) && queued==q+int(pass));
  assert(staged==queued && kept==2*queued && flushed==2*queued);
 };
 auto pair=[&](int i){return uint32_t(kernel.regs[i])|(uint32_t(kernel.regs[i+1])<<16);};
 for(int layer:{0,7,8,31}){run(true,51,8,layer);
  assert(pair(0)==a[0] && pair(2)==a[3] && pair(4)==a[1] && pair(32)==a[4] && pair(38)==a[2]);
  assert(pair(20)==g.address>>8 && pair(34)==u.address>>8 && pair(24)==a[5] && pair(36)==a[6]);
  assert(pair(8)==51 && pair(10)==256 && kernel.regs[12]==1024 && kernel.regs[13]==16);
  assert(pair(16)==2048 && pair(18)==512 && pair(22)==4096 && kernel.regs[26]==1 && kernel.regs[27]==layer);
  assert(kernel.regs[48]==32512 && kernel.regs[49]==13056);
  assert(pair(52)==a[1] && pair(54)==a[2] && pair(56)==a[1]);
  assert(kernel.counters==1 && kernel.regs[64]==8 && kernel.regs[65]==1 && kernel.regs[66]==1);
 }
 for(int layer:{32,65535})run(false,51,8,layer);
 for(int rows:{0,50,52,64})run(false,rows);
 for(int cores:{0,1,4,9})run(false,51,cores);
 for(size_t i=0;i<a.size();++i){const auto good=a[i];
  for(uint32_t bad:{base-256,good+1}){a[i]=bad;run(false);}a[i]=good;
  for(size_t j=0;j<a.size();++j)if(i!=j){a[i]=a[j];run(false);a[i]=good;
   if(bytes[j]>256){a[i]=a[j]+256;run(false);a[i]=good;}}
  a[i]=base+((SpmAllocator::SPM_USABLE-bytes[i])&~255u);run(true);a[i]+=256;run(false);a[i]=good;
 }
 for(auto* p:{&g,&u}){const auto good=*p;
  for(int failure=0;failure<11;++failure){*p=good;
   switch(failure){case 0:p->present=false;break;case 1:p->contiguous=false;break;
    case 2:p->dtype=at::kHalf;break;case 3:p->where.kind=c10::DeviceType::CPU;break;
    case 4:--p->shape[0];break;case 5:--p->shape[1];break;
    case 6:p->shape.push_back(1);break;case 7:p->address+=128;break;
    case 8:p->address=(uint64_t{1}<<40)-256;break;case 9:p->where.index=1;break;
    case 10:p->address=0;break;}run(false);}
  *p=good;p->address=(uint64_t{1}<<40)-p->nbytes();run(true);
  p->address=uint64_t{1}<<40;run(false);*p=good;
 }
 {available=false;const int f=fetched,s=staged,q=queued,k=kept;bool rejected=false;
  try{launch_hyvla_nvfp4_resident_swiglu_spm(a[0],a[1],a[2],a[3],a[4],g,u,a[5],a[6],0,51,8 EXTRA_ARG);}
  catch(const std::exception&){rejected=true;}
  assert(rejected && fetched==f+1 && staged==s && queued==q && kept==k);available=true;}
 spm.ready=false;run(false);spm.ready=true;
 spm.base=0xf8ff0000;for(auto& p:a)p+=spm.base-base;run(false);
}
'''
    source=source.replace('EXTRA_ARG', ', true' if overlap else '').replace('EXPECTED_ID', 'HYVLA_NVFP4_RESIDENT_SWIGLU_S51' + ('_OVERLAP' if overlap else ''))
    compile_run(tmp_path,'nvfp4_resident_swiglu',source.replace('LAUNCH',launch))
