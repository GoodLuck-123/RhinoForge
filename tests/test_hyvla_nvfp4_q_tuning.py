"""Execute the Q helper's actual ABI and failure boundaries."""
import re
import pytest
from test_hyvla_paired_kv import COMMON, ROOT, block, compile_run


@pytest.mark.parametrize('overlap',[False,True])
def test_actual_q_tuning_helper_owned_abi_bounds_and_dispatch(tmp_path,overlap):
    launch=block((ROOT/'src/fused/hyvla_nvfp4_q_tuning.h').read_text(),
                 'inline void launch_hyvla_nvfp4_q_tuning_spm(')
    source=COMMON+r'''
#include "ops/rpu_linear_tiling.h"
enum class KernelId {HYVLA_EXPERT_Q_NVFP4_S51_N32,HYVLA_EXPERT_Q_NVFP4_S51_OVERLAP};
constexpr auto selected=OVERLAP?KernelId::HYVLA_EXPERT_Q_NVFP4_S51_OVERLAP:KernelId::HYVLA_EXPERT_Q_NVFP4_S51_N32;
enum class GraphDdrRegisterRole {PackedFp4ModelWeight,Fp8BlockScale};
enum class GraphDdrRegisterAccess {Read};
enum class GraphDdrRegisterEncoding {DevAddrShift8LoHi};
struct GraphDdrRegisterAbi {GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;bool ready=true;uint32_t base=0x20000000;
 bool is_initialized()const{return ready;}uint32_t addr(int c,uint32_t o)const{return base+uint32_t(c)*0x1000000u+o;}} spm;
#define SPM_ALLOC spm
struct Kernel {std::array<uint16_t,68> regs{};void reset_regs(){regs.fill(0);}
 void set_regs(uint32_t i,uint16_t v){regs.at(i)=v;}
 void set_regs(uint32_t,int)=delete;void set_regs(uint32_t,uint32_t)=delete;} kernel;
int fetched=0,queued=0,staged=0,kept=0,flushed=0;bool available=true;
Kernel* get_kernel(KernelId id){assert(id==selected);++fetched;return available?&kernel:nullptr;}
#define GET_KERNEL(id) get_kernel(id)
struct Writer {std::vector<GraphDdrRegisterOperandSpec> ops;
 void write(Kernel& k){for(const auto& o:ops){auto a=o.tensor.address>>8;
  k.set_regs(o.abi.lo,uint16_t(a));k.set_regs(o.abi.hi,uint16_t(a>>16));}}};
struct RpuKernelGraph {static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 void keep_alive(const at::Tensor&){++kept;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> v){
  assert(id==selected && v.size()==2);++staged;
  for(size_t i=0;i<2;++i){const auto& o=v[i];assert(o.abi.lo==(i?20:2) && o.abi.hi==o.abi.lo+1);
   assert(o.abi.role==(i?GraphDdrRegisterRole::Fp8BlockScale:GraphDdrRegisterRole::PackedFp4ModelWeight));
   assert(o.abi.access==GraphDdrRegisterAccess::Read && o.abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
   assert(o.tensor.nbytes()==size_t(2048*(i?64:512)));}return {v};}};
void rpu_ddr_flush(void*){++flushed;}
namespace rhino_lkn {uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uintptr_t>(p);}}
struct Queue {bool broadcast=false;void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
  assert(broadcast && grid==std::vector<uint16_t>({OVERLAP?4u:8u,1,1}));
  assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));++queued;}} queue;
#define GET_QUEUE(n) (&::queue)
LAUNCH
int main(){
 const uint32_t base=spm.base;uint32_t input=base+0x10000,output=base+0x100000,table=base+0x700000;
 at::Tensor w,s;w.dtype=s.dtype=at::kByte;w.shape={2048,512};s.shape={64,2048};s.address+=0x200000;
 auto run=[&](bool pass,int rows=51,int cores=8,int layer=0){const int f=fetched,q=queued;bool ok=true;
  try{launch_hyvla_nvfp4_q_tuning_spm(input,w,s,table,output,uint16_t(layer),OVERLAP,rows,cores);}
  catch(const std::exception&){ok=false;}assert(ok==pass && fetched==f+int(pass) && queued==q+int(pass));
  assert(staged==queued && kept==2*queued && flushed==2*queued);};
 auto pair=[&](int i){return uint32_t(kernel.regs[i])|(uint32_t(kernel.regs[i+1])<<16);};
 for(int layer:{0,7,8,31}){run(true,51,8,layer);
  assert(pair(0)==input && pair(2)==w.address>>8 && pair(4)==output && pair(6)==0 && pair(8)==51 && pair(10)==256);
  assert(kernel.regs[12]==1024 && kernel.regs[13]==16 && kernel.regs[14]==0 && kernel.regs[15]==8);
  assert(pair(16)==2048 && pair(18)==512 && pair(20)==s.address>>8 && pair(22)==4096 && pair(24)==table);
  assert(kernel.regs[26]==1 && kernel.regs[27]==layer && kernel.regs[64]==(OVERLAP?4:8));}
 for(int r:{0,50,52,64})run(false,r);
 for(int c:{0,1,4,9})run(false,51,c);
 for(int l:{32,65535})run(false,51,8,l);
 const std::array<uint32_t*,3> addresses{&input,&output,&table};
 const std::array<uint32_t,3> bytes{51*1024*2,51*256*2,128};
 for(size_t i=0;i<3;++i){auto& a=*addresses[i];const auto good=a;
  for(uint32_t bad:{base-256,good+1}){a=bad;run(false);}a=good;
  for(size_t j=0;j<3;++j)if(i!=j){a=*addresses[j];run(false);a=good;
   if(bytes[j]>256){a=*addresses[j]+256;run(false);a=good;}}
  a=base+((SpmAllocator::SPM_USABLE-bytes[i])&~255u);run(true);a+=256;run(false);a=good;}
 for(auto* t:{&w,&s}){auto good=*t;
  for(int failure=0;failure<11;++failure){*t=good;
   switch(failure){case 0:t->present=false;break;case 1:t->contiguous=false;break;
    case 2:t->dtype=at::kHalf;break;case 3:t->where.kind=c10::DeviceType::CPU;break;
    case 4:--t->shape[0];break;case 5:--t->shape[1];break;case 6:t->shape.push_back(1);break;
    case 7:t->address+=128;break;case 8:t->address=(uint64_t{1}<<40)-256;break;
    case 9:t->where.index=1;break;case 10:t->address=0;break;}run(false);}
  *t=good;t->address=(uint64_t{1}<<40)-t->nbytes();run(true);t->address=uint64_t{1}<<40;run(false);*t=good;}
 available=false;const int f=fetched,q=queued,k=kept;bool rejected=false;
 try{launch_hyvla_nvfp4_q_tuning_spm(input,w,s,table,output,0,OVERLAP);}catch(const std::exception&){rejected=true;}
 assert(rejected && fetched==f+1 && queued==q && kept==k);available=true;
 spm.ready=false;run(false);spm.ready=true;
 spm.base=0xf8ff0000;for(auto* a:addresses)*a+=spm.base-base;run(false);
}
'''
    source=source.replace('LAUNCH',launch)
    source=re.sub(r'\bOVERLAP\b','true' if overlap else 'false',source)
    compile_run(tmp_path,'nvfp4_q_tuning',source)
