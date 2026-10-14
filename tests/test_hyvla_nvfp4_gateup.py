"""Execute actual NVFP4 paired/fused launcher's operand and SPM checks."""
import re

import pytest
from test_hyvla_paired_kv import COMMON, ROOT, block, compile_run


@pytest.mark.parametrize('resident,input_reuse,overlap', [(False, False, False), (True, False, False), (False, True, False), (False, False, True)])
def test_actual_nvfp4_gateup_ranges_scales_layer_and_registers(tmp_path, resident, input_reuse, overlap):
    launch=block((ROOT/'src/ops/rpu_hyvla_gateup_nvfp4.h').read_text(),
                 'inline void launch_hyvla_gateup_nvfp4(')
    source=COMMON+r'''
#include "ops/rpu_linear_tiling.h"
enum class KernelId {HYVLA_NVFP4_GATEUP_RESIDENT_S51,HYVLA_NVFP4_GATEUP_SWIGLU_S51,HYVLA_NVFP4_GATEUP_SWIGLU_INPUT_REUSE_S51,HYVLA_NVFP4_GATEUP_SWIGLU_S51_OVERLAP};
constexpr auto expected_id=OVERLAP?KernelId::HYVLA_NVFP4_GATEUP_SWIGLU_S51_OVERLAP:RESIDENT?KernelId::HYVLA_NVFP4_GATEUP_RESIDENT_S51:
 (INPUT_REUSE?KernelId::HYVLA_NVFP4_GATEUP_SWIGLU_INPUT_REUSE_S51:KernelId::HYVLA_NVFP4_GATEUP_SWIGLU_S51);
enum class GraphDdrRegisterRole {PackedFp4ModelWeight,Fp8BlockScale};
enum class GraphDdrRegisterAccess {Read};
enum class GraphDdrRegisterEncoding {DevAddrShift8LoHi};
struct GraphDdrRegisterAbi {GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct Kernel {std::array<uint16_t,132> regs{};int counters=0;
 void reset_regs(){regs.fill(0);counters=0;}
 uint32_t set_regs(uint32_t i,uint16_t x){regs.at(i)=x;return 0;}
 uint32_t set_regs(uint32_t i,uint32_t x){assert((i==128||i==130)&&x==0);++counters;regs.at(i)=x;return 0;}
 uint32_t set_regs(uint32_t,int32_t)=delete;
} kernel;
int queued=0,staged=0,kept=0;
struct Writer {std::vector<GraphDdrRegisterOperandSpec> operands;
 void write(Kernel& k){for(auto& o:operands){auto a=o.tensor.address>>8;
  k.set_regs(o.abi.lo,uint16_t(a));k.set_regs(o.abi.hi,uint16_t(a>>16));}}};
struct RpuKernelGraph {
 static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 void keep_alive(const at::Tensor&){++kept;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> v){
  assert(id==expected_id);
  assert(v.size()==(RESIDENT?2:4));++staged;
  for(auto& o:v){bool scale=o.abi.lo==20||o.abi.lo==34;
   assert(o.abi.role==(scale?GraphDdrRegisterRole::Fp8BlockScale:GraphDdrRegisterRole::PackedFp4ModelWeight));
   assert(o.tensor.nbytes()==size_t(scale?131072:1048576));assert(o.abi.hi==o.abi.lo+1);}
  return {v};}
};
struct Queue {void set_broadcast_mode(bool b){assert(b);}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
  assert(grid==std::vector<uint16_t>({8,1,1}));assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));++queued;}
} queue;
Kernel* get_kernel(KernelId id){assert(id==expected_id);return &kernel;}
#define GET_KERNEL(id) get_kernel(id)
#define GET_QUEUE(n) (&::queue)
namespace rhino_lkn {uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uintptr_t>(p);}}
void rpu_ddr_flush(void*){}
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;bool ready=true;
 uint32_t base=0x20000000;bool is_initialized()const{return ready;}
 uint32_t addr(int c,uint32_t o)const{return base+uint32_t(c)*0x1000000u+o;}} spm;
#define SPM_ALLOC spm
LAUNCH
int main(){
 std::array<at::Tensor,4> ts;
 ts[0].shape=ts[1].shape={2048,512};ts[2].shape=ts[3].shape={64,2048};
 for(int i=0;i<4;++i){ts[i].dtype=at::kByte;ts[i].address+=i*0x400000;}
 std::array<uint32_t,7> a{0x20000000,0x20020000,RESIDENT?0x20030000u:0u,
  RESIDENT?0x20100000u:0u,RESIDENT?0x20120000u:0u,0x20040000,0x20040100};
 auto run=[&](bool expected,int rows=51,int layer=0,bool reuse=INPUT_REUSE,bool overlap=OVERLAP){const int q=queued,s=staged,k=kept;bool ok=true;
  try{if(overlap)launch_hyvla_gateup_nvfp4(a[0],a[1],a[2],a[3],a[4],rows,ts[0],ts[1],ts[2],ts[3],a[5],a[6],layer,RESIDENT,reuse,true);
   else if(reuse)launch_hyvla_gateup_nvfp4(a[0],a[1],a[2],a[3],a[4],rows,ts[0],ts[1],ts[2],ts[3],a[5],a[6],layer,RESIDENT,true);
   else launch_hyvla_gateup_nvfp4(a[0],a[1],a[2],a[3],a[4],rows,ts[0],ts[1],ts[2],ts[3],a[5],a[6],layer,RESIDENT);}
  catch(const std::exception&){ok=false;}assert(ok==expected&&queued==q+int(expected));
  assert(staged==s+int(expected)&&kept==k+4*int(expected));};
 for(int l:{0,31}){
  run(true,51,l);auto pair=[&](int i){return uint32_t(kernel.regs[i])|(uint32_t(kernel.regs[i+1])<<16);};
  assert(pair(8)==51&&pair(10)==256&&kernel.regs[12]==1024&&kernel.regs[13]==16);
  assert(pair(20)==ts[2].address>>8&&pair(34)==ts[3].address>>8);
  assert(pair(2)==(RESIDENT?a[3]:ts[0].address>>8));
  assert(pair(32)==(RESIDENT?a[4]:ts[1].address>>8));
  assert(pair(24)==a[5]&&pair(36)==a[6]&&kernel.regs[27]==l);
  assert(pair(22)==4096&&kernel.regs[26]==1&&kernel.regs[64]==8&&kernel.regs[65]==1);
  assert(pair(38)==(RESIDENT?a[2]:0)&&kernel.counters==(RESIDENT?0:2));
 }
 if(RESIDENT)run(false,51,0,true);
 if(RESIDENT || INPUT_REUSE)run(false,51,0,INPUT_REUSE,true);
 if(OVERLAP)run(false,51,0,true);
 for(int l:{32,65535})run(false,51,l);
 for(int r:{0,20,50,64})run(false,r);
 std::vector<int> used=RESIDENT?std::vector<int>{0,1,2,3,4,5,6}:std::vector<int>{0,1,5,6};
 const std::array<uint32_t,7> bytes{51*1024*2,51*256*2,51*256*2,131072,131072,128,128};
 for(int i:used){auto saved=a[i];
  a[i]+=1;run(false);a[i]=0x1fffff00;run(false);a[i]=0x207fff00;run(false);
  a[i]=spm.base+((SpmAllocator::SPM_USABLE-bytes[i])/256)*256;run(true);
  a[i]+=256;run(false);a[i]=saved;
  for(int j:used)if(i!=j){a[i]=a[j];run(false);a[i]=saved;}
 }
 if(!RESIDENT){for(int i:{2,3,4}){a[i]=0x20100000;run(false);a[i]=0;}}
 for(int i=0;i<4;++i){auto t=ts[i];
  ts[i].present=false;run(false);ts[i]=t;ts[i].contiguous=false;run(false);ts[i]=t;
  ts[i].dtype=at::kHalf;run(false);ts[i]=t;ts[i].shape[0]-=1;run(false);ts[i]=t;
  ts[i].shape[1]-=1;run(false);ts[i]=t;ts[i].where.index=1;run(false);ts[i]=t;
  ts[i].where.kind=c10::DeviceType::CPU;run(false);ts[i]=t;
  if(!RESIDENT||i>=2){for(uintptr_t x:{uintptr_t(0),t.address+128,uintptr_t(1)<<40,(uintptr_t(1)<<40)-256}){ts[i].address=x;run(false);}ts[i]=t;}
 }
 spm.ready=false;run(false);spm.ready=true;
 const auto old=spm.base;spm.base=0xf8ff0000;
 for(int i:used)a[i]+=spm.base-old;
 run(false);
}
'''
    source = source.replace('LAUNCH', launch)
    source = re.sub(r'\bRESIDENT\b', 'true' if resident else 'false', source)
    source = re.sub(r'\bOVERLAP\b', 'true' if overlap else 'false', source)
    source = re.sub(r'\bINPUT_REUSE\b', 'true' if input_reuse else 'false', source)
    compile_run(tmp_path,'nvfp4_gateup',source)
