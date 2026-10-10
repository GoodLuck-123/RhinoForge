"""Execute the actual grouped-projection helper and selected scalar ABI on host."""
import re

import pytest
from test_hyvla_paired_kv import COMMON, ROOT, block, compile_run


@pytest.mark.parametrize('qkv', [False, True])
def test_actual_qkv_helper_operand_vectors_ranges_aliases_and_registers(tmp_path, qkv):
    launch = block((ROOT / 'src/fused/hyvla_nvfp4_qkv.h').read_text(),
                   'inline void launch_hyvla_nvfp4_qkv_spm(')
    source = COMMON + r'''
#include "ops/rpu_linear_tiling.h"
enum class KernelId {HYVLA_EXPERT_KV_NVFP4_S51,HYVLA_EXPERT_QKV_NVFP4_S51};
constexpr auto expected_id=QKV?KernelId::HYVLA_EXPERT_QKV_NVFP4_S51:KernelId::HYVLA_EXPERT_KV_NVFP4_S51;
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
Kernel* get_kernel(KernelId id){assert(id==expected_id);++fetched;return available?&kernel:nullptr;}
#define GET_KERNEL(id) get_kernel(id)
struct Writer {std::vector<GraphDdrRegisterOperandSpec> ops;
 void write(Kernel& k){for(const auto& o:ops){const uint64_t a=o.tensor.address>>8;
  k.set_regs(o.abi.lo,uint16_t(a));k.set_regs(o.abi.hi,uint16_t(a>>16));}}};
struct RpuKernelGraph {static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 void keep_alive(const at::Tensor& t){assert(t.defined());++kept;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> v){
  assert(id==expected_id && v.size()==(QKV?6:4));++staged;
  std::array<int,3> ws{2,32,40},ss{20,34,42};
  for(size_t i=0;i<v.size();++i){const auto& o=v[i];const bool s=i%2;const int p=i/2;
   assert(o.abi.role==(s?GraphDdrRegisterRole::Fp8BlockScale:GraphDdrRegisterRole::PackedFp4ModelWeight));
   assert(o.abi.access==GraphDdrRegisterAccess::Read && o.abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
   assert(o.abi.lo==(s?ss[p]:ws[p]) && o.abi.hi==o.abi.lo+1);
   assert(o.tensor.nbytes()==size_t((QKV&&p==0?2048:1024)*(s?64:512)));}
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
 const uint32_t base=spm.base;const size_t count=QKV?3:2;
 uint32_t input=base+0x10000;
 std::vector<uint32_t> out,table;
 std::vector<at::Tensor> weight(count),scale(count);
 for(size_t i=0;i<count;++i){int n=QKV&&i==0?2048:1024;
  weight[i].shape={n,512};weight[i].dtype=at::kByte;weight[i].address+=i*0x400000;
  scale[i].shape={64,n};scale[i].dtype=at::kByte;scale[i].address=weight[i].address+0x200000;
  out.push_back(base+0x100000+i*0x100000);table.push_back(base+0x700000+i*256);}
 auto run=[&](bool pass,int rows=51,int cores=8,int layer=0){
  const int f=fetched,q=queued;bool ok=true;
  try{launch_hyvla_nvfp4_qkv_spm(input,weight,scale,table,out,uint16_t(layer),QKV,rows,cores);}
  catch(const std::exception&){ok=false;}
  assert(ok==pass && fetched==f+int(pass) && queued==q+int(pass));
  assert(staged==queued && flushed==int(2*count)*queued && kept==int(2*count)*queued);
 };
 auto pair=[&](int i){return uint32_t(kernel.regs[i])|(uint32_t(kernel.regs[i+1])<<16);};
 const std::array<int,3> ws{2,32,40},ss{20,34,42},ts{24,36,44},os{4,38,46};
 for(int layer:{0,7,8,31}){run(true,51,8,layer);
  const int n=QKV?256:128;
  assert(pair(0)==input && pair(6)==0 && pair(8)==51 && pair(10)==uint32_t(n));
  assert(kernel.regs[12]==1024 && kernel.regs[13]==16 && kernel.regs[14]==0 && kernel.regs[15]==8);
  assert(pair(16)==2048 && pair(18)==uint32_t(2*n) && pair(22)==uint32_t(16*n));
  assert(kernel.regs[26]==1 && kernel.regs[27]==layer && kernel.regs[64]==8 && kernel.regs[65]==1 && kernel.regs[66]==1);
  for(size_t i=0;i<count;++i){assert(pair(ws[i])==weight[i].address>>8 && pair(ss[i])==scale[i].address>>8);
   assert(pair(ts[i])==table[i] && pair(os[i])==out[i]);}
 }
 for(int layer:{32,65535})run(false,51,8,layer);
 for(int rows:{0,50,52,64})run(false,rows);
 for(int cores:{0,1,4,9})run(false,51,cores);
 for(int which=0;which<4;++which){
  if(which==0){auto t=weight.back();weight.pop_back();run(false);weight.push_back(t);weight.push_back(t);run(false);weight.pop_back();}
  if(which==1){auto t=scale.back();scale.pop_back();run(false);scale.push_back(t);scale.push_back(t);run(false);scale.pop_back();}
  if(which==2){auto t=table.back();table.pop_back();run(false);table.push_back(t);table.push_back(t);run(false);table.pop_back();}
  if(which==3){auto t=out.back();out.pop_back();run(false);out.push_back(t);out.push_back(t);run(false);out.pop_back();}
 }
 std::vector<uint32_t*> addresses{&input};std::vector<uint32_t> bytes{51*1024*2};
 for(size_t i=0;i<count;++i){addresses.push_back(&out[i]);bytes.push_back(51*(QKV&&i==0?256:128)*2);
  addresses.push_back(&table[i]);bytes.push_back(128);}
 for(size_t i=0;i<addresses.size();++i){auto& a=*addresses[i];const auto good=a;
  for(uint32_t bad:{base-256,good+1}){a=bad;run(false);}a=good;
  for(size_t j=0;j<addresses.size();++j)if(i!=j){a=*addresses[j];run(false);a=good;
   if(bytes[j]>256){a=*addresses[j]+256;run(false);a=good;}}
  a=base+((SpmAllocator::SPM_USABLE-bytes[i])&~255u);run(true);a+=256;run(false);a=good;
 }
 for(auto* vector:{&weight,&scale})for(auto& t:*vector){auto good=t;
  for(int failure=0;failure<11;++failure){t=good;
   switch(failure){case 0:t.present=false;break;case 1:t.contiguous=false;break;
    case 2:t.dtype=at::kHalf;break;case 3:t.where.kind=c10::DeviceType::CPU;break;
    case 4:--t.shape[0];break;case 5:--t.shape[1];break;
    case 6:t.shape.push_back(1);break;case 7:t.address+=128;break;
    case 8:t.address=(uint64_t{1}<<40)-256;break;case 9:t.where.index=1;break;
    case 10:t.address=0;break;}run(false);}
  t=good;t.address=(uint64_t{1}<<40)-t.nbytes();run(true);
  t.address=uint64_t{1}<<40;run(false);t=good;
 }
 {available=false;const int f=fetched,s=staged,q=queued,k=kept;bool rejected=false;
  try{launch_hyvla_nvfp4_qkv_spm(input,weight,scale,table,out,0,QKV);}
  catch(const std::exception&){rejected=true;}
  assert(rejected && fetched==f+1 && staged==s && queued==q && kept==k);available=true;}
 spm.ready=false;run(false);spm.ready=true;
 spm.base=0xf8ff0000;for(auto* a:addresses)*a+=spm.base-base;run(false);
}
'''
    source = source.replace('LAUNCH', launch)
    source = re.sub(r'\bQKV\b', 'true' if qkv else 'false', source)
    compile_run(tmp_path, 'nvfp4_qkv', source)
