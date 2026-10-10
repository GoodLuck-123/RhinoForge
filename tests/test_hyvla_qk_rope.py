"""Exercise the shared-frequency launcher's actual memory and typed-owner boundary."""
from test_hyvla_paired_kv import COMMON, ROOT, block, compile_run


def test_actual_launcher_ranges_tables_and_typed_registers(tmp_path):
    launch = block((ROOT/'src/fused/hyvla_qk_rope.h').read_text(), 'inline void launch_hyvla_qk_rope_spm(')
    source = COMMON + r'''
enum class KernelId {HYVLA_EXPERT_QK_ROPE_S51};
enum class GraphDdrRegisterRole {SemanticInput};
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
struct Kernel {std::array<uint16_t,128> regs{};
 void reset_regs(){regs.fill(0);}
 void set_regs(uint32_t i,uint16_t v){regs.at(i)=v;}
 void set_regs(uint32_t,int)=delete;
} kernel;
int fetched=0,queued=0,staged=0,flushed=0;
Kernel* get_kernel(){++fetched;return &kernel;}
#define GET_KERNEL(id) get_kernel()
struct Writer {std::vector<GraphDdrRegisterOperandSpec> ops;
 void write(Kernel& k){for(const auto& o:ops){const uint64_t a=o.tensor.address>>8;
  k.set_regs(o.abi.lo,uint16_t(a));k.set_regs(o.abi.hi,uint16_t(a>>16));}}};
struct RpuKernelGraph {static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> v){
  assert(id==KernelId::HYVLA_EXPERT_QK_ROPE_S51 && v.size()==2);++staged;
  for(int i=0;i<2;++i){assert(v[i].abi.role==GraphDdrRegisterRole::SemanticInput && v[i].abi.access==GraphDdrRegisterAccess::Read);
   assert(v[i].abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi && v[i].abi.lo==2*i && v[i].abi.hi==2*i+1);}
  return {v};}};
void rpu_ddr_flush(void*){++flushed;}
namespace rhino_lkn {uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uintptr_t>(p);}}
struct Queue {bool broadcast=false;
 void set_broadcast_mode(bool b){broadcast=b;}
 void enqueu_kernel(Kernel&,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
  assert(broadcast && grid==std::vector<uint16_t>({1,1,1}));
  assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));++queued;}
} queue;
#define GET_QUEUE(n) (&::queue)
LAUNCH
int main(){
 const uint32_t base=spm.base;
 const std::array<uint32_t,4> good{base+0x10000,base+0x200000,base+0x400000,base+0x600000};
 at::Tensor cos,sin;cos.shape=sin.shape={320,64};sin.address+=0x100000;
 auto run=[&](std::array<uint32_t,4> a,bool pass,int rows=51,int pos=240,int cores=8){
  int before=fetched,previous=queued;bool ok=true;
  try{launch_hyvla_qk_rope_spm(a[0],a[1],a[2],a[3],cos,sin,rows,pos,cores);}
  catch(const std::exception&){ok=false;}
  assert(ok==pass && fetched==before+int(pass) && queued==previous+int(pass));
  assert(staged==queued && flushed==2*queued);
 };
 run(good,true);
 auto pair=[&](int i){return uint32_t(kernel.regs[i])|(uint32_t(kernel.regs[i+1])<<16);};
 assert(pair(0)==cos.address>>8 && pair(2)==sin.address>>8);
 assert(pair(4)==good[0] && pair(6)==good[1] && pair(8)==240);
 assert(pair(16)==good[2] && pair(18)==good[3]);
 assert(kernel.regs[10]==128 && kernel.regs[11]==2 && kernel.regs[12]==51);
 assert(kernel.regs[13]==128 && kernel.regs[14]==1 && kernel.regs[15]==1);
 assert(kernel.regs[64]==1 && kernel.regs[65]==1 && kernel.regs[66]==1);
 const std::array<uint32_t,4> bytes{51*256*2,51*256*2,51*128*2,51*128*2};
 for(int i=0;i<4;++i){
  for(auto b:{base-256,good[i]+1,base+SpmAllocator::SPM_USABLE-256}){auto a=good;a[i]=b;run(a,false);}
  for(int j=0;j<4;++j)if(i!=j){auto a=good;a[i]=good[j];run(a,false);
   a[i]=good[j]+256;run(a,false);}
  auto a=good;a[i]=base+SpmAllocator::SPM_USABLE-bytes[i];run(a,true);
  a[i]+=256;run(a,false);
 }
 for(int n:{0,50,52,64})run(good,false,n);
 for(int n:{-1,0,112,127,129,191,193,207,209,223,225,239,241,256})run(good,false,51,n);
 for(int n:{0,1,4,9})run(good,false,51,240,n);
 const auto original_cos=cos,original_sin=sin;
 for(int position:{128,144,160,176,192,208,224,240}){
  cos.shape=sin.shape={position+51,64};run(good,true,51,position);
  assert(pair(8)==uint32_t(position));
  for(auto* t:{&cos,&sin}){--t->shape[0];run(good,false,51,position);++t->shape[0];}
  --cos.shape[0];--sin.shape[0];run(good,false,51,position);
  cos.shape=sin.shape={position+51,64};++sin.shape[0];run(good,false,51,position);
 }
 cos=original_cos;sin=original_sin;
 for(auto* t:{&cos,&sin})for(int failure=0;failure<10;++failure){
  cos=original_cos;sin=original_sin;
  switch(failure){case 0:t->present=false;break;case 1:t->contiguous=false;break;
   case 2:t->dtype=at::kChar;break;case 3:t->where.kind=c10::DeviceType::CPU;break;
   case 4:t->shape={290,64};break;case 5:t->shape={320,63};break;
   case 6:t->shape={1,320,64};break;case 7:t->address+=128;break;
   case 8:t->address=(uint64_t{1}<<40)-256;break;case 9:t->shape={319,64};break;}
  run(good,false);
 }
 cos=original_cos;sin=original_sin;cos.shape=sin.shape={291,64};run(good,true);
 sin.where.index=1;run(good,false);sin.where.index=0;
 spm.ready=false;run(good,false);spm.ready=true;
 spm.base=0xf8ff0000;auto high=good;for(auto& a:high)a+=spm.base-base;run(high,false);
}
'''
    compile_run(tmp_path, 'qk_rope_launcher', source.replace('LAUNCH', launch))
