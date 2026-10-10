"""Exercise the RoPE plus padded head-norm launcher's actual memory and typed-owner boundary."""
from test_hyvla_paired_kv import COMMON, ROOT, block, compile_run


def test_actual_rope_norm_launcher_padded_ranges_tables_barriers_and_typed_registers(tmp_path):
    launch = block((ROOT/'src/fused/hyvla_qk_rope_norm_kv.h').read_text(), 'inline void launch_hyvla_qk_rope_norm_kv_spm(')
    source = COMMON + r'''
enum class KernelId {HYVLA_EXPERT_QK_ROPE_NORM_KV_S51};
enum class GraphDdrRegisterRole {SemanticInput,KeyCache,ValueCache};
enum class GraphDdrRegisterAccess {Read,Write};
enum class GraphDdrRegisterEncoding {DevAddrShift8LoHi};
struct GraphDdrRegisterAbi {GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};
struct GraphDdrRegisterOperandSpec {GraphDdrRegisterAbi abi;at::Tensor tensor;};
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;
 bool ready=true;uint32_t base=0x20000000;
 bool is_initialized()const{return ready;}
 uint32_t addr(int c,uint32_t o)const{return base+uint32_t(c)*0x1000000u+o;}
} spm;
#define SPM_ALLOC spm
struct Kernel {std::array<uint16_t,136> regs{};std::array<uint32_t,4> counters{99,99,99,99};
 void reset_regs(){regs.fill(0);counters={99,99,99,99};}
 void set_regs(uint32_t i,uint16_t v){assert(i<=66);regs.at(i)=v;}
 void set_regs(uint32_t i,uint32_t v){assert((i==128||i==130||i==132||i==134)&&v==0);counters.at((i-128)/2)=v;}
 void set_regs(uint32_t,int)=delete;
} kernel;
int fetched=0,queued=0,staged=0,flushed=0,kept=0;bool available=true;
Kernel* get_kernel(KernelId id){assert(id==KernelId::HYVLA_EXPERT_QK_ROPE_NORM_KV_S51);++fetched;return available?&kernel:nullptr;}
#define GET_KERNEL(id) get_kernel(id)
struct Writer {std::vector<GraphDdrRegisterOperandSpec> ops;
 void write(Kernel& k){for(const auto& o:ops){const uint64_t a=o.tensor.address>>8;
  k.set_regs(o.abi.lo,uint16_t(a));k.set_regs(o.abi.hi,uint16_t(a>>16));}}};
struct RpuKernelGraph {static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 void keep_alive(const at::Tensor& t){assert(t.defined());++kept;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> v){
  assert(id==KernelId::HYVLA_EXPERT_QK_ROPE_NORM_KV_S51 && v.size()==4);++staged;
  // Match the SDK typed DDR launch-register bound; the original88/92 packet
  // failed on hardware even though its parameter indices assembled correctly.
  for(const auto& o:v)assert(o.abi.lo>=0 && o.abi.hi<64);
  for(int i=0;i<2;++i){assert(v[i].abi.role==GraphDdrRegisterRole::SemanticInput && v[i].abi.access==GraphDdrRegisterAccess::Read);
   assert(v[i].abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi && v[i].abi.lo==2*i && v[i].abi.hi==2*i+1);}
  for(int i=2;i<4;++i){assert(v[i].abi.role==(i==2?GraphDdrRegisterRole::KeyCache:GraphDdrRegisterRole::ValueCache));
   assert(v[i].abi.access==GraphDdrRegisterAccess::Write && v[i].abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);
   assert(v[i].abi.lo==20+4*(i-2) && v[i].abi.hi==21+4*(i-2));}
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
 const std::array<uint32_t,7> good{base+0x10000,base+0x200000,base+0x400000,base+0x600000,base,base+0x8000,base+0x300000};
 at::Tensor key,value;key.address=0x18000000;value.address=0x19000000;
 auto plan=exact_plan();
 at::Tensor cos,sin;cos.shape=sin.shape={320,64};sin.address+=0x100000;
 auto run=[&](std::array<uint32_t,7> a,bool pass,int rows=51,int pos=240,int cores=8,int qr=112,int kr=64,double eps=1e-5){
  int before=fetched,previous=queued;bool ok=true;
  try{launch_hyvla_qk_rope_norm_kv_spm(a[0],a[1],a[2],a[3],a[4],a[5],a[6],cos,sin,key,value,rows,pos,qr,kr,eps,cores,plan);}
  catch(const std::exception&){ok=false;}
  assert(ok==pass && fetched==before+int(pass) && queued==previous+int(pass));
  assert(staged==queued && flushed==6*queued && kept==4*queued);
 };
 run(good,true);
 auto pair=[&](int i){return uint32_t(kernel.regs[i])|(uint32_t(kernel.regs[i+1])<<16);};
 assert(pair(0)==cos.address>>8 && pair(2)==sin.address>>8);
 assert(pair(4)==good[0] && pair(6)==good[1] && pair(8)==240);
 assert(pair(16)==good[2] && pair(18)==good[3]);
 assert(kernel.regs[10]==128 && kernel.regs[11]==2 && kernel.regs[12]==51);
 assert(kernel.regs[13]==128 && kernel.regs[14]==1 && kernel.regs[15]==1);
 assert(kernel.regs[64]==8 && kernel.regs[65]==1 && kernel.regs[66]==1);
 assert(kernel.counters[0]==0 && kernel.counters[1]==0 && kernel.counters[2]==0 && kernel.counters[3]==0);
 assert(pair(8)==240 && kernel.regs[14]==1 && kernel.regs[10]==128);
 assert(pair(16)==good[2] && pair(22)==good[6] && kernel.regs[26]==8);
 assert(pair(20)==key.address>>8 && pair(24)==value.address>>8);
 assert(kernel.regs[32]==128 && kernel.regs[33]==c10::Half(1.f/128).x && kernel.regs[34]==256 && kernel.regs[35]==14);
 assert(pair(36)==good[1] && pair(38)==good[0] && pair(40)==good[4] && kernel.regs[42]==c10::Half(1e-5).x);
 assert(kernel.regs[44]==128 && kernel.regs[45]==c10::Half(1.f/128).x && kernel.regs[46]==256 && kernel.regs[47]==8);
 assert(pair(48)==good[3] && pair(50)==good[2] && pair(52)==good[5] && kernel.regs[54]==c10::Half(1e-5).x);
 const std::array<uint32_t,7> bytes{112*128*2,112*128*2,64*128*2,64*128*2,256,256,64*128*2};
 for(int i=0;i<7;++i){
  for(auto b:{base-256,good[i]+1}){auto a=good;a[i]=b;run(a,false);}
  for(int j=0;j<7;++j)if(i!=j){auto a=good;a[i]=good[j];run(a,false);
   if(bytes[j]>256){a[i]=good[j]+256;run(a,false);}}
  auto a=good;a[i]=base+SpmAllocator::SPM_USABLE-bytes[i];run(a,true);
  a[i]+=256;run(a,false);
 }
 for(int n:{0,50,52,64})run(good,false,n);
 for(int n:{-1,0,112,127,129,191,193,207,209,223,225,239,241,256})run(good,false,51,n);
 for(int n:{0,1,4,9})run(good,false,51,240,n);
 for(int n:{0,102,111,113,128})run(good,false,51,240,8,n);
 for(int n:{0,51,63,65,112})run(good,false,51,240,8,112,n);
 for(double eps:{0.,1e-6,-1e-5})run(good,false,51,240,8,112,64,eps);
 const auto original_cos=cos,original_sin=sin;
 {const auto original_key=key,original_value=value;const auto original_plan=plan;
  for(int position:{128,144,160,176,192,208,224,240}){
   plan=exact_plan(position);
   cos.shape=sin.shape={position+51,64};
   const int blocks=(position+64+15)/16;
   key.shape[1]=value.shape[1]=blocks;run(good,true,51,position);
   assert(pair(8)==uint32_t(position)); // shared RoPE and writer position ABI2
   for(auto* t:{&key,&value}){--t->shape[1];run(good,false,51,position);++t->shape[1];}
   --key.shape[1];--value.shape[1];run(good,false,51,position);
   key.shape[1]=value.shape[1]=blocks;
   ++value.shape[1];run(good,false,51,position);--value.shape[1];
   for(auto* t:{&cos,&sin}){--t->shape[0];run(good,false,51,position);++t->shape[0];}
   --cos.shape[0];--sin.shape[0];run(good,false,51,position);
   cos.shape=sin.shape={position+51,64};++sin.shape[0];run(good,false,51,position);--sin.shape[0];
   for(int actual:{128,144,160,176,192,208,224,240})if(actual!=position){
    plan=exact_plan(actual);run(good,false,51,position);
   }
   for(auto wrong:{exact_plan(position,51,KvInsertRoute::V2,51),
                   exact_plan(position,64,KvInsertRoute::ALIGNED_V16,64)}){
    plan=wrong;run(good,false,51,position);
   }
  }
  key=original_key;value=original_value;plan=original_plan;
 }
 cos=original_cos;sin=original_sin;
 for(auto* t:{&cos,&sin})for(int failure=0;failure<11;++failure){
  cos=original_cos;sin=original_sin;
  switch(failure){case 0:t->present=false;break;case 1:t->contiguous=false;break;
   case 2:t->dtype=at::kChar;break;case 3:t->where.kind=c10::DeviceType::CPU;break;
   case 4:t->shape={290,64};break;case 5:t->shape={320,63};break;
   case 6:t->shape={1,320,64};break;case 7:t->address+=128;break;
   case 8:t->address=(uint64_t{1}<<40)-256;break;case 9:t->shape={319,64};break;case 10:t->address=0;break;}
  run(good,false);
 }
 cos=original_cos;sin=original_sin;cos.shape=sin.shape={291,64};run(good,true);
 sin.where.index=1;run(good,false);sin.where.index=0;
 {available=false;const int f=fetched,s=staged,q=queued,k=kept,flush=flushed;bool rejected=false;
  try{launch_hyvla_qk_rope_norm_kv_spm(good[0],good[1],good[2],good[3],good[4],good[5],good[6],cos,sin,key,value,51,240,112,64,1e-5,8,plan);}
  catch(const std::exception&){rejected=true;}
  assert(rejected && fetched==f+1 && staged==s && queued==q && kept==k && flushed==flush);available=true;}
 // Actual cache plan, full owner shape, all writable/read-only DDR overlaps.
 const auto saved_plan=plan;
 for(auto bad:{exact_plan(224),exact_plan(240,51,KvInsertRoute::V2,51),exact_plan(240,64,KvInsertRoute::ALIGNED_V16,64)}){
  plan=bad;run(good,false);
 }plan=saved_plan;
 const auto saved_key=key,saved_value=value;
 for(auto* t:{&key,&value})for(int failure=0;failure<15;++failure){
  key=saved_key;value=saved_value;
  switch(failure){case 0:t->present=false;break;case 1:t->contiguous=false;break;
   case 2:t->dtype=at::kByte;break;case 3:t->where.kind=c10::DeviceType::CPU;break;
   case 4:t->shape[0]=2;break;case 5:t->shape[1]=18;break;case 6:t->shape[2]=2;break;
   case 7:t->shape[3]=4;break;case 8:t->shape[4]=16;break;case 9:t->shape[5]=8;break;
   case 10:t->shape[6]=8;break;case 11:t->shape.pop_back();break;case 12:t->address+=128;break;
   case 13:t->address=0;break;case 14:t->address=(uint64_t{1}<<40)-256;break;}
  run(good,false);
 }
 key=saved_key;value=saved_value;key.shape[1]=value.shape[1]=19;run(good,true);
 value.shape[1]=20;run(good,false);value.shape[1]=19;
 value.where.index=1;run(good,false);value.where.index=0;
 for(auto* t:{&key,&value})for(const auto* other:{&cos,&sin,&key,&value})if(t!=other){
  const auto saved=*t;const auto at=other->address;
  t->address=at;run(good,false);t->address=at+256;run(good,false);*t=saved;
 }
 {const auto saved=value;value.address=key.address+key.nbytes();run(good,true);value=saved;}
 {const auto saved=key;key.address=(uint64_t{1}<<40)-key.nbytes();run(good,true);key.address+=256;run(good,false);key=saved;}
 spm.ready=false;run(good,false);spm.ready=true;
 spm.base=0xf8ff0000;auto high=good;for(auto& a:high)a+=spm.base-base;run(high,false);
}
'''
    compile_run(tmp_path, 'qk_rope_norm_kv_launcher', source.replace('LAUNCH', launch))
