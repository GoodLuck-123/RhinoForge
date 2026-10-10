"""Exercise the actual fused collective launcher at its memory/ABI boundaries."""
from test_hyvla_paired_kv import COMMON, ROOT, block, compile_run


def test_actual_launcher_ranges_aliases_typed_registers_and_peer_table(tmp_path):
    launch=block((ROOT/'src/fused/hyvla_ring_postnorm.h').read_text(), 'inline void launch_hyvla_ring_postnorm_spm(')
    source=COMMON+r'''
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;
 bool ready=true;uint32_t base=0x20000000;
 bool is_initialized()const{return ready;}
 uint32_t addr(int c,uint32_t o)const{return base+uint32_t(c)*0x1000000u+o;}
} spm;
#define SPM_ALLOC spm
struct Kernel {std::array<uint16_t,4160> regs{};uint32_t phase=99;
 void reset_regs(){regs.fill(0);phase=99;}
 void set_regs(uint32_t i,uint16_t v){regs.at(i)=v;}
 void set_regs(uint32_t i,uint32_t v){assert(i==128);phase=v;}
 void set_regs(uint32_t,int)=delete;
} kernel;
void rpu_set_legacy_scm_u16_checked(Kernel* k,uint32_t i,uint16_t v,const char*){k->set_regs(i,v);}
int fetched=0,queued=0;
struct RpuKernelGraph {static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 Kernel* get_kernel_reset(const std::string& s){assert(s=="hyvla_expert_ring_postnorm_s51");
  ++fetched;return &kernel;}};
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
 const std::array<uint32_t,4> good{base+0x10000,base+0x200000,base+0x400000,base};
 auto run=[&](std::array<uint32_t,4> a,bool pass,int rows=51,int cols=1024,int nr=64,double eps=1e-5,int cores=8){
  int before=fetched,previous=queued;bool ok=true;
  try{launch_hyvla_ring_postnorm_spm(a[0],a[1],a[2],a[3],rows,cols,nr,eps,cores);}
  catch(const std::exception&){ok=false;}
  assert(ok==pass && fetched==before+int(pass) && queued==previous+int(pass));
 };
 run(good,true);
 auto pair=[&](int i,int step=1){return uint32_t(kernel.regs[i])|(uint32_t(kernel.regs[i+step])<<16);};
 assert(pair(4)==good[1]-base && pair(12)==good[2] && pair(14)==good[1] && pair(16)==good[3]);
 assert(kernel.regs[0]==8 && kernel.regs[1]==1 && pair(2)==63488 && kernel.regs[6]==0);
 assert(kernel.regs[8]==1024 && kernel.regs[9]==c10::Half(1.f/1024).x && kernel.regs[10]==2048);
 assert(kernel.regs[11]==8 && kernel.regs[18]==c10::Half(1e-5).x && kernel.phase==0);
 for(int c=0;c<8;++c){
  assert(pair(4096+c,8)==spm.addr(c,good[0]-base)-base);
  assert(pair(4112+c,8)==spm.addr(c,good[2]-base)-base);
  assert(pair(4128+c,8)==uint32_t(c*6528*2) && pair(4144+c,8)==6528);
 }
 const std::array<uint32_t,4> bytes{51*1024*2,64*1024*2,64*1024*2,2048};
 for(int i=0;i<4;++i){
  for(auto b:{base-256,good[i]+1,base+SpmAllocator::SPM_USABLE-256}){auto a=good;a[i]=b;run(a,false);}
  for(int j=0;j<4;++j)if(i!=j){auto a=good;a[i]=good[j];run(a,false);
   a[i]=good[j]+256;run(a,false);}
  auto a=good;a[i]=base+SpmAllocator::SPM_USABLE-bytes[i];run(a,true);
  a[i]+=256;run(a,false);
 }
 for(int r:{0,50,52,64})run(good,false,r);
 for(int c:{0,128,1023,1025,2048})run(good,false,51,c);
 for(int n:{0,51,63,65})run(good,false,51,1024,n);
 for(double e:{0.,1e-6,-1e-5})run(good,false,51,1024,64,e);
 for(int n:{0,1,4,9})run(good,false,51,1024,64,1e-5,n);
 spm.ready=false;run(good,false);spm.ready=true;
 spm.base=0xf8ff0000;auto high=good;for(auto& a:high)a+=spm.base-base;run(high,false);
}
'''
    compile_run(tmp_path,'ring_postnorm_launcher',source.replace('LAUNCH',launch))
