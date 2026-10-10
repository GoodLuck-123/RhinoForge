"""Exercise the actual paired-norm launcher with owned SPM boundaries."""
from test_hyvla_paired_kv import COMMON, ROOT, block, compile_run
import pytest


@pytest.mark.parametrize("vector", (16, 32))
def test_actual_launcher_ranges_aliases_and_typed_registers(tmp_path, vector):
    launch=block((ROOT/'src/fused/hyvla_paired_rmsnorm.h').read_text(), 'inline void launch_hyvla_paired_rmsnorm_spm(')
    source=COMMON+r'''
enum class RpuRmsNormSpmRoute {BASE,V16_GRID8,V32_GRID8};
constexpr auto selected_route=RpuRmsNormSpmRoute::VWIDTH_GRID8;
constexpr int default_rows=DEFAULT_ROWS;
struct SpmAllocator {static constexpr uint32_t SPM_USABLE=8387584;
 bool ready=true;uint32_t base=0x20000000;
 bool is_initialized()const{return ready;}
 uint32_t addr(int c,uint32_t o)const{return base+uint32_t(c)*0x1000000u+o;}
} spm;
#define SPM_ALLOC spm
struct Kernel {std::array<uint16_t,68> regs{};void reset_regs(){regs.fill(0);}
 void set_regs(uint32_t i,uint16_t v){regs.at(i)=v;}
 void set_regs(uint32_t,int)=delete;
} kernel;
int fetched=0,queued=0;bool available=true;
struct RpuKernelGraph {static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
 Kernel* get_kernel_reset(const std::string& s){assert(s=="EXPECTED_SYMBOL");
  ++fetched;return available?&kernel:nullptr;}};
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
 const std::array<uint32_t,5> good{base+0x10000,base+0x200000,base+0x400000,base,base+4096};
 auto run=[&](std::array<uint32_t,5> a,bool pass,int rows=default_rows,int cols=2048,double eps=1e-5,int cores=8){
  int before=fetched,previous=queued;bool ok=true;
  try{launch_hyvla_paired_rmsnorm_spm(a[0],a[1],a[2],a[3],a[4],rows,cols,eps,cores,selected_route);}
  catch(const std::exception&){ok=false;}
  assert(ok==pass && fetched==before+int(pass) && queued==previous+int(pass));
 };
 run(good,true);
 auto pair=[&](int i){return uint32_t(kernel.regs[i])|(uint32_t(kernel.regs[i+1])<<16);};
 assert(pair(4)==good[0] && pair(6)==good[1] && pair(12)==good[2] && pair(8)==good[3] && pair(14)==good[4]);
 assert(kernel.regs[0]==2048 && kernel.regs[1]==c10::Half(1.f/2048).x && kernel.regs[2]==4096);
 assert(kernel.regs[3]==default_rows/8 && kernel.regs[10]==c10::Half(1e-5).x);
 for(int i=0;i<5;++i){
  for(auto b:{base-256,good[i]+1,base+SpmAllocator::SPM_USABLE-256}){auto a=good;a[i]=b;run(a,false);}
  for(int j=0;j<5;++j)if(i!=j){auto a=good;a[i]=good[j];run(a,false);
   a[i]=good[j]+256;run(a,false);}
  auto a=good;a[i]=base+SpmAllocator::SPM_USABLE-(i<3?default_rows*2048*2:4096);run(a,true);
  a[i]+=256;run(a,false);
 }
 for(int rows:{128,144,160,176,192,208,224,240}){
  if(selected_route==RpuRmsNormSpmRoute::V32_GRID8 && rows%32){run(good,false,rows);continue;}
  run(good,true,rows);assert(kernel.regs[3]==rows/8);
  auto a=good;a[2]=base+SpmAllocator::SPM_USABLE-rows*2048*2;run(a,true,rows);
  a[2]+=256;run(a,false,rows);
 }
 for(int r:{0,112,127,129,200,239,241,256})run(good,false,r);
 for(int c:{0,128,1024,2047,2049,4096})run(good,false,default_rows,c);
 for(double e:{0.,1e-6,-1e-5})run(good,false,default_rows,2048,e);
 for(int n:{0,1,4,9})run(good,false,default_rows,2048,1e-5,n);
 {const int before=queued;bool rejected=false;
  try{launch_hyvla_paired_rmsnorm_spm(good[0],good[1],good[2],good[3],good[4],default_rows,2048,1e-5,8,RpuRmsNormSpmRoute::BASE);}
  catch(const std::exception&){rejected=true;}assert(rejected && before==queued);}
 spm.ready=false;run(good,false);spm.ready=true;
 spm.base=0xf8ff0000;auto high=good;for(auto& a:high)a+=spm.base-base;run(high,false);
}
'''
    source = source.replace("WIDTH", str(vector)).replace("DEFAULT_ROWS", "240" if vector == 16 else "160")
    source = source.replace("EXPECTED_SYMBOL", "hyvla_mot_paired_rmsnorm_s240" if vector == 16 else "hyvla_mot_paired_rmsnorm_v32")
    compile_run(tmp_path,'paired_norm_launcher',source.replace('LAUNCH',launch))
