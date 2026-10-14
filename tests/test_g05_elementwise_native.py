"""Execute actual optional G05 wrappers and validate SPM/alias/register boundaries."""
from pathlib import Path
from test_siglip_cold_profile import _body
from test_g05_q_gate_native import _compile_run

ROOT=Path(__file__).resolve().parents[1]


def test_g05_elementwise_actual_wrappers_boundaries_and_original_aliases(tmp_path):
    functions=[]
    for path,name in [('rpu_eltwise_binary.cpp','rpu_launch_g05_sub_gated_residual_spm_kernel'),
                      ('rpu_eltwise_binary.cpp','rpu_launch_g05_rtc_sub_gated_residual_spm_kernel'),
                      ('rpu_rmsnorm.cpp','rpu_launch_g05_rtc_adarms_norm_shift_spm_kernel')]:
        source=(ROOT/'src/ops'/path).read_text()
        marker='void '+name+'('
        start=source.index(marker)
        functions.append(source[start:source.index('{',start)]+_body(source,marker))
    cpp=r'''
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>
#define TORCH_CHECK(x,...) do{if(!(x))throw std::runtime_error("rejected");}while(0)
// The launcher test checks known binary16 register values; it does not depend on
// the host compiler providing a native half type.
namespace c10{struct Half{uint16_t x;explicit Half(double f){
 if(f==1.0/1024)x=0x1400;else if(f==1e-6)x=0x0011;else if(f==1.0)x=0x3c00;else throw std::runtime_error("unexpected half input");
}};}
uint16_t half_to_u16(c10::Half h){return h.x;}
int64_t CeilDiv(int64_t n,int64_t d){return(n+d-1)/d;}
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8387584;bool initialized=true;bool is_initialized()const{return initialized;}uint32_t addr(int core,int off)const{assert(core==0);return 0x20000000u+off;}}allocator;
#define SPM_ALLOC allocator
#define IDS G05_SUB_GATED_RESIDUAL_H1024,G05_RTC_ADARMS_NORM_SHIFT_M32H1024,G05_RTC_SUB_GATED_RESIDUAL_M32H1024
enum class KernelId{IDS};constexpr auto SUB=KernelId::G05_SUB_GATED_RESIDUAL_H1024,RTC=KernelId::G05_RTC_ADARMS_NORM_SHIFT_M32H1024;
struct KernelCache{std::set<KernelId>missing;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId id)const{return !missing.count(id);}};
struct Kernel{KernelId id;std::map<int,uint16_t>regs;void reset_regs(){regs.clear();}void set_regs(int r,uint16_t v){regs[r]=v;}void set_regs(int,uint32_t)=delete;}kernel;
int lookups=0;
Kernel*get(KernelId id){++lookups;assert(KernelCache::instance().has_loaded(id));kernel.id=id;return &kernel;}
#define GET_KERNEL(id) get(id)
struct Submission{KernelId id;std::map<int,uint16_t>regs;std::vector<uint16_t>grid;std::vector<uint8_t>cores;};
struct Queue{bool broadcast=false;std::vector<Submission>submissions;void set_broadcast_mode(bool b){broadcast=b;}void enqueu_kernel(Kernel&k,std::vector<uint16_t>grid,std::vector<uint8_t>cores){assert(broadcast);submissions.push_back({k.id,k.regs,grid,cores});}}queue;
Queue*get_queue(int c){assert(c==1||c==8);return &queue;}
#define GET_QUEUE(c) get_queue(c)
__FUNCTIONS__
uint32_t pair(const Submission&s,int r){return uint32_t(s.regs.at(r))|(uint32_t(s.regs.at(r+1))<<16);}
template<class F>void reject(F f){auto n=queue.submissions.size();auto k=lookups;try{f();assert(false);}catch(const std::runtime_error&){}assert(n==queue.submissions.size()&&k==lookups);}
int main(){
 const uint32_t base=allocator.addr(0,0),x=base+0x10000,r=base+0x40000,g=base+0x70000,y=base+0x80000;
 auto gate=[&](uint32_t input,uint32_t residual,uint32_t output,int64_t rows,int cores=8){rpu_launch_g05_sub_gated_residual_spm_kernel(input,residual,g,output,rows,cores);};
 for(auto rows:{1,5,27,32,64})for(auto cores:{1,8})for(auto output:{x,r,y}){
  gate(x,r,output,rows,cores);const auto&s=queue.submissions.back();const auto blocks=(rows+15)/16,normal=(rows+blocks-1)/blocks;
  assert(s.id==SUB&&s.grid==std::vector<uint16_t>({uint16_t(blocks),1,1})&&s.cores.size()==size_t(cores));
  for(int c=0;c<cores;++c)assert(s.cores[c]==c);
  assert(s.regs.at(0)==normal&&s.regs.at(2)==rows-normal*(blocks-1));
  assert(pair(s,4)==g&&pair(s,6)==x&&pair(s,8)==output&&pair(s,14)==r);
  assert(pair(s,10)==uint32_t(normal*2)&&s.regs.at(12)==4&&s.regs.at(13)==64);
  for(int reg:{56,57,58})assert(s.regs.at(reg)==0x3c00);
  assert(s.regs.at(59)==0&&s.regs.at(60)==0&&s.regs.at(61)==1&&s.regs.at(62)==0x492&&s.regs.at(63)==0);
 }
 gate(x,x,x,32); // Whole alias is safe: all row reads precede retirement.
 for(auto rows:{-1,0,65})reject([&]{gate(x,r,x,rows);});
 for(auto cores:{0,2,7,9})reject([&]{gate(x,r,x,32,cores);});
 reject([&]{gate(x,r,x+256,32);});reject([&]{gate(x,r,r+256,32);});reject([&]{gate(x,r,g,32);});
 for(auto bad:{base-256,x+1,base+SpmAllocator::SPM_USABLE-256,0xffffff00u}){
  reject([&]{gate(bad,r,y,32);});reject([&]{gate(x,bad,y,32);});reject([&]{gate(x,r,bad,32);});
  reject([&]{rpu_launch_g05_sub_gated_residual_spm_kernel(x,r,bad,y,32,8);});
 }
 gate(base+SpmAllocator::SPM_USABLE-65536,r,y,32);
 const uint32_t cs=base+0xa0000,ch=cs+2048,ns=ch+2048,nh=ns+2048;
 auto norm=[&](uint32_t input,uint32_t output,int clean,double eps=1e-6,int cores=8){rpu_launch_g05_rtc_adarms_norm_shift_spm_kernel(input,output,cs,ch,ns,nh,clean,eps,cores);};
 for(int clean=0;clean<=32;++clean)for(auto output:{x,y}){
  norm(x,output,clean);const auto&s=queue.submissions.back();assert(s.id==RTC&&s.grid==std::vector<uint16_t>({32,1,1})&&s.cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));
  assert(s.regs.at(0)==1024&&s.regs.at(1)==0x1400&&pair(s,2)==2048&&s.regs.at(10)==half_to_u16(c10::Half(1e-6))&&s.regs.at(18)==clean);
  assert(pair(s,4)==x&&pair(s,6)==output&&pair(s,8)==cs&&pair(s,12)==ch&&pair(s,14)==ns&&pair(s,16)==nh);
 }
 for(int clean:{-1,33})reject([&]{norm(x,y,clean);});
 for(int cores:{0,1,2,7,9})reject([&]{norm(x,y,5,1e-6,cores);});
 for(double eps:{0.0,-1.0,1.001,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})reject([&]{norm(x,y,5,eps);});
 reject([&]{norm(x,x+256,5);});
 for(auto output:{cs,ch,ns,nh})reject([&]{norm(x,output,5);});
 for(int which=0;which<6;++which)for(auto bad:{base-256,x+1,base+SpmAllocator::SPM_USABLE-256,0xffffff00u}){
  std::vector<uint32_t>a{x,y,cs,ch,ns,nh};a[which]=bad;
  reject([&]{rpu_launch_g05_rtc_adarms_norm_shift_spm_kernel(a[0],a[1],a[2],a[3],a[4],a[5],5,1e-6,8);});
 }
 norm(base+SpmAllocator::SPM_USABLE-65536,y,5);
 for(auto id:{SUB,RTC}){KernelCache::instance().missing.insert(id);reject([&]{if(id==SUB)gate(x,r,x,32);else norm(x,x,5);});KernelCache::instance().missing.clear();}
 const auto RTCG=KernelId::G05_RTC_SUB_GATED_RESIDUAL_M32H1024;
 auto rtcgate=[&](uint32_t input,uint32_t residual,uint32_t clean,uint32_t noisy,uint32_t output,int split,int cores=8){rpu_launch_g05_rtc_sub_gated_residual_spm_kernel(input,residual,clean,noisy,output,split,cores);};
 for(int split=0;split<=32;++split)for(auto output:{x,r,y}){
  rtcgate(x,r,cs,ns,output,split);const auto&s=queue.submissions.back();
  assert(s.id==RTCG&&s.grid==std::vector<uint16_t>({4,1,1})&&s.cores.size()==8);
  assert(pair(s,4)==cs&&pair(s,18)==ns&&pair(s,6)==x&&pair(s,8)==output&&pair(s,14)==r&&s.regs.at(16)==split);
  assert(s.regs.at(0)==(split<=16?split:(split+1)/2)&&s.regs.at(2)==(split<=16?0:split/2));
  int noisy=32-split;assert(s.regs.at(20)==(noisy<=16?noisy:(noisy+1)/2)&&s.regs.at(22)==(noisy<=16?0:noisy/2));
  assert(s.regs.at(12)==4&&s.regs.at(13)==64&&s.regs.at(62)==0x492&&s.regs.at(63)==0);
 }
 for(int split:{-1,33})reject([&]{rtcgate(x,r,cs,ns,y,split);});
 for(int cores:{0,1,4,7,9})reject([&]{rtcgate(x,r,cs,ns,y,5,cores);});
 for(int which=0;which<5;++which)for(auto bad:{base-256,x+1,base+SpmAllocator::SPM_USABLE-256,0xffffff00u}){
  std::vector<uint32_t>a{x,r,cs,ns,y};a[which]=bad;reject([&]{rtcgate(a[0],a[1],a[2],a[3],a[4],5);});
 }
 for(auto output:{x+256,r+256,cs,ns})reject([&]{rtcgate(x,r,cs,ns,output,5);});
 // Both gates must be valid even at split endpoints, and unavailable payload fails before GET.
 reject([&]{rtcgate(x,r,cs,base-256,y,32);});reject([&]{rtcgate(x,r,base-256,ns,y,0);});
 KernelCache::instance().missing.insert(RTCG);reject([&]{rtcgate(x,r,cs,ns,y,5);});KernelCache::instance().missing.clear();
 allocator.initialized=false;reject([&]{rtcgate(x,r,cs,ns,y,5);});reject([&]{gate(x,r,x,32);});reject([&]{norm(x,x,5);});
}
'''
    _compile_run(tmp_path,'g05_elementwise',cpp.replace('__FUNCTIONS__','\n'.join(functions)))
