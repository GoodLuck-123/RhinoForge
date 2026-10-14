"""Execute the bounded production leaf and compare its packet to the measured micro."""
from pathlib import Path

from test_rmsnorm_capability_publication_host import _block, _compile_and_run


ROOT = Path(__file__).resolve().parents[1]


def test_actual_ragged_sdpa_packet_ranges_and_fail_closed_boundaries(tmp_path):
    ops = (ROOT / 'src/ops/rpu_sdpa_vctxlen.cpp').read_text()
    helpers = (ROOT / 'src/core/rpu_helpers.h').read_text()
    leaf = _block(ops, 'void rpu_launch_wall_base_vision_ragged_sdpa_spm(')
    common = '\n'.join(_block(ops, name) + (';' if name == 'struct SpmRegion {' else '') for name in (
        'struct SpmRegion {', 'size_t checked_spm_bytes(', 'SpmRegion checked_spm_region(',
        'template <size_t N>\nvoid check_spm_regions_disjoint(', 'void set_scm_spm_ptr(',
        'std::vector<uint8_t> spm_cores('))
    scm = _block(helpers, 'inline void rpu_set_scm_u32_checked(')
    bits = _block(helpers, 'inline uint32_t float32_to_uint32(')
    program = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
namespace c10 { using Half=uint16_t; }
enum class KernelId {WALL_BASE_VISION_RAGGED_SDPA};
struct Kernel {
 std::array<uint16_t, 4352> regs{};std::vector<uint32_t> scm_writes;int reject=-1,resets=0;
 void reset_regs(){regs.fill(0xa55a);scm_writes.clear();++resets;}
 uint32_t set_regs(uint32_t reg,uint16_t value){
  assert(reg<68);if(int(reg)==reject)return 5;regs.at(reg)=value;return 0;}
 uint32_t set_regs(uint32_t reg,uint32_t value){
  assert(reg>=4096&&reg%2==0);if(int(reg)==reject)return 5;
  regs.at(reg)=uint16_t(value);regs.at(reg+1)=uint16_t(value>>16);scm_writes.push_back(reg);return 0;}
 uint32_t set_regs(uint32_t,int32_t)=delete;
} kernel;
namespace rhino_lkn {using Kernel_t=Kernel;constexpr int kKernelRegOk=0;}
using rhino_lkn::Kernel_t;using rhino_lkn::kKernelRegOk;
constexpr int SCM_REG_OFFSET=4096;
SCM_HELPER
FLOAT_BITS
struct SpmAllocator {
 static constexpr uint32_t SPM_USABLE=8387584;bool ready=true;
 std::array<uint32_t,8> bases{0x20000000,0x21010000,0x22020000,0x23030000,
                             0x24040000,0x25050000,0x26060000,0x27070000};
 bool is_initialized()const{return ready;}
 uint32_t addr(int core,uint32_t offset)const{return bases.at(core)+offset;}
} spm;
#define SPM_ALLOC spm
struct KernelCache {bool loaded=true;static auto& instance(){static KernelCache c;return c;}
 bool has_loaded(KernelId id)const{assert(id==KernelId::WALL_BASE_VISION_RAGGED_SDPA);return loaded;}};
int lookups=0,queued=0;bool kernel_available=true;
struct RpuKernelGraph {static auto& active(){static RpuKernelGraph g;return g;}
 Kernel* get_kernel_reset(KernelId id){assert(id==KernelId::WALL_BASE_VISION_RAGGED_SDPA);++lookups;
  if(!kernel_available)return nullptr;kernel.reset_regs();return &kernel;}
 Kernel* get_kernel_reset(const std::string& name){assert(name=="wall_base_vision_ragged_sdpa_b1_s324_s336_h16d80");
  return get_kernel_reset(KernelId::WALL_BASE_VISION_RAGGED_SDPA);}
};
#define GET_KERNEL(id) RpuKernelGraph::active().get_kernel_reset(id)
struct Queue {bool broadcast=false;
 void set_broadcast_mode(bool value){broadcast=value;}
 void enqueu_kernel(Kernel&,const std::vector<uint16_t>& grid,const std::vector<uint8_t>& cores){
  assert(broadcast&&grid==std::vector<uint16_t>({2,1,9})&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));++queued;}
} queue;
Queue* get_queue(int cores){assert(cores==8);return &queue;}
#define GET_QUEUE(cores) get_queue(cores)
SPM_HELPERS
PRODUCTION_LEAF
uint32_t word(int r){return kernel.regs.at(r)|(uint32_t(kernel.regs.at(r+1))<<16);}
template<class F>void rejects(F run,bool before_lookup=true){
 int old=queued,look=lookups;bool failed=false;try{run();}catch(const std::exception&){failed=true;}
 assert(failed&&queued==old);if(before_lookup)assert(lookups==look);
}
int main(){
 for(int64_t rows:{324,336}){
  std::array<uint32_t,5> p{0x11000,0x101000,0x201000,0x301000,0x404000};
  for(auto& address:p)address+=spm.addr(0,0);
  const auto valid=p;double scale=1/std::sqrt(80.0);int cores=8;int64_t m=rows;
  auto run=[&]{rpu_launch_wall_base_vision_ragged_sdpa_spm(p[0],p[1],p[2],p[3],p[4],scale,m,cores);};
  // Frozen R48 micro ABI fixture: p0/p2 vary with M; FP32 scale is 1/sqrt(80).
  // Keep numeric expected data here so this test needs no untracked probe files.
  const std::array<uint16_t,68> expected{
   uint16_t(rows),0,uint16_t(rows),21,21,0,16,16,2,2,80,80,
   80,5,5,5,0,0,8,1,0,0,0,0,
   0,0,0,0,0,0,0,0,0,0,0,0,
   0,0,0,0,63790,15844,1,0,0,0,0,0,
   0,0,0,0,0,0,0,0,4,0,64,80,
   256,4,5,16,2,1,9,0};
  for(int repeat=0;repeat<2;++repeat){kernel.regs.fill(0xdead);queue.broadcast=false;run();
   assert(std::equal(expected.begin(),expected.end(),kernel.regs.begin())&&kernel.scm_writes.size()==48);
   assert(std::all_of(kernel.regs.begin()+68,kernel.regs.begin()+4096,[](auto v){return v==0xa55a;}));
   assert(std::all_of(kernel.regs.begin()+4192,kernel.regs.end(),[](auto v){return v==0xa55a;}));
   assert(kernel.regs[0]==rows&&kernel.regs[2]==rows&&kernel.regs[56]==4);
   for(int c=0;c<8;++c){const std::array<int,5> fields{0,8,10,6,4};
    for(int i=0;i<5;++i)assert(word(4096+c*12+fields[i])==(spm.addr(c,p[i]-spm.addr(0,0))>>5));
    assert(word(4096+c*12+2)==0);}
  }
  for(double value:{0.0,-0.25,0.125,1.0}){scale=value;run();assert(word(40)==float32_to_uint32(float(value)));}
  scale=1/std::sqrt(80.0);
  for(auto bad:{0,32,320,323,325,335,337,640,672}){m=bad;rejects(run);}m=rows;
  for(auto bad:{0,1,4,7,9}){cores=bad;rejects(run);}cores=8;
  for(double bad:{std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::max()}){scale=bad;rejects(run);}
  scale=1/std::sqrt(80.0);
  spm.ready=false;rejects(run);spm.ready=true;
  for(int i=0;i<5;++i){
   p=valid;p[i]=0;rejects(run);p=valid;p[i]=spm.addr(0,0)-32;rejects(run);
   p=valid;p[i]+=2;rejects(run);p=valid;p[i]=spm.addr(1,0);rejects(run);
  }p=valid;
  // Every pair is disjoint, including immutable inputs and mask, not only Q/out.
  for(int i=0;i<5;++i)for(int j=i+1;j<5;++j){
   p=valid;p[j]=p[i];rejects(run);p=valid;p[j]=p[i]+32;rejects(run);
  }p=valid;
  const std::array<uint32_t,5> sizes{uint32_t(rows*320),336*320,336*320,uint32_t(rows*320),uint32_t(rows*336*2)};
  // Exact usable-SPM endpoints are legal; the first aligned step beyond is not.
  for(int i=0;i<5;++i){p=valid;p[i]=spm.addr(0,SpmAllocator::SPM_USABLE-sizes[i]);run();p[i]+=32;rejects(run);}
  p=valid;
  if(rows==324){
   for(int i:{1,2}){p=valid;p[i]=spm.addr(0,SpmAllocator::SPM_USABLE-324*320);rejects(run);}
   p=valid;p[4]=spm.addr(0,SpmAllocator::SPM_USABLE-324*324*2);rejects(run);p=valid;
  }
  // Adjacent (non-overlapping) ranges remain legal.
  uint32_t offset=256;for(int i=0;i<5;++i){p[i]=spm.addr(0,offset);offset+=sizes[i];}run();p=valid;
  // Each peer address is checked in uint64 before the allocator's u32 addition.
  for(int c=0;c<8;++c){auto old=spm.bases[c];spm.bases[c]+=2;rejects(run);spm.bases[c]=old;}
  for(int c=1;c<8;++c){auto old=spm.bases[c];spm.bases[c]=0xffff0000;rejects(run);spm.bases[c]=old;}
  const uint32_t max_end=valid[4]-spm.addr(0,0)+sizes[4];auto old=spm.bases[7];
  spm.bases[7]=uint32_t((uint64_t{1}<<32)-max_end);run();spm.bases[7]+=32;rejects(run);spm.bases[7]=old;
  KernelCache::instance().loaded=false;rejects(run);KernelCache::instance().loaded=true;
  kernel_available=false;rejects(run,false);kernel_available=true;
  for(int reg:{0,56,67,4096,4098,4100,4180,4190}){kernel.reject=reg;rejects(run,false);kernel.reject=-1;}
  run();
 }
}
'''
    for key, value in (('SCM_HELPER', scm), ('FLOAT_BITS', bits), ('SPM_HELPERS', common),
                       ('PRODUCTION_LEAF', leaf)):
        program = program.replace(key, value)
    _compile_and_run(tmp_path, 'wall_base_ragged_sdpa_leaf', program)
