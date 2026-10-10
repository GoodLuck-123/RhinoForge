"""Compile the actual fixed G05 wrappers against a strict Graph/queue stub."""
from pathlib import Path
import importlib.util
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_actual_g05_vision_owner_wrappers_validate_spm_and_register_abi(tmp_path):
    source = (ROOT / 'src/ops/rpu_pi05_owner_norm.cpp').read_text()
    body = source[source.index('\nnamespace {\nconstexpr uint32_t G05VisionFullBytes'):]
    cpp = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include <c10/util/Half.h>
#include "rpu_pi05_owner_norm.h"
#define TORCH_CHECK(x,...) do {if(!(x))throw std::runtime_error("rejected");}while(0)
struct SpmAllocator {
    static constexpr uint32_t SPM_USABLE=8303616;
    bool initialized=true;
    bool is_initialized()const{return initialized;}
    uint32_t addr(uint32_t core,uint32_t offset)const{
        assert(core<8&&offset<SPM_USABLE);return 0x20000000+core*0x1000000+offset;
    }
} allocator;
#define SPM_ALLOC allocator
enum class KernelId {G05_VISION_OWNER_REDUCE_LAYERNORM_M256N1024,
    G05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M256N1024};
constexpr auto OWNER=KernelId::G05_VISION_OWNER_REDUCE_LAYERNORM_M256N1024;
constexpr auto COMPACT=KernelId::G05_VISION_XOR3_COMPACT_RESIDUAL_RAW_FULL_M256N1024;
struct KernelCache {
    bool owner=true,compact=true;
    static KernelCache& instance(){static KernelCache c;return c;}
    bool has_loaded(KernelId id)const{return id==OWNER?owner:compact;}
};
enum class GraphKernelNoDdrProof {RingAllReduceSpmResidual};
constexpr int kKernelRegOk=0;
struct Kernel {
    KernelId id;std::map<uint32_t,uint16_t> regs,scm;
    void reset_regs(){regs.clear();scm.clear();}
    int set_regs(uint32_t i,uint16_t v){assert(i<4096);regs[i]=v;return kKernelRegOk;}
    int set_regs(uint32_t,uint32_t)=delete;
};
struct Submission {KernelId id;std::map<uint32_t,uint16_t> regs,scm;};
struct RpuKernelGraph {
    bool census=true;std::optional<KernelId> pending,staged;
    int stages=0,gets=0;std::vector<Submission> submissions;
    Kernel owner{OWNER,{},{}},compact{COMPACT,{},{}};
    static RpuKernelGraph& active(){static RpuKernelGraph g;return g;}
    bool kernel_register_census_active()const{return census;}
    void stage_kernel_no_ddr(KernelId id,GraphKernelNoDdrProof proof){
        assert(census&&!pending&&!staged&&proof==GraphKernelNoDdrProof::RingAllReduceSpmResidual);
        staged=id;++stages;
    }
    Kernel* get(KernelId id){
        assert(!pending&&KernelCache::instance().has_loaded(id));
        assert(census?staged==id:!staged);pending=id;++gets;return id==OWNER?&owner:&compact;
    }
};
#define GET_KERNEL(id) RpuKernelGraph::active().get(id)
void rpu_set_legacy_scm_u16_checked(Kernel* k,uint32_t reg,uint16_t value,const char*){
    assert(reg>=4096&&reg<4096+80);k->scm[reg]=value;
}
struct Queue {
    bool broadcast=false;
    void set_broadcast_mode(bool value){broadcast=value;}
    void enqueu_kernel(Kernel& kernel,std::vector<uint16_t> grid,std::vector<uint8_t> cores){
        auto& g=RpuKernelGraph::active();assert(broadcast&&g.pending==kernel.id);
        assert(grid==std::vector<uint16_t>({8,1,1}));
        assert(cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));
        assert(g.census?g.staged==kernel.id:!g.staged);
        g.submissions.push_back({kernel.id,kernel.regs,kernel.scm});
        g.pending.reset();g.staged.reset();broadcast=false;
    }
} queue;
Queue* get_queue(int cores){assert(cores==8);return &queue;}
#define GET_QUEUE(cores) get_queue(cores)
''' + body + r'''
uint32_t pair(const std::map<uint32_t,uint16_t>& regs,int index,int high_offset=1){
    return uint32_t(regs.at(index))|(uint32_t(regs.at(index+high_offset))<<16);
}
void owner(const std::array<uint64_t,6>& a){
    rpu_launch_g05_vision_owner_reduce_layernorm_m256n1024_kernel(a[0],a[1],a[2],a[3],a[4],a[5]);
}
template<class F>void rejected(F run){
    auto& g=RpuKernelGraph::active();const int gets=g.gets,stages=g.stages;
    const auto submits=g.submissions.size();
    try{run();assert(false);}catch(const std::runtime_error&){}
    assert(g.gets==gets&&g.stages==stages&&g.submissions.size()==submits&&!g.pending&&!g.staged);
}
void check_collective(const Submission& s,uint32_t partial,uint32_t residual,uint32_t raw,uint32_t normalized){
    const auto& r=s.regs;const uint32_t base=allocator.addr(0,0);
    assert(r.at(0)==8&&r.at(1)==1&&pair(r,2)==63488&&pair(r,4)==residual-base&&r.at(6)==0);
    assert(r.at(42)==0&&r.at(64)==8&&r.at(65)==1&&r.at(66)==1);
    for(int i=128;i<=255;++i)assert(r.at(i)==0);
    assert(s.scm.size()==80);
    for(int c=0;c<8;++c){
        assert(pair(s.scm,4096+c,8)==allocator.addr(c,partial-base)-base);
        assert(pair(s.scm,4096+16+c,8)==allocator.addr(c,raw-base)-base);
        assert(pair(s.scm,4096+32+c,8)==uint32_t(c*65536));
        assert(pair(s.scm,4096+48+c,8)==32768);
        assert(pair(s.scm,4096+64+c,8)==(normalized?allocator.addr(c,normalized-base)-base:0));
    }
}
int main(){
    const uint32_t base=allocator.addr(0,0);
    const std::array<uint64_t,6> valid{base+0x10000,base+0x100000,base+0x200000,
                                    base+0x300000,base+0x400000,base+0x401000};
    auto& g=RpuKernelGraph::active();
    for(int pass=0;pass<4;++pass){
        g.census=pass<2;auto a=valid;
        for(auto& address:a)address+=pass*256;
        if(pass%2)a[3]=a[0];
        owner(a);
        const auto& s=g.submissions.back();assert(s.id==OWNER);
        check_collective(s,a[0],a[1],a[2],a[3]);
        assert(pair(s.regs,8)==a[2]&&pair(s.regs,10)==a[3]);
        assert(pair(s.regs,12)==a[4]&&pair(s.regs,14)==a[5]);
        assert(s.regs.at(16)==32&&s.regs.at(17)==1024&&pair(s.regs,18)==0x3a800000);
        assert(s.regs.at(20)==0x1432&&s.regs.at(21)==0&&pair(s.regs,22)==0);
        assert(pair(s.regs,24)==2048&&s.regs.at(26)==2&&s.regs.at(27)==32);
        assert(s.regs.at(28)==0x5000&&s.regs.at(29)==1);
        const uint64_t partial=base+0x500000+pass*256;
        rpu_launch_g05_vision_compact_residual_m256n1024_kernel(partial,a[2],a[1]);
        const auto& c=g.submissions.back();assert(c.id==COMPACT);
        check_collective(c,partial,a[2],a[1],0);
        for(int i=8;i<=29;++i)assert(c.regs.count(i)==0);
    }
    g.census=true;
    for(int index=0;index<6;++index){
        for(uint64_t invalid:std::array<uint64_t,5>{uint64_t(base-256),uint64_t(base+SpmAllocator::SPM_USABLE),
                             valid[index]+1,(1ULL<<32)+valid[index],UINT64_MAX-255}){
            auto a=valid;a[index]=invalid;rejected([&]{owner(a);});
        }
    }
    for(int first=0;first<6;++first)for(int second=first+1;second<6;++second){
        if(first==0&&second==3)continue;
        auto a=valid;a[second]=a[first];rejected([&]{owner(a);});
    }
    auto overlap=valid;overlap[3]=overlap[0]+256;rejected([&]{owner(overlap);});
    auto at_end=valid;at_end[5]=base+SpmAllocator::SPM_USABLE-2048;owner(at_end);
    at_end[5]+=256;rejected([&]{owner(at_end);});
    allocator.initialized=false;rejected([&]{owner(valid);});allocator.initialized=true;
    auto& cache=KernelCache::instance();cache.owner=false;rejected([&]{owner(valid);});cache.owner=true;
    cache.compact=false;rejected([&]{rpu_launch_g05_vision_compact_residual_m256n1024_kernel(valid[0],valid[2],valid[1]);});cache.compact=true;
    for(const auto& a:std::vector<std::array<uint64_t,3>>{
        {valid[0],valid[0],valid[1]},{valid[0],valid[2],valid[0]},
        {valid[0],valid[2],valid[2]},{valid[0]+1,valid[2],valid[1]},
        {valid[0],valid[2]+1,valid[1]},{valid[0],valid[2],uint64_t(base)+SpmAllocator::SPM_USABLE-256}})
        rejected([&]{rpu_launch_g05_vision_compact_residual_m256n1024_kernel(a[0],a[1],a[2]);});
    assert(!g.pending&&!g.staged);
}
'''
    path = tmp_path / 'wrapper.cpp'
    path.write_text(cpp)
    executable = tmp_path / 'wrapper'
    torch_include = Path(importlib.util.find_spec('torch').origin).parent / 'include'
    subprocess.run(['c++', '-std=c++17', '-O0', '-I', str(ROOT / 'src/ops'),
                    '-I', str(torch_include), str(path), '-o', str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
