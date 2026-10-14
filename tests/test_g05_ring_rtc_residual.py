"""Actual native ring submission and model COMPLETE/replicated-owner boundaries."""
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def _with_actual_g05_site_helpers(cpp):
    """Keep compiled model doubles on the production precision/site mapping."""
    source = (ROOT / 'src/fused/rpu_qwen3_5_model.cpp').read_text()
    helpers = ('text_gateup_pair_site', 'action_gateup_pair_site',
               'action_nvfp4_swiglu_site', 'action_nvfp4_o_resident_site',
               'action_nvfp4_down_resident_site', 'high_precision_site')
    definitions = []
    constants = []
    for high_site in ('QWEN35_ACTION_SILU_HIGH_MUL_SITE',
                      'QWEN35_ACTION_NEWTON_NORM_SHIFT_SITE'):
        if (high_site in cpp and not re.search(
                r'constexpr\s+int64_t\s+' + high_site + r'\s*=', cpp)):
            constants.append(high_site)
    # Historical doubles exercise BASE. Keep that default explicit while
    # dedicated HIGH tests supply the actual policy, helpers and descriptors.
    if re.search(r'\bhigh_precision_\b', cpp) and not re.search(r'\bbool\s+high_precision_\b', cpp):
        definitions.append('bool high_precision_ = false;')
    if ('high_precision_arguments(' in cpp and not re.search(
            r'std::vector\s*<\s*int64_t\s*>\s+high_precision_arguments\(', cpp)):
        definitions.append('std::vector<int64_t> high_precision_arguments() const '
                           '{ TORCH_CHECK(!high_precision_); return {}; }')
    if ('consume_high_precision_math(' in cpp and not re.search(
            r'void\s+consume_high_precision_math\(', cpp)):
        definitions.append('void consume_high_precision_math() '
                           '{ TORCH_CHECK(false, "BASE double cannot execute HIGH"); }')
    if ('action_silu_high_mul_available_' in cpp and not re.search(
            r'\bbool\s+action_silu_high_mul_available_\b', cpp)):
        definitions.append('bool action_silu_high_mul_available_ = false;')
    if ('use_action_silu_high_mul(' in cpp and not re.search(
            r'bool\s+use_action_silu_high_mul\(', cpp)):
        definitions.append('bool use_action_silu_high_mul(int64_t) const '
                           '{ TORCH_CHECK(!high_precision_); return false; }')
    if ('action_silu_high_mul_arguments(' in cpp and not re.search(
            r'std::vector\s*<\s*int64_t\s*>\s+action_silu_high_mul_arguments\(', cpp)):
        definitions.append('std::vector<int64_t> action_silu_high_mul_arguments() const '
                           '{ TORCH_CHECK(!high_precision_); return {}; }')
    if ('prepare_action_silu_high_mul(' in cpp and not re.search(
            r'void\s+prepare_action_silu_high_mul\(', cpp)):
        definitions.append('void prepare_action_silu_high_mul() '
                           '{ TORCH_CHECK(!high_precision_); }')
    if ('launch_action_silu_high_mul(' in cpp and not re.search(
            r'void\s+launch_action_silu_high_mul\(', cpp)):
        definitions.append('void launch_action_silu_high_mul(int,const ChunkInfo&) '
                           '{ TORCH_CHECK(false, "BASE double cannot execute HIGH fusion"); }')
    if (('action_newton_norm_shift_available_' in cpp or
         'use_action_newton_norm_shift(' in cpp) and not re.search(
            r'\bbool\s+action_newton_norm_shift_available_\b', cpp)):
        definitions.append('bool action_newton_norm_shift_available_ = false;')
    if ('use_action_newton_norm_shift(' in cpp and not re.search(
            r'bool\s+use_action_newton_norm_shift\(', cpp)):
        definitions.append('bool use_action_newton_norm_shift(int64_t) const '
                           '{ TORCH_CHECK(!high_precision_ || '
                           '!action_newton_norm_shift_available_); return false; }')
    if ('action_newton_norm_shift_arguments(' in cpp and not re.search(
            r'std::vector\s*<\s*int64_t\s*>\s+action_newton_norm_shift_arguments\(', cpp)):
        definitions.append('std::vector<int64_t> action_newton_norm_shift_arguments() const '
                           '{ TORCH_CHECK(!high_precision_); return {}; }')
    if ('prepare_action_newton_norm_shift(' in cpp and not re.search(
            r'void\s+prepare_action_newton_norm_shift\(', cpp)):
        definitions.append('void prepare_action_newton_norm_shift() '
                           '{ TORCH_CHECK(!high_precision_); }')
    if ('apply_action_newton_norm_shift(' in cpp and not re.search(
            r'void\s+apply_action_newton_norm_shift\(', cpp)):
        definitions.append('void apply_action_newton_norm_shift(uint32_t,uint32_t,const ChunkInfo&) '
                           '{ TORCH_CHECK(false, "BASE double cannot execute HIGH normalization"); }')
    declarations = []
    if 'unary_precision(' in cpp and not re.search(r'RpuUnaryPrecision\s+unary_precision\(', cpp):
        definitions.append('RpuUnaryPrecision unary_precision() const ' + _block(
            source, 'RpuUnaryPrecision Qwen3_5Model::unary_precision('))
        if not re.search(r'enum\s+class\s+RpuUnaryPrecision\b', cpp):
            declarations.append('enum class RpuUnaryPrecision : uint8_t ' + _block((ROOT / 'src/core/rpu_math_precision.h').read_text(),
                                       'enum class RpuUnaryPrecision') + ';')
        # Existing launcher doubles now accept the real optional precision
        # argument and reject a HIGH launch in a BASE-only fixture.
        cpp = re.sub(
            r'(void\s+rpu_launch_eltwise_unary_spm_kernel\([^{};]*?)(\)\s*\{)',
            lambda match: (match[1] + ', RpuUnaryPrecision precision = RpuUnaryPrecision::BASE'
                           + match[2] + '\n assert(precision == RpuUnaryPrecision::BASE);')
            if match[1].count(',') == 5 else match[0], cpp)
    if 'rmsnorm_route(' in cpp and not re.search(r'RpuRmsNormSpmRoute\s+rmsnorm_route\(', cpp):
        decls = (ROOT / 'src/core/rpu_kernel_decls.h').read_text()
        norm_enum = 'enum class RpuRmsNormSpmRoute : int64_t ' + _block(decls, 'enum class RpuRmsNormSpmRoute') + ';'
        if re.search(r'enum\s+class\s+RpuRmsNormSpmRoute\b', cpp):
            cpp = re.sub(r'enum\s+class\s+RpuRmsNormSpmRoute[^{}]*\{[^{}]*\};',
                         '', cpp)
        declarations.append(norm_enum)
        declarations.append('inline constexpr int64_t RPU_RMSNORM_SCM_U16_MAX = 65535;\n'
                            'inline constexpr int64_t RPU_RMSNORM_MAX_COLS = '
                            'RPU_RMSNORM_SCM_U16_MAX / 2;\n'
                            'RpuRmsNormSpmRoute rpu_resolve_high_precision_rmsnorm_spm_route('
                            'int64_t rows, int64_t cols) ' + _block(
                                decls, 'inline RpuRmsNormSpmRoute rpu_resolve_high_precision_rmsnorm_spm_route('))
        definitions.append('RpuRmsNormSpmRoute rmsnorm_route(int64_t rows, int64_t cols) const '
                           + _block(source, 'RpuRmsNormSpmRoute Qwen3_5Model::rmsnorm_route('))
        cpp = re.sub(
            r'(void\s+rpu_launch_rmsnorm_spm_kernel\([^{};]*?)(\)\s*\{)',
            lambda match: (match[1] + ', RpuRmsNormSpmRoute route = RpuRmsNormSpmRoute::BASE, int cores = 8'
                           + match[2] + '\n assert(route == RpuRmsNormSpmRoute::BASE && cores == 8);')
            if match[1].count(',') == 5 else match[0], cpp)
    for helper in helpers:
        if helper + '(' not in cpp or re.search(r'int64_t\s+' + helper + r'\(\)\s*const\s*{', cpp):
            continue
        body = _block(source, 'int64_t Qwen3_5Model::' + helper + '(')
        definitions.append('int64_t ' + helper + '() const ' + body)
        for constant in re.findall(r'QWEN35_[A-Z0-9_]+', body):
            if constant in constants or re.search(r'constexpr\s+int64_t\s+' + constant + r'\s*=', cpp):
                continue
            constants.append(constant)
    if not definitions and not constants and not declarations:
        return cpp
    declarations.extend(re.search(r'constexpr int64_t ' + constant + r'\s*=.*?;', source, re.S)[0]
                        for constant in constants)
    match = re.search(r'struct Model\s*{', cpp)
    assert match, 'site helpers require the actual Model double'
    cpp = cpp[:match.end()] + '\n'.join(definitions) + '\n' + cpp[match.end():]
    # Precision types must precede launcher prototypes as well as Model.
    global_point = re.search(r'void\s+rpu_launch_(?:eltwise_unary|rmsnorm)_spm_kernel\(', cpp)
    at = global_point.start() if global_point else match.start()
    return cpp[:at] + '\n'.join(declarations) + '\n' + cpp[at:]

def _compile_run(tmp_path, name, cpp):
    cpp = _with_actual_g05_site_helpers(cpp)
    path, exe = tmp_path / (name + '.cpp'), tmp_path / name
    path.write_text(cpp)
    compiled = subprocess.run(['c++', '-std=c++17', '-O0', '-Wall', '-Wextra', '-Werror',
                               '-I', str(ROOT / 'src/ops'), str(path), '-o', str(exe)],
                              capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(exe)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

def _block(text, marker):
    begin = text.index("{", text.index(marker))
    depth = 0
    for end in range(begin, len(text)):
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
            if depth == 0:
                return text[begin:end + 1]
    raise AssertionError(marker)


def test_actual_ring_wrapper_guards_shared_scm_and_no_ddr_census(tmp_path):
    source=(ROOT/'src/core/rpu_runtime_extras.cpp').read_text()
    functions=["struct RingAdaRmsOperands"+_block(source,"struct RingAdaRmsOperands")+";"]
    for name in ['ring_all_reduce_sum_residual_admitted','launch_ring_all_reduce_sum_residual','rpu_launch_g05_ring_rtc_residual_spm_kernel']:
        marker=('bool ' if name.startswith('ring_') else 'void ')+name+'('
        start=source.index(marker)
        functions.append(source[start:source.index('{',start)]+_block(source,marker))
    ids=sorted(set(re.findall(r'KernelId::(\w+)',functions[1]+functions[2])))
    cpp=r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
#define TORCH_CHECK(x,...) do{if(!(x))throw std::runtime_error("rejected");}while(0)
#define TORCH_INTERNAL_ASSERT(x,...) TORCH_CHECK(x)
namespace c10{struct Half{uint16_t x;explicit Half(double v):x(v==1.0/1024?0x1400:0x11){}};}
constexpr int kKernelRegOk=0;
constexpr int kRingCoreNum=8;constexpr uint32_t kRingWarpNum=8,kRingRoundElemMax=31*256*8;constexpr uint64_t kRingNopaceChunkBytes=11*1024;
struct SpmAllocator{static constexpr uint32_t SPM_USABLE=8387584;bool initialized=true;bool is_initialized()const{return initialized;}uint32_t addr(int core,uint32_t off)const{return 0x40000000u+core*0x1000000u+off;}}allocator;
#define SPM_ALLOC allocator
namespace v3{enum class FmbSharedAllReduceRouteSelector{RING_PI05_XOR3=9};int64_t fmb_ring_all_reduce_route_selector(int64_t,int64_t,bool,int){return 9;}}
uint64_t RpuGetDevAddr(c10::Half*){return 0;}
enum class KernelId{__IDS__};
struct Kernel_t{KernelId id;std::map<int,uint16_t>regs;int reset_regs(){regs.clear();return kKernelRegOk;}int set_regs(uint32_t i,uint16_t v){regs[i]=v;return 0;}}kernel;
struct KernelCache{bool loaded=true;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId)const{return loaded;}Kernel_t* get(KernelId)const{return loaded?&kernel:nullptr;}};
int gets=0,stages=0;
enum class GraphKernelNoDdrProof{RingAllReduceSpmResidual};
struct RpuKernelGraph{bool census=true;static RpuKernelGraph&active(){static RpuKernelGraph g;return g;}bool kernel_register_census_active()const{return census;}void stage_kernel_no_ddr(KernelId id,GraphKernelNoDdrProof p){assert(id==KernelId::G05_RING_RTC_SUB_GATED_RESIDUAL_M32H1024&&p==GraphKernelNoDdrProof::RingAllReduceSpmResidual);++stages;}};
Kernel_t*get(KernelId id){assert(stages>gets);++gets;kernel.id=id;return &kernel;}
#define GET_KERNEL(id) get(id)
void rpu_set_legacy_scm_u16_checked(Kernel_t*k,uint32_t reg,uint16_t value,const char*){k->set_regs(reg,value);}
struct Submission{KernelId id;std::map<int,uint16_t>regs;std::vector<uint16_t>grid;std::vector<uint8_t>cores;};
struct Queue{bool broadcast=false;std::vector<Submission>submissions;void set_broadcast_mode(bool b){broadcast=b;}void enqueu_kernel(Kernel_t&k,std::vector<uint16_t>grid,std::vector<uint8_t>cores){assert(broadcast);submissions.push_back({k.id,k.regs,grid,cores});}}queue;
Queue*get_queue(int cores){assert(cores==8);return &queue;}
#define GET_QUEUE(c) get_queue(c)
struct G05RingRtcEpilogue{uint32_t clean_gate,noisy_gate;int64_t clean_rows;};
struct G05ActionTailNorm{uint32_t clean_scale,clean_shift,noisy_scale,noisy_shift;double eps;};
__FUNCTIONS__
uint32_t pair(const Submission&s,int r){return uint32_t(s.regs.at(r))|(uint32_t(s.regs.at(r+1))<<16);}
template<class F>void reject(F f){auto n=queue.submissions.size();int k=gets,c=stages;try{f();assert(false);}catch(const std::runtime_error&){}assert(n==queue.submissions.size()&&k==gets&&c==stages);}
int main(){
 const uint32_t base=allocator.addr(0,0),x=base+0x10000,r=base+0x30000,y=base+0x50000,c=base+0x70000,n=c+6144;
 auto call=[&](uint32_t a,uint32_t b,uint32_t clean,uint32_t noisy,uint32_t out,int split,int cores=8){rpu_launch_g05_ring_rtc_residual_spm_kernel(a,b,clean,noisy,out,split,cores);};
 for(int split=0;split<=32;++split)for(auto noisy:{n,c}){
  call(x,r,c,noisy,y,split);const auto&s=queue.submissions.back();assert(s.id==KernelId::G05_RING_RTC_SUB_GATED_RESIDUAL_M32H1024&&s.grid==std::vector<uint16_t>({8,1,1})&&s.cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));
  assert(s.regs.at(0)==8&&s.regs.at(1)==1&&pair(s,2)==63488&&pair(s,4)==r-base&&s.regs.at(6)==0);
  assert(pair(s,18)==c&&pair(s,20)==r&&pair(s,22)==noisy&&s.regs.at(24)==split&&s.regs.at(64)==8&&s.regs.at(65)==1&&s.regs.at(66)==1);
  for(int core=0;core<8;++core)for(int k=0;k<4;++k){uint32_t actual=uint32_t(s.regs.at(4096+k*16+core))|(uint32_t(s.regs.at(4096+k*16+core+8))<<16);uint32_t expected[]={allocator.addr(core,x-base)-base,allocator.addr(core,y-base)-base,uint32_t(core*8192),4096};assert(actual==expected[k]);}
 }
 for(int split:{-1,33})reject([&]{call(x,r,c,n,y,split);});
 for(int cores:{0,1,4,7,9})reject([&]{call(x,r,c,n,y,5,cores);});
 for(int index=0;index<5;++index)for(auto bad:{base-256,x+1,base+SpmAllocator::SPM_USABLE-256,0xffffff00u}){std::vector<uint32_t>a{x,r,c,n,y};a[index]=bad;reject([&]{call(a[0],a[1],a[2],a[3],a[4],5);});}
 for(auto out:{x,x+256,r,r+256,c,n})reject([&]{call(x,r,c,n,out,5);});
 for(auto gate:{x,r,y})reject([&]{call(x,r,gate,n,y,5);});
 reject([&]{call(x,r,c,c+256,y,5);});
 reject([&]{call(x,x,c,n,y,5);});
 KernelCache::instance().loaded=false;reject([&]{call(x,r,c,n,y,5);});KernelCache::instance().loaded=true;
 allocator.initialized=false;reject([&]{call(x,r,c,n,y,5);});
}
'''
    _compile_run(tmp_path,'ring_rtc_wrapper',cpp.replace('__IDS__',','.join(ids)).replace('__FUNCTIONS__','\n'.join(functions)))


def test_actual_model_composed_route_replication_and_old_ref_fallback(tmp_path):
    source=(ROOT/'src/fused/rpu_qwen3_5_model.cpp').read_text()
    methods=[('bool use_action_mod_body()const','bool Qwen3_5Model::use_action_mod_body('),
             ('bool use_action_ring_rtc_residual(int64_t rows)const','bool Qwen3_5Model::use_action_ring_rtc_residual('),
             ('std::vector<int64_t>action_ring_rtc_residual_arguments()const','std::vector<int64_t> Qwen3_5Model::action_ring_rtc_residual_arguments('),
             ('void launch_action_ring_rtc_residual(uint32_t input,uint32_t residual,uint32_t output,int64_t ring_site,const ChunkInfo&chunk)','void Qwen3_5Model::launch_action_ring_rtc_residual(')]
    constants='\n'.join(re.findall(r'constexpr int64_t QWEN35_(?:TEXT_(?:FULL|MLP)_ALL_REDUCE|ACTION_RING_RTC_RESIDUAL)_SITE\s*=.*?;',source,re.S))
    blocks=[]
    for method in ['build_full_attention','emit_mlp_and_output']:
        body=_block(source,'void Qwen3_5Model::'+method+'(');start=body.index('if (use_action_ring_rtc_residual(seq_len))');stop=body.index('\n    // ',start)
        blocks.append(body[start:stop])
    cpp=r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define DWIDTH 2
#define TORCH_CHECK(x,...) do{if(!(x))throw std::runtime_error("rejected");}while(0)
__CONSTANTS__
enum class FmbRouteFamily{ALL_REDUCE,GRAPH_SCHEDULE};
struct ChunkInfo{int64_t len=32,offset=0,idx=2;};
struct Route{FmbRouteFamily family;int64_t site,selector,flags;std::vector<int64_t>args;int64_t invocation;};
struct Call{uint32_t input,residual,clean,noisy,output;int64_t rows;int cores;};
std::vector<Call> calls;int original_ring=0,original_gate=0;
void rpu_launch_g05_ring_rtc_residual_spm_kernel(uint32_t a,uint32_t b,uint32_t c,uint32_t n,uint32_t o,int64_t clean,int cores){calls.push_back({a,b,c,n,o,clean,cores});}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t,uint32_t,uint32_t,int64_t,int64_t,int,int){++original_ring;}
struct Model{
 bool use_action_nvfp4(int64_t)const{return false;}

 bool action_ring_rtc_residual_available_=true,z2_bound_=false,linear_acc32_=false,action_mod_body_fp16_=false,action_mod_body_w8_=true;
 bool action_mode_=true,action_step_active_=true,action_loop_active_=true,has_gdn_=false,action_rtc_active_=true,complete=true;
 int64_t action_num_steps_=10,action_len_=32,action_rtc_prefix_len_=5,layers=24,h=1024,cores=8,tp=8;
 uint32_t adaptive_mod_row_offset_=49*6144;
 std::vector<Route>routes;int consumed=0;
 int64_t hidden_size()const{return h;}int num_cores()const{return cores;}int64_t num_layers()const{return layers;}int attn_tp()const{return tp;}int mlp_tp()const{return tp;}
 bool has_complete_physical_manifest()const{return complete;}Model&ctx(){return *this;}
 uint32_t addr(int,const char*name)const{std::string s(name);return 0x40000000+(s=="oproj"?0x10000:s=="down"?0x30000:s=="residual1"?0x50000:s=="residual2"?0x70000:0x90000);}
 int64_t text_ring_route(int64_t,int64_t)const{return 2;}
 uint32_t layer_input_residual_addr(int,const ChunkInfo&)const{return addr(0,"residual1");}
 void consume_physical_route(FmbRouteFamily family,int64_t site,int64_t sel,int64_t flags,std::vector<int64_t>args,int64_t invocation){for(auto&r:routes)if(r.family==family&&r.site==site){TORCH_CHECK(sel==r.selector&&flags==r.flags&&args==r.args&&invocation==r.invocation);++consumed;return;}TORCH_CHECK(false);}
 void consume_manifest_route(FmbRouteFamily f,int64_t site,int64_t sel,int64_t invocation){consume_physical_route(f,site,sel,0,{},invocation);}
 void apply_adaptive_residual_gate(uint32_t,uint32_t,const ChunkInfo&){++original_gate;}
 __METHODS__
 void plan(ChunkInfo chunk={}){routes={{FmbRouteFamily::ALL_REDUCE,QWEN35_TEXT_FULL_ALL_REDUCE_SITE,2,0,{},chunk.idx},{FmbRouteFamily::ALL_REDUCE,QWEN35_TEXT_MLP_ALL_REDUCE_SITE,2,0,{},chunk.idx}};if(use_action_ring_rtc_residual(chunk.len))routes.push_back({FmbRouteFamily::GRAPH_SCHEDULE,QWEN35_ACTION_RING_RTC_RESIDUAL_SITE,1,0,action_ring_rtc_residual_arguments(),chunk.idx});}
 void attention(ChunkInfo chunk={}){int64_t seq_len=chunk.len;int layer_idx=0;int tp=attn_tp();__ATTENTION__}
 void mlp(ChunkInfo chunk={}){int64_t seq_len=chunk.len;uint32_t residual_buf=addr(0,"residual2");__MLP__}
};
template<class F>void reject(F f){auto n=calls.size();try{f();assert(false);}catch(const std::runtime_error&){}assert(n==calls.size());}
int main(){
 Model m;m.plan();assert(m.routes.size()==3);auto expected=m.action_ring_rtc_residual_arguments();assert(expected==std::vector<int64_t>({1,32,1024,8,16,1,1,4,8,4,1,0,32,6144}));
 for(int clean=0;clean<=32;++clean){m.action_rtc_prefix_len_=clean;m.attention();m.mlp();for(size_t i=calls.size()-2;i<calls.size();++i){auto&c=calls[i];assert(c.rows==clean&&c.cores==8&&c.clean==m.addr(0,"adaptive_mod")+m.adaptive_mod_row_offset_+4096&&c.noisy==c.clean+6144);}}
 assert(original_ring==0&&original_gate==0&&m.consumed==33*4);
 m.action_rtc_active_=false;m.attention();assert(calls.back().rows==0&&calls.back().clean==calls.back().noisy);
 for(int i=0;i<14;++i){auto bad=m;++bad.routes.back().args[i];reject([&]{bad.attention();});}
 for(int route=0;route<3;++route)for(int fault=0;fault<7;++fault){auto bad=m;auto&r=bad.routes[route];if(fault==0)++r.site;if(fault==1)++r.selector;if(fault==2)++r.flags;if(fault==3)++r.invocation;if(fault==4)r.family=r.family==FmbRouteFamily::ALL_REDUCE?FmbRouteFamily::GRAPH_SCHEDULE:FmbRouteFamily::ALL_REDUCE;if(fault==5)bad.complete=false;if(fault==6)bad.routes.erase(bad.routes.begin()+route);reject([&]{if(route==1)bad.mlp();else bad.attention();});}
 for(int operand=0;operand<3;++operand){std::vector<uint32_t>a{m.addr(0,"oproj"),m.addr(0,"residual1"),m.addr(0,"residual2")};a[operand]+=256;reject([&]{m.launch_action_ring_rtc_residual(a[0],a[1],a[2],QWEN35_TEXT_FULL_ALL_REDUCE_SITE,{});});}
 reject([&]{m.attention({32,1,2});});reject([&]{m.launch_action_ring_rtc_residual(0,0,0,123,{});});
 for(int fault=0;fault<12;++fault){if(fault==1)continue;auto old=m;if(fault==0)old.action_ring_rtc_residual_available_=false;if(fault==1)old.linear_acc32_=true;if(fault==2)old.action_mod_body_w8_=false;if(fault==3)old.action_loop_active_=false;if(fault==4)old.action_step_active_=false;if(fault==5)old.action_num_steps_=1;if(fault==6)old.action_len_=16;if(fault==7)old.action_mode_=false;if(fault==8)old.has_gdn_=true;if(fault==9)old.tp=4;if(fault==10)old.z2_bound_=true;if(fault==11)old.h=2048;assert(!old.use_action_ring_rtc_residual(32));}
 // Both proven FP16 and W8 owners reuse the same neutral ring with selected-mode receipts.
 for(bool acc32:{false,true})for(bool fp16:{false,true}){auto x=m;x.linear_acc32_=acc32;x.action_mod_body_w8_=!fp16;x.action_mod_body_fp16_=fp16;x.plan();assert(x.routes.size()==3&&x.action_ring_rtc_residual_arguments()[4]==(acc32?32:16));const auto before=calls.size();x.attention();x.mlp();assert(calls.size()==before+2);x.routes.back().args[4]=acc32?16:32;reject([&]{x.attention();});}
 auto old=m;old.action_ring_rtc_residual_available_=false;old.plan();assert(old.routes.size()==2);old.attention();old.mlp();assert(original_ring==2&&original_gate==2);
}
'''
    cpp=cpp.replace('__CONSTANTS__',constants).replace('__METHODS__','\n'.join(d+_block(source,mark) for d,mark in methods)).replace('__ATTENTION__',blocks[0]).replace('__MLP__',blocks[1])
    _compile_run(tmp_path,'ring_rtc_model',cpp)
