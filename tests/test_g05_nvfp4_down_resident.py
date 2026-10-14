"""Run actual NVFP4 first10 Down install, typed DMA and absolute-SPM launcher bodies."""
from pathlib import Path
import re
from test_g05_q_gate_native import _compile_run
from test_siglip_cold_profile import _body
from test_g05_vision_owner_model import _block

ROOT = Path(__file__).resolve().parents[1]


def test_down_nvfp4_original_owner_validation_typed_preload_and_spm_registers(tmp_path):
    source = (ROOT / 'src/ops/rpu_linear.cpp').read_text()
    functions = []
    for marker in ('void check_pi05_resident_spm(',
                   'std::pair<at::Tensor, at::Tensor> rpu_pack_g05_down_resident_nvfp4(',
                   'void rpu_launch_g05_down_resident_nvfp4_preload(',
                   'void rpu_launch_g05_down_resident_nvfp4_acc16_kernel('):
        start = source.index(marker)
        functions.append(source[start:source.index('{', start)] + _body(source, marker))
    cpp = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>
#define TORCH_CHECK(ok,...) do{if(!(ok))throw std::runtime_error("rejected");}while(0)
namespace c10{using Half=uint16_t;}
int copies=0,allocations=0,flushes=0,lookups=0,dmas=0,stages=0;
namespace at{
constexpr int kByte=1,kHalf=2,kPrivateUse1=3;using IntArrayRef=std::vector<int64_t>;
struct Device{int kind=kPrivateUse1,index=0;int type()const{return kind;}bool operator==(Device b)const{return kind==b.kind&&index==b.index;}};
struct Tensor{uint64_t address=0x1234000000ULL;bool present=true,contig=true;int dtype=kByte;int64_t offset=0;Device dev;IntArrayRef shape{1024,2048};
 bool defined()const{return present;}int scalar_type()const{return dtype;}Device device()const{return dev;}bool is_contiguous()const{return contig;}int64_t storage_offset()const{return offset;}IntArrayRef sizes()const{return shape;}
 size_t nbytes()const{size_t n=1;for(auto v:shape)n*=v;return n*(dtype==kHalf?2:1);}template<class T>T*data_ptr()const{return reinterpret_cast<T*>(address);}
 Tensor cpu()const{++copies;auto t=*this;t.dev.kind=0;return t;}
 Tensor view(IntArrayRef dims)const{auto t=*this;t.shape=dims;assert(t.nbytes()==nbytes());return t;}
 Tensor permute(IntArrayRef dims)const{auto t=*this;assert(dims==IntArrayRef({2,0,1,3})||dims==IntArrayRef({1,0,2,3,4}));for(size_t i=0;i<dims.size();++i)t.shape[i]=shape[dims[i]];return t;}
 Tensor contiguous()const{return *this;}Tensor to(Device d)const{auto t=*this;t.dev=d;t.address=0x2300000000ULL+(++allocations)*0x10000000ULL;return t;}
};using TensorList=std::vector<Tensor>;
Tensor cat(const std::vector<Tensor>& v,int axis){assert(v.size()==10&&axis==1);auto t=v[0];for(const auto&x:v)assert(x.shape==t.shape&&x.dtype==kByte&&x.dev.kind==0);t.shape[1]*=10;return t;}}
struct SpmAllocator{static constexpr size_t SPM_USABLE=8387584;bool initialized=true;bool is_initialized()const{return initialized;}uint32_t addr(int,int)const{return 0x20000000;}}allocator;
#define SPM_ALLOC allocator
uint64_t RpuGetDevAddr(void*p){return reinterpret_cast<uint64_t>(p);}template<class T>void rpu_ddr_flush(T*){++flushes;}template<class T>void rpu_ddr_flush_force(T*){++flushes;}
bool capturing=false;struct RpuKernelGraph{static bool has_active(){return capturing;}};
enum class KernelId{G05_DOWN_RESIDENT_NVFP4_ACC16_M32N1024K512_N64};
struct KernelCache{bool available=true;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId)const{return available;}};
struct Kernel{std::map<int,uint16_t>regs;void reset_regs(){regs.clear();}void set_regs(int r,uint16_t v){regs[r]=v;}void set_regs(int,uint32_t)=delete;}kernel;
Kernel* get_kernel(KernelId){++lookups;return &kernel;}
#define GET_KERNEL(id) get_kernel(id)
struct Queue{bool broadcast=false;int launches=0;void set_broadcast_mode(bool b){broadcast=b;}void enqueu_kernel(Kernel&,std::vector<uint16_t>grid,std::vector<uint8_t>cores){assert(broadcast&&grid==std::vector<uint16_t>({16,1,1})&&cores==std::vector<uint8_t>({0,1,2,3,4,5,6,7}));++launches;}}queue;
Queue*get_queue(int n){assert(n==8);return &queue;}
#define GET_QUEUE(n) get_queue(n)
enum class GraphOuterFastDmaSide{Source};
namespace graph_dma{bool pending=false;at::Tensor owner;bool active(){return true;}void stage_census_fixed_burst(const at::Tensor&t,GraphOuterFastDmaSide,size_t offset,size_t bytes,size_t packets){assert(!pending&&offset==0&&bytes==t.nbytes()&&packets==8&&t.dtype==at::kByte);owner=t;pending=true;++stages;}}
void rpu_launch_ddr_scatter_spm_dma(c10::Half*p,int64_t n,int64_t stride,uint32_t destination,int cores){assert(graph_dma::pending&&reinterpret_cast<uint64_t>(p)==graph_dma::owner.address&&n*2==stride&&stride==graph_dma::owner.shape[1]&&cores==8&&destination%256==0);graph_dma::pending=false;++dmas;}
__FUNCTIONS__
template<class F>void reject(F f){int c=copies,a=allocations,fl=flushes,l=lookups,d=dmas,s=stages,la=queue.launches;try{f();assert(false);}catch(const std::runtime_error&){}assert(c==copies&&a==allocations&&fl==flushes&&l==lookups&&d==dmas&&s==stages&&la==queue.launches&&!graph_dma::pending);}
uint32_t gp(int r){return uint32_t(kernel.regs.at(r))|(uint32_t(kernel.regs.at(r+1))<<16);}
int main(){
 std::vector<at::Tensor>w(10),s(10);for(int i=0;i<10;++i){w[i].address+=i*0x400000ULL;s[i].address=w[i].address+0x200000;s[i].shape={256,1024};}
 auto pack=[&]{return rpu_pack_g05_down_resident_nvfp4(w,s);};
 auto packed=pack();assert(copies==20&&allocations==2&&packed.first.shape==at::IntArrayRef({8,2621440})&&packed.second.shape==at::IntArrayRef({8,327680})&&packed.first.dtype==at::kByte&&packed.second.dtype==at::kByte);
 capturing=true;reject(pack);capturing=false;
 for(int layer=0;layer<10;++layer)for(int family=0;family<2;++family)for(int fault=0;fault<11;++fault){auto&t=family?s[layer]:w[layer];auto save=t;if(fault==0)t.present=false;if(fault==1)t.dtype=at::kHalf;if(fault==2)t.contig=false;if(fault==3)t.dev.kind=0;if(fault==4)t.dev.index=1;if(fault==5)t.offset=1;if(fault==6)--t.shape[0];if(fault==7)t.address=0;if(fault==8)++t.address;if(fault==9)t.address=1ULL<<40;if(fault==10)t.address=(1ULL<<40)-256;reject(pack);t=save;}
 {auto save=s[9];s[9].address=w[0].address;reject(pack);s[9]=save;}
 {auto save=w;w.pop_back();reject(pack);w=save;}
 for(int family=0;family<2;++family){auto t=family?packed.second:packed.first;auto preload=[&]{rpu_launch_g05_down_resident_nvfp4_preload(t,0x20100000,family);};preload();assert(graph_dma::owner.address==t.address);for(int fault=0;fault<9;++fault){auto save=t;if(fault==0)t.dtype=at::kHalf;if(fault==1)t.offset=1;if(fault==2)t.contig=false;if(fault==3)t.dev.kind=0;if(fault==4)--t.shape[1];if(fault==5)t.address=0;if(fault==6)++t.address;if(fault==7)t.address=1ULL<<40;if(fault==8)t.address=(1ULL<<40)-256;reject(preload);t=save;}}
 reject([&]{rpu_launch_g05_down_resident_nvfp4_preload(packed.first,0x20100000,2);});
 reject([&]{rpu_launch_g05_down_resident_nvfp4_preload(packed.first,0x207fff00,0);});
 std::array<uint32_t,5>a{0x20010000,0x20100000,0x20020000,0x20080000,0x200a0000};
 auto launch=[&](uint16_t layer=9){rpu_launch_g05_down_resident_nvfp4_acc16_kernel(a[0],a[1],a[2],a[3],a[4],layer);};
 for(int layer:{0,8,9}){launch(layer);assert(gp(0)==a[0]&&gp(2)==a[1]&&gp(4)==a[2]&&gp(20)==a[3]&&gp(24)==a[4]);assert(gp(10)==1024&&kernel.regs.at(12)==512&&kernel.regs.at(13)==8&&kernel.regs.at(15)==8&&gp(16)==1024&&gp(18)==2048&&gp(22)==2048&&kernel.regs.at(26)==0&&kernel.regs.at(27)==layer);}
 reject([&]{launch(10);});reject([&]{launch(uint16_t(-1));});
 for(int i=0;i<5;++i){auto save=a[i];for(auto bad:{0u,save+1,0x207fffe0u}){a[i]=bad;reject([&]{launch();});}a[i]=save;}
 for(int i=0;i<5;++i)for(int j=0;j<i;++j){auto save=a[i];a[i]=a[j];reject([&]{launch();});a[i]=save;}
 allocator.initialized=false;reject([&]{launch();});allocator.initialized=true;KernelCache::instance().available=false;reject([&]{launch();});
}
'''
    _compile_run(tmp_path, 'g05_nvfp4_down_native', cpp.replace('__FUNCTIONS__', '\n'.join(functions)))


def test_down_nvfp4_complete_routes_cold_capability_lifetime_and_fallback(tmp_path):
    source = (ROOT / 'src/fused/rpu_qwen3_5_model.cpp').read_text()
    methods = [
        ('void prepare_action_nvfp4_down_resident()', 'void Qwen3_5Model::prepare_action_nvfp4_down_resident('),
        ('bool use_action_nvfp4_down_resident(int64_t rows,int64_t layer)const', 'bool Qwen3_5Model::use_action_nvfp4_down_resident('),
        ('std::vector<int64_t>action_nvfp4_down_resident_arguments(int64_t layer,int64_t preload)const', 'std::vector<int64_t> Qwen3_5Model::action_nvfp4_down_resident_arguments('),
        ('void preload_action_nvfp4_down_resident()', 'void Qwen3_5Model::preload_action_nvfp4_down_resident('),
        ('void launch_action_nvfp4_down_resident(int layer,const ChunkInfo&chunk)', 'void Qwen3_5Model::launch_action_nvfp4_down_resident('),
    ]
    definitions = '\n'.join(decl + _block(source, marker) for decl, marker in methods)
    constants = '\n'.join(re.findall(r'constexpr int64_t QWEN35_ACTION_DOWN_(?:RESIDENT|PRELOAD)_NVFP4_SITE\s*=.*?;', source, re.S))
    manifest_start = source.index('FmbPhysicalManifest Qwen3_5Model::') if 'FmbPhysicalManifest Qwen3_5Model::' in source else source.index('        if (use_action_nvfp4_down_resident(chunk.len,0))')
    linear = 'if(use_action_nvfp4_down_resident(chunk.len,0))' + _block(source[manifest_start:], 'if (use_action_nvfp4_down_resident(chunk.len,0))')
    preload = 'if(use_action_nvfp4_down_resident(physical_len,0))' + _block(source, 'if (use_action_nvfp4_down_resident(physical_len,0))')
    declarations = 'if(use_action_nvfp4_down_resident(action_len_,0))' + _block(source[source.index('std::vector<BufferDecl> decls;'):], 'if (use_action_nvfp4_down_resident(action_len_,0))')
    mlp = _block(source, 'void Qwen3_5Model::emit_mlp_and_output(')
    mlp = mlp[mlp.index('    if (use_action_nvfp4_down_resident(seq_len,layer_idx))'):mlp.index('    // Split the last decoder-layer output')]
    constants += '\n'.join(re.findall(r'constexpr int64_t QWEN35_TEXT_MLP_ALL_REDUCE_SITE\s*=.*?;', source, re.S))
    header = (ROOT / 'src/fused/rpu_qwen3_5_model.h').read_text()
    layout_body = _block(source, 'int64_t Qwen3_5Model::subclass_layout_hash(')
    layout = 'if(use_action_nvfp4_down_resident(action_len_,0))' + _block(layout_body, 'if (use_action_nvfp4_down_resident(action_len_,0))')
    identity = _block(header, 'std::vector<int64_t> kvinsert_cost_weight_identity() const')
    begin = identity.index('            identity.push_back(static_cast<int64_t>(action_nvfp4_down_resident_available_));')
    end = identity.index('            identity.push_back(static_cast<int64_t>(action_o_resident_available_));', begin)
    identity = identity[begin:end]
    cpp = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define TORCH_CHECK(ok,...) do{if(!(ok))throw std::runtime_error("rejected");}while(0)
namespace at{struct Tensor{int owner=0;bool defined()const{return owner!=0;}};}
void append_kvinsert_cost_tensor_identity(std::vector<int64_t>&v,const at::Tensor&t){v.push_back(t.owner);}
namespace detail{int64_t layout_mix(int64_t a,int64_t b){return (uint64_t(a)*1099511628211ULL)^uint64_t(b);}}
struct Weights{at::Tensor down_w,down_ws,down_negative_zero;};struct ChunkInfo{int64_t idx=0,offset=0,len=32;};
struct G05Nvfp4Tables{std::vector<at::Tensor>values=std::vector<at::Tensor>(5);static constexpr const char*names[]={"qts","ots","gts","uts","dts"};};
enum class KernelId{G05_DOWN_RESIDENT_NVFP4_ACC16_M32N1024K512_N64,G05_DOWN_RESIDENT_NVFP4_ACC32_M32N1024K512_N64};struct KernelCache{bool available=true,acc32=true;static KernelCache&instance(){static KernelCache c;return c;}bool has_loaded(KernelId id)const{return available&&(acc32||id!=KernelId::G05_DOWN_RESIDENT_NVFP4_ACC32_M32N1024K512_N64);}};
enum class FmbRouteFamily{LINEAR,MUTABLE_DMA,ALL_REDUCE};enum class FmbLinearRouteSelector{G05_NVFP4_RESIDENT_ACC16=10,G05_NVFP4_RESIDENT_ACC32=14};
enum class Qwen35TextMutableDmaRoute{DDR_SCATTER_TO_SPM=8};int64_t qwen35_text_dma_route(Qwen35TextMutableDmaRoute){return 8;}
struct Route{int64_t site;FmbRouteFamily family;int64_t selector,flags;std::vector<int64_t>args;int64_t invocation;};struct Manifest{std::vector<Route>routes;};
enum class StorageClass{Temp};enum class BufferScope{LayerWide};struct BufferDecl{const char*name;int64_t bytes,first,last;StorageClass storage;int per_layer;const char*alias;BufferScope scope;};
__CONSTANTS__
int packs=0,generation=0,dmas=0,launches=0;bool reject_pack=false;
std::vector<std::string>order;
void rpu_prepare_ring_all_reduce_input(uint32_t,int64_t,int64_t,int64_t,int64_t){order.push_back("prepare");}
void rpu_launch_all_reduce_sum_residual_kernel(uint32_t,uint32_t,uint32_t,int64_t,int64_t,int64_t,int64_t){order.push_back("reduce");}
std::pair<at::Tensor,at::Tensor>rpu_pack_g05_down_resident_nvfp4(const std::vector<at::Tensor>&w,const std::vector<at::Tensor>&s){TORCH_CHECK(!reject_pack,"pack failed");assert(w.size()==10&&s.size()==10);for(int i=0;i<10;++i)assert(w[i].owner==100+i&&s[i].owner==200+i);++packs;return {{1000+generation},{2000+generation}};}
void rpu_launch_g05_down_resident_nvfp4_preload(const at::Tensor&t,uint32_t dst,int64_t family){assert(t.owner==(family?2000:1000)+generation&&dst==uint32_t(family?0x20400000:0x20100000));++dmas;}
void rpu_launch_g05_down_resident_nvfp4_acc16_kernel(uint32_t x,uint32_t w,uint32_t y,uint32_t s,uint32_t ts,uint16_t layer){assert(layer<10&&x==0x20010000&&y==0x20020000&&w==0x20100000u+layer*262144u&&s==0x20400000u+layer*32768u&&ts==0x20080000);++launches;order.push_back("resident-down");}
void rpu_launch_g05_down_resident_nvfp4_acc32_kernel(uint32_t x,uint32_t w,uint32_t y,uint32_t s,uint32_t ts,uint16_t layer){rpu_launch_g05_down_resident_nvfp4_acc16_kernel(x,w,y,s,ts,layer);order.back()="resident-down-acc32";}
struct Model{
 template<class...T>bool action_mlp_tail_handoff(T...){return false;}
 template<class...T>void launch_action_mlp_tail(T...){assert(false);}
 static constexpr int64_t QWEN35_TEXT_MLP_TAIL_SITE=8475656447499929675LL;
 bool text_mlp_tail_available_=false;
 void prepare_text_mlp_tail(){}
 template<class...T>bool use_text_mlp_tail(T...)const{return false;}
 template<class...T>bool text_mlp_tail_handoff(T...){return false;}
 std::vector<int64_t>text_mlp_tail_arguments()const{return {};}
 template<class...T>void launch_text_mlp_tail(T...){assert(false);}

 bool action_nvfp4_o_resident_available_=true;
 bool use_action_nvfp4_o_resident(int64_t rows)const{return action_nvfp4_o_resident_available_&&nv&&rows==32&&use_action_mod_body();}
 bool linear_acc32_=false;
 bool nv=true,action_gateup_pair_nvfp4_=true,action_gateup_pair_available_=true,action_silu_round_mul_available_=true,body_mode=true,complete=true;
 bool action_nvfp4_down_resident_available_=false;std::pair<at::Tensor,at::Tensor>action_nvfp4_down_resident_;
 int64_t action_len_=32,action_num_steps_=10,body_iter=0;G05Nvfp4Tables action_nvfp4_;std::vector<Weights>layer_weights_=std::vector<Weights>(24);std::vector<Route>routes;
 Model(){for(int i=0;i<24;++i){layer_weights_[i].down_w.owner=100+i;layer_weights_[i].down_ws.owner=200+i;}}
 bool use_action_nvfp4(int64_t rows)const{return nv&&rows==32;}bool use_action_mod_body()const{return body_mode&&action_num_steps_==10;}
 Model&ctx(){return *this;}bool has_complete_physical_manifest()const{return complete;}
 uint32_t addr(int,const char*n)const{std::string s(n);return s=="gate"?0x20010000:s=="down"?0x20020000:s=="action_nvfp4_down_weight"?0x20100000:s=="action_nvfp4_down_scale"?0x20400000:0x20080000;}
 void consume_physical_route(FmbRouteFamily f,int64_t site,int64_t selector,int64_t flags,std::vector<int64_t>args,int64_t invocation){auto i=std::find_if(routes.begin(),routes.end(),[&](const auto&r){return r.site==site&&r.invocation==invocation;});TORCH_CHECK(i!=routes.end()&&i->family==f&&i->selector==selector&&i->flags==flags&&i->args==args,"stale route");}
 bool action_mode_=true;
 int64_t num_cores()const{return 8;}
 int64_t mlp_tp()const{return 8;}int64_t intermediate_size()const{return 4096;}
 bool use_action_down_partial(int64_t,int64_t)const{return false;}
 void launch_action_down_partial(int,const ChunkInfo&){assert(false);}
 template<class...T>void launch_linear(T...){order.push_back("ddr-down");}
 bool use_action_ring_rtc_residual(int64_t)const{return false;}
 template<class...T>void launch_action_ring_rtc_residual(T...){assert(false);}
 int64_t text_ring_route(int64_t,int64_t)const{return 2;}
 uint32_t layer_input_residual_addr(int,const ChunkInfo&)const{return 0x20060000;}
 void consume_manifest_route(FmbRouteFamily f,int64_t site,int64_t,int64_t){assert(f==FmbRouteFamily::ALL_REDUCE&&site==QWEN35_TEXT_MLP_ALL_REDUCE_SITE);}
 void apply_adaptive_residual_gate(uint32_t,uint32_t,const ChunkInfo&){order.push_back("residual");}
 void mlp_down(int layer_idx){ChunkInfo chunk;const auto&lw=layer_weights_[layer_idx];int64_t seq_len=32,h=1024;uint32_t product_buf=addr(0,"gate"),residual_buf=0x20060000;const at::Tensor*down_scale=&lw.down_ws;__MLP__}
 __METHODS__
 int64_t hash()const{int64_t hash=0;__LAYOUT__ return hash;}
 std::vector<int64_t>cost()const{std::vector<int64_t>identity;__IDENTITY__ return identity;}
 void plan(){Manifest manifest;ChunkInfo chunk;int64_t physical_len=32;__LINEAR__ __PRELOAD__ routes=manifest.routes;for(const auto&r:routes)for(auto a:r.args)assert(a>=0);}
 std::vector<BufferDecl>buffers(){std::vector<BufferDecl>decls;__DECLS__ return decls;}
};
template<class F>void rejects(F f){int p=packs,d=dmas,l=launches;try{f();assert(false);}catch(const std::runtime_error&){}assert(packs==p&&dmas==d&&launches==l);}
int main(){Model m;m.prepare_action_nvfp4_down_resident();assert(packs==1&&m.use_action_nvfp4_down_resident(32,0)&&!m.use_action_nvfp4_down_resident(16,0));m.plan();assert(m.routes.size()==12);auto b=m.buffers();assert(b.size()==2&&b[0].bytes==2621440&&b[1].bytes==327680);for(auto&v:b)assert(v.first==0&&v.last==31&&v.scope==BufferScope::LayerWide);
 for(int body=0;body<10;++body){m.body_iter=body;m.preload_action_nvfp4_down_resident();for(int l=0;l<10;++l)m.launch_action_nvfp4_down_resident(l,{});}assert(dmas==2&&launches==100);
 for(int layer=0;layer<24;++layer){order.clear();m.mlp_down(layer);assert(order==std::vector<std::string>({layer<10?"resident-down":"ddr-down","reduce","residual"}));}
 {auto off=m;off.action_nvfp4_down_resident_available_=false;for(int layer:{0,9,10,23}){order.clear();off.mlp_down(layer);assert(order==std::vector<std::string>({"ddr-down","reduce","residual"}));}}
 m.body_iter=0;for(int fault=0;fault<4;++fault){auto bad=m;ChunkInfo c;if(fault==0)bad.complete=false;if(fault==1)c.idx=1;if(fault==2)c.offset=1;if(fault==3)c.len=16;rejects([&]{bad.launch_action_nvfp4_down_resident(0,c);});}
 for(int l:{-1,10})rejects([&]{m.launch_action_nvfp4_down_resident(l,{});});
 for(size_t route=0;route<m.routes.size();++route)for(size_t arg=0;arg<m.routes[route].args.size();++arg){auto bad=m;++bad.routes[route].args[arg];if(route<10)rejects([&]{bad.launch_action_nvfp4_down_resident(route,{});});else rejects([&]{bad.preload_action_nvfp4_down_resident();});}
 for(int fault=0;fault<3;++fault){auto off=m;if(fault==0)off.nv=false;if(fault==1)off.action_nvfp4_o_resident_available_=false;if(fault==2)KernelCache::instance().available=false;int p=packs;off.prepare_action_nvfp4_down_resident();assert(!off.action_nvfp4_down_resident_available_&&!off.action_nvfp4_down_resident_.first.defined()&&packs==p);off.plan();assert(off.routes.empty());KernelCache::instance().available=true;}
 {auto bad=m;reject_pack=true;rejects([&]{bad.prepare_action_nvfp4_down_resident();});assert(!bad.action_nvfp4_down_resident_available_&&!bad.action_nvfp4_down_resident_.first.defined());reject_pack=false;}
 m.body_mode=false;assert(!m.use_action_nvfp4_down_resident(32,0)&&m.buffers().empty());m.body_mode=true;
 auto old_hash=m.hash();auto old_cost=m.cost();for(int owner=0;owner<2;++owner){auto changed=m;auto& tensor=owner?changed.action_nvfp4_down_resident_.second:changed.action_nvfp4_down_resident_.first;++tensor.owner;assert(changed.hash()!=old_hash&&changed.cost()!=old_cost);}
 {auto disabled=m;disabled.action_nvfp4_down_resident_available_=false;assert(disabled.hash()!=old_hash&&disabled.cost()!=old_cost);}
 ++generation;m.prepare_action_nvfp4_down_resident();assert(m.action_nvfp4_down_resident_.first.owner==1001);m.plan();m.preload_action_nvfp4_down_resident();
 assert(m.action_nvfp4_down_resident_arguments(0,-1).back()==1);m.action_nvfp4_down_resident_available_=false;assert(m.action_nvfp4_down_resident_arguments(0,-1).back()==0);
}
'''
    for key,value in [('CONSTANTS',constants),('METHODS',definitions),('LINEAR',linear),('PRELOAD',preload),('DECLS',declarations),('MLP',mlp),('LAYOUT',layout),('IDENTITY',identity)]:
        cpp = cpp.replace('__'+key+'__',value)
    _compile_run(tmp_path, 'g05_nvfp4_down_model', cpp)
    acc32 = cpp.replace('bool linear_acc32_=false;', 'bool linear_acc32_=true;', 1)
    head, main = acc32.split('int main(){', 1)
    main = main.replace('"resident-down"', '"resident-down-acc32"')
    checks = 'assert(m.routes[0].selector==14&&m.action_nvfp4_down_resident_arguments(0,-1)[11]==32);'
    checks += 'auto alias=m;alias.routes[0].site=5936546264671002449LL;rejects([&]{alias.launch_action_nvfp4_down_resident(0,{});});auto stale=m;stale.routes[0].args[11]=16;rejects([&]{stale.launch_action_nvfp4_down_resident(0,{});});'
    checks += 'stale=m;stale.body_iter=0;stale.routes[10].args[11]=16;rejects([&]{stale.preload_action_nvfp4_down_resident();});'
    checks += 'KernelCache::instance().acc32=false;auto old=m;old.prepare_action_nvfp4_down_resident();assert(!old.action_nvfp4_down_resident_available_);old.plan();assert(old.routes.empty());KernelCache::instance().acc32=true;'
    main = main.replace(' for(int body=0;', checks+' for(int body=0;', 1)
    _compile_run(tmp_path, 'g05_nvfp4_down_acc32_model', head+'int main(){'+main)
