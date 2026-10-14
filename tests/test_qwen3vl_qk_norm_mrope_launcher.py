"""Execute the production exact QK launcher: real body, SDK overloads, failure boundaries."""
from pathlib import Path
import subprocess
from test_qwen3vl_prefill_gateup_launcher import definition

ROOT=Path(__file__).resolve().parents[1]
NAME='rpu_launch_qwen3vl_4b_decode_qk_norm_mrope_spm_kernel'
KV_NAME='rpu_launch_qwen3vl_4b_decode_qk_norm_mrope_kv_insert_spm_kernel'


def test_actual_launcher_dynamic_position_tables_and_rejections(tmp_path):
    source=(ROOT/'src/ops/rpu_mrope.cpp').read_text()
    wrapper=definition(source,'static void launch_qwen3vl_4b_decode_qk_norm_mrope(')
    wrapper+='\n'+definition(source,f'void {NAME}(')+'\n'+definition(source,f'void {KV_NAME}(')
    decls=(ROOT/'src/core/rpu_kernel_decls.h').read_text();decl=''
    for name in (NAME,KV_NAME):
        start=decls.index(f'void {name}(');decl+=decls[start:decls.index(';',start)+1]+'\n'
    harness=r'''
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>
template<class...T>void check(bool x,T&&...){if(!x)throw std::runtime_error("reject");}
#define TORCH_CHECK(...) check(__VA_ARGS__)
namespace c10 {enum class DeviceType{PrivateUse1,CPU};enum class Layout{Strided,Sparse};
struct Half {float value;uint16_t x;Half(float f):value(f),x(f==1.0f/128?0x2000:0x11){}operator float()const{return value;}};}
namespace at {constexpr int kHalf=1,kInt=2,kFloat=3;
struct Tensor {std::vector<int64_t> shape{1024,64};int type=kHalf;bool present=true,contiguous=true;
 c10::DeviceType device_kind=c10::DeviceType::PrivateUse1;c10::Layout layout_kind=c10::Layout::Strided;
 uint64_t address=0x1200000000ULL;int64_t offset=0;size_t bytes=1024*128;std::vector<int32_t> values=std::vector<int32_t>(1024*3,0);
 bool defined()const{return present;}struct Device{c10::DeviceType kind;c10::DeviceType type()const{return kind;}};Device device()const{return {device_kind};}
 int scalar_type()const{return type;}c10::Layout layout()const{return layout_kind;}bool is_contiguous()const{return contiguous;}
 int64_t dim()const{return shape.size();}int64_t size(int i)const{return shape.at(i);}const auto& sizes()const{return shape;}
 int64_t storage_offset()const{return offset;}struct Storage{size_t bytes;size_t nbytes()const{return bytes;}};Storage storage()const{return {bytes};}
 size_t nbytes()const{size_t n=type==kInt?4:2;for(auto d:shape)n*=d;return n;}
 void* data_ptr()const{return reinterpret_cast<void*>(address);}template<class T>T* data_ptr()const{return reinterpret_cast<T*>(const_cast<int32_t*>(values.data()));}
};}
struct SpmAllocator{static constexpr uint64_t SPM_USABLE=8387584;};
struct {bool initialized=true;bool is_initialized()const{return initialized;}uint32_t addr(int c,int o)const{assert(c==0);return 0x40000000+o;}}SPM_ALLOC;
enum class KernelId{QWEN3VL_QK_NORM_MROPE_D128,QWEN3VL_QK_NORM_MROPE_KV_INSERT_D128};
enum class GraphDdrRegisterRole{ModelWeight,SemanticInput,KeyCache,ValueCache};enum class GraphDdrRegisterAccess{Read,Write};enum class GraphDdrRegisterEncoding{DevAddrShift8LoHi};
struct Abi{GraphDdrRegisterRole role;GraphDdrRegisterAccess access;GraphDdrRegisterEncoding encoding;int lo,hi;};struct GraphDdrRegisterOperandSpec{Abi abi;at::Tensor tensor;};
int submits=0,stages=0,flushes=0,keeps=0,writes=0,wide=0,addresses=0;
std::vector<uint64_t> flushed;
uint64_t RpuGetDevAddr(void* ptr){++addresses;return reinterpret_cast<uint64_t>(ptr);}
struct Kernel_t{std::array<uint16_t,68> r{};std::array<uint16_t,48> mask{};
 void reset_regs(){r.fill(0);}void set_regs(int i,uint16_t v){r.at(i)=v;}
 void set_regs(int i,int16_t v){set_regs(i,uint16_t(v));}
 void set_regs(int i,uint32_t v){++wide;check(i%2==0);r.at(i)=uint16_t(v);r.at(i+1)=uint16_t(v>>16);}
 void set_regs(int i,int32_t v){set_regs(i,uint32_t(v));}
 uint32_t pair(int i)const{return r[i]|uint32_t(r[i+1])<<16;}
}kernel;
bool missing=false;Kernel_t* get_kernel(KernelId){return missing?nullptr:&kernel;}
#define GET_KERNEL(id) get_kernel(id)
struct Writer{std::vector<GraphDdrRegisterOperandSpec> operands;void write(Kernel_t& k){++writes;for(const auto& op:operands){auto a=op.tensor.address>>8;k.set_regs(op.abi.lo,uint16_t(a));k.set_regs(op.abi.hi,uint16_t(a>>16));}}};
struct RpuKernelGraph{bool deny=false;static auto& active(){static RpuKernelGraph g;return g;}
 Writer stage_kernel_ddr_registers(KernelId id,std::vector<GraphDdrRegisterOperandSpec> ops){++stages;check(!deny);
 const int count=id==KernelId::QWEN3VL_QK_NORM_MROPE_D128?3:5;assert(int(ops.size())==count);
 const int regs[]={6,8,12,24,26};const GraphDdrRegisterRole roles[]={GraphDdrRegisterRole::ModelWeight,GraphDdrRegisterRole::ModelWeight,GraphDdrRegisterRole::SemanticInput,GraphDdrRegisterRole::KeyCache,GraphDdrRegisterRole::ValueCache};
 for(int i=0;i<count;++i){assert(ops[i].abi.lo==regs[i]&&ops[i].abi.hi==regs[i]+1&&ops[i].abi.role==roles[i]);assert(ops[i].abi.access==(i<3?GraphDdrRegisterAccess::Read:GraphDdrRegisterAccess::Write)&&ops[i].abi.encoding==GraphDdrRegisterEncoding::DevAddrShift8LoHi);check(ops[i].tensor.address&&ops[i].tensor.address%256==0&&ops[i].tensor.address<(1ULL<<40));}return {ops};}
 void keep_alive(const at::Tensor&){++keeps;}};
void rpu_ddr_flush(void* ptr){++flushes;flushed.push_back(reinterpret_cast<uint64_t>(ptr));}
constexpr int SCM_REG_OFFSET=4096;
auto rpu_compute_mrope_strobe_masks(std::vector<int> sections,int d){assert(sections==std::vector<int>({24,20,20})&&d==128);std::array<std::array<uint16_t,16>,3> m{};for(int c=0;c<64;++c){int a=c<60?c%3:0;m[a][c/16]|=uint16_t(1u<<(c%16));}return m;}
void rpu_set_legacy_scm_u16_checked(Kernel_t* k,int i,uint16_t v,const char*){k->mask.at(i-4096)=v;}
struct Queue {bool broadcast=false;void set_broadcast_mode(bool v){broadcast=v;}
 void enqueu_kernel(Kernel_t&,std::vector<int> grid,std::vector<int> cores){assert(broadcast&&writes>submits&&wide==0);assert(grid==std::vector<int>({5,1,1})&&cores==std::vector<int>({0,1,2,3,4,5,6,7}));++submits;}}queue;
Queue* get_queue(int n){assert(n==8);queue.broadcast=false;return &queue;}
#define GET_QUEUE(n) get_queue(n)
__DECL__
__WRAPPER__
using Inputs=std::array<at::Tensor,3>;
Inputs good(){Inputs in;in[1].address+=0x100000;in[2].address+=0x200000;in[2].type=at::kInt;in[2].shape={1024,3};in[2].bytes=(1024+8)*12;return in;}
void launch(const Inputs& in,int64_t p=15,std::array<uint32_t,4> a={0x40001000,0x40002000,0x40003000,0x40004000},double eps=1e-6){
 __NAME__(a[0],a[1],a[2],a[3],in[0],in[1],in[2],p,eps);}
void rejects(std::function<void()> f,bool before_stage=true){int old=submits,st=stages,fl=flushes,ke=keeps;bool rejected=false;try{f();}catch(const std::runtime_error&){rejected=true;}assert(rejected&&submits==old&&flushes==fl&&keeps==ke);if(before_stage)assert(st==stages);}
using KvInputs=std::array<at::Tensor,5>;
KvInputs good_kv(){KvInputs in;auto old=good();for(int i=0;i<3;++i)in[i]=old[i];
 in[0].shape[0]=in[1].shape[0]=2048;in[0].bytes=in[1].bytes=2048*128;
 in[2].shape[0]=2056;in[2].bytes=2056*12;in[2].values.resize(2056*3);
 for(int i=3;i<5;++i){in[i].address+=uint64_t(i)*0x1000000;in[i].shape={1,128,1,8,8,16,16};in[i].bytes=in[i].nbytes();}return in;}
void launch_kv(const KvInputs& in,int64_t p=15,uint32_t v=0x40005000,double eps=1e-6){
 __KV_NAME__(0x40001000,0x40002000,v,0x40003000,0x40004000,in[0],in[1],in[2],in[3],in[4],p,eps);}
int main(){
 auto masks=rpu_compute_mrope_strobe_masks({24,20,20},128);std::array<uint16_t,68> a{};
 for(int step:{0,1,0}){auto in=good();int p=15+step;for(auto& t:in)t.address+=step*256;in[2].values[p*3]=3;in[2].values[p*3+1]=7;in[2].values[p*3+2]=11;
 auto before=in[2].values;launch(in,p);assert(in[2].values==before);assert(kernel.pair(0)==uint32_t(p)&&kernel.r[2]==128&&kernel.r[3]==4);
 assert(kernel.pair(4)==0x40001000&&kernel.pair(10)==0x40002000&&kernel.pair(14)==0x40003000&&kernel.pair(16)==0x40004000);
 for(auto pair:std::vector<std::pair<int,int>>{{0,6},{1,8},{2,12}})assert(kernel.pair(pair.second)==uint32_t(in[pair.first].address>>8));
 assert(kernel.r[18]==0x2000&&kernel.r[19]==0x11&&kernel.r[20]==5&&kernel.r[21]==8&&kernel.r[64]==5&&kernel.r[65]==1&&kernel.r[66]==1);
 for(int i=0;i<48;++i)assert(kernel.mask[i]==masks[i/16][i%16]);
 if(submits==1)a=kernel.r;else assert((kernel.r==a)==(step==0));}
 assert(flushes==6&&keeps==9&&writes==3);
 for(int i=0;i<3;++i){auto in=good();in[i].present=false;rejects([&]{launch(in);});in=good();in[i].contiguous=false;rejects([&]{launch(in);});in=good();in[i].device_kind=c10::DeviceType::CPU;rejects([&]{launch(in);});in=good();in[i].layout_kind=c10::Layout::Sparse;rejects([&]{launch(in);});in=good();in[i].type=at::kFloat;rejects([&]{launch(in);});
 for(uint64_t a:{0ULL,0x1200000001ULL,1ULL<<40}){in=good();in[i].address=a;rejects([&]{launch(in);},false);}}
 for(int64_t p:std::vector<int64_t>{-1,1024,INT64_MAX})rejects([&]{launch(good(),p);});
 for(int axis=0;axis<3;++axis)for(int val:{-1,1024}){auto in=good();in[2].values[15*3+axis]=val;rejects([&]{launch(in);});}
 {auto in=good();in[0].shape[0]=15;in[1].shape[0]=15;rejects([&]{launch(in);});}
 {auto in=good();in[1].shape[1]=32;rejects([&]{launch(in);});}
 {auto in=good();in[2].shape[1]=4;rejects([&]{launch(in);});}
 {auto in=good();launch(in,1023);in[2].bytes=1024*12;rejects([&]{launch(in,1023);});in[2].bytes+=20;launch(in,1023);in[2].offset=1;rejects([&]{launch(in,1023);});}
 {auto in=good();in[2].offset=-1;rejects([&]{launch(in);});}
 for(double eps:{0.,-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})rejects([&]{launch(good(),15,{0x40001000,0x40002000,0x40003000,0x40004000},eps);});
 const std::array<uint32_t,4> original{0x40001000,0x40002000,0x40003000,0x40004000};
 for(int i=0;i<4;++i){for(uint32_t addr:{0x3fffff00u,0x40001001u,0x40800000u}){auto a=original;a[i]=addr;rejects([&]{launch(good(),15,a);});}for(int j=0;j<i;++j){auto a=original;a[i]=a[j];rejects([&]{launch(good(),15,a);});}}
 SPM_ALLOC.initialized=false;rejects([&]{launch(good());});SPM_ALLOC.initialized=true;
 missing=true;rejects([&]{launch(good());});missing=false;RpuKernelGraph::active().deny=true;rejects([&]{launch(good());},false);RpuKernelGraph::active().deny=false;
 // New wrapper executes that same arithmetic setup, with two typed DDR writers.
 for(int p:{0,15,16,31,32,511,512,1535,1536,2047}){
  auto in=good_kv();for(auto& t:in)t.address+=uint64_t(p)*256;
  in[2].values[p*3]=p;in[2].values[p*3+1]=17;in[2].values[p*3+2]=3;auto before=in[2].values;
  int old_submits=submits,old_flush=flushes,old_keeps=keeps;size_t old_order=flushed.size();launch_kv(in,p);
  assert(submits==old_submits+1&&flushes==old_flush+6&&keeps==old_keeps+5&&in[2].values==before);
  assert(kernel.pair(0)==uint32_t(p)&&kernel.r[2]==128&&kernel.r[3]==4&&kernel.pair(22)==0x40005000);
  assert(kernel.pair(4)==0x40001000&&kernel.pair(10)==0x40002000&&kernel.pair(14)==0x40003000&&kernel.pair(16)==0x40004000);
  const int regs[]={6,8,12,24,26};for(int i=0;i<5;++i)assert(kernel.pair(regs[i])==uint32_t(in[i].address>>8));
  assert(kernel.r[18]==0x2000&&kernel.r[19]==0x11&&kernel.r[20]==5&&kernel.r[21]==8);
  for(int i=0;i<48;++i)assert(kernel.mask[i]==masks[i/16][i%16]);
  assert(std::vector<uint64_t>(flushed.begin()+old_order,flushed.end())==std::vector<uint64_t>({in[0].address,in[1].address,in[3].address,in[4].address,in[3].address,in[4].address}));
 }
 for(int i=0;i<5;++i){
  auto in=good_kv();in[i].present=false;rejects([&]{launch_kv(in);});
  in=good_kv();in[i].contiguous=false;rejects([&]{launch_kv(in);});
  in=good_kv();in[i].device_kind=c10::DeviceType::CPU;rejects([&]{launch_kv(in);});
  in=good_kv();in[i].layout_kind=c10::Layout::Sparse;rejects([&]{launch_kv(in);});
  in=good_kv();in[i].type=at::kFloat;rejects([&]{launch_kv(in);});
  for(uint64_t a:{0ULL,0x1200000001ULL,1ULL<<40}){in=good_kv();in[i].address=a;rejects([&]{launch_kv(in);},false);}
 }
 for(int i:{3,4}){
  auto in=good_kv();in[i].shape.pop_back();rejects([&]{launch_kv(in);});
  for(int d=0;d<7;++d){in=good_kv();in[i].shape[d]+=1;rejects([&]{launch_kv(in);});}
  in=good_kv();in[3].shape[1]=in[4].shape[1]=0;rejects([&]{launch_kv(in);});
  in=good_kv();in[3].shape[1]=in[4].shape[1]=uint64_t(UINT32_MAX)/16+1;rejects([&]{launch_kv(in);});
  in=good_kv();in[i].address=(1ULL<<40)-256;rejects([&]{launch_kv(in);},false);
  for(int ro=0;ro<3;++ro){in=good_kv();in[i].address=in[ro].address+256;rejects([&]{launch_kv(in);},false);}
 }
 for(uint64_t delta:{0ULL,256ULL,4194048ULL}){auto in=good_kv();in[4].address=in[3].address+delta;rejects([&]{launch_kv(in);},false);}
 {auto in=good_kv();in[3].shape[1]=in[4].shape[1]=1;launch_kv(in,15);rejects([&]{launch_kv(in,16);});}
 for(int64_t p:{-1,2048})rejects([&]{launch_kv(good_kv(),p);});
 for(int axis=0;axis<3;++axis)for(int val:{-1,2048}){auto in=good_kv();in[2].values[15*3+axis]=val;rejects([&]{launch_kv(in);});}
 {auto in=good_kv();in[2].bytes=2048*12;rejects([&]{launch_kv(in,2047);});}
 for(uint32_t v:{0x3fffff00u,0x40005001u,0x40800000u,0x40001100u,0x40002000u,0x40003000u,0x40004000u})rejects([&]{launch_kv(good_kv(),15,v);});
 for(double eps:{0.,-1.,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})rejects([&]{launch_kv(good_kv(),15,0x40005000,eps);});
 missing=true;rejects([&]{launch_kv(good_kv());});missing=false;
 SPM_ALLOC.initialized=false;rejects([&]{launch_kv(good_kv());});SPM_ALLOC.initialized=true;
 {int old=addresses;RpuKernelGraph::active().deny=true;rejects([&]{launch_kv(good_kv());},false);assert(addresses==old);RpuKernelGraph::active().deny=false;}
 bool odd=false;try{kernel.set_regs(13,1);}catch(const std::runtime_error&){odd=true;}assert(odd);
}
'''
    harness=harness.replace('__DECL__',decl).replace('__WRAPPER__',wrapper).replace('__NAME__',NAME).replace('__KV_NAME__',KV_NAME)
    cpp=tmp_path/'qk.cpp';cpp.write_text(harness);exe=tmp_path/'qk'
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror',str(cpp),'-o',str(exe)],check=True,capture_output=True,text=True)
    subprocess.run([str(exe)],check=True,capture_output=True,text=True)
