"""Run the whole actual cat with real ATen and host SDK/cache boundary doubles.

This does not execute an device kernel or establish physical SDK barrier/performance
coverage. Device labels, allocation, cache maintenance and the barrier are the
only substituted boundaries; validation and the complete native copy body run.
"""
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def _function(source, marker):
    start = source.index(marker)
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
                    lambda match: ' ' * len(match[0]), source)
    brace = masked.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        depth += (masked[end] == '{') - (masked[end] == '}')
        end += 1
    return source[start:end]


def test_portable_whole_cat_acquires_before_memcpy_and_publishes_fresh_output(tmp_path, monkeypatch):
    monkeypatch.setenv('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
    import torch

    profile = (ROOT / 'src/core/rpu_profile.h').read_text()
    source = (ROOT / 'src/ops/rpu_concat.cpp').read_text()
    candidate = _function(source, 'at::Tensor rpu_cat(')
    out_body = _function(source, 'at::Tensor& rpu_cat_out(')
    # Host boundary doubles work without initializing the SDK submodule. When
    # its normal source is present, use the actual owner definitions/prototypes
    # and complete Batch loop; never require a specific SDK archive or hash.
    sdk_root = ROOT / 'third_party/rhino-launch-kernel'
    sdk_types = r"""
    enum MemFlag { read_write };
    enum MemStride { kStride1B };
    enum MemType { cpu_ddr_memory };
    class Buffer_t {
    public:
      Buffer_t(size_t n,MemType,MemFlag,MemStride,uint64_t):cpu_ptr_(nullptr),size_(n){}
      virtual ~Buffer_t()=default;
      virtual void* get_cpu_ptr() const {return cpu_ptr_;}
      virtual size_t get_memory_size() const {return size_;}
    protected:void* cpu_ptr_;size_t size_;
    };
    class HostDDR_t:public Buffer_t {
    public:HostDDR_t(size_t,MemFlag,MemStride,uint64_t);~HostDDR_t();
    protected:class Impl;std::unique_ptr<Impl> pImpl;
    };
    HostDDR_t* RpuGetHostddr(void* ptr);
    void RpuDdrFlushBatch(void** addrs,size_t* sizes,size_t count);
    """
    sdk_batch = r"""
    void sdk_batch_actual(void** addrs,size_t* sizes,size_t count){
      if(count==0)return;
      for(size_t i=0;i<count;++i){
        if(addrs[i]==nullptr||sizes[i]==0)continue;
        const auto start=reinterpret_cast<uintptr_t>(addrs[i])&R1_CACHE_LINE_MASK;
        const auto end=(reinterpret_cast<uintptr_t>(addrs[i])+sizes[i]+63)&R1_CACHE_LINE_MASK;
        for(uintptr_t p=start;p<end;p+=64)maintain(reinterpret_cast<void*>(p),64);
      }
      ++sdk_completions;
    }
    """
    files = [sdk_root / name for name in ('export_include/rhino_launch_buffer.h',
             'export_include/rhino_launch_def.h', 'src/r1_buffer.cc')]
    if all(path.is_file() for path in files):
        header, definitions, implementation = [path.read_text() for path in files]
        sdk_types = '\n'.join(_function(definitions, '\nenum ' + name + ' {') + ';'
                              for name in ('MemFlag', 'MemStride', 'MemType'))
        sdk_types += '\n' + _function(header, 'class Buffer_t {') + ';'
        sdk_types += '\n' + _function(header, 'class RHINO_EXPORT_API HostDDR_t :') + ';'
        sdk_types += '\n' + '\n'.join(line for line in header.splitlines() if
            'HostDDR_t *RpuGetHostddr(void *ptr);' in line or
            'void RpuDdrFlushBatch(void **addrs, size_t *sizes, size_t count);' in line)
        sdk_batch = _function(implementation, 'void rhino_lkn::RpuDdrFlushBatch(')
        sdk_batch = sdk_batch.replace('void rhino_lkn::RpuDdrFlushBatch(', 'void sdk_batch_actual(')
        sdk_batch = sdk_batch.replace('#if defined(__aarch64__)', '#if 1')
        instruction = '__asm__ volatile("dc civac, %0" : : "r"(p) : "memory");'
        completion = '__asm__ volatile("dsb ish" ::: "memory");'
        assert sdk_batch.count(instruction) == sdk_batch.count(completion) == 1
        sdk_batch = sdk_batch.replace(instruction, 'maintain(reinterpret_cast<void*>(p), 64);')
        sdk_batch = sdk_batch.replace(completion, '++sdk_completions;')
    macros = profile[profile.index('#define rpu_ddr_flush_force_sized('):].split('\n\n', 1)[0]

    def host_boundary(body, name):
        body = body.replace('at::Tensor rpu_cat(', 'at::Tensor ' + name + '(')
        for variable in ('t', 'inp'):
            body = body.replace(variable + '.device().type() == c10::DeviceType::PrivateUse1',
                                'rpu_label(' + variable + ')')
            body = body.replace(variable + '.device().type() != c10::DeviceType::PrivateUse1',
                                '!rpu_label(' + variable + ')')
        body = body.replace('target_device = t.device();', 'target_device = labelled_device(t);')
        body = body.replace('t.contiguous()', 'labelled_contiguous(t)')
        body = body.replace('at::empty(output_shape, first.options().device(target_device))',
                            'labelled_empty(output_shape, first.options().device(target_device))')
        body = body.replace('at::empty({0}, tensors.front().options())',
                            'labelled_empty_all(tensors.front())')
        body = body.replace('#if defined(__aarch64__)', '#if 1')
        body = body.replace('__asm__ volatile("dsb sy" ::: "memory");', 'system_barrier();')
        body = body.replace('memcpy(dst, src, copy_size);', 'tracked_memcpy(dst, src, copy_size);')
        return body

    cpp = r'''
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/intrusive_ptr.h>
#include <cassert>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#define RHINO_EXPORT_API
namespace rhino_lkn {
__SDK_TYPES__
class HostDDR_t::Impl {};
HostDDR_t::HostDDR_t(size_t n,MemFlag f,MemStride s,uint64_t a)
 :Buffer_t(n,cpu_ddr_memory,f,s,a){}
HostDDR_t::~HostDDR_t()=default;
}
bool g_rpu_ddr_flush_enabled=false,g_rpu_ddr_flush_force_enabled=true;
struct Owner:rhino_lkn::HostDDR_t{
 at::Tensor backing;std::vector<uint8_t>device,dirty;bool lookup=true;
 Owner(at::Tensor b):HostDDR_t(b.nbytes(),rhino_lkn::read_write,rhino_lkn::kStride1B,64),
  backing(std::move(b)),device(backing.nbytes(),0xa7),dirty(backing.nbytes(),0){cpu_ptr_=backing.data_ptr();}
 void metadata(uintptr_t base,size_t extent){cpu_ptr_=reinterpret_cast<void*>(base);size_=extent;}
 uintptr_t actual_base()const{return reinterpret_cast<uintptr_t>(backing.data_ptr());}
};
std::vector<std::unique_ptr<Owner>> owners;
std::unordered_map<const c10::TensorImpl*,
 c10::weak_intrusive_ptr<c10::TensorImpl,c10::UndefinedTensorImpl>> rpu_labels;
struct Range{void* ptr;size_t bytes;};
std::vector<Range>scalars;std::vector<std::vector<Range>>batches;
std::vector<std::string>timeline;at::Tensor forced_output;
int copied=0,system_completions=0,sdk_completions=0,owner_lookups=0;
bool rpu_label(const at::Tensor& t){auto i=rpu_labels.find(t.unsafeGetTensorImpl());
 return i!=rpu_labels.end()&&!i->second.expired();}
void label(const at::Tensor& t){rpu_labels.insert_or_assign(t.unsafeGetTensorImpl(),
 c10::weak_intrusive_ptr<c10::TensorImpl,c10::UndefinedTensorImpl>(t.getIntrusivePtr()));}
c10::Device labelled_device(const at::Tensor& t){return rpu_label(t)?c10::Device(c10::DeviceType::PrivateUse1):t.device();}
Owner& locate(void* ptr,size_t n,size_t& off){const auto p=reinterpret_cast<uintptr_t>(ptr);
 for(auto i=owners.rbegin();i!=owners.rend();++i){auto& o=**i;auto b=o.actual_base();
  if(p>=b&&p-b<=o.device.size()&&n<=o.device.size()-(p-b)){off=p-b;return o;}}
 TORCH_CHECK(false,"fixture lost actual allocation");}
at::Tensor bank(at::IntArrayRef shape,at::ScalarType dtype=at::kHalf,int64_t offset=32){
 int64_t n=1;for(auto d:shape)n*=d;
 auto backing=at::zeros({((n+offset+64+31)/32)*32},dtype);
 assert(reinterpret_cast<uintptr_t>(backing.data_ptr())%64==0);
 auto t=backing.narrow(0,offset,n).view(shape);
 owners.emplace_back(std::make_unique<Owner>(backing));label(t);return t;
}
void maintain(void* ptr,size_t bytes){size_t off;auto& o=locate(ptr,bytes,off);
 const uintptr_t lo=reinterpret_cast<uintptr_t>(ptr)&~uintptr_t(63);
 const uintptr_t hi=(reinterpret_cast<uintptr_t>(ptr)+bytes+63)&~uintptr_t(63);
 assert(lo>=o.actual_base()&&hi-o.actual_base()<=o.device.size());
 auto* host=static_cast<uint8_t*>(o.backing.data_ptr());
 for(size_t j=lo-o.actual_base();j<hi-o.actual_base();++j){
  if(o.dirty[j])o.device[j]=host[j];else host[j]=o.device[j];o.dirty[j]=0;
 }
}
void mark_cpu(const at::Tensor& t){size_t off;auto& o=locate(t.data_ptr(),t.nbytes(),off);
 const uintptr_t lo=reinterpret_cast<uintptr_t>(t.data_ptr())&~uintptr_t(63);
 const uintptr_t hi=(reinterpret_cast<uintptr_t>(t.data_ptr())+t.nbytes()+63)&~uintptr_t(63);
 std::fill(o.dirty.begin()+lo-o.actual_base(),o.dirty.begin()+hi-o.actual_base(),1);}
at::Tensor values(at::IntArrayRef shape,at::ScalarType dtype,int marker){
 int64_t n=1;for(auto d:shape)n*=d;
 auto v=(at::arange(n,at::kFloat)*0.125-marker).reshape(shape).to(dtype);
 if(dtype==at::kHalf){auto* p=static_cast<uint16_t*>(v.data_ptr());
  const uint16_t signed_bits[]={0,0x8000,1,0x8001,0x3c00,0xbc00,0x7bff,0xfbff};
  for(int64_t i=0;i<n&&i<8;++i)p[i]=signed_bits[i];}
 return v;
}
void device_write(const at::Tensor& t,const at::Tensor& v){
 t.copy_(v);size_t off;auto& o=locate(t.data_ptr(),0,off);
 std::memcpy(o.device.data(),o.backing.data_ptr(),o.device.size());
 std::fill(o.dirty.begin(),o.dirty.end(),0);t.fill_(37);
}
at::Tensor labelled_contiguous(const at::Tensor& t){
 if(!rpu_label(t)||t.is_contiguous())return t.contiguous();
 // A device contiguous kernel consumes DDR; only this device/allocator boundary
 // is doubled. ATen still performs the actual stride materialization.
 size_t off;auto& o=locate(t.data_ptr(),0,off);
 std::memcpy(o.backing.data_ptr(),o.device.data(),o.device.size());
 auto contiguous=t.contiguous();auto result=bank(contiguous.sizes(),contiguous.scalar_type());
 result.copy_(contiguous);mark_cpu(result);maintain(result.data_ptr(),result.nbytes());return result;
}
at::Tensor labelled_empty(at::IntArrayRef shape,const at::TensorOptions& options){
 if(options.device().type()!=c10::DeviceType::PrivateUse1){auto t=at::empty(shape,options);
  owners.emplace_back(std::make_unique<Owner>(t));return t;}
 auto t=forced_output.defined()?forced_output:bank(shape,options.dtype().toScalarType());
 assert(t.sizes()==shape);label(t);mark_cpu(t);return t;
}
at::Tensor labelled_empty_all(const at::Tensor& t){auto result=at::empty({0},t.options());
 if(rpu_label(t))label(result);return result;}
void system_barrier(){++system_completions;timeline.push_back("dsb-sy");}
void tracked_memcpy(void* dst,const void* src,size_t n){
 timeline.push_back("memcpy");++copied;std::memcpy(dst,src,n);
 // Model the actual CPU stores made by the unchanged native memcpy.
 size_t off;auto& o=locate(dst,n,off);
 std::fill(o.dirty.begin()+off,o.dirty.begin()+off+n,1);
}
constexpr size_t R1_CACHE_LINE_SIZE=64;
constexpr uintptr_t R1_CACHE_LINE_MASK=~uintptr_t(63);
__SDK_BATCH_BODY__
namespace rhino_lkn{
HostDDR_t* RpuGetHostddr(void* ptr){++owner_lookups;size_t off;auto& o=locate(ptr,0,off);return o.lookup?&o:nullptr;}
void RpuDdrFlush(void* ptr,size_t bytes){scalars.push_back({ptr,bytes});timeline.push_back("scalar");maintain(ptr,bytes);}
void RpuDdrFlushBatch(void** ptrs,size_t* sizes,size_t count){
 assert(g_rpu_ddr_flush_force_enabled&&count>1&&copied==0);
 std::vector<Range> packet;for(size_t i=0;i<count;++i)packet.push_back({ptrs[i],sizes[i]});
 sdk_batch_actual(ptrs,sizes,count);batches.push_back(packet);timeline.push_back("batch");
}
}
__MACROS__
__CANDIDATE__
__OUT__
void reset_calls(){scalars.clear();batches.clear();timeline.clear();copied=system_completions=sdk_completions=owner_lookups=0;}
void reset(){owners.clear();rpu_labels.clear();forced_output=at::Tensor();reset_calls();g_rpu_ddr_flush_force_enabled=true;}
bool bytes_equal(const at::Tensor& a,const at::Tensor& b){
 return a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type()&&
  (a.nbytes()==0||std::memcmp(a.contiguous().data_ptr(),b.contiguous().data_ptr(),a.nbytes())==0);}
void output_published(const at::Tensor& out,const at::Tensor& expected){
 assert(bytes_equal(out,expected));size_t off;auto& o=locate(out.data_ptr(),out.nbytes(),off);
 assert(std::memcmp(o.device.data()+off,expected.contiguous().data_ptr(),out.nbytes())==0);
 assert(scalars.back().ptr==out.data_ptr()&&scalars.back().bytes==out.nbytes());
 assert(timeline.back()=="scalar");
}
void exact_acquires(const std::vector<at::Tensor>& in,const at::Tensor& out,bool batch){
 std::vector<at::Tensor> rpu;for(const auto& t:in)if(t.numel()&&rpu_label(t))rpu.push_back(t);
 if(batch){assert(batches.size()==1&&system_completions==1&&sdk_completions==1&&scalars.size()==1);
  assert(batches[0].size()==rpu.size());
  for(size_t i=0;i<rpu.size();++i)assert(batches[0][i].ptr==rpu[i].data_ptr()&&batches[0][i].bytes==rpu[i].nbytes());
  assert(timeline[0]=="batch"&&timeline[1]=="dsb-sy"&&timeline[2]=="memcpy");
 }else{assert(batches.empty()&&system_completions==0&&sdk_completions==0&&scalars.size()==rpu.size()+1);
  for(size_t i=0;i<rpu.size();++i)assert(scalars[i].ptr==rpu[i].data_ptr()&&scalars[i].bytes==rpu[i].nbytes());
  for(size_t i=0;i<rpu.size();++i)assert(timeline[i]=="scalar");
 }
 for(auto& t:in)assert(t.data_ptr()!=out.data_ptr());
}
std::string failure(const std::vector<at::Tensor>& in,int64_t dim){
 reset_calls();try{rpu_cat(in,dim);}catch(const c10::Error& e){
  assert(scalars.empty()&&batches.empty()&&copied==0);return e.what_without_backtrace();}
 TORCH_CHECK(false,"fixture expected an error");}
int main(){
 c10::InferenceMode inference;int cases=0;
 // Multi-dimensional, negative dim, mixed CPU/device, signed FP16 bytes and saved
 // output retention. Device-written DDR differs from stale CPU before acquire.
 for(auto dtype:{at::kHalf,at::kFloat,at::kInt,at::kByte})for(int dim:{0,1,2,-1})for(bool cpu_first:{false,true}){
  reset();std::vector<at::Tensor> in,expected;std::vector<int64_t> shape{2,3,4};
  for(int i=0;i<4;++i){shape[dim<0?2:dim]=i+1;auto v=values(shape,dtype,i+3);expected.push_back(v);
   if(i==(cpu_first?0:3))in.push_back(v.clone());else{auto t=bank(shape,dtype);device_write(t,v);in.push_back(t);}}
  auto oracle=at::cat(expected,dim);auto out=rpu_cat(in,dim);output_published(out,oracle);exact_acquires(in,out,true);
  auto saved=out.clone();for(auto& t:in)if(rpu_label(t)){t.fill_(9);mark_cpu(t);}
  reset_calls();auto changed=rpu_cat(in,dim);assert(out.data_ptr()!=changed.data_ptr()&&bytes_equal(out,saved));++cases;
 }
 // Same address A/B/A through public device-written and raw dirty CPU aliases;
 // output tensors retained across requests must never change afterward.
 for(bool raw_alias:{false,true}){
  reset();std::vector<at::Tensor> in{bank({2,5}),bank({2,3}),bank({2,4})};
  std::vector<at::Tensor> saved,oracle_saved;
  for(int marker:{3,17,3}){
   std::vector<at::Tensor> expected;
   for(size_t i=0;i<in.size();++i){auto v=values(in[i].sizes(),at::kHalf,marker+i);expected.push_back(v);
    if(raw_alias){std::memcpy(in[i].data_ptr(),v.data_ptr(),v.nbytes());mark_cpu(in[i]);}
    else device_write(in[i],v);}
   reset_calls();auto oracle=at::cat(expected,1);auto out=rpu_cat(in,1);
   output_published(out,oracle);exact_acquires(in,out,true);
   for(size_t j=0;j<saved.size();++j){assert(bytes_equal(saved[j],oracle_saved[j]));assert(saved[j].data_ptr()!=out.data_ptr());}
   saved.push_back(out);oracle_saved.push_back(oracle);
  }
  assert(bytes_equal(saved.front(),saved.back())&&!bytes_equal(saved.front(),saved[1]));++cases;
 }
 // Fresh-input A/B/A drops caller inputs after each request while retaining
 // every output. Weak labels cannot turn a recycled TensorImpl into an device.
 reset();{
  std::vector<at::Tensor> saved,truth_saved;
  for(int marker:{4,19,4}){
   std::vector<at::Tensor> in,truth;
   for(int width:{3,4,5}){auto t=bank({2,width});auto v=values(t.sizes(),at::kHalf,marker+width);
    device_write(t,v);in.push_back(t);truth.push_back(v);}
   reset_calls();auto result=rpu_cat(in,1);auto oracle=at::cat(truth,1);
   output_published(result,oracle);exact_acquires(in,result,true);
   in.clear();truth.clear();
   for(size_t i=0;i<saved.size();++i)assert(bytes_equal(saved[i],truth_saved[i])&&saved[i].data_ptr()!=result.data_ptr());
   saved.push_back(result);truth_saved.push_back(oracle);
  }
  assert(bytes_equal(saved.front(),saved.back())&&!bytes_equal(saved.front(),saved[1]));++cases;
 }
 // Offset views sharing a real full backing owner are valid: owners need not
 // be disjoint, only the rounded input/output ranges must be disjoint.
 reset();auto slab=bank({224});std::vector<at::Tensor> views;
 for(int offset:{0,64,128}){auto t=slab.narrow(0,offset,32).view({2,16});label(t);views.push_back(t);}
 std::vector<at::Tensor> expected;
 for(size_t i=0;i<views.size();++i){auto v=values(views[i].sizes(),at::kHalf,i+5);views[i].copy_(v);expected.push_back(v);}
 {size_t off;auto& o=locate(slab.data_ptr(),0,off);std::memcpy(o.device.data(),o.backing.data_ptr(),o.device.size());slab.fill_(31);}
 auto out=rpu_cat(views,1);output_published(out,at::cat(expected,1));exact_acquires(views,out,true);++cases;
 // Overlapping input aliases are read-only and may share an owner/cache line;
 // preserve every logical span in the Batch instead of deduplicating it.
 reset();slab=bank({96});slab.copy_(values(slab.sizes(),at::kHalf,12));views.clear();expected.clear();
 for(int offset:{0,8,16}){auto t=slab.narrow(0,offset,32).view({2,16});label(t);views.push_back(t);expected.push_back(t.clone());}
 {size_t off;auto& o=locate(slab.data_ptr(),0,off);std::memcpy(o.device.data(),o.backing.data_ptr(),o.device.size());slab.fill_(31);}
 out=rpu_cat(views,1);output_published(out,at::cat(expected,1));exact_acquires(views,out,true);++cases;
 // Noncontiguous device inputs retain the original real ATen materialization.
 reset();std::vector<at::Tensor> strided,truth;
 for(int i=0;i<3;++i){auto base=bank({4,3});auto t=base.transpose(0,1);label(t);
  auto v=values(t.sizes(),at::kHalf,i+7);device_write(t,v);strided.push_back(t);truth.push_back(v);}
 out=rpu_cat(strided,1);output_published(out,at::cat(truth,1));assert(batches.size()==1);++cases;
 // Each failed SDK metadata proof falls back to the complete original scalar
 // order/ranges. The SDK double retains its actual storage for legacy scalar.
 for(int endpoint=0;endpoint<3;++endpoint)for(int fault=0;fault<9;++fault){
  reset();std::vector<at::Tensor> in,truth;
  for(int i=0;i<3;++i){auto t=bank({2,17},at::kHalf,33);auto v=values(t.sizes(),at::kHalf,i+8);
   device_write(t,v);in.push_back(t);truth.push_back(v);}
  size_t off;auto& o=locate(in[endpoint].data_ptr(),0,off);const auto p=reinterpret_cast<uintptr_t>(in[endpoint].data_ptr());
  switch(fault){
   case 0:o.lookup=false;break;
   case 1:o.metadata(0,o.device.size());break;
   case 2:o.metadata(o.actual_base(),0);break;
   case 3:o.metadata(o.actual_base(),off+in[endpoint].nbytes()-1);break;
   case 4:o.metadata(o.actual_base(),std::numeric_limits<size_t>::max());break;
   case 5:o.metadata(p,in[endpoint].nbytes()+64);break;
   case 6:o.metadata(o.actual_base(),off+in[endpoint].nbytes());break;
   case 7:o.metadata(std::numeric_limits<uintptr_t>::max()-16,64);break;
   case 8:o.metadata(p+in[endpoint].nbytes()+64,64);break;
  }
  out=rpu_cat(in,1);output_published(out,at::cat(truth,1));exact_acquires(in,out,false);++cases;
 }
 // One device input retains scalar acquisition; plain CPU input has no SDK owner.
 reset();auto one=bank({2,3});auto v=values(one.sizes(),at::kHalf,5);device_write(one,v);
 auto skipped=at::empty({0},at::kFloat);label(skipped);
 std::vector<at::Tensor> mixed{one,values({2,4},at::kHalf,6),skipped};
 out=rpu_cat(mixed,1);output_published(out,at::cat(std::vector<at::Tensor>{v,mixed[1]},1));exact_acquires(mixed,out,false);++cases;
 // Force OFF preserves the old no-maintenance behavior on coherent inputs.
 reset();std::vector<at::Tensor> coherent{bank({2,3}),bank({2,4}),bank({2,5})};
 for(auto& t:coherent)t.copy_(values(t.sizes(),at::kHalf,6));
 for(auto& o:owners)o->lookup=false;g_rpu_ddr_flush_force_enabled=false;
 out=rpu_cat(coherent,1);assert(bytes_equal(out,at::cat(coherent,1))&&scalars.empty()&&batches.empty()&&system_completions==0&&owner_lookups==0);++cases;
 // CPU-only is still the unchanged generic memcpy path, including a real
 // noncontiguous input and an empty tensor skipped by the original algorithm.
 reset();std::vector<at::Tensor> cpu{values({3,2},at::kFloat,2).transpose(0,1),values({2,4},at::kFloat,3),at::empty({0},at::kFloat)};
 // CPU allocation must also be known to the cache/copy boundary instrumentation.
 // No SDK maintenance is required, and the actual output bytes use ATen oracle.
 out=rpu_cat(cpu,1);assert(bytes_equal(out,at::cat(cpu,1))&&scalars.empty()&&batches.empty());++cases;
 // Rounded-only overlap with output falls back, while logical memcpy ranges
 // are disjoint. No undefined overlapping memcpy is introduced by the fixture.
 reset();auto shared=bank({64});auto a=shared.narrow(0,0,4);auto b=shared.narrow(0,8,4);
 label(a);label(b);a.copy_(values(a.sizes(),at::kHalf,3));b.copy_(values(b.sizes(),at::kHalf,4));
 truth={a.clone(),b.clone()};mark_cpu(a);mark_cpu(b);forced_output=shared.narrow(0,16,8);
 out=rpu_cat(std::vector<at::Tensor>{a,b},0);output_published(out,at::cat(truth,0));
 exact_acquires({a,b},out,false);forced_output=at::Tensor();++cases;
 // Original exceptions and their precedence occur before any acquire/memcpy.
 reset();auto x=bank({2,3});auto y=bank({2,4});auto z=bank({2,4},at::kFloat);
 std::vector<std::pair<std::vector<at::Tensor>,int64_t>> invalid{
  {{},0},{{at::ones({},at::kHalf)},0},{{x,y},3},{{x,bank({2,3,4},at::kFloat)},1},
  {{x,z},0},{{x,y},0}};
 const std::vector<std::string> expected_errors={"expected a non-empty list", "zero-dimensional",
  "out of range", "same number of dimensions", "same dtype", "sizes must match"};
 for(size_t i=0;i<invalid.size();++i){auto& item=invalid[i];
  assert(failure(item.first,item.second).find(expected_errors[i])!=std::string::npos);++cases;}
 reset();auto empty=at::empty({0},at::kHalf);out=rpu_cat(std::vector<at::Tensor>{empty,empty},-19);
 assert(out.numel()==0&&scalars.empty()&&batches.empty()&&copied==0);++cases;
 // The unchanged out wrapper executes the whole actual cat; result resize/copy
 // still uses real ATen. It does not bypass acquisition or output publication.
 reset();auto p=bank({2,3}),q=bank({2,4});auto pv=values(p.sizes(),at::kHalf,4),qv=values(q.sizes(),at::kHalf,5);
 device_write(p,pv);device_write(q,qv);auto destination=at::empty({1},at::kHalf);
 auto& returned=rpu_cat_out(std::vector<at::Tensor>{p,q},1,destination);
 assert(&returned==&destination&&bytes_equal(destination,at::cat(std::vector<at::Tensor>{pv,qv},1)));++cases;
 std::cout<<"actual-whole-cat-cases="<<cases<<"\n";
}
'''
    cpp = cpp.replace('__SDK_TYPES__', sdk_types).replace('__MACROS__', macros)
    cpp = cpp.replace('__SDK_BATCH_BODY__', sdk_batch)
    cpp = cpp.replace('__CANDIDATE__', host_boundary(candidate, 'rpu_cat'))
    cpp = cpp.replace('__OUT__', out_body)
    compiler = shutil.which('c++')
    assert compiler, 'actual cat contract requires a C++ compiler'
    path, binary = tmp_path / 'cat.cpp', tmp_path / 'cat'
    path.write_text(cpp)
    torch_root = Path(torch.__file__).resolve().parent
    command = [compiler, '-std=c++17', '-O0', '-D_GLIBCXX_USE_CXX11_ABI=' +
               str(int(torch.compiled_with_cxx11_abi())), '-I', str(torch_root / 'include'),
               '-I', str(torch_root / 'include/torch/csrc/api/include'), str(path),
               '-L', str(torch_root / 'lib'), '-Wl,-rpath,' + str(torch_root / 'lib'),
               '-ltorch_cpu', '-lc10', '-o', str(binary)]
    built = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, built.stdout + built.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'actual-whole-cat-cases=' in result.stdout
