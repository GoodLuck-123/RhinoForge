"""Fake dispatch and public freshness schema for multi-part device composition."""
import ast
from pathlib import Path
import re

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

ROOT = Path(__file__).resolve().parents[1]


def test_parts_fake_dispatch_is_shape_only_and_rejects_real_inputs():
    source = ast.parse((ROOT / "python/rpu_backend/graph/fake_dispatch.py").read_text())
    functions = []
    for node in source.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_has_fake", "_require_fake", "_meta_assemble_inputs_embeds_parts"
        }:
            node.decorator_list = []
            functions.append(node)
    namespace = {"torch": torch, "FakeTensor": FakeTensor}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "fake_dispatch.py", "exec"), namespace)
    call = namespace["_meta_assemble_inputs_embeds_parts"]
    vocab = torch.empty(320, 2048, dtype=torch.float16)
    ids = torch.empty(480, dtype=torch.int32)
    part = torch.empty(144, 2048, dtype=torch.float16)
    runs = torch.tensor([[5, 144]])
    with pytest.raises(RuntimeError, match="non-Fake"):
        call(vocab, ids, [part], runs)
    with FakeTensorMode() as mode:
        fake_vocab, fake_ids, fake_part, fake_runs = map(mode.from_tensor, (vocab, ids, part, runs))
        first = call(fake_vocab, fake_ids, [fake_part], fake_runs)
        second = call(fake_vocab, fake_ids, [fake_part], fake_runs)
        assert isinstance(first, FakeTensor) and first.shape == (480, 2048)
        assert first.dtype == torch.float16 and first.device == fake_vocab.device
        assert first is not second and first is not fake_vocab and first is not fake_part
        fake_prefix = mode.from_tensor(torch.empty(480, 2048, dtype=torch.float16))
        copied = call(fake_prefix, fake_ids, [fake_part], fake_runs, True)
        assert copied.shape == (480, 2048) and copied is not fake_prefix
        with pytest.raises(RuntimeError, match="prefix rows"):
            call(fake_vocab, fake_ids, [fake_part], fake_runs, True)


def test_parts_schema_has_no_mutable_or_returned_alias():
    source = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    match = re.search(r'm\.def\("(assemble_inputs_embeds_parts[^"\n]+)"', source)
    assert match is not None
    schema = torch._C.parse_schema(match[1])
    assert [argument.name for argument in schema.arguments] == ["vocab", "ids", "merged_parts", "runs", "prefix_is_embeddings"]
    assert schema.arguments[-1].default_value is False
    assert all(argument.alias_info is None for argument in schema.arguments)
    assert len(schema.returns) == 1 and schema.returns[0].alias_info is None


def test_preassembled_parts_single_batch_uses_real_DMA_guards_and_recovers(tmp_path):
    from test_gr00t_native_position_cache import test_backbone_parts_actual_native_dma_publication_and_spm_admission
    test_backbone_parts_actual_native_dma_publication_and_spm_admission(tmp_path, _extended_checks=True)


def _run_preassembled_parts_native_fixture(path, cpp, *, extended_checks=False):
    """Use GR's byte-store fixture with actual checked DMA and sync submission.

    The existing GR node keeps its geometry/flags/admission checks. The extended
    node adds SDK-stage failures, owner rejection, recovery and actual Hy runs.
    """
    from test_siglip_attention_dry_live_contract import _function, _run_native

    memcpy = (ROOT / "src/ops/rpu_memcpy.cpp").read_text()
    helper = _function(memcpy, "void rpu_launch_assembled_prefix_parts_dma_immediate(")
    submit = _function(memcpy, "static void submit_immediate_batch(")
    cache = (ROOT / "src/core/rpu_kernel_cache.h").read_text()
    checked = "\n".join(_function(cache, marker) for marker in (
        "inline void rpu_add_dma_checked(::rhino_lkn::Queue_t& q,\n                                const RpuDmaEndpoint& src,",
        "inline void rpu_add_dma_checked(::rhino_lkn::Queue_t& q,\n                                uint64_t src_addr,"))
    assert cpp.count("bool device_dirty=false;") == 1
    cpp = cpp.replace("bool device_dirty=false;", "bool device_dirty=false,registered=true;")
    # Preserve the existing allocation/device-label doubles, but use actual
    # ATen backing and byte-offset views for every positive caller and result.
    # Malformed metadata fields remain explicit admission controls.
    cpp = cpp.replace('#include <algorithm>', '#include <ATen/ATen.h>\n#include <ATen/Parallel.h>\n#include <algorithm>')
    cpp = cpp.replace('namespace c10 {using Half=uint16_t;}\n', '')
    cpp = cpp.replace(' size_t bytes;std::shared_ptr<uint8_t>cpu;', ' at::Tensor backing;size_t bytes;std::shared_ptr<uint8_t>cpu;')
    cpp = cpp.replace('assert(cpu);std::memset(cpu.get(),0,n);++allocations;',
                      'assert(cpu);std::memset(cpu.get(),0,n);backing=at::from_blob(cpu.get(),{int64_t(n)},at::kByte);++allocations;')
    cpp = cpp.replace('int allocations=0,dma_count=0,submits=0,fail_submit=0;',
                      'int allocations=0,dma_count=0,submits=0,fail_submit=0,flush_calls=0;')
    cpp = cpp.replace('void RpuDdrFlush(void*p,size_t bytes){', 'void RpuDdrFlush(void*p,size_t bytes){++flush_calls;')
    cpp = cpp.replace('void RpuDdrFlush(void*p){', 'void RpuDdrFlush(void*p){++flush_calls;')
    cpp = cpp.replace('uint64_t RpuGetDevAddr(const void*p){return reinterpret_cast<uint64_t>(p);}',
                      'const void*bad_dma_pointer=nullptr;uint64_t bad_dma_address=0;\nuint64_t RpuGetDevAddr(const void*p){return p==bad_dma_pointer?bad_dma_address:reinterpret_cast<uint64_t>(p);}')
    begin = cpp.index('namespace at {\nenum ScalarType')
    end = cpp.index('std::vector<uint8_t>spm', begin)
    tensors = r'''
struct HostDevice{c10::DeviceType value=at::kPrivateUse1;c10::DeviceType type()const{return value;}bool is_cpu()const{return value==at::kCPU;}bool operator==(HostDevice o)const{return value==o.value;}};
struct HostOptions{at::ScalarType dtype;HostDevice device;};
struct HostTensor{
 std::shared_ptr<Memory>m;std::vector<int64_t>sizes;at::ScalarType dtype=at::kHalf;HostDevice dev;size_t offset=0;bool contiguous=true,grad=false;
 HostTensor()=default;
 HostTensor(std::vector<int64_t>s,at::ScalarType d=at::kHalf,HostDevice v={}):sizes(s),dtype(d),dev(v){
  size_t count=1;for(auto x:s)count*=x;m=std::make_shared<Memory>(count*c10::elementSize(d));memories.push_back(m);
 }
 at::Tensor value()const{
  size_t count=1;for(auto s:sizes)count*=s;
  return m->backing.narrow(0,offset,count*c10::elementSize(dtype)).view(dtype).view(sizes);
 }
 int dim()const{return sizes.size();}int64_t size(int n)const{return sizes.at(n);}at::ScalarType scalar_type()const{return dtype;}
 HostDevice device()const{return dev;}bool is_contiguous()const{
  size_t bytes=c10::elementSize(dtype);for(auto s:sizes){if(uint64_t(s)>std::numeric_limits<size_t>::max()/bytes)return contiguous;bytes*=s;}
  return contiguous&&(bytes>m->bytes-offset||value().is_contiguous());
 }bool requires_grad()const{return grad;}
 void*data_ptr()const{return m->backing.data_ptr<uint8_t>()+offset;}template<class T>T*data_ptr()const{return reinterpret_cast<T*>(data_ptr());}
 size_t nbytes()const{size_t n=c10::elementSize(dtype);for(auto s:sizes)n*=s;return n<=m->bytes-offset?value().nbytes():n;}HostOptions options()const{return{dtype,dev};}
};
using HostTensorList=std::vector<HostTensor>;
namespace at{HostTensor empty(std::initializer_list<int64_t>s,HostOptions o){return HostTensor(s,o.dtype,o.device);}}
'''
    cpp = cpp[:begin] + cpp[end:]
    cpp = cpp.replace('at::TensorList', 'HostTensorList').replace('at::Tensor', 'HostTensor')
    # Reinsert real ATen types after adapting only the production functions and
    # original metadata fixture; the backing itself is never replaced.
    cpp = cpp.replace(' HostTensor backing;', ' at::Tensor backing;')
    cpp = cpp.replace('std::vector<uint8_t>spm', tensors + '\nstd::vector<uint8_t>spm', 1)
    cpp = cpp.replace('catch(const std::runtime_error&)', 'catch(const std::exception&)')
    cpp = cpp.replace("#include <stdexcept>", "#include <stdexcept>\n#include <string>")
    old_queue = 'struct Queue_t{std::vector<Node>nodes;void add_barrier(uint8_t,uint8_t){}};'
    queue = r'''
bool scratch_live();
size_t live_scratch_bytes=0;
int queue_clears=0,builds=0,enqueues=0,discards=0,adds=0;
std::string fail_stage;
int fail_add=0;
struct Buffer{uint64_t base;std::shared_ptr<Memory>memory;};
void execute_node(const Node&n){
 if(n.dst>=SPM_BASE&&n.dst-SPM_BASE<spm.size()){auto[m,o]=lookup(n.src);assert(n.bytes<=m->bytes-o&&n.bytes<=spm.size()-(n.dst-SPM_BASE));std::memcpy(spm.data()+n.dst-SPM_BASE,m->device.data()+o,n.bytes);}
 else{assert(n.src>=SPM_BASE&&n.bytes<=spm.size()-(n.src-SPM_BASE));auto[m,o]=lookup(n.dst);assert(n.bytes<=m->bytes-o);std::memcpy(m->device.data()+o,spm.data()+n.src-SPM_BASE,n.bytes);m->device_dirty=true;}
}
struct Queue_t{
 std::vector<Node>nodes;std::vector<std::shared_ptr<Memory>>leases;bool built=false,inflight=false;
 void add_barrier(uint8_t,uint8_t){}
 uint32_t add_dma_kernel(const Buffer&s,size_t so,const Buffer&d,size_t offset,size_t bytes,int channel){
  assert(channel==0);++adds;if(fail_stage=="add"&&adds==fail_add)return 11;
  assert(bytes>0&&bytes%16==0&&bytes<=(8ULL<<20));
  for(auto address:{s.base+so,d.base+offset})if(address>=SPM_BASE&&address-SPM_BASE<spm.size())assert(address-SPM_BASE<=live_scratch_bytes&&bytes<=live_scratch_bytes-(address-SPM_BASE));
  nodes.push_back({s.base+so,d.base+offset,bytes});leases.push_back(s.memory);leases.push_back(d.memory);++dma_count;return 0;
 }
 uint32_t build_batch(){++builds;assert(!built&&!inflight);if(fail_stage=="build")return 12;built=true;return 0;}
 uint32_t enqueu_batch(bool wait_finish){
  assert(wait_finish&&built&&!inflight);++enqueues;++submits;
  if(fail_stage=="enqueue"||fail_submit==submits){inflight=true;return 13;}
  for(auto&n:nodes)execute_node(n);return 0;
 }
 uint32_t discard_completed_batch(){
  ++discards;assert(!inflight);if(fail_stage=="discard")return 14;nodes.clear();leases.clear();built=false;return 0;
 }
 void destroy_failed(){
  assert(scratch_live());if(inflight){for(auto&n:nodes)execute_node(n);inflight=false;}
  nodes.clear();leases.clear();built=false;
 }
};
namespace rhino_lkn{using Queue_t=::Queue_t;}
struct RpuDeviceDmaEndpointRequest{uint64_t device_addr;size_t bytes;const char*where;};
struct RpuDmaEndpoint{
 std::shared_ptr<Buffer>owner;size_t offset=0;
 explicit operator bool()const{return bool(owner);}
};
std::vector<std::shared_ptr<Buffer>>rpu_resolve_device_dma_endpoints(const RpuDeviceDmaEndpointRequest*requests,size_t count,RpuDmaEndpoint*endpoints){
 std::vector<std::shared_ptr<Buffer>>lease;
 for(size_t i=0;i<count;++i){auto&r=requests[i];std::shared_ptr<Buffer>b;
  if(r.device_addr>=SPM_BASE&&r.device_addr-SPM_BASE<spm.size()){
   TORCH_CHECK(scratch_live()&&r.device_addr-SPM_BASE<=live_scratch_bytes&&r.bytes<=live_scratch_bytes-(r.device_addr-SPM_BASE));b=std::make_shared<Buffer>(Buffer{SPM_BASE,{}});
  }else{
   auto[m,o]=lookup(r.device_addr);TORCH_CHECK(m->registered&&r.bytes<=m->bytes-o);
   auto owned=std::find_if(memories.begin(),memories.end(),[&](auto&x){return x.get()==m;});assert(owned!=memories.end());
   b=std::make_shared<Buffer>(Buffer{reinterpret_cast<uint64_t>(m->cpu.get()),*owned});
  }
  endpoints[i]={b,size_t(r.device_addr-b->base)};lease.push_back(b);
 }return lease;
}
'''
    # The earlier fixture places the SPM constant after Queue_t; this queue
    # must reference it while executing its SDK transport delegate.
    cpp = cpp.replace('struct Node{uint64_t src,dst;size_t bytes;};',
                      'constexpr uint64_t SPM_BASE=0x10000000;\nstruct Node{uint64_t src,dst;size_t bytes;};')
    cpp = cpp.replace('constexpr uint64_t SPM_BASE=0x10000000;\nuint64_t spm_unified', 'uint64_t spm_unified')
    assert cpp.count(old_queue) == 1
    cpp = cpp.replace(old_queue, queue)
    old_cache = 'struct QueueCache{Queue_t queue;static QueueCache&instance(){static QueueCache q;return q;}Queue_t*get(uint8_t n){assert(n==1);return &queue;}};'
    new_cache = 'struct QueueCache{Queue_t queue;static QueueCache&instance(){static QueueCache q;return q;}Queue_t*get(uint8_t n){assert(n==1);return &queue;}void clear(){++queue_clears;queue.destroy_failed();}};'
    assert cpp.count(old_cache) == 1
    cpp = cpp.replace(old_cache, new_cache)
    old_checked = _function(cpp, 'void rpu_add_dma_checked(Queue_t&q,')
    cpp = cpp.replace(old_checked, checked)
    old_submit = _function(cpp, 'void submit_immediate_batch(Queue_t*q,')
    cpp = cpp.replace(old_submit, submit)
    cpp = cpp.replace('assert(r.bytes==921600&&size_t(r.bytes)<=available);',
                      'assert(r.bytes>0&&size_t(r.bytes)<=SPM_PLANNING_BUDGET&&size_t((r.bytes+255)/256*256)<=available);')
    cpp = cpp.replace('bool SpmAllocator::scratch=false;', 'bool scratch_live(){return SpmAllocator::scratch;}\nbool SpmAllocator::scratch=false;')
    cpp = cpp.replace('scratch=true;return{0};', 'scratch=true;live_scratch_bytes=r.bytes;return{0};')
    cpp = cpp.replace('scratch=false;++releases;', 'scratch=false;live_scratch_bytes=0;++releases;')
    assembler_marker = 'at::Tensor rpu_assemble_inputs_embeds_parts('
    assembler_marker = assembler_marker.replace('at::Tensor', 'HostTensor')
    helper = helper.replace('at::TensorList', 'HostTensorList').replace('at::Tensor', 'HostTensor')
    cpp = cpp.replace(assembler_marker, helper + '\n' + assembler_marker, 1)
    cpp = cpp.replace('auto output=invoke(prefix,ids,parts,runs);\n    assert(output.m',
                      'int b=builds,e=enqueues,d=discards,a=adds;auto output=invoke(prefix,ids,parts,runs);assert(builds==b+1&&enqueues==e+1&&discards==d+1&&adds==a+5);\n    assert(output.m')
    old_fail = ''' fail_submit=submits+2;int released=SpmAllocator::releases;bool failed=false;
 try{invoke(prefix,ids,parts,runs);}catch(const std::exception&){failed=true;}
 assert(failed&&!SpmAllocator::scratch&&SpmAllocator::releases==released+1);
 fail_submit=0;auto output=invoke(prefix,ids,parts,runs);assert(output.m->bytes==BYTES&&!SpmAllocator::scratch);'''
    cases = r'''
 // SDK failures at every stage preserve scratch until queue destruction, do
 // not touch a separately owned Graph queue, and leave the next batch usable.
 Queue_t graph_owned;graph_owned.nodes.push_back({123,456,16});
 SpmAllocator::available=65*H*2;
 for(const char*stage:{"add","build","enqueue","discard"}){
  fail_stage=stage;fail_add=adds+2;int released=SpmAllocator::releases,c=queue_clears;bool failed=false;
  try{invoke(prefix,ids,parts,runs);}catch(const std::runtime_error&){failed=true;}
  assert(failed&&!SpmAllocator::scratch&&SpmAllocator::releases==released+1&&queue_clears==c+1);
  assert(QueueCache::instance().queue.nodes.empty()&&!QueueCache::instance().queue.inflight&&!QueueCache::instance().queue.built);
  assert(graph_owned.nodes.size()==1&&graph_owned.nodes[0].src==123);
  fail_stage.clear();int b=builds,e=enqueues,d=discards;auto good=invoke(prefix,ids,parts,runs);
  assert(good.m->bytes==BYTES&&builds==b+1&&enqueues==e+1&&discards==d+1&&!SpmAllocator::scratch);
 }
 SpmAllocator::available=SpmAllocator::SPM_PLANNING_BUDGET;
 // Full managed owner preflight rejects even a late malformed part before
 // publication, append/build/execute or mutation of the shared queue.
 for(int which:{0,1}){
  auto bad_prefix=prefix;auto bad_parts=parts;
  if(which==0){bad_prefix.m=std::make_shared<Memory>(256);memories.push_back(bad_prefix.m);}
  else bad_parts[1].m->registered=false;
  g_rpu_ddr_flush_enabled=true;int b=builds,e=enqueues,c=queue_clears,a=adds,f=flush_calls;bool failed=false;
  try{invoke(bad_prefix,ids,bad_parts,runs);}catch(const std::runtime_error&){failed=true;}
  assert(failed&&builds==b&&enqueues==e&&adds==a&&queue_clears==c&&flush_calls==f&&!SpmAllocator::scratch);
  features.m->registered=true;
 }
 // Aligned but overflowing device ranges fail complete preflight before any
 // publication/queue mutation; no partial batch is left behind for recovery.
 for(int which:{0,1}){
  bad_dma_pointer=which?parts.back().data_ptr():prefix.data_ptr();
  bad_dma_address=std::numeric_limits<uint64_t>::max()-(which?15:255);
  int a=adds,b=builds,e=enqueues,c=queue_clears,f=flush_calls;bool failed=false;
  try{invoke(prefix,ids,parts,runs);}catch(const std::exception&){failed=true;}
  assert(failed&&adds==a&&builds==b&&enqueues==e&&queue_clears==c&&flush_calls==f&&!SpmAllocator::scratch);
  bad_dma_pointer=nullptr;
 }
 // Actual Hy 2cam P160/g7 geometry: fourteen contiguous seven-row parts,
 // separated by split-token rows in each camera's 7x8 prefix grid.
 // It is one ordered 16-DMA batch instead of 16 sync SDK submissions.
 at::Tensor hy_prefix({160,H}),hy_ids({160},at::kInt),hy_runs({14,2},at::kLong,{at::kCPU}),hy_features({98,H});
 for(int row=0;row<160;++row)std::fill_n(hy_prefix.data_ptr<uint16_t>()+row*H,H,uint16_t(1000+row));
 for(int row=0;row<98;++row)std::fill_n(hy_features.data_ptr<uint16_t>()+row*H,H,uint16_t(200+row));
 rhino_lkn::RpuDdrFlush(hy_prefix.data_ptr());rhino_lkn::RpuDdrFlush(hy_features.data_ptr());
 at::TensorList hy_parts;for(int i=0;i<14;++i){auto part=hy_features;part.sizes={7,H};part.offset=i*7*H*2;hy_parts.push_back(part);hy_runs.data_ptr<int64_t>()[2*i]=3+(i/7)*58+(i%7)*8;hy_runs.data_ptr<int64_t>()[2*i+1]=7;}
 int b=builds,e=enqueues,d=discards,a=adds;auto hy=invoke(hy_prefix,hy_ids,hy_parts,hy_runs);
 assert(builds==b+1&&enqueues==e+1&&discards==d+1&&adds==a+16&&!SpmAllocator::scratch);
 for(int row=0;row<160;++row){uint16_t expected=1000+row;for(int i=0;i<14;++i){int start=3+(i/7)*58+(i%7)*8;if(row>=start&&row<start+7)expected=200+i*7+row-start;}for(int col=0;col<H;++col)assert(reinterpret_cast<uint16_t*>(hy.m->device.data())[row*H+col]==expected);}

 // The real four-bank GR caller: N900,12 parts, and one921600B scratch
 // whose row boundaries split image runs and cross component banks.
 auto packed_case=[&](int64_t bank_rows,size_t free_bytes,bool all_flags){
  const int64_t rows=4*bank_rows;
  HostTensor packed({rows,H}),packed_ids({rows},at::kInt),packed_features({4*3*L,H}),packed_runs({12,2},at::kLong,{at::kCPU});
  HostTensorList packed_parts;auto*r=packed_runs.data_ptr<int64_t>();
  for(int bank=0;bank<4;++bank)for(int camera=0;camera<3;++camera){int index=bank*3+camera;
   auto part=packed_features;part.sizes={L,H};part.offset=index*L*H*2;packed_parts.push_back(part);
   r[2*index]=bank*bank_rows+run[2*camera];r[2*index+1]=L;
   assert(part.value().storage().unsafeGetStorageImpl()==packed_features.value().storage().unsafeGetStorageImpl());
  }
  SpmAllocator::available=free_bytes;
  for(bool normal:{false,true})for(bool force:{false,true}){
   if(!all_flags&&(!normal||!force))continue;
   g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=force;
   std::fill(packed.m->device.begin(),packed.m->device.end(),0);std::fill(packed_features.m->device.begin(),packed_features.m->device.end(),0);
   auto prefix_version=packed.value()._version(),parts_version=packed_features.value()._version();
   std::vector<HostTensor>retained;std::vector<std::vector<uint8_t>>saved;
   for(uint16_t value:{uint16_t(1),uint16_t(7),uint16_t(1)}){
    for(int bank=0;bank<4;++bank){std::fill_n(packed.data_ptr<uint16_t>()+bank*bank_rows*H,bank_rows*H,uint16_t(11+bank));
     for(int camera=0;camera<3;++camera)std::fill_n(packed_parts[bank*3+camera].data_ptr<uint16_t>(),L*H,uint16_t(value+bank*10+camera));}
    rpu_ddr_flush_force(packed.data_ptr());rpu_ddr_flush_force(packed_features.data_ptr());
    auto expected=packed.m->device;if(normal)std::memcpy(expected.data(),packed.data_ptr(),packed.nbytes());
    for(int index=0;index<12;++index){auto&part=packed_parts[index];const auto*src=normal?static_cast<uint8_t*>(part.data_ptr()):packed_features.m->device.data()+part.offset;
     std::memcpy(expected.data()+r[2*index]*H*2,src,L*H*2);}
    int b=builds,e=enqueues,d=discards;auto out=invoke(packed,packed_ids,packed_parts,packed_runs);
    assert(out.m->device==expected&&out.value().is_contiguous()&&out.value().sizes()==at::IntArrayRef({rows,H}));
    assert(builds==b+1&&enqueues==e+1&&discards==d+1&&!SpmAllocator::scratch);
    assert(packed.value()._version()==prefix_version&&packed_features.value()._version()==parts_version);
    assert(out.value().storage().unsafeGetStorageImpl()!=packed.value().storage().unsafeGetStorageImpl());
    for(size_t i=0;i<retained.size();++i)assert(retained[i].m->device==saved[i]);
    retained.push_back(out);saved.push_back(expected);
   }
  }
  if(bank_rows==512){
   // True preserves the pre-existing per-bank P512 envelope, while False
   // still rejects full-N beyond the original gather scratch planning budget.
   bool rejected=false;int a=allocations,d=dma_count,s=submits;
   try{rpu_assemble_inputs_embeds_parts(packed,packed_ids,packed_parts,packed_runs,false);}catch(const std::exception&){rejected=true;}
   assert(rejected&&allocations==a&&dma_count==d&&submits==s);
  }
  SpmAllocator::available=SpmAllocator::SPM_PLANNING_BUDGET;
 };
 packed_case(225,65*H*2+127,true);
 packed_case(225,H*2,true);
 packed_case(512,225*H*2,true);

 // H16 exposes256B allocation alignment with8-row chunks and a3-row tail.
 HostTensor narrow_prefix({43,16}),narrow_ids({43},at::kInt),narrow_features({19,16}),narrow_runs({1,2},at::kLong,{at::kCPU});
 std::fill_n(narrow_prefix.data_ptr<uint16_t>(),43*16,11);std::fill_n(narrow_features.data_ptr<uint16_t>(),19*16,7);
 narrow_runs.data_ptr<int64_t>()[0]=7;narrow_runs.data_ptr<int64_t>()[1]=19;
 g_rpu_ddr_flush_enabled=true;SpmAllocator::available=511;int nb=builds,ne=enqueues,nd=discards;
 auto narrow=invoke(narrow_prefix,narrow_ids,{narrow_features},narrow_runs);
 assert(builds==nb+1&&enqueues==ne+1&&discards==nd+1&&!SpmAllocator::scratch);
 for(int row=0;row<43;++row)for(int col=0;col<16;++col)assert(reinterpret_cast<uint16_t*>(narrow.m->device.data())[row*16+col]==uint16_t(row>=7&&row<26?7:11));
 SpmAllocator::available=SpmAllocator::SPM_PLANNING_BUDGET;
'''
    cases = cases.replace('catch(const std::runtime_error&)', 'catch(const std::exception&)')
    assert cpp.count(old_fail) == 1
    cases = cases.replace('at::TensorList', 'HostTensorList').replace('at::Tensor', 'HostTensor')
    cpp = cpp.replace(old_fail, cases if extended_checks else old_fail.replace("submits+2", "submits+1"))
    _run_native(path, cpp, torch)
