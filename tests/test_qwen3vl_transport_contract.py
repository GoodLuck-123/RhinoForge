"""Exercise the native host-visible transport bodies without a board.

The doubles retain byte storage, aliases and version counters. No device kernel,
cache coherency, or model numerical/performance claim is made by these tests.
"""
from pathlib import Path
import ast
import hashlib
import re
import shutil
import subprocess

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def transport_binary(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = (ROOT / "src/core/rpu_backend.cpp").read_text()
    def function(name):
        start = source.index(name)
        start = source.rfind("\n", 0, start) + 1
        return source[start:source.index("\n}", start) + 2]
    start = source.index("uint16_t fp16_order_key(")
    end = source.index("\nat::Tensor host_top1_impl(", start)
    functions = [source[start:end]]
    # Include a local helper if the native implementation uses one for physical
    # overlap; actual tensor pointers can alias across distinct Storage objects.
    before = source[:source.index("void rpu_qwen3vl_scatter_row_parts_unflushed_(")]
    for match in re.finditer(r"(?:static )?bool ([A-Za-z_][A-Za-z_0-9]*overlap[A-Za-z_0-9]*)\(", before):
        functions.append(function(match.group(1) + "("))
    for name in ("rpu_qwen3vl_scatter_row_parts_unflushed_", "rpu_qwen3vl_gather_embedding_from_rpu_unflushed_",
                 "rpu_qwen3vl_publish_inputs", "rpu_zero_contiguous_tensor_blocks_sized", "rpu_lm_head_logits_top2_rescore_cpu"):
        functions.append(function(name + "("))
    cpp = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <c10/util/Half.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while(false)
#define TORCH_INTERNAL_ASSERT(ok, ...) TORCH_CHECK(ok)
namespace c10 {
enum class DeviceType { CPU, PrivateUse1, CUDA };
}
namespace at {
enum ScalarType { kHalf, kLong, kFloat };
constexpr auto kCPU=c10::DeviceType::CPU;
struct Device {
    c10::DeviceType kind;
    bool is_cpu() const { return kind==c10::DeviceType::CPU; }
    c10::DeviceType type() const { return kind; }
};
struct TensorImpl { int version=0; void bump_version() { ++version; } };
struct Sizes: std::vector<int64_t> {
    using std::vector<int64_t>::vector;
    std::vector<int64_t> vec() const { return {begin(),end()}; }
};
struct TensorOptions {
    ScalarType scalar=kLong; c10::DeviceType where=kCPU;
    TensorOptions& dtype(ScalarType value) { scalar=value; return *this; }
    TensorOptions& device(c10::DeviceType value) { where=value; return *this; }
};
struct Tensor {
    std::shared_ptr<std::vector<uint8_t>> bytes;
    std::shared_ptr<TensorImpl> impl=std::make_shared<TensorImpl>();
    std::shared_ptr<int> storage=std::make_shared<int>(0);
    Sizes shape; size_t offset=0; ScalarType dtype=kHalf;
    c10::DeviceType where=c10::DeviceType::PrivateUse1;
    bool contiguous=true, grad=false, inference=false;
    Tensor()=default;
    Tensor(std::initializer_list<int64_t> dims, ScalarType scalar=kHalf,
           c10::DeviceType device=c10::DeviceType::PrivateUse1):shape(dims),dtype(scalar),where(device) {
        bytes=std::make_shared<std::vector<uint8_t>>(nbytes());
    }
    size_t element_size() const { return dtype==kHalf?2:dtype==kLong?8:4; }
    int64_t numel() const { int64_t n=1; for(auto d:shape)n*=d; return n; }
    size_t nbytes() const { return numel()*element_size(); }
    Device device() const { return {where}; }
    ScalarType scalar_type() const { return dtype; }
    int64_t dim() const { return shape.size(); }
    int64_t size(int64_t axis) const { return shape[axis<0?shape.size()+axis:axis]; }
    Sizes sizes() const { return shape; }
    bool is_contiguous() const { return contiguous; }
    bool requires_grad() const { return grad; }
    bool is_inference() const { return inference; }
    TensorImpl* unsafeGetTensorImpl() const { return impl.get(); }
    void* data_ptr() const { return bytes->data()+offset; }
    template<class T> T* data_ptr() const { return reinterpret_cast<T*>(data_ptr()); }
    Tensor reshape(const std::vector<int64_t>& dims) const { Tensor t=*this; t.shape.assign(dims.begin(),dims.end()); return t; }
    Tensor view(std::initializer_list<int64_t> dims, size_t byte_offset=0) const {
        Tensor t=*this; t.shape=dims; t.offset+=byte_offset; return t;
    }
};
using TensorList=std::vector<Tensor>;
Tensor empty(std::initializer_list<int64_t> dims, TensorOptions options) { return Tensor(dims,options.scalar,options.where); }
enum class MemOverlapStatus { No, Partial };
MemOverlapStatus get_overlap_status(const Tensor&a,const Tensor&b) {
    // ATen's storage check intentionally cannot see independent wrappers of
    // one pointer, reproducing the backend's CPU zero-copy storage boundary.
    if(a.storage!=b.storage)return MemOverlapStatus::No;
    auto x=reinterpret_cast<uintptr_t>(a.data_ptr()),y=reinterpret_cast<uintptr_t>(b.data_ptr());
    return a.nbytes()&&b.nbytes()&&(x<=y?y-x<a.nbytes():x-y<b.nbytes())?MemOverlapStatus::Partial:MemOverlapStatus::No;
}
void assert_no_overlap(const Tensor&a,const Tensor&b) { TORCH_CHECK(get_overlap_status(a,b)==MemOverlapStatus::No); }
}
int syncs=0,flushes=0,batches=0;
std::vector<std::pair<void*,size_t>> flushed_ranges;
std::vector<std::vector<uint8_t>> published_bytes;
std::function<void()> pending_graph_write;
bool require_zero_publication=false;
bool graph_aware_rpu_to_cpu_zerocopy_sync() {
    ++syncs;
    if(pending_graph_write) { auto write=std::move(pending_graph_write);write(); }
    return true;
}
bool g_rpu_ddr_flush_force_enabled=true;
bool g_rpu_ddr_flush_enabled=false;
namespace rhino_lkn { void RpuDdrFlushBatch(void** ptrs,size_t* sizes,size_t count) {
    ++batches; flushes+=count;
    for(size_t i=0;i<count;++i) {
        flushed_ranges.emplace_back(ptrs[i],sizes[i]);
        auto p=static_cast<uint8_t*>(ptrs[i]);
        if(require_zero_publication)assert(std::all_of(p,p+sizes[i],[](uint8_t x){return x==0;}));
        published_bytes.emplace_back(p,p+sizes[i]);
    }
} }
struct Publication { void* pointer; size_t bytes; std::vector<uint8_t> device_bytes; };
std::vector<Publication> publications;
namespace rhino_lkn {
void RpuDdrFlush(void* pointer,size_t bytes) {
    ++flushes;
    auto p=static_cast<uint8_t*>(pointer);
    publications.push_back({pointer,bytes,{p,p+bytes}});
}
}
#define rpu_ddr_flush_force_sized(ptr,nbytes) do { \
    if(g_rpu_ddr_flush_force_enabled) \
        rhino_lkn::RpuDdrFlush(const_cast<void*>(static_cast<const void*>(ptr)),nbytes); \
} while(false)
__FUNCTIONS__
using at::Tensor;
Tensor ids(std::initializer_list<int64_t> values) {
    Tensor t({static_cast<int64_t>(values.size())},at::kLong,c10::DeviceType::CPU);
    std::copy(values.begin(),values.end(),t.data_ptr<int64_t>());return t;
}
void fill(Tensor t,float value) { for(int64_t i=0;i<t.numel();++i)t.data_ptr<c10::Half>()[i]=c10::Half(value); }
void values(Tensor t,std::initializer_list<float> values) { int i=0;for(float v:values)t.data_ptr<c10::Half>()[i++]=c10::Half(v); }
std::vector<uint8_t> snapshot(Tensor t) { auto p=static_cast<uint8_t*>(t.data_ptr()); return {p,p+t.nbytes()}; }
template<class F> void rejected_unchanged(Tensor t,F run) {
    auto before=snapshot(t);int version=t.impl->version;bool caught=false;
    try { run(); } catch(const std::runtime_error&) { caught=true; }
    assert(caught && snapshot(t)==before && t.impl->version==version);
}
int main(int argc,char**argv) {
    std::string test=argv[1];
    if(test=="scatter") {
        Tensor dst({4,4});fill(dst,-2);Tensor a({1,4},at::kHalf,c10::DeviceType::CPU),b({1,4});fill(a,3);fill(b,7);
        rpu_qwen3vl_scatter_row_parts_unflushed_(dst,ids({0,2}),{a,b});
        assert(dst.impl->version==1 && static_cast<float>(dst.data_ptr<c10::Half>()[8])==7 && static_cast<float>(dst.data_ptr<c10::Half>()[4])==-2 && flushes==2);
        assert(publications[0].pointer==dst.data_ptr() && publications[0].bytes==8);
        assert(publications[1].pointer==dst.data_ptr<c10::Half>()+8 && publications[1].bytes==8);
        assert(publications[1].device_bytes==snapshot(b));
        auto same=dst.view({1,4});
        rpu_qwen3vl_scatter_row_parts_unflushed_(dst,ids({0}),{same});
        assert(dst.impl->version==2 && flushes==3);
        rejected_unchanged(dst,[&]{rpu_qwen3vl_scatter_row_parts_unflushed_(dst,ids({0,4}),{a,b});});
        rejected_unchanged(dst,[&]{rpu_qwen3vl_scatter_row_parts_unflushed_(dst,ids({2,0}),{a,b});});
        b.contiguous=false;
        rejected_unchanged(dst,[&]{rpu_qwen3vl_scatter_row_parts_unflushed_(dst,ids({0,2}),{a,b});});
        rejected_unchanged(dst,[&]{rpu_qwen3vl_scatter_row_parts_unflushed_(dst,ids({1}),{same});});
        same.storage=std::make_shared<int>(1);same.where=c10::DeviceType::CPU;
        rejected_unchanged(dst,[&]{rpu_qwen3vl_scatter_row_parts_unflushed_(dst,ids({1}),{same});});
        rpu_qwen3vl_scatter_row_parts_unflushed_(dst,ids({0}),{same});
        Tensor control_dst({2,8}),control_src({2,8});fill(control_src,2);
        auto alias=control_dst.view({2});alias.dtype=at::kLong;alias.where=c10::DeviceType::CPU;
        alias.storage=std::make_shared<int>(2);
        alias.data_ptr<int64_t>()[0]=0;alias.data_ptr<int64_t>()[1]=1;
        rejected_unchanged(control_dst,[&]{rpu_qwen3vl_scatter_row_parts_unflushed_(control_dst,alias,{control_src});});
    } else if(test=="gather") {
        // Execute the real native consumer for both admitted IDs devices. Raw
        // writes retain the address/version and must be observed on every call.
        for(auto device:{c10::DeviceType::CPU,c10::DeviceType::PrivateUse1}) {
            Tensor out({2,4}),weight({3,4});
            values(weight,{0,1,2,3,4,5,6,7,8,9,10,11});
            auto tokens=ids({0,1});tokens.where=device;
            auto alias=tokens;auto pointer=tokens.data_ptr();
            std::vector<unsigned char> saved;
            for(int value:{0,2,0}) {
                alias.data_ptr<int64_t>()[0]=value;
                rpu_qwen3vl_gather_embedding_from_rpu_unflushed_(out,weight,tokens,-1);
                assert(static_cast<float>(out.data_ptr<c10::Half>()[0])==value*4);
                assert(tokens.impl->version==0 && tokens.data_ptr()==pointer);
                if(saved.empty())saved=snapshot(out);
                else assert((snapshot(out)==saved)==(value==0));
            }
            assert(out.impl->version==3 && flushes==3 && batches==0);
            assert(publications.back().device_bytes==snapshot(out));
            flushes=0;publications.clear();
            tokens.where=c10::DeviceType::CUDA;
            rejected_unchanged(out,[&]{rpu_qwen3vl_gather_embedding_from_rpu_unflushed_(out,weight,tokens,-1);});
        }
        Tensor dst({3,4}),vocab({3,4});fill(dst,-2);values(vocab,{0,1,2,3,4,5,6,7,8,9,10,11});
        rpu_qwen3vl_gather_embedding_from_rpu_unflushed_(dst,vocab,ids({2,-1,0}),-1);
        assert(dst.impl->version==1 && static_cast<float>(dst.data_ptr<c10::Half>()[0])==8 && static_cast<float>(dst.data_ptr<c10::Half>()[4])==-2);
        assert(flushes==1 && publications[0].pointer==dst.data_ptr() && publications[0].bytes==dst.nbytes());
        assert(publications[0].device_bytes==snapshot(dst));
        rejected_unchanged(dst,[&]{rpu_qwen3vl_gather_embedding_from_rpu_unflushed_(dst,vocab,ids({0,1,3}),-1);});
        rejected_unchanged(dst,[&]{rpu_qwen3vl_gather_embedding_from_rpu_unflushed_(dst,dst,ids({0,1,2}),-1);});
        Tensor control_dst({2,8}),control_vocab({2,8});fill(control_vocab,0);
        Tensor alias=control_dst.view({2});alias.dtype=at::kLong;
        alias.data_ptr<int64_t>()[0]=0;alias.data_ptr<int64_t>()[1]=1;
        rejected_unchanged(control_dst,[&]{rpu_qwen3vl_gather_embedding_from_rpu_unflushed_(control_dst,control_vocab,alias,-1);});
        alias.storage=std::make_shared<int>(1);alias.where=c10::DeviceType::CPU;
        rejected_unchanged(control_dst,[&]{rpu_qwen3vl_gather_embedding_from_rpu_unflushed_(control_dst,control_vocab,alias,-1);});
        auto detached_vocab=dst;detached_vocab.storage=std::make_shared<int>(2);
        rejected_unchanged(dst,[&]{rpu_qwen3vl_gather_embedding_from_rpu_unflushed_(dst,detached_vocab,ids({0,1,2}),-1);});
    } else if(test=="publish") {
        Tensor owner({8,4}),other({2,4});fill(owner,3);fill(other,7);
        auto view=owner.view({2,4},16),alias=view;
        alias.storage=std::make_shared<int>(2); // distinct wrapper, same live bytes
        auto before=snapshot(owner);
        rpu_qwen3vl_publish_inputs({view,other,alias,Tensor({0,4})});
        assert(batches==1 && flushes==2 && snapshot(owner)==before && owner.impl->version==0);
        assert(flushed_ranges[0]==std::make_pair(view.data_ptr(),size_t(16)));
        assert(flushed_ranges[1]==std::make_pair(other.data_ptr(),other.nbytes()));
        for(int fault=0;fault<4;++fault) {
            auto bad=other;
            if(fault==0)bad.where=c10::DeviceType::CPU;
            if(fault==1)bad.dtype=at::kFloat;
            if(fault==2)bad.contiguous=false;
            if(fault==3)bad.grad=true;
            rejected_unchanged(owner,[&]{rpu_qwen3vl_publish_inputs({view,bad});});
            assert(batches==1); // invalid later entries cannot partially publish
        }
        g_rpu_ddr_flush_force_enabled=false;
        rpu_qwen3vl_publish_inputs({view});assert(batches==1);
        g_rpu_ddr_flush_force_enabled=true;
        rpu_qwen3vl_publish_inputs({Tensor({0,4})});assert(batches==1);
    } else if(test=="publication") {
        // A stable host-visible slot is consumed repeatedly by DMA. Its
        // published bytes must change with the token, not with its address.
        Tensor dst({1,4}),vocab({2,4});fill(dst,-2);values(vocab,{0,1,2,3,4,5,6,7});
        rpu_qwen3vl_gather_embedding_from_rpu_unflushed_(dst,vocab,ids({0}),-1);
        auto first=publications.back().device_bytes;
        rpu_qwen3vl_gather_embedding_from_rpu_unflushed_(dst,vocab,ids({1}),-1);
        assert(publications.back().pointer==dst.data_ptr());
        assert(publications.back().device_bytes!=first && publications.back().device_bytes==snapshot(dst));
        // No invalid caller may write or publish a partially checked request.
        auto count=publications.size();
        rejected_unchanged(dst,[&]{rpu_qwen3vl_gather_embedding_from_rpu_unflushed_(dst,vocab,ids({2}),-1);});
        assert(publications.size()==count);
        // The explicit boundary-flush opt-out is retained; the consumer will
        // still see the previously published row in this non-coherent double.
        g_rpu_ddr_flush_force_enabled=false;
        rpu_qwen3vl_gather_embedding_from_rpu_unflushed_(dst,vocab,ids({0}),-1);
        assert(publications.size()==count && publications.back().device_bytes!=snapshot(dst));
        Tensor src({1,4},at::kHalf,c10::DeviceType::CPU);fill(src,8);
        rpu_qwen3vl_scatter_row_parts_unflushed_(dst,ids({0}),{src});
        assert(publications.size()==count);
        g_rpu_ddr_flush_force_enabled=true;
        rpu_qwen3vl_scatter_row_parts_unflushed_(dst,ids({0}),{src});
        assert(publications.back().device_bytes==snapshot(dst));
        Tensor empty({0,4});count=publications.size();
        rpu_qwen3vl_gather_embedding_from_rpu_unflushed_(empty,vocab,ids({}),-1);
        assert(publications.size()==count);
        // Empty-width rows are valid transport inputs, with no DDR range to
        // publish even though the requested row indices are non-empty.
        Tensor empty_width({2,0}),empty_source({2,0},at::kHalf,c10::DeviceType::CPU);
        rpu_qwen3vl_scatter_row_parts_unflushed_(empty_width,ids({0,1}),{empty_source});
        assert(publications.size()==count);
    } else if(test=="zero") {
        Tensor a({1,3,4}),b({1,3,4});fill(a,3);fill(b,7);b.contiguous=false;
        rejected_unchanged(a,[&]{rpu_zero_contiguous_tensor_blocks_sized({a,b},1);});
        assert(b.impl->version==0);
        b.contiguous=true;b.inference=true;
        rpu_zero_contiguous_tensor_blocks_sized({a,b},1);
        assert(a.impl->version==1 && b.impl->version==0 && flushes==2);
        assert(static_cast<float>(a.data_ptr<c10::Half>()[0])==3 && static_cast<float>(a.data_ptr<c10::Half>()[4])==0);
        rejected_unchanged(a,[&]{rpu_zero_contiguous_tensor_blocks_sized({a,b},3);});
    } else if(test=="zero_reset") {
        require_zero_publication=true;
        for(bool normal:{false,true}) for(bool force:{false,true}) {
            g_rpu_ddr_flush_enabled=normal;
            g_rpu_ddr_flush_force_enabled=force;
            syncs=flushes=batches=0;flushed_ranges.clear();published_bytes.clear();
            Tensor a({1,1,128}),b({1,1,128}),empty({1,1,0});b.inference=true;
            auto alias=a.view({1,1,128});auto ptr=a.data_ptr();
            for(int round=0;round<3;++round) {
                // Raw CPU writes do not bump the original/shared versions.
                fill(a,round==1?9:3);fill(b,7);
                assert(alias.impl->version==round && empty.impl->version==2*round);
                pending_graph_write=[&]{fill(a,11);fill(b,13);};
                rpu_zero_contiguous_tensor_blocks_sized({a,b,empty,empty},0);
                assert(!pending_graph_write && syncs==round+1 && a.data_ptr()==ptr);
                assert(alias.impl->version==round+1 && b.impl->version==0 && empty.impl->version==2*(round+1));
                assert(std::all_of(a.bytes->begin(),a.bytes->end(),[](uint8_t x){return x==0;}));
                assert(std::all_of(b.bytes->begin(),b.bytes->end(),[](uint8_t x){return x==0;}));
                assert(batches==(force?round+1:0) && flushes==(force?2*(round+1):0));
            }
            if(force) {
                assert(flushed_ranges[0]==std::make_pair(a.data_ptr(),size_t(256)));
                assert(flushed_ranges[1]==std::make_pair(b.data_ptr(),size_t(256)));
                assert(published_bytes.size()==6);
            }
            // A bad later tensor must fail before Graph sync, version changes,
            // host writes or SDK publication, even with force disabled.
            for(int fault=0;fault<5;++fault) {
                auto bad=b;
                if(fault==0)bad.where=c10::DeviceType::CPU;
                if(fault==1)bad.contiguous=false;
                if(fault==2)bad.grad=true;
                if(fault==3)bad.shape={2,1,64};
                if(fault==4)bad.shape={256};
                auto before=syncs,old_batches=batches;
                rejected_unchanged(a,[&]{rpu_zero_contiguous_tensor_blocks_sized({a,bad},0);});
                assert(syncs==before && batches==old_batches);
            }
        }
    } else if(test=="top2") {
        Tensor logits({2,4}),hidden({2,2}),weight({4,2},at::kHalf,c10::DeviceType::CPU);
        values(logits,{1,3,2,0,5,4,1,0});values(hidden,{1,0,0,1});values(weight,{0,2,1,1,2,0,9,9});
        auto before=snapshot(logits);auto out=rpu_lm_head_logits_top2_rescore_cpu(logits,hidden,weight,2);
        assert(out.device().is_cpu() && out.scalar_type()==at::kLong && out.sizes()==at::Sizes({2}));
        assert(out.data_ptr<int64_t>()[0]==2 && out.data_ptr<int64_t>()[1]==0 && snapshot(logits)==before);
        assert(logits.impl->version==0 && syncs==1 && flushes==2);
        auto second=rpu_lm_head_logits_top2_rescore_cpu(logits,hidden,weight,2);assert(second.data_ptr()!=out.data_ptr());
        rejected_unchanged(logits,[&]{rpu_lm_head_logits_top2_rescore_cpu(logits,hidden,weight,3);});
        Tensor row({65});fill(row,-1);row.data_ptr<c10::Half>()[31]=c10::Half(2);row.data_ptr<c10::Half>()[32]=c10::Half(2);
        auto tie=top2_row_fp16(row.data_ptr<c10::Half>(),65);assert(tie.first==31 && tie.second==32);
        row.data_ptr<c10::Half>()[0].x=0x7e00;row.data_ptr<c10::Half>()[64]=c10::Half(3);
        auto mixed=top2_row_fp16(row.data_ptr<c10::Half>(),65);assert(mixed.first==64 && mixed.second==31);
    } else if(test=="guards") {
        Tensor dst({2,4}),src({2,4});auto idx=ids({0,1});
        dst.where=c10::DeviceType::CPU;
        rejected_unchanged(dst,[&]{rpu_qwen3vl_scatter_row_parts_unflushed_(dst,idx,{src});});
        dst.where=c10::DeviceType::PrivateUse1;dst.grad=true;
        rejected_unchanged(dst,[&]{rpu_qwen3vl_scatter_row_parts_unflushed_(dst,idx,{src});});
        dst.grad=false;idx.dtype=at::kFloat;
        rejected_unchanged(dst,[&]{rpu_qwen3vl_scatter_row_parts_unflushed_(dst,idx,{src});});
    } else return 2;
}
'''.replace("__FUNCTIONS__", "\n".join(functions))
    directory = tmp_path_factory.mktemp("qwen3vl-transport")
    path, binary = directory / "transport.cpp", directory / "transport"
    path.write_text(cpp)
    (directory / "source.sha256").write_text(hashlib.sha256(source.encode()).hexdigest() + "\n")
    torch_include = Path(torch.__file__).resolve().parent / "include"
    result = subprocess.run([compiler, "-std=c++17", "-O0", f"-I{torch_include}",
                             str(path), "-o", str(binary)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return binary


@pytest.mark.parametrize("case", ["scatter", "gather", "publish", "publication", "zero", "zero_reset", "top2", "guards"])
def test_native_transport_success_failure_ownership_and_versions(transport_binary, case):
    subprocess.run([str(transport_binary), case], check=True, capture_output=True, text=True)


def _fake_functions():
    tree = ast.parse((ROOT / "python/rpu_backend/graph/fake_dispatch.py").read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and (node.name in {"_has_fake", "_require_fake"}
                or node.name.startswith(("_meta_zero_contiguous_tensor_blocks", "_meta_lm_head_logits_top2", "_meta_qwen3vl_scatter", "_meta_qwen3vl_gather", "_meta_qwen3vl_publish"))):
            node.decorator_list = []
            selected.append(node)
    namespace = {"torch": torch, "FakeTensor": FakeTensor}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "fake_dispatch.py", "exec"), namespace)
    return namespace


def test_fake_transport_never_reads_or_writes_real_tensor_storage():
    functions = _fake_functions()
    real = torch.ones(2, 4)
    with pytest.raises(RuntimeError, match="non-Fake"):
        functions["_meta_qwen3vl_scatter_row_parts_unflushed_"](real, torch.tensor([0, 1]), [real])
    with pytest.raises(RuntimeError, match="non-Fake"):
        functions["_meta_qwen3vl_gather_embedding_from_rpu_unflushed_"](real, real, torch.tensor([0, 1]))
    with pytest.raises(RuntimeError, match="non-Fake"):
        functions["_meta_qwen3vl_publish_inputs"]([real])
    with FakeTensorMode() as mode:
        fake = mode.from_tensor(real)
        assert functions["_meta_qwen3vl_scatter_row_parts_unflushed_"](fake, torch.tensor([0, 1]), [fake]) is None
        assert functions["_meta_qwen3vl_gather_embedding_from_rpu_unflushed_"](fake, fake, torch.tensor([0, 1])) is None
        assert functions["_meta_qwen3vl_publish_inputs"]([fake]) is None
        assert functions["_meta_zero_contiguous_tensor_blocks_sized"]([fake], 1) is None
        out = functions["_meta_lm_head_logits_top2_rescore_cpu"](fake, fake, fake)
        assert isinstance(out, FakeTensor) and out.device.type == "cpu" and out.dtype == torch.int64
        assert out.shape == (2,)
    assert torch.equal(real, torch.ones_like(real))


def test_mutation_schemas_expose_destination_aliases_and_fresh_top2_result():
    text = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    expected = {"zero_contiguous_tensor_blocks_sized", "qwen3vl_scatter_row_parts_unflushed_",
                "qwen3vl_gather_embedding_from_rpu_unflushed_", "qwen3vl_publish_inputs", "lm_head_logits_top2_rescore_cpu"}
    found = {}
    for match in re.finditer(r'm\.def\((.*?)\);', text, re.S):
        pieces = re.findall(r'"(?:\\.|[^"\\])*"', match.group(1))
        if not pieces:
            continue
        schema = "".join(ast.literal_eval(piece) for piece in pieces)
        name = schema.partition("(")[0]
        if name in expected:
            found[name] = torch._C.parse_schema(schema)
    assert set(found) == expected
    for name, schema in found.items():
        if name == "lm_head_logits_top2_rescore_cpu":
            assert all(arg.alias_info is None for arg in schema.arguments)
            assert schema.returns[0].alias_info is None
        else:
            assert schema.arguments[0].alias_info.is_write
            assert not schema.returns
