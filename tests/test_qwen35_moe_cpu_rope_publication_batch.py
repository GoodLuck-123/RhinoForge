"""Run the complete native MoE RoPE setter with real ATen CPU values/storage.

ROOT executes this fixture. Device labels, device allocation/copy publication and
SDK DDR visibility are host doubles; this is not hardware or Graph evidence.
"""
from pathlib import Path
import re
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _function(source, marker):
    start = source.index(marker)
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"',
                    lambda match: " " * len(match[0]), source)
    brace = masked.index("{", start)
    end, depth = brace + 1, 1
    while depth:
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    return source[start:end]


def _host_boundaries(source):
    # Ordinary CPU and modeled public/private device use real CPU ATen storage.
    # Substitute labels only; keep all native math/shape/receipt/span guards.
    for name in ("cos", "sin", "src"):
        source = source.replace(name + ".device().is_cpu()", "cpu_label(" + name + ")")
        source = source.replace(name + ".device().type() == at::kPrivateUse1",
                                "(!cpu_label(" + name + ") && " + name + ".device().is_cpu())")
    source = source.replace(".device(at::kPrivateUse1)", ".device(at::kCPU)")
    source = source.replace("at::empty(", "empty_rpu(")
    for target, value in (("aligned", "src"), ("prefill_cos_", "cos"),
                          ("prefill_sin_", "sin")):
        source = source.replace(target + ".copy_(" + value + ")",
                                "copy_to_rpu(" + target + ", " + value + ")")
    for index, name in ((0, "cos"), (1, "sin")):
        copy = ("std::memcpy(targets[" + str(index) + "].data_ptr(), " + name +
                ".const_data_ptr(), " + name + ".nbytes());")
        if copy in source:
            source = source.replace(copy, copy + "\n                mark_dirty(targets[" + str(index) + "]);")
    return source


def test_actual_complete_moe_rope_setter_private_batch_and_public_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    cpp_source = (ROOT / "src/fused/rpu_qwen3_5_moe_model.cpp").read_text()
    header = (ROOT / "src/fused/rpu_qwen3_5_moe_model.h").read_text()
    profile = (ROOT / "src/core/rpu_profile.h").read_text()
    setter = _function(cpp_source, "void Qwen3_5MoeModel::set_prefill_rope(")
    aligned_copy = _function(cpp_source, "at::Tensor keep_aligned_rpu_copy(")
    span = _function(header, "    struct PrefillRopeCpuSpan {") + ";"
    macro = profile[profile.index("#define rpu_ddr_flush_force_sized("):].split("\n\n", 1)[0]
    # Ensure the host executes the whole production setter, including the old
    # receipt and fallback, instead of a copied miniature candidate algorithm.
    assert "prefill_rope_cpu_published_ = false;" in setter
    assert setter.count("::rhino_lkn::RpuDdrFlushBatch(ranges, sizes, 2);") == 1
    assert setter.count("std::memcpy(targets[") == 2
    assert setter.index("std::memcpy(targets[1]") < setter.index("RpuDdrFlushBatch")
    assert setter.index("RpuDdrFlushBatch") < setter.index('asm volatile("dsb sy"')
    assert "if (!cpu_pair_batched" in setter and "else if (!cpu_pair_batched)" in setter
    cpp = r'''
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/intrusive_ptr.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
#include <unordered_map>
bool g_rpu_ddr_flush_enabled=true, g_rpu_ddr_flush_force_enabled=true;
struct Store {
    void* base;
    size_t bytes;
    c10::weak_intrusive_ptr<c10::StorageImpl> owner;
    std::vector<uint8_t> ddr, dirty;
};
std::vector<Store> stores;
std::unordered_map<const c10::TensorImpl*,
    c10::weak_intrusive_ptr<c10::TensorImpl,c10::UndefinedTensorImpl>> cpu_labels;
int batch_calls=0, scalar_calls=0, owner_calls=0, sdk_fault=0, fault_target=0;
int flushes_until_failure=-1;
struct Event {void* pointer;size_t bytes;};
std::vector<Event> events;
bool cpu_label(const at::Tensor& t) {
    auto it=cpu_labels.find(t.unsafeGetTensorImpl());
    return it!=cpu_labels.end()&&!it->second.expired();
}
void label_cpu(const at::Tensor& t) {
    cpu_labels.insert_or_assign(t.unsafeGetTensorImpl(),
        c10::weak_intrusive_ptr<c10::TensorImpl,c10::UndefinedTensorImpl>(t.getIntrusivePtr()));
}
Store& locate(void* p,size_t n,size_t& offset) {
    auto address=reinterpret_cast<uintptr_t>(p);
    for(auto it=stores.rbegin();it!=stores.rend();++it) {
        auto base=reinterpret_cast<uintptr_t>(it->base);
        if(!it->owner.expired()&&address>=base&&address-base<=it->bytes
                &&n<=it->bytes-(address-base)) {offset=address-base;return *it;}
    }
    TORCH_CHECK(false,"host DDR fixture: unowned span");
}
at::Tensor empty_rpu(at::IntArrayRef shape,at::TensorOptions options) {
    auto result=at::empty(shape,options.device(at::kCPU));
    std::memset(result.data_ptr(),0x51,result.nbytes());
    stores.push_back({result.data_ptr(),result.nbytes(),
        c10::weak_intrusive_ptr<c10::StorageImpl>(result.storage().getWeakStorageImpl()),
        std::vector<uint8_t>(result.nbytes(),0x51),std::vector<uint8_t>(result.nbytes(),0)});
    return result;
}
void mark_dirty(const at::Tensor& t) {
    size_t offset;auto& store=locate(t.data_ptr(),t.nbytes(),offset);
    std::fill(store.dirty.begin()+offset,store.dirty.begin()+offset+t.nbytes(),1);
}
std::vector<uint8_t> logical_bytes(const at::Tensor& t) {
    auto materialized=t.resolve_conj().resolve_neg().contiguous().clone();
    const auto* p=static_cast<const uint8_t*>(materialized.const_data_ptr());
    return {p,p+materialized.nbytes()};
}
std::vector<uint8_t> device_bytes(const at::Tensor& t) {
    size_t offset;auto& store=locate(t.data_ptr(),t.nbytes(),offset);
    return {store.ddr.begin()+offset,store.ddr.begin()+offset+t.nbytes()};
}
namespace rhino_lkn {
uint64_t RpuGetDevAddr(void* p){return reinterpret_cast<uint64_t>(p);}
struct HostDDR_t {void* base;size_t bytes;
    void* get_cpu_ptr(){return base;}size_t get_memory_size(){return bytes;}};
HostDDR_t* RpuGetHostddr(void* p) {
    size_t offset;auto& store=locate(p,0,offset);static HostDDR_t owner;
    owner={store.base,store.bytes};const int call=owner_calls++;
    if(sdk_fault&&call==fault_target) {
        const auto address=reinterpret_cast<uintptr_t>(p);
        if(sdk_fault==1)return nullptr;
        if(sdk_fault==2)owner.base=nullptr;
        if(sdk_fault==3)owner.bytes=0;
        if(sdk_fault==4)owner.base=reinterpret_cast<void*>(address+1);
        if(sdk_fault==5)owner.bytes=offset+186-1;
        if(sdk_fault==6)owner.bytes=std::numeric_limits<size_t>::max();
        if(sdk_fault==7)owner.bytes=offset+186; // logical fits,64B rounded tail does not
    }
    return &owner;
}
void flush_range(void* p,size_t n) {
    size_t offset;auto& store=locate(p,n,offset);events.push_back({p,n});
    if(flushes_until_failure==0){flushes_until_failure=-1;TORCH_CHECK(false,"injected publication failure");}
    if(flushes_until_failure>0)--flushes_until_failure;
    if(n==0)return;
    const auto begin=offset&~size_t(63),end=(offset+n+63)&~size_t(63);
    assert(end<=store.bytes);auto* cpu=static_cast<uint8_t*>(store.base);
    for(size_t i=begin;i<end;++i) {
        if(store.dirty[i])store.ddr[i]=cpu[i];else cpu[i]=store.ddr[i];
        store.dirty[i]=0;
    }
}
void RpuDdrFlush(void* p,size_t n){++scalar_calls;flush_range(p,n);}
void RpuDdrFlushBatch(void** ranges,size_t* sizes,size_t count) {
    ++batch_calls;assert(count==2);
    // Both full current CPU copies must exist before the first publication.
    for(size_t i=0;i<count;++i) {
        size_t offset;auto& store=locate(ranges[i],sizes[i],offset);
        assert(std::all_of(store.dirty.begin()+offset,
                          store.dirty.begin()+offset+sizes[i],[](uint8_t x){return x==1;}));
    }
    for(size_t i=0;i<count;++i)flush_range(ranges[i],sizes[i]);
}
}
using rhino_lkn::RpuGetDevAddr;
__FORCE_MACRO__
void copy_to_rpu(at::Tensor& target,const at::Tensor& source) {
    if(target.is_same(source))return;
    if(!cpu_label(source))rpu_ddr_flush_force_sized(source.data_ptr(),source.nbytes());
    target.copy_(source); // real ATen supplies lazy-neg materialization and math
    mark_dirty(target);
    rpu_ddr_flush_force_sized(target.data_ptr(),target.nbytes());
}
__ALIGNED_COPY__
namespace v3 {
struct Qwen3_5MoeModel {
    at::Tensor prefill_cos_,prefill_sin_;
    __SPAN__
    std::array<PrefillRopeCpuSpan,2> prefill_rope_cpu_owners_{};
    uint64_t prefill_rope_cpu_generation_=0,generation=1;
    bool prefill_rope_cpu_owned_=false,prefill_rope_cpu_published_=false;
    uint64_t installed_model_state_generation(){return generation;}
    void set_prefill_rope(const at::Tensor&,const at::Tensor&);
};
__SETTER__
}
void reset_events(){batch_calls=scalar_calls=owner_calls=0;events.clear();}
at::Tensor cpu_table(int64_t rows,int64_t width,float bias=0) {
    auto t=(at::arange(rows*width,at::kFloat).mul(0.0001).sub(0.5).add(bias))
        .view({rows,width}).to(at::kHalf);label_cpu(t);return t;
}
at::Tensor public_table(int64_t rows,int64_t width,float bias) {
    auto c=cpu_table(rows,width,bias);
    auto t=keep_aligned_rpu_copy(c,"fixture public device",256,true);
    return t; // deliberately retain a public device alias, not a CPU label
}
void oracle(const v3::Qwen3_5MoeModel& m,const at::Tensor& c,const at::Tensor& s,bool device=true) {
    assert(at::equal(m.prefill_cos_,c.resolve_neg().resolve_conj()));
    assert(at::equal(m.prefill_sin_,s.resolve_neg().resolve_conj()));
    assert(at::isfinite(m.prefill_cos_).all().item<bool>());
    assert(at::isfinite(m.prefill_sin_).all().item<bool>());
    if(device){assert(device_bytes(m.prefill_cos_)==logical_bytes(c));
               assert(device_bytes(m.prefill_sin_)==logical_bytes(s));}
}
void unchanged_edges(const at::Tensor& t) {
    size_t offset;auto& store=locate(t.data_ptr(),t.nbytes(),offset);
    for(size_t i=0;i<store.bytes;++i)
        if(i<offset||i>=offset+t.nbytes())assert(store.ddr[i]==0x51);
}
template<class F>void fail_before_write(F call) {
    reset_events();bool failed=false;try{call();}catch(const c10::Error&){failed=true;}
    assert(failed&&batch_calls==0&&scalar_calls==0);
}
int main(){int scenarios=0;
for(bool normal:{false,true})for(bool inference:{false,true})for(int rows:{1,13,64,4096}) {
    ++scenarios;c10::InferenceMode mode(inference);
    g_rpu_ddr_flush_enabled=normal;g_rpu_ddr_flush_force_enabled=true;
    auto c=cpu_table(rows,32),s=cpu_table(rows,32,0.125);
    auto a_c=c.clone(),a_s=s.clone();v3::Qwen3_5MoeModel m;
    reset_events();m.set_prefill_rope(c,s);assert(batch_calls==1&&scalar_calls==0);
    oracle(m,c,s);unchanged_edges(m.prefill_cos_);unchanged_edges(m.prefill_sin_);
    auto* cp=m.prefill_cos_.data_ptr();auto* sp=m.prefill_sin_.data_ptr();
    const auto cv=inference?-1:c._version(),sv=inference?-1:s._version();
    reset_events();m.set_prefill_rope(c,s);assert(batch_calls==0&&scalar_calls==0);
    // Same pointer/raw bytes mutation, no ATen version increment.
    c.data_ptr<c10::Half>()[0]=c10::Half(-0.25);
    reset_events();m.set_prefill_rope(c,s);assert(batch_calls==1&&scalar_calls==0);oracle(m,c,s);
    s.data_ptr<c10::Half>()[0]=c10::Half(0.375);
    reset_events();m.set_prefill_rope(c,s);assert(batch_calls==1&&scalar_calls==0);oracle(m,c,s);
    assert((inference?-1:c._version())==cv&&(inference?-1:s._version())==sv);
    c.copy_(a_c);s.copy_(a_s);reset_events();m.set_prefill_rope(c,s);
    assert(batch_calls==1&&scalar_calls==0);oracle(m,c,s);
    auto fresh_c=c.clone(),fresh_s=s.clone();label_cpu(fresh_c);label_cpu(fresh_s);
    reset_events();m.set_prefill_rope(fresh_c,fresh_s);assert(batch_calls==0&&scalar_calls==0);
    assert(m.prefill_cos_.data_ptr()==cp&&m.prefill_sin_.data_ptr()==sp);
    ++m.generation;reset_events();m.set_prefill_rope(c,s);assert(batch_calls==1&&scalar_calls==0);
    auto old_c=m.prefill_cos_,old_s=m.prefill_sin_;auto old_cb=logical_bytes(old_c),old_sb=logical_bytes(old_s);
    auto different_c=cpu_table(rows+1,32),different_s=cpu_table(rows+1,32,0.25);
    reset_events();m.set_prefill_rope(different_c,different_s);assert(batch_calls==1&&scalar_calls==0);
    oracle(m,different_c,different_s);assert(logical_bytes(old_c)==old_cb&&logical_bytes(old_s)==old_sb);
}
// Force-off preserves original scalar-copy semantics and invalidates receipt.
for(bool inference:{false,true}){
    ++scenarios;c10::InferenceMode mode(inference);v3::Qwen3_5MoeModel m;
    auto c=cpu_table(13,32),s=cpu_table(13,32,0.25);m.set_prefill_rope(c,s);
    c.data_ptr<c10::Half>()[0]=c10::Half(0.5);g_rpu_ddr_flush_force_enabled=false;
    reset_events();m.set_prefill_rope(c,s);assert(batch_calls==0&&scalar_calls==0&&!m.prefill_rope_cpu_published_);
    oracle(m,c,s,false);g_rpu_ddr_flush_force_enabled=true;
    reset_events();m.set_prefill_rope(c,s);assert(batch_calls==1&&scalar_calls==0);oracle(m,c,s);
}
// Both SDK owner lookups can fail independently; no optimized writes/publication.
for(int fault=1;fault<=7;++fault)for(int target:{0,1}) {
    ++scenarios;v3::Qwen3_5MoeModel m;auto c=cpu_table(3,31),s=cpu_table(3,31,0.125);
    sdk_fault=fault;fault_target=target;reset_events();m.set_prefill_rope(c,s);
    assert(batch_calls==0&&scalar_calls==2);oracle(m,c,s);sdk_fault=0;
}
// SDK ownership alone cannot bypass the real ATen backing's64B rounded extent.
{
    ++scenarios;v3::Qwen3_5MoeModel m;auto c=cpu_table(3,31),s=cpu_table(3,31,0.125);m.set_prefill_rope(c,s);
    auto hold=m.prefill_cos_;m.prefill_cos_=at::from_blob(hold.data_ptr(),hold.sizes(),at::kHalf);
    c.data_ptr<c10::Half>()[0]=c10::Half(0.25);reset_events();m.set_prefill_rope(c,s);
    assert(batch_calls==0&&scalar_calls==2);oracle(m,c,s);
}
// Public device aliases stay public; CPU subsequent writes must keep original fallback.
{
    ++scenarios;v3::Qwen3_5MoeModel m;auto c=public_table(13,32,0),s=public_table(13,32,0.25);
    reset_events();m.set_prefill_rope(c,s);assert(batch_calls==0&&!m.prefill_rope_cpu_owned_);
    assert(m.prefill_cos_.is_same(c)&&m.prefill_sin_.is_same(s));oracle(m,c,s);
    auto cp=cpu_table(13,32,0.5),sp=cpu_table(13,32,0.75);
    reset_events();m.set_prefill_rope(cp,sp);assert(batch_calls==0&&scalar_calls==2&&!m.prefill_rope_cpu_published_);
    oracle(m,cp,sp);
}
// Dirty public device source / offset view and a mixed CPU-device pair acquire current bytes.
{
    ++scenarios;v3::Qwen3_5MoeModel m;auto c=cpu_table(13,32),s=cpu_table(13,32,0.125);m.set_prefill_rope(c,s);
    auto rc=public_table(14,32,0.25).narrow(0,1,13);rc.data_ptr<c10::Half>()[0]=c10::Half(-0.75);mark_dirty(rc);
    auto rs=public_table(13,32,0.5);reset_events();m.set_prefill_rope(rc,rs);
    assert(batch_calls==0&&scalar_calls==4);oracle(m,rc,rs);
    c.data_ptr<c10::Half>()[0]=c10::Half(0.75);reset_events();m.set_prefill_rope(c,rs);
    assert(batch_calls==0&&scalar_calls==3&&!m.prefill_rope_cpu_published_);oracle(m,c,rs);
}
// A CPU view of either private backing never gains Batch authority.
{
    ++scenarios;v3::Qwen3_5MoeModel m;auto c=cpu_table(13,32),s=cpu_table(13,32,0.125);m.set_prefill_rope(c,s);
    auto alias=m.prefill_cos_.view(m.prefill_cos_.sizes());label_cpu(alias);
    alias.data_ptr<c10::Half>()[0]=c10::Half(0.25);mark_dirty(alias);
    reset_events();m.set_prefill_rope(alias,s);assert(batch_calls==0&&!m.prefill_rope_cpu_published_);
    oracle(m,alias,s);
}
// Lazy-neg raw storage is not its logical value: never reuse a raw-byte receipt.
{
    ++scenarios;v3::Qwen3_5MoeModel m;auto c=cpu_table(13,32),s=cpu_table(13,32,0.125);m.set_prefill_rope(c,s);
    auto nc=at::_neg_view(c),ns=at::_neg_view(s);label_cpu(nc);label_cpu(ns);
    assert(nc.is_neg()&&ns.is_neg()&&nc.is_contiguous());reset_events();m.set_prefill_rope(nc,ns);
    assert(batch_calls==0&&scalar_calls==2&&!m.prefill_rope_cpu_published_);oracle(m,nc,ns);
}
// CPU aliases in unused alignment tails also fall back before either fast write.
{
    ++scenarios;v3::Qwen3_5MoeModel m;auto c=cpu_table(1,1),s=cpu_table(1,1,0.125);m.set_prefill_rope(c,s);
    auto* tail=static_cast<uint8_t*>(m.prefill_cos_.data_ptr())+64;
    const auto storage=m.prefill_cos_.storage();
    assert(reinterpret_cast<uintptr_t>(tail)+2<=
        reinterpret_cast<uintptr_t>(storage.data_ptr().get())+storage.nbytes());
    auto alias=at::from_blob(tail,{1,1},at::kHalf);label_cpu(alias);
    reset_events();m.set_prefill_rope(alias,s);
    assert(batch_calls==0&&scalar_calls==2&&!m.prefill_rope_cpu_published_);oracle(m,alias,s);
}
// Existing dtype/dim/shape/stride failures occur before any writes.
{
    ++scenarios;v3::Qwen3_5MoeModel m;auto c=cpu_table(13,32),s=cpu_table(13,32,0.125);m.set_prefill_rope(c,s);
    auto old=logical_bytes(m.prefill_cos_);auto fp32=c.to(at::kFloat);label_cpu(fp32);
    fail_before_write([&]{m.set_prefill_rope(fp32,fp32);});
    auto noncontiguous=c.transpose(0,1);label_cpu(noncontiguous);
    fail_before_write([&]{m.set_prefill_rope(noncontiguous,noncontiguous);});
    auto mismatch=cpu_table(14,32);fail_before_write([&]{m.set_prefill_rope(c,mismatch);});
    auto flat=c.flatten();label_cpu(flat);fail_before_write([&]{m.set_prefill_rope(flat,flat);});
    auto complex=at::ones({13,32},at::kComplexFloat).conj();label_cpu(complex);
    fail_before_write([&]{m.set_prefill_rope(complex,complex);});
    fail_before_write([&]{m.set_prefill_rope(at::Tensor(),s);});
    assert(logical_bytes(m.prefill_cos_)==old);
}
// A partial SDK failure must invalidate the pair before any later retry can
// accept equal CPU mappings as proof of both current device tables.
for(bool fallback:{false,true}) {
    ++scenarios;v3::Qwen3_5MoeModel m;auto c=cpu_table(13,32),s=cpu_table(13,32,0.125);
    m.set_prefill_rope(c,s);c.data_ptr<c10::Half>()[0]=c10::Half(0.75);
    s.data_ptr<c10::Half>()[0]=c10::Half(-0.5);
    if(fallback){sdk_fault=1;fault_target=0;}
    reset_events();flushes_until_failure=1;bool failed=false;
    try{m.set_prefill_rope(c,s);}catch(const c10::Error&){failed=true;}
    assert(failed&&!m.prefill_rope_cpu_published_);
    assert(device_bytes(m.prefill_cos_)==logical_bytes(c));
    assert(device_bytes(m.prefill_sin_)!=logical_bytes(s));
    sdk_fault=0;reset_events();m.set_prefill_rope(c,s);
    assert(batch_calls==1&&scalar_calls==0&&m.prefill_rope_cpu_published_);oracle(m,c,s);
    reset_events();m.set_prefill_rope(c,s);assert(batch_calls==0&&scalar_calls==0);
}
std::cout<<"WHOLE_NATIVE_ROPE_SETTER_ATEN_CPU_ORACLE_SCENARIOS="<<scenarios<<"\n";
}
'''
    cpp = (cpp.replace("__FORCE_MACRO__", macro)
           .replace("__ALIGNED_COPY__", _host_boundaries(aligned_copy))
           .replace("__SPAN__", span)
           .replace("__SETTER__", _host_boundaries(setter)))
    compiler = shutil.which("c++")
    assert compiler, "whole native setter fixture requires C++ compiler"
    source, binary = tmp_path / "moe_rope_setter.cpp", tmp_path / "moe_rope_setter"
    source.write_text(cpp)
    torch_root = Path(torch.__file__).resolve().parent
    command = [compiler, "-std=c++17", "-O0", "-D_GLIBCXX_USE_CXX11_ABI=" +
               str(int(torch.compiled_with_cxx11_abi())),
               "-I", str(torch_root / "include"),
               "-I", str(torch_root / "include/torch/csrc/api/include"), str(source),
               "-L", str(torch_root / "lib"), "-Wl,-rpath," + str(torch_root / "lib"),
               "-ltorch_cpu", "-lc10", "-o", str(binary)]
    built = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert built.returncode == 0, built.stdout + built.stderr
    executed = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert "WHOLE_NATIVE_ROPE_SETTER_ATEN_CPU_ORACLE_SCENARIOS=" in executed.stdout
