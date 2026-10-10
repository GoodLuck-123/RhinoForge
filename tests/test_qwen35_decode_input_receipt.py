"""Actual decode ingress/gather, real ATen storage, host allocation/SDK doubles.

The complete production functions execute in the existing native host helper.
Only the private device allocation and weight device label are adapted to CPU.
DDR is a separate byte shadow; FMB computation, the planner, SDK completion and
hardware Graph execution are doubles. This does not measure hits/performance or
admit concurrent native calls/multiple forwards in an unfinished raw Graph.
"""
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _byte_literals(frames):
    return "{" + ",".join("{" + ",".join(map(str, frame)) + "}" for frame in frames) + "}"


@pytest.mark.parametrize("family", ["dense", "moe"])
def test_actual_native_decode_cpu_input_receipt(tmp_path, monkeypatch, family):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    from test_siglip_attention_dry_live_contract import _function, _run_native

    filename, owner = {
        "dense": ("rpu_qwen3_5_model", "Qwen3_5Model"),
        "moe": ("rpu_qwen3_5_moe_model", "Qwen3_5MoeModel"),
    }[family]
    source = (ROOT / "src/fused" / f"{filename}.cpp").read_text()
    prepare = _function(source, f"at::Tensor {owner}::prepare_decode_cpu_input(")
    forward = _function(source, f"at::Tensor {owner}::forward(")
    # Explicit allocator double, not a reconstructed receipt implementation.
    # All Tensor dtype/shape/stride/storage/version checks remain real ATen.
    assert prepare.count("at::empty(") == 1
    prepare = prepare.replace("at::empty(", "allocate_private_owner(").replace(
        "at::kPrivateUse1", "at::kCPU")
    definitions = (prepare + "\n" + forward).replace(f"{owner}::", "Model::")
    header = (ROOT / "src/fused" / f"{filename}.h").read_text().splitlines()
    first = next(i for i, line in enumerate(header)
                 if re.search(r"at::Tensor\s+decode_cpu_input_owner_;", line))
    last = next(i for i, line in enumerate(header)
                if re.search(r"std::mutex\s+decode_cpu_input_mutex_;", line))
    fields = "\n".join(header[first:last + 1])
    fmb = (ROOT / "src/core/fused_model_base.cpp").read_text()
    begin = fmb.index("    // Layer-0 mutable DMA")
    live_input = fmb[begin:fmb.index("    const bool skip_preload_replay", begin)]
    profile = (ROOT / "src/core/rpu_profile.h").read_text()
    macro = profile[profile.index("#define rpu_ddr_flush_force_sized("):].split("\n\n", 1)[0]

    # Stage actual NumPy raw-alias mutations, without a Tensor version bump.
    # The standalone C++ fixture replays these bytes via an ATen raw pointer;
    # the production C++ code, not Python, decides every receipt hit/miss.
    staged = torch.zeros((1, 1, 32), dtype=torch.float16)
    alias, version = staged.numpy(), staged._version
    frames = []
    for value in (1, 1, 3, 1):
        alias.fill(value)
        assert staged._version == version
        frames.append(alias.tobytes())

    cpp = r'''
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <c10/util/intrusive_ptr.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

bool g_rpu_ddr_flush_force_enabled = true;
bool rpu_get_ddr_flush_force() {return g_rpu_ddr_flush_force_enabled;}
struct Store {
    void* pointer;
    size_t bytes;
    c10::weak_intrusive_ptr<c10::TensorImpl,c10::UndefinedTensorImpl> tensor;
    std::vector<uint8_t> ddr;
    explicit Store(const at::Tensor& t)
        : pointer(t.data_ptr()), bytes(t.nbytes()), tensor(t.getIntrusivePtr()),
          ddr(t.nbytes(), 0xa7) {}
};
std::vector<std::unique_ptr<Store>> stores;
int allocations=0, flushes=0, address_reads=0, consumers=0;
bool fail_allocation=false, fail_publication=false;
Store& locate(void* pointer) {
    // An expired logical allocation at a recycled numerical address is not
    // the new owner. The registry itself retains no Tensor/storage.
    for (auto i=stores.rbegin(); i!=stores.rend(); ++i)
        if ((*i)->pointer==pointer && !(*i)->tensor.expired()) return **i;
    throw std::runtime_error("consumer has no live Tensor owner");
}
std::vector<uint8_t> bytes(const at::Tensor& t) {
    auto* p=static_cast<const uint8_t*>(t.const_data_ptr());
    return {p,p+t.nbytes()};
}
at::Tensor allocate_private_owner(at::IntArrayRef shape, at::TensorOptions options) {
    ++allocations;
    if (std::exchange(fail_allocation,false)) throw std::runtime_error("allocation failed");
    auto out=at::empty(shape,options);
    stores.push_back(std::make_unique<Store>(out));
    return out;
}
namespace rhino_lkn {
uint64_t RpuGetDevAddr(void* pointer) {
    ++address_reads;
    locate(pointer);
    return reinterpret_cast<uint64_t>(pointer);
}
void RpuDdrFlush(void* pointer,size_t size) {
    ++flushes;
    if (std::exchange(fail_publication,false)) throw std::runtime_error("publication failed");
    auto& store=locate(pointer);
    assert(size==store.bytes && size>0);
    std::memcpy(store.ddr.data(),pointer,size);
}
}
__MACRO__
struct PendingRead {
    std::function<uint64_t()> current_base;
    uint64_t emitted_base;
    at::Tensor output;
};
struct Graph {
    std::vector<at::Tensor> held;
    std::vector<PendingRead> pending;
    void keep_alive(const at::Tensor& t) {held.push_back(t);}
    void end() {
        for (auto& read:pending) {
            // Execution happens after the native private_hidden local died.
            const uint64_t base=read.current_base();
            assert(base==read.emitted_base);
            auto& store=locate(reinterpret_cast<void*>(base));
            assert(read.output.nbytes()==store.bytes);
            std::memcpy(read.output.data_ptr(),store.ddr.data(),store.bytes);
        }
        pending.clear();
        held.clear();
    }
} graph;
struct RpuKernelGraph {
    static bool has_active() {return true;}
    static Graph& active() {return graph;}
};
struct Model {
    __FIELDS__
    bool action_mode_=false, z2_bound_=false;
    int64_t mrope_pos_delta_=0;
    uint64_t generation=1, input_base=0;
    int public_delegations=0;
    std::vector<int64_t> positions;
    int64_t hidden_size() const {return 32;}
    uint64_t installed_model_state_generation() const {return generation;}
    uint64_t& current_hidden_in_src_base() {return input_base;}
    at::Tensor prepare_decode_cpu_input(const at::Tensor&);
    at::Tensor forward(const at::Tensor&,std::vector<at::Tensor>&,
        std::vector<at::Tensor>&,std::vector<at::Tensor>&,std::vector<at::Tensor>&,
        const std::optional<at::Tensor>&,int64_t,bool,int64_t,at::IntArrayRef);
    at::Tensor forward_impl(const at::Tensor& hidden_states,
        std::vector<at::Tensor>&,std::vector<at::Tensor>&,
        std::vector<at::Tensor>&,std::vector<at::Tensor>&,
        const std::optional<at::Tensor>&,int64_t position,bool,
        int64_t,at::IntArrayRef) {
        if (!hidden_states.device().is_cpu()) {
            // A Meta label exercises only the unchanged non-CPU delegation;
            // this double makes no claim that Meta is a valid device input.
            ++public_delegations;
            return at::zeros({1,1,32},at::kHalf);
        }
        ++consumers;
        positions.push_back(position);
        __LIVE_INPUT__
        auto output=at::empty(hidden_states.sizes(),hidden_states.options());
        graph.pending.push_back({[this]{return input_base;},input_base,output});
        return output;
    }
    at::Tensor call(const at::Tensor& input,int64_t position=1,bool causal=true,
                    std::optional<at::Tensor> mask=std::nullopt,int64_t chunk=0,
                    std::vector<int64_t> descriptor={3,7}) {
        std::vector<at::Tensor> caches;
        return forward(input,caches,caches,caches,caches,mask,position,causal,chunk,descriptor);
    }
};
__DEFINITIONS__
int64_t version(const at::Tensor& t) {return t.is_inference()?-1:t._version();}
void raw_write(const at::Tensor& t,const std::vector<uint8_t>& value) {
    assert(t.nbytes()==value.size());
    std::memcpy(t.data_ptr(),value.data(),value.size());
}
void rejected_unchanged(Model& m,const std::function<void()>& run) {
    const int a=allocations,f=flushes,c=consumers;
    auto owner=m.decode_cpu_input_owner_;
    const auto saved=bytes(owner), receipt=m.decode_cpu_input_bytes_;
    const auto generation=m.decode_cpu_input_generation_;
    const bool published=m.decode_cpu_input_published_;
    bool rejected=false;
    try {run();} catch(const std::exception&) {rejected=true;}
    assert(rejected && allocations==a && flushes==f && consumers==c);
    assert(graph.held.empty() && graph.pending.empty());
    assert(m.decode_cpu_input_owner_.data_ptr()==owner.data_ptr());
    assert(bytes(owner)==saved && m.decode_cpu_input_bytes_==receipt);
    assert(m.decode_cpu_input_generation_==generation && m.decode_cpu_input_published_==published);
}
const std::vector<std::vector<uint8_t>> frames=__FRAMES__;
int main() {
    for (bool inference:{false,true}) {
        c10::InferenceMode mode(inference);
        stores.clear(); graph.held.clear(); graph.pending.clear();
        allocations=flushes=address_reads=consumers=0;
        g_rpu_ddr_flush_force_enabled=true;
        Model m;
        auto input=at::zeros({1,1,32},at::kHalf);
        const int64_t input_version=version(input);
        std::vector<at::Tensor> retained_outputs, old_graph_leases;
        std::vector<std::vector<uint8_t>> saved_outputs, saved_owners;
        const int expected[]={1,1,2,3};
        for (size_t i=0;i<frames.size();++i) {
            raw_write(input,frames[i]); assert(version(input)==input_version);
            auto output=m.call(input,1+i);
            assert(allocations==expected[i] && flushes==expected[i]);
            assert(consumers==int(i+1) && address_reads==consumers);
            assert(m.input_base==reinterpret_cast<uint64_t>(m.decode_cpu_input_owner_.data_ptr()));
            // Retain old leases separately to prove replacement never rewrites
            // their CPU/DDR bytes. Every actual call still ends before the next;
            // this is not a multiple-forward unfinished raw-Graph protocol.
            old_graph_leases.push_back(graph.held.back());
            saved_owners.push_back(frames[i]);
            graph.end();
            assert(bytes(output)==frames[i]);
            retained_outputs.push_back(output); saved_outputs.push_back(frames[i]);
            for(size_t j=0;j<old_graph_leases.size();++j) {
                assert(bytes(old_graph_leases[j])==saved_owners[j]);
                assert(locate(old_graph_leases[j].data_ptr()).ddr==saved_owners[j]);
                assert(bytes(retained_outputs[j])==saved_outputs[j]);
                if(j!=i)assert(retained_outputs[j].data_ptr()!=output.data_ptr());
            }
        }
        // A fresh CPU allocation containing identical bytes is a real hit.
        auto fresh=input.clone(); assert(fresh.data_ptr()!=input.data_ptr());
        auto output=m.call(fresh,8); graph.end();
        assert(allocations==3 && flushes==3 && consumers==5 && bytes(output)==frames.back());
        // The final byte differs even though the rest of the hidden row matches.
        auto tail=frames.back(); tail.back()^=1;
        raw_write(input,tail); assert(version(input)==input_version);
        auto old=m.decode_cpu_input_owner_;
        output=m.call(input,9); graph.end();
        assert(allocations==4 && flushes==4 && bytes(output)==tail);
        assert(m.decode_cpu_input_owner_.data_ptr()!=old.data_ptr() && bytes(old)==frames.back());
        old=m.decode_cpu_input_owner_; ++m.generation;
        output=m.call(input,10); graph.end();
        assert(allocations==5 && flushes==5 && bytes(output)==tail);
        assert(m.decode_cpu_input_owner_.data_ptr()!=old.data_ptr());

        // Disabled boundary maintenance never makes an unflushed owner valid.
        g_rpu_ddr_flush_force_enabled=false;
        for(int i=0;i<2;++i) {
            output=m.call(input,11+i);
            assert(!m.decode_cpu_input_published_ && !m.decode_cpu_input_owner_.defined());
            auto& live=locate(reinterpret_cast<void*>(m.input_base));
            assert(!live.tensor.expired() && graph.held.size()==1);
            graph.end();
            assert(live.tensor.expired());
        }
        assert(allocations==7 && flushes==5);
        g_rpu_ddr_flush_force_enabled=true;
        output=m.call(input,13); graph.end();
        assert(allocations==8 && flushes==6 && bytes(output)==tail);

        // Failures cannot leave a usable receipt; retry must publish afresh.
        for(bool allocation_fault:{true,false}) {
            old=m.decode_cpu_input_owner_; const auto old_bytes=bytes(old);
            auto changed=tail; changed.front()^=uint8_t(allocation_fault?1:2);
            raw_write(input,changed);
            if(allocation_fault)fail_allocation=true;else fail_publication=true;
            const int prior_consumers=consumers;
            bool rejected=false;
            try {m.call(input,14);} catch(const std::exception&) {rejected=true;}
            assert(rejected && !m.decode_cpu_input_published_ && consumers==prior_consumers);
            assert(graph.held.empty() && graph.pending.empty() && bytes(old)==old_bytes);
            const int prior_flushes=flushes;
            output=m.call(input,15); graph.end();
            assert(flushes==prior_flushes+1 && bytes(output)==changed);
            assert(m.decode_cpu_input_owner_.data_ptr()!=old.data_ptr() && bytes(old)==old_bytes);
            tail=changed;
        }

        // Actual ingress validation runs before receipt/allocation/publication.
        for(auto invalid:{at::zeros({1,32},at::kHalf),at::zeros({2,1,32},at::kHalf),
                          at::zeros({1,2,32},at::kHalf),at::zeros({1,1,31},at::kHalf),
                          at::zeros({1,1,32},at::kFloat),
                          at::zeros({1,1,64},at::kHalf).slice(2,0,64,2)})
            rejected_unchanged(m,[&]{m.call(invalid);});
        auto grad=at::zeros({1,1,32},at::kHalf);
        if(!inference) {
            grad.set_requires_grad(true);
            rejected_unchanged(m,[&]{m.call(grad);});
        }
        rejected_unchanged(m,[&]{m.call(at::Tensor{});});
        rejected_unchanged(m,[&]{m.call(input,0);});
        rejected_unchanged(m,[&]{m.call(input,1,false);});
        rejected_unchanged(m,[&]{m.call(input,1,true,at::ones({1,1}));});
        rejected_unchanged(m,[&]{m.call(input,1,true,std::nullopt,64);});
        rejected_unchanged(m,[&]{m.call(input,1,true,std::nullopt,0,{});});
        m.mrope_pos_delta_=1;
        rejected_unchanged(m,[&]{m.call(input);}); m.mrope_pos_delta_=0;
        __DENSE_GUARDS__
        // Non-CPU input goes to the old forward_impl directly, never into the
        // receipt. Meta is only a device-label control for this stub delegate.
        const int a=allocations,f=flushes;
        auto public_input=at::empty({1,1,32},at::TensorOptions().dtype(at::kHalf).device(at::kMeta));
        output=m.call(public_input);
        assert(m.public_delegations==1 && allocations==a && flushes==f);
        assert(graph.held.empty() && graph.pending.empty());
        for(size_t j=0;j<retained_outputs.size();++j)
            assert(bytes(retained_outputs[j])==saved_outputs[j]);
        // Stored registry records are weak; releasing model/Graph owners can
        // retire the actual ATen allocation rather than hiding a lease leak.
        old=at::Tensor{}; old_graph_leases.clear();
        auto* final_pointer=m.decode_cpu_input_owner_.data_ptr();
        auto& final_store=locate(final_pointer);
        m.decode_cpu_input_owner_=at::Tensor{};
        assert(final_store.tensor.expired());
    }
    std::cout << "actual receipt/ingress checked\n";
}
'''
    dense_guards = """
        m.action_mode_=true;
        rejected_unchanged(m,[&]{m.call(input);}); m.action_mode_=false;
        m.z2_bound_=true;
        rejected_unchanged(m,[&]{m.call(input);}); m.z2_bound_=false;
    """ if family == "dense" else ""
    _run_native(tmp_path, cpp.replace("__MACRO__", macro).replace("__FIELDS__", fields)
                .replace("__LIVE_INPUT__", live_input).replace("__DEFINITIONS__", definitions)
                .replace("__FRAMES__", _byte_literals(frames)).replace("__DENSE_GUARDS__", dense_guards), torch)


def test_actual_decode_embedding_cpu_reads_current_weight_and_ids(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    from test_siglip_attention_dry_live_contract import _function, _run_native

    source = (ROOT / "src/core/rpu_backend.cpp").read_text()
    gather = _function(source, "at::Tensor rpu_qwen3_5_decode_embedding_cpu(")
    # The one device weight-label guard is mapped to CPU, not the gather math.
    assert gather.count("c10::DeviceType::PrivateUse1") == 1
    gather = gather.replace("c10::DeviceType::PrivateUse1", "c10::DeviceType::CPU")
    weight = torch.zeros((4, 32), dtype=torch.float16)
    ids = torch.ones((1, 1), dtype=torch.int64)
    weight_alias, ids_alias = weight.numpy(), ids.numpy()
    versions = (weight._version, ids._version)
    weight_frames, expected, id_frames = [], [], []
    for token, value in ((1, 1), (1, 3), (2, 5), (1, 1)):
        ids_alias.fill(token)
        weight_alias[token].fill(value)
        assert (weight._version, ids._version) == versions
        weight_frames.append(weight_alias.tobytes())
        id_frames.append(token)
        expected.append(torch.nn.functional.embedding(ids, weight).numpy().tobytes())
    cpp = r'''
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <vector>
int weight_reads=0;
at::Tensor rpu_to_cpu_zerocopy(const at::Tensor& weight,bool flush) {
    assert(!flush); ++weight_reads; return weight;
}
__GATHER__
std::vector<uint8_t> bytes(const at::Tensor& t) {
    auto* p=static_cast<const uint8_t*>(t.const_data_ptr());return {p,p+t.nbytes()};
}
int64_t version(const at::Tensor& t) {return t.is_inference()?-1:t._version();}
const std::vector<std::vector<uint8_t>> weights=__WEIGHTS__, expected=__EXPECTED__;
const std::vector<int64_t> tokens=__TOKENS__;
void invalid(const at::Tensor& weight,const at::Tensor& ids) {
    const int before=weight_reads;bool rejected=false;
    try {rpu_qwen3_5_decode_embedding_cpu(weight,ids,-1,false,false);}
    catch(const std::exception&) {rejected=true;}
    assert(rejected && weight_reads==before);
}
int main() {
    for(bool inference:{false,true}) {
        c10::InferenceMode mode(inference);
        auto weight=at::zeros({4,32},at::kHalf), ids=at::ones({1,1},at::kLong);
        const auto wv=version(weight),iv=version(ids);
        std::vector<at::Tensor> outputs;
        weight_reads=0;
        for(size_t i=0;i<tokens.size();++i) {
            std::memcpy(weight.data_ptr(),weights[i].data(),weights[i].size());
            *ids.data_ptr<int64_t>()=tokens[i];
            assert(version(weight)==wv && version(ids)==iv);
            auto out=rpu_qwen3_5_decode_embedding_cpu(weight,ids,-1,false,false);
            assert(weight_reads==int(i+1) && out.device().is_cpu() && out.scalar_type()==at::kHalf);
            assert(out.sizes()==at::IntArrayRef({1,1,32}) && bytes(out)==expected[i]);
            for(size_t j=0;j<outputs.size();++j)
                assert(outputs[j].data_ptr()!=out.data_ptr() && bytes(outputs[j])==expected[j]);
            outputs.push_back(out);
        }
        invalid(at::zeros({4,32},at::kFloat),ids);
        invalid(weight.view({128}),ids);
        invalid(weight.transpose(0,1),ids);
        invalid(at::empty({4,32},at::TensorOptions().dtype(at::kHalf).device(at::kMeta)),ids);
        invalid(weight,at::ones({1,1},at::kInt));
        invalid(weight,ids.view({1}));
        invalid(weight,at::ones({2,1},at::kLong));
        invalid(weight,at::empty({1,0},at::kLong));
        invalid(weight,at::ones({1,6},at::kLong).slice(1,0,6,2));
        invalid(weight,at::empty({1,1},at::TensorOptions().dtype(at::kLong).device(at::kMeta)));
        // Invalid token values fail in the actual shared CPU at::embedding;
        // they do not alter current raw weights or retained fresh outputs.
        for(int64_t value:{int64_t(-1),int64_t(4)}) {
            *ids.data_ptr<int64_t>()=value;const int before=weight_reads;
            bool rejected=false;
            try {rpu_qwen3_5_decode_embedding_cpu(weight,ids,-1,false,false);}
            catch(const std::exception&) {rejected=true;}
            assert(rejected && weight_reads==before+1);
        }
        assert(version(weight)==wv && version(ids)==iv);
        for(size_t j=0;j<outputs.size();++j)assert(bytes(outputs[j])==expected[j]);
        // Multi-token private prefill reads every current index/current weight
        // byte. The independent index_select oracle is not a gather replica.
        for(int64_t length:{int64_t(7),int64_t(64),int64_t(65)}) {
            auto input=at::empty({1,length},at::kLong);
            const auto input_version=version(input);
            std::vector<at::Tensor> kept;
            std::vector<std::vector<uint8_t>> saved;
            for(size_t frame:{size_t(0),size_t(0),size_t(2),size_t(0)}) {
                std::memcpy(weight.data_ptr(),weights[frame].data(),weights[frame].size());
                for(int64_t row=0;row<length;++row)
                    input.data_ptr<int64_t>()[row]=(row+frame)%4;
                assert(version(input)==input_version && version(weight)==wv);
                auto oracle=weight.index_select(0,input.view({length})).view({1,length,32});
                const int before=weight_reads;
                auto out=rpu_qwen3_5_decode_embedding_cpu(weight,input,-1,false,false);
                assert(weight_reads==before+1 && out.device().is_cpu());
                assert(out.sizes()==at::IntArrayRef({1,length,32}) && out.is_contiguous());
                assert(bytes(out)==bytes(oracle));
                for(size_t j=0;j<kept.size();++j)
                    assert(kept[j].data_ptr()!=out.data_ptr() && bytes(kept[j])==saved[j]);
                kept.push_back(out);saved.push_back(bytes(out));
            }
            assert(saved.front()==saved.back() && saved[0]!=saved[2]);
            // A legal first N-1 indices cannot hide a bad final index.
            for(int64_t token:{int64_t(-1),int64_t(4)}) {
                input.data_ptr<int64_t>()[length-1]=token;
                const int before=weight_reads;bool rejected=false;
                try {rpu_qwen3_5_decode_embedding_cpu(weight,input,-1,false,false);}
                catch(const std::exception&) {rejected=true;}
                assert(rejected && weight_reads==before+1);
                for(size_t j=0;j<kept.size();++j)assert(bytes(kept[j])==saved[j]);
            }
        }
    }
    std::cout << "actual CPU gather checked\n";
}
'''
    _run_native(tmp_path, cpp.replace("__GATHER__", gather)
                .replace("__WEIGHTS__", _byte_literals(weight_frames))
                .replace("__EXPECTED__", _byte_literals(expected))
                .replace("__TOKENS__", "{" + ",".join(map(str, id_frames)) + "}"), torch)
