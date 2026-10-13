"""Exercise FMB's actual manifest-prefix recording and preload-window check."""
from pathlib import Path

from test_siglip_attention_dry_live_contract import _function, _run_native


ROOT = Path(__file__).resolve().parents[1]


def test_composite_manifest_prefix_is_exact_and_does_not_count_as_preload(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    impl = (ROOT / "src/core/fused_model_base_impl.h").read_text()
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    predicate = _function(impl, "        bool composite_preload_prefix_valid() const")
    run = _function(base, "at::Tensor FusedModelBase::run_all_layers_impl(")
    record = _function(run, "    if (defer_physical_manifest_branch) {")
    for marker in ("void SpmCompositeTraceCoordinator::finish_occurrence(",
                   "FusedModelBase::seal_spm_pipeline_build_trace("):
        assert "candidate.composite_preload_prefix_valid()" in _function(base, marker)
    source = r'''
#include <c10/util/Exception.h>
#include <cassert>
#include <cstdint>
#include <iostream>
struct RpuKernelGraph;
struct GraphOpStreamStamp {
    const void* graph=nullptr;
    int state=1;
    uint64_t build_generation=7, signature_identity=8, signature_segment_key=9;
    size_t position=20;
};
struct RpuKernelGraph {
    struct State { static constexpr int RECORDING=1, REPLAYING=2; };
    static RpuKernelGraph& active() { static RpuKernelGraph graph; return graph; }
    GraphOpStreamStamp stamp;
    int emitted_nodes=1;
    bool drift=false;
    int state() const { return stamp.state; }
    GraphOpStreamStamp op_stream_stamp() { stamp.graph=this; return stamp; }
    void record_branch(uint64_t key, int, const char*) { emit(key); }
    void record_physical_manifest_branch(uint64_t key) { emit(key); }
    void emit(uint64_t key) {
        TORCH_CHECK(key != 0, "empty manifest");
        stamp.position+=emitted_nodes;
        if(drift) ++stamp.build_generation;
    }
};
struct Candidate {
    const void* graph_identity=&RpuKernelGraph::active();
    uint64_t graph_build_generation=7, graph_signature_identity=8,
        graph_signature_segment_key=9;
    size_t composite_outer_begin=20, graph_node_begin=20;
    uint64_t manifest_prefix_fingerprint=0;
    size_t manifest_prefix_begin=0, manifest_prefix_end=0;
    bool composite_preload_follower=true, cpu_dry=false;
''' + predicate + r'''
};
struct Probe {
    struct Impl {
        Candidate pipeline_build_trace_;
        struct Context {
            uint64_t physical_manifest_fingerprint() const { return 42; }
        } ctx_;
    } impl;
    Impl* pimpl_=&impl;
    void record() {
        const bool defer_physical_manifest_branch=true, build_trace_run=true;
        bool matching_physical_manifest_replay_branch=false;
''' + record + r'''
    }
};
template<class F> void rejects(F f) {
    bool failed=false;
    try { f(); } catch(const c10::Error&) { failed=true; }
    assert(failed);
}
int main() {
    auto& graph=RpuKernelGraph::active();
    Probe p;
    auto& c=p.impl.pipeline_build_trace_;
    assert(c.composite_preload_prefix_valid()); // original unspecified empty prefix
    p.record();
    c.graph_node_begin=graph.stamp.position;
    assert(c.manifest_prefix_fingerprint==42);
    assert(c.composite_preload_prefix_valid()); // exactly one authenticated node
    c.composite_preload_follower=false;
    assert(!c.composite_preload_prefix_valid()); // metadata is not a leader preload
    ++c.graph_node_begin;
    assert(c.composite_preload_prefix_valid()); // leader includes a real preload
    c.composite_preload_follower=true;
    assert(!c.composite_preload_prefix_valid()); // extra DMA/Kernel/Branch rejected
    --c.graph_node_begin;
    ++c.manifest_prefix_begin;
    assert(!c.composite_preload_prefix_valid());
    --c.manifest_prefix_begin;
    ++c.manifest_prefix_end;
    assert(!c.composite_preload_prefix_valid()); // cannot authenticate two nodes
    --c.manifest_prefix_end;
    rejects([&]{p.record();}); // duplicate receipt
    for(int emitted : {0,2}) {
        graph=RpuKernelGraph{}; graph.emitted_nodes=emitted;
        Probe bad; rejects([&]{bad.record();});
        assert(bad.impl.pipeline_build_trace_.manifest_prefix_fingerprint==0);
    }
    graph=RpuKernelGraph{}; graph.drift=true;
    Probe drift; rejects([&]{drift.record();});
    graph=RpuKernelGraph{}; graph.stamp.state=RpuKernelGraph::State::REPLAYING;
    Probe replay; rejects([&]{replay.record();});
    std::cout << "exact manifest prefix PASS\n";
}
'''
    assert "exact manifest prefix PASS" in _run_native(tmp_path, source, torch)
