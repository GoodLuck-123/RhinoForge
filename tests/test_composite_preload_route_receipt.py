"""Run the real composite preload receipt control flow without an device."""
from pathlib import Path

from test_siglip_attention_dry_live_contract import _function, _run_native


ROOT = Path(__file__).resolve().parents[1]


def test_composite_preload_receipts_are_lexical_subsets(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    impl = (ROOT / "src/core/fused_model_base_impl.h").read_text()
    header = (ROOT / "src/core/fused_model_base.h").read_text()
    runtime = _function(impl, "    struct CompositeOccurrenceRuntime") + ";"
    methods = []
    for return_type, name in (
        ("void", "arm_spm_pipeline_composite_occurrence_runtime"),
        ("void", "finish_spm_pipeline_composite_occurrence_runtime"),
        ("void", "cancel_spm_pipeline_composite_occurrence_runtime"),
        ("bool", "consume_spm_pipeline_composite_preload_follower"),
    ):
        method = _function(base, return_type + " FusedModelBase::" + name)
        methods.append(method.replace("FusedModelBase::", "Probe::")
                       .replace("RpuKernelGraph", "FakeGraph")
                       .replace("GraphOpStreamStamp", "FakeStamp"))
    preload = _function(base, "void FusedModelBase::run_preload_callbacks_(")
    # Preserve the real callback loop and its receipt/bookkeeping, omitting
    # only the pre-existing Graph trace plumbing (covered by pipeline tests).
    preload = preload[preload.index("    // Capture only the delta"):].rsplit("}", 1)[0]
    context = "\n".join(_function(header, marker) for marker in (
        "    std::vector<uint8_t> consumed_physical_route_receipt() const",
        "    void inherit_physical_route_subset(",
    ))
    cancel = _function(base, "void SpmCompositeTraceCoordinator::cancel() noexcept")
    cancel_receipt = _function(cancel, "    if (pimpl_->producer_owner != nullptr &&")
    source = r'''
#include "core/fused_model_base.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <map>
using namespace v3;
struct FakeGraph;
struct FakeStamp {
    enum State { RECORDING, REPLAYING } state = RECORDING;
    uint64_t build_generation=4, signature_identity=5, signature_segment_key=6;
};
struct FakeGraph {
    using State=FakeStamp::State;
    static inline bool active_scope=true;
    static bool has_active() { return active_scope; }
    static FakeGraph& active() { static FakeGraph g; return g; }
    FakeStamp stamp;
    FakeStamp op_stream_stamp() const { return stamp; }
};
struct FakeAllocator {
    uint64_t generation=13;
    uint64_t persistent_generation() const { return generation; }
    uint32_t addr(int, uint32_t offset) const { return offset; }
} SPM_ALLOC;
struct Context {
    FmbPhysicalManifestConsumer physical_manifest_consumer_;
    bool has_complete_physical_manifest() const { return physical_manifest_consumer_.complete(); }
    uint64_t physical_manifest_fingerprint() const { return physical_manifest_consumer_.fingerprint(); }
''' + context + r'''
};
struct Probe {
    struct Impl {
''' + runtime + r'''
        CompositeOccurrenceRuntime composite_occurrence_runtime_;
        struct Snapshot {
            bool valid=true, cpu_dry=false;
            uint64_t layout_hash=11, allocation_hash=12, persistent_generation=13;
        } pipeline_manifest_snapshot_;
        uint64_t cached_preload_gen_=13;
        bool preload_callbacks_dirty_=false;
        bool resident_preload_priming_=false;
        std::vector<std::string> resident_preload_names_;
        std::array<uint64_t,3> composite_hidden_in_src_bases_{{1,2,3}};
        Context ctx_;
        std::map<std::string,uint32_t> persistent_offsets_{{"tables",256}};
        std::vector<std::map<std::string,uint32_t>> persistent_per_layer_offsets_;
    } storage;
    Impl* pimpl_=&storage;
    struct Decl {
        std::string name="tables";
        StorageClass storage=StorageClass::Persistent;
        int per_layer=0;
        std::function<void(Probe&,int,uint32_t)> preload_callback;
    };
    auto spm_fmb_composite_occurrence_capability() const {
        return SpmFmbCompositeOccurrenceCapability::StableThreeInputSlotsSameOwnerPreload;
    }
    uint64_t spm_fmb_composite_occurrence_policy_fingerprint() const { return 14; }
    auto spm_fmb_preload_capability() const {
        return SpmFmbPreloadCapability::PersistentOutsideResolvedWindow;
    }
    void arm_spm_pipeline_composite_occurrence_runtime(uint64_t,size_t,
        SpmFmbCompositeOccurrenceCapability,uint64_t,uint64_t,uint64_t,uint64_t,uint64_t,bool);
    void finish_spm_pipeline_composite_occurrence_runtime(uint64_t,size_t);
    void cancel_spm_pipeline_composite_occurrence_runtime() noexcept;
    bool consume_spm_pipeline_composite_preload_follower();
    // This owner tests composite preload receipts with residency unbound.
    bool resident_preloads_enabled() const { return false; }
    void validate_resident_preloads_() { throw std::runtime_error("unexpected resident owner"); }
    void preload(const std::vector<Decl>& decls) {
        auto invoke_preload_callback = [this](const Decl& d,int layer,uint32_t addr) {
            d.preload_callback(*this,layer,addr);
        };
''' + preload + r'''
    }
    void begin(size_t ordinal,const FmbPhysicalExecutionManifest& manifest,uint64_t identity=20) {
        const auto& s=storage.pipeline_manifest_snapshot_;
        arm_spm_pipeline_composite_occurrence_runtime(identity,ordinal,
            spm_fmb_composite_occurrence_capability(),14,s.layout_hash,s.allocation_hash,
            s.persistent_generation,15,ordinal!=0);
        storage.ctx_.physical_manifest_consumer_.reset(manifest);
        storage.composite_occurrence_runtime_.input_slot_consumed=true;
    }
    void consume(int64_t site) {
        storage.ctx_.physical_manifest_consumer_.consume_route(
            site < 3 ? FmbRouteFamily::MUTABLE_DMA : FmbRouteFamily::LINEAR,
            site,1,0);
    }
    void body() { consume(3); consume(4); }
    void finish(size_t ordinal) {
        storage.ctx_.physical_manifest_consumer_.require_all_consumed();
        finish_spm_pipeline_composite_occurrence_runtime(20,ordinal);
    }
};
struct CoordinatorProbe {
    struct Impl {
        Probe* producer_owner=nullptr;
        std::weak_ptr<uint64_t> producer_owner_generation;
        uint64_t composite_identity=20;
    } state;
    Impl* pimpl_=&state;
    void cancel() {
''' + cancel_receipt + r'''
    }
};
''' + "\n".join(methods) + r'''
template<class F> void rejects(F&& f,const char* contains) {
    bool rejected=false;
    try { f(); } catch(const c10::Error& e) {
        rejected=std::string(e.what()).find(contains)!=std::string::npos;
        if(!rejected) std::cerr << e.what() << "\n";
    }
    assert(rejected);
}
FmbPhysicalExecutionManifest manifest() {
    FmbPhysicalExecutionManifest m;
    m.state=FmbPhysicalManifestState::COMPLETE;
    for(int64_t site=1;site<=4;++site)
        m.routes.push_back({site,site<3?FmbRouteFamily::MUTABLE_DMA:FmbRouteFamily::LINEAR,1,0,{},0});
    std::sort(m.routes.begin(),m.routes.end(),[](const auto& a,const auto& b) {
        return std::tie(a.family,a.site_id,a.invocation)<std::tie(b.family,b.site_id,b.invocation);
    });
    return m;
}
void leader(Probe& p,const FmbPhysicalExecutionManifest& m) {
    p.begin(0,m);
    assert(!p.consume_spm_pipeline_composite_preload_follower());
    p.consume(4); // A real dynamic-config receipt must not leak to followers.
    Probe::Decl d; d.preload_callback=[](Probe& owner,int,uint32_t) {
        owner.consume(1); owner.consume(2);
    };
    p.preload({d}); p.consume(3); p.finish(0);
}
int main() {
    auto m=manifest();
    for(auto state:{FakeStamp::RECORDING,FakeStamp::REPLAYING}) {
        FakeGraph::active().stamp.state=state;
        Probe p; leader(p,m);
        assert((p.storage.composite_occurrence_runtime_.preload_route_receipt==
                std::vector<uint8_t>{0,0,1,1}));
        for(size_t ordinal=1;ordinal<3;++ordinal) {
            p.begin(ordinal,m);
            assert(p.consume_spm_pipeline_composite_preload_follower());
            rejects([&]{p.finish(ordinal);},"was not consumed");
            assert((p.storage.ctx_.physical_manifest_consumer_.consumed_route_receipt()==
                    std::vector<uint8_t>{0,0,1,1}));
            p.body(); p.finish(ordinal);
        }
        assert(!p.storage.composite_occurrence_runtime_.armed);
        assert(!p.storage.composite_occurrence_runtime_.preload_routes_recorded);
        assert(p.storage.composite_occurrence_runtime_.preload_route_receipt.empty());
        // Fresh outer invocation must run a fresh leader; no carry across calls.
        p.begin(1,m);
        rejects([&]{p.consume_spm_pipeline_composite_preload_follower();},"no leader route receipt");
        p.cancel_spm_pipeline_composite_occurrence_runtime();
    }
    FakeGraph::active().stamp=FakeStamp{};
    { Probe p; leader(p,m); auto changed=m; changed.routes[0].selector=2;
      p.begin(1,changed);
      rejects([&]{p.consume_spm_pipeline_composite_preload_follower();},"manifest identity");
      assert((p.storage.ctx_.physical_manifest_consumer_.consumed_route_receipt()==std::vector<uint8_t>(4,0)));
      p.cancel_spm_pipeline_composite_occurrence_runtime(); }
    // Every retained authority must match the immediately preceding occurrence.
    for(int kind=0;kind<11;++kind) {
        Probe p; leader(p,m); auto& r=p.storage.composite_occurrence_runtime_;
        switch(kind) {
            case 0: ++r.composite_identity; break;
            case 1: ++r.producer_local_ordinal; break;
            case 2: ++r.policy_fingerprint; break;
            case 3: ++r.expected_layout_hash; break;
            case 4: ++r.expected_allocation_hash; break;
            case 5: ++r.expected_persistent_generation; break;
            case 6: ++r.composite_occurrence_profile_hash; break;
            case 7: r.graph_identity=nullptr; break;
            case 8: ++r.graph_build_generation; break;
            case 9: ++r.graph_signature_identity; break;
            case 10: ++r.graph_signature_segment_key; break;
        }
        rejects([&]{p.begin(1,m);},"leader Graph/owner authority");
        p.cancel_spm_pipeline_composite_occurrence_runtime();
    }
    { Probe p; leader(p,m); p.begin(1,m); p.storage.preload_callbacks_dirty_=true;
      rejects([&]{p.consume_spm_pipeline_composite_preload_follower();},"current clean");
      p.cancel_spm_pipeline_composite_occurrence_runtime(); }
    { Probe p; leader(p,m); p.begin(1,m); ++SPM_ALLOC.generation;
      rejects([&]{p.consume_spm_pipeline_composite_preload_follower();},"current clean");
      --SPM_ALLOC.generation; p.cancel_spm_pipeline_composite_occurrence_runtime(); }
    { Probe p; leader(p,m); Probe another; another.begin(1,m);
      rejects([&]{another.consume_spm_pipeline_composite_preload_follower();},"no leader route receipt");
      p.cancel_spm_pipeline_composite_occurrence_runtime();
      another.cancel_spm_pipeline_composite_occurrence_runtime(); }
    // A partial callback failure cannot publish a reusable receipt.
    { Probe p; p.begin(0,m); p.consume_spm_pipeline_composite_preload_follower();
      Probe::Decl d; d.preload_callback=[](Probe& owner,int,uint32_t) {
          owner.consume(1); TORCH_CHECK(false,"callback failed");
      };
      rejects([&]{p.preload({d});},"callback failed");
      assert(!p.storage.composite_occurrence_runtime_.preload_routes_recorded);
      p.cancel_spm_pipeline_composite_occurrence_runtime();
      assert(p.storage.composite_occurrence_runtime_.preload_route_receipt.empty()); }
    // Failure between occurrences has no active scope; the coordinator still
    // clears the completed leader's receipt before another Graph can start.
    { Probe p; leader(p,m); CoordinatorProbe c;
      auto generation=std::make_shared<uint64_t>(1);
      c.state.producer_owner=&p; c.state.producer_owner_generation=generation;
      assert(!p.storage.composite_occurrence_runtime_.armed);
      c.cancel(); assert(!p.storage.composite_occurrence_runtime_.preload_routes_recorded);
      assert(p.storage.composite_occurrence_runtime_.preload_route_receipt.empty());
      p.begin(1,m);
      rejects([&]{p.consume_spm_pipeline_composite_preload_follower();},"no leader route receipt");
      p.cancel_spm_pipeline_composite_occurrence_runtime(); }
    // Receipt bit/size/fingerprint validation is atomic; body omissions persist.
    { FmbPhysicalManifestConsumer c(m); const auto zero=c.consumed_route_receipt();
      rejects([&]{c.inherit_consumed_route_subset(c.fingerprint(),{1,1,0,2});},"invalid preload receipt bit");
      assert(c.consumed_route_receipt()==zero);
      rejects([&]{c.inherit_consumed_route_subset(c.fingerprint(),{1});},"manifest identity");
      rejects([&]{c.inherit_consumed_route_subset(c.fingerprint()+1,{0,0,1,1});},"manifest identity");
      assert(c.consumed_route_receipt()==zero);
      c.inherit_consumed_route_subset(c.fingerprint(),{0,0,1,1});
      rejects([&]{c.require_all_consumed();},"was not consumed"); }
    // Legacy CPU-dry/UNSPECIFIED contract remains a receipt-free path.
    { Probe p; FmbPhysicalExecutionManifest legacy;
      for(size_t ordinal=0;ordinal<3;++ordinal) {
          p.begin(ordinal,legacy);
          assert(p.consume_spm_pipeline_composite_preload_follower()==(ordinal!=0));
          p.finish(ordinal);
      } }
    std::cout << "composite preload subset and authority: PASS\n";
}
'''
    result = _run_native(
        tmp_path, source, torch,
        extra_sources=(ROOT / "src/core/fmb_three_stage_chunk_plan.cpp",),
    )
    assert "composite preload subset and authority: PASS" in result


def test_cancel_clears_receipt_between_occurrences():
    base = (ROOT / "src/core/fused_model_base.cpp").read_text()
    cancel = _function(base, "void SpmCompositeTraceCoordinator::cancel() noexcept")
    assert "producer_owner_generation.expired()" in cancel
    assert "cancel_spm_pipeline_composite_occurrence_runtime()" in cancel
    assert ".composite_identity == pimpl_->composite_identity" in cancel
    assert cancel.index("cancel_spm_pipeline_composite_occurrence_runtime()") < cancel.index("pimpl_->completed.clear()")


def test_existing_native_manifest_consumer_regressions(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    _run_native(
        tmp_path, (ROOT / "tests/test_fmb_three_stage_chunk_plan.cpp").read_text(),
        torch, extra_sources=(ROOT / "src/core/fmb_three_stage_chunk_plan.cpp",),
    )
