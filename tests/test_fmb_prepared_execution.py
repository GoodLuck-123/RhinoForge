"""Exercise immutable descriptor reuse and live SPM admission without hardware."""
from pathlib import Path

from test_siglip_attention_dry_live_contract import _function, _run_native

ROOT = Path(__file__).resolve().parents[1]


def test_prepared_native_plans_preserve_request_owner_and_resource_boundaries(tmp_path, monkeypatch):
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch

    core = (ROOT / "src/core/fused_model_base.cpp").read_text()
    allocator = (ROOT / "src/core/rpu_spm_allocator.cpp").read_text()
    # Compile the actual pure allocator and cache methods. Only a counter is
    # inserted at the allocator entry, so a warm hit proves planning was avoided.
    plan = _function(allocator, "SpmAllocator::AliasedPlan SpmAllocator::plan_temporary_aliased(")
    plan = plan.replace("{", "{\n    ++allocator_plans;", 1)
    source = r'''
#include "core/fused_model_base.h"
#include "core/rpu_spm_allocator.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <stdexcept>
int allocator_plans=0;
template<class T> T Align(T n,T alignment) { return (n+alignment-1)/alignment*alignment; }
'''
    source += plan + "\nnamespace v3 {\n"
    for marker in (
        "struct FixedSpmOverhead", "static FixedSpmOverhead estimate_fixed_overhead(",
        "int64_t detail::estimate_temporary_total(",
        "bool detail::FmbPreparedSpmCostCache::Input::matches(",
        "detail::FmbPreparedSpmCost detail::FmbPreparedSpmCostCache::prepare(",
        "int64_t detail::FmbPreparedSpmCost::available(",
        "bool detail::FmbPreparedSpmCost::fits(",
    ):
        source += _function(core, marker) + (";\n" if marker.startswith("struct ") else "\n")
    source += r'''
struct Owner {
    struct State {
        int64_t planner_cache_epoch_=0;
        FmbPreparedStageCandidateCache prepared_stage_candidates_;
        detail::FmbPreparedSpmCostCache prepared_spm_costs_;
    } state;
    State* pimpl_=&state;
    std::shared_ptr<const FmbPreparedStageCandidate> prepare_stage_candidate(at::IntArrayRef) const;
    void invalidate_planner_cache();
};
'''
    for marker in (
        "std::shared_ptr<const FmbPreparedStageCandidate>\nFusedModelBase::prepare_stage_candidate(",
        "void FusedModelBase::invalidate_planner_cache(",
    ):
        source += _function(core, marker).replace("FusedModelBase::", "Owner::") + "\n"
    source += r'''
}
using namespace v3;
template<class F> void rejects(F operation) {
    bool rejected=false;try {operation();} catch(const c10::Error&) {rejected=true;}
    assert(rejected);
}
std::vector<int64_t> wire(int64_t argument, int64_t rows=32, int64_t position=0) {
    LayoutContext layout;layout.chunk_size=16;layout.max_kv_seq_len=4096;
    auto plan=compose_fmb_default_three_stage_chunk_plan(layout,rows,position,ChunkMode::SEQUENTIAL);
    FmbPhysicalExecutionManifest m;
    m.state=FmbPhysicalManifestState::COMPLETE;
    m.logical_length=rows;m.physical_length=rows;
    m.kv_logical_length=position+rows;m.kv_insert_physical_rows=rows;
    m.graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;
    m.linear_accumulation=FmbLinearAccumulationPolicy::ACC16;
    m.routes={{101,FmbRouteFamily::LINEAR,1,0,{argument},0}};
    return encode_fmb_prefill_stage_candidate({rows,16,16,std::move(plan),std::move(m)});
}
int main() {
    Owner a,b;
    auto words=wire(7);
    auto first=a.prepare_stage_candidate(words);
    auto same_words=words; // New Python/vector storage must hit by complete value.
    assert(first==a.prepare_stage_candidate(same_words));
    assert(first!=b.prepare_stage_candidate(words));
    assert(first->topology()==FmbChunkTopology::SIMC);
    first->validate_request(32,0);
    rejects([&]{first->validate_request(16,0);});
    rejects([&]{first->validate_request(32,1);});
    auto corrupt=words;corrupt.back()^=1; // Preserve every encoded fingerprint.
    rejects([&]{a.prepare_stage_candidate(corrupt);});
    assert(a.state.prepared_stage_candidates_.size()==1);
    // Absolute chunk positions are not in the stage fingerprint. Exact wire
    // comparison must still distinguish a different continuation schedule.
    auto continuation=wire(7,32,8);
    auto second=a.prepare_stage_candidate(continuation);
    assert(second!=first);
    assert(second->topology()==first->topology());
    second->validate_request(32,8);
    rejects([&]{second->validate_request(32,0);});

    FmbPhysicalManifestConsumer consumer;
    consumer.reset_prepared(first);
    assert(&consumer.manifest()==&first->candidate().physical_manifest);
    rejects([&]{consumer.require_all_consumed();});
    consumer.consume_route(FmbRouteFamily::LINEAR,101,1,0,{7});
    consumer.require_all_consumed();
    consumer.reset_prepared(first); // A cache hit must not inherit consumption.
    rejects([&]{consumer.require_all_consumed();});
    rejects([&]{consumer.consume_route(FmbRouteFamily::LINEAR,101,2,0,{7});});
    rejects([&]{consumer.inherit_matching_graph_replay_receipt(first->manifest_fingerprint()+1);});
    consumer.inherit_matching_graph_replay_receipt(first->manifest_fingerprint());
    consumer.require_all_consumed();
    for(int n=0;n<40;++n) a.prepare_stage_candidate(wire(100+n));
    assert(a.state.prepared_stage_candidates_.size()==FmbPreparedStageCandidateCache::kCapacity);
    // Eviction and owner invalidation cannot dangle an active consumer.
    assert(consumer.manifest().routes[0].arguments[0]==7);
    assert(a.prepare_stage_candidate(words)!=first);
    auto before=a.prepare_stage_candidate(words);
    a.invalidate_planner_cache();
    assert(a.state.planner_cache_epoch_==1 && a.state.prepared_stage_candidates_.size()==0);
    assert(a.prepare_stage_candidate(words)!=before);
    assert(consumer.manifest().routes[0].arguments[0]==7);

    auto canonical=decode_fmb_prefill_stage_candidate(wire(9,1));
    canonical.physical_manifest.graph_lifecycle=FmbGraphLifecycle::RETAINED_CACHE;
    canonical.physical_manifest.kv_logical_length=4096;
    auto decode=a.prepare_stage_candidate(encode_fmb_prefill_stage_candidate(canonical));
    assert(decode->topology()==FmbChunkTopology::SISC);
    auto rebased=decode->candidate();
    rebase_fmb_retained_decode_candidate(rebased,123);
    assert(rebased.stage_plan.compute.chunks[0].kv_seq_len==124);
    assert(decode->candidate().stage_plan.compute.chunks[0].kv_seq_len==1);
    rejects([&]{rebase_fmb_retained_decode_candidate(rebased,-1);});

    // Topology is derived from the admitted spans/chunks, independently of
    // continuation position. Switching the immutable descriptor must replace
    // the cached value, including multi-input layouts with just one chunk.
    auto multi=decode_fmb_prefill_stage_candidate(wire(11));
    multi.stage_plan.spans={{0,16,0},{16,16,1}};
    auto mimc=a.prepare_stage_candidate(encode_fmb_prefill_stage_candidate(multi));
    assert(mimc->topology()==FmbChunkTopology::MIMC);
    multi.qkv_chunk_size=multi.compute_chunk_size=32;
    multi.stage_plan.qkv=multi.stage_plan.compute={
        {32,1},{{0,0,32,32}}};
    auto misc=a.prepare_stage_candidate(encode_fmb_prefill_stage_candidate(multi));
    assert(misc!=mimc && misc->topology()==FmbChunkTopology::MISC);
    rejects([&]{misc->validate_request(32,1);});
    assert(mimc->topology()==FmbChunkTopology::MIMC);

    std::vector<BufferDecl> decls={
        {"weight",1024,0,0,StorageClass::Persistent},
        {"scratch_a",4096,0,1,StorageClass::Temp},
        {"scratch_b",8192,2,3,StorageClass::Temp}};
    auto cost=a.state.prepared_spm_costs_.prepare(decls);
    assert(cost.persistent==1024 && cost.temporary==8192);
    int cold_plans=allocator_plans;
    for(int n=0;n<5;++n) a.state.prepared_spm_costs_.prepare(decls);
    assert(allocator_plans==cold_plans);
    assert(cost.fits(80000,false,SpmAllocator::SPM_PLANNING_BUDGET));
    assert(!cost.fits(70000,false,SpmAllocator::SPM_PLANNING_BUDGET));
    // Persistent residency and live resources are evaluated per call.
    assert(!cost.fits(65536+8192,false,SpmAllocator::SPM_PLANNING_BUDGET));
    assert(cost.fits(65536+8192,true,SpmAllocator::SPM_PLANNING_BUDGET));
    auto changed=decls;changed[2].phase_start=1;
    assert(a.state.prepared_spm_costs_.prepare(changed).temporary==12288);
    assert(allocator_plans==cold_plans+1);
    changed=decls;changed[2].alias_of="scratch_a";
    assert(a.state.prepared_spm_costs_.prepare(changed).temporary==4096);
    changed=decls;changed[0].storage=StorageClass::TempPerLayer;changed[0].per_layer=3;
    auto per_layer=a.state.prepared_spm_costs_.prepare(changed);
    assert(per_layer.persistent==0 && per_layer.temp_per_layer==3072);
    for(int n=0;n<40;++n) { changed=decls;changed[1].size=256*(n+1);a.state.prepared_spm_costs_.prepare(changed); }
    assert(a.state.prepared_spm_costs_.size()==v3::detail::FmbPreparedSpmCostCache::kCapacity);
    a.invalidate_planner_cache();
    assert(a.state.prepared_spm_costs_.size()==0);
    std::cout << "prepared descriptors, owner invalidation, route receipts and live SPM budgets: PASS\n";
}
'''
    result = _run_native(tmp_path, source, torch,
                         extra_sources=(ROOT / "src/core/fmb_three_stage_chunk_plan.cpp",))
    assert "live SPM budgets: PASS" in result
