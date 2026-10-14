"""CPU-only execution of the actual scoped Z2 native Graph census collector."""
from pathlib import Path
import shutil
import subprocess

from test_native_kvinsert_cost_domain import _body

ROOT = Path(__file__).resolve().parents[1]


def test_actual_graph_census_keeps_unique_objects_and_direct_child_multiplicity(tmp_path):
    body = _body((ROOT / "src/fused/rpu_qwen3vl_multiview_spm_z2.cpp").read_text(),
                 "retained_graph_stats")
    harness = r'''
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while(0)
enum class GraphNodeKind { Kernel, Dma, ChildGraph };
struct RpuKernelGraph;
struct Child { std::shared_ptr<RpuKernelGraph> child; };
struct Node { GraphNodeKind kind=GraphNodeKind::Kernel;Child data;
 const Child& as_child_graph() const {return data;} };
struct GraphStats {
 uint64_t graph_lifetime_id=1,build_generation=1,execution_ordinal=0;
 size_t child_graph_count=0,kernel_count=0,dma_count=0,segment_count=0;
 uint64_t hwperf_evidence_failure_total=0;
 std::vector<uint64_t> child_graph_lifetime_ids;
 std::vector<int> segment_census;
};
struct RpuKernelGraph {
 enum class State { BUILT, RECORDING };
 State phase=State::BUILT;bool reusable=true;GraphStats stats;std::vector<Node> nodes;
 State state() const {return phase;} bool replayable() const{return reusable;}
 GraphStats debug_stats() const {return stats;}
 struct Plan {const std::vector<Node>* steps;};
 Plan replay_plan() const {return {&nodes};}
};
std::vector<GraphStats> retained_graph_stats(const RpuKernelGraph* root) __BODY__
template<class F> void rejects(F f) {bool caught=false;try{f();}catch(const std::runtime_error&){caught=true;}assert(caught);}
void child(RpuKernelGraph& parent,const std::shared_ptr<RpuKernelGraph>& value){
 parent.nodes.push_back({GraphNodeKind::ChildGraph,{value}});
 ++parent.stats.child_graph_count;parent.stats.child_graph_lifetime_ids.push_back(value->stats.graph_lifetime_id);
}
int main(){
 assert(retained_graph_stats(nullptr).empty());
 auto root=std::make_shared<RpuKernelGraph>();root->stats.graph_lifetime_id=9;
 auto mid=std::make_shared<RpuKernelGraph>();mid->stats.graph_lifetime_id=3;
 auto leaf=std::make_shared<RpuKernelGraph>();leaf->stats.graph_lifetime_id=5;
 leaf->stats.dma_count=3;leaf->stats.segment_count=1;leaf->stats.segment_census={0,3};
 leaf->nodes.push_back({GraphNodeKind::Dma,{}});
 child(*root,mid);child(*root,mid);child(*root,leaf);child(*mid,leaf);
 auto query=[&](){return retained_graph_stats(root.get());};
 auto first=query();assert(first.size()==3);
 assert(first[0].graph_lifetime_id==3 && first[1].graph_lifetime_id==5 && first[2].graph_lifetime_id==9);
 assert(first[2].segment_count==0 && first[2].child_graph_count==3);
 assert(first[2].child_graph_lifetime_ids==std::vector<uint64_t>({3,3,5}));
 assert(first[1].kernel_count==0 && first[1].dma_count==3 && first[1].segment_census==std::vector<int>({0,3}));
 // Copies remain immutable, but another query sees repeated real invocations.
 root->stats.execution_ordinal=2;mid->stats.execution_ordinal=4;leaf->stats.execution_ordinal=6;
 leaf->stats.hwperf_evidence_failure_total=7;
 auto after=query();assert(first[1].execution_ordinal==0 && after[1].execution_ordinal==6);
 assert(after[1].hwperf_evidence_failure_total==7 && after[2].execution_ordinal==2);
 leaf->phase=RpuKernelGraph::State::RECORDING;rejects(query);leaf->phase=RpuKernelGraph::State::BUILT;
 leaf->reusable=false;rejects(query);leaf->reusable=true;
 leaf->stats.build_generation=0;rejects(query);leaf->stats.build_generation=1;
 root->stats.child_graph_lifetime_ids.pop_back();rejects(query);root->stats.child_graph_lifetime_ids.push_back(5);
 root->nodes[0].data.child.reset();rejects(query);root->nodes[0].data.child=mid;
 child(*leaf,root);rejects(query);leaf->nodes.pop_back();leaf->stats.child_graph_count=0;leaf->stats.child_graph_lifetime_ids.clear();
 leaf->stats.graph_lifetime_id=3;rejects(query);leaf->stats.graph_lifetime_id=5;
 auto deep=std::make_shared<RpuKernelGraph>();auto cursor=deep;
 for(int i=0;i<65;++i){auto next=std::make_shared<RpuKernelGraph>();next->stats.graph_lifetime_id=i+100;child(*cursor,next);cursor=next;}
 rejects([&](){retained_graph_stats(deep.get());});
}
'''.replace("__BODY__", body)
    source, binary = tmp_path / "census.cpp", tmp_path / "census"
    source.write_text(harness)
    result = subprocess.run([shutil.which("g++"), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                             str(source), "-o", str(binary)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(binary)], check=True)


def test_scoped_graph_query_reuses_locked_native_authority_and_registered_stats():
    native = (ROOT / "src/fused/rpu_qwen3vl_multiview_spm_z2.cpp").read_text()
    getter = _body(native, "rpu_lingbot2_multiview_spm_z2_graph_stats")
    assert "RpuExecutionCleanupGuard" in getter and "stats_impl(" in getter
    stats = _body(native, "stats_impl")
    assert stats.index("std::lock_guard<std::mutex> lock") < stats.index("retained_graph_stats(")
    assert stats.index("validate_planner_identity(") < stats.index("retained_graph_stats(")
    assert stats.index("installed_child_identity(") < stats.index("retained_graph_stats(")
    assert "CanaryPhase::Reserved || s.phase == CanaryPhase::Retained" in stats
    bindings = (ROOT / "src/graph/graph_pybind.cpp").read_text()
    assert 'm.def("lingbot2_multiview_spm_z2_graph_stats"' in bindings
    assert "return rpu_lingbot2_multiview_spm_z2_graph_stats(" in bindings
    assert 'py::class_<GraphStats>(m, "GraphStats")' in bindings
