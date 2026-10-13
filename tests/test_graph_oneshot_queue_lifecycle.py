"""Execute the one-shot queue lifecycle with a host SDK double."""
from pathlib import Path
import shutil
import subprocess

import pytest

from test_fmb_standalone_layout_rebuild import _definition

ROOT = Path(__file__).resolve().parents[1]


def test_cached_oneshot_queues_release_after_last_use_and_on_failure(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    production = _definition(
        (ROOT / "src/graph/graph_runtime_execute.cpp").read_text(),
        "void RpuKernelGraph::execute_graph_oneshot()",
    )
    queue_cache = _definition(
        (ROOT / "src/core/rpu_kernel_cache.h").read_text(), "class QueueCache {",
    ) + ";"
    program = r'''
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while (0)
namespace c10 {
template<class F> struct ScopeExit {F callback; bool active=true;
 ~ScopeExit(){if(active)callback();} void release(){active=false;}};
template<class F> ScopeExit<F> make_scope_exit(F callback){return {callback};}
}
bool log_at(int){return false;}
int submits=0, fail_submit=0, fail_data=-1, fail_record=0, prepares=0, fail_prepare=0;
int created=0,dirty=0;std::array<int,8> destroyed{};
std::vector<std::pair<char,int>> events;
namespace rhino_lkn {
struct Queue_t {
 int cores=0, discards=0; bool retained=false,built=false;
 explicit Queue_t(uint8_t n):cores(n){++created;}
 ~Queue_t(){if(retained)--dirty;++destroyed[cores-1];}
 uint32_t enqueu_batch(bool wait){assert(wait&&retained&&built);++submits;
  events.push_back({'s',cores});return submits==fail_submit?1:0;}
 uint32_t discard_completed_batch(){
  if(!built)return 1;
  ++discards;retained=false;built=false;--dirty;
  events.push_back({'d',cores});return 0;}
};
}
'''+queue_cache+r'''
struct Tensor {bool defined()const{return false;}size_t numel()const{return 0;}
 size_t element_size()const{return 0;}};
enum class HostCallbackTier {Tier1,Tier2,Tier3};
struct HostCallback {HostCallbackTier tier=HostCallbackTier::Tier1;std::vector<Tensor> output_pool;};
enum class GraphNodeKind {Kernel,HostCallback,Tier3Oneshot};
struct Node {GraphNodeKind kind=GraphNodeKind::Kernel;int id=0;HostCallback hc;
 HostCallback& as_host_callback(){return hc;}};
struct Segment {std::vector<int> core_ids;size_t start_idx,end_idx;};
struct Signature {std::string op_id_str;};
struct Stats {size_t kernel_count=0,data_node_count=0,segment_count=0;
 size_t host_callback_tier1_count=0,host_callback_tier2_count=0,host_callback_tier3_count=0;
 size_t host_callback_output_bytes=0,tier3_oneshot_count=0,hw_batch_submit_total=0;
 std::string non_replayable_reason;};
struct HwPerfConfig {bool enabled=false;std::string output_dir;};
HwPerfConfig rpu_hw_perf_snapshot(){return {};}
struct RpuDmaSubmissionLease {};
void mirror_per_segment_replay_count(Stats&,const std::vector<Segment>&){}
class RpuKernelGraph {public:
 std::vector<Node> nodes_;std::vector<Segment> segments_;Stats last_stats_;
 std::string non_replayable_reason_;std::optional<Signature> pending_signature_;
 Signature built_signature_;
 void begin_execution_stats(){last_stats_={};}
 void build_segments_from_nodes(){}
 void execute_data_node(Node& node){TORCH_CHECK(node.id!=fail_data);}
 uint32_t prepare_segment_queue(Segment&,rhino_lkn::Queue_t& q,bool,
                               RpuDmaSubmissionLease&,bool retain_replay_state){
  assert(!retain_replay_state);if(!q.retained)++dirty;q.retained=true;q.built=false;++prepares;
  if(prepares==2){if(fail_prepare==1)return 1;TORCH_CHECK(fail_prepare!=2);}
  q.built=true;return 0;}
 void record_segment_execution(Segment&,rhino_lkn::Queue_t&,const char*,
                               const std::string&,bool,bool,const std::string&){
  TORCH_CHECK(submits!=fail_record);}
 void execute_graph_oneshot();
};
'''+production+r'''
RpuKernelGraph setup(){
 QueueCache::instance().clear();created=0;dirty=0;destroyed={};events.clear();submits=0;fail_submit=0;fail_data=-1;fail_record=0;prepares=0;fail_prepare=0;
 RpuKernelGraph graph;
 for(int i=0;i<5;++i)graph.nodes_.push_back({i%2?GraphNodeKind::HostCallback:GraphNodeKind::Kernel,i,{}});
 graph.segments_={{{0,1},0,1},{{0,1,2,3},2,3},{{0,1},4,5}};
 return graph;
}
void empty_batches(){assert(dirty==0);}
int main(){
 {
  auto graph=setup();graph.execute_graph_oneshot();auto& cache=QueueCache::instance();
  assert(submits==3&&created==2&&graph.last_stats_.hw_batch_submit_total==3);
  assert(cache.get(2)->discards==1&&cache.get(4)->discards==1);
  const std::vector<std::pair<char,int>> expected={{'s',2},{'s',4},{'d',4},{'s',2},{'d',2}};
  assert(events==expected&&graph.segments_.empty());empty_batches();
 }
 // Preparation (status/exception), submission, host work and bookkeeping failures
 // must retire every retained batch before the caller handles the exception.
 for(int failure=0;failure<5;++failure){
  auto graph=setup();if(failure==0)fail_submit=2;else if(failure==1)fail_data=3;else if(failure==2)fail_record=2;else fail_prepare=failure-2;
  bool rejected=false;try{graph.execute_graph_oneshot();}catch(const std::runtime_error&){rejected=true;}
  assert(rejected);empty_batches();
  assert(destroyed[1]==1);
  if(failure==1)assert(destroyed[3]==0); // Already retired after its last use.
  else assert(destroyed[3]==1);
  const int before=created;QueueCache::instance().get(2);assert(created==before+1);
 }
 // Validate every segment before acquiring a queue or executing a data node.
 for(size_t bad:{size_t(0),size_t(9)}){
  auto graph=setup();graph.segments_[2].core_ids.resize(bad);
  bool rejected=false;try{graph.execute_graph_oneshot();}catch(const std::runtime_error&){rejected=true;}
  assert(rejected&&created==0&&submits==0);empty_batches();
 }
 auto graph=setup();graph.nodes_.clear();graph.execute_graph_oneshot();
 assert(created==0);empty_batches();
}
'''
    source, binary = tmp_path / "oneshot.cpp", tmp_path / "oneshot"
    source.write_text(program)
    build = subprocess.run(
        [compiler, "-std=c++17", "-O0", "-Wall", "-Wextra", str(source), "-o", str(binary)],
        text=True, capture_output=True, timeout=60,
    )
    assert build.returncode == 0, build.stderr
    result = subprocess.run([str(binary)], text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
