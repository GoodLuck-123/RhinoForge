// Device contract tests; build with RHINOFORGE_BUILD_DEVICE_TESTS.
// Run on an exclusively locked board with the matching Release runtime.

#include "graph/graph_infra.h"
#include "graph/graph_runtime.h"
#include "graph/execution_coordinator.h"
#include "rpu_spm_pool.h"

#include <ATen/core/Dict.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

struct ThrowOnArmedCopy {
    bool* armed = nullptr;

    explicit ThrowOnArmedCopy(bool* value) : armed(value) {}
    ThrowOnArmedCopy(const ThrowOnArmedCopy& other) : armed(other.armed) {
        if (*armed) throw std::runtime_error("injected callback copy failure");
    }
    ThrowOnArmedCopy(ThrowOnArmedCopy&&) noexcept = default;
    void operator()() const {}
};

int g_passed = 0;
int g_failed = 0;
const char* g_section = "(uninit)";

void section(const char* name) {
    g_section = name;
    std::printf("\n--- %s ---\n", name);
}

#define EXPECT(cond, label)                                                   \
    do {                                                                      \
        if (cond) {                                                           \
            ++g_passed;                                                       \
            std::printf("  [PASS] %s\n", label);                              \
        } else {                                                              \
            ++g_failed;                                                       \
            std::fprintf(stderr,                                              \
                         "  [FAIL] %s (%s:%d in %s)\n",                       \
                         label, __FILE__, __LINE__, g_section);               \
        }                                                                    \
    } while (0)

using State = RpuKernelGraph::State;

// no-op 节点工厂:RpuKernelGraph::end() 对 nodes_.empty() 直接落回 PASSTHROUGH
// (设计如此,空 capture 不值得 BUILT),所以测试要在 RECORDING scope 里注入
// 至少一个节点。BranchNodeData 是纯 marker,execute_data_node 对它直接 break
// (graph_runtime_execute.cpp Branch case),不需要任何硬件资源,正好用来撑起
// 非空 graph。(原来用的 FlushNodeData 随 dead flush shim 一起删掉了。)
inline void inject_noop_branch_node(RpuKernelGraph& g) {
    g.capture_data(BranchNodeData{});
}

// 构造两个明显不同的 sig:模拟 KV_FIRST 的 kv_insert phase 和 compute phase
// (op_id 区分 + dyn_dims chunk_size 差异)。
GraphSignature make_sig_kv_insert(int64_t chunk_size) {
    GraphSignature sig;
    sig.op_id = 1001;
    sig.shapes = {1, 32, 8, 64};
    sig.dyn_dims = {chunk_size};
    sig.dtypes = {2};
    sig.flags = {0};
    sig.branch_key = 42;
    sig.segment_key = 1;
    return sig;
}

GraphSignature make_sig_compute(int64_t chunk_size) {
    GraphSignature sig;
    sig.op_id = 1002;
    sig.shapes = {1, 32, 8, 64};
    sig.dyn_dims = {chunk_size};
    sig.dtypes = {2};
    sig.flags = {0};
    sig.branch_key = 42;
    sig.segment_key = 2;
    return sig;
}

GraphSignature make_wall_prefill_sig(int64_t policy_version) {
    GraphSignature sig;
    sig.op_id = 1003;
    sig.shapes = {640, 2048, 320};
    sig.dyn_dims = {36, policy_version};
    sig.dtypes = {2};
    return sig;
}

// ----------------------------------------------------------------------
// ① 顺序两次 capture,两 graph 互相隔离
// ----------------------------------------------------------------------
void test_sequential_two_phase_capture() {
    section("sequential two-phase capture (RECORDING -> BUILT for both)");
    RpuGraphCache cache;
    auto sig_kv = make_sig_kv_insert(128);
    auto sig_co = make_sig_compute(128);

    EXPECT(SPM_POOL.defer_depth() == 0, "pre: SPM_POOL.defer_depth == 0");

    // Phase 1 — kv_insert
    {
        auto& g = cache.get_or_create(sig_kv);
        g.begin(sig_kv);
        EXPECT(g.state() == State::RECORDING, "phase1 begin -> RECORDING");
        EXPECT(SPM_POOL.defer_depth() == 1,
               "phase1 begin -> SPM_POOL defer_depth=1");
        inject_noop_branch_node(g);
        g.end();
        EXPECT(g.state() == State::BUILT, "phase1 end -> BUILT");
        EXPECT(SPM_POOL.defer_depth() == 0,
               "phase1 end -> SPM_POOL defer_depth back to 0");
    }

    // Phase 2 — compute
    {
        auto& g = cache.get_or_create(sig_co);
        EXPECT(g.state() == State::PASSTHROUGH,
               "phase2 new graph: PASSTHROUGH (not leaked from phase1)");
        g.begin(sig_co);
        EXPECT(g.state() == State::RECORDING, "phase2 begin -> RECORDING");
        EXPECT(SPM_POOL.defer_depth() == 1,
               "phase2 begin -> SPM_POOL defer_depth=1");
        inject_noop_branch_node(g);
        g.end();
        EXPECT(g.state() == State::BUILT, "phase2 end -> BUILT");
        EXPECT(SPM_POOL.defer_depth() == 0,
               "phase2 end -> SPM_POOL defer_depth back to 0");
    }

    // 两 entry 都活在 cache 里
    EXPECT(cache.lookup(sig_kv) != nullptr, "phase1 entry survives in cache");
    EXPECT(cache.lookup(sig_co) != nullptr, "phase2 entry survives in cache");

    // 缓存里两 graph 状态独立
    EXPECT(cache.lookup(sig_kv)->state() == State::BUILT,
           "phase1 graph stays BUILT");
    EXPECT(cache.lookup(sig_co)->state() == State::BUILT,
           "phase2 graph stays BUILT");
}

// ----------------------------------------------------------------------
// ② 第二轮 forward 两 graph 都命中 BUILT,走 REPLAYING
// ----------------------------------------------------------------------
void test_second_forward_both_replay() {
    section("second forward: both BUILT -> REPLAYING");
    RpuGraphCache cache;
    auto sig_kv = make_sig_kv_insert(128);
    auto sig_co = make_sig_compute(128);

    // 第一轮:RECORDING -> BUILT
    {
        auto& g = cache.get_or_create(sig_kv);
        g.begin(sig_kv);
        inject_noop_branch_node(g);
        g.end();
    }
    {
        auto& g = cache.get_or_create(sig_co);
        g.begin(sig_co);
        inject_noop_branch_node(g);
        g.end();
    }

    EXPECT(cache.lookup(sig_kv)->state() == State::BUILT,
           "after forward 1: phase1 BUILT");
    EXPECT(cache.lookup(sig_co)->state() == State::BUILT,
           "after forward 1: phase2 BUILT");

    // 第二轮:BUILT -> REPLAYING -> BUILT。REPLAY 期 caller 必须重 capture
    // 相同 kind/序的 node 以推 cursor —— RpuKernelGraph::end() 在 REPLAYING
    // 退出时 TORCH_CHECK(cursor_ == nodes_.size())。
    {
        auto& g = cache.get_or_create(sig_kv);
        g.begin(sig_kv);
        EXPECT(g.state() == State::REPLAYING,
               "forward 2 phase1 begin -> REPLAYING");
        inject_noop_branch_node(g);
        g.end();
        EXPECT(g.state() == State::BUILT,
               "forward 2 phase1 end -> BUILT");
    }
    {
        auto& g = cache.get_or_create(sig_co);
        g.begin(sig_co);
        EXPECT(g.state() == State::REPLAYING,
               "forward 2 phase2 begin -> REPLAYING");
        inject_noop_branch_node(g);
        g.end();
        EXPECT(g.state() == State::BUILT,
               "forward 2 phase2 end -> BUILT");
    }

    EXPECT(SPM_POOL.defer_depth() == 0,
           "after forward 2: SPM_POOL defer_depth == 0");
}

// ----------------------------------------------------------------------
// ③ 单边失效:phase1 sig 变了(chunk_size 128 -> 64)→ phase1 miss + 重录,
//    phase2 sig 没变 → 仍命中 REPLAYING。验证一个 sig miss 不会扯坏另一个的状态。
// ----------------------------------------------------------------------
void test_partial_miss() {
    section("partial miss: phase1 sig changes, phase2 stable");
    RpuGraphCache cache;
    auto sig_kv_a = make_sig_kv_insert(128);
    auto sig_kv_b = make_sig_kv_insert(64);   // chunk_size 不一样
    auto sig_co = make_sig_compute(128);

    // 第一轮:用 chunk_size=128
    {
        auto& g = cache.get_or_create(sig_kv_a);
        g.begin(sig_kv_a);
        inject_noop_branch_node(g);
        g.end();
    }
    {
        auto& g = cache.get_or_create(sig_co);
        g.begin(sig_co);
        inject_noop_branch_node(g);
        g.end();
    }

    // 第二轮:phase1 换 chunk_size=64,phase2 不变
    {
        auto& g = cache.get_or_create(sig_kv_b);
        EXPECT(g.state() == State::PASSTHROUGH,
               "forward 2 phase1 new sig: fresh PASSTHROUGH");
        g.begin(sig_kv_b);
        EXPECT(g.state() == State::RECORDING,
               "forward 2 phase1 new sig -> RECORDING");
        inject_noop_branch_node(g);
        g.end();
        EXPECT(g.state() == State::BUILT,
               "forward 2 phase1 new sig end -> BUILT");
    }
    {
        auto& g = cache.get_or_create(sig_co);
        g.begin(sig_co);
        EXPECT(g.state() == State::REPLAYING,
               "forward 2 phase2 stable sig -> REPLAYING (not affected by phase1 miss)");
        inject_noop_branch_node(g);
        g.end();
    }

    // 老的 chunk_size=128 entry 没被踩掉
    EXPECT(cache.lookup(sig_kv_a) != nullptr,
           "old phase1 entry (chunk=128) survives — LRU 还没启用,容量足");
}

// ----------------------------------------------------------------------
// ④ Wall-OSS prefill policy version 是真实 cache-key consumer。
// ----------------------------------------------------------------------
void test_wall_prefill_policy_version_key() {
    section("Wall-OSS prefill policy version separates cache entries");
    RpuGraphCache cache;
    auto sig_v2 = make_wall_prefill_sig(2);
    auto sig_v3 = make_wall_prefill_sig(3);

    EXPECT(sig_v2 != sig_v3, "policy version changes GraphSignature equality");
    {
        auto& g = cache.get_or_create(sig_v2);
        g.begin(sig_v2);
        inject_noop_branch_node(g);
        g.end();
    }
    EXPECT(cache.lookup(sig_v2) != nullptr, "version 2 entry is cached");
    EXPECT(cache.lookup(sig_v3) == nullptr, "version 3 lookup misses version 2");

    {
        auto& g = cache.get_or_create(sig_v3);
        EXPECT(g.state() == State::PASSTHROUGH,
               "version 3 miss creates a fresh graph");
        g.begin(sig_v3);
        inject_noop_branch_node(g);
        g.end();
    }
    EXPECT(cache.size() == 2, "version 2 and 3 occupy distinct cache entries");
}

// ----------------------------------------------------------------------
// ⑤ 防御性:嵌套 begin(同 graph 重入) 应抛
// ----------------------------------------------------------------------
void test_same_graph_double_begin_rejected() {
    section("defensive: same graph double begin throws");
    RpuGraphCache cache;
    auto sig = make_sig_kv_insert(128);
    auto& g = cache.get_or_create(sig);
    g.begin(sig);
    bool threw = false;
    try {
        g.begin(sig);   // 应该抛
    } catch (const std::exception&) {
        threw = true;
    }
    EXPECT(threw, "double begin on same graph throws");
    if (g.state() == State::RECORDING) {
        g.abort();   // 干净退出
    }
    EXPECT(SPM_POOL.defer_depth() == 0,
           "after abort: SPM_POOL defer_depth == 0");
}

// ----------------------------------------------------------------------
// ⑥ Public process reset invalidates every registered top-level graph.
// ----------------------------------------------------------------------
void test_process_reset_invalidates_registered_graphs() {
    section("process reset invalidates cache and raw top-level graphs");
    RpuGraphCache cache;
    auto sig_cache = make_sig_kv_insert(128);
    auto sig_raw = make_sig_compute(128);

    auto& cached = cache.get_or_create(sig_cache);
    cached.begin(sig_cache);
    inject_noop_branch_node(cached);
    cached.end();

    auto raw = make_registered_rpu_kernel_graph();
    raw->begin(sig_raw);
    inject_noop_branch_node(*raw);
    raw->end();

    EXPECT(cached.state() == State::BUILT, "cached graph built before reset");
    EXPECT(raw->state() == State::BUILT, "raw graph built before reset");

    invalidate_registered_rpu_kernel_graphs();

    EXPECT(cached.state() == State::PASSTHROUGH,
           "cached graph invalidated by process reset");
    EXPECT(raw->state() == State::PASSTHROUGH,
           "raw graph invalidated by process reset");
    EXPECT(cache.size() == 1,
           "reset preserves the cache key slot for deterministic rebuild");

    cached.begin(sig_cache);
    EXPECT(cached.state() == State::RECORDING,
           "first capture after reset records instead of replaying stale graph");
    inject_noop_branch_node(cached);
    cached.end();
    EXPECT(cached.state() == State::BUILT,
           "cached graph rebuilds after process reset");
}

// ----------------------------------------------------------------------
// ⑦ Reset is all-or-nothing when any registered graph scope is active.
// ----------------------------------------------------------------------
void test_process_reset_rejects_active_scope_without_partial_mutation() {
    section("process reset rejects active scope before mutating peers");
    RpuGraphCache cache;
    auto sig_cache = make_sig_kv_insert(128);
    auto sig_active = make_sig_compute(128);

    auto& cached = cache.get_or_create(sig_cache);
    cached.begin(sig_cache);
    inject_noop_branch_node(cached);
    cached.end();

    auto active = make_registered_rpu_kernel_graph();
    active->begin(sig_active);

    bool threw = false;
    try {
        invalidate_registered_rpu_kernel_graphs();
    } catch (const std::exception&) {
        threw = true;
    }
    EXPECT(threw, "reset throws while a registered graph is RECORDING");
    EXPECT(cached.state() == State::BUILT,
           "reset rejection leaves peer graph unchanged");
    EXPECT(active->state() == State::RECORDING,
           "reset rejection leaves active graph scope unchanged");

    active->abort();
    invalidate_registered_rpu_kernel_graphs();
    EXPECT(cached.state() == State::PASSTHROUGH,
           "reset succeeds after active scope aborts");
    EXPECT(SPM_POOL.defer_depth() == 0,
           "reset rejection/abort leaves SPM defer depth balanced");
}

// ----------------------------------------------------------------------
// ⑧ Board-free BUILD topology stamp: Branch nodes execute as no-ops, so this
// covers BUILD, ordinary replay, both fast-skip cursors, and stale generation
// rejection without creating a Queue_t or touching /dev/rpu.
// ----------------------------------------------------------------------
void test_branch_only_op_stream_stamp_contract() {
    section("Branch-only op-stream stamp and BUILD topology identity");
    RpuKernelGraph graph;
    auto sig = make_sig_compute(128);
    sig.segment_key = 0x53504d08;

    graph.begin(sig);
    const auto build_begin = graph.op_stream_stamp();
    EXPECT(build_begin.state == State::RECORDING &&
               build_begin.position == 0 && build_begin.extent == 0 &&
               build_begin.build_generation != 0 &&
               build_begin.signature_identity != 0 &&
               build_begin.signature_segment_key == sig.segment_key,
           "fresh signed BUILD exposes generation/signature at position zero");

    graph.record_branch(0xA1, {}, "body");
    const auto body_end = graph.op_stream_stamp();
    graph.mark_post_fn_cursor();
    graph.record_branch(0xB2, {}, "post");
    const auto build_end = graph.op_stream_stamp();

    const auto body_digest =
        graph.digest_op_stream_window(build_begin, body_end);
    const auto post_digest =
        graph.digest_op_stream_window(body_end, build_end);
    const auto full_digest =
        graph.digest_op_stream_window(build_begin, build_end);
    const auto empty_digest =
        graph.digest_op_stream_window(body_end, body_end);
    EXPECT(body_end.position == 1 && build_end.position == 2,
           "RECORDING stamps advance exactly once per Branch node");
    EXPECT(body_digest.node_count == 1 && post_digest.node_count == 1 &&
               full_digest.node_count == 2 &&
               body_digest.contains(GraphNodeKind::Branch) &&
               post_digest.contains(GraphNodeKind::Branch) &&
               !full_digest.contains(GraphNodeKind::Kernel),
           "window digest reports exact Branch-only topology");
    const std::vector<GraphNodeKind> branch_kinds{GraphNodeKind::Branch};
    const std::vector<GraphNodeKind> full_branch_kinds{
        GraphNodeKind::Branch, GraphNodeKind::Branch};
    EXPECT(body_digest.node_kinds == branch_kinds &&
               post_digest.node_kinds == branch_kinds &&
               full_digest.node_kinds == full_branch_kinds &&
               empty_digest.node_kinds.empty(),
           "actual ordered node vectors cover only the requested BUILD window");
    EXPECT(body_digest.topology_hash != post_digest.topology_hash &&
               full_digest.topology_hash != 0 &&
               empty_digest.node_count == 0 &&
               empty_digest.kind_mask == 0 &&
               empty_digest.topology_hash != 0,
           "topology hash distinguishes Branch keys and seals empty no-op windows");

    const uint64_t first_generation = build_begin.build_generation;
    graph.end();
    const uint64_t first_topology_hash = graph.build_topology_hash();
    const GraphNodeKindMask first_kind_mask = graph.build_node_kind_mask();
    EXPECT(graph.state() == State::BUILT && first_topology_hash != 0 &&
               first_topology_hash == full_digest.topology_hash &&
               first_kind_mask == graph_node_kind_bit(GraphNodeKind::Branch) &&
               graph.built_signature_identity() ==
                   build_begin.signature_identity &&
               graph.is_op_stream_stamp_current(build_begin),
           "successful signed BUILD commits the full topology and keeps stamps current");

    graph.begin(sig);
    const auto replay_begin = graph.op_stream_stamp();
    graph.record_branch(0xA1, {}, "body replay");
    const auto replay_mid = graph.op_stream_stamp();
    graph.record_branch(0xB2, {}, "post replay");
    const auto replay_end = graph.op_stream_stamp();
    EXPECT(replay_begin.state == State::REPLAYING &&
               replay_begin.position == 0 && replay_mid.position == 1 &&
               replay_end.position == 2 &&
               replay_end.build_generation == first_generation,
           "ordinary REPLAY advances cursor while retaining BUILD generation");
    graph.end();

    graph.begin(sig);
    const auto layer_skip_begin = graph.op_stream_stamp();
    graph.skip_layer_body_for_fast_replay();
    const auto layer_skip_post = graph.op_stream_stamp();
    graph.record_branch(0xB2, {}, "post after layer skip");
    const auto layer_skip_end = graph.op_stream_stamp();
    EXPECT(layer_skip_begin.position == 0 && layer_skip_post.position == 1 &&
               layer_skip_end.position == 2,
           "layer-body fast replay lands at post_fn cursor then consumes post");
    graph.end();

    graph.begin(sig);
    const auto full_skip_begin = graph.op_stream_stamp();
    graph.skip_op_stream_for_fast_replay();
    const auto full_skip_end = graph.op_stream_stamp();
    EXPECT(full_skip_begin.position == 0 && full_skip_end.position == 2 &&
               full_skip_end.extent == 2,
           "full fast replay advances directly to graph extent");
    graph.end();

    graph.invalidate();
    EXPECT(graph.state() == State::PASSTHROUGH &&
               graph.build_topology_hash() == 0 &&
               !graph.is_op_stream_stamp_current(build_begin),
           "invalidate clears committed topology and makes old stamps stale");

    graph.begin(sig);
    const auto rebuild_begin = graph.op_stream_stamp();
    graph.record_branch(0xA1, {}, "body rebuilt");
    graph.mark_post_fn_cursor();
    graph.record_branch(0xB2, {}, "post rebuilt");
    const auto rebuild_end = graph.op_stream_stamp();
    const auto rebuild_digest =
        graph.digest_op_stream_window(rebuild_begin, rebuild_end);
    graph.end();
    EXPECT(rebuild_begin.build_generation > first_generation &&
               !graph.is_op_stream_stamp_current(build_begin) &&
               graph.is_op_stream_stamp_current(rebuild_begin),
           "fresh RECORDING generation rejects the prior BUILD stamps");
    EXPECT(rebuild_digest.topology_hash == first_topology_hash &&
               graph.build_topology_hash() == first_topology_hash &&
               rebuild_digest.node_kinds == full_digest.node_kinds,
           "identical A/B/A Branch topology and ordered kinds survive rebuild");
}

// DMA/Barrier are recorded through the production entrypoints, then aborted
// before end(): these topology-only fixtures have no managed device endpoints
// and must never prepare a queue or execute DMA. Branch-only tests above cover
// actual BUILD and REPLAY execution separately.
void test_dma_barrier_op_stream_windows() {
    section("actual DMA/Barrier order, subwindows, and stale BUILD rejection");
    RpuKernelGraph graph;
    auto sig = make_sig_compute(128);
    sig.segment_key = 0x53504d0C;

    graph.begin(sig);
    const auto begin = graph.op_stream_stamp();
    graph.record_dma_fixed(0x1000, 0x2000, 64, 0);
    graph.record_barrier(0, 1);
    const auto vision_end = graph.op_stream_stamp();
    graph.record_dma_fixed(0x2000, 0x3000, 128, 1);
    graph.record_barrier(1, 0);
    const auto handoff_end = graph.op_stream_stamp();
    graph.record_barrier(0, 1);
    graph.record_dma_fixed(0x3000, 0x4000, 256, 0);
    const auto end = graph.op_stream_stamp();

    const auto vision = graph.digest_op_stream_window(begin, vision_end);
    const auto handoff =
        graph.digest_op_stream_window(vision_end, handoff_end);
    const auto text = graph.digest_op_stream_window(handoff_end, end);
    const auto full = graph.digest_op_stream_window(begin, end);
    const std::vector<GraphNodeKind> dma_barrier{
        GraphNodeKind::Dma, GraphNodeKind::Barrier};
    const std::vector<GraphNodeKind> barrier_dma{
        GraphNodeKind::Barrier, GraphNodeKind::Dma};
    const std::vector<GraphNodeKind> actual_order{
        GraphNodeKind::Dma, GraphNodeKind::Barrier,
        GraphNodeKind::Dma, GraphNodeKind::Barrier,
        GraphNodeKind::Barrier, GraphNodeKind::Dma};
    EXPECT(vision.begin == 0 && vision.end == handoff.begin &&
               handoff.end == text.begin && text.end == 6 &&
               full.node_count == 6 && full.node_kinds == actual_order,
           "adjacent actual phase windows retain every DMA and Barrier node");
    EXPECT(vision.node_kinds == dma_barrier &&
               handoff.node_kinds == dma_barrier &&
               text.node_kinds == barrier_dma &&
               text.node_count == vision.node_count &&
               text.kind_mask == vision.kind_mask,
           "equal counts and kind masks cannot replace actual ordered node kinds");
    EXPECT(full.kind_mask == (graph_node_kind_bit(GraphNodeKind::Dma) |
                              graph_node_kind_bit(GraphNodeKind::Barrier)) &&
               full.node_kinds.size() == full.node_count &&
               full.topology_hash != 0 &&
               handoff.topology_hash != text.topology_hash,
           "Barrier remains visible without inventing a Kernel or work packet");
    auto stitched = vision.node_kinds;
    stitched.insert(stitched.end(), handoff.node_kinds.begin(),
                    handoff.node_kinds.end());
    stitched.insert(stitched.end(), text.node_kinds.begin(), text.node_kinds.end());
    EXPECT(stitched == full.node_kinds,
           "phase vectors reconstruct the whole stream with no gaps or suffix");

    bool reverse_rejected = false;
    try {
        (void)graph.digest_op_stream_window(handoff_end, vision_end);
    } catch (const std::exception& error) {
        reverse_rejected = std::string(error.what()).find("invalid node range") !=
                           std::string::npos;
    }
    EXPECT(reverse_rejected,
           "reversed real stamps fail before publishing an ordered vector");
    graph.abort();
    EXPECT(graph.state() == State::PASSTHROUGH &&
               !graph.is_op_stream_stamp_current(begin) &&
               SPM_POOL.defer_depth() == 0,
           "topology-only DMA fixture aborts without queue execution");

    graph.begin(sig);
    bool stale_rejected = false;
    try {
        (void)graph.digest_op_stream_window(begin, end);
    } catch (const std::exception& error) {
        stale_rejected = std::string(error.what()).find("stale BUILD generation") !=
                         std::string::npos;
    }
    EXPECT(stale_rejected,
           "new BUILD rejects old phase stamps despite the same graph and signature");
    graph.abort();
}

// ----------------------------------------------------------------------
// ⑨ Once a BUILD has published callback windows/topology, post-BUILD child
// replacement must fail before mutating the parent. A caller may invalidate and
// rebuild to obtain a fresh generation, but cannot silently stale old stamps.
// ----------------------------------------------------------------------
void test_observed_build_rejects_post_build_child_replacement() {
    section("observed BUILD rejects post-BUILD child replacement");
    RpuKernelGraph graph;
    auto sig = make_sig_compute(64);
    sig.segment_key = 0x53504d09;

    graph.begin(sig);
    const auto build_begin = graph.op_stream_stamp();
    inject_noop_branch_node(graph);
    const auto build_end = graph.op_stream_stamp();
    const auto window =
        graph.digest_op_stream_window(build_begin, build_end);
    graph.end();

    const uint64_t original_hash = graph.build_topology_hash();
    EXPECT(window.contains(GraphNodeKind::Branch),
           "observed parent window retains its Branch topology");

    bool threw = false;
    std::string what_msg;
    try {
        // The observed-topology seal is intentionally checked before child
        // validation, so a null child proves the parent fails without mutation.
        graph.replace_window_with_child(0, 1, nullptr);
    } catch (const std::exception& e) {
        threw = true;
        what_msg = e.what();
    }
    EXPECT(threw && what_msg.find("observed BUILD topology is sealed") !=
                        std::string::npos,
           "observed parent rejects replacement with an invalidate/rebuild gate");
    EXPECT(graph.state() == State::BUILT && graph.graph_size() == 1 &&
               graph.build_topology_hash() == original_hash &&
               graph.is_op_stream_stamp_current(build_begin),
           "rejected replacement leaves nodes, topology hash, and old stamp unchanged");
}

// ----------------------------------------------------------------------
// ⑩ A stamp's owner pointer alone cannot reject same-address object reuse.
// Reconstruct the graph in identical storage with the same signature/topology;
// the process-wide BUILD nonce must still reject the dead object's stamp.
// Every active scope is closed before its object's lifetime ends.
// ----------------------------------------------------------------------
void test_build_generation_rejects_same_address_aba() {
    section("process-unique BUILD generation rejects same-address ABA");
    std::aligned_storage_t<sizeof(RpuKernelGraph), alignof(RpuKernelGraph)>
        graph_storage;
    auto sig = make_sig_compute(32);
    sig.segment_key = 0x53504d0A;

    RpuKernelGraph* first =
        ::new (static_cast<void*>(&graph_storage)) RpuKernelGraph();
    first->begin(sig);
    const GraphOpStreamStamp stale_stamp = first->op_stream_stamp();
    first->record_branch(0xABA, {}, "same-address topology");
    first->end();
    const uint64_t first_generation = stale_stamp.build_generation;
    const uint64_t first_hash = first->build_topology_hash();
    first->~RpuKernelGraph();

    RpuKernelGraph* second =
        ::new (static_cast<void*>(&graph_storage)) RpuKernelGraph();
    second->begin(sig);
    const GraphOpStreamStamp current_stamp = second->op_stream_stamp();
    second->record_branch(0xABA, {}, "same-address topology");
    second->end();
    const uint64_t second_hash = second->build_topology_hash();

    EXPECT(current_stamp.graph == stale_stamp.graph &&
               current_stamp.build_generation != first_generation &&
               !second->is_op_stream_stamp_current(stale_stamp) &&
               second->is_op_stream_stamp_current(current_stamp),
           "same address cannot revive a prior object's BUILD stamp");
    EXPECT(first_hash != 0 && second_hash == first_hash &&
               second->build_node_kind_mask() ==
                   graph_node_kind_bit(GraphNodeKind::Branch),
           "BUILD nonce stays outside deterministic topology identity");
    second->~RpuKernelGraph();
}

// ----------------------------------------------------------------------
// ⑪ Topology observation must reject replay-equality inputs that have no
// pointer-free canonical encoding. A GenericDict is a valid non-Tensor
// IValue, but its pointer-backed equality contract is intentionally outside
// the initial topology schema. The failure occurs before graph execution.
// ----------------------------------------------------------------------
void test_observed_topology_rejects_unsupported_ivalue() {
    section("observed topology rejects unsupported IValue");
    RpuKernelGraph graph;
    auto sig = make_sig_compute(16);
    sig.segment_key = 0x53504d0B;

    graph.begin(sig);
    const GraphOpStreamStamp stale_on_abort = graph.op_stream_stamp();
    HostCallbackNodeData callback;
    c10::impl::GenericDict dict(
        c10::StringType::get(), c10::IntType::get());
    dict.insert(c10::IValue(std::string("key")), c10::IValue(int64_t{1}));
    callback.args_template.emplace_back(std::move(dict));
    graph.capture_data(std::move(callback));

    bool threw = false;
    std::string what_msg;
    try {
        graph.end();
    } catch (const std::exception& e) {
        threw = true;
        what_msg = e.what();
    }
    EXPECT(threw &&
               what_msg.find("unsupported replay-equality IValue") !=
                   std::string::npos,
           "unsupported pointer-bearing IValue fails closed before BUILD");
    graph.abort();
    EXPECT(graph.state() == State::PASSTHROUGH &&
               !graph.is_op_stream_stamp_current(stale_on_abort),
               "abort after topology rejection clears the observed BUILD");
}

// ----------------------------------------------------------------------
// ⑫ Retained-arena publication is explicit.  Existing physical-lease users
// may still BUILD and REPLAY signed Graphs without creating a provisional
// retained root, then release the physical claim exactly as before.  The full
// opaque authority/guard transaction is covered with a real SpmPipelineLease
// by test_spm_pipeline_contract.
// ----------------------------------------------------------------------
void test_physical_graph_retention_is_explicit_opt_in() {
    section("physical Graph retention is explicit opt-in");
    auto physical = RpuExecutionCoordinator::enter_physical(
        "ordinary physical Graph compatibility canary");
    RpuKernelGraph graph;
    auto sig = make_sig_compute(16);
    sig.segment_key = 0x53504d0C;

    graph.begin(sig);
    inject_noop_branch_node(graph);
    graph.end();
    EXPECT(graph.state() == State::BUILT &&
               !graph.has_provisional_physical_build() &&
               !graph.has_retained_physical_arena_guard(),
           "ordinary signed physical BUILD is not provisionally retained");

    graph.begin(sig);
    EXPECT(graph.state() == State::REPLAYING,
           "ordinary physical Graph still enters stable REPLAY");
    inject_noop_branch_node(graph);
    graph.end();
    EXPECT(graph.state() == State::BUILT &&
               !graph.has_provisional_physical_build(),
           "ordinary physical REPLAY remains outside retained registry");

    bool released = true;
    try {
        RpuExecutionCoordinator::exit_physical(
            physical, "ordinary physical Graph compatibility cleanup");
    } catch (const std::exception&) {
        released = false;
    }
    EXPECT(released && !physical.valid(),
           "ordinary physical claim releases without guard teardown");
}

// ----------------------------------------------------------------------
// ⑬ Native hot reconfigure is one same-thread, process-quiescent transaction.
// Staging has no effect; commit invalidates the old generation; an apply error
// rolls every attempted participant back before releasing the claim.
// ----------------------------------------------------------------------
void test_native_hot_reconfigure_transaction() {
    section("native hot-reconfigure transaction");
    RpuKernelGraph graph;
    const auto sig = make_sig_compute(16);
    graph.begin(sig);
    inject_noop_branch_node(graph);
    graph.end();

    int control = 1;
    bool committing_destroy_blocked = false;
    uint64_t token = RpuExecutionCoordinator::begin_reconfigure("test begin");
    RpuExecutionCoordinator::stage_reconfigure(
        token, &control,
        [&] {
            try {
                RpuExecutionCoordinator::
                    check_reconfigure_participant_destroy_allowed(
                        &control, "test committing destroy");
            } catch (const std::exception&) {
                committing_destroy_blocked = true;
            }
            control = 2;
            graph.invalidate();
        },
        [&] { control = 1; graph.invalidate(); },
        "test stage");
    EXPECT(control == 1 && graph.state() == State::BUILT,
           "stage leaves the visible generation unchanged");

    bool staged_destroy_blocked = false;
    try {
        RpuExecutionCoordinator::check_reconfigure_participant_destroy_allowed(
            &control, "test staged destroy");
    } catch (const std::exception&) {
        staged_destroy_blocked = true;
    }
    EXPECT(staged_destroy_blocked,
           "a staged participant cannot be destroyed before retirement");

    bool graph_blocked = false;
    try {
        (void)RpuExecutionCoordinator::enter_graph("test foreign graph");
    } catch (const std::exception&) {
        graph_blocked = true;
    }
    EXPECT(graph_blocked, "transaction excludes graph execution");
    RpuExecutionCoordinator::commit_reconfigure(token, "test commit");
    bool post_commit_destroy_allowed = true;
    try {
        RpuExecutionCoordinator::check_reconfigure_participant_destroy_allowed(
            &control, "test post-commit destroy");
    } catch (const std::exception&) {
        post_commit_destroy_allowed = false;
    }
    EXPECT(control == 2 && graph.state() == State::PASSTHROUGH &&
               committing_destroy_blocked && post_commit_destroy_allowed,
           "commit blocks participant destruction until callbacks retire");

    graph.begin(sig);
    EXPECT(graph.state() == State::RECORDING,
           "first forward after commit rebuilds");
    inject_noop_branch_node(graph);
    graph.end();
    graph.begin(sig);
    EXPECT(graph.state() == State::REPLAYING,
           "second forward after commit replays the rebuilt graph");
    inject_noop_branch_node(graph);
    graph.end();

    token = RpuExecutionCoordinator::begin_reconfigure("test abort begin");
    RpuExecutionCoordinator::stage_reconfigure(
        token, &control, [&] { control = 3; }, [&] { control = 2; },
        "test abort stage");
    RpuExecutionCoordinator::abort_reconfigure(token, "test abort");
    bool post_abort_destroy_allowed = true;
    try {
        RpuExecutionCoordinator::check_reconfigure_participant_destroy_allowed(
            &control, "test post-abort destroy");
    } catch (const std::exception&) {
        post_abort_destroy_allowed = false;
    }
    EXPECT(control == 2 && post_abort_destroy_allowed,
           "abort discards the stage and releases destroy admission");

    token = RpuExecutionCoordinator::begin_reconfigure(
        /*attempt_token=*/101, "test attempt begin");
    bool wrong_attempt_rejected = false;
    try {
        (void)RpuExecutionCoordinator::abort_reconfigure_attempt(
            102, "test wrong attempt abort");
    } catch (const std::exception&) {
        wrong_attempt_rejected = true;
    }
    EXPECT(wrong_attempt_rejected,
           "attempt abort rejects a stale/foreign attempt identity");
    EXPECT(RpuExecutionCoordinator::abort_reconfigure_attempt(
               101, "test exact attempt abort"),
           "exact attempt abort releases a begin whose return was not stored");
    EXPECT(!RpuExecutionCoordinator::abort_reconfigure_attempt(
               101, "test inactive attempt abort"),
           "attempt abort reports an already-inactive transaction");
    token = RpuExecutionCoordinator::begin_reconfigure(
        /*attempt_token=*/103, "test after attempt abort");
    RpuExecutionCoordinator::abort_reconfigure(
        token, "test after attempt abort cleanup");

    int peer = 4;
    token = RpuExecutionCoordinator::begin_reconfigure("test rollback begin");
    RpuExecutionCoordinator::stage_reconfigure(
        token, &control, [&] { control = 5; }, [&] { control = 2; },
        "test rollback first");
    RpuExecutionCoordinator::stage_reconfigure(
        token, &peer,
        [&] { peer = 6; throw std::runtime_error("injected apply failure"); },
        [&] { peer = 4; }, "test rollback second");
    bool commit_failed = false;
    try {
        RpuExecutionCoordinator::commit_reconfigure(token, "test rollback commit");
    } catch (const std::exception&) {
        commit_failed = true;
    }
    EXPECT(commit_failed && control == 2 && peer == 4,
           "failed commit restores every attempted participant");

    token = RpuExecutionCoordinator::begin_reconfigure("test thread begin");
    std::atomic<bool> foreign_rejected{false};
    std::thread foreign([&] {
        try {
            RpuExecutionCoordinator::stage_reconfigure(
                token, &peer, [] {}, [] {}, "foreign stage");
        } catch (const std::exception&) {
            foreign_rejected.store(true);
        }
    });
    foreign.join();
    RpuExecutionCoordinator::abort_reconfigure(token, "test thread abort");
    EXPECT(foreign_rejected.load(), "opaque token is bound to its owner thread");

    int quiesce_calls = 0;
    token = RpuExecutionCoordinator::begin_reconfigure_with_quiesce(
        [&] { ++quiesce_calls; }, "test quiesce begin");
    EXPECT(quiesce_calls == 1,
           "quiesce callback runs once while the transaction owns execution");
    RpuExecutionCoordinator::abort_reconfigure(token, "test quiesce abort");

    bool quiesce_failed = false;
    try {
        (void)RpuExecutionCoordinator::begin_reconfigure_with_quiesce(
            [] { throw std::runtime_error("injected quiesce failure"); },
            "test failed quiesce");
    } catch (const std::exception&) {
        quiesce_failed = true;
    }
    token = RpuExecutionCoordinator::begin_reconfigure(
        "test after failed quiesce");
    RpuExecutionCoordinator::abort_reconfigure(
        token, "test after failed quiesce abort");
    EXPECT(quiesce_failed,
           "failed quiesce releases the process-wide transaction claim");

    bool throw_on_copy = false;
    token = RpuExecutionCoordinator::begin_reconfigure(
        "test callback-copy begin");
    RpuExecutionCoordinator::stage_reconfigure(
        token, &throw_on_copy, ThrowOnArmedCopy(&throw_on_copy), [] {},
        "test callback-copy stage");
    throw_on_copy = true;
    bool callback_copy_failed = false;
    try {
        RpuExecutionCoordinator::commit_reconfigure(
            token, "test callback-copy commit");
    } catch (const std::exception&) {
        callback_copy_failed = true;
    }
    RpuExecutionCoordinator::abort_reconfigure(
        token, "test callback-copy abort");
    token = RpuExecutionCoordinator::begin_reconfigure(
        "test after callback-copy failure");
    RpuExecutionCoordinator::abort_reconfigure(
        token, "test after callback-copy failure abort");
    EXPECT(callback_copy_failed,
           "callback copy failure leaves the transaction abortable");
}

}  // namespace

int main() {
    std::printf("[main] start\n"); std::fflush(stdout);
    test_sequential_two_phase_capture();
    test_second_forward_both_replay();
    test_partial_miss();
    test_wall_prefill_policy_version_key();
    test_same_graph_double_begin_rejected();
    test_process_reset_invalidates_registered_graphs();
    test_process_reset_rejects_active_scope_without_partial_mutation();
    test_branch_only_op_stream_stamp_contract();
    test_dma_barrier_op_stream_windows();
    test_observed_build_rejects_post_build_child_replacement();
    test_build_generation_rejects_same_address_aba();
    test_observed_topology_rejects_unsupported_ivalue();
    test_physical_graph_retention_is_explicit_opt_in();
    test_native_hot_reconfigure_transaction();

    std::printf("\n=== Summary ===\n");
    std::printf("Passed: %d\n", g_passed);
    std::printf("Failed: %d\n", g_failed);
    return g_failed == 0 ? 0 : 1;
}
