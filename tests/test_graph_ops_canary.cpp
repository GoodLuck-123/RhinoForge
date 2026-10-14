// Device contract tests; build with RHINOFORGE_BUILD_DEVICE_TESTS.
// Run on an exclusively locked board with the matching Release runtime.

#include "graph/graph_infra.h"
#include "graph/graph_runtime.h"
#include "rpu_spm_pool.h"

#include <cstdio>
#include <cstdlib>

namespace {

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
using ::rhino_lkn::read_write;
using ::rhino_lkn::kStride32B;

GraphSignature make_sig(int64_t op_id) {
    GraphSignature sig;
    sig.op_id = op_id;
    sig.shapes = {1, 64};
    sig.dyn_dims = {64};
    sig.dtypes = {2};
    sig.flags = {0};
    sig.branch_key = 0;
    sig.segment_key = 0;
    return sig;
}

// ----------------------------------------------------------------------
// ① graph_acquire_local_spm 三态稳定性
// ----------------------------------------------------------------------
void test_local_spm_lifecycle() {
    section("graph_acquire_local_spm: RECORDING captures, REPLAYING returns same buf");

    // P0 — PASSTHROUGH 路径:acquire/release 等价 new/delete,各次 acquire 都
    // 是独立堆分配,地址不该跨 acquire 复用。
    EXPECT(RpuKernelGraph::active().state() == State::PASSTHROUGH,
           "pre: outside any begin/end -> PASSTHROUGH");
    auto* p0 = graph_acquire_local_spm(1024, read_write, kStride32B, 2, 0);
    auto* p1 = graph_acquire_local_spm(1024, read_write, kStride32B, 2, 1);
    EXPECT(p0 != nullptr && p1 != nullptr, "PASSTHROUGH: both acquires non-null");
    EXPECT(p0 != p1, "PASSTHROUGH: two acquires distinct (new/delete path)");
    graph_release_local_spm(p0);
    graph_release_local_spm(p1);

    // RECORDING — 三只 buf 进 pool,end 时 owned by graph;reset_cursor 后
    // REPLAYING 同序 acquire 必须拿回同一只(地址匹配)。
    RpuGraphCache cache;
    auto sig = make_sig(/*op_id=*/4001);
    auto& g = cache.get_or_create(sig);

    g.begin(sig);
    EXPECT(g.state() == State::RECORDING, "begin(sig) -> RECORDING");

    auto* rec_a = graph_acquire_local_spm(2048, read_write, kStride32B, 2, 0);
    auto* rec_b = graph_acquire_local_spm(4096, read_write, kStride32B, 2, 1);
    auto* rec_c = graph_acquire_local_spm(8192, read_write, kStride32B, 2, 2);

    EXPECT(rec_a != nullptr && rec_b != nullptr && rec_c != nullptr,
           "RECORDING: 3 acquires non-null");
    EXPECT(rec_a != rec_b && rec_b != rec_c && rec_a != rec_c,
           "RECORDING: 3 acquires distinct slots");

    void* cpu_a = rec_a->get_cpu_ptr();
    void* cpu_b = rec_b->get_cpu_ptr();
    void* cpu_c = rec_c->get_cpu_ptr();

    // release 在 RECORDING 期是 no-op(所有权归 pool)。
    graph_release_local_spm(rec_a);
    graph_release_local_spm(rec_b);
    graph_release_local_spm(rec_c);

    // 撑非空 graph,end 才会落 BUILT(空 graph 会回 PASSTHROUGH)。
    g.capture_data(BranchNodeData{});
    g.end();
    EXPECT(g.state() == State::BUILT, "end() with non-empty graph -> BUILT");

    // 第二次同 sig:命中 BUILT -> REPLAYING,reset_cursor 后 acquire 必须按
    // RECORDING 期顺序回到 rec_a / rec_b / rec_c。
    g.begin(sig);
    EXPECT(g.state() == State::REPLAYING, "begin(sig) second time -> REPLAYING");

    auto* rep_a = graph_acquire_local_spm(2048, read_write, kStride32B, 2, 0);
    auto* rep_b = graph_acquire_local_spm(4096, read_write, kStride32B, 2, 1);
    auto* rep_c = graph_acquire_local_spm(8192, read_write, kStride32B, 2, 2);

    EXPECT(rep_a == rec_a, "REPLAYING acquire #1 == RECORDING acquire #1 (slot 0)");
    EXPECT(rep_b == rec_b, "REPLAYING acquire #2 == RECORDING acquire #2 (slot 1)");
    EXPECT(rep_c == rec_c, "REPLAYING acquire #3 == RECORDING acquire #3 (slot 2)");
    EXPECT(rep_a->get_cpu_ptr() == cpu_a, "REPLAYING get_cpu_ptr #1 stable");
    EXPECT(rep_b->get_cpu_ptr() == cpu_b, "REPLAYING get_cpu_ptr #2 stable");
    EXPECT(rep_c->get_cpu_ptr() == cpu_c, "REPLAYING get_cpu_ptr #3 stable");

    graph_release_local_spm(rep_a);
    graph_release_local_spm(rep_b);
    graph_release_local_spm(rep_c);
    // REPLAYING 期也要走一遍 capture_data 推进 cursor,跟 RECORDING 期录的
    // 节点对账;否则 end() 会因为 cursor != nodes_.size() 抛。
    g.capture_data(BranchNodeData{});
    g.end();
    EXPECT(g.state() == State::BUILT, "REPLAYING end -> BUILT");
    EXPECT(SPM_POOL.defer_depth() == 0, "defer_depth back to 0 after replay end");
}

// ----------------------------------------------------------------------
// ② graph_acquire_global_spm 三态稳定性(rpu_sdpa 路径)
// ----------------------------------------------------------------------
void test_global_spm_lifecycle() {
    section("graph_acquire_global_spm: same contract as local, GlobalSPM_t");

    EXPECT(RpuKernelGraph::active().state() == State::PASSTHROUGH,
           "pre: PASSTHROUGH");
    auto* p0 = graph_acquire_global_spm(1024, read_write, kStride32B, 2, 0);
    auto* p1 = graph_acquire_global_spm(1024, read_write, kStride32B, 2, 0);
    EXPECT(p0 != nullptr && p1 != nullptr && p0 != p1,
           "PASSTHROUGH: distinct heap allocations");
    graph_release_global_spm(p0);
    graph_release_global_spm(p1);

    RpuGraphCache cache;
    auto sig = make_sig(/*op_id=*/4002);
    auto& g = cache.get_or_create(sig);

    g.begin(sig);
    auto* rec_a = graph_acquire_global_spm(1024, read_write, kStride32B, 2, 0);
    auto* rec_b = graph_acquire_global_spm(2048, read_write, kStride32B, 2, 0);
    void* cpu_a = rec_a->get_cpu_ptr();
    void* cpu_b = rec_b->get_cpu_ptr();
    graph_release_global_spm(rec_a);
    graph_release_global_spm(rec_b);
    g.capture_data(BranchNodeData{});
    g.end();
    EXPECT(g.state() == State::BUILT, "global RECORDING end -> BUILT");

    g.begin(sig);
    EXPECT(g.state() == State::REPLAYING, "global second begin -> REPLAYING");
    auto* rep_a = graph_acquire_global_spm(1024, read_write, kStride32B, 2, 0);
    auto* rep_b = graph_acquire_global_spm(2048, read_write, kStride32B, 2, 0);
    EXPECT(rep_a == rec_a, "global REPLAYING #1 == RECORDING #1");
    EXPECT(rep_b == rec_b, "global REPLAYING #2 == RECORDING #2");
    EXPECT(rep_a->get_cpu_ptr() == cpu_a, "global REPLAYING get_cpu_ptr #1 stable");
    EXPECT(rep_b->get_cpu_ptr() == cpu_b, "global REPLAYING get_cpu_ptr #2 stable");
    graph_release_global_spm(rep_a);
    graph_release_global_spm(rep_b);
    g.capture_data(BranchNodeData{});
    g.end();
    EXPECT(SPM_POOL.defer_depth() == 0, "defer_depth back to 0");
}

// ----------------------------------------------------------------------
// ③ Mixed local + global in 同一 graph,各自 pool 独立 cursor
// ----------------------------------------------------------------------
void test_mixed_pool_cursors() {
    section("mixed local+global pool: each pool has its own cursor");

    RpuGraphCache cache;
    auto sig = make_sig(/*op_id=*/4003);
    auto& g = cache.get_or_create(sig);

    g.begin(sig);
    auto* L1 = graph_acquire_local_spm(1024, read_write, kStride32B, 2, 0);
    auto* G1 = graph_acquire_global_spm(2048, read_write, kStride32B, 2, 0);
    auto* L2 = graph_acquire_local_spm(4096, read_write, kStride32B, 2, 1);
    auto* G2 = graph_acquire_global_spm(8192, read_write, kStride32B, 2, 1);
    graph_release_local_spm(L1);
    graph_release_local_spm(L2);
    graph_release_global_spm(G1);
    graph_release_global_spm(G2);
    g.capture_data(BranchNodeData{});
    g.end();
    EXPECT(g.state() == State::BUILT, "mixed pool end -> BUILT");

    g.begin(sig);
    auto* L1r = graph_acquire_local_spm(1024, read_write, kStride32B, 2, 0);
    auto* G1r = graph_acquire_global_spm(2048, read_write, kStride32B, 2, 0);
    auto* L2r = graph_acquire_local_spm(4096, read_write, kStride32B, 2, 1);
    auto* G2r = graph_acquire_global_spm(8192, read_write, kStride32B, 2, 1);
    EXPECT(L1r == L1 && L2r == L2,
           "local pool cursor not contaminated by global acquires");
    EXPECT(G1r == G1 && G2r == G2,
           "global pool cursor not contaminated by local acquires");
    graph_release_local_spm(L1r);
    graph_release_local_spm(L2r);
    graph_release_global_spm(G1r);
    graph_release_global_spm(G2r);
    g.capture_data(BranchNodeData{});
    g.end();
}

// ----------------------------------------------------------------------
// ④ GET_QUEUE 在 PASSTHROUGH/RECORDING/REPLAYING 都返回 RpuQueue proxy
// ----------------------------------------------------------------------
//
// 故意不测 GET_KERNEL:KernelCache 是 lazy 初始化,canary 在 init() 之前跑
// 会拿到 nullptr 然后 get_kernel_reset 段错。GET_KERNEL 的 graph-aware 改造
// 由 P1.4 commit 的全量编译通过 + P5.5 Path B token-match 覆盖。
void test_get_queue_three_states() {
    section("GET_QUEUE returns RpuQueue proxy in all 3 states");

    // PASSTHROUGH
    {
        EXPECT(RpuKernelGraph::active().state() == State::PASSTHROUGH,
               "pre: PASSTHROUGH");
        auto* wq = GET_QUEUE(8);
        EXPECT(wq != nullptr, "PASSTHROUGH: GET_QUEUE(8) non-null");
        wq->set_broadcast_mode(true);
        wq->set_flush_icache(false);
    }

    RpuGraphCache cache;
    auto sig = make_sig(/*op_id=*/4004);
    auto& g = cache.get_or_create(sig);

    // RECORDING
    g.begin(sig);
    {
        EXPECT(g.state() == State::RECORDING, "begin(sig) -> RECORDING");
        auto* wq = GET_QUEUE(8);
        EXPECT(wq != nullptr, "RECORDING: GET_QUEUE(8) non-null");
        wq->set_broadcast_mode(true);
        wq->set_flush_icache(false);
        // 同一 core_num 下重复 GET_QUEUE 应返回同一 RpuQueue 实例(单例)
        auto* wq2 = GET_QUEUE(8);
        EXPECT(wq == wq2, "RECORDING: GET_QUEUE(8) returns same singleton");
    }
    g.capture_data(BranchNodeData{});
    g.end();

    // REPLAYING
    g.begin(sig);
    {
        EXPECT(g.state() == State::REPLAYING, "second begin -> REPLAYING");
        auto* wq = GET_QUEUE(8);
        EXPECT(wq != nullptr, "REPLAYING: GET_QUEUE(8) non-null");
        wq->set_broadcast_mode(true);
    }
    g.capture_data(BranchNodeData{});
    g.end();
    EXPECT(SPM_POOL.defer_depth() == 0, "defer_depth back to 0 after GET_QUEUE test");
}

}  // namespace

int main() {
    std::printf("[main] start\n");
    test_local_spm_lifecycle();
    test_global_spm_lifecycle();
    test_mixed_pool_cursors();
    test_get_queue_three_states();

    std::printf("\n=== Summary ===\n");
    std::printf("Passed: %d\n", g_passed);
    std::printf("Failed: %d\n", g_failed);
    return g_failed == 0 ? 0 : 1;
}
