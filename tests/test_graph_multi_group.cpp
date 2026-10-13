// Device contract tests; build with RHINOFORGE_BUILD_DEVICE_TESTS.
// Run on an exclusively locked board with the matching Release runtime.

#include "graph/graph_infra.h"
#include "graph/graph_runtime.h"
#include "rpu_spm_pool.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

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

// 每个 group 一个 sig — op_id + branch_key 区分 group_idx。
// 模拟 SigLIP encoder 24 层 / cross_batch=6 → 4 group:每个 group 跑 6 个 layer。
GraphSignature make_group_sig(int group_idx) {
    GraphSignature sig;
    sig.op_id = 3000;
    sig.shapes = {1, 256, 1024};
    sig.dyn_dims = {6};                   // group 内 layer 数
    sig.dtypes = {2};
    sig.flags = {static_cast<uint32_t>(group_idx)};
    sig.branch_key = static_cast<uint64_t>(group_idx);
    sig.segment_key = 0;
    return sig;
}

constexpr int kLayersPerGroup = 3;   // 测试用简化的 group 内 op 数(原型是 6)

// 在 RECORDING 或 REPLAYING 内,给 group 注入 kLayersPerGroup 个 memcpy node。
// dst/src 用 buf_in/buf_out,bytes 固定;模拟一个 group 内的 layer chain。
void simulate_group_layers(RpuKernelGraph& g,
                            void* buf_in, void* buf_out,
                            size_t bytes) {
    for (int L = 0; L < kLayersPerGroup; ++L) {
        MemcpyNodeData m;
        m.dst = buf_out;
        m.src = buf_in;
        m.bytes = bytes;
        m.kind = CopyKind::HOST_MEMCPY;
        g.capture_data(m);
    }
}

// ----------------------------------------------------------------------
// ① K 个 group 顺序 RECORDING -> BUILT,cache 同时存住 K 个 entry
// ----------------------------------------------------------------------
void test_k_groups_sequential_record() {
    section("K=5 groups sequential record (cache holds 5 entries)");
    RpuGraphCache cache;
    constexpr int K = 5;
    constexpr size_t kBytes = 256;

    std::vector<std::vector<uint8_t>> bufs_in(K);
    std::vector<std::vector<uint8_t>> bufs_out(K);
    for (int i = 0; i < K; ++i) {
        bufs_in[i].assign(kBytes, static_cast<uint8_t>(0x10 + i));
        bufs_out[i].assign(kBytes, 0);
    }

    EXPECT(SPM_POOL.defer_depth() == 0, "pre: defer_depth == 0");

    for (int gi = 0; gi < K; ++gi) {
        auto sig = make_group_sig(gi);
        auto& g = cache.get_or_create(sig);
        EXPECT(g.state() == State::PASSTHROUGH,
               "group new graph fresh PASSTHROUGH");
        g.begin(sig);
        EXPECT(g.state() == State::RECORDING, "group begin -> RECORDING");
        EXPECT(SPM_POOL.defer_depth() == 1,
               "group begin -> defer_depth=1 (non-nested)");
        simulate_group_layers(g, bufs_in[gi].data(), bufs_out[gi].data(), kBytes);
        g.end();
        EXPECT(g.state() == State::BUILT, "group end -> BUILT");
        EXPECT(SPM_POOL.defer_depth() == 0, "group end -> defer_depth=0");
    }

    // 5 个 entry 在 cache 里并存,且各自独立 BUILT
    for (int gi = 0; gi < K; ++gi) {
        auto sig = make_group_sig(gi);
        auto* g = cache.lookup(sig);
        char lbl[64];
        std::snprintf(lbl, sizeof(lbl), "group %d entry survives + BUILT", gi);
        EXPECT(g != nullptr && g->state() == State::BUILT, lbl);
    }

    // RECORDING 期 execute 真的跑了 memcpy:bufs_out[gi] = 源头 bufs_in[gi]
    for (int gi = 0; gi < K; ++gi) {
        char lbl[80];
        std::snprintf(lbl, sizeof(lbl),
                      "group %d: bufs_out[%d][0] == 0x%x", gi, gi, 0x10 + gi);
        EXPECT(bufs_out[gi][0] == static_cast<uint8_t>(0x10 + gi), lbl);
    }
}

// ----------------------------------------------------------------------
// ② 第二轮 forward:K 个 group 全 BUILT->REPLAYING->BUILT,buffer 跨轮 rebind
// ----------------------------------------------------------------------
void test_k_groups_replay_with_address_drift() {
    section("K=5 groups replay with fresh buffers each forward");
    RpuGraphCache cache;
    constexpr int K = 5;
    constexpr size_t kBytes = 256;

    // Forward 1:每个 group 用 bufs_in_r1[gi] / bufs_out_r1[gi]
    std::vector<std::vector<uint8_t>> bufs_in_r1(K), bufs_out_r1(K);
    for (int i = 0; i < K; ++i) {
        bufs_in_r1[i].assign(kBytes, static_cast<uint8_t>(0x20 + i));
        bufs_out_r1[i].assign(kBytes, 0);
    }
    for (int gi = 0; gi < K; ++gi) {
        auto sig = make_group_sig(gi);
        auto& g = cache.get_or_create(sig);
        g.begin(sig);
        simulate_group_layers(g, bufs_in_r1[gi].data(), bufs_out_r1[gi].data(),
                              kBytes);
        g.end();
    }
    EXPECT(SPM_POOL.defer_depth() == 0, "after forward 1: defer_depth=0");

    // Forward 2:全新 buffer,期望全部 REPLAYING
    std::vector<std::vector<uint8_t>> bufs_in_r2(K), bufs_out_r2(K);
    for (int i = 0; i < K; ++i) {
        bufs_in_r2[i].assign(kBytes, static_cast<uint8_t>(0xC0 + i));
        bufs_out_r2[i].assign(kBytes, 0);
    }
    for (int gi = 0; gi < K; ++gi) {
        auto sig = make_group_sig(gi);
        auto& g = cache.get_or_create(sig);
        g.begin(sig);
        EXPECT(g.state() == State::REPLAYING, "forward 2 group -> REPLAYING");
        simulate_group_layers(g, bufs_in_r2[gi].data(), bufs_out_r2[gi].data(),
                              kBytes);
        g.end();
        EXPECT(g.state() == State::BUILT, "forward 2 group end -> BUILT");
    }
    EXPECT(SPM_POOL.defer_depth() == 0, "after forward 2: defer_depth=0");

    // 关键:每个 group 的 buf 跨轮独立,REPLAY 用 forward 2 地址
    for (int gi = 0; gi < K; ++gi) {
        char lbl[96];
        std::snprintf(lbl, sizeof(lbl),
                      "forward 2 group %d: bufs_out_r2 == 0x%x (replay 用新地址)",
                      gi, 0xC0 + gi);
        EXPECT(bufs_out_r2[gi][0] == static_cast<uint8_t>(0xC0 + gi), lbl);

        std::snprintf(lbl, sizeof(lbl),
                      "forward 2 group %d: bufs_out_r1 == 0x%x (forward 1 没被踩)",
                      gi, 0x20 + gi);
        EXPECT(bufs_out_r1[gi][0] == static_cast<uint8_t>(0x20 + gi), lbl);
    }
}

// ----------------------------------------------------------------------
// ③ 选择性 forward:只跑 K/2 个 group(模拟 SigLIP 动态 batch / 减少 image),
//    跳过的 group entry 保留 BUILT 不被未访问影响
// ----------------------------------------------------------------------
void test_partial_group_forward() {
    section("partial group forward: skip half, untouched groups stay BUILT");
    RpuGraphCache cache;
    constexpr int K = 4;
    constexpr size_t kBytes = 128;

    // 先 RECORDING 全部 4 个 group
    std::vector<std::vector<uint8_t>> bufs_in(K), bufs_out(K);
    for (int i = 0; i < K; ++i) {
        bufs_in[i].assign(kBytes, static_cast<uint8_t>(0x40 + i));
        bufs_out[i].assign(kBytes, 0);
    }
    for (int gi = 0; gi < K; ++gi) {
        auto sig = make_group_sig(gi);
        auto& g = cache.get_or_create(sig);
        g.begin(sig);
        simulate_group_layers(g, bufs_in[gi].data(), bufs_out[gi].data(),
                              kBytes);
        g.end();
    }
    for (int gi = 0; gi < K; ++gi) {
        auto* g = cache.lookup(make_group_sig(gi));
        EXPECT(g && g->state() == State::BUILT,
               "after warm-up: all 4 groups BUILT");
    }

    // 第二轮:只 REPLAY group 0 和 2,跳过 1 和 3。
    std::vector<std::vector<uint8_t>> bufs_in2(K), bufs_out2(K);
    for (int i = 0; i < K; ++i) {
        bufs_in2[i].assign(kBytes, static_cast<uint8_t>(0x80 + i));
        bufs_out2[i].assign(kBytes, 0);
    }
    for (int gi : {0, 2}) {
        auto sig = make_group_sig(gi);
        auto& g = cache.get_or_create(sig);
        g.begin(sig);
        EXPECT(g.state() == State::REPLAYING,
               "visited group -> REPLAYING");
        simulate_group_layers(g, bufs_in2[gi].data(), bufs_out2[gi].data(),
                              kBytes);
        g.end();
    }

    // 跳过的 group entry 状态:还是 BUILT,没被任何 forward 影响
    for (int gi : {1, 3}) {
        auto* g = cache.lookup(make_group_sig(gi));
        char lbl[80];
        std::snprintf(lbl, sizeof(lbl),
                      "skipped group %d: still BUILT (no side effect)", gi);
        EXPECT(g && g->state() == State::BUILT, lbl);
    }
    // 跳过的 group 的 buffer 没被写入第二轮数据
    for (int gi : {1, 3}) {
        char lbl[80];
        std::snprintf(lbl, sizeof(lbl),
                      "skipped group %d: forward2 buf NOT written (still 0)", gi);
        EXPECT(bufs_out2[gi][0] == 0, lbl);
    }
    // 跑过的 group:用了第二轮地址 + 数据
    for (int gi : {0, 2}) {
        char lbl[80];
        std::snprintf(lbl, sizeof(lbl),
                      "visited group %d: bufs_out2 == 0x%x", gi, 0x80 + gi);
        EXPECT(bufs_out2[gi][0] == static_cast<uint8_t>(0x80 + gi), lbl);
    }

    EXPECT(SPM_POOL.defer_depth() == 0, "after partial forward: defer_depth=0");
}

// ----------------------------------------------------------------------
// ④ 新 sig 追加:warm-up K group 之后,又访问 K+1 个新 sig
//    (模拟 cross_batch_size 切换或 chunk_size 漂移)。
//    旧 K 个 entry 应保留,新 K+1 个 entry 触发 RECORDING。
// ----------------------------------------------------------------------
void test_new_sigs_appended() {
    section("new sigs appended: old groups untouched, new groups RECORDING");
    RpuGraphCache cache;
    constexpr int K = 3;
    constexpr size_t kBytes = 64;

    std::vector<std::vector<uint8_t>> bufs(K * 2 + 1);
    for (auto& b : bufs) b.assign(kBytes, 0);

    // 预热 K 个旧 group sig
    for (int gi = 0; gi < K; ++gi) {
        auto sig = make_group_sig(gi);
        auto& g = cache.get_or_create(sig);
        g.begin(sig);
        simulate_group_layers(g, bufs[gi].data(), bufs[K + gi].data(), kBytes);
        g.end();
    }

    // 加 K+1=4 个新 sig (group_idx 从 100 起避免跟旧 sig 撞)
    for (int gi = 100; gi < 100 + K + 1; ++gi) {
        auto sig = make_group_sig(gi);
        auto& g = cache.get_or_create(sig);
        EXPECT(g.state() == State::PASSTHROUGH,
               "new sig: fresh PASSTHROUGH (not hijacked by old entries)");
        g.begin(sig);
        EXPECT(g.state() == State::RECORDING, "new sig begin -> RECORDING");
        simulate_group_layers(g, bufs[0].data(), bufs[1].data(), kBytes);
        g.end();
        EXPECT(g.state() == State::BUILT, "new sig end -> BUILT");
    }

    // 旧 K 个 entry 仍 BUILT 没被新访问污染
    for (int gi = 0; gi < K; ++gi) {
        auto* g = cache.lookup(make_group_sig(gi));
        char lbl[80];
        std::snprintf(lbl, sizeof(lbl),
                      "old group %d still BUILT after new-sig burst", gi);
        EXPECT(g && g->state() == State::BUILT, lbl);
    }
    // 新 K+1 个 entry 都 BUILT
    for (int gi = 100; gi < 100 + K + 1; ++gi) {
        auto* g = cache.lookup(make_group_sig(gi));
        char lbl[80];
        std::snprintf(lbl, sizeof(lbl), "new group %d BUILT", gi);
        EXPECT(g && g->state() == State::BUILT, lbl);
    }

    EXPECT(SPM_POOL.defer_depth() == 0, "after sigs appended: defer_depth=0");
}

}  // namespace

int main() {
    std::printf("[main] start\n"); std::fflush(stdout);
    test_k_groups_sequential_record();
    test_k_groups_replay_with_address_drift();
    test_partial_group_forward();
    test_new_sigs_appended();

    std::printf("\n=== Summary ===\n");
    std::printf("Passed: %d\n", g_passed);
    std::printf("Failed: %d\n", g_failed);
    return g_failed == 0 ? 0 : 1;
}
