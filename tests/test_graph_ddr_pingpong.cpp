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

GraphSignature make_sig() {
    GraphSignature sig;
    sig.op_id = 2001;       // DDR_PINGPONG forward 用一个 sig
    sig.shapes = {1, 128, 1024};   // batch, seq, hidden
    sig.dyn_dims = {128};          // chunk_size
    sig.dtypes = {2};
    sig.flags = {static_cast<uint32_t>(2)};   // 2 = DDR_PINGPONG (cosmetic)
    sig.branch_key = 99;
    sig.segment_key = 0;
    return sig;
}

// 模拟一次 forward 内 N 层 DDR_PINGPONG 链:
//   L=0: in=A, out=B   →  memcpy(B, A, bytes)
//   L=1: in=B, out=A   →  memcpy(A, B, bytes)
//   L=2: in=A, out=B
//   L=3: in=B, out=A
// 调用者负责 begin/end RECORDING 或 REPLAYING scope。
constexpr int kNumLayers = 4;

void simulate_pingpong_forward(RpuKernelGraph& g,
                                void* bufA, void* bufB,
                                size_t bytes) {
    for (int L = 0; L < kNumLayers; ++L) {
        void* in  = (L % 2 == 0) ? bufA : bufB;
        void* out = (L % 2 == 0) ? bufB : bufA;
        MemcpyNodeData m;
        m.dst = out;
        m.src = in;
        m.bytes = bytes;
        m.kind = CopyKind::HOST_MEMCPY;
        g.capture_data(m);
    }
}

// ----------------------------------------------------------------------
// ① 单次 forward:RECORDING -> BUILT,memcpy 链顺序正确
// ----------------------------------------------------------------------
void test_record_pingpong_chain() {
    section("record pingpong chain (RECORDING -> BUILT)");
    RpuGraphCache cache;
    auto sig = make_sig();

    constexpr size_t kBytes = 1024;
    std::vector<uint8_t> bufA(kBytes, 0xAA);  // 初始化 bufA = 0xAA
    std::vector<uint8_t> bufB(kBytes, 0xBB);  // bufB = 0xBB

    auto& g = cache.get_or_create(sig);
    g.begin(sig);
    EXPECT(g.state() == State::RECORDING, "begin -> RECORDING");
    simulate_pingpong_forward(g, bufA.data(), bufB.data(), kBytes);
    g.end();
    EXPECT(g.state() == State::BUILT, "end -> BUILT (graph has 4 memcpy nodes)");

    // 4 层完了:bufA 应被 bufB 的内容(其实也是 bufA 内容,因为链路头是 bufA)
    // 覆盖。最终 bufA = bufB = 0xAA。
    EXPECT(bufA[0] == 0xAA, "after RECORDING execute: bufA[0] == 0xAA (chain ends in A)");
    EXPECT(bufB[0] == 0xAA, "after RECORDING execute: bufB[0] == 0xAA (链路头是 A)");
    EXPECT(SPM_POOL.defer_depth() == 0, "defer_depth back to 0");
}

// ----------------------------------------------------------------------
// ② 跨 forward 地址漂移:第二轮换全新 bufA2/bufB2,replay 用新地址
// ----------------------------------------------------------------------
void test_replay_with_address_drift() {
    section("replay with bufA/bufB address drift");
    RpuGraphCache cache;
    auto sig = make_sig();
    constexpr size_t kBytes = 1024;

    // Forward 1:用 bufA1/bufB1
    std::vector<uint8_t> bufA1(kBytes, 0x11);
    std::vector<uint8_t> bufB1(kBytes, 0x22);
    {
        auto& g = cache.get_or_create(sig);
        g.begin(sig);
        simulate_pingpong_forward(g, bufA1.data(), bufB1.data(), kBytes);
        g.end();
        EXPECT(g.state() == State::BUILT, "forward 1 BUILT");
    }
    EXPECT(bufA1[0] == 0x11, "forward 1 final bufA1 == 0x11 (源头)");

    // Forward 2:用完全新分配的 bufA2/bufB2,地址不同
    std::vector<uint8_t> bufA2(kBytes, 0xCD);  // 新数据
    std::vector<uint8_t> bufB2(kBytes, 0xEF);
    EXPECT(bufA1.data() != bufA2.data(),
           "bufA1 / bufA2 拿到不同堆地址(模拟跨 forward 的 at::empty 漂移)");

    {
        auto& g = cache.get_or_create(sig);
        g.begin(sig);
        EXPECT(g.state() == State::REPLAYING, "forward 2 begin -> REPLAYING");
        simulate_pingpong_forward(g, bufA2.data(), bufB2.data(), kBytes);
        g.end();
        EXPECT(g.state() == State::BUILT, "forward 2 end -> BUILT");
    }

    // 关键校验:forward 2 实际 ::memcpy 跑的应该是 bufA2/bufB2,不是 bufA1/bufB1。
    // 若 capture_data 没正确覆写 rec.dst/rec.src,execute 会拿陈旧 bufA1 指针
    // memcpy(可能 use-after-free,这里 vector 还活着所以不崩,但内容就错了)。
    EXPECT(bufA2[0] == 0xCD,
           "forward 2 final bufA2 == 0xCD (replay 用新地址,不动旧 bufA1)");
    EXPECT(bufB2[0] == 0xCD,
           "forward 2 final bufB2 == 0xCD (链路头 bufA2 覆盖到 bufB2)");
    EXPECT(bufA1[0] == 0x11,
           "forward 1 buffer bufA1 没被 forward 2 误碰 (no cross-forward leak)");
    EXPECT(bufB1[0] == 0x11,
           "forward 1 buffer bufB1 没被 forward 2 误碰");
}

// ----------------------------------------------------------------------
// ③ 多次 replay 各自地址漂移
// ----------------------------------------------------------------------
void test_multiple_replays_each_with_new_buffers() {
    section("multiple forwards, each with fresh bufA/bufB");
    RpuGraphCache cache;
    auto sig = make_sig();
    constexpr size_t kBytes = 256;

    // 三轮 forward,每轮的 bufA/bufB 都不同
    uint8_t fingerprints[3] = {0x10, 0x20, 0x30};

    for (int round = 0; round < 3; ++round) {
        std::vector<uint8_t> bufA(kBytes, fingerprints[round]);
        std::vector<uint8_t> bufB(kBytes, 0);

        auto& g = cache.get_or_create(sig);
        g.begin(sig);
        if (round == 0) {
            EXPECT(g.state() == State::RECORDING,
                   "round 0 first time -> RECORDING");
        } else {
            EXPECT(g.state() == State::REPLAYING,
                   "round 1+ -> REPLAYING");
        }
        simulate_pingpong_forward(g, bufA.data(), bufB.data(), kBytes);
        g.end();

        // 链路头 bufA 的 fingerprint 在四层 memcpy 后应出现在两个 buf 里
        char label[64];
        std::snprintf(label, sizeof(label),
                      "round %d: bufA[0] == fingerprint 0x%x", round,
                      fingerprints[round]);
        EXPECT(bufA[0] == fingerprints[round], label);
        std::snprintf(label, sizeof(label),
                      "round %d: bufB[0] == fingerprint 0x%x", round,
                      fingerprints[round]);
        EXPECT(bufB[0] == fingerprints[round], label);
    }
}

// ----------------------------------------------------------------------
// ④ 防御:同 cursor 上 RECORDING 录 bytes=1024,REPLAY 给 bytes=512 → 抛
// ----------------------------------------------------------------------
void test_bytes_mismatch_rejected() {
    section("defensive: memcpy bytes mismatch on replay throws");
    RpuGraphCache cache;
    auto sig = make_sig();

    constexpr size_t kBytes1 = 1024;
    std::vector<uint8_t> bufA(kBytes1, 0);
    std::vector<uint8_t> bufB(kBytes1, 0);

    {
        auto& g = cache.get_or_create(sig);
        g.begin(sig);
        simulate_pingpong_forward(g, bufA.data(), bufB.data(), kBytes1);
        g.end();
        EXPECT(g.state() == State::BUILT, "forward 1 BUILT");
    }

    // Forward 2:故意把 bytes 从 1024 改成 512
    bool threw = false;
    std::string what_msg;
    try {
        auto& g = cache.get_or_create(sig);
        g.begin(sig);
        // 第一个节点就给错 bytes
        MemcpyNodeData m;
        m.dst = bufB.data();
        m.src = bufA.data();
        m.bytes = 512;             // RECORDING 时是 1024
        m.kind = CopyKind::HOST_MEMCPY;
        g.capture_data(m);
    } catch (const std::exception& e) {
        threw = true;
        what_msg = e.what();
    }
    EXPECT(threw, "bytes mismatch throws on replay");
    EXPECT(what_msg.find("bytes mismatch") != std::string::npos,
           "error message mentions 'bytes mismatch'");

    // 抛过之后 graph 仍在 REPLAYING,要 abort 让 active_stack / defer_depth 清干净
    auto& g = cache.get_or_create(sig);
    if (g.state() == State::REPLAYING) {
        g.abort();
    }
    EXPECT(SPM_POOL.defer_depth() == 0, "after abort: defer_depth == 0");
}

}  // namespace

int main() {
    std::printf("[main] start\n"); std::fflush(stdout);
    test_record_pingpong_chain();
    test_replay_with_address_drift();
    test_multiple_replays_each_with_new_buffers();
    test_bytes_mismatch_rejected();

    std::printf("\n=== Summary ===\n");
    std::printf("Passed: %d\n", g_passed);
    std::printf("Failed: %d\n", g_failed);
    return g_failed == 0 ? 0 : 1;
}
