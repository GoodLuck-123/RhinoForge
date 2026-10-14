"""Execute the actual C224 owner-ring/norm packet and fail-closed boundaries."""
from pathlib import Path

from test_wall_base_paired_kv import block, compile_run


ROOT = Path(__file__).resolve().parents[1]


def test_actual_prefill_owner_norm_packet_and_failures(tmp_path):
    runtime = (ROOT / 'src/core/rpu_runtime_extras.cpp').read_text()
    launch = '\n'.join(block(runtime, marker) for marker in (
        'static void launch_wall_base_owner_ring_norm(',
        'void rpu_launch_wall_base_prefill_o_owner_ring_norm_m224_kernel(',
    ))
    cache = (ROOT / 'src/core/rpu_kernel_cache.h').read_text()
    pieces = {
        'KERNEL_IDS': block(cache, 'enum class KernelId : uint16_t') + ';',
        'KERNEL_NAMES': block(cache, 'static constexpr const char* KERNEL_ID_NAMES[]') + ';',
        'OPTIONAL': block((ROOT / 'src/core/rpu_kernel_cache.inc').read_text(),
                          'static bool is_optional_kernel('),
        'SCM': block((ROOT / 'src/core/rpu_helpers.h').read_text(),
                     'inline void rpu_set_legacy_scm_u16_checked('),
        'LAUNCH': launch,
    }
    source = r'''
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <c10/util/Half.h>
#define TORCH_CHECK(ok, ...) do { if (!(ok)) throw std::runtime_error(#ok); } while (false)
OPTIONAL
KERNEL_IDS
KERNEL_NAMES
constexpr KernelId ID = KernelId::WALL_BASE_PREFILL_O_OWNER_RING_NORM_M224;
enum class GraphKernelNoDdrProof {RingAllReduceSpmResidual};
struct Kernel {
    std::array<uint16_t, 4352> regs{};
    std::vector<uint32_t> word_writes, counter_writes;
    int reject = -1;
    void reset_regs() { regs.fill(0xffff); word_writes.clear(); counter_writes.clear(); }
    uint32_t set_regs(uint32_t reg, uint16_t value) {
        if (int(reg) == reject) return 5;
        regs.at(reg) = value; word_writes.push_back(reg); return 0;
    }
    uint32_t set_regs(uint32_t reg, uint32_t value) {
        if (int(reg) == reject) return 5;
        assert(reg % 2 == 0 && reg >= 128 && reg < 256);
        regs.at(reg) = uint16_t(value); regs.at(reg + 1) = uint16_t(value >> 16);
        counter_writes.push_back(reg); return 0;
    }
    uint32_t set_regs(uint32_t, int32_t) = delete;
} kernel;
namespace rhino_lkn { using Kernel_t = Kernel; constexpr int kKernelRegOk = 0; }
using rhino_lkn::kKernelRegOk;
constexpr int SCM_REG_OFFSET = 4096;
SCM
struct KernelCache {
    std::unordered_set<KernelId> loaded{ID};
    bool available = true;
    static auto& instance() { static KernelCache c; return c; }
    bool has_loaded(KernelId id) const { return loaded.count(id); }
};
int staged = 0, queued = 0, fetched = 0;
struct RpuKernelGraph {
    bool census = false, reject = false;
    static auto& active() { static RpuKernelGraph g; return g; }
    bool kernel_register_census_active() const { return census; }
    void stage_kernel_no_ddr(KernelId id, GraphKernelNoDdrProof proof) {
        assert(id == ID && census && proof == GraphKernelNoDdrProof::RingAllReduceSpmResidual);
        if (reject) throw std::runtime_error("proof denied");
        ++staged;
    }
};
struct Queue {
    bool broadcast = false;
    std::array<uint16_t, 4352> packet{};
    void set_broadcast_mode(bool value) { broadcast = value; }
    void enqueu_kernel(Kernel& k, std::vector<uint16_t> grid, std::vector<uint8_t> cores) {
        assert(broadcast && grid == std::vector<uint16_t>({8, 1, 1}));
        assert(cores == std::vector<uint8_t>({0, 1, 2, 3, 4, 5, 6, 7}));
        assert(k.counter_writes.size() == 64);
        packet = k.regs;
        for (int i = 0; i < 64; ++i) {
            const uint32_t reg = 128 + 2 * i;
            assert(k.counter_writes[i] == reg && packet[reg] == 0 && packet[reg + 1] == 0);
            // Simulate live epochs so later packets must clear every pair.
            k.regs[reg] = 19; k.regs[reg + 1] = 3;
        }
        ++queued;
    }
} queue;
Kernel* get_kernel(KernelId id) {
    assert(id == ID); ++fetched;
    return KernelCache::instance().available ? &kernel : nullptr;
}
Queue* get_queue(int cores) { assert(cores == 8); return &queue; }
#define GET_KERNEL(id) get_kernel(id)
#define GET_QUEUE(n) get_queue(n)
struct SpmAllocator {
    static constexpr uint32_t SPM_USABLE = 8387584;
    bool ready = true;
    std::array<uint32_t, 8> bases{0x20000000, 0x21000000, 0x22300000, 0x23800000,
                                 0x24700000, 0x25100000, 0x26300000, 0x27000000};
    bool is_initialized() const { return ready; }
    uint32_t addr(int core, uint32_t offset) const { return bases.at(core) + offset; }
} spm;
#define SPM_ALLOC spm
LAUNCH
int main() {
    assert(is_optional_kernel("wall_base_prefill_o_owner_ring_norm_m224_h2048"));
    assert(!is_optional_kernel("llama_insert_kv_cache_multiwarp_v16"));
    assert(std::string(KERNEL_ID_NAMES[static_cast<size_t>(ID)]) ==
           "wall_base_prefill_o_owner_ring_norm_m224_h2048");
    static_assert(ID != KernelId::WALL_BASE_VISION_OWNER_RING_NORM_M672 &&
                  ID != KernelId::WALL_BASE_VISION_DOWN_OWNER_RING_NORM_M672);
    constexpr uint32_t base = 0x20000000, full = 917504, gamma_bytes = 4096;
    constexpr uint32_t partial = base + 4096, residual = partial + full + 256;
    constexpr uint32_t raw = residual + full + 256, gamma = raw + full + 256;
    using Slabs = std::array<uint32_t, 4>;
    const Slabs exact{partial, residual, raw, gamma};
    const std::array<uint32_t, 4> bytes{full, full, full, gamma_bytes};
    auto run = [&](Slabs slabs, bool expected, double eps = 1e-6) {
        const int before = queued;
        bool accepted = true;
        try {
            rpu_launch_wall_base_prefill_o_owner_ring_norm_m224_kernel(
                slabs[0], slabs[1], slabs[2], slabs[3], eps);
        } catch (const std::exception&) { accepted = false; }
        assert(accepted == expected && queued == before + int(expected));
    };
    auto check_packet = [&] {
        // Frozen successful micro ABI, including untouched words. The expected
        // buffer is independent of the shared production packet construction.
        std::array<uint16_t, 4352> expected; expected.fill(0xffff);
        auto word32 = [&](int reg, uint32_t value) {
            expected[reg] = uint16_t(value); expected[reg + 1] = uint16_t(value >> 16);
        };
        expected[0] = 8; expected[1] = 1; word32(2, 63488);
        word32(4, residual - base); expected[6] = 0;
        expected[8] = 2048; expected[9] = c10::Half(1.0f / 2048).x;
        expected[10] = 4096; expected[11] = 4;
        word32(12, raw); word32(14, residual); word32(16, gamma);
        expected[18] = c10::Half(1e-6).x;
        expected[64] = 8; expected[65] = 1; expected[66] = 1;
        for (int core = 0; core < 8; ++core) {
            const uint32_t peer = spm.bases[core] - base;
            for (const auto [field, value] : std::array<std::pair<int, uint32_t>, 5>{{
                 {0, peer + partial - base}, {16, peer + raw - base},
                 {32, core * 114688u}, {48, 57344u}, {64, peer + residual - base}}}) {
                expected[(4096 + field + core) ^ 1] = uint16_t(value);
                expected[(4096 + field + 8 + core) ^ 1] = uint16_t(value >> 16);
            }
        }
        for (int reg = 128; reg < 256; ++reg) expected[reg] = 0;
        assert(queue.packet == expected);
    };
    for (bool census : {false, true}) {
        RpuKernelGraph::active().census = census;
        for (int replay = 0; replay < 3; ++replay) {
            const int old_staged = staged;
            queue.broadcast = false;
            run(exact, true); check_packet();
            assert(staged == old_staged + int(census));
        }
    }
    RpuKernelGraph::active().census = false;
    // Every written parameter, SCM halfword and counter pair must reject a
    // failed register write before enqueue. Counters use the 32-bit overload.
    const auto words = kernel.word_writes, counters = kernel.counter_writes;
    for (const auto& regs : {words, counters}) for (uint32_t reg : regs) {
        kernel.reject = reg; run(exact, false); kernel.reject = -1;
    }
    for (int i = 0; i < 4; ++i) {
        for (uint32_t address : {uint32_t{0}, base - 256, exact[i] - base,
                                 exact[i] + 2, exact[i] + 32, base + 0x1000000u}) {
            auto bad = exact; bad[i] = address; run(bad, false);
        }
        auto edge = exact;
        edge[i] = base + SpmAllocator::SPM_USABLE - bytes[i];
        run(edge, true); edge[i] += 256; run(edge, false);
        for (int j = 0; j < i; ++j) {
            for (uint32_t overlap : {0u, 256u, bytes[j] - 256}) {
                auto bad = exact; bad[i] = bad[j] + overlap; run(bad, false);
            }
        }
    }
    // Exactly adjacent full-sized operands are legal; all four remain distinct.
    run({partial, partial + full, partial + 2 * full, partial + 3 * full}, true);
    spm.ready = false; run(exact, false); spm.ready = true;
    const auto bases = spm.bases;
    const uint32_t last_end = gamma - base + gamma_bytes;
    for (int core = 1; core < 8; ++core) {
        spm.bases[core] = uint32_t(uint64_t{UINT32_MAX} + 1 - last_end);
        run(exact, true);  // Last byte equals UINT32_MAX, with no wrap.
        spm.bases[core] += 256; run(exact, false);
        spm.bases[core] = base - 256; run(exact, false);
        spm.bases = bases;
    }
    for (double eps : {0.0, -1.0, 1e-5, std::nextafter(1e-6, 1.0),
                       std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) run(exact, false, eps);
    auto& cache = KernelCache::instance();
    // New symbol alone suffices; old Vision payloads cannot substitute for it.
    cache.loaded.clear(); run(exact, false);
    cache.loaded = {KernelId::WALL_BASE_VISION_OWNER_RING_NORM_M672,
                    KernelId::WALL_BASE_VISION_DOWN_OWNER_RING_NORM_M672};
    run(exact, false);
    cache.loaded.insert(ID); run(exact, true);
    cache.loaded = {ID}; run(exact, true);
    cache.available = false; run(exact, false); cache.available = true;
    RpuKernelGraph::active().census = true;
    RpuKernelGraph::active().reject = true;
    const int old_fetched = fetched, old_staged = staged;
    run(exact, false);
    assert(fetched == old_fetched && staged == old_staged);
    RpuKernelGraph::active().reject = false;
    cache.loaded.clear(); run(exact, false);
    assert(fetched == old_fetched && staged == old_staged);
    cache.loaded = {ID}; run(exact, true); check_packet();
    // The internal shape guard is still bounded even though three fixed public
    // wrappers now share the body. No untested C256/C320 route is admitted.
    for (auto shape : std::array<std::pair<uint32_t, uint32_t>, 5>{{
             {256, 2048}, {320, 2048}, {224, 1280}, {672, 2048}, {0, 2048}}}) {
        const int before = queued;
        bool rejected = false;
        try {
            launch_wall_base_owner_ring_norm(partial, residual, raw, gamma, 1e-6,
                ID, true, shape.first, shape.second);
        } catch (const std::exception&) { rejected = true; }
        assert(rejected && queued == before);
    }
}
'''
    for key, value in pieces.items():
        source = source.replace('\n' + key + '\n', '\n' + value + '\n')
    compile_run(tmp_path, 'wall_prefill_owner_norm', source)
