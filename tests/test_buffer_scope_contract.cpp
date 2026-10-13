// Device contract tests; build with RHINOFORGE_BUILD_DEVICE_TESTS.
// Run on an exclusively locked board with the matching Release runtime.

#include "core/fused_model_base.h"
#include "core/rpu_spm_allocator.h"
#include "graph/execution_coordinator.h"
#include "graph/graph_runtime.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

int g_passed = 0;
int g_failed = 0;

#define EXPECT(cond, label)                                                   \
    do {                                                                      \
        if (cond) {                                                           \
            ++g_passed;                                                       \
            std::printf("  [PASS] %s\n", label);                              \
        } else {                                                              \
            ++g_failed;                                                       \
            std::fprintf(stderr, "  [FAIL] %s (%s:%d)\n",                    \
                         label, __FILE__, __LINE__);                           \
        }                                                                     \
    } while (0)

bool disjoint(uint32_t a_offset, uint32_t a_bytes,
              uint32_t b_offset, uint32_t b_bytes) {
    const uint64_t a_end = static_cast<uint64_t>(a_offset) + a_bytes;
    const uint64_t b_end = static_cast<uint64_t>(b_offset) + b_bytes;
    return a_end <= b_offset || b_end <= a_offset;
}

class PlannerBudgetModel final : public v3::FusedModelBase {
public:
    PlannerBudgetModel() {
        set_model_params(1, 1, 16, 16, 16);
        set_num_layers(1);
    }

    int64_t resolve(int64_t chunk_size) {
        set_chunk_size_override(chunk_size);
        return resolve_current();
    }

    int64_t resolve_current(
        int64_t seq_len = 16, int64_t position = 0) {
        return resolve_chunk_size_for_shape(
            seq_len, position, std::nullopt, /*is_causal=*/true);
    }

    size_t resolve_stage_domain(
        int64_t seq_len = 16, int64_t position = 0) {
        return resolve_prefill_stage_domain_for_shape(
                   seq_len, position, std::nullopt, /*is_causal=*/true)
            .size();
    }

    void resolve_causal_shape(int64_t execution_len) {
        (void)resolve_spm_pipeline_causal_prefill_shape_for_cpu_contract(
            execution_len);
    }

    int declare_calls() const { return declare_calls_.load(); }

protected:
    std::vector<v3::BufferDecl> declare_buffers(
        const v3::LayoutContext&) override {
        ++declare_calls_;
        return {{"temp", 96 * 1024, 0, 0, v3::StorageClass::Temp}};
    }

    v3::ModelStaticConfig static_config() override {
        v3::ModelStaticConfig cfg;
        cfg.num_layers = 1;
        return cfg;
    }

    v3::ModelDynamicConfig dynamic_config(
        const v3::ChunkPlan&) override { return {}; }

    void build_layer_subgraph(int, const v3::ChunkInfo&) override {}

private:
    std::atomic<int> declare_calls_{0};
};

class DualPlannerBudgetModel final : public v3::FusedModelBase {
public:
    DualPlannerBudgetModel() {
        set_model_params(1, 1, 16, 16, 16);
        set_num_layers(1);
    }

    void resolve() {
        set_chunk_size_override(16);
        (void)resolve_spm_pipeline_causal_prefill_shape_for_cpu_contract(16);
    }

    void set_inter_layer_io(v3::InterLayerIO io) { io_ = io; }
    void set_chunk_outer(bool enabled) { chunk_outer_ = enabled; }
    void set_joint_floor_case(bool enabled) { joint_floor_case_ = enabled; }

protected:
    std::vector<v3::BufferDecl> declare_buffers(
        const v3::LayoutContext& ctx) override {
        const int64_t bytes = joint_floor_case_
            ? (ctx.kv_insert_chunk_size == 0 ? 16 * 1024 : 48 * 1024)
            : (ctx.kv_insert_chunk_size == 0 ? 32 * 1024 : 96 * 1024);
        std::vector<v3::BufferDecl> decls{
            {"temp", bytes, 0, 0, v3::StorageClass::Temp}};
        if (joint_floor_case_) {
            decls.push_back(
                {"fixed", 32 * 1024, 0, 0,
                 v3::StorageClass::Persistent});
        }
        return decls;
    }

    v3::ModelStaticConfig static_config() override {
        v3::ModelStaticConfig cfg;
        cfg.num_layers = 1;
        cfg.kv_first_fn =
            static_cast<void (v3::FusedModelBase::*)(
                int, const v3::ChunkInfo&)>(
                &DualPlannerBudgetModel::kv_first_stub);
        cfg.kv_first_chunk_plan_fn =
            static_cast<v3::ChunkPlan (v3::FusedModelBase::*)(
                const v3::ChunkPlan&)>(
                &DualPlannerBudgetModel::kv_first_plan);
        return cfg;
    }

    v3::ModelDynamicConfig dynamic_config(
        const v3::ChunkPlan&) override {
        v3::ModelDynamicConfig cfg;
        cfg.chunk_mode = v3::ChunkMode::KV_FIRST;
        cfg.inter_layer_io = io_;
        cfg.chunk_outer_within_group = chunk_outer_;
        return cfg;
    }

    void build_layer_subgraph(int, const v3::ChunkInfo&) override {}

private:
    void kv_first_stub(int, const v3::ChunkInfo&) {}
    v3::ChunkPlan kv_first_plan(const v3::ChunkPlan&) { return {8, 2}; }
    v3::InterLayerIO io_ = v3::InterLayerIO::AUTO;
    bool chunk_outer_ = false;
    bool joint_floor_case_ = false;
};

class FixedAlignmentBudgetModel final : public v3::FusedModelBase {
public:
    FixedAlignmentBudgetModel() {
        set_model_params(1, 1, 16, 16, 16);
        set_num_layers(1000);
    }

    void resolve() {
        set_chunk_size_override(16);
        (void)resolve_chunk_size_for_shape(
            16, 0, std::nullopt, /*is_causal=*/true);
    }

protected:
    std::vector<v3::BufferDecl> declare_buffers(
        const v3::LayoutContext&) override {
        return {
            {"fixed", 1, 0, 0, v3::StorageClass::PersistentPerLayer,
             1000},
            {"temp", 8'100'000, 0, 0, v3::StorageClass::Temp},
        };
    }

    v3::ModelStaticConfig static_config() override {
        v3::ModelStaticConfig cfg;
        cfg.num_layers = 1000;
        return cfg;
    }

    v3::ModelDynamicConfig dynamic_config(
        const v3::ChunkPlan&) override { return {}; }

    void build_layer_subgraph(int, const v3::ChunkInfo&) override {}
};

std::string planner_error(PlannerBudgetModel& model, int64_t chunk_size) {
    try {
        (void)model.resolve(chunk_size);
    } catch (const c10::Error& error) {
        return error.what_without_backtrace();
    }
    return {};
}

std::string dual_planner_error(DualPlannerBudgetModel& model) {
    try {
        model.resolve();
    } catch (const c10::Error& error) {
        return error.what_without_backtrace();
    }
    return {};
}

std::string fixed_alignment_error(FixedAlignmentBudgetModel& model) {
    try {
        model.resolve();
    } catch (const c10::Error& error) {
        return error.what_without_backtrace();
    }
    return {};
}

}  // namespace

int main() {
    using v3::BufferDecl;
    using v3::BufferScope;
    using v3::StorageClass;

    // Layer-loop peak:
    //   LayerWide 512 + max(KvInsert 256, Compute 512) = 1024.
    // Outside-layer-loop peak:
    //   pre phase max = 512 + 256; post phase = 768; peak = 768.
    // Contract total: max(1024, 768) = 1024.
    const std::vector<BufferDecl> decls = {
        {"layer", 512, 0, 2, StorageClass::Temp, 0, nullptr,
         BufferScope::LayerWide},
        {"kv", 256, 1, 1, StorageClass::Temp, 0, nullptr,
         BufferScope::KvInsert},
        {"compute", 512, 1, 1, StorageClass::Temp, 0, nullptr,
         BufferScope::Compute},
        {"pre_a", 512, 1, 1, StorageClass::Temp, 0, nullptr,
         BufferScope::OutsideLayerLoop},
        {"pre_b", 256, 1, 1, StorageClass::Temp, 0, nullptr,
         BufferScope::OutsideLayerLoop},
        {"post", 768, 2, 2, StorageClass::Temp, 0, nullptr,
         BufferScope::OutsideLayerLoop},
    };

    std::vector<SpmAllocator::AllocRequest> requests;
    requests.reserve(decls.size());
    for (const BufferDecl& decl : decls) {
        requests.push_back({
            decl.size,
            decl.phase_start,
            decl.phase_end,
            static_cast<int>(decl.scope),
        });
    }

    const int64_t estimated = v3::detail::estimate_temporary_total(decls);
    const SpmAllocator::AliasedPlan plan =
        SpmAllocator::plan_temporary_aliased(requests);

    EXPECT(plan.offsets.size() == decls.size(), "one offset per declaration");
    if (plan.offsets.size() == decls.size()) {
        const uint32_t layer = plan.offsets[0];
        const uint32_t kv = plan.offsets[1];
        const uint32_t compute = plan.offsets[2];
        const uint32_t pre_a = plan.offsets[3];
        const uint32_t pre_b = plan.offsets[4];
        const uint32_t post = plan.offsets[5];

        EXPECT(kv == compute,
               "KvInsert and Compute alias across disjoint loop phases");
        EXPECT(pre_a == layer,
               "pre-layer temporary aliases the layer-loop allocation");
        EXPECT(post == pre_a,
               "post-layer temporary aliases the disjoint pre-layer phase");

        EXPECT(disjoint(layer, 512, kv, 256),
               "LayerWide does not overlap KvInsert");
        EXPECT(disjoint(layer, 512, compute, 512),
               "LayerWide does not overlap Compute");
        EXPECT(disjoint(pre_a, 512, pre_b, 256),
               "co-live OutsideLayerLoop buffers do not overlap");
    }

    EXPECT(estimated == 1024, "estimator reports the scope-contract peak");
    EXPECT(plan.peak_bytes == 1024, "allocator planner reports the same peak");
    EXPECT(static_cast<uint64_t>(estimated) == plan.peak_bytes,
           "allocator and estimator peaks agree");

    // Fragmentation regression: the maximum simultaneous live bytes are 2816,
    // but first-fit needs 3840 because the phase-3 1792-byte slot does not fit
    // in the 1024-byte hole below the long-lived buffer. Chunk budgeting must
    // predict the allocator's real request, not the theoretical live-byte sum.
    const std::vector<BufferDecl> fragmented = {
        {"short_0", 1024, 2, 2, StorageClass::Temp},
        {"late", 1280, 4, 6, StorageClass::Temp},
        {"short_1", 1792, 3, 3, StorageClass::Temp},
        {"long", 1024, 2, 5, StorageClass::Temp},
    };
    std::vector<SpmAllocator::AllocRequest> fragmented_requests;
    for (const BufferDecl& decl : fragmented) {
        fragmented_requests.push_back({
            decl.size,
            decl.phase_start,
            decl.phase_end,
            static_cast<int>(decl.scope),
        });
    }
    const int64_t fragmented_estimate =
        v3::detail::estimate_temporary_total(fragmented);
    const SpmAllocator::AliasedPlan fragmented_plan =
        SpmAllocator::plan_temporary_aliased(fragmented_requests);
    EXPECT(fragmented_plan.peak_bytes == 3840,
           "first-fit planner accounts for lifecycle fragmentation");
    EXPECT(static_cast<uint64_t>(fragmented_estimate) == fragmented_plan.peak_bytes,
           "estimator matches fragmented allocator plan");

    // Exact Qwen3-VL multiview Text C225 layout.  The MLP executes
    // up-write -> gate*up (last up read) -> down-write -> all-reduce (last
    // down read), so the sealed composite may hand one slot from up [7,7] to
    // down [8,8].  The ordinary conservative down [7,8] layout remains the
    // control arm.
    const auto text_requests = [](int down_phase_start) {
        constexpr int layer = static_cast<int>(BufferScope::LayerWide);
        return std::vector<SpmAllocator::AllocRequest>{
            {1'152'000, 1, 8, layer},  // residual1
            {1'152'000, 1, 8, layer},  // input_norm
            {230'400, 2, 5, layer},    // q
            {57'600, 2, 4, layer},     // k
            {57'600, 2, 4, layer},     // v
            {230'400, 3, 5, layer},    // output
            {1'152'000, 5, 5, layer},  // oproj
            {131'072, 5, 5, layer},    // sdpa_tmp
            {547'328, 7, 8, layer},    // gate
            {547'328, 7, 7, layer},    // up
            {1'152'000, down_phase_start, 8, layer},  // down
            {50'176, 8, 9, layer},     // lm_head_out
        };
    };
    const SpmAllocator::AliasedPlan text_conservative =
        SpmAllocator::plan_temporary_aliased(text_requests(7));
    const SpmAllocator::AliasedPlan text_composite =
        SpmAllocator::plan_temporary_aliased(text_requests(8));
    EXPECT(text_conservative.peak_bytes == 4'550'656,
           "ordinary C225 Text keeps the conservative down lifetime");
    EXPECT(text_composite.peak_bytes == 4'163'072,
           "sealed C225 Text reclaims the dead up/down overlap");
    EXPECT(text_composite.offsets.size() == 12 &&
               text_composite.offsets[9] == 2'851'328 &&
               text_composite.offsets[10] == text_composite.offsets[9],
           "sealed C225 Text hands the exact up slot to down");

    // Exact Wall-OSS RTC P640/C320 A16-resident layout. Chunk-outer
    // traversal reuses this one-C320 allocation across both chunks.
    constexpr int wall_scope = static_cast<int>(BufferScope::LayerWide);
    const std::vector<SpmAllocator::AllocRequest> wall_p640_c320 = {
        {1'310'720, 1, 8, wall_scope},  // residual1
        {1'310'720, 1, 8, wall_scope},  // input_norm
        {163'840, 2, 5, wall_scope},    // q
        {81'920, 2, 4, wall_scope},     // k
        {81'920, 2, 4, wall_scope},     // v
        {163'840, 3, 5, wall_scope},    // output
        {1'310'720, 5, 5, wall_scope},  // oproj
        {81'920, 5, 5, wall_scope},     // sdpa_tmp
        {880'640, 7, 8, wall_scope},    // gate
        {880'640, 7, 7, wall_scope},    // up
        {1'310'720, 7, 8, wall_scope},  // down
        {50'176, 8, 9, wall_scope},     // lm_head_out
    };
    const SpmAllocator::AliasedPlan wall_plan =
        SpmAllocator::plan_temporary_aliased(wall_p640_c320);
    constexpr uint64_t kFullVlaPersistentBytes = 591'360;
    constexpr uint64_t kAlignedSpmBudgetBytes = 8'303'616;
    EXPECT(wall_plan.peak_bytes == 5'693'440,
           "Wall P640/C320 exact allocator peak is stable");
    EXPECT(wall_plan.peak_bytes + kFullVlaPersistentBytes == 6'284'800 &&
               wall_plan.peak_bytes + kFullVlaPersistentBytes <=
                   kAlignedSpmBudgetBytes,
           "Wall P640/C320 plus full-VLA persistent state fits SPM");

    // Opt-in staged P640: packed hidden/Q/K/V + one C320 attention/result
    // scratch, one compact half-0 residual shard, M640 gate, and C320
    // down/scratch. Both K/V halves enter the cache before SDPA, so output and
    // sdpa_tmp may reuse the now-dead packed K and V slots respectively.
    constexpr uint64_t kWallQHalfBytes = 163'840;
    constexpr uint64_t kWallKvHalfBytes = 81'920;
    const std::vector<SpmAllocator::AllocRequest> wall_p640_staged = {
        {2'621'440, 1, 8, wall_scope},  // packed residual1 [640,2048]
        {1'310'720, 1, 8, wall_scope},  // input_norm / half-1 full residual
        {163'840, 1, 8, wall_scope},    // half-0 compact residual shard
        {2 * kWallQHalfBytes, 2, 5, wall_scope},   // packed q
        {2 * kWallKvHalfBytes, 2, 2, wall_scope},  // packed k
        {2 * kWallKvHalfBytes, 2, 2, wall_scope},  // packed v
        {kWallQHalfBytes, 3, 4, wall_scope},       // C320 output
        {1'310'720, 4, 5, wall_scope},  // oproj
        {kWallKvHalfBytes, 3, 3, wall_scope},  // C320 sdpa_tmp
        {1'761'280, 7, 8, wall_scope},  // M640 gate
        {1'310'720, 7, 8, wall_scope},  // C320 down / bounded up-tile scratch
    };
    const SpmAllocator::AliasedPlan wall_staged_plan =
        SpmAllocator::plan_temporary_aliased(wall_p640_staged);
    EXPECT(wall_staged_plan.peak_bytes == 7'168'000,
           "Wall staged P640 exact allocator peak is stable");
    EXPECT(wall_staged_plan.offsets.size() == 11 &&
               wall_staged_plan.offsets[6] ==
                   wall_staged_plan.offsets[4] &&
               wall_staged_plan.offsets[8] ==
                   wall_staged_plan.offsets[5] &&
               wall_staged_plan.offsets[3] !=
                   wall_staged_plan.offsets[7] &&
               wall_staged_plan.offsets[9] !=
                   wall_staged_plan.offsets[10],
           "Wall staged P640 aliases output/K and sdpa_tmp/V while keeping packed Q live");
    EXPECT(
        disjoint(wall_staged_plan.offsets[9], 1'761'280,
                 wall_staged_plan.offsets[10], 1'310'720) &&
            disjoint(wall_staged_plan.offsets[9], 1'761'280,
                     wall_staged_plan.offsets[0], 2'621'440) &&
            disjoint(wall_staged_plan.offsets[10], 1'310'720,
                     wall_staged_plan.offsets[0] + 1'310'720, 1'310'720),
        "Wall staged packed Down input and both C320 outputs are disjoint");
    EXPECT(wall_staged_plan.peak_bytes + kFullVlaPersistentBytes ==
               7'759'360 &&
               wall_staged_plan.peak_bytes + kFullVlaPersistentBytes <=
                   kAlignedSpmBudgetBytes,
           "Wall staged P640 plus full-VLA persistent state fits SPM");

    int64_t balanced = 0;
    for (int64_t cs : {512, 448, 384, 320, 256}) {
        if (v3::detail::prefer_balanced_chunk(576, cs, balanced)) balanced = cs;
    }
    EXPECT(balanced == 320,
           "same chunk count prefers the most balanced legal tail");
    EXPECT(!v3::detail::prefer_balanced_chunk(576, 192, balanced),
           "tail balance never increases the minimum chunk count");

    {
        DualPlannerBudgetModel invalid_io;
        invalid_io.set_inter_layer_io(static_cast<v3::InterLayerIO>(99));
        EXPECT(dual_planner_error(invalid_io).find(
                   "invalid inter-layer I/O mode") != std::string::npos,
               "live planning rejects an invalid inter-layer I/O mode");

        DualPlannerBudgetModel invalid_outer;
        invalid_outer.set_inter_layer_io(v3::InterLayerIO::SPM_RESIDENT);
        invalid_outer.set_chunk_outer(true);
        EXPECT(dual_planner_error(invalid_outer).find(
                   "chunk_outer_within_group requires") != std::string::npos,
               "live planning rejects an invalid chunk-outer combination");

        FixedAlignmentBudgetModel fixed_alignment;
        EXPECT(fixed_alignment_error(fixed_alignment).find(
                   "SPM budget") != std::string::npos,
               "fixed budget rounds each per-layer declaration to 256 bytes");
    }

    // Planning either borrows this thread's active Graph claim or takes the
    // existing process-exclusive cleanup claim. A foreign owner must fail
    // before declare_buffers observes mutable model/allocator state.
    {
        PlannerBudgetModel model;
        const int before_overflow = model.declare_calls();
        auto rejects_shape = [&](auto&& query) {
            try {
                query();
            } catch (const c10::Error& error) {
                return std::string(error.what_without_backtrace()).find(
                           "RPU_PLANNER_REJECT:CAPABILITY") !=
                    std::string::npos;
            }
            return false;
        };
        constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
        EXPECT(
            rejects_shape([&] { (void)model.resolve_current(kMax, 0); }) &&
                rejects_shape([&] {
                    (void)model.resolve_current(16, kMax - 15);
                }) &&
                rejects_shape([&] {
                    (void)model.resolve_stage_domain(kMax, 0);
                }) &&
                rejects_shape([&] {
                    (void)model.resolve_stage_domain(16, kMax - 15);
                }) &&
                rejects_shape([&] { model.resolve_causal_shape(kMax); }) &&
                model.declare_calls() == before_overflow,
            "all native planning entries reject shape overflow before declarations");

        model.resolve(16);

        RpuKernelGraph graph;
        graph.begin();
        const int before_foreign = model.declare_calls();
        std::string foreign_error;
        std::string foreign_domain_error;
        std::thread foreign([&] {
            try {
                (void)model.resolve_current();
            } catch (const c10::Error& error) {
                foreign_error = error.what_without_backtrace();
            }
            try {
                (void)model.resolve_stage_domain();
            } catch (const c10::Error& error) {
                foreign_domain_error = error.what_without_backtrace();
            }
        });
        foreign.join();
        EXPECT(!foreign_error.empty() && !foreign_domain_error.empty() &&
                   model.declare_calls() == before_foreign,
               "foreign Graph owner rejects scalar/domain planning before declarations");
        EXPECT(model.resolve_current() == 16,
               "planner borrows the current thread's active Graph claim");
        graph.abort();
        EXPECT(model.resolve_stage_domain() > 0,
               "stage-domain planning succeeds after the foreign owner exits");

        auto physical = RpuExecutionCoordinator::enter_physical(
            "planner physical-plus-Graph contract");
        const int before_physical_only = model.declare_calls();
        std::string physical_only_error;
        try {
            (void)model.resolve_current();
        } catch (const c10::Error& error) {
            physical_only_error = error.what_without_backtrace();
        }
        EXPECT(physical_only_error.find(
                   "no process-active Graph or physical") !=
                   std::string::npos &&
                   model.declare_calls() == before_physical_only,
               "planner rejects physical ownership without an active Graph");
        RpuKernelGraph physical_graph;
        physical_graph.begin();
        EXPECT(model.resolve_current() == 16,
               "planner borrows a current physical-plus-Graph owner");
        physical_graph.abort();
        RpuExecutionCoordinator::exit_physical(
            physical, "planner physical-plus-Graph cleanup");
    }

    // Leave 128 KiB below a synthetic persistent floor. After FMB's 64 KiB
    // physical reserve, neither auto nor an explicit 16-row chunk may admit
    // this 96 KiB temporary layout. The explicit path used to bypass this
    // actual-free check and fail only during real allocation.
    {
        PlannerBudgetModel model;
        SPM_ALLOC.init();
        SPM_ALLOC.alloc_super_persistent(
            SpmAllocator::SPM_USABLE - 128 * 1024);
        const std::string auto_error = planner_error(model, 0);
        const std::string explicit_error = planner_error(model, 16);
        DualPlannerBudgetModel dual_model;
        const std::string dual_error = dual_planner_error(dual_model);
        DualPlannerBudgetModel joint_model;
        joint_model.set_joint_floor_case(true);
        const std::string joint_error = dual_planner_error(joint_model);
        EXPECT(auto_error.find("RPU_PLANNER_REJECT:NO_FEASIBLE:") !=
                   std::string::npos,
               "auto rejects a chunk above post-reset free SPM");
        EXPECT(explicit_error.find("SPM budget") != std::string::npos,
               "explicit rejects the same chunk above post-reset free SPM");
        EXPECT(auto_error.find("available=65536 bytes") != std::string::npos,
               "auto reports the reserved post-reset budget");
        EXPECT(explicit_error.find("available=65536 bytes") != std::string::npos,
               "explicit reports the same reserved post-reset budget");
        EXPECT(dual_error.find("final joint compute/KV_FIRST layout") !=
                   std::string::npos &&
                   dual_error.find("available=65536") != std::string::npos,
               "final dual plan uses the same post-reset SPM budget");
        EXPECT(joint_error.find("final joint compute/KV_FIRST layout") !=
                   std::string::npos &&
                   joint_error.find("fixed=32768") != std::string::npos &&
                   joint_error.find("available=32768") != std::string::npos,
               "first layout jointly budgets new persistent and temporary SPM");
    }

    std::printf("\n%d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}
