// Device contract tests; build with RHINOFORGE_BUILD_DEVICE_TESTS.
// Run on an exclusively locked board with the matching Release runtime.

#include "core/fused_model_base.h"
#include "core/rpu_caching_allocator.h"
#include "core/rpu_kernel_cache.h"
#include "core/rpu_kernel_decls.h"
#include "core/rpu_spm_pipeline.h"
#include "core/rpu_tensor_ops.h"
#include "graph/graph_runtime.h"

#include <c10/util/Exception.h>
#include <ATen/EmptyTensor.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <unistd.h>
#include <vector>

// Use the production TP packing, not a second test-only implementation.
at::Tensor tp_row_swizzle_mc_weight(const at::Tensor&, int64_t, int64_t);
at::Tensor tp_col_swizzle_mc_weight(const at::Tensor&, int64_t, int64_t);

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

template <typename Fn>
bool throws_c10(Fn&& fn) {
    try {
        fn();
    } catch (const c10::Error&) {
        return true;
    }
    return false;
}

template <typename Fn>
bool throws_c10_containing(Fn&& fn, const char* expected) {
    try {
        fn();
    } catch (const c10::Error& error) {
        return std::string(error.what()).find(expected) != std::string::npos;
    }
    return false;
}

template <typename T, typename = void>
struct has_public_offset_method : std::false_type {};

template <typename T>
struct has_public_offset_method<
    T, std::void_t<decltype(std::declval<const T&>().offset())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_public_address_method : std::false_type {};

template <typename T>
struct has_public_address_method<
    T, std::void_t<decltype(std::declval<const T&>().address())>>
    : std::true_type {};

template <typename T, typename = void>
struct has_public_resolve_method : std::false_type {};

template <typename T>
struct has_public_resolve_method<
    T, std::void_t<decltype(std::declval<const T&>().resolve())>>
    : std::true_type {};

struct DebugDmaSnapshot {
    std::string variant;
    uint64_t src_addr = 0;
    uint64_t dst_addr = 0;
    size_t bytes = 0;
    uint8_t channel = 0;
    bool semantic_endpoint = false;
    bool canonical_member = false;
    bool typed_spm_peer = false;
};

uint64_t json_u64_field(const std::string& line, const char* key) {
    const std::string marker = std::string("\"") + key + "\":";
    const size_t begin = line.find(marker);
    TORCH_CHECK(begin != std::string::npos,
                "contract debug JSON is missing field ", key);
    const char* value = line.c_str() + begin + marker.size();
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    TORCH_CHECK(end != value,
                "contract debug JSON has an invalid numeric field ", key);
    return static_cast<uint64_t>(parsed);
}

std::string json_string_field(const std::string& line, const char* key) {
    const std::string marker = std::string("\"") + key + "\":\"";
    const size_t begin = line.find(marker);
    TORCH_CHECK(begin != std::string::npos,
                "contract debug JSON is missing field ", key);
    const size_t value_begin = begin + marker.size();
    const size_t value_end = line.find('"', value_begin);
    TORCH_CHECK(value_end != std::string::npos,
                "contract debug JSON has an unterminated string field ",
                key);
    return line.substr(value_begin, value_end - value_begin);
}

std::vector<DebugDmaSnapshot> debug_dma_snapshots(
    const RpuKernelGraph& graph) {
    std::vector<DebugDmaSnapshot> result;
    const std::string diagnostics = graph.dump_nodes_jsonl();
    TORCH_CHECK(diagnostics.find("\"src_addr\":") == std::string::npos &&
                    diagnostics.find("\"dst_addr\":") == std::string::npos &&
                    diagnostics.find("\"live_base\":") == std::string::npos,
                "public Graph diagnostics must not expose allocation addresses");
    // Keep address checks inside this process; the public JSON is pointer-free.
    const auto plan = graph.replay_plan();
    TORCH_CHECK(plan.steps != nullptr, "contract graph has no replay steps");
    size_t node_index = 0;
    std::istringstream input(diagnostics);
    std::string line;
    while (std::getline(input, line)) {
        if (line.find("\"kind\":\"Dma\"") == std::string::npos) {
            continue;
        }
        while (node_index < plan.steps->size() &&
               (*plan.steps)[node_index].kind != GraphNodeKind::Dma) {
            ++node_index;
        }
        TORCH_CHECK(node_index < plan.steps->size(),
                    "diagnostic DMA has no matching replay step");
        const auto& dma = (*plan.steps)[node_index++].as_dma();
        result.push_back({
            json_string_field(line, "variant"),
            dma.src_addr,
            dma.dst_addr,
            static_cast<size_t>(json_u64_field(line, "bytes")),
            static_cast<uint8_t>(json_u64_field(line, "channel")),
            line.find("\"semantic_endpoint_id\":") != std::string::npos,
            line.find("\"semantic_canonical_member_emission\":true") !=
                std::string::npos,
            line.find("\"semantic_spm_peer_bound\":true") !=
                std::string::npos,
        });
        const auto& snapshot = result.back();
        const char* variant = dma.variant == DmaNodeData::Variant::Fixed
            ? "fixed" : dma.variant == DmaNodeData::Variant::MutableSrc
            ? "msrc" : "mdst";
        TORCH_CHECK(snapshot.variant == variant &&
                        snapshot.bytes == dma.bytes &&
                        snapshot.channel == dma.channel &&
                        snapshot.semantic_endpoint == (dma.semantic_endpoint_id != 0) &&
                        snapshot.canonical_member == dma.semantic_canonical_member_emission &&
                        snapshot.typed_spm_peer == (dma.semantic_spm_peer_id != 0),
                    "diagnostic DMA metadata differs from the replay step");
    }
    while (node_index < plan.steps->size() &&
           (*plan.steps)[node_index].kind != GraphNodeKind::Dma) {
        ++node_index;
    }
    TORCH_CHECK(node_index == plan.steps->size(),
                "replay DMA is missing from public diagnostics");
    return result;
}

uint64_t managed_spm_root_address(int core) {
    auto* root = SPM_ALLOC.root_buffer(core);
    TORCH_CHECK(root != nullptr,
                "schema contract requires an initialized SPM root for core ",
                core);
    return root->get_rpu_addr();
}

struct InactiveGraphStamp {
    RpuKernelGraph::State state;
    size_t graph_size;
    uint64_t build_generation;
    bool replayable;
    bool has_built_signature;

    bool operator==(const InactiveGraphStamp& other) const {
        return state == other.state && graph_size == other.graph_size &&
            build_generation == other.build_generation &&
            replayable == other.replayable &&
            has_built_signature == other.has_built_signature;
    }
};

InactiveGraphStamp inactive_graph_stamp(const RpuKernelGraph& graph) {
    return {
        graph.state(),
        graph.graph_size(),
        graph.build_generation(),
        graph.replayable(),
        graph.has_built_signature(),
    };
}

v3::BufferDecl temp_region(const char* name,
                           int64_t bytes,
                           int first_event,
                           int last_event) {
    v3::BufferDecl decl;
    decl.name = name;
    decl.size = bytes;
    decl.phase_start = first_event;
    decl.phase_end = last_event;
    decl.storage = v3::StorageClass::Temp;
    decl.scope = v3::BufferScope::LayerWide;
    return decl;
}

v3::SpmDense2DSpec dense_fp16(int64_t rows, int64_t cols) {
    v3::SpmDense2DSpec spec;
    spec.rows = rows;
    spec.cols = cols;
    return spec;
}

v3::SpmScratchDecl typed_scratch(uint32_t id,
                                 int64_t bytes,
                                 int first_event,
                                 int last_event) {
    return {v3::SpmScratchId(id), bytes, first_event, last_event};
}

v3::SpmIdentityRoute identity_route(uint32_t id,
                                    const v3::SpmDense2DSpec& spec,
                                    int first_write,
                                    int last_read) {
    return v3::SpmIdentityRoute::bind(
        v3::SpmPortId(id), spec, spec, first_write, last_read);
}

v3::SpmRowSliceRoute row_slice_route(
    uint32_t id,
    const v3::SpmDense2DSpec& produced,
    const v3::SpmDense2DSpec& storage,
    int64_t dst_row_begin,
    int64_t dst_row_count,
    int first_write,
    int last_read) {
    return v3::SpmRowSliceRoute::bind(
        v3::SpmPortId(id), produced, storage, dst_row_begin,
        dst_row_count, first_write, last_read);
}

v3::SpmPermutationRowRunsRoute permutation_row_runs_route(
    uint32_t source_id,
    const v3::SpmDense2DSpec& source,
    uint32_t destination_id,
    const v3::SpmDense2DSpec& destination,
    const std::vector<int32_t>& permutation,
    const std::vector<v3::SpmDstRowRun>& runs,
    const v3::SpmPermutationRowRunsEvents& events) {
    return v3::SpmPermutationRowRunsRoute::bind(
        v3::SpmPortId(source_id), source, v3::SpmPortId(destination_id),
        destination, permutation, runs, events);
}

class SealedManifestContractModel final : public v3::FusedModelBase {
public:
    enum class DeclMode {
        Normal,
        OverlappingSubrangeAlias,
        OversizedAlias,
        CrossScopeAlias,
        TemporaryCallback,
        TemporaryReverse,
        BiggerLater,
        InvalidScope,
        InvalidStorage,
    };

    explicit SealedManifestContractModel(DeclMode mode = DeclMode::Normal)
        : mode_(mode) {
        set_model_params(1, 1, 16, 16, 32);
        set_num_layers(2);
    }

    v3::SpmPipelineComponentLayout prepare_cpu() {
        v3::LayoutContext context;
        context.chunk_size = 16;
        context.num_layers = 2;
        return prepare_spm_pipeline_component_for_cpu_contract(
            context, v3::compose_fmb_default_three_stage_chunk_plan(
                         context, /*execution_len=*/16, /*position=*/0,
                         v3::ChunkMode::SEQUENTIAL));
    }

    v3::SpmPipelineComponentLayout resolve_cpu() {
        v3::LayoutContext context;
        context.chunk_size = 16;
        context.num_layers = 2;
        return resolve_spm_pipeline_component_layout_for_cpu_contract(
            context, v3::compose_fmb_default_three_stage_chunk_plan(
                         context, /*execution_len=*/16, /*position=*/0,
                         v3::ChunkMode::SEQUENTIAL));
    }

    void set_schedule(const v3::SpmFmbOccurrenceSchedule& schedule) {
        set_spm_pipeline_occurrence_schedule(schedule);
    }

    v3::SpmFmbSealedPhaseManifest seal(uint32_t arena,
                                        uint32_t base = 0) const {
        return seal_spm_pipeline_manifest(v3::SpmScratchId(arena), base);
    }

    void invalidate_for_contract() { invalidate_model_state(); }
    void rename_root_after_prepare() { root_name_ = "mutated_root"; }
    void set_decl_mode(DeclMode mode) { mode_ = mode; }

protected:
    std::vector<v3::BufferDecl> declare_buffers(
        const v3::LayoutContext&) override {
        using v3::BufferDecl;
        using v3::BufferScope;
        using v3::StorageClass;
        std::vector<BufferDecl> decls;
        decls.push_back({root_name_.c_str(), 512, 0, 1,
                         StorageClass::Temp, 0, nullptr,
                         BufferScope::LayerWide});
        decls.push_back({"root_synonym", 0, 0, 1,
                         StorageClass::Temp, 0, root_name_.c_str(),
                         BufferScope::LayerWide});
        const int subrange_phase =
            mode_ == DeclMode::OverlappingSubrangeAlias ? 1 : 2;
        const int64_t subrange_bytes =
            mode_ == DeclMode::OversizedAlias ? 768 : 256;
        const BufferScope subrange_scope =
            mode_ == DeclMode::CrossScopeAlias
            ? BufferScope::Compute : BufferScope::LayerWide;
        decls.push_back({"root_subrange", subrange_bytes,
                         subrange_phase, subrange_phase,
                         StorageClass::Temp, 0, root_name_.c_str(),
                         subrange_scope});
        const int64_t later_bytes =
            mode_ == DeclMode::BiggerLater ? 768 : 512;
        decls.push_back({"later", later_bytes, 3, 4, StorageClass::Temp, 0,
                         nullptr, BufferScope::LayerWide});
        decls.push_back({"per_layer", 256, 0, 1,
                         StorageClass::TempPerLayer, 2, nullptr,
                         BufferScope::Compute});
        decls.push_back({"persistent", 256, 0, 0,
                         StorageClass::Persistent});
        BufferDecl persistent_per_layer{
            "persistent_per_layer", 256, 0, 0,
            StorageClass::PersistentPerLayer, 2};
        persistent_per_layer.reverse_layer_alloc = true;
        persistent_per_layer.preload_callback =
            [](v3::FusedModelBase&, int, uint32_t) {};
        decls.push_back(std::move(persistent_per_layer));
        if (mode_ == DeclMode::TemporaryCallback) {
            decls[0].preload_callback =
                [](v3::FusedModelBase&, int, uint32_t) {};
        }
        if (mode_ == DeclMode::TemporaryReverse) {
            decls[0].reverse_layer_alloc = true;
        }
        if (mode_ == DeclMode::InvalidScope) {
            decls[0].scope = static_cast<BufferScope>(255);
        }
        if (mode_ == DeclMode::InvalidStorage) {
            decls[0].storage = static_cast<StorageClass>(255);
        }
        return decls;
    }

    v3::ModelStaticConfig static_config() override {
        v3::ModelStaticConfig config;
        config.num_layers = 2;
        return config;
    }

    v3::ModelDynamicConfig dynamic_config(const v3::ChunkPlan&) override {
        return {};
    }

    void build_layer_subgraph(int, const v3::ChunkInfo&) override {}

private:
    DeclMode mode_;
    std::string root_name_ = "root";
};

v3::SpmFmbOccurrence occurrence(
    uint32_t key,
    v3::SpmFmbOccurrenceKind kind,
    v3::BufferScope scope,
    int layer,
    int local_begin,
    int local_end,
    uint32_t global_origin,
    uint32_t body = 1,
    uint32_t group = 0,
    uint32_t chunk = 0,
    uint32_t repetition = 0) {
    return {key, kind, body, group, chunk, repetition, scope, layer,
            local_begin, local_end,
            v3::SpmPipelineEvent(global_origin)};
}

v3::SpmFmbOccurrenceSchedule valid_fmb_schedule(uint64_t version = 1) {
    using v3::BufferScope;
    using v3::SpmFmbOccurrenceKind;
    v3::SpmFmbOccurrenceSchedule schedule;
    schedule.version = version;
    schedule.occurrences = {
        occurrence(1, SpmFmbOccurrenceKind::LayerBody,
                   BufferScope::LayerWide, 0, 0, 2, 0, 1, 0, 0, 0),
        occurrence(2, SpmFmbOccurrenceKind::LayerBody,
                   BufferScope::LayerWide, 1, 3, 4, 10, 1, 0, 0, 1),
        occurrence(3, SpmFmbOccurrenceKind::LayerBody,
                   BufferScope::Compute, 0, 0, 1, 20, 1, 1, 0, 0),
        occurrence(4, SpmFmbOccurrenceKind::LayerBody,
                   BufferScope::Compute, 1, 0, 1, 30, 1, 1, 0, 1),
        occurrence(5, SpmFmbOccurrenceKind::PostFn,
                   BufferScope::OutsideLayerLoop, -1, 0, 0, 40,
                   2, 0, 0, 0),
    };
    schedule.use_extensions.push_back(
        {"root", -1, 5, 0, 0});
    return schedule;
}

class ResolvedTraversalContractBase : public v3::FusedModelBase {
public:
    enum class Traversal {
        Sequential,
        SpmResidentSingleLayerGroups,
        ChunkOuter,
        KvFirst,
    };

    enum class OpaqueCallback {
        None,
        Preload,
        PreLayers,
        PostLayers,
        PostFn,
        PostFnYield,
        CallbackYieldPostFn,
        DoubleCallbackYieldPostFn,
    };

    enum class TraceCallback {
        ActiveChunk0NoopFollowers,
        ActiveChunk0NoopFollowersAlternateTopology,
        AccessWithoutNode,
        NodeWithoutAccess,
        KvScopeDrift,
        PerLayerDrift,
        AllFacades,
        ChildGraphNode,
        HostCallbackNode,
        Tier3Node,
        LayerProducerYield,
        KvProducerYield,
    };

    enum class PreloadAccess {
        None,
        Persistent,
        Temporary,
    };

    explicit ResolvedTraversalContractBase(
        Traversal traversal = Traversal::Sequential)
        : traversal_(traversal) {
        set_model_params(1, 1, 16, 16, 32);
        set_num_layers(kNumLayers);
    }

    v3::SpmPipelineComponentLayout prepare_cpu(
        int64_t chunk_size = 16,
        int64_t max_kv_seq_len = 32,
        int64_t kv_insert_chunk_size = 0) {
        v3::LayoutContext context;
        context.chunk_size = chunk_size;
        context.kv_insert_chunk_size = kv_insert_chunk_size;
        context.max_kv_seq_len = max_kv_seq_len;
        context.num_layers = kNumLayers;
        const v3::ChunkMode mode = traversal_ == Traversal::KvFirst
            ? v3::ChunkMode::KV_FIRST
            : v3::ChunkMode::SEQUENTIAL;
        return prepare_spm_pipeline_component_for_cpu_contract(
            context, v3::compose_fmb_default_three_stage_chunk_plan(
                         context, max_kv_seq_len, /*position=*/0, mode));
    }

    v3::SpmPipelineComponentLayout prepare_live(
        int64_t chunk_size = 16,
        int64_t max_kv_seq_len = 32,
        int64_t kv_insert_chunk_size = 0) {
        v3::LayoutContext context;
        context.chunk_size = chunk_size;
        context.kv_insert_chunk_size = kv_insert_chunk_size;
        context.max_kv_seq_len = max_kv_seq_len;
        context.num_layers = kNumLayers;
        const v3::ChunkMode mode = traversal_ == Traversal::KvFirst
            ? v3::ChunkMode::KV_FIRST
            : v3::ChunkMode::SEQUENTIAL;
        return prepare_spm_pipeline_component(
            context, v3::compose_fmb_default_three_stage_chunk_plan(
                         context, max_kv_seq_len, /*position=*/0, mode));
    }

    v3::SpmFmbResolvedExecutionProfile resolve(
        const v3::SpmFmbResolvedProfileRequest& request) {
        return resolve_spm_pipeline_execution_profile_for_cpu_contract(
            request);
    }

    v3::SpmPipelineCausalPrefillShape resolve_causal_prefill(
        int64_t execution_len) {
        return resolve_spm_pipeline_causal_prefill_shape_for_cpu_contract(
            execution_len);
    }

    v3::SpmFmbResolvedExecutionProfile resolve_live(
        const v3::SpmFmbResolvedProfileRequest& request) {
        return resolve_spm_pipeline_execution_profile_for_build_trace(
            request);
    }

    v3::SpmFmbResolvedPhaseManifest seal(
        const v3::SpmFmbResolvedExecutionProfile& profile,
        uint32_t arena,
        uint32_t base = 0) const {
        return seal_spm_pipeline_resolved_manifest(
            profile, v3::SpmScratchId(arena), base);
    }

    v3::SpmFmbResolvedPhaseManifest seal_live(
        const v3::SpmFmbResolvedExecutionProfile& profile,
        uint32_t arena,
        uint32_t base = 0) const {
        return seal_spm_pipeline_live_resolved_manifest(
            profile, v3::SpmScratchId(arena), base);
    }

    v3::SpmFmbConsumerRowSliceEndpoint seal_consumer_endpoint(
        const v3::SpmFmbResolvedExecutionProfile& profile,
        const v3::SpmFmbResolvedPhaseManifest& manifest,
        const char* buffer_name,
        int allocation_layer,
        v3::SpmChunkKey consumer_chunk,
        const v3::SpmDense2DSpec& storage_spec,
        int64_t row_begin,
        int64_t row_count,
        const v3::SpmFmbConsumerOccurrenceSelector& occurrence) const {
        return seal_spm_pipeline_consumer_row_slice_endpoint(
            profile, manifest, buffer_name, allocation_layer,
            consumer_chunk, storage_spec, row_begin, row_count, occurrence);
    }

    void cancel_cpu_prepare() {
        cancel_spm_pipeline_component_for_cpu_contract();
    }

    void begin_trace(
        const v3::SpmFmbResolvedExecutionProfile& profile) {
        begin_spm_pipeline_build_trace(profile);
    }

    void drive_trace_cpu(
        const v3::SpmFmbResolvedExecutionProfile& profile) {
        drive_spm_pipeline_build_trace_for_cpu_contract(profile);
    }

    v3::SpmFmbSealedBuildTrace seal_trace(
        const v3::SpmFmbResolvedExecutionProfile& profile,
        const RpuKernelGraph& graph) {
        return seal_spm_pipeline_build_trace(profile, graph);
    }

    v3::SpmFmbSealedPostFnYields seal_post_fn_yields(
        const v3::SpmFmbSealedBuildTrace& trace,
        const RpuKernelGraph& graph) const {
        return seal_spm_pipeline_post_fn_yields(trace, graph);
    }

    v3::SpmFmbSealedCallbackYields seal_callback_yields(
        const v3::SpmFmbSealedBuildTrace& trace,
        const RpuKernelGraph& graph) const {
        return seal_spm_pipeline_callback_yields(trace, graph);
    }

    void validate_callback_yields(
        const v3::SpmFmbSealedCallbackYields& yields,
        const RpuKernelGraph& graph) const {
        validate_spm_pipeline_callback_yields(yields, graph);
    }

    void arm_callback_yield_replay(
        const v3::SpmFmbSealedCallbackYields& yields,
        RpuKernelGraph& graph) const {
        arm_spm_pipeline_callback_yield_replay(yields, graph);
    }

    void drive_callback_yield_replay(
        const v3::SpmFmbSealedCallbackYields& yields) {
        drive_spm_pipeline_callback_yield_replay_for_cpu_contract(yields);
    }

    void drive_composite_callback_yield_replay() {
        drive_spm_pipeline_composite_callback_yield_replay_for_cpu_contract();
    }

    void validate_post_fn_yields(
        const v3::SpmFmbSealedPostFnYields& yields,
        const RpuKernelGraph& graph) const {
        validate_spm_pipeline_post_fn_yields(yields, graph);
    }

    void arm_post_fn_yield_replay(
        const v3::SpmFmbSealedPostFnYields& yields,
        RpuKernelGraph& graph) const {
        arm_spm_pipeline_post_fn_yield_replay(yields, graph);
    }

    void drive_post_fn_yield_replay(
        const v3::SpmFmbSealedPostFnYields& yields) {
        drive_spm_pipeline_post_fn_yield_replay_for_cpu_contract(yields);
    }

    void validate_trace(
        const v3::SpmFmbSealedBuildTrace& trace,
        const RpuKernelGraph& graph) const {
        validate_spm_pipeline_build_trace(trace, graph);
    }

    v3::SpmFmbSealedActiveGroups seal_active_groups(
        const v3::SpmFmbSealedBuildTrace& trace,
        const RpuKernelGraph& graph) const {
        return seal_spm_pipeline_active_groups(trace, graph);
    }

    void validate_active_groups(
        const v3::SpmFmbSealedActiveGroups& groups,
        const RpuKernelGraph& graph) const {
        validate_spm_pipeline_active_groups(groups, graph);
    }

    v3::SpmFmbSealedDenseDdrMembers seal_dense_ddr_members(
        const v3::SpmFmbSealedActiveGroups& groups,
        const RpuKernelGraph& graph) const {
        return seal_spm_pipeline_dense_ddr_members(groups, graph);
    }

    void validate_dense_ddr_members(
        const v3::SpmFmbSealedDenseDdrMembers& members,
        const RpuKernelGraph& graph) const {
        validate_spm_pipeline_dense_ddr_members(members, graph);
    }

    GraphDmaSemanticEndpoint mint_dma_endpoint(
        const v3::SpmFmbResolvedExecutionProfile& profile,
        size_t layer_group_ordinal,
        size_t member_ordinal,
        v3::SpmFmbDmaEndpointRole role,
        uint8_t expected_occurrence_count) const {
        return mint_spm_pipeline_dma_endpoint(
            profile, layer_group_ordinal, member_ordinal, role,
            expected_occurrence_count);
    }

    v3::SpmFmbSealedDmaEndpointBindings seal_dma_endpoint_bindings(
        const v3::SpmFmbSealedDenseDdrMembers& members,
        const RpuKernelGraph& graph) const {
        return seal_spm_pipeline_dma_endpoint_bindings(members, graph);
    }

    void validate_dma_endpoint_bindings(
        const v3::SpmFmbSealedDmaEndpointBindings& bindings,
        const RpuKernelGraph& graph) const {
        validate_spm_pipeline_dma_endpoint_bindings(bindings, graph);
    }

    void arm_dma_endpoint_replay(
        const v3::SpmFmbSealedDmaEndpointBindings& bindings,
        RpuKernelGraph& graph) const {
        arm_spm_pipeline_dma_endpoint_replay(bindings, graph);
    }

    void cancel_trace() { cancel_spm_pipeline_build_trace(); }
    void invalidate_for_contract() { invalidate_model_state(); }

    void drive_preload_for_contract() {
        drive_preload_callbacks_for_test();
    }

    void touch_unscoped_for_contract() {
        (void)addr_offset("resolved_layerwide");
    }

    void touch_persistent_unscoped_for_contract() {
        (void)addr_offset("resolved_persistent");
    }

    void set_traversal(Traversal traversal) { traversal_ = traversal; }
    void set_body_iterations(int64_t iterations) {
        body_iterations_ = iterations;
    }
    void set_opaque_callback(OpaqueCallback callback) {
        opaque_callback_ = callback;
    }
    void set_outside_temp(bool enabled) { outside_temp_ = enabled; }
    void set_persistent_preload(bool enabled) {
        persistent_preload_ = enabled;
    }
    void set_preload_access(PreloadAccess access) {
        preload_access_ = access;
    }
    void set_temporary_preload(bool enabled) {
        temporary_preload_ = enabled;
    }
    void set_trace_callback(TraceCallback callback) {
        trace_callback_ = callback;
    }
    void set_trace_per_layer(bool enabled) {
        trace_per_layer_ = enabled;
    }
    void set_hidden_size_for_contract(int64_t hidden_size) {
        set_model_params(1, 1, 16, hidden_size, 32);
    }
    std::pair<uintptr_t, uint64_t> exchange_composite_input_slot(
        uint64_t value) {
        return exchange_spm_pipeline_composite_input_slot_for_cpu_contract(
            value);
    }
    bool consume_composite_preload_role() {
        return consume_spm_pipeline_composite_preload_role_for_cpu_contract();
    }

protected:
    std::vector<v3::BufferDecl> declare_buffers(
        const v3::LayoutContext&) override {
        using v3::BufferDecl;
        using v3::BufferScope;
        using v3::StorageClass;
        BufferDecl persistent{
            "resolved_persistent", 256, 0, 0,
            StorageClass::Persistent};
        if (persistent_preload_) {
            persistent.preload_callback =
                [](v3::FusedModelBase& base, int, uint32_t) {
                    static_cast<ResolvedTraversalContractBase&>(base)
                        .emit_preload_access();
                };
        }
        BufferDecl compute{
            "resolved_compute", 256, 2, 3, StorageClass::Temp, 0,
            nullptr, BufferScope::Compute};
        if (temporary_preload_) {
            compute.preload_callback =
                [](v3::FusedModelBase&, int, uint32_t) {};
        }
        std::vector<BufferDecl> decls = {
            {"resolved_layerwide", 256, 0, 1, StorageClass::Temp, 0,
             nullptr, BufferScope::LayerWide},
            std::move(compute),
            std::move(persistent),
        };
        if (traversal_ == Traversal::KvFirst) {
            decls.push_back(
                {"resolved_kv_insert", 256, 4, 5, StorageClass::Temp, 0,
                 nullptr, BufferScope::KvInsert});
        }
        if (outside_temp_) {
            decls.push_back(
                {"resolved_outside", 256, 6, 7, StorageClass::Temp, 0,
                 nullptr, BufferScope::OutsideLayerLoop});
        }
        if (trace_per_layer_) {
            decls.push_back(
                {"resolved_per_layer", 256, 2, 3,
                 StorageClass::TempPerLayer, kNumLayers, nullptr,
                 BufferScope::Compute});
        }
        return decls;
    }

    v3::ModelStaticConfig static_config() override {
        v3::ModelStaticConfig config;
        config.num_layers = kNumLayers;
        config.cross_layer_batch_size =
            traversal_ == Traversal::SpmResidentSingleLayerGroups
            ? 1
            : kNumLayers;
        config.body_iterations = body_iterations_;
        if (traversal_ == Traversal::KvFirst) {
            config.kv_first_fn =
                static_cast<void (v3::FusedModelBase::*)(
                    int, const v3::ChunkInfo&)>(
                    &ResolvedTraversalContractBase::kv_first_stub);
        }
        switch (opaque_callback_) {
        case OpaqueCallback::None:
            break;
        case OpaqueCallback::Preload:
            config.preload_fn =
                static_cast<void (v3::FusedModelBase::*)()>(
                    &ResolvedTraversalContractBase::callback_stub);
            break;
        case OpaqueCallback::PreLayers:
            config.pre_layers_fn =
                static_cast<void (v3::FusedModelBase::*)()>(
                    &ResolvedTraversalContractBase::callback_stub);
            break;
        case OpaqueCallback::PostLayers:
            config.post_layers_fn =
                static_cast<void (v3::FusedModelBase::*)()>(
                    &ResolvedTraversalContractBase::callback_stub);
            break;
        case OpaqueCallback::PostFn:
            config.post_fn =
                static_cast<void (v3::FusedModelBase::*)()>(
                    &ResolvedTraversalContractBase::callback_stub);
            break;
        case OpaqueCallback::PostFnYield:
            config.post_fn =
                static_cast<void (v3::FusedModelBase::*)()>(
                    &ResolvedTraversalContractBase::post_fn_yield_stub);
            break;
        case OpaqueCallback::CallbackYieldPostFn:
            config.post_fn =
                static_cast<void (v3::FusedModelBase::*)()>(
                    &ResolvedTraversalContractBase::callback_yield_stub);
            break;
        case OpaqueCallback::DoubleCallbackYieldPostFn:
            config.post_fn =
                static_cast<void (v3::FusedModelBase::*)()>(
                    &ResolvedTraversalContractBase::
                        double_callback_yield_stub);
            break;
        }
        return config;
    }

    v3::ModelDynamicConfig dynamic_config(
        const v3::ChunkPlan&) override {
        v3::ModelDynamicConfig config;
        switch (traversal_) {
        case Traversal::Sequential:
            config.chunk_mode = v3::ChunkMode::SEQUENTIAL;
            config.inter_layer_io = v3::InterLayerIO::DDR_PINGPONG;
            break;
        case Traversal::SpmResidentSingleLayerGroups:
            config.chunk_mode = v3::ChunkMode::SEQUENTIAL;
            config.inter_layer_io = v3::InterLayerIO::SPM_RESIDENT;
            break;
        case Traversal::ChunkOuter:
            config.chunk_mode = v3::ChunkMode::SEQUENTIAL;
            config.inter_layer_io = v3::InterLayerIO::SPM_RESIDENT;
            config.chunk_outer_within_group = true;
            break;
        case Traversal::KvFirst:
            config.chunk_mode = v3::ChunkMode::KV_FIRST;
            config.inter_layer_io = v3::InterLayerIO::DDR_PINGPONG;
            break;
        }
        return config;
    }

    void build_layer_subgraph(
        int layer, const v3::ChunkInfo& chunk) override {
        const bool active_first_run =
            trace_callback_ == TraceCallback::ActiveChunk0NoopFollowers ||
            trace_callback_ ==
                TraceCallback::ActiveChunk0NoopFollowersAlternateTopology;
        if (active_first_run &&
            chunk.idx != 0) {
            return;
        }
        if (trace_callback_ == TraceCallback::NodeWithoutAccess) {
            emit_trace_nodes(layer, chunk);
            return;
        }
        if (trace_callback_ == TraceCallback::LayerProducerYield) {
            (void)addr_offset("resolved_layerwide");
            (void)addr_offset("resolved_compute");
            emit_trace_nodes(layer, chunk);
            if (chunk.idx == 0) {
                emit_callback_yield();
            }
            return;
        }
        if (trace_callback_ == TraceCallback::PerLayerDrift) {
            (void)layer_addr_offset(
                (layer + 1) % static_cast<int>(kNumLayers),
                "resolved_per_layer");
            emit_trace_nodes(layer, chunk);
            return;
        }
        if (trace_callback_ == TraceCallback::AllFacades) {
            (void)addr(0, "resolved_layerwide");
            (void)addr_offset("resolved_compute");
            (void)layer_addr(layer, 0, "resolved_per_layer");
            (void)layer_addr_offset(layer, "resolved_per_layer");
            emit_trace_nodes(layer, chunk);
            return;
        }
        (void)addr_offset("resolved_layerwide");
        (void)addr_offset("resolved_compute");
        if (trace_callback_ == TraceCallback::AccessWithoutNode) {
            return;
        }
        if (trace_callback_ == TraceCallback::ChildGraphNode) {
            RpuKernelGraph::active().capture_data(ChildGraphNodeData{});
            return;
        }
        if (trace_callback_ == TraceCallback::HostCallbackNode) {
            RpuKernelGraph::active().capture_data(HostCallbackNodeData{});
            return;
        }
        if (trace_callback_ == TraceCallback::Tier3Node) {
            RpuKernelGraph::active().capture_data(Tier3OneshotNodeData{});
            return;
        }
        emit_trace_nodes(layer, chunk);
        if (trace_callback_ ==
            TraceCallback::ActiveChunk0NoopFollowersAlternateTopology) {
            RpuKernelGraph::active().record_branch(
                (static_cast<uint64_t>(layer + 1) << 40) | 0xa9,
                GraphSignature{}, "spm_schema_v9_alt_topology");
            RpuKernelGraph::active().capture_data(BranchNodeData{});
        }
    }

    void callback_stub() {}
    void emit_preload_access() {
        if (preload_access_ == PreloadAccess::Persistent) {
            (void)addr_offset("resolved_persistent");
        } else if (preload_access_ == PreloadAccess::Temporary) {
            (void)addr_offset("resolved_compute");
        }
    }
    void post_fn_yield_stub() {
        auto yield = begin_spm_pipeline_post_fn_dense_yield(
            "resolved_layerwide", /*allocation_layer=*/-1,
            dense_fp16(/*rows=*/2, /*cols=*/64));
        const uint32_t input = addr(0, "resolved_compute");
        const uint32_t residual = addr(0, "resolved_persistent");
        rpu_launch_all_reduce_sum_residual_kernel(
            input, residual, yield.target(), /*M=*/2, /*N=*/64,
            /*input_num_cores=*/8, /*output_num_cores=*/8);
        yield.finish();
    }
    void callback_yield_stub() { emit_callback_yield(); }
    void double_callback_yield_stub() {
        emit_callback_yield();
        emit_callback_yield();
    }
    void emit_callback_yield() {
        auto yield = begin_spm_pipeline_dense_producer_yield(
            "resolved_layerwide", /*allocation_layer=*/-1,
            dense_fp16(/*rows=*/2, /*cols=*/64));
        const uint32_t input = addr(0, "resolved_compute");
        const uint32_t residual = addr(0, "resolved_persistent");
        rpu_launch_all_reduce_sum_residual_kernel(
            input, residual, yield.target(), /*M=*/2, /*N=*/64,
            /*input_num_cores=*/8, /*output_num_cores=*/8);
        yield.finish();
    }
    void kv_first_stub(int layer, const v3::ChunkInfo& chunk) {
        if (trace_callback_ == TraceCallback::KvScopeDrift) {
            (void)addr_offset("resolved_compute");
            return;
        }
        if (trace_callback_ == TraceCallback::KvProducerYield) {
            emit_callback_yield();
            return;
        }
        (void)addr_offset("resolved_layerwide");
        (void)addr_offset("resolved_kv_insert");
        emit_trace_nodes(layer, chunk);
    }

    void emit_trace_nodes(int layer, const v3::ChunkInfo& chunk) {
        auto& graph = RpuKernelGraph::active();
        graph.record_branch(
            (static_cast<uint64_t>(layer + 1) << 32) |
                static_cast<uint32_t>(chunk.idx + 1),
            GraphSignature{}, "spm_schema_v8_cpu_fixture");
        graph.capture_data(BranchNodeData{});
    }

private:
    static constexpr int64_t kNumLayers = 2;
    Traversal traversal_;
    int64_t body_iterations_ = 1;
    OpaqueCallback opaque_callback_ = OpaqueCallback::None;
    bool outside_temp_ = false;
    bool persistent_preload_ = false;
    bool temporary_preload_ = false;
    PreloadAccess preload_access_ = PreloadAccess::None;
    TraceCallback trace_callback_ =
        TraceCallback::ActiveChunk0NoopFollowers;
    bool trace_per_layer_ = false;
};

class ResolvedTraversalContractModel final
    : public ResolvedTraversalContractBase {
public:
    using ResolvedTraversalContractBase::ResolvedTraversalContractBase;

protected:
    v3::SpmFmbTraversalCapability
    spm_fmb_traversal_capability() const override {
        return v3::SpmFmbTraversalCapability::CanonicalTraversal;
    }
    v3::SpmFmbActiveGroupCapability
    spm_fmb_active_group_capability() const override {
        return v3::SpmFmbActiveGroupCapability::
            CanonicalActiveFirstChunkRun;
    }
};

class ResolvedTraversalPostFnYieldContractModel final
    : public ResolvedTraversalContractBase {
public:
    using ResolvedTraversalContractBase::ResolvedTraversalContractBase;

protected:
    v3::SpmFmbTraversalCapability
    spm_fmb_traversal_capability() const override {
        return v3::SpmFmbTraversalCapability::CanonicalTraversal;
    }
    v3::SpmFmbPostFnYieldCapability
    spm_fmb_post_fn_yield_capability() const override {
        return v3::SpmFmbPostFnYieldCapability::
            CanonicalDenseReplicatedFp16;
    }
    uint64_t spm_fmb_post_fn_yield_policy_fingerprint() const override {
        return 0x706f7374666e0001ULL;
    }
};

class ResolvedTraversalCallbackYieldContractModel final
    : public ResolvedTraversalContractBase {
public:
    using ResolvedTraversalContractBase::ResolvedTraversalContractBase;

protected:
    v3::SpmFmbTraversalCapability
    spm_fmb_traversal_capability() const override {
        return v3::SpmFmbTraversalCapability::CanonicalTraversal;
    }
    v3::SpmFmbLayerProducerYieldCapability
    spm_fmb_layer_producer_yield_capability() const override {
        return v3::SpmFmbLayerProducerYieldCapability::
            CanonicalDenseReplicatedFp16;
    }
    uint64_t
    spm_fmb_layer_producer_yield_policy_fingerprint() const override {
        return 0x63616c6c62616301ULL;
    }
};

class ResolvedTraversalCompositeSlotsContractModel final
    : public ResolvedTraversalContractBase {
public:
    using ResolvedTraversalContractBase::ResolvedTraversalContractBase;
    void set_composite_slots_enabled(bool enabled) {
        composite_slots_enabled_ = enabled;
    }

protected:
    v3::SpmFmbTraversalCapability
    spm_fmb_traversal_capability() const override {
        return v3::SpmFmbTraversalCapability::CanonicalTraversal;
    }
    v3::SpmFmbPreloadCapability
    spm_fmb_preload_capability() const override {
        return v3::SpmFmbPreloadCapability::
            PersistentOutsideResolvedWindow;
    }
    v3::SpmFmbLayerProducerYieldCapability
    spm_fmb_layer_producer_yield_capability() const override {
        return v3::SpmFmbLayerProducerYieldCapability::
            CanonicalDenseReplicatedFp16;
    }
    uint64_t
    spm_fmb_layer_producer_yield_policy_fingerprint() const override {
        return UINT64_C(0x63616c6c62616301);
    }
    v3::SpmFmbCompositeOccurrenceCapability
    spm_fmb_composite_occurrence_capability() const override {
        return composite_slots_enabled_
            ? v3::SpmFmbCompositeOccurrenceCapability::
                  StableThreeInputSlotsSameOwnerPreload
            : v3::SpmFmbCompositeOccurrenceCapability::Unsealed;
    }
    uint64_t
    spm_fmb_composite_occurrence_policy_fingerprint() const override {
        return composite_slots_enabled_
            ? UINT64_C(0x33696e7075740001)
            : 0;
    }

private:
    bool composite_slots_enabled_ = true;
};

class ResolvedTraversalPersistentPreloadContractModel final
    : public ResolvedTraversalContractBase {
public:
    using ResolvedTraversalContractBase::ResolvedTraversalContractBase;

protected:
    v3::SpmFmbTraversalCapability
    spm_fmb_traversal_capability() const override {
        return v3::SpmFmbTraversalCapability::CanonicalTraversal;
    }
    v3::SpmFmbPreloadCapability
    spm_fmb_preload_capability() const override {
        return v3::SpmFmbPreloadCapability::
            PersistentOutsideResolvedWindow;
    }
};

// Live fixture for the Graph-independent composite reservation.  The optional
// stable-producer mode admits only the first lexical occurrence so the physical
// adoption/current-generation guard can be tested without inventing a kernel
// body or exposing a raw SPM address.
class CompositeReservationGeometryContractModel final
    : public v3::FusedModelBase {
public:
    CompositeReservationGeometryContractModel(
        size_t root_bytes,
        size_t total_temporary_bytes,
        int64_t num_layers,
        int64_t hidden_size,
        bool stable_producer = false)
        : root_bytes_(root_bytes),
          total_temporary_bytes_(total_temporary_bytes),
          num_layers_(num_layers),
          stable_producer_(stable_producer) {
        TORCH_CHECK(root_bytes_ > 0 &&
                        total_temporary_bytes_ >= root_bytes_ &&
                        num_layers_ > 0 && hidden_size > 0,
                    "invalid composite reservation geometry fixture");
        set_model_params(/*num_q_heads=*/1, /*num_kv_heads=*/1,
                         /*head_dim=*/16, hidden_size,
                         /*intermediate_size=*/32);
        set_num_layers(num_layers_);
    }

    v3::SpmPipelineComponentLayout prepare_live() {
        v3::LayoutContext context;
        context.chunk_size = 16;
        context.max_kv_seq_len = 16;
        context.num_layers = num_layers_;
        return prepare_spm_pipeline_component(
            context, v3::compose_fmb_default_three_stage_chunk_plan(
                         context, /*execution_len=*/16, /*position=*/0,
                         v3::ChunkMode::SEQUENTIAL));
    }

    v3::SpmFmbResolvedExecutionProfile resolve_live() {
        v3::SpmFmbResolvedProfileRequest request;
        request.version = 1;
        request.chunks = {{0, 0, 16, 16}};
        return resolve_spm_pipeline_execution_profile_for_build_trace(
            request);
    }

    v3::SpmFmbResolvedPhaseManifest seal_live(
        const v3::SpmFmbResolvedExecutionProfile& profile,
        uint32_t arena,
        uint32_t base) const {
        return seal_spm_pipeline_live_resolved_manifest(
            profile, v3::SpmScratchId(arena), base);
    }

    v3::SpmFmbConsumerRowSliceEndpoint seal_endpoint(
        const v3::SpmFmbResolvedExecutionProfile& profile,
        const v3::SpmFmbResolvedPhaseManifest& manifest,
        const v3::SpmDense2DSpec& storage_spec,
        int64_t row_begin,
        int64_t row_count,
        int layer) const {
        const v3::SpmFmbConsumerOccurrenceSelector occurrence{
            v3::SpmFmbOccurrenceKind::LayerBody,
            /*body_id=*/0, /*group_id=*/0, layer,
            /*chunk_index=*/0};
        return seal_spm_pipeline_consumer_row_slice_endpoint(
            profile, manifest, "root", /*allocation_layer=*/-1,
            v3::SpmChunkKey(v3::SpmPortId(301), 0), storage_spec,
            row_begin, row_count, occurrence);
    }

protected:
    std::vector<v3::BufferDecl> declare_buffers(
        const v3::LayoutContext&) override {
        using v3::BufferDecl;
        using v3::BufferScope;
        using v3::StorageClass;
        std::vector<BufferDecl> result = {
            {"root", static_cast<int64_t>(root_bytes_), 0, 1,
             StorageClass::Temp, 0, nullptr, BufferScope::LayerWide},
        };
        if (total_temporary_bytes_ != root_bytes_) {
            result.push_back(
                {"tail",
                 static_cast<int64_t>(total_temporary_bytes_ - root_bytes_),
                 // Keep the root at [0,1], but widen the callback so the same
                 // root has non-adjacent canonical event slices across layers.
                 0, 2, StorageClass::Temp, 0, nullptr,
                 BufferScope::LayerWide});
        }
        return result;
    }

    v3::ModelStaticConfig static_config() override {
        v3::ModelStaticConfig config;
        config.num_layers = num_layers_;
        config.cross_layer_batch_size = num_layers_;
        config.body_iterations = 1;
        return config;
    }

    v3::ModelDynamicConfig dynamic_config(
        const v3::ChunkPlan&) override {
        v3::ModelDynamicConfig config;
        config.chunk_mode = v3::ChunkMode::SEQUENTIAL;
        config.inter_layer_io = v3::InterLayerIO::DDR_PINGPONG;
        return config;
    }

    void build_layer_subgraph(int, const v3::ChunkInfo&) override {}

    v3::SpmFmbTraversalCapability
    spm_fmb_traversal_capability() const override {
        return v3::SpmFmbTraversalCapability::CanonicalTraversal;
    }

    v3::SpmFmbPreloadCapability
    spm_fmb_preload_capability() const override {
        return stable_producer_
            ? v3::SpmFmbPreloadCapability::
                  PersistentOutsideResolvedWindow
            : v3::SpmFmbPreloadCapability::Unsealed;
    }

    v3::SpmFmbLayerProducerYieldCapability
    spm_fmb_layer_producer_yield_capability() const override {
        return stable_producer_
            ? v3::SpmFmbLayerProducerYieldCapability::
                  CanonicalDenseReplicatedFp16
            : v3::SpmFmbLayerProducerYieldCapability::Unsealed;
    }

    uint64_t
    spm_fmb_layer_producer_yield_policy_fingerprint() const override {
        return stable_producer_ ? UINT64_C(0x63616c6c62616301) : 0;
    }

    v3::SpmFmbCompositeOccurrenceCapability
    spm_fmb_composite_occurrence_capability() const override {
        return stable_producer_
            ? v3::SpmFmbCompositeOccurrenceCapability::
                  StableThreeInputSlotsSameOwnerPreload
            : v3::SpmFmbCompositeOccurrenceCapability::Unsealed;
    }

    uint64_t
    spm_fmb_composite_occurrence_policy_fingerprint() const override {
        return stable_producer_ ? UINT64_C(0x6c696e67626f7432) : 0;
    }

private:
    size_t root_bytes_;
    size_t total_temporary_bytes_;
    int64_t num_layers_;
    bool stable_producer_ = false;
};

class ResolvedTraversalUngroupedContractModel final
    : public ResolvedTraversalContractBase {
public:
    using ResolvedTraversalContractBase::ResolvedTraversalContractBase;

protected:
    v3::SpmFmbTraversalCapability
    spm_fmb_traversal_capability() const override {
        return v3::SpmFmbTraversalCapability::CanonicalTraversal;
    }
};

class ResolvedTraversalDenseDdrContractModel final
    : public ResolvedTraversalContractBase {
public:
    using ResolvedTraversalContractBase::ResolvedTraversalContractBase;

protected:
    v3::SpmFmbTraversalCapability
    spm_fmb_traversal_capability() const override {
        return v3::SpmFmbTraversalCapability::CanonicalTraversal;
    }
    v3::SpmFmbActiveGroupCapability
    spm_fmb_active_group_capability() const override {
        return v3::SpmFmbActiveGroupCapability::
            CanonicalActiveFirstChunkRun;
    }
    v3::SpmFmbDenseDdrMemberCapability
    spm_fmb_dense_ddr_member_capability() const override {
        return v3::SpmFmbDenseDdrMemberCapability::
            IndependentLocalSequentialDdrPingPong;
    }
};

class ResolvedTraversalDmaEndpointContractModel final
    : public ResolvedTraversalContractBase {
public:
    enum class ReplayProbe {
        None,
        MutableSlot,
        MutableOffset,
        Bytes,
        Channel,
        EndpointId,
        MutableFixedPeer,
        FixedPeer,
    };

    ResolvedTraversalDmaEndpointContractModel()
        : ddr_allocation_(
              rpu::RPUCachingAllocatorAdapter::get().allocate(
                  kDdrAllocationBytes)),
          ddr_base_(::rhino_lkn::RpuGetDevAddr(ddr_allocation_.get())),
          live_input_base_(ddr_address(kLiveInputOffset)),
          alternate_live_input_base_(
              ddr_address(kAlternateLiveInputOffset)) {
        TORCH_CHECK(ddr_base_ != 0,
                    "schema-v11 test fixture DDR allocation has no device "
                    "address");
    }

    void configure_dma_endpoints(
        const v3::SpmFmbResolvedExecutionProfile& profile,
        const std::vector<v3::ChunkInfo>& chunks,
        int ingress_lanes = 2) {
        TORCH_CHECK(ingress_lanes >= 1 && ingress_lanes <= 8,
                    "test fixture ingress_lanes must be 1..8");
        chunks_ = chunks;
        declared_ingress_lanes_ = ingress_lanes;
        emitted_ingress_lanes_ = ingress_lanes;
        endpoints_.clear();
        endpoints_.reserve(
            static_cast<size_t>(num_layers()) * chunks_.size() * 2);
        for (size_t layer = 0;
             layer < static_cast<size_t>(num_layers()); ++layer) {
            for (size_t member = 0; member < chunks_.size(); ++member) {
                endpoints_.push_back(mint_dma_endpoint(
                    profile, layer, member,
                    v3::SpmFmbDmaEndpointRole::Ingress,
                    static_cast<uint8_t>(declared_ingress_lanes_)));
                endpoints_.push_back(mint_dma_endpoint(
                    profile, layer, member,
                    v3::SpmFmbDmaEndpointRole::Egress,
                    /*expected_occurrence_count=*/1));
            }
        }
    }

    void set_live_input_base(uint64_t value) { live_input_base_ = value; }
    uint64_t replay_live_input_base() const {
        return ddr_address(kReplayLiveInputOffset);
    }
    void set_omit_last_egress(bool enabled) {
        omit_last_egress_ = enabled;
    }
    void set_alternate_topology(bool enabled) {
        alternate_topology_ = enabled;
    }
    void set_emitted_ingress_lanes(int lanes) {
        TORCH_CHECK(lanes >= 1 && lanes <= declared_ingress_lanes_,
                    "test fixture emitted ingress lanes are invalid");
        emitted_ingress_lanes_ = lanes;
    }
    void set_channel_gap(bool enabled) { channel_gap_ = enabled; }
    void set_split_mutable_slots(bool enabled) {
        split_mutable_slots_ = enabled;
    }
    void set_misorder_first_endpoint(bool enabled) {
        misorder_first_endpoint_ = enabled;
    }
    void set_inject_legacy_dma(bool enabled) {
        inject_legacy_dma_ = enabled;
    }

    void emit_first_ingress_outside_callback() {
        const auto& ingress = endpoint(
            /*layer=*/0, /*member=*/0,
            v3::SpmFmbDmaEndpointRole::Ingress);
        const auto& chunk = chunks_.at(0);
        graph_dma::emit_mutable_src(
            ingress, &live_input_base_, chunk.offset * 32,
            ingress_spm_address(/*layer=*/0, /*member=*/0, /*lane=*/0),
            chunk.len * 32, /*channel=*/0);
    }

    void replay_dma_fixture() {
        const bool rejected = emit_replay_fixture(ReplayProbe::None);
        TORCH_INTERNAL_ASSERT(!rejected);
    }

    bool replay_dma_fixture_with_probe(ReplayProbe probe) {
        return emit_replay_fixture(probe);
    }

protected:
    v3::SpmFmbTraversalCapability
    spm_fmb_traversal_capability() const override {
        return v3::SpmFmbTraversalCapability::CanonicalTraversal;
    }
    v3::SpmFmbActiveGroupCapability
    spm_fmb_active_group_capability() const override {
        return v3::SpmFmbActiveGroupCapability::
            CanonicalActiveFirstChunkRun;
    }
    v3::SpmFmbDenseDdrMemberCapability
    spm_fmb_dense_ddr_member_capability() const override {
        return v3::SpmFmbDenseDdrMemberCapability::
            IndependentLocalSequentialDdrPingPong;
    }
    v3::SpmFmbDmaEndpointCapability
    spm_fmb_dma_endpoint_capability() const override {
        return v3::SpmFmbDmaEndpointCapability::
            DenseDdrMemberRoleBindings;
    }

    void build_layer_subgraph(
        int layer, const v3::ChunkInfo& chunk) override {
        if (chunk.idx != 0) return;
        TORCH_CHECK(!chunks_.empty() &&
                        endpoints_.size() ==
                            static_cast<size_t>(num_layers()) *
                                chunks_.size() * 2,
                    "schema-v11 test fixture endpoints are not configured");
        auto& graph = RpuKernelGraph::active();
        if (graph.state() == RpuKernelGraph::State::RECORDING) {
            // CPU-dry facade offsets exist only while the FMB BUILD trace is
            // armed.  REPLAY re-emits the sealed Graph nodes and must not
            // pretend that a physical SPM allocation was adopted.
            (void)addr_offset("resolved_layerwide");
            (void)addr_offset("resolved_compute");
        }
        graph.record_branch(
            (static_cast<uint64_t>(layer + 1) << 32) | 1,
            GraphSignature{}, "spm_schema_v11_dma_fixture");
        graph.capture_data(BranchNodeData{});

        for (size_t member = 0; member < chunks_.size(); ++member) {
            const auto& member_chunk = chunks_[member];
            const size_t byte_begin =
                static_cast<size_t>(member_chunk.offset) * 32;
            const size_t byte_count =
                static_cast<size_t>(member_chunk.len) * 32;
            const auto& ingress = endpoint(
                layer, member, v3::SpmFmbDmaEndpointRole::Ingress);
            if (misorder_first_endpoint_ && layer == 0 && member == 0) {
                const auto& early_egress = endpoint(
                    layer, member, v3::SpmFmbDmaEndpointRole::Egress);
                graph_dma::emit_fixed(
                    early_egress, egress_spm_address(layer, member),
                    fixed_egress_address(byte_begin),
                    byte_count, /*channel=*/0);
            }
            for (int lane = 0; lane < emitted_ingress_lanes_; ++lane) {
                const int channel = channel_gap_ && lane == 1 ? 2 : lane;
                const uint64_t spm_dst =
                    ingress_spm_address(layer, member, lane);
                if (layer == 0) {
                    const uint64_t* live_base =
                        split_mutable_slots_ && lane == 1
                        ? &alternate_live_input_base_
                        : &live_input_base_;
                    maybe_probe_mutable_ingress(
                        member, channel, ingress, byte_begin, byte_count,
                        spm_dst);
                    graph_dma::emit_mutable_src(
                        ingress, live_base,
                        static_cast<int64_t>(byte_begin), spm_dst,
                        byte_count, static_cast<uint8_t>(channel));
                } else {
                    graph_dma::emit_fixed(
                        ingress, fixed_ingress_address(byte_begin),
                        spm_dst, byte_count,
                        static_cast<uint8_t>(channel));
                }
            }

            if (omit_last_egress_ &&
                layer == num_layers() - 1 &&
                member + 1 == chunks_.size()) {
                continue;
            }
            const auto& egress = endpoint(
                layer, member, v3::SpmFmbDmaEndpointRole::Egress);
            const uint64_t spm_src = egress_spm_address(layer, member);
            maybe_probe_fixed_egress(
                layer, member, egress, spm_src, byte_begin, byte_count);
            graph_dma::emit_fixed(
                egress, spm_src, fixed_egress_address(byte_begin),
                byte_count, /*channel=*/0);
        }

        if (inject_legacy_dma_) {
            graph.record_dma_fixed(
                ddr_address(kLegacySrcOffset),
                ddr_address(kLegacyDstOffset),
                /*bytes=*/16, /*channel=*/0);
        }

        if (alternate_topology_) {
            graph.record_branch(
                (static_cast<uint64_t>(layer + 1) << 40) | 0xb1,
                GraphSignature{}, "spm_schema_v11_alt_topology");
            graph.capture_data(BranchNodeData{});
        }
    }

private:
    static constexpr size_t kDdrAllocationBytes = 32 * 1024;
    static constexpr size_t kLiveInputOffset = 0x0000;
    static constexpr size_t kAlternateLiveInputOffset = 0x1000;
    static constexpr size_t kReplayLiveInputOffset = 0x2000;
    static constexpr size_t kFixedIngressOffset = 0x3000;
    static constexpr size_t kFixedEgressOffset = 0x4000;
    static constexpr size_t kLegacySrcOffset = 0x5000;
    static constexpr size_t kLegacyDstOffset = 0x6000;
    static constexpr size_t kIngressSpmOffset = 0x10000;
    static constexpr size_t kEgressSpmOffset = 0x50000;
    static constexpr size_t kLayerSpmStride = 0x10000;
    static constexpr size_t kMemberSpmStride = 0x1000;

    uint64_t ddr_address(size_t offset) const {
        return ddr_base_ + offset;
    }

    uint64_t spm_address(int core, size_t offset) const {
        auto* root = SPM_ALLOC.root_buffer(core);
        TORCH_CHECK(root != nullptr,
                    "schema-v11 test fixture requires initialized SPM root "
                    "for core ", core);
        return root->get_rpu_addr() + offset;
    }

    uint64_t ingress_spm_address(
        int layer, size_t member, int lane) const {
        return spm_address(
            lane, kIngressSpmOffset +
                static_cast<size_t>(layer) * kLayerSpmStride +
                member * kMemberSpmStride);
    }

    uint64_t egress_spm_address(int layer, size_t member) const {
        return spm_address(
            /*core=*/0, kEgressSpmOffset +
                static_cast<size_t>(layer) * kLayerSpmStride +
                member * kMemberSpmStride);
    }

    uint64_t fixed_ingress_address(size_t byte_begin) const {
        return ddr_address(kFixedIngressOffset + byte_begin);
    }

    uint64_t fixed_egress_address(size_t byte_begin) const {
        return ddr_address(kFixedEgressOffset + byte_begin);
    }

    size_t endpoint_index(
        int layer,
        size_t member,
        v3::SpmFmbDmaEndpointRole role) const {
        return (static_cast<size_t>(layer) * chunks_.size() + member) * 2 +
            (role == v3::SpmFmbDmaEndpointRole::Egress ? 1 : 0);
    }

    const GraphDmaSemanticEndpoint& endpoint(
        int layer,
        size_t member,
        v3::SpmFmbDmaEndpointRole role) const {
        return endpoints_.at(endpoint_index(layer, member, role));
    }

    void record_probe_result(bool rejected) {
        TORCH_CHECK(!probe_attempted_,
                    "schema-v11 test fixture attempted multiple probes");
        probe_attempted_ = true;
        probe_rejected_ = rejected;
    }

    void maybe_probe_mutable_ingress(
        size_t member,
        int channel,
        const GraphDmaSemanticEndpoint& ingress,
        size_t byte_begin,
        size_t byte_count,
        uint64_t spm_dst) {
        if (probe_ == ReplayProbe::None ||
            probe_ == ReplayProbe::FixedPeer ||
            member != 0 || channel != 0) {
            return;
        }
        const bool rejected = throws_c10([&] {
            switch (probe_) {
            case ReplayProbe::MutableSlot:
                graph_dma::emit_mutable_src(
                    ingress, &alternate_live_input_base_,
                    static_cast<int64_t>(byte_begin), spm_dst,
                    byte_count, /*channel=*/0);
                break;
            case ReplayProbe::MutableOffset:
                graph_dma::emit_mutable_src(
                    ingress, &live_input_base_,
                    static_cast<int64_t>(byte_begin + rpu::kAlignment),
                    spm_dst,
                    byte_count, /*channel=*/0);
                break;
            case ReplayProbe::Bytes:
                graph_dma::emit_mutable_src(
                    ingress, &live_input_base_,
                    static_cast<int64_t>(byte_begin), spm_dst,
                    byte_count + rpu::kAlignment, /*channel=*/0);
                break;
            case ReplayProbe::Channel:
                graph_dma::emit_mutable_src(
                    ingress, &live_input_base_,
                    static_cast<int64_t>(byte_begin), spm_dst,
                    byte_count, /*channel=*/1);
                break;
            case ReplayProbe::EndpointId: {
                const auto& wrong = endpoint(
                    /*layer=*/0, /*member=*/1,
                    v3::SpmFmbDmaEndpointRole::Ingress);
                const auto& wrong_chunk = chunks_.at(1);
                graph_dma::emit_mutable_src(
                    wrong, &live_input_base_,
                    wrong_chunk.offset * 32, spm_dst,
                    wrong_chunk.len * 32, /*channel=*/0);
                break;
            }
            case ReplayProbe::MutableFixedPeer:
                graph_dma::emit_mutable_src(
                    ingress, &live_input_base_,
                    static_cast<int64_t>(byte_begin),
                    spm_dst + rpu::kAlignment,
                    byte_count, /*channel=*/0);
                break;
            case ReplayProbe::FixedPeer:
            case ReplayProbe::None:
                TORCH_INTERNAL_ASSERT(false);
            }
        });
        record_probe_result(rejected);
    }

    void maybe_probe_fixed_egress(
        int layer,
        size_t member,
        const GraphDmaSemanticEndpoint& egress,
        uint64_t spm_src,
        size_t byte_begin,
        size_t byte_count) {
        if (probe_ != ReplayProbe::FixedPeer || layer != 0 || member != 0) {
            return;
        }
        const bool rejected = throws_c10([&] {
            graph_dma::emit_fixed(
                egress, spm_src + rpu::kAlignment,
                fixed_egress_address(byte_begin),
                byte_count, /*channel=*/0);
        });
        record_probe_result(rejected);
    }

    bool emit_replay_fixture(ReplayProbe probe) {
        probe_ = probe;
        probe_attempted_ = false;
        probe_rejected_ = false;
        for (int layer = 0; layer < num_layers(); ++layer) {
            for (const auto& chunk : chunks_) {
                build_layer_subgraph(layer, chunk);
            }
        }
        const bool result = probe_attempted_ && probe_rejected_;
        probe_ = ReplayProbe::None;
        return result;
    }

    std::vector<v3::ChunkInfo> chunks_;
    std::vector<GraphDmaSemanticEndpoint> endpoints_;
    int declared_ingress_lanes_ = 2;
    int emitted_ingress_lanes_ = 2;
    at::DataPtr ddr_allocation_;
    uint64_t ddr_base_ = 0;
    uint64_t live_input_base_ = 0;
    uint64_t alternate_live_input_base_ = 0;
    bool omit_last_egress_ = false;
    bool alternate_topology_ = false;
    bool channel_gap_ = false;
    bool split_mutable_slots_ = false;
    bool misorder_first_endpoint_ = false;
    bool inject_legacy_dma_ = false;
    ReplayProbe probe_ = ReplayProbe::None;
    bool probe_attempted_ = false;
    bool probe_rejected_ = false;
};

class ResolvedTraversalRuntimeMemberContractModel
    : public ResolvedTraversalContractBase {
public:
    using ResolvedTraversalContractBase::ResolvedTraversalContractBase;

    void replay_runtime_members(
        const v3::SpmFmbSealedDmaEndpointBindings& bindings) {
        drive_spm_pipeline_dma_endpoint_replay_for_cpu_contract(bindings);
    }

    bool count_outside_callback_rejected() const {
        return throws_c10_containing(
            [&] { (void)current_spm_pipeline_dense_ddr_member_count(); },
            "no active current callback");
    }

    bool escaped_context_outside_callback_rejected() const {
        if (!escaped_build_ref_.has_value()) return false;
        return throws_c10_containing(
            [&] {
                (void)spm_pipeline_member_local_context(
                    *escaped_build_ref_);
            },
            "no active current callback");
    }

    bool safe_build_probes_ok() const { return safe_build_probes_ok_; }
    bool safe_replay_probes_ok() const { return safe_replay_probes_ok_; }
    bool escaped_follower_rejected() const {
        return escaped_follower_rejected_;
    }
    bool escaped_replay_rejected() const {
        return escaped_replay_rejected_;
    }
    bool local_context_ok() const { return local_context_ok_; }
    void set_alternate_topology(bool enabled) {
        alternate_topology_ = enabled;
    }
    void set_omit_first_egress(bool enabled) {
        omit_first_egress_ = enabled;
    }
    void set_inject_opaque_after_cover(bool enabled) {
        inject_opaque_after_cover_ = enabled;
    }
    void set_bypass_probe_child(RpuKernelGraph* child) {
        bypass_probe_child_ = child;
    }
    void set_bypass_probe_foreign_graph(
        RpuKernelGraph* graph,
        ::rhino_lkn::Kernel_t* kernel) {
        bypass_probe_foreign_graph_ = graph;
        bypass_probe_foreign_kernel_ = kernel;
    }

protected:
    virtual const char* runtime_ingress_buffer(int) const {
        return "resolved_layerwide";
    }
    virtual const char* runtime_egress_buffer(int) const {
        return "resolved_layerwide";
    }
    virtual void run_typed_spm_peer_probe(
        const v3::SpmFmbRuntimeDenseDdrMemberRef&,
        int,
        size_t) {}

    v3::SpmFmbTraversalCapability
    spm_fmb_traversal_capability() const override {
        return v3::SpmFmbTraversalCapability::CanonicalTraversal;
    }
    v3::SpmFmbActiveGroupCapability
    spm_fmb_active_group_capability() const override {
        return v3::SpmFmbActiveGroupCapability::
            CanonicalActiveFirstChunkRun;
    }
    v3::SpmFmbDenseDdrMemberCapability
    spm_fmb_dense_ddr_member_capability() const override {
        return v3::SpmFmbDenseDdrMemberCapability::
            IndependentLocalSequentialDdrPingPong;
    }
    v3::SpmFmbDmaEndpointCapability
    spm_fmb_dma_endpoint_capability() const override {
        return v3::SpmFmbDmaEndpointCapability::
            DenseDdrMemberRoleBindings;
    }
    v3::SpmFmbRuntimeMemberCapability
    spm_fmb_runtime_member_capability() const override {
        return v3::SpmFmbRuntimeMemberCapability::
            CurrentCallbackDenseDdrRefs;
    }

    void build_layer_subgraph(
        int layer, const v3::ChunkInfo& chunk) override {
        auto& graph = RpuKernelGraph::active();
        if (chunk.idx != 0) {
            if (!escaped_follower_rejected_ &&
                escaped_build_ref_.has_value()) {
                const GraphOpStreamStamp before = graph.op_stream_stamp();
                const bool rejected = throws_c10_containing(
                    [&] {
                        (void)spm_pipeline_member_local_context(
                            *escaped_build_ref_);
                    },
                    "reference escaped its current callback");
                const GraphOpStreamStamp after = graph.op_stream_stamp();
                escaped_follower_rejected_ =
                    rejected && stamps_equal(before, after);
            }
            return;
        }

        graph.record_branch(
            (static_cast<uint64_t>(layer + 1) << 32) | 1,
            GraphSignature{}, "spm_schema_v12_runtime_member_fixture");
        graph.capture_data(BranchNodeData{});

        const bool recording =
            graph.state() == RpuKernelGraph::State::RECORDING;
        if (!recording && escaped_build_ref_.has_value() &&
            !escaped_replay_rejected_) {
            const GraphOpStreamStamp before = graph.op_stream_stamp();
            const bool rejected = throws_c10_containing(
                [&] {
                    (void)spm_pipeline_member_local_context(
                        *escaped_build_ref_);
                },
                "reference escaped its current callback");
            const GraphOpStreamStamp after = graph.op_stream_stamp();
            escaped_replay_rejected_ = rejected && stamps_equal(before, after);
        }

        const size_t member_count =
            current_spm_pipeline_dense_ddr_member_count();
        TORCH_CHECK(member_count == 3,
                    "schema-v12 fixture expected three runtime members");
        for (size_t ordinal = 0; ordinal < member_count; ++ordinal) {
            auto member = current_spm_pipeline_dense_ddr_member(ordinal);
            const auto local = spm_pipeline_member_local_context(member);
            local_context_ok_ = local_context_ok_ &&
                local.member_ordinal() == ordinal &&
                local.local_row_count() == 16 &&
                local.local_position_begin() == 0 &&
                local.local_kv_seq_len() == 16;
            if (recording && layer == 0 && ordinal == 0 &&
                !escaped_build_ref_.has_value()) {
                escaped_build_ref_ = member;
            }
            if (layer == 0 && ordinal == 0) {
                const bool probes_ok = run_safe_bypass_probes(
                    graph, layer, chunk);
                if (recording) {
                    safe_build_probes_ok_ = probes_ok;
                } else {
                    safe_replay_probes_ok_ = probes_ok;
                }
            }
            run_typed_spm_peer_probe(member, layer, ordinal);
            emit_spm_pipeline_member_ingress_dma(
                member, runtime_ingress_buffer(layer));
            if (omit_first_egress_ && layer == 0 && ordinal == 0) {
                return;
            }
            emit_spm_pipeline_member_egress_dma(
                member, runtime_egress_buffer(layer));
        }
        if (alternate_topology_) {
            graph.record_branch(
                (static_cast<uint64_t>(layer + 1) << 40) | 0xc1,
                GraphSignature{}, "spm_schema_v12_alt_topology");
            graph.capture_data(BranchNodeData{});
        }
        if (inject_opaque_after_cover_ && layer == 0) {
            graph.capture_data(HostCallbackNodeData{});
        }
    }

private:
    static bool stamps_equal(
        const GraphOpStreamStamp& lhs,
        const GraphOpStreamStamp& rhs) {
        return lhs.graph == rhs.graph && lhs.state == rhs.state &&
            lhs.build_generation == rhs.build_generation &&
            lhs.signature_identity == rhs.signature_identity &&
            lhs.signature_segment_key == rhs.signature_segment_key &&
            lhs.position == rhs.position && lhs.extent == rhs.extent;
    }

    bool run_safe_bypass_probes(
        RpuKernelGraph& graph,
        int layer,
        const v3::ChunkInfo& chunk) {
        const auto safe_probe = [&](auto&& fn, const char* message) {
            const GraphOpStreamStamp before = graph.op_stream_stamp();
            const bool rejected = throws_c10_containing(fn, message);
            const GraphOpStreamStamp after = graph.op_stream_stamp();
            return rejected && stamps_equal(before, after);
        };
        const auto safe_external_graph_probe = [&]
                (RpuKernelGraph& target, auto&& fn, const char* message) {
            const GraphOpStreamStamp outer_before = graph.op_stream_stamp();
            const InactiveGraphStamp target_before =
                inactive_graph_stamp(target);
            const bool rejected = throws_c10_containing(fn, message);
            const GraphOpStreamStamp outer_after = graph.op_stream_stamp();
            const InactiveGraphStamp target_after =
                inactive_graph_stamp(target);
            return rejected && stamps_equal(outer_before, outer_after) &&
                target_before == target_after;
        };
        uint64_t raw_live_base = UINT64_C(0x23000000);
        const bool child_replay_probes_ok =
            bypass_probe_child_ == nullptr ||
            (safe_external_graph_probe(
                 *bypass_probe_child_,
                 [&] { bypass_probe_child_->replay_prepared_child({}); },
                 "direct child replay is forbidden") &&
             safe_external_graph_probe(
                 *bypass_probe_child_,
                 [&] {
                     bypass_probe_child_
                         ->replay_prepared_child_with_data_patches({}, {});
                 },
                 "direct child replay is forbidden"));
        const bool foreign_graph_probes_ok =
            bypass_probe_foreign_graph_ == nullptr ||
            (bypass_probe_foreign_kernel_ != nullptr &&
             safe_external_graph_probe(
                 *bypass_probe_foreign_graph_,
                 [&] {
                     (void)bypass_probe_foreign_graph_->get_kernel_reset(
                         KernelId::BINARY_SAMESHAPE);
                 },
                 "foreign Graph kernel lookup") &&
             safe_external_graph_probe(
                 *bypass_probe_foreign_graph_,
                 [&] {
                     bypass_probe_foreign_graph_->enqueue(
                         *bypass_probe_foreign_kernel_, {1}, {0});
                 },
                 "foreign Graph kernel enqueue") &&
             safe_external_graph_probe(
                 *bypass_probe_foreign_graph_,
                 [&] { bypass_probe_foreign_graph_->sync_point(); },
                 "foreign Graph sync_point"));
        return safe_probe(
                   [&] {
                       emit_layer_input_dma(
                           layer, chunk, "resolved_layerwide");
                   },
                   "legacy layer DMA helper is forbidden") &&
            safe_probe(
                   [&] {
                       emit_layer_output_dma(
                           layer, chunk, "resolved_layerwide");
                   },
                   "legacy layer DMA helper is forbidden") &&
            safe_probe(
                   [&] {
                       emit_layer_chain_scratch_store(
                           layer, chunk, "resolved_layerwide");
                   },
                   "legacy layer DMA helper is forbidden") &&
            safe_probe(
                   [&] {
                       (void)stable_layer_input_ddr_ptr(layer, chunk);
                   },
                   "legacy DDR pointer helper is forbidden") &&
            safe_probe(
                   [&] {
                       graph.record_dma_fixed(
                           UINT64_C(0x61000000), UINT64_C(0x41000000),
                           /*bytes=*/16, /*channel=*/0);
                   },
                   "legacy fixed DMA is forbidden") &&
            safe_probe(
                   [&] {
                       graph.record_dma_mutable_src(
                           &raw_live_base, /*src_offset=*/0,
                           UINT64_C(0x41000000), /*bytes=*/16,
                           /*channel=*/0);
                   },
                   "legacy MutableSrc DMA is forbidden") &&
            safe_probe(
                   [&] {
                       graph.record_dma_mutable_dst(
                           UINT64_C(0x41000000), &raw_live_base,
                           /*dst_offset=*/0, /*bytes=*/16,
                           /*channel=*/0);
                   },
                   "legacy MutableDst DMA is forbidden") &&
            safe_probe(
                   [&] {
                       MemcpyNodeData memcpy;
                       memcpy.bytes = 16;
                       memcpy.kind = CopyKind::DDR_TO_DDR;
                       graph.capture_data(std::move(memcpy));
                   },
                   "Memcpy/Memset data node is forbidden") &&
            safe_probe(
                   [&] {
                       MemsetNodeData memset;
                       memset.bytes = 16;
                       graph.capture_data(std::move(memset));
                   },
                   "Memcpy/Memset data node is forbidden") &&
            safe_probe(
                   [&] {
                       (void)layer_chain_scratch_ddr_ptr(layer, chunk);
                   },
                   "legacy DDR pointer helper is forbidden") &&
            safe_probe(
                   [&] {
                       (void)graph.get_kernel_reset(
                           std::string("ddr2spm_multi_core_v2"));
                   },
                   "legacy DDR/SPM transport kernel is forbidden") &&
            safe_probe(
                   [&] {
                       (void)graph.get_kernel_reset(
                           std::string("spm2ddr_multi_core_v2"));
                   },
                   "dynamic legacy DDR/SPM transport kernel is forbidden") &&
            safe_probe(
                   [&] {
                       graph.mark_non_replayable(
                           "schema-v12 safe probe");
                   },
                   "mark_non_replayable is forbidden") &&
            safe_probe(
                   [&] {
                       RpuKernelGraph nested;
                       GraphSignature nested_signature;
                       nested_signature.op_id = 0xcafe;
                       nested.begin(nested_signature);
                   },
                   "nested Graph begin is forbidden") &&
            safe_probe(
                   [&] { graph.abort(); },
                   "abort is forbidden") &&
            safe_probe(
                   [&] { graph.invalidate(); },
                   "Graph invalidation is forbidden while a Graph scope is "
                   "active") &&
            safe_probe(
                   [&] {
                       rpu_launch_ddr_broadcast_spm_dma_mutable(
                           &raw_live_base, /*src_offset_bytes=*/0,
                           /*num_elements=*/256,
                           /*spm_addr_unified=*/0x41000000,
                           /*num_cores=*/8);
                   },
                   "barrier requires the current framework DMA burst permit") &&
            child_replay_probes_ok && foreign_graph_probes_ok;
    }

    std::optional<v3::SpmFmbRuntimeDenseDdrMemberRef> escaped_build_ref_;
    bool safe_build_probes_ok_ = false;
    bool safe_replay_probes_ok_ = false;
    bool escaped_follower_rejected_ = false;
    bool escaped_replay_rejected_ = false;
    bool local_context_ok_ = true;
    bool alternate_topology_ = false;
    bool omit_first_egress_ = false;
    bool inject_opaque_after_cover_ = false;
    RpuKernelGraph* bypass_probe_child_ = nullptr;
    RpuKernelGraph* bypass_probe_foreign_graph_ = nullptr;
    ::rhino_lkn::Kernel_t* bypass_probe_foreign_kernel_ = nullptr;
};

class ResolvedTraversalTypedSpmPeerContractModel final
    : public ResolvedTraversalRuntimeMemberContractModel {
public:
    enum class LayoutPattern {
        MixedA,
        MixedB,
        AllShared,
        AllPacked,
        UnalignedShared,
        UnalignedPacked,
        UnalignedSlack,
        TailPacked,
        Undersized,
        Unsupported,
    };

    using ResolvedTraversalRuntimeMemberContractModel::
        ResolvedTraversalRuntimeMemberContractModel;

    void set_layout_pattern(LayoutPattern pattern) { pattern_ = pattern; }
    void set_peer_capability_enabled(bool enabled) {
        peer_capability_enabled_ = enabled;
    }
    void set_alias_replay_probe(bool enabled) {
        alias_replay_probe_ = enabled;
        alias_probe_attempted_ = false;
        alias_probe_ok_ = false;
    }
    bool alias_probe_ok() const {
        return alias_probe_attempted_ && alias_probe_ok_;
    }

    v3::SpmFmbSealedSpmPeerBindings seal_spm_peer_bindings(
        const v3::SpmFmbSealedDmaEndpointBindings& bindings,
        const RpuKernelGraph& graph) const {
        return seal_spm_pipeline_spm_peer_bindings(bindings, graph);
    }
    void validate_spm_peer_bindings(
        const v3::SpmFmbSealedSpmPeerBindings& bindings,
        const RpuKernelGraph& graph) const {
        validate_spm_pipeline_spm_peer_bindings(bindings, graph);
    }
    void arm_spm_peer_replay(
        const v3::SpmFmbSealedSpmPeerBindings& bindings,
        RpuKernelGraph& graph) const {
        arm_spm_pipeline_spm_peer_replay(bindings, graph);
    }
    void replay_spm_peers(
        const v3::SpmFmbSealedSpmPeerBindings& bindings) {
        drive_spm_pipeline_spm_peer_replay_for_cpu_contract(bindings);
    }

protected:
    v3::SpmFmbSpmPeerCapability
    spm_fmb_spm_peer_capability() const override {
        return peer_capability_enabled_
            ? v3::SpmFmbSpmPeerCapability::TypedDensePackedOrShared
            : v3::SpmFmbSpmPeerCapability::Unsealed;
    }
    uint64_t spm_fmb_spm_peer_policy_fingerprint() const override {
        return peer_capability_enabled_
            ? static_cast<uint64_t>(pattern_) + 1
            : 0;
    }

    std::vector<v3::BufferDecl> declare_buffers(
        const v3::LayoutContext&) override {
        using v3::BufferDecl;
        using v3::BufferScope;
        using v3::StorageClass;
        return {
            {"spm_shared", 512, 0, 1, StorageClass::Temp, 0,
             nullptr, BufferScope::LayerWide},
            {"spm_shared_alias", 512, 0, 1, StorageClass::Temp, 0,
             "spm_shared", BufferScope::LayerWide},
            {"spm_packed", 1536, 0, 1, StorageClass::Temp, 0,
             nullptr, BufferScope::LayerWide},
            {"spm_undersized", 256, 0, 1, StorageClass::Temp, 0,
             nullptr, BufferScope::LayerWide},
            {"spm_unsupported", 768, 0, 1, StorageClass::Temp, 0,
             nullptr, BufferScope::LayerWide},
            {"spm_shared_480", 480, 0, 1, StorageClass::Temp, 0,
             nullptr, BufferScope::LayerWide},
            {"spm_packed_1440", 1440, 0, 1, StorageClass::Temp, 0,
             nullptr, BufferScope::LayerWide},
            {"spm_slack_496", 496, 0, 1, StorageClass::Temp, 0,
             nullptr, BufferScope::LayerWide},
            {"spm_packed_tail", 1280, 0, 1, StorageClass::Temp, 0,
             nullptr, BufferScope::LayerWide},
            {"resolved_compute", 256, 2, 3, StorageClass::Temp, 0,
             nullptr, BufferScope::Compute},
            {"resolved_persistent", 256, 0, 0,
             StorageClass::Persistent},
        };
    }

    const char* runtime_ingress_buffer(int layer) const override {
        switch (pattern_) {
        case LayoutPattern::MixedA:
            return layer == 0 ? "spm_shared" : "spm_packed";
        case LayoutPattern::MixedB:
            return layer == 0 ? "spm_packed" : "spm_shared";
        case LayoutPattern::AllShared:
            return "spm_shared";
        case LayoutPattern::AllPacked:
            return "spm_packed";
        case LayoutPattern::UnalignedShared:
            return "spm_shared_480";
        case LayoutPattern::UnalignedPacked:
            return "spm_packed_1440";
        case LayoutPattern::UnalignedSlack:
            return "spm_slack_496";
        case LayoutPattern::TailPacked:
            return "spm_packed_tail";
        case LayoutPattern::Undersized:
            return "spm_undersized";
        case LayoutPattern::Unsupported:
            return "spm_unsupported";
        }
        TORCH_INTERNAL_ASSERT(false);
        return nullptr;
    }

    const char* runtime_egress_buffer(int layer) const override {
        switch (pattern_) {
        case LayoutPattern::MixedA:
            return layer == 0 ? "spm_packed" : "spm_shared";
        case LayoutPattern::MixedB:
            return layer == 0 ? "spm_shared" : "spm_packed";
        case LayoutPattern::AllShared:
            return "spm_shared";
        case LayoutPattern::AllPacked:
            return "spm_packed";
        case LayoutPattern::UnalignedShared:
            return "spm_shared_480";
        case LayoutPattern::UnalignedPacked:
            return "spm_packed_1440";
        case LayoutPattern::UnalignedSlack:
            return "spm_slack_496";
        case LayoutPattern::TailPacked:
            return "spm_packed_tail";
        case LayoutPattern::Undersized:
            return "spm_undersized";
        case LayoutPattern::Unsupported:
            return "spm_unsupported";
        }
        TORCH_INTERNAL_ASSERT(false);
        return nullptr;
    }

    void run_typed_spm_peer_probe(
        const v3::SpmFmbRuntimeDenseDdrMemberRef& member,
        int layer,
        size_t ordinal) override {
        if (!alias_replay_probe_ || alias_probe_attempted_ || layer != 0 ||
            ordinal != 0 ||
            RpuKernelGraph::active().state() !=
                RpuKernelGraph::State::REPLAYING) {
            return;
        }
        alias_probe_attempted_ = true;
        const GraphOpStreamStamp before =
            RpuKernelGraph::active().op_stream_stamp();
        const bool rejected = throws_c10_containing(
            [&] {
                emit_spm_pipeline_member_ingress_dma(
                    member, "spm_shared_alias");
            },
            "differs from the sealed token");
        const GraphOpStreamStamp after =
            RpuKernelGraph::active().op_stream_stamp();
        alias_probe_ok_ = rejected && before.graph == after.graph &&
            before.state == after.state &&
            before.build_generation == after.build_generation &&
            before.signature_identity == after.signature_identity &&
            before.signature_segment_key == after.signature_segment_key &&
            before.position == after.position &&
            before.extent == after.extent;
    }

private:
    LayoutPattern pattern_ = LayoutPattern::MixedA;
    bool peer_capability_enabled_ = true;
    bool alias_replay_probe_ = false;
    bool alias_probe_attempted_ = false;
    bool alias_probe_ok_ = false;
};

v3::SpmFmbResolvedProfileRequest resolved_request(
    uint64_t version = 1,
    bool include_kv_insert = false) {
    v3::SpmFmbResolvedProfileRequest request;
    request.version = version;
    request.chunks = {
        {0, 0, 16, 16},
        {1, 16, 16, 32},
    };
    if (include_kv_insert) {
        request.kv_insert_chunks = {
            {0, 0, 8, 8},
            {1, 8, 8, 16},
            {2, 16, 8, 24},
            {3, 24, 8, 32},
        };
    }
    return request;
}

v3::SpmFmbResolvedProfileRequest resolved_three_chunk_request(
    uint64_t version = 9) {
    v3::SpmFmbResolvedProfileRequest request;
    request.version = version;
    request.chunks = {
        {0, 0, 16, 16},
        {1, 16, 16, 32},
        {2, 32, 16, 48},
    };
    return request;
}

v3::SpmFmbResolvedProfileRequest resolved_wall_exact_mimc_request() {
    v3::SpmFmbResolvedProfileRequest request;
    request.version = 16;
    request.chunks = {
        {0, 0, 576, 576},
        {1, 576, 576, 1152},
        {2, 1152, 576, 1728},
    };
    request.kv_insert_chunks = {{0, 0, 1728, 1728}};
    return request;
}

v3::SpmFmbResolvedProfileRequest resolved_tail_chunk_request(
    uint64_t version) {
    v3::SpmFmbResolvedProfileRequest request;
    request.version = version;
    request.chunks = {
        {0, 0, 16, 16},
        {1, 16, 16, 32},
        {2, 32, 8, 40},
    };
    return request;
}

// These checks exercise native numerical paths missing from the schema-only
// cases below. They are hardware checks, not CPU-dry support evidence.
void check_native_parity(const at::Tensor& actual, const at::Tensor& reference,
                         at::ScalarType dtype, const char* boundary) {
    TORCH_CHECK(actual.device().is_cpu() && reference.device().is_cpu() &&
                    actual.scalar_type() == dtype && reference.scalar_type() == at::kFloat &&
                    actual.sizes() == reference.sizes() && actual.dim() == 2 && actual.numel() > 0,
                boundary, ": shape/dtype/nonempty boundary");
    auto values = actual.to(at::kFloat);
    TORCH_CHECK(at::isfinite(values).all().item<bool>() &&
                    at::isfinite(reference).all().item<bool>(), boundary, ": nonfinite values");
    auto dot = (values * reference).sum(1);
    auto ref_sq = reference.square().sum(1);
    auto out_sq = values.square().sum(1);
    auto err_sq = (values - reference).square().sum(1);
    TORCH_CHECK(ref_sq.min().item<float>() > 0 && out_sq.min().item<float>() > 0,
                boundary, ": degenerate numerical oracle");
    const float cosine = (dot / (ref_sq * out_sq).sqrt()).min().item<float>();
    const float relative_l2 = (err_sq / ref_sq).sqrt().max().item<float>();
    std::printf("  NUMERIC %s min_row_cos=%.9g max_row_relative_l2=%.9g\n",
                boundary, cosine, relative_l2);
    TORCH_CHECK(cosine >= 0.999f && relative_l2 <= 0.01f,
                boundary, ": direction or magnitude mismatch");
}

void check_native_replay(RpuGraphCache& cache, int iteration,
                         uint64_t& generation, uint64_t& lifetime) {
    const auto snapshots = cache.snapshot();
    TORCH_CHECK(cache.size() == 1 && cache.cache_invariant_ok() && snapshots.size() == 1,
                "native numerical gate: unstable cache");
    const auto& stats = snapshots.front();
    if (iteration == 0) {
        generation = stats.build_generation;
        lifetime = stats.graph_lifetime_id;
    }
    TORCH_CHECK(stats.kernel_count > 0 && stats.segment_count > 0 &&
                    generation > 0 && lifetime > 0 && stats.build_generation == generation &&
                    stats.graph_lifetime_id == lifetime && stats.execution_ordinal == uint64_t(iteration + 1) &&
                    stats.replay_count == size_t(iteration) && stats.recapture_count == 0 &&
                    !graph_dma::active() && SPM_POOL.defer_depth() == 0,
                "native numerical gate: BUILD -> lookup-only REPLAY or retirement drift");
}

at::Tensor native_owned_tensor(const at::Tensor& cpu) {
    TORCH_CHECK(cpu.device().is_cpu() && cpu.is_contiguous(), "owned input requires contiguous CPU tensor");
    auto result = at::Tensor(at::detail::empty_strided_generic(
        cpu.sizes(), cpu.strides(), &rpu::RPUCachingAllocatorAdapter::get(),
        c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), cpu.scalar_type()));
    std::memcpy(result.data_ptr(), cpu.data_ptr(), cpu.nbytes());
    ::rhino_lkn::RpuDdrFlush(result.data_ptr());
    return result;
}

at::Tensor native_host_snapshot(const at::Tensor& tensor) {
    ::rhino_lkn::RpuDdrFlush(tensor.data_ptr());
    return at::from_blob(tensor.data_ptr(), tensor.sizes(),
                         at::TensorOptions().dtype(tensor.scalar_type()).device(at::kCPU)).clone();
}

// CPU ATen allocations are bank-aligned. All transfers are bank multiples;
// avoid unaligned scalar reads/writes through the mapped SPM aperture.
at::Tensor native_spm_snapshot(int core, uint32_t offset, int64_t rows,
                               int64_t cols, at::ScalarType dtype) {
    auto result = at::empty({rows, cols}, at::TensorOptions().dtype(dtype).device(at::kCPU));
    std::memcpy(result.data_ptr(), SPM_ALLOC.cpu_ptr(core, offset), result.nbytes());
    return result;
}

class NumericMlpContractModel final : public v3::FusedModelBase {
public:
    NumericMlpContractModel(v3::ActivationKind activation, bool fused, bool wall)
        : activation_(activation), fused_(fused), wall_(wall),
          hidden_(wall ? 1024 : 128), intermediate_(wall ? 2048 : 256) {
        set_model_params(8, 8, hidden_ / 8, hidden_, intermediate_);
        set_num_layers(2);
        for (int index = 0; index < 3; ++index) {
            const int64_t n = index == 2 ? hidden_ : intermediate_;
            const int64_t k = index == 2 ? intermediate_ : hidden_;
            auto cpu = at::zeros({n, k}, at::TensorOptions().dtype(at::kFloat));
            auto* ptr = cpu.data_ptr<float>();
            for (int64_t row = 0; row < n; ++row) {
                ptr[row * k + row % k] = (index == 1 ? -0.5f : 0.75f);
                ptr[row * k + (row + 17) % k] = 0.25f;
            }
            logical_[index] = cpu;
            auto physical = wall ? (cpu * 32).to(at::kChar) : cpu.to(at::kHalf);
            physical = index == 2 ? tp_row_swizzle_mc_weight(physical, 8, wall ? 1 : 2)
                                  : tp_col_swizzle_mc_weight(physical, 8, wall ? 1 : 2);
            weight_[index] = native_owned_tensor(physical);
            if (wall) scale_[index] = native_owned_tensor(at::full({n}, 1.f / 32, at::kHalf));
        }
    }

    at::Tensor reference(at::Tensor input, int layers, bool skip_down = false) const {
        input = input.to(at::kFloat).reshape({-1, hidden_});
        for (int layer = 0; layer < layers; ++layer) {
            auto gate = at::mm(input, logical_[0].t());
            gate = activation_ == v3::ActivationKind::GELU ? at::gelu(gate, "tanh") : at::silu(gate);
            gate = gate * at::mm(input, logical_[1].t());
            if (skip_down) return gate;
            input = input + at::mm(gate, logical_[2].t());
        }
        return input;
    }

    void prepare_wall() {
        v3::LayoutContext context;
        context.chunk_size = 32;  // Legal allocation; the original exact kernel consumes M20.
        context.max_kv_seq_len = 32;
        context.num_layers = 2;
        prepare_spm_pipeline_component(context, v3::compose_fmb_default_three_stage_chunk_plan(
            context, 32, 0, v3::ChunkMode::SEQUENTIAL));
    }
    void emit_wall(const at::Tensor& input, uint64_t& live_base) {
        live_base = ::rhino_lkn::RpuGetDevAddr(input.data_ptr());
        rpu_launch_ddr_broadcast_spm_dma_mutable(live_base, input, 0, 20 * hidden_, addr(0, "residual1"), 8);
        emit_mlp_pipeline(weight_[0], weight_[1], weight_[2], 20, activation_,
                          scale_[0], scale_[1], scale_[2], 0, 0, 0, 0, false, false, false, true, true);
    }
    at::Tensor wall_result(int core) const {
        return native_spm_snapshot(core, addr(0, "gate") - SPM_ALLOC.addr(0, 0),
                                   20, intermediate_ / 8, at::kHalf);
    }

protected:
    std::vector<v3::BufferDecl> declare_buffers(const v3::LayoutContext& context) override {
        const auto hbytes = context.rows() * hidden_ * 2;
        const auto ibytes = context.rows() * intermediate_ / 8 * 2;
        return {{"residual1", hbytes, 0, 10}, {"residual2", hbytes, 0, 10},
                {"down", hbytes, 0, 10}, {"gate", ibytes, 0, 10}, {"up", ibytes, 0, 10}};
    }
    v3::ModelStaticConfig static_config() override {
        v3::ModelStaticConfig config;
        config.num_layers = 2;
        config.cross_layer_batch_size = 2;
        return config;
    }
    v3::ModelDynamicConfig dynamic_config(const v3::ChunkPlan&) override {
        v3::ModelDynamicConfig config;
        config.inter_layer_io = v3::InterLayerIO::DDR_PINGPONG;
        return config;
    }
    void build_layer_subgraph(int layer, const v3::ChunkInfo& chunk) override {
        emit_layer_input_dma(layer, chunk);
        rpu_launch_spm_local_copy_dma(addr(0, "residual1"), chunk.len * hidden_, addr(0, "residual2"), 8);
        emit_mlp_pipeline(weight_[0], weight_[1], weight_[2], chunk.len, activation_,
                          {}, {}, {}, 0, 0, 0, 0, false, false, false, fused_);
        // The legacy scratch helper has no current production caller. Exercise
        // its explicit API, making the result depend on that write: destroy the
        // SPM result, reload the owned scratch, then publish. Canonical member
        // models must still reject this helper in the original cases below.
        emit_layer_chain_scratch_store(layer, chunk, "residual1");
        rpu_launch_spm_local_copy_dma(addr(0, "residual2"), chunk.len * hidden_, addr(0, "residual1"), 8);
        rpu_launch_ddr_broadcast_spm_dma(const_cast<c10::Half*>(layer_chain_scratch_ddr_ptr(layer, chunk)),
                                       chunk.len * hidden_, addr(0, "residual1"), 8);
        emit_layer_output_dma(layer, chunk);
    }

private:
    v3::ActivationKind activation_;
    bool fused_, wall_;
    int64_t hidden_, intermediate_;
    std::array<at::Tensor, 3> logical_, weight_, scale_;
};

void test_shared_fmb_numerical_graph() {
    for (int mode = 0; mode < 4; ++mode) {
        const bool wall = mode == 3;
        NumericMlpContractModel model(mode == 1 ? v3::ActivationKind::GELU : v3::ActivationKind::SILU,
                                      mode >= 2, wall);
        const int64_t rows = wall ? 20 : 32, hidden = wall ? 1024 : 128;
        std::array<at::Tensor, 2> cpu, input;
        for (int side = 0; side < 2; ++side) {
            cpu[side] = (((at::arange(rows * hidden, at::kFloat) + side * 7).remainder(29) - 14) / 16)
                            .reshape({1, rows, hidden}).to(at::kHalf);
            input[side] = native_owned_tensor(cpu[side]);
        }
        std::vector<at::Tensor> keys(2, native_owned_tensor(at::zeros({1, 32, 8, 16}, at::kHalf)));
        auto values = keys;  // MLP fixture never reads/writes KV, no KV coverage claim.
        if (wall) model.prepare_wall();
        RpuGraphCache cache(1);
        GraphSignature signature;
        signature.op_id = 9810 + mode;
        signature.shapes = {rows, hidden};
        uint64_t generation = 0, lifetime = 0, live_base = 0;
        std::array<at::Tensor, 2> captured;
        for (int iteration = 0; iteration < 4; ++iteration) {
            const int side = iteration % 2;
            if (iteration == 3) cache.clear();  // Fresh PASSTHROUGH control for B.
            auto* graph = iteration == 3 ? nullptr
                          : iteration == 0 ? &cache.get_or_create(signature) : cache.lookup(signature);
            TORCH_CHECK(iteration == 3 || graph != nullptr, "MLP lookup-only READY miss");
            if (graph) {
                graph->begin(signature);
                TORCH_CHECK(graph->state() == (iteration == 0 ? RpuKernelGraph::State::RECORDING
                                                               : RpuKernelGraph::State::REPLAYING),
                            "MLP expected one BUILD followed by REPLAY");
            }
            const bool replay = graph && graph->state() == RpuKernelGraph::State::REPLAYING;
            at::Tensor output;
            try {
                if (wall) model.emit_wall(input[side], live_base);
                else output = model.run_all_layers(input[side], keys, values, std::nullopt, 0, true);
                if (graph) {
                    graph->end();
                    cache.touch_entry(signature, replay);
                }
            } catch (...) { if (graph) graph->abort(); cache.clear(); throw; }
            const auto reference = model.reference(cpu[side], wall ? 1 : 2, wall);
            at::Tensor actual;
            if (wall) {
                std::vector<at::Tensor> shards;
                for (int core = 0; core < 8; ++core) shards.push_back(model.wall_result(core));
                actual = at::cat(shards, 1);
            } else actual = native_host_snapshot(output).reshape({rows, hidden});
            check_native_parity(actual, reference, at::kHalf, wall ? "FMB exact M20 W8 gate-up" : "FMB two-layer MLP/DMA");
            if (iteration < 2) captured[side] = actual.clone();
            if (iteration == 1) TORCH_CHECK(!at::equal(captured[0], actual), "changed live input reused old MLP output");
            if (iteration >= 2) TORCH_CHECK(at::equal(captured[side], actual),
                                           "MLP REPLAY/PASSTHROUGH is not byte-exact");
            if (graph) check_native_replay(cache, iteration, generation, lifetime);
        }
        cache.clear();
        TORCH_CHECK(cache.size() == 0 && cache.cache_invariant_ok() && !graph_dma::active(),
                    "MLP graph must retire before model, weights, mutable slots and owned DDR");
    }
    EXPECT(true, "shared FMB numerical, mutable-input and stable Graph contract");
}

void test_shared_mixed_ring_numerical_graph() {
    SPM_ALLOC.init();
    const size_t before = SPM_ALLOC.temporary_mark();
    // Five mixed-ring APIs and all precision branches of the four postnorm
    // wrappers. These are isolated original APIs, not a Wall model claim.
    for (int mode = 0; mode < 13; ++mode) {
        SpmAllocator::ScopedTemporary scope;
        const bool postnorm = mode >= 5;
        const bool local = mode == 2 || (mode >= 8 && mode <= 10) || mode == 12;
        const bool vision = mode < 11;
        const int precision = postnorm && vision ? (mode - 5) % 3 : 2;
        const bool partial_bf16 = mode == 0 || mode == 3 || (postnorm && vision && precision == 1);
        const bool residual_bf16 = !partial_bf16 && !(postnorm && vision && precision == 0);
        const auto partial_dtype = partial_bf16 ? at::kBFloat16 : at::kHalf;
        const auto residual_dtype = residual_bf16 ? at::kBFloat16 : at::kHalf;
        const auto raw_dtype = partial_bf16 || residual_bf16 ? at::kBFloat16 : at::kHalf;
        const int64_t rows = vision ? 576 : 320, cols = vision ? 1280 : 2048;
        const int64_t count = rows * cols, shard = count / 8;
        const uint32_t partial = SPM_ALLOC.alloc_temporary(count * 2);
        const uint32_t residual = SPM_ALLOC.alloc_temporary((local ? shard : count) * 2);
        const uint32_t output = mode == 3 || mode == 4 || (postnorm && vision) ? partial
                              : mode == 11 ? residual : SPM_ALLOC.alloc_temporary(count * 2);
        const uint32_t raw = local ? residual : SPM_ALLOC.alloc_temporary(shard * 2);
        const uint32_t gamma = SPM_ALLOC.alloc_temporary(cols * 2);
        auto address = [](uint32_t offset) { return SPM_ALLOC.addr(0, offset); };
        constexpr double eps = 1.e-5;
        const auto gamma_cpu = (0.75 + at::arange(cols, at::kFloat).remainder(7) / 32).to(at::kHalf);
        auto launch = [&] {
            const auto p = address(partial), r = address(residual), o = address(output);
            const auto s = address(raw), g = address(gamma);
            switch (mode) {
            case 0: rpu_launch_all_reduce_bf16_partial_fp16_residual_spm_kernel(p, r, o, rows, cols); break;
            case 1: rpu_launch_all_reduce_fp16_partial_bf16_residual_spm_kernel(p, r, o, rows, cols); break;
            case 2: rpu_launch_all_reduce_fp16_partial_bf16_residual_local_spm_kernel(p, r, o, rows, cols); break;
            case 3: rpu_launch_wall_vision_all_reduce_bf16_partial_fp16_residual_m576_inplace_kernel(p, r); break;
            case 4: rpu_launch_wall_vision_all_reduce_fp16_partial_bf16_residual_m576_inplace_kernel(p, r); break;
            case 5: rpu_launch_wall_down_postnorm_m576_fp16_partial_fp16_residual_kernel(p, r, s, g, eps); break;
            case 6: rpu_launch_wall_down_postnorm_m576_bf16_partial_fp16_residual_kernel(p, r, s, g, eps); break;
            case 7: rpu_launch_wall_down_postnorm_m576_fp16_partial_bf16_residual_kernel(p, r, s, g, eps); break;
            case 8: rpu_launch_wall_ring_postnorm_m576_fp16_partial_fp16_residual_local_inout_kernel(p, r, g, eps); break;
            case 9: rpu_launch_wall_ring_postnorm_m576_bf16_partial_fp16_residual_local_inout_kernel(p, r, g, eps); break;
            case 10: rpu_launch_wall_ring_postnorm_m576_fp16_partial_bf16_residual_local_inout_kernel(p, r, g, eps); break;
            case 11: rpu_launch_wall_ring_postnorm_fp16_partial_bf16_residual_spm_kernel(p, r, s, g, eps); break;
            case 12: rpu_launch_wall_ring_postnorm_fp16_partial_bf16_residual_local_spm_kernel(p, r, o, g, eps); break;
            }
        };
        if (mode == 0) EXPECT(throws_c10([&] {
                   rpu_launch_all_reduce_fp16_partial_bf16_residual_spm_kernel(
                       address(partial), address(residual), address(output), 16, cols);
               }), "mixed ring rejects unsupported shape before launch");
        if (postnorm) EXPECT(throws_c10([&] {
                   if (mode == 11)
                       rpu_launch_wall_ring_postnorm_fp16_partial_bf16_residual_spm_kernel(
                           address(partial), address(partial), address(raw), address(gamma), eps);
                   else if (mode == 12)
                       rpu_launch_wall_ring_postnorm_fp16_partial_bf16_residual_local_spm_kernel(
                           address(partial), address(residual), address(residual), address(gamma), eps);
                   else if (local)
                       rpu_launch_wall_ring_postnorm_m576_fp16_partial_fp16_residual_local_inout_kernel(
                           address(partial), address(residual), address(partial), eps);
                   else
                       rpu_launch_wall_down_postnorm_m576_fp16_partial_fp16_residual_kernel(
                           address(partial), address(partial), address(raw), address(gamma), eps);
               }), "postnorm rejects destructive residual/gamma alias before launch");
        RpuGraphCache cache(1);
        GraphSignature signature;
        signature.op_id = 9830 + mode;
        signature.shapes = {rows, cols};
        uint64_t generation = 0, lifetime = 0;
        std::array<std::array<at::Tensor, 8>, 2> captured;
        std::array<std::array<at::Tensor, 8>, 2> captured_raw;
        const auto poison = at::full({rows, cols}, -1, at::kShort); // 0xffff is NaN in FP16 and BF16.
        for (int iteration = 0; iteration < 4; ++iteration) {
            const int side = iteration % 2;
            const auto indices = at::arange(count, at::kFloat).reshape({rows, cols});
            const auto residual_cpu = ((indices.remainder(31) - 15) / 32 + side * 0.25).to(residual_dtype);
            auto reference = residual_cpu.to(at::kFloat);
            for (int core = 0; core < 8; ++core) {
                if (output != partial && output != residual)
                    std::memcpy(SPM_ALLOC.cpu_ptr(core, output), poison.data_ptr(), count * 2);
                if (postnorm && !local)
                    std::memcpy(SPM_ALLOC.cpu_ptr(core, raw), poison.data_ptr(), shard * 2);
                const auto partial_cpu = (((indices + 3 * core + 7 * side).remainder(23) - 11) / 64 +
                                           (core + 1) / 32.f).to(partial_dtype);
                reference = reference + partial_cpu.to(at::kFloat);
                const auto residual_part = local ? residual_cpu.reshape({-1}).slice(0, core * shard, (core + 1) * shard)
                                                 : residual_cpu;
                std::memcpy(SPM_ALLOC.cpu_ptr(core, partial), partial_cpu.data_ptr(), partial_cpu.nbytes());
                std::memcpy(SPM_ALLOC.cpu_ptr(core, residual), residual_part.data_ptr(), residual_part.nbytes());
                std::memcpy(SPM_ALLOC.cpu_ptr(core, gamma), gamma_cpu.data_ptr(), gamma_cpu.nbytes());
            }
            const auto normalized = postnorm ? reference * at::rsqrt(reference.square().mean(1, true) + eps) *
                                                      gamma_cpu.to(at::kFloat) : reference;
            if (iteration == 3) cache.clear();
            auto* graph = iteration == 3 ? nullptr
                          : iteration == 0 ? &cache.get_or_create(signature) : cache.lookup(signature);
            TORCH_CHECK(iteration == 3 || graph != nullptr, "mixed ring lookup-only READY miss");
            if (graph) {
                graph->begin(signature);
                TORCH_CHECK(graph->state() == (iteration == 0 ? RpuKernelGraph::State::RECORDING
                                                               : RpuKernelGraph::State::REPLAYING),
                            "mixed ring expected one BUILD followed by REPLAY");
            }
            const bool replay = graph && graph->state() == RpuKernelGraph::State::REPLAYING;
            try {
                launch();
                if (graph) {
                    graph->end();
                    cache.touch_entry(signature, replay);
                }
            }
            catch (...) { if (graph) graph->abort(); cache.clear(); throw; }
            for (int core = 0; core < 8; ++core) {
                const auto actual = native_spm_snapshot(core, output, rows, cols, postnorm ? at::kHalf : raw_dtype);
                check_native_parity(actual, normalized, postnorm ? at::kHalf : raw_dtype, "mixed ring replicated output");
                if (postnorm) {
                    const auto raw_actual = native_spm_snapshot(core, raw, rows / 8, cols, raw_dtype);
                    check_native_parity(raw_actual, reference.slice(0, core * rows / 8, (core + 1) * rows / 8),
                                         raw_dtype, "mixed ring compact raw carry");
                    if (iteration < 2) captured_raw[side][core] = raw_actual.clone();
                    else TORCH_CHECK(at::equal(captured_raw[side][core], raw_actual),
                                      "compact carry REPLAY/PASSTHROUGH is not byte-exact");
                }
                if (iteration < 2) captured[side][core] = actual.clone();
                if (iteration == 1) TORCH_CHECK(!at::equal(captured[0][core], actual), "changed ring input replayed stale output");
                if (iteration >= 2) TORCH_CHECK(at::equal(captured[side][core], actual),
                                               "ring REPLAY/PASSTHROUGH is not byte-exact");
            }
            if (graph) check_native_replay(cache, iteration, generation, lifetime);
        }
        cache.clear();
        TORCH_CHECK(cache.size() == 0 && cache.cache_invariant_ok() && !graph_dma::active(),
                    "mixed ring graph must retire before its SPM allocations");
    }
    TORCH_CHECK(SPM_ALLOC.temporary_mark() == before && SPM_POOL.defer_depth() == 0,
                "mixed ring restores the exact SPM watermark");
    EXPECT(true, "shared mixed ring numerical, raw carry and stable Graph contract");
}

}  // namespace

static int run_contracts() {
    const char* lkn_skip = std::getenv("LKN_SKIP");
    if (lkn_skip == nullptr || std::strcmp(lkn_skip, "1") != 0 ||
        ::geteuid() != 0) {
        std::fprintf(
            stderr,
            "[FATAL] schema-v11 managed-address Graph DMA tests require root "
            "and explicit process-start LKN_SKIP=1 acknowledgement (Release "
            "still submits hardware)\n");
        return 2;
    }
    try {
        // Standalone native entry bypasses the Python composition startup.
        initialize_rpu_tensor_runtime();
        KernelCache::instance().initialize();
        test_shared_fmb_numerical_graph();
        test_shared_mixed_ring_numerical_graph();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[FAIL] shared native numerical contract: %s\n", error.what());
        return 1;
    }
    using v3::BufferScope;
    using v3::SpmDense2DSpec;
    using v3::SpmBorrowedConsumerChunkRef;
    using v3::SpmChunkedPermutationRoute;
    using v3::SpmChunkKey;
    using v3::SpmComponentPhaseManifest;
    using v3::SpmConsumerTransform;
    using v3::SpmDirectProducedChunkRef;
    using v3::SpmDirectProducedPayloadRef;
    using v3::SpmDirectProducedPayloadRunAddRoute;
    using v3::SpmDirectProducedRowRunsRoute;
    using v3::SpmFmbConsumerRowSliceEndpoint;
    using v3::SpmIdentityRoute;
    using v3::SpmPhaseSlice;
    using v3::SpmPermutationRowRunsEvents;
    using v3::SpmPermutationRowRunsRoute;
    using v3::SpmPipelineLease;
    using v3::SpmPipelineEvent;
    using v3::SpmPipelinePlan;
    using v3::SpmPayloadPartition;
    using v3::SpmPortDistribution;
    using v3::SpmPortDType;
    using v3::SpmPortId;
    using v3::SpmRowSliceRoute;
    using v3::SpmRetainedPayloadChunkRef;
    using v3::SpmSourceChunkRef;
    using v3::SpmDstRowRun;
    using v3::SpmFmbResolvedExecutionProfile;
    using v3::SpmFmbResolvedPhaseManifest;
    using v3::SpmFmbDirectProducedPostFnBinding;
    using v3::SpmFmbDirectProducedPostFnRunAddBinding;
    using v3::SpmFmbCompositeDirectProducedBinding;
    using v3::SpmFmbCompositeRunAddBinding;
    using v3::SpmFmbCompositePhysicalReservation;
    using v3::SpmFmbCompositePhysicalPlan;
    using v3::SpmFmbPostFnYieldScope;
    using v3::SpmFmbPostFnYieldTarget;
    using v3::SpmFmbSealedActiveGroups;
    using v3::SpmFmbSealedBuildTrace;
    using v3::SpmFmbSealedDenseDdrMembers;
    using v3::SpmFmbSealedDmaEndpointBindings;
    using v3::SpmFmbSealedCallbackYields;
    using v3::SpmCompositeOccurrenceScope;
    using v3::SpmCompositeOuterFastReplayTicket;
    using v3::SpmCompositeTraceCoordinator;
    using v3::SpmFmbCompletedBuildTrace;
    using v3::SpmFmbSealedCompositeTrace;
    using v3::SpmFmbSealedPostFnYields;
    using v3::SpmFmbSealedPhaseManifest;
    using v3::SpmScratchDecl;
    using v3::SpmScratchId;
    using v3::StorageClass;

    static_assert(!std::is_copy_constructible_v<SpmPipelineLease>);
    static_assert(!std::is_copy_assignable_v<SpmPipelineLease>);
    static_assert(!std::is_move_constructible_v<SpmPipelineLease>);
    static_assert(!std::is_move_assignable_v<SpmPipelineLease>);
    static_assert(!std::is_default_constructible_v<
                  SpmFmbCompletedBuildTrace>);
    static_assert(!std::is_copy_constructible_v<
                  SpmFmbCompletedBuildTrace>);
    static_assert(!std::is_default_constructible_v<
                  SpmFmbSealedCompositeTrace>);
    static_assert(!std::is_aggregate_v<SpmFmbSealedCompositeTrace>);
    static_assert(!std::is_default_constructible_v<
                  SpmCompositeOccurrenceScope>);
    static_assert(!std::is_copy_constructible_v<
                  SpmCompositeOccurrenceScope>);
    static_assert(!std::is_default_constructible_v<
                  SpmCompositeOuterFastReplayTicket>);
    static_assert(!std::is_copy_constructible_v<
                  SpmCompositeOuterFastReplayTicket>);
    static_assert(std::is_move_constructible_v<
                  SpmCompositeOuterFastReplayTicket>);
    static_assert(!std::is_move_assignable_v<
                  SpmCompositeOuterFastReplayTicket>);
    static_assert(noexcept(
        std::declval<SpmCompositeTraceCoordinator&>()
            .commit_outer_fast_replay(
                std::declval<SpmCompositeOuterFastReplayTicket&&>())));
    static_assert(!std::is_copy_constructible_v<
                  SpmCompositeTraceCoordinator>);
    static_assert(!std::is_move_constructible_v<
                  SpmCompositeTraceCoordinator>);
    static_assert(!std::is_default_constructible_v<
                  SpmFmbCompositeDirectProducedBinding>);
    static_assert(!std::is_aggregate_v<
                  SpmFmbCompositeDirectProducedBinding>);
    static_assert(!std::is_constructible_v<
                  SpmFmbCompositeDirectProducedBinding,
                  SpmPhaseSlice>);
    static_assert(!std::is_default_constructible_v<
                  SpmFmbCompositeRunAddBinding>);
    static_assert(!std::is_aggregate_v<SpmFmbCompositeRunAddBinding>);
    static_assert(!std::is_constructible_v<
                  SpmFmbCompositeRunAddBinding,
                  SpmPhaseSlice>);
    static_assert(!std::is_convertible_v<
                  SpmFmbCompositeDirectProducedBinding,
                  SpmFmbCompositeRunAddBinding>);
    static_assert(!std::is_convertible_v<
                  SpmFmbCompositeRunAddBinding,
                  SpmFmbCompositeDirectProducedBinding>);
    static_assert(!std::is_default_constructible_v<
                  SpmFmbCompositePhysicalReservation>);
    static_assert(!std::is_aggregate_v<
                  SpmFmbCompositePhysicalReservation>);
    static_assert(!std::is_constructible_v<
                  SpmFmbCompositePhysicalReservation,
                  SpmPipelinePlan>);
    static_assert(!has_public_offset_method<
                  SpmFmbCompositePhysicalReservation>::value);
    static_assert(!has_public_address_method<
                  SpmFmbCompositePhysicalReservation>::value);
    static_assert(!has_public_resolve_method<
                  SpmFmbCompositePhysicalReservation>::value);
    static_assert(!std::is_default_constructible_v<
                  SpmFmbCompositePhysicalPlan>);
    static_assert(!std::is_aggregate_v<SpmFmbCompositePhysicalPlan>);
    static_assert(!std::is_constructible_v<
                  SpmFmbCompositePhysicalPlan,
                  SpmPipelinePlan>);
    static_assert(!has_public_offset_method<
                  SpmFmbCompositePhysicalPlan>::value);
    static_assert(!has_public_address_method<
                  SpmFmbCompositePhysicalPlan>::value);
    static_assert(!has_public_resolve_method<
                  SpmFmbCompositePhysicalPlan>::value);
    static_assert(std::is_same_v<
                  decltype(SpmFmbCompositePhysicalPlan::commit(
                      std::declval<const
                          SpmFmbCompositePhysicalReservation&>(),
                      std::declval<const SpmFmbSealedCompositeTrace&>(),
                      std::declval<const std::vector<
                          SpmFmbCompositeDirectProducedBinding>&>(),
                      std::declval<const std::vector<
                          SpmFmbCompositeRunAddBinding>&>())),
                  SpmFmbCompositePhysicalPlan>);
    using CompositeCommitFn = SpmFmbCompositePhysicalPlan (*)(
        const SpmFmbCompositePhysicalReservation&,
        const SpmFmbSealedCompositeTrace&,
        const std::vector<SpmFmbCompositeDirectProducedBinding>&,
        const std::vector<SpmFmbCompositeRunAddBinding>&);
    static_assert(!std::is_invocable_v<
                  CompositeCommitFn,
                  const SpmFmbCompositePhysicalReservation&,
                  const SpmFmbSealedCompositeTrace&,
                  const std::vector<SpmFmbCompositeRunAddBinding>&,
                  const std::vector<
                      SpmFmbCompositeDirectProducedBinding>&>);
    static_assert(std::is_same_v<
                  decltype(SpmPipelineLease::acquire_physical(
                      std::declval<const
                          SpmFmbCompositePhysicalReservation&>())),
                  std::unique_ptr<SpmPipelineLease>>);
    static_assert(std::is_same_v<
                  decltype(SpmPipelineLease::acquire_physical(
                      std::declval<const SpmFmbCompositePhysicalPlan&>())),
                  std::unique_ptr<SpmPipelineLease>>);
    EXPECT(true, "lease is statically non-copyable and non-movable");

    EXPECT(SpmAllocator::SPM_PLANNING_BUDGET == 8303708,
           "canonical budget is floor(0.99 * SPM_USABLE)");

    const std::vector<v3::BufferDecl> regions = {
        temp_region("producer", 1024, 0, 2),
        temp_region("route", 512, 1, 3),
        temp_region("consumer", 1024, 3, 4),
    };
    const SpmPipelinePlan plan = SpmPipelinePlan::compile(regions, 256);

    EXPECT(plan.effective_budget_bytes() == 8303616,
           "effective 256-byte-aligned ceiling is 8,303,616 bytes");
    EXPECT(plan.peak_bytes() == 1536,
           "plan reuses non-overlapping coarse regions");
    EXPECT(plan.reserved_top_bytes() == 256 && plan.total_bytes() == 1792,
           "plan accounts for aligned top reservation");
    EXPECT(plan.region_count() == 3 && plan.contains("route") &&
               !plan.contains("missing"),
           "plan seals named coarse regions");

    {
        auto lease = SpmPipelineLease::acquire(plan, 17);
        const auto producer = lease.view("producer");
        const auto route = lease.view("route");
        const auto consumer = lease.view("consumer");
        EXPECT(producer.resolve_relative(lease, 17) == 0,
               "producer uses first-fit offset zero");
        EXPECT(route.resolve_relative(lease, 17) == 1024,
               "overlapping route receives a disjoint offset");
        EXPECT(consumer.resolve_relative(lease, 17) == 0,
               "consumer aliases producer after its last event");

        const auto subview = lease.view("route", 128, 256);
        EXPECT(subview.resolve_relative(lease, 17) == 1152 &&
                   subview.size_bytes() == 256,
               "checked subview resolves a relative byte range");
        EXPECT(throws_c10([&] { lease.view("route", 384, 256); }),
               "out-of-range subview is rejected");
        EXPECT(throws_c10([&] { lease.view("missing"); }),
               "unknown region name is rejected");
        EXPECT(throws_c10([&] { route.resolve_relative(lease, 18); }),
               "allocator-generation drift invalidates resolution");
        EXPECT(throws_c10([&] { SpmPipelineLease::acquire(plan, 17); }),
               "only one process-global logical lease may be active");

        std::atomic<bool> wrong_thread_rejected{false};
        std::atomic<bool> cross_thread_acquire_rejected{false};
        std::thread wrong_thread([&] {
            wrong_thread_rejected.store(
                throws_c10([&] { lease.view("producer"); }));
            cross_thread_acquire_rejected.store(throws_c10([&] {
                auto nested = SpmPipelineLease::acquire(plan, 17);
            }));
        });
        wrong_thread.join();
        EXPECT(wrong_thread_rejected.load(),
               "lease operations require the owner thread");
        EXPECT(cross_thread_acquire_rejected.load(),
               "process-global lease rejects acquisition on another thread");
        lease.release();
        lease.release();
        EXPECT(!lease.active(), "release is idempotent");
    }

    // A view is bound to both plan hash and acquisition epoch.
    {
        auto first = SpmPipelineLease::acquire(plan, 21);
        const auto stale = first.view("producer");
        first.release();

        auto second = SpmPipelineLease::acquire(plan, 21);
        EXPECT(throws_c10([&] { stale.resolve_relative(second, 21); }),
               "old view is rejected after lease epoch changes");
        second.release();

        std::vector<v3::BufferDecl> changed_regions = regions;
        changed_regions[0].phase_end = 1;
        const SpmPipelinePlan changed =
            SpmPipelinePlan::compile(changed_regions, 256);
        auto changed_lease = SpmPipelineLease::acquire(changed, 21);
        EXPECT(throws_c10(
                   [&] { stale.resolve_relative(changed_lease, 21); }),
               "view is rejected against a different plan hash");
        changed_lease.release();
    }

    // The lease owns a plan copy: resolving remains valid after the source plan
    // temporary has been destroyed.
    {
        auto lease = SpmPipelineLease::acquire(
            SpmPipelinePlan::compile(regions, 256), 31);
        EXPECT(lease.view("consumer").resolve_relative(lease, 31) == 0,
               "lease does not depend on caller plan lifetime");
    }
    {
        auto lease = SpmPipelineLease::acquire(plan, 32);
        EXPECT(lease.active(), "destructor released the preceding lease");
    }

    // Physical arenas add process-wide SDK/allocator ownership on top of the
    // logical lease epoch. The owner may nest Graph scopes, but seal/release
    // require global Graph quiescence and foreign threads must fail before any
    // allocator/global mutation.
    if (!SPM_ALLOC.is_initialized()) SPM_ALLOC.init();
    const size_t physical_top = SPM_ALLOC.persistent_used() +
                                SPM_ALLOC.super_persistent_used();
    const SpmPipelinePlan physical_plan = SpmPipelinePlan::compile(
        {temp_region("physical_arena", 256, 0, 0)}, physical_top);
    {
        auto physical = SpmPipelineLease::acquire_physical(physical_plan);
        physical->seal_physical();
        EXPECT(physical->active() && physical->physical() &&
                   physical->physical_sealed(),
               "physical acquisition publishes a sealed owner lease");

        RpuKernelGraph nested;
        GraphSignature nested_signature;
        nested_signature.op_id = 701;
        nested.begin(nested_signature);
        nested.record_branch(1, GraphSignature{}, "physical owner graph");
        const uint64_t nested_generation_before = SPM_ALLOC.generation();
        const size_t nested_temporary_before = SPM_ALLOC.temporary_used();
        EXPECT(throws_c10_containing(
                   [&] { physical->release(); },
                   "operation requires no process-active Graph"),
               "physical release rejects while its owner Graph is active");
        EXPECT(physical->active() && physical->physical_sealed() &&
                   SPM_ALLOC.generation() == nested_generation_before &&
                   SPM_ALLOC.temporary_used() == nested_temporary_before,
               "graph-active physical release rejection has zero mutation");
        nested.end();
        EXPECT(nested.state() == RpuKernelGraph::State::BUILT,
               "physical owner may run a nested Graph on the same thread");

        std::atomic<bool> foreign_release_rejected{false};
        std::atomic<bool> foreign_graph_rejected{false};
        std::atomic<bool> foreign_direct_rejected{false};
        std::thread foreign_physical([&] {
            foreign_release_rejected.store(
                throws_c10_containing(
                    [&] { physical->release(); },
                    "owned by another thread"));
            foreign_graph_rejected.store(
                throws_c10_containing(
                    [&] {
                        RpuKernelGraph graph;
                        GraphSignature signature;
                        signature.op_id = 702;
                        graph.begin(signature);
                    },
                    "owned by another thread"));
            foreign_direct_rejected.store(
                throws_c10_containing(
                    [&] {
                        RpuKernelGraph::check_direct_immediate_dma_allowed(
                            "foreign physical direct canary");
                    },
                    "owned by another thread"));
        });
        foreign_physical.join();
        EXPECT(foreign_release_rejected.load() &&
                   foreign_graph_rejected.load() &&
                   foreign_direct_rejected.load(),
               "foreign release/Graph/direct paths reject under physical owner");
        EXPECT(physical->active() && physical->physical_sealed(),
               "foreign rejection leaves the physical lease unchanged");
        physical->validate_physical(
            physical->epoch(), physical_plan.hash());
        physical->release();
        const bool allocator_released =
            !throws_c10([&] { SPM_ALLOC.reset_temporary(); });
        EXPECT(!physical->active() && allocator_released,
               "owner release clears allocator and process physical claims");
    }

    {
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 703;
        graph.begin(signature);
        const uint64_t generation_before = SPM_ALLOC.generation();
        const size_t temporary_before = SPM_ALLOC.temporary_used();
        EXPECT(throws_c10_containing(
                   [&] {
                       auto forbidden =
                           SpmPipelineLease::acquire_physical(physical_plan);
                   },
                   "requires a process-quiescent"),
               "physical acquisition rejects while any Graph is active");
        EXPECT(SPM_ALLOC.generation() == generation_before &&
                   SPM_ALLOC.temporary_used() == temporary_before,
               "Graph-active rejection precedes allocator side effects");
        graph.abort();
    }

    const SpmPipelinePlan same = SpmPipelinePlan::compile(regions, 256);
    EXPECT(plan.hash() == same.hash(),
           "identical declarations produce a deterministic hash");
    EXPECT(plan.hash() == 0x6263d54ad751a479ULL,
           "FNV plan hash is stable across processes");
    std::vector<v3::BufferDecl> changed_lifetime = regions;
    changed_lifetime[1].phase_end = 2;
    EXPECT(plan.hash() !=
               SpmPipelinePlan::compile(changed_lifetime, 256).hash(),
           "semantic lifetime change updates the plan hash");

    // Typed V2: one dense FP16 replicated identity port plus a coarse scratch
    // region.  Equal numeric IDs are legal because their strong types occupy
    // independent namespaces.
    static_assert(!std::is_same_v<SpmScratchId, SpmPortId>);
    static_assert(!std::is_convertible_v<uint32_t, SpmScratchId>);
    static_assert(!std::is_convertible_v<uint32_t, SpmPortId>);
    EXPECT(true, "typed V2 scratch and port IDs are strong distinct types");

    const SpmDense2DSpec vl_spec = dense_fp16(82, 2048);
    EXPECT(vl_spec.storage_bytes() == 335872,
           "dense FP16 storage size is derived from rows and cols");
    const SpmIdentityRoute vl_route =
        identity_route(1, vl_spec, 0, 1);
    const std::vector<SpmScratchDecl> typed_scratch_regions = {
        typed_scratch(1, 4096, 0, 1),
    };
    const std::vector<SpmIdentityRoute> typed_routes = {vl_route};
    const SpmPipelinePlan typed_plan = SpmPipelinePlan::compile(
        typed_scratch_regions, typed_routes, 256);

    EXPECT(typed_plan.region_count() == 2 &&
               typed_plan.contains(SpmScratchId(1)) &&
               typed_plan.contains(SpmPortId(1)) &&
               !typed_plan.contains("1"),
           "typed plan keeps scratch, port, and legacy-name namespaces separate");
    EXPECT(typed_plan.port_spec(SpmPortId(1)) == vl_spec,
           "typed plan seals the exact dense port spec");
    EXPECT(typed_plan.peak_bytes() == 339968 &&
               typed_plan.total_bytes() == 340224,
           "overlapping typed scratch and port receive disjoint aligned ranges");

    {
        auto lease = SpmPipelineLease::acquire(typed_plan, 41);
        const auto scratch = lease.scratch(SpmScratchId(1));
        const auto port = lease.port(SpmPortId(1));
        EXPECT(scratch.resolve_relative(lease, 41) == 0 &&
                   scratch.size_bytes() == 4096,
               "typed scratch resolves without a region string");
        EXPECT(port.resolve_relative(lease, 41) == 4096 &&
                   port.size_bytes() == 335872 && port.spec() == vl_spec,
               "typed port resolves with its immutable spec");
        EXPECT(throws_c10([&] { scratch.region_name(); }),
               "typed scratch cannot be reinterpreted through a string name");
        EXPECT(throws_c10([&] { lease.view("1"); }),
               "legacy string lookup cannot reach a typed region");
        EXPECT(throws_c10([&] { lease.scratch(SpmScratchId(2)); }),
               "unknown typed scratch ID is rejected");
        EXPECT(throws_c10([&] { lease.port(SpmPortId(2)); }),
               "unknown typed port ID is rejected");
        EXPECT(throws_c10([&] { port.resolve_relative(lease, 42); }),
               "typed port detects allocator-generation drift");

        lease.release();
        auto next = SpmPipelineLease::acquire(typed_plan, 41);
        EXPECT(throws_c10([&] { port.resolve_relative(next, 41); }),
               "typed port is rejected after its lease epoch changes");
        next.release();
    }

    const SpmPipelinePlan typed_same = SpmPipelinePlan::compile(
        typed_scratch_regions, typed_routes, 256);
    EXPECT(typed_plan.hash() == typed_same.hash(),
           "identical typed declarations produce a deterministic hash");
    EXPECT(typed_plan.hash() == 0x6a10cbac97ce9d65ULL,
           "typed V2 FNV plan hash is stable across processes");
    EXPECT(typed_plan.hash() != plan.hash(),
           "typed V2 and legacy V1 plans use distinct hash schemas");
    EXPECT(typed_plan.hash() !=
               SpmPipelinePlan::compile(
                   {typed_scratch(2, 4096, 0, 1)}, typed_routes, 256)
                   .hash(),
           "scratch ID participates in the typed plan hash");
    EXPECT(typed_plan.hash() !=
               SpmPipelinePlan::compile(
                   typed_scratch_regions,
                   {identity_route(2, vl_spec, 0, 1)}, 256)
                   .hash(),
           "port ID participates in the typed plan hash");
    EXPECT(typed_plan.hash() !=
               SpmPipelinePlan::compile(
                   typed_scratch_regions,
                   {identity_route(1, vl_spec, 0, 2)}, 256)
                   .hash(),
           "route lifetime participates in the typed plan hash");
    EXPECT(typed_plan.hash() !=
               SpmPipelinePlan::compile(
                   typed_scratch_regions,
                   {identity_route(1, dense_fp16(164, 1024), 0, 1)},
                   256)
                   .hash(),
           "same-byte different-shape port changes the typed plan hash");

    EXPECT(throws_c10([&] {
               SpmIdentityRoute::bind(
                   SpmPortId(1), dense_fp16(82, 2048),
                   dense_fp16(164, 1024), 0, 1);
           }),
           "identity binding rejects same-byte different-shape endpoints");
    EXPECT(throws_c10([&] {
               SpmDense2DSpec required = vl_spec;
               required.dtype = static_cast<SpmPortDType>(2);
               SpmIdentityRoute::bind(
                   SpmPortId(1), vl_spec, required, 0, 1);
           }),
           "identity binding rejects an unsupported dtype");
    EXPECT(throws_c10([&] {
               SpmDense2DSpec required = vl_spec;
               required.distribution =
                   static_cast<SpmPortDistribution>(2);
               SpmIdentityRoute::bind(
                   SpmPortId(1), vl_spec, required, 0, 1);
           }),
           "identity binding rejects an unsupported distribution");
    EXPECT(throws_c10([&] {
               SpmIdentityRoute::bind(
                   SpmPortId(0), vl_spec, vl_spec, 0, 1);
           }),
           "identity binding rejects reserved port ID zero");
    EXPECT(throws_c10([&] {
               SpmIdentityRoute::bind(
                   SpmPortId(1), vl_spec, vl_spec, -1, 1);
           }) &&
               throws_c10([&] {
                   SpmIdentityRoute::bind(
                       SpmPortId(1), vl_spec, vl_spec, 2, 1);
               }),
           "identity binding rejects invalid lifetimes");

    EXPECT(throws_c10([&] { dense_fp16(0, 2048).storage_bytes(); }) &&
               throws_c10([&] { dense_fp16(82, 0).storage_bytes(); }) &&
               throws_c10([&] { dense_fp16(-1, 2048).storage_bytes(); }),
           "dense spec rejects non-positive dimensions");
    EXPECT(throws_c10([&] {
               dense_fp16(std::numeric_limits<int64_t>::max(),
                          std::numeric_limits<int64_t>::max())
                   .storage_bytes();
           }),
           "dense spec rejects storage-size overflow");

    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile(
                   std::vector<SpmScratchDecl>{},
                   std::vector<SpmIdentityRoute>{});
           }),
           "empty typed plan is rejected");
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile(
                   {typed_scratch(0, 256, 0, 0)}, typed_routes);
           }),
           "typed plan rejects reserved scratch ID zero");
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile(
                   {typed_scratch(1, 256, 0, 0),
                    typed_scratch(1, 256, 1, 1)},
                   typed_routes);
           }),
           "typed plan rejects duplicate scratch IDs");
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile(
                   typed_scratch_regions, {vl_route, vl_route});
           }),
           "typed plan rejects duplicate port IDs");
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile(
                   {typed_scratch(1, 0, 0, 0)}, typed_routes);
           }) &&
               throws_c10([&] {
                   SpmPipelinePlan::compile(
                       {typed_scratch(1, -1, 0, 0)}, typed_routes);
               }),
           "typed plan rejects non-positive scratch sizes");
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile(
                   {typed_scratch(1, 256, -1, 0)}, typed_routes);
           }) &&
               throws_c10([&] {
                   SpmPipelinePlan::compile(
                       {typed_scratch(1, 256, 2, 1)}, typed_routes);
               }),
           "typed plan rejects invalid scratch lifetimes");
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile(
                   std::vector<SpmScratchDecl>{},
                   {identity_route(1, dense_fp16(1, 4151809), 0, 0)});
           }),
           "typed port above the aligned canonical ceiling is rejected");

    const SpmPipelinePlan typed_alias = SpmPipelinePlan::compile(
        {typed_scratch(1, 1024, 0, 0)},
        {identity_route(1, dense_fp16(1, 512), 1, 1)});
    {
        auto lease = SpmPipelineLease::acquire(typed_alias, 43);
        EXPECT(lease.scratch(SpmScratchId(1)).resolve_relative(lease, 43) ==
                   0 &&
                   lease.port(SpmPortId(1)).resolve_relative(lease, 43) == 0 &&
                   typed_alias.peak_bytes() == 1024,
               "typed non-overlapping scratch and port reuse first-fit storage");
    }

    // Typed row-slice route: a dense producer aliases one sealed contiguous
    // row range inside a larger dense consumer storage port.
    const SpmDense2DSpec merger_storage = dense_fp16(64, 32);
    const SpmDense2DSpec merger_produced = dense_fp16(16, 32);
    const SpmRowSliceRoute merger_route = row_slice_route(
        3, merger_produced, merger_storage, 24, 16, 0, 1);
    const std::vector<SpmRowSliceRoute> row_routes = {merger_route};
    const SpmPipelinePlan row_plan = SpmPipelinePlan::compile(
        {typed_scratch(7, 4096, 0, 1)}, row_routes, 256);

    EXPECT(row_plan.port_spec(SpmPortId(3)) == merger_storage &&
               row_plan.peak_bytes() == 8192 &&
               row_plan.total_bytes() == 8448,
           "row-slice plan allocates the complete consumer storage port");
    {
        auto lease = SpmPipelineLease::acquire(row_plan, 47);
        const auto storage = lease.port(SpmPortId(3));
        const auto slice = storage.row_slice(24, 16);
        EXPECT(storage.spec() == merger_storage &&
                   storage.size_bytes() == merger_storage.storage_bytes(),
               "row-slice storage view preserves the full consumer spec");
        EXPECT(slice.spec() == merger_produced &&
                   slice.size_bytes() == merger_produced.storage_bytes() &&
                   slice.resolve_relative(lease, 47) == 4096 + 24 * 32 * 2,
               "checked row slice preserves dtype/layout and advances by dense rows");
        EXPECT(throws_c10([&] { storage.row_slice(25, 16); }),
               "row slice must match the destination range sealed in the plan");
        EXPECT(throws_c10([&] { storage.row_slice(-1, 16); }) &&
                   throws_c10([&] { storage.row_slice(24, 0); }) &&
                   throws_c10([&] {
                       storage.row_slice(std::numeric_limits<int64_t>::max(),
                                         1);
                   }),
               "row slice rejects negative, empty, out-of-range, and overflow inputs");
        EXPECT(throws_c10([&] { slice.row_slice(0, 16); }),
               "a planned row slice cannot be sliced again implicitly");
        EXPECT(throws_c10([&] {
                   lease.port(SpmPortId(3)).row_slice(24, 16)
                       .resolve_relative(lease, 48);
               }),
               "row-slice view retains allocator-generation checks");
        lease.release();

        auto next = SpmPipelineLease::acquire(row_plan, 47);
        EXPECT(throws_c10([&] { slice.resolve_relative(next, 47); }),
               "row-slice view retains lease epoch checks");
        next.release();
    }
    {
        auto lease = SpmPipelineLease::acquire(typed_plan, 49);
        EXPECT(throws_c10(
                   [&] { lease.port(SpmPortId(1)).row_slice(0, 16); }),
               "identity port cannot use an undeclared row-slice transform");
    }

    const SpmPipelinePlan row_same = SpmPipelinePlan::compile(
        {typed_scratch(7, 4096, 0, 1)}, row_routes, 256);
    EXPECT(row_plan.hash() == row_same.hash(),
           "identical row-slice declarations produce a deterministic hash");
    EXPECT(row_plan.hash() == 0x184e87d60bf76457ULL,
           "typed row-slice V3 FNV plan hash is stable across processes");
    EXPECT(row_plan.hash() != typed_plan.hash(),
           "row-slice and identity plans use distinct hash schemas");
    EXPECT(row_plan.hash() !=
               SpmPipelinePlan::compile(
                   {typed_scratch(7, 4096, 0, 1)},
                   {row_slice_route(3, merger_produced, merger_storage,
                                    25, 16, 0, 1)},
                   256)
                   .hash(),
           "destination row begin participates in the row-slice plan hash");
    EXPECT(row_plan.hash() !=
               SpmPipelinePlan::compile(
                   {typed_scratch(7, 4096, 0, 1)},
                   {row_slice_route(3, dense_fp16(8, 32), merger_storage,
                                    24, 8, 0, 1)},
                   256)
                   .hash(),
           "destination row count participates in the row-slice plan hash");

    EXPECT(throws_c10([&] {
               row_slice_route(3, dense_fp16(15, 32), merger_storage,
                               24, 16, 0, 1);
           }) &&
               throws_c10([&] {
                   row_slice_route(3, dense_fp16(16, 16), merger_storage,
                                   24, 16, 0, 1);
               }),
           "row-slice binding rejects producer row and column shape mismatch");
    EXPECT(throws_c10([&] {
               SpmDense2DSpec produced = merger_produced;
               produced.dtype = static_cast<SpmPortDType>(2);
               row_slice_route(3, produced, merger_storage, 24, 16, 0, 1);
           }),
           "row-slice binding rejects dtype mismatch");
    EXPECT(throws_c10([&] {
               SpmDense2DSpec produced = merger_produced;
               produced.distribution =
                   static_cast<SpmPortDistribution>(2);
               row_slice_route(3, produced, merger_storage, 24, 16, 0, 1);
           }),
           "row-slice binding rejects distribution mismatch");
    EXPECT(throws_c10([&] {
               row_slice_route(3, merger_produced, merger_storage,
                               -1, 16, 0, 1);
           }) &&
               throws_c10([&] {
                   row_slice_route(3, merger_produced, merger_storage,
                                   24, 0, 0, 1);
               }) &&
               throws_c10([&] {
                   row_slice_route(3, merger_produced, merger_storage,
                                   56, 16, 0, 1);
               }) &&
               throws_c10([&] {
                   row_slice_route(
                       3, merger_produced, merger_storage,
                       std::numeric_limits<int64_t>::max(),
                       std::numeric_limits<int64_t>::max(), 0, 1);
               }),
           "row-slice binding rejects invalid and overflowing destination ranges");
    EXPECT(throws_c10([&] {
               row_slice_route(0, merger_produced, merger_storage,
                               24, 16, 0, 1);
           }) &&
               throws_c10([&] {
                   row_slice_route(3, merger_produced, merger_storage,
                                   24, 16, 2, 1);
           }),
           "row-slice binding rejects reserved IDs and invalid lifetimes");

    // Typed permutation/row-runs Z1 route: a sealed source-row permutation is
    // lowered explicitly into ordered destination runs.  Only the source and
    // full destination storage occupy the plan; there is no hidden index or
    // transform scratch region.
    const SpmDense2DSpec perm_source = dense_fp16(6, 32);
    const SpmDense2DSpec perm_destination = dense_fp16(12, 32);
    const std::vector<int32_t> permutation = {2, 0, 1, 5, 3, 4};
    const std::vector<SpmDstRowRun> permutation_runs = {
        {1, 3},
        {8, 3},
    };
    const SpmPermutationRowRunsEvents permutation_events = {1, 0, 2, 3};
    const SpmPermutationRowRunsRoute permutation_route =
        permutation_row_runs_route(
            4, perm_source, 5, perm_destination, permutation,
            permutation_runs, permutation_events);
    const std::vector<SpmPermutationRowRunsRoute> permutation_routes = {
        permutation_route,
    };
    const SpmPipelinePlan permutation_plan = SpmPipelinePlan::compile(
        {typed_scratch(9, 256, 4, 4)}, permutation_routes, 256);

    EXPECT(permutation_plan.region_count() == 3 &&
               permutation_plan.contains(SpmScratchId(9)) &&
               permutation_plan.contains(SpmPortId(4)) &&
               permutation_plan.contains(SpmPortId(5)),
           "permutation plan contains scratch plus source/destination ports only");
    EXPECT(permutation_plan.peak_bytes() == 1280 &&
               permutation_plan.total_bytes() == 1536,
           "source and destination overlap at transform while later scratch aliases");
    EXPECT(permutation_plan.port_spec(SpmPortId(4)) == perm_source &&
               permutation_plan.port_spec(SpmPortId(5)) == perm_destination,
           "permutation plan seals both endpoint specs");
    EXPECT(permutation_plan.permutation_row_runs_hash(
               SpmPortId(4), SpmPortId(5)) == permutation_route.hash(),
           "plan exposes the exact sealed route hash");

    {
        auto lease = SpmPipelineLease::acquire(permutation_plan, 53);
        const auto checked = lease.permutation_row_runs(
            SpmPortId(4), SpmPortId(5));
        EXPECT(checked.source().spec() == perm_source &&
                   checked.source().resolve_relative(lease, 53) == 768 &&
                   checked.destination().spec() == perm_destination &&
                   checked.destination().resolve_relative(lease, 53) == 0,
               "checked route preserves source/destination roles and offsets");
        EXPECT(checked.permutation() == permutation &&
                   checked.runs().size() == 2 &&
                   checked.runs()[0].dst_row_begin == 1 &&
                   checked.runs()[0].row_count == 3 &&
                   checked.runs()[1].dst_row_begin == 8 &&
                   checked.runs()[1].row_count == 3 &&
                   checked.route_hash() == permutation_route.hash(),
               "checked route exports the complete lowering metadata");
        const auto lowered = checked.lower_contiguous_runs();
        EXPECT(lowered.size() == 4 &&
                   lowered[0].source_row == 2 &&
                   lowered[0].destination_row == 1 &&
                   lowered[0].row_count == 1 &&
                   lowered[1].source_row == 0 &&
                   lowered[1].destination_row == 2 &&
                   lowered[1].row_count == 2 &&
                   lowered[2].source_row == 5 &&
                   lowered[2].destination_row == 8 &&
                   lowered[2].row_count == 1 &&
                   lowered[3].source_row == 3 &&
                   lowered[3].destination_row == 9 &&
                   lowered[3].row_count == 2,
               "row-run lowering is maximal, ordered, and covers every source row");
        EXPECT(lease.scratch(SpmScratchId(9)).resolve_relative(lease, 53) == 0,
               "scratch after destination last-read reuses its offset");
        EXPECT(throws_c10([&] {
                   checked.source().row_slice(0, perm_source.rows);
               }) &&
                   throws_c10([&] {
                       checked.destination().row_slice(1, 3);
                   }),
               "permutation endpoints cannot use undeclared row-slice views");
        EXPECT(throws_c10([&] {
                   lease.permutation_row_runs(SpmPortId(5), SpmPortId(4));
               }),
               "checked route rejects a reversed or unsealed port pair");
        EXPECT(throws_c10([&] {
                   checked.source().resolve_relative(lease, 54);
               }) &&
                   throws_c10([&] {
                       checked.destination().resolve_relative(lease, 54);
                   }),
               "checked endpoints retain allocator-generation checks");

        const auto stale_source = checked.source();
        const auto stale_destination = checked.destination();
        lease.release();
        auto next = SpmPipelineLease::acquire(permutation_plan, 53);
        EXPECT(throws_c10([&] {
                   stale_source.resolve_relative(next, 53);
               }) &&
                   throws_c10([&] {
                       stale_destination.resolve_relative(next, 53);
                   }),
               "checked endpoints retain lease-epoch checks");
        next.release();
    }

    const auto identity_lower_route = permutation_row_runs_route(
        30, dense_fp16(6, 32), 31, dense_fp16(6, 32),
        std::vector<int32_t>{0, 1, 2, 3, 4, 5},
        std::vector<SpmDstRowRun>{{0, 6}},
        SpmPermutationRowRunsEvents{0, 0, 1, 2});
    const auto identity_lower_plan = SpmPipelinePlan::compile(
        std::vector<SpmScratchDecl>{},
        std::vector<SpmPermutationRowRunsRoute>{identity_lower_route});
    {
        auto lease = SpmPipelineLease::acquire(identity_lower_plan, 53);
        const auto lowered = lease.permutation_row_runs(
            SpmPortId(30), SpmPortId(31)).lower_contiguous_runs();
        EXPECT(lowered.size() == 1 &&
                   lowered[0].source_row == 0 &&
                   lowered[0].destination_row == 0 &&
                   lowered[0].row_count == 6,
               "whole-source identity lowers to one contiguous run");
        lease.release();
    }

    const auto boundary_lower_route = permutation_row_runs_route(
        32, dense_fp16(6, 32), 33, dense_fp16(7, 32),
        std::vector<int32_t>{0, 1, 2, 3, 4, 5},
        std::vector<SpmDstRowRun>{{0, 3}, {3, 3}},
        SpmPermutationRowRunsEvents{0, 0, 1, 2});
    const auto boundary_lower_plan = SpmPipelinePlan::compile(
        std::vector<SpmScratchDecl>{},
        std::vector<SpmPermutationRowRunsRoute>{boundary_lower_route});
    {
        auto lease = SpmPipelineLease::acquire(boundary_lower_plan, 53);
        const auto lowered = lease.permutation_row_runs(
            SpmPortId(32), SpmPortId(33)).lower_contiguous_runs();
        EXPECT(lowered.size() == 2 &&
                   lowered[0].source_row == 0 &&
                   lowered[0].destination_row == 0 &&
                   lowered[0].row_count == 3 &&
                   lowered[1].source_row == 3 &&
                   lowered[1].destination_row == 3 &&
                   lowered[1].row_count == 3,
               "contiguous source rows never merge across a destination-run boundary");
        lease.release();
    }

    const SpmPipelinePlan permutation_same = SpmPipelinePlan::compile(
        {typed_scratch(9, 256, 4, 4)}, permutation_routes, 256);
    EXPECT(permutation_plan.hash() == permutation_same.hash(),
           "identical permutation declarations produce a deterministic hash");
    EXPECT(permutation_route.hash() == 0xe49b60f29cef3be3ULL &&
               permutation_plan.hash() == 0xe180059c7dd2f853ULL,
           "permutation route and schema-v4 plan hashes are stable across processes");
    EXPECT(permutation_plan.hash() != typed_plan.hash() &&
               permutation_plan.hash() != row_plan.hash(),
           "permutation row-runs uses a distinct schema-v4 plan hash");

    const auto changed_source_id = permutation_row_runs_route(
        6, perm_source, 5, perm_destination, permutation,
        permutation_runs, permutation_events);
    const auto changed_destination_id = permutation_row_runs_route(
        4, perm_source, 6, perm_destination, permutation,
        permutation_runs, permutation_events);
    const auto changed_shape = permutation_row_runs_route(
        4, dense_fp16(6, 64), 5, dense_fp16(12, 64), permutation,
        permutation_runs, permutation_events);
    const auto changed_permutation = permutation_row_runs_route(
        4, perm_source, 5, perm_destination,
        std::vector<int32_t>{0, 2, 1, 5, 3, 4}, permutation_runs,
        permutation_events);
    const auto changed_runs = permutation_row_runs_route(
        4, perm_source, 5, perm_destination, permutation,
        std::vector<SpmDstRowRun>{{2, 3}, {8, 3}}, permutation_events);
    EXPECT(permutation_route.hash() != changed_source_id.hash() &&
               permutation_route.hash() != changed_destination_id.hash() &&
               permutation_route.hash() != changed_shape.hash() &&
               permutation_route.hash() != changed_permutation.hash() &&
               permutation_route.hash() != changed_runs.hash(),
           "endpoint IDs/specs, permutation values, and runs enter the route hash");
    EXPECT(permutation_plan.hash() !=
               SpmPipelinePlan::compile(
                   {typed_scratch(9, 256, 4, 4)},
                   std::vector<SpmPermutationRowRunsRoute>{
                       changed_permutation},
                   256)
                   .hash() &&
               permutation_plan.hash() !=
                   SpmPipelinePlan::compile(
                       {typed_scratch(9, 256, 4, 4)},
                       std::vector<SpmPermutationRowRunsRoute>{changed_runs},
                       256)
                       .hash(),
           "permutation and row-run changes enter the schema-v4 plan hash");

    bool every_event_changes_hash = true;
    const std::vector<SpmPermutationRowRunsEvents> changed_events = {
        {0, 0, 2, 3},
        {1, 1, 2, 3},
        {1, 0, 1, 3},
        {1, 0, 2, 4},
    };
    for (const auto& events : changed_events) {
        every_event_changes_hash = every_event_changes_hash &&
            permutation_route.hash() !=
                permutation_row_runs_route(
                    4, perm_source, 5, perm_destination, permutation,
                    permutation_runs, events)
                    .hash();
    }
    EXPECT(every_event_changes_hash,
           "every source/destination/transform event enters the route hash");

    EXPECT(throws_c10([&] {
               permutation_row_runs_route(
                   0, perm_source, 5, perm_destination, permutation,
                   permutation_runs, permutation_events);
           }) &&
               throws_c10([&] {
                   permutation_row_runs_route(
                       4, perm_source, 0, perm_destination, permutation,
                       permutation_runs, permutation_events);
               }) &&
               throws_c10([&] {
                   permutation_row_runs_route(
                       4, perm_source, 4, perm_destination, permutation,
                       permutation_runs, permutation_events);
               }),
           "permutation binding rejects reserved and identical endpoint IDs");
    EXPECT(throws_c10([&] {
               permutation_row_runs_route(
                   4, perm_source, 5, dense_fp16(12, 16), permutation,
                   permutation_runs, permutation_events);
           }) &&
               throws_c10([&] {
                   permutation_row_runs_route(
                       4, dense_fp16(6, 24), 5, dense_fp16(12, 24),
                       permutation, permutation_runs, permutation_events);
               }),
           "permutation binding rejects column mismatch and non-v16 columns");
    EXPECT(throws_c10([&] {
               SpmDense2DSpec bad = perm_source;
               bad.dtype = static_cast<SpmPortDType>(2);
               permutation_row_runs_route(
                   4, bad, 5, perm_destination, permutation,
                   permutation_runs, permutation_events);
           }) &&
               throws_c10([&] {
                   SpmDense2DSpec bad = perm_source;
                   bad.distribution =
                       static_cast<SpmPortDistribution>(2);
                   permutation_row_runs_route(
                       4, bad, 5, perm_destination, permutation,
                       permutation_runs, permutation_events);
               }),
           "permutation binding rejects unsupported dtype and distribution");

    EXPECT(throws_c10([&] {
               permutation_row_runs_route(
                   4, perm_source, 5, perm_destination,
                   std::vector<int32_t>{}, permutation_runs,
                   permutation_events);
           }) &&
               throws_c10([&] {
                   permutation_row_runs_route(
                       4, perm_source, 5, perm_destination,
                       std::vector<int32_t>{0, 1, 2}, permutation_runs,
                       permutation_events);
               }),
           "permutation binding rejects empty and wrong-length mappings");
    EXPECT(throws_c10([&] {
               permutation_row_runs_route(
                   4, perm_source, 5, perm_destination,
                   std::vector<int32_t>{-1, 0, 1, 2, 3, 4},
                   permutation_runs, permutation_events);
           }) &&
               throws_c10([&] {
                   permutation_row_runs_route(
                       4, perm_source, 5, perm_destination,
                       std::vector<int32_t>{0, 1, 2, 3, 4, 6},
                       permutation_runs, permutation_events);
               }) &&
               throws_c10([&] {
                   permutation_row_runs_route(
                       4, perm_source, 5, perm_destination,
                       std::vector<int32_t>{0, 1, 2, 3, 4, 4},
                       permutation_runs, permutation_events);
               }),
           "permutation binding rejects negative, out-of-range, and duplicate rows");

    EXPECT(throws_c10([&] {
               permutation_row_runs_route(
                   4, perm_source, 5, perm_destination, permutation, {},
                   permutation_events);
           }) &&
               throws_c10([&] {
                   permutation_row_runs_route(
                       4, perm_source, 5, perm_destination, permutation,
                       std::vector<SpmDstRowRun>{{-1, 6}},
                       permutation_events);
               }) &&
               throws_c10([&] {
                   permutation_row_runs_route(
                       4, perm_source, 5, perm_destination, permutation,
                       std::vector<SpmDstRowRun>{{1, 0}},
                       permutation_events);
               }) &&
               throws_c10([&] {
                   permutation_row_runs_route(
                       4, perm_source, 5, perm_destination, permutation,
                       std::vector<SpmDstRowRun>{{10, 6}},
                       permutation_events);
               }),
           "permutation binding rejects empty, negative, zero, and out-of-range runs");
    EXPECT(throws_c10([&] {
               permutation_row_runs_route(
                   4, perm_source, 5, perm_destination, permutation,
                   std::vector<SpmDstRowRun>{{8, 3}, {1, 3}},
                   permutation_events);
           }) &&
               throws_c10([&] {
                   permutation_row_runs_route(
                       4, perm_source, 5, perm_destination, permutation,
                       std::vector<SpmDstRowRun>{{1, 4}, {4, 2}},
                       permutation_events);
               }) &&
               throws_c10([&] {
                   permutation_row_runs_route(
                       4, perm_source, 5, perm_destination, permutation,
                       std::vector<SpmDstRowRun>{{1, 2}, {8, 3}},
                       permutation_events);
               }) &&
               throws_c10([&] {
                   permutation_row_runs_route(
                       4, perm_source, 5, perm_destination, permutation,
                       std::vector<SpmDstRowRun>{{1, 4}, {8, 4}},
                       permutation_events);
               }),
           "permutation binding rejects unordered, overlapping, short, and excess runs");

    EXPECT(throws_c10([&] {
               permutation_row_runs_route(
                   4, perm_source, 5, perm_destination, permutation,
                   permutation_runs,
                   SpmPermutationRowRunsEvents{-1, 0, 2, 3});
           }) &&
               throws_c10([&] {
                   permutation_row_runs_route(
                       4, perm_source, 5, perm_destination, permutation,
                       permutation_runs,
                       SpmPermutationRowRunsEvents{3, 0, 2, 3});
               }) &&
               throws_c10([&] {
                   permutation_row_runs_route(
                       4, perm_source, 5, perm_destination, permutation,
                       permutation_runs,
                       SpmPermutationRowRunsEvents{1, -1, 2, 3});
               }) &&
               throws_c10([&] {
                   permutation_row_runs_route(
                       4, perm_source, 5, perm_destination, permutation,
                       permutation_runs,
                       SpmPermutationRowRunsEvents{1, 3, 2, 3});
               }) &&
               throws_c10([&] {
                   permutation_row_runs_route(
                       4, perm_source, 5, perm_destination, permutation,
                       permutation_runs,
                       SpmPermutationRowRunsEvents{1, 0, 4, 3});
               }),
           "permutation binding rejects invalid source/destination/transform events");
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile(
                   std::vector<SpmScratchDecl>{},
                   std::vector<SpmPermutationRowRunsRoute>{});
           }),
           "permutation plan rejects an empty route list");
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile(
                   std::vector<SpmScratchDecl>{},
                   std::vector<SpmPermutationRowRunsRoute>{
                       permutation_route, permutation_route});
           }),
           "permutation plan rejects duplicate endpoint ports");
    EXPECT(throws_c10([&] {
               permutation_plan.permutation_row_runs_hash(
                   SpmPortId(5), SpmPortId(4));
           }),
           "permutation plan rejects an unknown route pair");

    // Mixed typed plan: retain independent dense identity ports alongside the
    // existing permutation edge without weakening plan-wide port uniqueness.
    // This is composition only; a shared source / 1->N consumer is still
    // rejected rather than inferred from repeated endpoint IDs.
    const std::vector<SpmIdentityRoute> mixed_identity_routes = {
        identity_route(6, dense_fp16(256, 1024), 1, 2),
        identity_route(7, dense_fp16(64, 2048), 2, 5),
    };
    const SpmPipelinePlan mixed_plan = SpmPipelinePlan::compile(
        {typed_scratch(10, 4096, 0, 5)}, mixed_identity_routes,
        permutation_routes, 256);
    EXPECT(mixed_plan.region_count() == 5 &&
               mixed_plan.contains(SpmScratchId(10)) &&
               mixed_plan.contains(SpmPortId(6)) &&
               mixed_plan.contains(SpmPortId(7)) &&
               mixed_plan.contains(SpmPortId(4)) &&
               mixed_plan.contains(SpmPortId(5)),
           "mixed plan contains one scratch, two retained ports, and one "
           "permutation endpoint pair");
    EXPECT(mixed_plan.peak_bytes() == 791808 &&
               mixed_plan.total_bytes() == 792064 &&
               mixed_plan.port_spec(SpmPortId(6)) ==
                   dense_fp16(256, 1024) &&
               mixed_plan.port_spec(SpmPortId(7)) ==
                   dense_fp16(64, 2048),
           "mixed plan applies one exact first-fit timeline to both route "
           "families");
    EXPECT(mixed_plan.permutation_row_runs_hash(
               SpmPortId(4), SpmPortId(5)) == permutation_route.hash(),
           "mixed plan preserves permutation route lookup and hash identity");
    {
        auto lease = SpmPipelineLease::acquire(mixed_plan, 57);
        const auto raw_tap = lease.port(SpmPortId(6));
        const auto merged = lease.port(SpmPortId(7));
        const auto checked = lease.permutation_row_runs(
            SpmPortId(4), SpmPortId(5));
        EXPECT(raw_tap.resolve_relative(lease, 57) == 4864 &&
                   merged.resolve_relative(lease, 57) == 529664 &&
                   checked.source().resolve_relative(lease, 57) == 529152 &&
                   checked.destination().resolve_relative(lease, 57) == 4096,
               "mixed lease resolves retained and permutation ports from one "
               "checked epoch");
        lease.release();
    }

    const SpmPipelinePlan mixed_same = SpmPipelinePlan::compile(
        {typed_scratch(10, 4096, 0, 5)}, mixed_identity_routes,
        permutation_routes, 256);
    EXPECT(mixed_plan.hash() == mixed_same.hash() &&
               mixed_plan.hash() == 0x5d6b6616091162beULL,
           "mixed identity/permutation FNV plan hash is stable across "
           "processes");
    EXPECT(mixed_plan.hash() != typed_plan.hash() &&
               mixed_plan.hash() != permutation_plan.hash() &&
               mixed_plan.hash() != row_plan.hash(),
           "mixed composition has a distinct schema without changing old "
           "typed hashes");
    EXPECT(mixed_plan.hash() !=
               SpmPipelinePlan::compile(
                   {typed_scratch(10, 4096, 0, 5)},
                   std::vector<SpmIdentityRoute>{
                       identity_route(6, dense_fp16(256, 1024), 1, 3),
                       mixed_identity_routes[1]},
                   permutation_routes, 256)
                   .hash() &&
               mixed_plan.hash() !=
                   SpmPipelinePlan::compile(
                       {typed_scratch(10, 4096, 0, 5)},
                       mixed_identity_routes,
                       std::vector<SpmPermutationRowRunsRoute>{
                           changed_permutation},
                       256)
                       .hash(),
           "identity lifetime and permutation mapping both enter the mixed "
           "plan hash");
    EXPECT(mixed_plan.hash() !=
               SpmPipelinePlan::compile(
                   {typed_scratch(10, 4096, 0, 5)},
                   std::vector<SpmIdentityRoute>{
                       mixed_identity_routes[1], mixed_identity_routes[0]},
                   permutation_routes, 256)
                   .hash(),
           "mixed route declaration order enters the plan hash");

    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile(
                   std::vector<SpmScratchDecl>{},
                   std::vector<SpmIdentityRoute>{}, permutation_routes);
           }) &&
               throws_c10([&] {
                   SpmPipelinePlan::compile(
                       std::vector<SpmScratchDecl>{}, mixed_identity_routes,
                       std::vector<SpmPermutationRowRunsRoute>{});
               }),
           "mixed overload requires both independent route families");
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile(
                   std::vector<SpmScratchDecl>{},
                   std::vector<SpmIdentityRoute>{
                       identity_route(4, dense_fp16(256, 1024), 1, 2)},
                   permutation_routes);
           }) &&
               throws_c10([&] {
                   const auto second_consumer = permutation_row_runs_route(
                       4, perm_source, 8, perm_destination, permutation,
                       permutation_runs, permutation_events);
                   SpmPipelinePlan::compile(
                       std::vector<SpmScratchDecl>{}, mixed_identity_routes,
                       std::vector<SpmPermutationRowRunsRoute>{
                           permutation_route, second_consumer});
               }),
           "mixed plan rejects cross-kind ID collisions and shared-source "
           "multi-consumer routing");

    // Qwen3-VL DS3 is three independent retained identity values, not one
    // shared-source 1->N route.  All three values span the coarse Vision->Text
    // boundary, so their closed [0,2] lifetimes must remain mutually disjoint
    // and disjoint from both component scratch arenas.
    std::vector<int32_t> qwen3vl_identity_permutation(64);
    std::iota(qwen3vl_identity_permutation.begin(),
              qwen3vl_identity_permutation.end(), int32_t{0});
    const auto qwen3vl_pooler_route = permutation_row_runs_route(
        1, dense_fp16(64, 2048), 2, dense_fp16(128, 2048),
        qwen3vl_identity_permutation,
        std::vector<SpmDstRowRun>{{4, 64}},
        SpmPermutationRowRunsEvents{0, 1, 1, 2});
    const SpmPipelinePlan qwen3vl_ds3_plan = SpmPipelinePlan::compile(
        std::vector<SpmScratchDecl>{
            typed_scratch(1, 1966080, 0, 0),
            typed_scratch(2, 1966080, 2, 2)},
        std::vector<SpmIdentityRoute>{
            identity_route(3, dense_fp16(64, 2048), 0, 2),
            identity_route(4, dense_fp16(64, 2048), 0, 2),
            identity_route(5, dense_fp16(64, 2048), 0, 2)},
        std::vector<SpmPermutationRowRunsRoute>{qwen3vl_pooler_route},
        /*reserved_top_bytes=*/628736);
    EXPECT(qwen3vl_ds3_plan.peak_bytes() == 4718592 &&
               qwen3vl_ds3_plan.total_bytes() == 5347328 &&
               qwen3vl_ds3_plan.effective_budget_bytes() == 8303616,
           "Qwen3-VL DS3 exact mixed plan fits below the aligned canonical "
           "ceiling");
    EXPECT(qwen3vl_ds3_plan.hash() == 0x9c693d7058865be3ULL,
           "Qwen3-VL DS3 exact mixed plan hash is stable across processes");
    {
        auto lease = SpmPipelineLease::acquire(qwen3vl_ds3_plan, 58);
        const auto route = lease.permutation_row_runs(
            SpmPortId(1), SpmPortId(2));
        EXPECT(
            lease.scratch(SpmScratchId(1)).resolve_relative(lease, 58) == 0 &&
                lease.port(SpmPortId(3)).resolve_relative(lease, 58) ==
                    1966080 &&
                lease.port(SpmPortId(4)).resolve_relative(lease, 58) ==
                    2228224 &&
                lease.port(SpmPortId(5)).resolve_relative(lease, 58) ==
                    2490368 &&
                route.source().resolve_relative(lease, 58) == 2752512 &&
                route.destination().resolve_relative(lease, 58) == 0 &&
                lease.scratch(SpmScratchId(2)).resolve_relative(lease, 58) ==
                    2752512,
            "Qwen3-VL DS3 first-fit offsets preserve all closed lifetimes");
        lease.release();
    }

    // LingBot2 exact three-camera route-only lower bound.  This deliberately
    // models only one compact [192,2560] pooler source, the full [225,2560]
    // Text destination, and three independent retained [192,2560] DeepStack
    // values.  It excludes both component scratch arenas and the live
    // persistent top, so neither this peak nor its budget headroom is evidence
    // of physical fit/admission.  Keep this fixture logical: it must never call
    // SpmPipelineLease::acquire_physical().
    std::vector<int32_t> lingbot_identity_permutation(192);
    std::iota(lingbot_identity_permutation.begin(),
              lingbot_identity_permutation.end(), int32_t{0});
    const auto lingbot_pooler_route = permutation_row_runs_route(
        60, dense_fp16(192, 2560), 61, dense_fp16(225, 2560),
        lingbot_identity_permutation,
        std::vector<SpmDstRowRun>{{1, 64}, {67, 64}, {133, 64}},
        SpmPermutationRowRunsEvents{0, 1, 1, 2});
    const SpmPipelinePlan lingbot_route_only_plan = SpmPipelinePlan::compile(
        std::vector<SpmScratchDecl>{},
        std::vector<SpmIdentityRoute>{
            identity_route(62, dense_fp16(192, 2560), 0, 2),
            identity_route(63, dense_fp16(192, 2560), 0, 2),
            identity_route(64, dense_fp16(192, 2560), 0, 2)},
        std::vector<SpmPermutationRowRunsRoute>{lingbot_pooler_route});
    constexpr size_t kLingbotRetainedRouteOnlyBytes =
        225 * 2560 * sizeof(c10::Half) +
        3 * 192 * 2560 * sizeof(c10::Half);
    EXPECT(kLingbotRetainedRouteOnlyBytes == 4101120,
           "LingBot2 retained route-only lower bound is destination plus three DeepStack ports");
    EXPECT(lingbot_route_only_plan.peak_bytes() == 5084160,
           "LingBot2 route-only peak includes the transient pooler source but no scratch or persistent bytes");
    // A full-N768 coarse Vision component is impossible even before the
    // pooler destination and persistent top are counted.  At every post-tap
    // layer's phase 5, residual1, input_norm, oproj, and fc1 are all live:
    //   3 * [768,1024] + [768,4096/8] = 5,505,024 bytes/core.
    // Once tap 17 has run, all three compact DeepStack values must survive
    // the remaining Vision layers, adding 2,949,120 bytes/core.  These byte
    // ranges overlap in time and therefore cannot alias in the coarse plan.
    constexpr size_t kLingbotN768VisionPhase5LiveBytes =
        3 * 768 * 1024 * sizeof(c10::Half) +
        768 * (4096 / 8) * sizeof(c10::Half);
    constexpr size_t kLingbotCompactDs3Bytes =
        3 * 192 * 2560 * sizeof(c10::Half);
    constexpr size_t kLingbotN768CoarseLowerBound =
        kLingbotN768VisionPhase5LiveBytes + kLingbotCompactDs3Bytes;
    EXPECT(kLingbotN768VisionPhase5LiveBytes == 5505024 &&
               kLingbotN768CoarseLowerBound == 8454144 &&
               kLingbotN768CoarseLowerBound >
                   lingbot_route_only_plan.effective_budget_bytes(),
           "LingBot2 full-N768 coarse Vision plus retained DS3 exceeds the aligned canonical ceiling before destination or persistent bytes");
    {
        // A logical lease exposes sealed lowering metadata without reserving
        // the hardware arena; physical acquisition is forbidden for this
        // intentionally incomplete lower-bound plan.
        auto lease = SpmPipelineLease::acquire(
            lingbot_route_only_plan, 59);
        const auto lowered = lease.permutation_row_runs(
            SpmPortId(60), SpmPortId(61)).lower_contiguous_runs();
        EXPECT(lowered.size() == 3 &&
                   lowered[0].source_row == 0 &&
                   lowered[0].destination_row == 1 &&
                   lowered[0].row_count == 64 &&
                   lowered[1].source_row == 64 &&
                   lowered[1].destination_row == 67 &&
                   lowered[1].row_count == 64 &&
                   lowered[2].source_row == 128 &&
                   lowered[2].destination_row == 133 &&
                   lowered[2].row_count == 64,
               "LingBot2 identity route lowers to the exact three visual prefix runs");
        lease.release();
    }

    // Wall-OSS's exact profile uses 144 merged rows per image.  The typed
    // contract has no legacy 64-row/256-byte index-pointer restriction because
    // lowering uses contiguous SPM slices and may split down to one row.
    std::vector<int32_t> wall_permutation(432);
    for (int32_t image = 0; image < 3; ++image) {
        for (int32_t row = 0; row < 144; ++row) {
            wall_permutation[image * 144 + row] =
                image * 144 + (143 - row);
        }
    }
    const auto wall_route = permutation_row_runs_route(
        20, dense_fp16(432, 16), 21, dense_fp16(480, 16),
        wall_permutation,
        std::vector<SpmDstRowRun>{{0, 144}, {160, 144}, {320, 144}},
        SpmPermutationRowRunsEvents{0, 0, 1, 2});
    const SpmPipelinePlan wall_plan = SpmPipelinePlan::compile(
        std::vector<SpmScratchDecl>{},
        std::vector<SpmPermutationRowRunsRoute>{wall_route});
    EXPECT(wall_plan.region_count() == 2 &&
               wall_plan.contains(SpmPortId(20)) &&
               wall_plan.contains(SpmPortId(21)),
           "three 144-row Wall-OSS runs compile without hidden metadata storage");

    // Schema-v5 phase overlay: production-like Wall geometry streams three
    // 144-row sources through one reused Vision backing, retains only routed
    // image rows, and borrows two C320 Text residual chunks at one reused
    // fixed placement.  No full [432,2048] source or P640 destination is added.
    static_assert(!std::is_convertible_v<uint32_t, SpmPipelineEvent>);
    static_assert(!std::is_convertible_v<uint32_t, SpmChunkKey>);
    EXPECT(true, "schema-v5 event and chunk keys are strong types");

    static_assert(!std::is_default_constructible_v<
                  SpmFmbSealedPhaseManifest>);
    static_assert(!std::is_aggregate_v<SpmFmbSealedPhaseManifest>);
    EXPECT(true,
           "FMB-sealed manifest cannot be publicly forged as an aggregate");

    SealedManifestContractModel sealed_model;
    const auto pure_layout = sealed_model.resolve_cpu();
    const auto sealed_layout = sealed_model.prepare_cpu();
    EXPECT(pure_layout.temporary_bytes == 1024 &&
               pure_layout.layout_hash == sealed_layout.layout_hash &&
               sealed_layout.temporary_bytes == 1024 &&
               sealed_layout.layout_hash != 0,
           "pure and prepared CPU planners share exact TempPerLayer plus first-fit layout");
    EXPECT(throws_c10([&] { sealed_model.seal(200); }),
           "missing model-owned occurrence schedule fails loudly");

    // The BufferDecl name pointers are model-owned and may change after
    // prepare.  Sealing must consume the allocation-time owned snapshot, not
    // fresh/shallow declarations.
    sealed_model.rename_root_after_prepare();
    sealed_model.set_schedule(valid_fmb_schedule());
    const SpmFmbSealedPhaseManifest sealed_a = sealed_model.seal(200);
    EXPECT(sealed_a.required_extent() == 1024 &&
               sealed_a.allocation_count() == 6 &&
               sealed_a.range_count() == 6,
           "sealed manifest canonicalizes Temp, TempPerLayer, synonym, "
           "subrange, and appended PostFn use");
    EXPECT(sealed_a.layout_hash() != 0 &&
               sealed_a.persistent_hash() != 0 &&
               sealed_a.allocation_hash() != 0 &&
               sealed_a.schedule_hash() != 0 &&
               sealed_a.range_hash() != 0 &&
               sealed_a.identity_hash() != 0,
           "sealed manifest binds nonzero structural and schedule hashes");

    const SpmPipelinePlan sealed_plan =
        SpmPipelinePlan::compile_fmb_sealed_phase_overlay({sealed_a});
    EXPECT(sealed_plan.phase_manifest_count() == 1 &&
               sealed_plan.peak_bytes() == 1024,
           "opaque FMB seal compiles as a placed logical overlay without routes");
    {
        auto logical = SpmPipelineLease::acquire(sealed_plan, 71);
        EXPECT(logical.plan_hash() == sealed_plan.hash(),
               "CPU-dry FMB-sealed plan admits a checked logical lease");
    }
    EXPECT(throws_c10_containing(
               [&] {
                   auto physical =
                       SpmPipelineLease::acquire_physical(sealed_plan);
               },
               "not on the physical allow-list"),
           "CPU-dry FMB seal is rejected by positive physical admission allow-list");

    {
        SealedManifestContractModel live_stale_model;
        live_stale_model.prepare_cpu();
        live_stale_model.set_schedule(valid_fmb_schedule());
        const auto live_stale_seal = live_stale_model.seal(206);
        const auto live_stale_plan =
            SpmPipelinePlan::compile_fmb_sealed_phase_overlay(
                {live_stale_seal});
        auto live_stale_lease =
            SpmPipelineLease::acquire(live_stale_plan, 77);
        live_stale_model.invalidate_for_contract();
        const bool stale_access = throws_c10_containing(
            [&] { live_stale_lease.view("missing"); }, "is stale");
        live_stale_lease.release();
        EXPECT(stale_access && !live_stale_lease.active(),
               "owner generation is revalidated during a live lease while "
               "release remains recoverable");
    }

    auto failed_schedule = valid_fmb_schedule();
    failed_schedule.use_extensions[0].name = "unknown";
    sealed_model.set_schedule(failed_schedule);
    EXPECT(throws_c10([&] { sealed_model.seal(200); }),
           "failed replacement schedule cannot mint a partial seal");
    {
        auto old_after_failed_schedule =
            SpmPipelineLease::acquire(sealed_plan, 76);
        EXPECT(old_after_failed_schedule.plan_hash() == sealed_plan.hash(),
               "failed replacement schedule does not poison old A seal");
    }

    auto schedule_b = valid_fmb_schedule(2);
    schedule_b.occurrences[0].chunk_ordinal = 3;
    sealed_model.set_schedule(schedule_b);
    const SpmFmbSealedPhaseManifest sealed_b = sealed_model.seal(200);
    EXPECT(sealed_b.schedule_hash() != sealed_a.schedule_hash() &&
               sealed_b.identity_hash() != sealed_a.identity_hash(),
           "execution-profile or schedule-version change alters the seal");
    {
        auto old_schedule_lease = SpmPipelineLease::acquire(sealed_plan, 72);
        EXPECT(old_schedule_lease.plan_hash() == sealed_plan.hash(),
               "valid replacement schedule does not poison an immutable old seal");
    }
    sealed_model.set_schedule(valid_fmb_schedule());
    const SpmFmbSealedPhaseManifest sealed_a_again = sealed_model.seal(200);
    EXPECT(sealed_a_again.schedule_hash() == sealed_a.schedule_hash() &&
               sealed_a_again.identity_hash() == sealed_a.identity_hash(),
           "A/B/A occurrence schedules canonicalize to stable hashes");

    // A new prepare commits a new allocation-owned generation.  Old sealed
    // plans must fail before taking process-global lease ownership.
    sealed_model.prepare_cpu();
    EXPECT(throws_c10_containing(
               [&] {
                   auto stale = SpmPipelineLease::acquire(sealed_plan, 73);
               },
               "is stale"),
           "new prepare invalidates prior FMB seal and compiled plan");

    {
        SealedManifestContractModel identity_model;
        const auto layout_a = identity_model.prepare_cpu();
        identity_model.set_schedule(valid_fmb_schedule());
        const auto identity_a = identity_model.seal(205);
        const auto plan_a =
            SpmPipelinePlan::compile_fmb_sealed_phase_overlay({identity_a});

        identity_model.set_decl_mode(
            SealedManifestContractModel::DeclMode::BiggerLater);
        const auto layout_b = identity_model.prepare_cpu();
        identity_model.set_schedule(valid_fmb_schedule());
        const auto identity_b = identity_model.seal(205);
        EXPECT(layout_a.temporary_bytes == 1024 &&
                   layout_b.temporary_bytes == 1280 &&
                   identity_a.allocation_hash() !=
                       identity_b.allocation_hash(),
               "A allocation identity cannot be paired with B declaration "
               "extent or offsets");
        EXPECT(throws_c10([&] {
                   auto stale_a = SpmPipelineLease::acquire(plan_a, 75);
               }),
               "B declaration prepare retires the A allocation seal");

        identity_model.set_decl_mode(
            SealedManifestContractModel::DeclMode::Normal);
        identity_model.prepare_cpu();
        identity_model.set_schedule(valid_fmb_schedule());
        const auto identity_a_again = identity_model.seal(205);
        EXPECT(identity_a_again.layout_hash() == identity_a.layout_hash() &&
                   identity_a_again.allocation_hash() ==
                       identity_a.allocation_hash(),
               "A/B/A declaration planning restores the canonical A identity");
    }

    {
        SealedManifestContractModel missing_occurrence_model;
        missing_occurrence_model.prepare_cpu();
        auto missing = valid_fmb_schedule();
        missing.occurrences.erase(missing.occurrences.begin() + 1);
        missing_occurrence_model.set_schedule(missing);
        EXPECT(throws_c10([&] { missing_occurrence_model.seal(201); }),
               "every concrete temporary allocation needs an ordinary occurrence");
    }
    {
        SealedManifestContractModel invalid_schedule_model;
        invalid_schedule_model.prepare_cpu();
        auto duplicate = valid_fmb_schedule();
        duplicate.occurrences[1].key = duplicate.occurrences[0].key;
        EXPECT(throws_c10([&] {
                   invalid_schedule_model.set_schedule(duplicate);
               }),
               "duplicate occurrence key is rejected before commit");
        auto overlapping = valid_fmb_schedule();
        overlapping.occurrences[1].global_origin = SpmPipelineEvent(1);
        EXPECT(throws_c10([&] {
                   invalid_schedule_model.set_schedule(overlapping);
               }),
               "closed global occurrence windows must be disjoint");
    }
    {
        SealedManifestContractModel extension_model;
        extension_model.prepare_cpu();
        auto unknown = valid_fmb_schedule();
        unknown.use_extensions[0].name = "unknown";
        extension_model.set_schedule(unknown);
        EXPECT(throws_c10([&] { extension_model.seal(202); }),
               "named use extension rejects unknown allocation");

        auto duplicate = valid_fmb_schedule();
        duplicate.use_extensions.push_back(duplicate.use_extensions[0]);
        extension_model.set_schedule(duplicate);
        EXPECT(throws_c10([&] { extension_model.seal(202); }),
               "named use extension key is unique");

        auto outside = valid_fmb_schedule();
        outside.use_extensions[0].local_phase_end = 1;
        extension_model.set_schedule(outside);
        EXPECT(throws_c10([&] { extension_model.seal(202); }),
               "appended use window must stay inside its occurrence");
    }
    {
        SealedManifestContractModel alias_collision(
            SealedManifestContractModel::DeclMode::OverlappingSubrangeAlias);
        alias_collision.prepare_cpu();
        alias_collision.set_schedule(valid_fmb_schedule());
        EXPECT(throws_c10_containing(
                   [&] { alias_collision.seal(203); },
                   "distinct logical allocations"),
               "nonzero alias overlapping its root in byte x event fails closed");
    }
    EXPECT(throws_c10([&] {
               SealedManifestContractModel model(
                   SealedManifestContractModel::DeclMode::OversizedAlias);
               model.prepare_cpu();
           }) &&
               throws_c10([&] {
                   SealedManifestContractModel model(
                       SealedManifestContractModel::DeclMode::CrossScopeAlias);
                   model.prepare_cpu();
               }),
           "alias span and unsupported cross-scope alias are rejected");
    EXPECT(throws_c10([&] {
               SealedManifestContractModel model(
                   SealedManifestContractModel::DeclMode::TemporaryCallback);
               model.prepare_cpu();
           }) &&
               throws_c10([&] {
                   SealedManifestContractModel model(
                       SealedManifestContractModel::DeclMode::TemporaryReverse);
                   model.prepare_cpu();
           }),
           "temporary callback and illegal reverse allocation metadata are rejected");
    EXPECT(throws_c10_containing(
               [&] {
                   SealedManifestContractModel model(
                       SealedManifestContractModel::DeclMode::InvalidScope);
                   model.prepare_cpu();
               },
               "invalid buffer scope") &&
               throws_c10_containing(
                   [&] {
                       SealedManifestContractModel model(
                           SealedManifestContractModel::DeclMode::InvalidStorage);
                       model.prepare_cpu();
                   },
                   "invalid storage class"),
           "owned manifest rejects invalid scope and storage enums");

    std::optional<SpmPipelinePlan> expired_owner_plan;
    {
        auto owner = std::make_unique<SealedManifestContractModel>();
        owner->prepare_cpu();
        owner->set_schedule(valid_fmb_schedule());
        const auto seal = owner->seal(204);
        expired_owner_plan.emplace(
            SpmPipelinePlan::compile_fmb_sealed_phase_overlay({seal}));
    }
    EXPECT(throws_c10_containing(
               [&] {
                   auto stale_owner =
                       SpmPipelineLease::acquire(*expired_owner_plan, 74);
               },
               "owner expired"),
           "compiled FMB seal fails after its owning model is destroyed");

    // Schema-v7 replaces model-authored occurrence lists with an opaque
    // profile materialized from FusedModelBase's canonical traversal helper.
    static_assert(!std::is_default_constructible_v<
                  SpmFmbResolvedExecutionProfile>);
    static_assert(!std::is_aggregate_v<SpmFmbResolvedExecutionProfile>);
    static_assert(!std::is_default_constructible_v<
                  SpmFmbResolvedPhaseManifest>);
    static_assert(!std::is_aggregate_v<SpmFmbResolvedPhaseManifest>);
    static_assert(!std::is_constructible_v<
                  SpmFmbResolvedPhaseManifest,
                  SpmFmbSealedPhaseManifest>);
    static_assert(!std::is_constructible_v<
                  SpmFmbSealedPhaseManifest,
                  SpmFmbResolvedPhaseManifest>);
    static_assert(!std::is_convertible_v<
                  SpmFmbResolvedPhaseManifest,
                  SpmFmbSealedPhaseManifest>);
    static_assert(!std::is_convertible_v<
                  SpmFmbSealedPhaseManifest,
                  SpmFmbResolvedPhaseManifest>);
    static_assert(!std::is_default_constructible_v<
                  SpmFmbConsumerRowSliceEndpoint>);
    static_assert(!std::is_aggregate_v<
                  SpmFmbConsumerRowSliceEndpoint>);
    static_assert(!std::is_constructible_v<
                  SpmFmbConsumerRowSliceEndpoint, SpmPhaseSlice>);
    static_assert(!std::is_convertible_v<
                  SpmFmbConsumerRowSliceEndpoint, SpmPhaseSlice>);
    EXPECT(true,
           "schema-v7 profile/manifest and consumer endpoint are opaque and "
           "have no public range conversion");

    const auto request_a = resolved_request();
    ResolvedTraversalContractModel resolved_model;
    const auto resolved_layout = resolved_model.prepare_cpu();
    const SpmFmbResolvedExecutionProfile profile_a =
        resolved_model.resolve(request_a);
    const SpmFmbResolvedPhaseManifest resolved_a =
        resolved_model.seal(profile_a, 210);
    EXPECT(resolved_layout.temporary_bytes == 512 &&
               profile_a.step_count() == 4 &&
               profile_a.build_body_iterations() == 1 &&
               profile_a.profile_hash() != 0,
           "schema-v7 resolves ordinary layer-outer SEQUENTIAL traversal");
    EXPECT(resolved_a.required_extent() == 512 &&
               resolved_a.allocation_count() == 2 &&
               resolved_a.range_count() == 8 &&
               resolved_a.profile_hash() == profile_a.profile_hash() &&
               resolved_a.identity_hash() != 0 && resolved_a.cpu_dry(),
           "SEQUENTIAL seal maps LayerWide and Compute scope-mask uses");

    {
        ResolvedTraversalContractModel owned_profile;
        owned_profile.prepare_cpu();

        RpuKernelGraph graph;
        graph.begin();
        std::atomic<bool> foreign_graph_rejected{false};
        std::thread foreign_graph([&] {
            foreign_graph_rejected.store(throws_c10(
                [&] { (void)owned_profile.resolve(request_a); }));
        });
        foreign_graph.join();
        graph.abort();

        auto physical = RpuExecutionCoordinator::enter_physical(
            "resolved profile foreign physical contract");
        std::atomic<bool> foreign_physical_rejected{false};
        std::thread foreign_physical([&] {
            foreign_physical_rejected.store(throws_c10(
                [&] { (void)owned_profile.resolve(request_a); }));
        });
        foreign_physical.join();
        RpuExecutionCoordinator::exit_physical(
            physical, "resolved profile foreign physical cleanup");

        const auto after_rejection = owned_profile.resolve(request_a);
        EXPECT(foreign_graph_rejected.load() &&
                   foreign_physical_rejected.load() &&
                   after_rejection.profile_hash() != 0,
               "profile resolver rejects foreign owners without corrupting its snapshot");
        owned_profile.cancel_cpu_prepare();
    }

    const SpmPipelinePlan resolved_plan =
        SpmPipelinePlan::compile_fmb_resolved_phase_overlay({resolved_a});
    EXPECT(resolved_plan.phase_manifest_count() == 1 &&
               resolved_plan.peak_bytes() == 512,
           "schema-v7 opaque seal compiles as a CPU-only logical overlay");
    {
        auto logical = SpmPipelineLease::acquire(resolved_plan, 81);
        EXPECT(logical.plan_hash() == resolved_plan.hash(),
               "schema-v7 CPU-dry overlay admits a logical lease");
    }
    EXPECT(throws_c10_containing(
               [&] {
                   auto physical =
                       SpmPipelineLease::acquire_physical(resolved_plan);
               },
               "phase-overlay admission 4"),
           "schema-v7 admission 4 remains outside the physical allow-list");

    // A live resolved manifest closes the allocation/profile/range identity
    // gap without granting physical authority on its own.  The frozen CPU
    // compiler must reject it until an opaque consumer endpoint and a sealed
    // producer-yield Graph window are both present.
    {
        ResolvedTraversalContractModel live_model;
        const auto live_layout = live_model.prepare_live();
        const auto live_profile = live_model.resolve_live(request_a);
        const auto live_manifest =
            live_model.seal_live(live_profile, 218);
        const v3::SpmFmbConsumerOccurrenceSelector consumer_occurrence{
            v3::SpmFmbOccurrenceKind::LayerBody,
            /*body_id=*/0, /*group_id=*/0, /*layer=*/0,
            /*chunk_index=*/0};
        const auto consumer_endpoint = live_model.seal_consumer_endpoint(
            live_profile, live_manifest, "resolved_layerwide",
            /*allocation_layer=*/-1,
            SpmChunkKey(SpmPortId(90), 0), dense_fp16(2, 64),
            /*row_begin=*/1, /*row_count=*/1, consumer_occurrence);
        const auto consumer_endpoint_again =
            live_model.seal_consumer_endpoint(
                live_profile, live_manifest, "resolved_layerwide",
                /*allocation_layer=*/-1,
                SpmChunkKey(SpmPortId(90), 0), dense_fp16(2, 64),
                /*row_begin=*/1, /*row_count=*/1,
                consumer_occurrence);
        EXPECT(live_layout.temporary_bytes == 512 &&
                   !live_profile.cpu_dry() &&
                   !live_manifest.cpu_dry() &&
                   live_manifest.required_extent() == 512 &&
                   live_manifest.profile_hash() ==
                       live_profile.profile_hash() &&
                   live_manifest.identity_hash() != 0 &&
                   consumer_endpoint.identity_hash() != 0 &&
                   consumer_endpoint.identity_hash() ==
                       consumer_endpoint_again.identity_hash() &&
                   consumer_endpoint.storage_spec() == dense_fp16(2, 64) &&
                   consumer_endpoint.row_begin() == 1 &&
                   consumer_endpoint.row_count() == 1,
               "live resolved manifest seals real FMB allocation identity without physical authority");
        const auto changed_consumer_endpoint =
            live_model.seal_consumer_endpoint(
                live_profile, live_manifest, "resolved_layerwide",
                /*allocation_layer=*/-1,
                SpmChunkKey(SpmPortId(90), 0), dense_fp16(2, 64),
                /*row_begin=*/0, /*row_count=*/1,
                consumer_occurrence);
        EXPECT(changed_consumer_endpoint.identity_hash() !=
                   consumer_endpoint.identity_hash(),
               "consumer endpoint hash seals the exact owned row slice");
        EXPECT(
            throws_c10_containing(
                [&] {
                    auto bad_occurrence = consumer_occurrence;
                    bad_occurrence.chunk_index = 99;
                    live_model.seal_consumer_endpoint(
                        live_profile, live_manifest,
                        "resolved_layerwide", -1,
                        SpmChunkKey(SpmPortId(90), 0),
                        dense_fp16(2, 64), 1, 1, bad_occurrence);
                },
                "matched 0 resolved steps") &&
                throws_c10_containing(
                    [&] {
                        live_model.seal_consumer_endpoint(
                            live_profile, live_manifest, "missing", -1,
                            SpmChunkKey(SpmPortId(90), 0),
                            dense_fp16(2, 64), 1, 1,
                            consumer_occurrence);
                    },
                    "no owned allocation") &&
                throws_c10_containing(
                    [&] {
                        live_model.seal_consumer_endpoint(
                            live_profile, live_manifest,
                            "resolved_layerwide", -1,
                            SpmChunkKey(SpmPortId(90), 0),
                            dense_fp16(2, 64), 2, 1,
                            consumer_occurrence);
                    },
                    "out of bounds"),
            "consumer endpoint rejects unsealed occurrence, allocation, and row geometry");
        EXPECT(throws_c10_containing(
                   [&] {
                       SpmPipelinePlan::compile_fmb_resolved_phase_overlay(
                           {live_manifest});
                   },
                   "rejects a live manifest"),
               "CPU resolved compiler rejects a live token before endpoint and producer-yield sealing");
        EXPECT(throws_c10_containing(
                   [&] { live_model.seal(live_profile, 218); },
                   "CPU-contract allocation snapshot"),
               "legacy schema-v7 seal remains CPU-dry only");
        EXPECT(throws_c10_containing(
                   [&] {
                       resolved_model.seal_consumer_endpoint(
                           profile_a, resolved_a, "resolved_layerwide", -1,
                           SpmChunkKey(SpmPortId(90), 0),
                           dense_fp16(2, 64), 1, 1,
                           consumer_occurrence);
                   },
                   "requires live allocation"),
               "consumer endpoint cannot be minted from CPU-dry authority");
    }

    std::optional<SpmFmbCompositePhysicalReservation>
        stale_composite_reservation;
    // Graph-independent two-phase reservation for the exact LingBot2
    // sequential N256x3 geometry.  Direct endpoints are image-major final
    // yields; retained endpoints are tap-major/image-minor for layers 0/1/2.
    {
        constexpr size_t kConsumerRootBytes = 1'152'000;
        constexpr size_t kConsumerTemporaryBytes = 4'163'072;
        constexpr size_t kProducerTemporaryBytes = 1'966'080;
        constexpr size_t kPersistentTopBytes = 774'144;
        constexpr size_t kExpectedDynamicBytes = 7'112'192;
        constexpr size_t kExpectedTotalBytes = 7'886'336;
        constexpr size_t kExpectedHeadroomBytes = 417'280;
        constexpr size_t kProducerCount = 3;
        constexpr size_t kYieldsPerProducer = 4;
        constexpr uint64_t kPolicy =
            UINT64_C(0x6c696e67626f7432);

        CompositeReservationGeometryContractModel consumer(
            kConsumerRootBytes, kConsumerTemporaryBytes,
            /*num_layers=*/3, /*hidden_size=*/2560);
        CompositeReservationGeometryContractModel producer(
            kProducerTemporaryBytes, kProducerTemporaryBytes,
            /*num_layers=*/1, /*hidden_size=*/1024,
            /*stable_producer=*/true);
        // Producer-first is intentional: consumer's first live prepare advances
        // the process persistent generation after the producer snapshot seals.
        const auto producer_layout = producer.prepare_live();
        const auto consumer_layout = consumer.prepare_live();
        const auto consumer_profile = consumer.resolve_live();
        const auto producer_profile = producer.resolve_live();
        const auto consumer_manifest = consumer.seal_live(
            consumer_profile, /*arena=*/401, /*base=*/0);
        const auto producer_manifest = producer.seal_live(
            producer_profile, /*arena=*/402,
            /*base=*/kConsumerRootBytes);
        const SpmDense2DSpec consumer_storage = dense_fp16(225, 2560);
        const SpmDense2DSpec produced = dense_fp16(64, 2560);
        const std::array<int64_t, kProducerCount> row_begins =
            {1, 67, 133};

        std::vector<SpmFmbConsumerRowSliceEndpoint> direct;
        direct.reserve(kProducerCount);
        for (size_t image = 0; image < kProducerCount; ++image) {
            direct.push_back(consumer.seal_endpoint(
                consumer_profile, consumer_manifest, consumer_storage,
                row_begins[image], /*row_count=*/64, /*layer=*/0));
        }
        std::vector<SpmFmbConsumerRowSliceEndpoint> run_add;
        run_add.reserve((kYieldsPerProducer - 1) * kProducerCount);
        for (int tap = 0; tap < 3; ++tap) {
            for (size_t image = 0; image < kProducerCount; ++image) {
                run_add.push_back(consumer.seal_endpoint(
                    consumer_profile, consumer_manifest, consumer_storage,
                    row_begins[image], /*row_count=*/64, /*layer=*/tap));
            }
        }

        const auto reservation =
            SpmFmbCompositePhysicalReservation::
                compile_repeated_dense_producer(
                    producer_manifest, consumer_manifest, direct, run_add,
                    produced, kYieldsPerProducer, kPolicy,
                    kPersistentTopBytes);
        const auto reservation_again =
            SpmFmbCompositePhysicalReservation::
                compile_repeated_dense_producer(
                    producer_manifest, consumer_manifest, direct, run_add,
                    produced, kYieldsPerProducer, kPolicy,
                    kPersistentTopBytes);
        EXPECT(
            consumer_layout.temporary_bytes == kConsumerTemporaryBytes &&
                producer_layout.temporary_bytes ==
                    kProducerTemporaryBytes &&
                reservation.dynamic_bytes() == kExpectedDynamicBytes &&
                reservation.reserved_top_bytes() == kPersistentTopBytes &&
                reservation.total_bytes() == kExpectedTotalBytes &&
                reservation.effective_budget_bytes() == 8'303'616 &&
                reservation.headroom_bytes() == kExpectedHeadroomBytes &&
                reservation.producer_occurrence_count() == kProducerCount &&
                reservation.yields_per_producer() == kYieldsPerProducer &&
                reservation.producer_yield_count() == 12 &&
                reservation.direct_binding_count() == 3 &&
                reservation.run_add_binding_count() == 9 &&
                reservation.retained_payload_count() == 9 &&
                reservation.hash() != 0 &&
                reservation.hash() == reservation_again.hash(),
            "LingBot2 pre-BUILD reservation locks T+DS9 exact fit and a "
            "Graph-independent stable placement hash");

        auto missing_run_add = run_add;
        missing_run_add.pop_back();
        auto duplicate_direct = direct;
        duplicate_direct[1] = direct[0];
        auto wrong_tap_order = run_add;
        std::rotate(wrong_tap_order.begin(),
                    wrong_tap_order.begin() + kProducerCount,
                    wrong_tap_order.end());
        auto mixed_callback = run_add;
        mixed_callback[1] = run_add[kProducerCount + 1];
        SpmDense2DSpec invalid_dtype = produced;
        invalid_dtype.dtype = static_cast<SpmPortDType>(99);
        EXPECT(
            throws_c10([&] {
                SpmFmbCompositePhysicalReservation::
                    compile_repeated_dense_producer(
                        producer_manifest, consumer_manifest, direct,
                        missing_run_add, produced, kYieldsPerProducer,
                        kPolicy, kPersistentTopBytes);
            }) &&
                throws_c10([&] {
                    SpmFmbCompositePhysicalReservation::
                        compile_repeated_dense_producer(
                            producer_manifest, consumer_manifest,
                            duplicate_direct, run_add, produced,
                            kYieldsPerProducer, kPolicy,
                            kPersistentTopBytes);
                }) &&
                throws_c10([&] {
                    SpmFmbCompositePhysicalReservation::
                        compile_repeated_dense_producer(
                            producer_manifest, consumer_manifest, direct,
                            wrong_tap_order, produced, kYieldsPerProducer,
                            kPolicy, kPersistentTopBytes);
                }) &&
                throws_c10([&] {
                    SpmFmbCompositePhysicalReservation::
                        compile_repeated_dense_producer(
                            producer_manifest, consumer_manifest, direct,
                            mixed_callback, produced, kYieldsPerProducer,
                            kPolicy, kPersistentTopBytes);
                }) &&
                throws_c10([&] {
                    SpmFmbCompositePhysicalReservation::
                        compile_repeated_dense_producer(
                            producer_manifest, consumer_manifest, direct,
                            run_add, invalid_dtype, kYieldsPerProducer,
                            kPolicy, kPersistentTopBytes);
                }),
            "composite reservation rejects missing cells, duplicate rows, "
            "tap/callback-order drift, and non-FP16 producer geometry");

        const size_t live_persistent_top =
            SPM_ALLOC.persistent_used() +
            SPM_ALLOC.super_persistent_used();
        const auto live_reservation =
            SpmFmbCompositePhysicalReservation::
                compile_repeated_dense_producer(
                    producer_manifest, consumer_manifest, direct, run_add,
                    produced, kYieldsPerProducer, kPolicy,
                    live_persistent_top);
        auto physical =
            SpmPipelineLease::acquire_physical(live_reservation);
        SpmCompositeTraceCoordinator coordinator(
            /*expected_occurrence_count=*/4,
            /*expected_producer_yield_count=*/12, kPolicy);
        coordinator.adopt_physical_component(
            producer, *physical, /*producer=*/true);
        coordinator.adopt_physical_component(
            consumer, *physical, /*producer=*/false);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 7998;
        signature.segment_key = producer_profile.profile_hash();
        graph.begin(signature);
        coordinator.begin_build(
            graph, *physical, /*require_committed=*/false);
        bool first_live_occurrence_admitted = false;
        try {
            auto occurrence = coordinator.begin_build_occurrence(
                producer, producer_profile);
            first_live_occurrence_admitted = true;
        } catch (...) {
        }
        coordinator.cancel();
        graph.abort();
        coordinator.release_physical_components(*physical);
        physical->release();
        EXPECT(
            first_live_occurrence_admitted,
            "live composite adoption rebases a sibling-advanced persistent "
            "generation before stable-three occurrence admission");
        stale_composite_reservation.emplace(reservation);
    }
    EXPECT(throws_c10_containing(
               [&] {
                   auto stale = SpmPipelineLease::acquire_physical(
                       *stale_composite_reservation);
               },
               "owner"),
           "pre-BUILD reservation fails closed after either component owner "
           "expires");

    {
        ResolvedTraversalContractModel short_prefill_model;
        const auto shape = short_prefill_model.resolve_causal_prefill(225);
        EXPECT(shape.resolved_chunk_size == 240 &&
                   shape.chunks.size() == 1 &&
                   shape.chunks[0].idx == 0 &&
                   shape.chunks[0].offset == 0 &&
                   shape.chunks[0].len == 225 &&
                   shape.chunks[0].kv_seq_len == 225 &&
                   shape.allocation_layout.chunk_size == 225 &&
                   shape.allocation_layout.max_kv_seq_len == 225,
               "causal dry shape distinguishes S225 nominal C240 from allocation C225");
    }

    {
        ResolvedTraversalContractModel dry_cancel_model;
        const auto first_layout = dry_cancel_model.prepare_cpu();
        const auto first_profile = dry_cancel_model.resolve(request_a);
        const auto first_manifest =
            dry_cancel_model.seal(first_profile, 219);
        const auto first_plan =
            SpmPipelinePlan::compile_fmb_resolved_phase_overlay(
                {first_manifest});
        dry_cancel_model.cancel_cpu_prepare();
        EXPECT(throws_c10_containing(
                   [&] {
                       auto stale = SpmPipelineLease::acquire(first_plan, 89);
                   },
                   "stale after relayout, prepare, or model-state change"),
               "CPU-dry cancel retires every previously sealed token");

        const auto second_layout = dry_cancel_model.prepare_cpu();
        const auto second_profile = dry_cancel_model.resolve(request_a);
        const auto second_manifest =
            dry_cancel_model.seal(second_profile, 219);
        const auto second_plan =
            SpmPipelinePlan::compile_fmb_resolved_phase_overlay(
                {second_manifest});
        EXPECT(first_layout.temporary_bytes ==
                       second_layout.temporary_bytes &&
                   first_layout.layout_hash == second_layout.layout_hash &&
                   first_profile.profile_hash() ==
                       second_profile.profile_hash() &&
                   first_manifest.identity_hash() ==
                       second_manifest.identity_hash() &&
                   first_plan.hash() == second_plan.hash(),
               "CPU-dry prepare/cancel/prepare is A/A deterministic");
        dry_cancel_model.cancel_cpu_prepare();
        EXPECT(throws_c10_containing(
                   [&] { dry_cancel_model.cancel_cpu_prepare(); },
                   "requires one prepared CPU-dry manifest"),
               "CPU-dry cancel rejects a double retirement");
    }

    resolved_model.set_traversal(
        ResolvedTraversalContractBase::Traversal::ChunkOuter);
    const SpmFmbResolvedExecutionProfile profile_b =
        resolved_model.resolve(request_a);
    const SpmFmbResolvedPhaseManifest resolved_b =
        resolved_model.seal(profile_b, 210);
    EXPECT(profile_b.step_count() == 4 &&
               profile_b.profile_hash() != profile_a.profile_hash() &&
               resolved_b.identity_hash() != resolved_a.identity_hash(),
           "chunk-outer traversal changes the resolved topology identity");
    resolved_model.set_traversal(
        ResolvedTraversalContractBase::Traversal::Sequential);
    const SpmFmbResolvedExecutionProfile profile_a_again =
        resolved_model.resolve(request_a);
    const SpmFmbResolvedPhaseManifest resolved_a_again =
        resolved_model.seal(profile_a_again, 210);
    EXPECT(profile_a_again.profile_hash() == profile_a.profile_hash() &&
               resolved_a_again.range_hash() == resolved_a.range_hash() &&
               resolved_a_again.identity_hash() ==
                   resolved_a.identity_hash(),
           "schema-v7 A/B/A topology resolution is deterministic");

    {
        ResolvedTraversalContractModel repeated_model;
        repeated_model.set_body_iterations(3);
        repeated_model.prepare_cpu();
        const auto repeated_profile = repeated_model.resolve(request_a);
        const auto repeated_seal =
            repeated_model.seal(repeated_profile, 211);
        EXPECT(repeated_profile.build_body_iterations() == 3 &&
                   repeated_profile.step_count() == 12 &&
                   repeated_profile.profile_hash() !=
                       profile_a.profile_hash() &&
                   repeated_seal.range_count() == 24,
               "body_iterations repeat the canonical BUILD traversal exactly");
    }

    {
        ResolvedTraversalContractModel kv_model(
            ResolvedTraversalContractBase::Traversal::KvFirst);
        kv_model.prepare_cpu(16, 32, 8);
        const auto kv_profile = kv_model.resolve(
            resolved_request(1, true));
        const auto kv_seal = kv_model.seal(kv_profile, 212);
        EXPECT(kv_profile.step_count() == 12 &&
                   kv_profile.profile_hash() != profile_a.profile_hash() &&
                   kv_seal.allocation_count() == 3 &&
                   kv_seal.range_count() == 24,
               "KV_FIRST resolves four insert plus two compute chunks per "
               "layer and maps LayerWide/KvInsert/Compute scope masks");

        ResolvedTraversalContractModel wall_model(
            ResolvedTraversalContractBase::Traversal::KvFirst);
        wall_model.prepare_cpu(576, 1728, 1728);
        const auto wall_request = resolved_wall_exact_mimc_request();
        const auto wall_profile = wall_model.resolve(wall_request);
        const auto wall_profile_again = wall_model.resolve(wall_request);
        EXPECT(wall_profile.step_count() == 8 &&
                   wall_profile.profile_hash() ==
                       wall_profile_again.profile_hash() &&
                   wall_profile.profile_hash() != kv_profile.profile_hash(),
               "Wall exact MIMC resolves one packed KV_FIRST plus three "
               "compute callbacks per layer with a stable profile hash");
    }

    {
        ResolvedTraversalContractBase unsealed_model;
        unsealed_model.prepare_cpu();
        EXPECT(throws_c10_containing(
                   [&] { unsealed_model.resolve(request_a); },
                   "CanonicalTraversal"),
               "default traversal capability is Unsealed and fails closed");
    }

    const auto opaque_callback_rejected =
        [&](ResolvedTraversalContractBase::OpaqueCallback callback,
            const char* reason) {
            ResolvedTraversalContractModel model;
            model.set_opaque_callback(callback);
            model.prepare_cpu();
            return throws_c10_containing(
                [&] { model.resolve(request_a); }, reason);
        };
    EXPECT(opaque_callback_rejected(
               ResolvedTraversalContractBase::OpaqueCallback::Preload,
               "preload_fn"),
           "schema-v7 rejects an opaque preload_fn");
    EXPECT(opaque_callback_rejected(
               ResolvedTraversalContractBase::OpaqueCallback::PreLayers,
               "pre/post layer callbacks") &&
               opaque_callback_rejected(
                   ResolvedTraversalContractBase::OpaqueCallback::PostLayers,
                   "pre/post layer callbacks"),
           "schema-v7 rejects opaque pre_layers_fn and post_layers_fn");
    EXPECT(opaque_callback_rejected(
               ResolvedTraversalContractBase::OpaqueCallback::PostFn,
               "post_fn scope"),
           "schema-v7 rejects an opaque post_fn");
    {
        ResolvedTraversalContractModel preload_decl_model;
        preload_decl_model.set_persistent_preload(true);
        preload_decl_model.prepare_cpu();
        EXPECT(throws_c10_containing(
                   [&] { preload_decl_model.resolve(request_a); },
                   "BufferDecl preload callback"),
               "schema-v7 rejects an opaque BufferDecl preload callback");
    }
    {
        ResolvedTraversalPersistentPreloadContractModel preload_model;
        preload_model.set_persistent_preload(true);
        preload_model.prepare_cpu();
        const auto preload_profile = preload_model.resolve(request_a);
        const auto preload_manifest =
            preload_model.seal(preload_profile, 217);
        EXPECT(preload_profile.profile_hash() != profile_a.profile_hash() &&
                   preload_manifest.profile_hash() ==
                       preload_profile.profile_hash() &&
                   preload_manifest.range_count() == resolved_a.range_count(),
               "persistent preload capability seals callbacks outside the resolved temporary window");

        ResolvedTraversalPersistentPreloadContractModel temp_preload_model;
        temp_preload_model.set_temporary_preload(true);
        EXPECT(throws_c10_containing(
                   [&] { temp_preload_model.prepare_cpu(); },
                   "temporary buffer 'resolved_compute' cannot declare a "
                   "preload callback"),
               "owned allocation snapshot rejects temporary preload before capability resolution");
    }

    {
        ResolvedTraversalContractModel outside_model;
        outside_model.set_outside_temp(true);
        outside_model.prepare_cpu();
        EXPECT(throws_c10_containing(
                   [&] { outside_model.resolve(request_a); },
                   "OutsideLayerLoop temporary"),
               "schema-v7 rejects an unowned OutsideLayerLoop temporary");
    }
    {
        ResolvedTraversalContractModel mismatch_model;
        mismatch_model.prepare_cpu();
        auto wrong_chunk_size = resolved_request();
        wrong_chunk_size.chunks = {
            {0, 0, 8, 8},
            {1, 8, 8, 16},
            {2, 16, 8, 24},
            {3, 24, 8, 32},
        };
        EXPECT(throws_c10_containing(
                   [&] { mismatch_model.resolve(wrong_chunk_size); },
                   "allocation-time chunk_size"),
               "schema-v7 rejects request/allocation chunk-size drift");

        auto malformed_chunks = resolved_request();
        malformed_chunks.chunks[1].offset = 15;
        EXPECT(throws_c10_containing(
                   [&] { mismatch_model.resolve(malformed_chunks); },
                   "contiguous idx/offset"),
               "schema-v7 rejects a malformed resolved chunk vector");

        auto overflowing_chunks = resolved_request();
        overflowing_chunks.chunks[0].kv_seq_len =
            std::numeric_limits<int64_t>::max();
        overflowing_chunks.chunks[1].kv_seq_len =
            std::numeric_limits<int64_t>::max();
        EXPECT(throws_c10_containing(
                   [&] { mismatch_model.resolve(overflowing_chunks); },
                   "calculation overflows"),
               "schema-v7 rejects ChunkInfo arithmetic overflow");

        const auto old_profile = mismatch_model.resolve(request_a);
        mismatch_model.prepare_cpu(32, 32);
        EXPECT(throws_c10_containing(
                   [&] { mismatch_model.seal(old_profile, 213); },
                   "execution profile owner is stale"),
               "layout re-prepare invalidates an old resolved profile");
    }

    // Schema-v8 observes the exact BUILD callback windows and FMB facade
    // allocation accesses.  It is an opaque CPU-dry token only: there is no
    // plan compiler or physical admission for this type.
    static_assert(!std::is_default_constructible_v<
                  SpmFmbSealedBuildTrace>);
    static_assert(!std::is_aggregate_v<SpmFmbSealedBuildTrace>);
    static_assert(!std::is_constructible_v<
                  SpmFmbResolvedPhaseManifest,
                  SpmFmbSealedBuildTrace>);
    static_assert(!std::is_constructible_v<
                  SpmFmbSealedBuildTrace,
                  SpmFmbResolvedPhaseManifest>);
    EXPECT(true,
           "schema-v8 BUILD trace token is opaque and has no schema-v7 "
           "manifest conversion");
    static_assert(!std::is_default_constructible_v<
                  SpmFmbSealedActiveGroups>);
    static_assert(!std::is_aggregate_v<SpmFmbSealedActiveGroups>);
    static_assert(!std::is_constructible_v<
                  SpmFmbSealedActiveGroups,
                  SpmFmbSealedBuildTrace>);
    static_assert(!std::is_constructible_v<
                  SpmFmbResolvedPhaseManifest,
                  SpmFmbSealedActiveGroups>);
    EXPECT(true,
           "schema-v9 active groups are opaque and have no trace-only or "
           "manifest conversion");
    static_assert(!std::is_default_constructible_v<
                  SpmFmbSealedDenseDdrMembers>);
    static_assert(!std::is_aggregate_v<SpmFmbSealedDenseDdrMembers>);
    static_assert(!std::is_constructible_v<
                  SpmFmbSealedDenseDdrMembers,
                  SpmFmbSealedActiveGroups>);
    static_assert(!std::is_constructible_v<
                  SpmFmbResolvedPhaseManifest,
                  SpmFmbSealedDenseDdrMembers>);
    EXPECT(true,
           "schema-v10 dense-DDR members are opaque and have no group-only "
           "or manifest conversion");
    static_assert(!std::is_default_constructible_v<
                  GraphDmaSemanticEndpoint>);
    static_assert(!std::is_aggregate_v<GraphDmaSemanticEndpoint>);
    static_assert(!std::is_constructible_v<
                  GraphDmaSemanticEndpoint, uint64_t>);
    static_assert(!std::is_convertible_v<
                  GraphDmaSemanticEndpoint, uint64_t>);
    static_assert(!std::is_default_constructible_v<
                  SpmFmbSealedDmaEndpointBindings>);
    static_assert(!std::is_aggregate_v<
                  SpmFmbSealedDmaEndpointBindings>);
    static_assert(!std::is_constructible_v<
                  SpmFmbSealedDmaEndpointBindings,
                  SpmFmbSealedDenseDdrMembers>);
    static_assert(!std::is_constructible_v<
                  SpmFmbResolvedPhaseManifest,
                  SpmFmbSealedDmaEndpointBindings>);
    EXPECT(true,
           "schema-v11 DMA endpoints and sealed bindings are opaque, "
           "FMB-minted, and have no numeric or physical conversion");

    static_assert(!std::is_default_constructible_v<
                  SpmFmbPostFnYieldTarget>);
    static_assert(!std::is_aggregate_v<SpmFmbPostFnYieldTarget>);
    static_assert(!std::is_copy_constructible_v<
                  SpmFmbPostFnYieldTarget>);
    static_assert(!std::is_move_constructible_v<
                  SpmFmbPostFnYieldTarget>);
    static_assert(!std::is_constructible_v<
                  SpmFmbPostFnYieldTarget, uint32_t>);
    static_assert(!std::is_convertible_v<
                  SpmFmbPostFnYieldTarget, uint32_t>);
    static_assert(!std::is_default_constructible_v<
                  SpmFmbPostFnYieldScope>);
    static_assert(!std::is_copy_constructible_v<
                  SpmFmbPostFnYieldScope>);
    static_assert(!std::is_default_constructible_v<
                  SpmFmbSealedPostFnYields>);
    static_assert(!std::is_aggregate_v<SpmFmbSealedPostFnYields>);
    static_assert(!std::is_constructible_v<
                  SpmFmbSealedPostFnYields, SpmFmbSealedBuildTrace>);
    static_assert(!std::is_default_constructible_v<
                  SpmFmbSealedCallbackYields>);
    static_assert(!std::is_aggregate_v<SpmFmbSealedCallbackYields>);
    static_assert(!std::is_constructible_v<
                  SpmFmbSealedCallbackYields,
                  SpmFmbSealedBuildTrace>);
    static_assert(!std::is_default_constructible_v<
                  SpmFmbDirectProducedPostFnBinding>);
    static_assert(!std::is_aggregate_v<
                  SpmFmbDirectProducedPostFnBinding>);
    static_assert(!std::is_default_constructible_v<
                  SpmFmbDirectProducedPostFnRunAddBinding>);
    static_assert(!std::is_aggregate_v<
                  SpmFmbDirectProducedPostFnRunAddBinding>);
    EXPECT(true,
           "typed PostFn yield target/scope/seal/direct/RunAdd bindings "
           "remain opaque and cannot expose or accept a raw SPM address");

    {
        ResolvedTraversalUngroupedContractModel model;
        model.set_opaque_callback(
            ResolvedTraversalContractBase::OpaqueCallback::PostFnYield);
        model.prepare_cpu();
        EXPECT(throws_c10_containing(
                   [&] { (void)model.resolve(request_a); },
                   "post_fn scope"),
               "typed PostFn yield callback fails closed without the model "
               "capability opt-in");
    }

    {
        ResolvedTraversalPostFnYieldContractModel model;
        model.set_opaque_callback(
            ResolvedTraversalContractBase::OpaqueCallback::PostFnYield);
        model.prepare_cpu();
        const auto profile = model.resolve(request_a);
        // Unlike older marker-only fixtures, this proof records one
        // real existing terminal kernel. Release submits it; initialization
        // supplies the Program_t needed to clone the Graph node under test.
        KernelCache::instance().initialize();
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 8001;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const SpmFmbSealedBuildTrace trace =
            model.seal_trace(profile, graph);
        const SpmFmbSealedPostFnYields yields =
            model.seal_post_fn_yields(trace, graph);
        model.validate_post_fn_yields(yields, graph);
        const uint64_t build_generation = graph.build_generation();
        graph.begin(signature);
        model.arm_post_fn_yield_replay(yields, graph);
        model.drive_post_fn_yield_replay(yields);
        graph.end();
        model.validate_post_fn_yields(yields, graph);
        EXPECT(profile.step_count() == 5 &&
                   trace.callback_count() == 5 &&
                   yields.cpu_dry() && yields.yield_count() == 1 &&
                   yields.yield_hash() != 0 &&
                   yields.source_trace_hash() == trace.trace_hash() &&
                   graph.build_generation() == build_generation,
               "opted-in typed PostFn all-reduce terminal writer BUILDs, "
               "seals, and validates one stable opaque producer yield "
               "through ordinary REPLAY");
    }

    {
        ResolvedTraversalCallbackYieldContractModel model;
        model.set_opaque_callback(
            ResolvedTraversalContractBase::OpaqueCallback::
                CallbackYieldPostFn);
        model.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::
                LayerProducerYield);
        model.prepare_cpu();
        const auto profile = model.resolve(request_a);
        KernelCache::instance().initialize();
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 8002;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const SpmFmbSealedBuildTrace trace =
            model.seal_trace(profile, graph);
        const SpmFmbSealedCallbackYields yields =
            model.seal_callback_yields(trace, graph);
        model.validate_callback_yields(yields, graph);
        const uint64_t build_generation = graph.build_generation();
        graph.begin(signature);
        model.arm_callback_yield_replay(yields, graph);
        model.drive_callback_yield_replay(yields);
        graph.end();
        model.validate_callback_yields(yields, graph);
        EXPECT(profile.step_count() == 5 &&
                   trace.callback_count() == 5 &&
                   yields.cpu_dry() && yields.yield_count() == 3 &&
                   yields.yield_hash() != 0 &&
                   yields.source_trace_hash() == trace.trace_hash() &&
                   graph.build_generation() == build_generation,
               "callback producer authority seals two ordered LayerBody "
               "terminal writers plus one PostFn writer and consumes the "
               "same graph-wide marker universe through ordinary REPLAY");
    }

    {
        ResolvedTraversalCallbackYieldContractModel model(
            ResolvedTraversalContractBase::Traversal::KvFirst);
        model.set_opaque_callback(
            ResolvedTraversalContractBase::OpaqueCallback::
                CallbackYieldPostFn);
        model.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::
                LayerProducerYield);
        model.prepare_cpu(16, 32, 8);
        const auto profile = model.resolve(
            resolved_request(/*version=*/1, /*include_kv_insert=*/true));
        KernelCache::instance().initialize();
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 80021;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const SpmFmbSealedBuildTrace trace =
            model.seal_trace(profile, graph);
        const SpmFmbSealedCallbackYields yields =
            model.seal_callback_yields(trace, graph);
        model.validate_callback_yields(yields, graph);
        const uint64_t build_generation = graph.build_generation();
        graph.begin(signature);
        model.arm_callback_yield_replay(yields, graph);
        model.drive_callback_yield_replay(yields);
        graph.end();
        model.validate_callback_yields(yields, graph);
        EXPECT(profile.step_count() == 13 &&
                   trace.callback_count() == 13 &&
                   yields.cpu_dry() && yields.yield_count() == 3 &&
                   yields.yield_hash() != 0 &&
                   yields.source_trace_hash() == trace.trace_hash() &&
                   graph.build_generation() == build_generation,
               "callback producer capability accepts eight zero-yield "
               "KV_FIRST callbacks while sealing only the two LayerBody "
               "and one PostFn terminal writers through stable REPLAY");
    }

    {
        ResolvedTraversalCallbackYieldContractModel model(
            ResolvedTraversalContractBase::Traversal::KvFirst);
        model.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::KvProducerYield);
        model.prepare_cpu(16, 32, 8);
        const auto profile = model.resolve(
            resolved_request(/*version=*/1, /*include_kv_insert=*/true));
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 80022;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        const bool rejected = throws_c10_containing(
            [&] { model.drive_trace_cpu(profile); },
            "requires an armed resolved PostFn or LayerBody");
        graph.abort();
        EXPECT(rejected &&
                   graph.state() == RpuKernelGraph::State::PASSTHROUGH &&
                   !RpuKernelGraph::has_active(),
               "callback producer capability still rejects an actual "
               "yield from a KV_FIRST callback before Graph commit");
    }

    {
        constexpr size_t kOccurrences = 3;
        constexpr size_t kProducerYields = 9;
        constexpr uint64_t kPolicy =
            UINT64_C(0x636f6d706f736974);

        ResolvedTraversalCallbackYieldContractModel model;
        model.set_opaque_callback(
            ResolvedTraversalContractBase::OpaqueCallback::
                CallbackYieldPostFn);
        model.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::
                LayerProducerYield);
        model.prepare_cpu();
        const auto profile = model.resolve(request_a);
        KernelCache::instance().initialize();

        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 8003;
        signature.segment_key =
            profile.profile_hash() ^ UINT64_C(1);
        if (signature.segment_key == 0) {
            signature.segment_key = UINT64_C(2);
        }

        SpmCompositeTraceCoordinator build(
            kOccurrences, kProducerYields, kPolicy);
        graph.begin(signature);
        build.begin_build(graph);
        for (size_t ordinal = 0; ordinal < kOccurrences; ++ordinal) {
            auto occurrence =
                build.begin_build_occurrence(model, profile);
            model.drive_trace_cpu(profile);
            occurrence.finish();
        }
        const bool build_count_ok =
            build.completed_occurrence_count() == kOccurrences;
        graph.end();

        const size_t built_graph_size = graph.graph_size();
        const uint64_t build_generation = graph.build_generation();
        const SpmFmbSealedCompositeTrace sealed =
            build.seal_build(graph);

        SpmCompositeTraceCoordinator replay(
            kOccurrences, kProducerYields, kPolicy);
        graph.begin(signature);
        replay.begin_replay(sealed, graph);
        for (size_t ordinal = 0; ordinal < kOccurrences; ++ordinal) {
            auto occurrence =
                replay.begin_replay_occurrence(model);
            model.drive_composite_callback_yield_replay();
            occurrence.finish();
        }
        replay.finish_replay();
        graph.end();

        EXPECT(
            signature.segment_key != profile.profile_hash() &&
                build_count_ok && sealed.composite_hash() != 0 &&
                sealed.occurrence_count() == kOccurrences &&
                sealed.producer_yield_count() == kProducerYields &&
                replay.completed_occurrence_count() == kOccurrences &&
                graph.state() == RpuKernelGraph::State::BUILT &&
                graph.replayable() &&
                graph.graph_size() == built_graph_size &&
                graph.build_generation() == build_generation,
            "three ordered callback-yield occurrences BUILD once and "
            "consume the exact retained outer Graph through stable REPLAY");
    }

    {
        constexpr size_t kOccurrences = 3;
        constexpr size_t kProducerYields = 9;
        constexpr uint64_t kPolicy =
            UINT64_C(0x33736c6f74746564);
        constexpr uint64_t kLegacySentinel =
            UINT64_C(0x1e6ac90000000001);

        ResolvedTraversalCompositeSlotsContractModel model;
        model.set_opaque_callback(
            ResolvedTraversalContractBase::OpaqueCallback::
                CallbackYieldPostFn);
        model.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::
                LayerProducerYield);
        model.set_persistent_preload(true);
        model.prepare_cpu();
        model.set_composite_slots_enabled(false);
        const auto ordinary_profile = model.resolve(request_a);
        model.set_composite_slots_enabled(true);
        const auto profile = model.resolve(request_a);
        const auto legacy_before =
            model.exchange_composite_input_slot(kLegacySentinel);

        KernelCache::instance().initialize();
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 8005;
        signature.segment_key =
            profile.profile_hash() ^ UINT64_C(0x35);
        if (signature.segment_key == 0) {
            signature.segment_key = UINT64_C(0x35);
        }

        std::array<uintptr_t, kOccurrences> build_slot_addresses{};
        std::array<uint64_t, kOccurrences> build_previous_values{};
        std::array<bool, kOccurrences> build_follower_roles{};
        SpmCompositeTraceCoordinator build(
            kOccurrences, kProducerYields, kPolicy);
        graph.begin(signature);
        build.begin_build(graph);
        for (size_t ordinal = 0; ordinal < kOccurrences; ++ordinal) {
            auto occurrence =
                build.begin_build_occurrence(model, profile);
            const auto slot = model.exchange_composite_input_slot(
                UINT64_C(0xb000000000000000) + ordinal);
            build_slot_addresses[ordinal] = slot.first;
            build_previous_values[ordinal] = slot.second;
            build_follower_roles[ordinal] =
                model.consume_composite_preload_role();
            model.drive_trace_cpu(profile);
            occurrence.finish();
        }
        graph.end();
        const size_t built_graph_size = graph.graph_size();
        const uint64_t build_generation = graph.build_generation();
        const SpmFmbSealedCompositeTrace sealed =
            build.seal_build(graph);

        std::array<uintptr_t, kOccurrences> replay_slot_addresses{};
        std::array<uint64_t, kOccurrences> replay_previous_values{};
        std::array<bool, kOccurrences> replay_follower_roles{};
        SpmCompositeTraceCoordinator replay(
            kOccurrences, kProducerYields, kPolicy);
        graph.begin(signature);
        replay.begin_replay(sealed, graph);
        for (size_t ordinal = 0; ordinal < kOccurrences; ++ordinal) {
            auto occurrence =
                replay.begin_replay_occurrence(model);
            const auto slot = model.exchange_composite_input_slot(
                UINT64_C(0xc000000000000000) + ordinal);
            replay_slot_addresses[ordinal] = slot.first;
            replay_previous_values[ordinal] = slot.second;
            replay_follower_roles[ordinal] =
                model.consume_composite_preload_role();
            model.drive_composite_callback_yield_replay();
            occurrence.finish();
        }
        replay.finish_replay();
        graph.end();
        const auto legacy_after =
            model.exchange_composite_input_slot(kLegacySentinel);

        bool build_slots_started_zero = true;
        bool replay_restored_each_slot = true;
        bool roles_exact = true;
        for (size_t ordinal = 0; ordinal < kOccurrences; ++ordinal) {
            build_slots_started_zero =
                build_slots_started_zero &&
                build_previous_values[ordinal] == 0;
            replay_restored_each_slot =
                replay_restored_each_slot &&
                replay_slot_addresses[ordinal] ==
                    build_slot_addresses[ordinal] &&
                replay_previous_values[ordinal] ==
                    UINT64_C(0xb000000000000000) + ordinal;
            roles_exact =
                roles_exact &&
                build_follower_roles[ordinal] == (ordinal != 0) &&
                replay_follower_roles[ordinal] == (ordinal != 0);
        }
        const bool build_slots_distinct =
            build_slot_addresses[0] != build_slot_addresses[1] &&
            build_slot_addresses[0] != build_slot_addresses[2] &&
            build_slot_addresses[1] != build_slot_addresses[2];

        EXPECT(
            ordinary_profile.profile_hash() == profile.profile_hash() &&
                legacy_before.first == legacy_after.first &&
                legacy_after.second == kLegacySentinel &&
                build_slots_started_zero && build_slots_distinct &&
                replay_restored_each_slot && roles_exact &&
                sealed.occurrence_count() == kOccurrences &&
                sealed.producer_yield_count() == kProducerYields &&
                graph.state() == RpuKernelGraph::State::BUILT &&
                graph.replayable() &&
                graph.graph_size() == built_graph_size &&
                graph.build_generation() == build_generation,
            "explicit three-slot same-owner preload policy preserves the "
            "ordinary profile hash and restores distinct mutable inputs "
            "through BUILD plus stable REPLAY");
    }

    {
        constexpr size_t kOccurrences = 4;
        constexpr size_t kProducerYields = 12;
        constexpr uint64_t kPolicy =
            UINT64_C(0x346f757465726661);

        ResolvedTraversalCompositeSlotsContractModel producer;
        producer.set_opaque_callback(
            ResolvedTraversalContractBase::OpaqueCallback::
                DoubleCallbackYieldPostFn);
        producer.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::
                LayerProducerYield);
        producer.set_persistent_preload(true);
        producer.prepare_cpu();
        const auto producer_profile = producer.resolve(request_a);

        ResolvedTraversalContractModel consumer;
        consumer.prepare_cpu();
        const auto consumer_profile = consumer.resolve(request_a);

        KernelCache::instance().initialize();
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 80051;
        signature.segment_key =
            producer_profile.profile_hash() ^ UINT64_C(0x4f51);
        if (signature.segment_key == 0) {
            signature.segment_key = UINT64_C(0x4f51);
        }

        SpmCompositeTraceCoordinator build(
            kOccurrences, kProducerYields, kPolicy);
        graph.begin(signature);
        build.begin_build(graph);
        for (size_t ordinal = 0; ordinal < 3; ++ordinal) {
            auto occurrence =
                build.begin_build_occurrence(producer, producer_profile);
            (void)producer.exchange_composite_input_slot(
                UINT64_C(0xe000000000000000) + ordinal);
            (void)producer.consume_composite_preload_role();
            producer.drive_trace_cpu(producer_profile);
            occurrence.finish();
        }
        {
            auto occurrence =
                build.begin_build_occurrence(consumer, consumer_profile);
            consumer.drive_trace_cpu(consumer_profile);
            occurrence.finish();
        }
        graph.end();
        const SpmFmbSealedCompositeTrace sealed =
            build.seal_build(graph);

        SpmCompositeTraceCoordinator replay(
            kOccurrences, kProducerYields, kPolicy);
        graph.begin(signature);
        replay.begin_replay(sealed, graph);
        const bool uncommitted_physical_rejected = throws_c10_containing(
            [&] { (void)replay.validate_outer_fast_replay(); },
            "one committed physical lease");
        const bool unchanged_before_cancel =
            replay.completed_occurrence_count() == 0 &&
            graph.op_stream_stamp().position == 0;
        replay.cancel();
        graph.abort();

        EXPECT(
            sealed.occurrence_count() == kOccurrences &&
                sealed.producer_yield_count() == kProducerYields &&
                uncommitted_physical_rejected &&
                unchanged_before_cancel &&
                graph.state() == RpuKernelGraph::State::PASSTHROUGH &&
                !RpuKernelGraph::has_active(),
            "outer-fast composite ticket rejects the exact sealed "
            "4-occurrence/3-producer/12-yield cursor-zero replay until "
            "both owners share one committed physical lease");

        RpuKernelGraph reordered_graph;
        GraphSignature reordered_signature = signature;
        reordered_signature.op_id = 80052;
        reordered_signature.segment_key ^= UINT64_C(0x100);
        if (reordered_signature.segment_key == 0) {
            reordered_signature.segment_key = UINT64_C(0x4f52);
        }
        SpmCompositeTraceCoordinator reordered_build(
            kOccurrences, kProducerYields, kPolicy);
        reordered_graph.begin(reordered_signature);
        reordered_build.begin_build(reordered_graph);
        for (size_t ordinal = 0; ordinal < 2; ++ordinal) {
            auto occurrence = reordered_build.begin_build_occurrence(
                producer, producer_profile);
            (void)producer.exchange_composite_input_slot(
                UINT64_C(0xe100000000000000) + ordinal);
            (void)producer.consume_composite_preload_role();
            producer.drive_trace_cpu(producer_profile);
            occurrence.finish();
        }
        {
            auto occurrence = reordered_build.begin_build_occurrence(
                consumer, consumer_profile);
            consumer.drive_trace_cpu(consumer_profile);
            occurrence.finish();
        }
        {
            auto occurrence = reordered_build.begin_build_occurrence(
                producer, producer_profile);
            (void)producer.exchange_composite_input_slot(
                UINT64_C(0xe100000000000002));
            (void)producer.consume_composite_preload_role();
            producer.drive_trace_cpu(producer_profile);
            occurrence.finish();
        }
        reordered_graph.end();
        const SpmFmbSealedCompositeTrace reordered_sealed =
            reordered_build.seal_build(reordered_graph);

        SpmCompositeTraceCoordinator reordered_replay(
            kOccurrences, kProducerYields, kPolicy);
        reordered_graph.begin(reordered_signature);
        reordered_replay.begin_replay(
            reordered_sealed, reordered_graph);
        const bool owner_order_rejected = throws_c10_containing(
            [&] {
                (void)reordered_replay.validate_outer_fast_replay();
            },
            "3-producer/1-consumer/12-yield owner order");
        const bool reordered_cursor_unchanged =
            reordered_replay.completed_occurrence_count() == 0 &&
            reordered_graph.op_stream_stamp().position == 0;
        reordered_replay.cancel();
        reordered_graph.abort();

        EXPECT(
            reordered_sealed.occurrence_count() == kOccurrences &&
                reordered_sealed.producer_yield_count() ==
                    kProducerYields &&
                owner_order_rejected && reordered_cursor_unchanged &&
                reordered_graph.state() ==
                    RpuKernelGraph::State::PASSTHROUGH &&
                !RpuKernelGraph::has_active(),
            "outer-fast composite ticket rejects a sealed P/P/C/P owner "
            "order before physical or census publication");
    }

    {
        constexpr uint64_t kPolicy =
            UINT64_C(0x33736c6f7473746f);
        constexpr uint64_t kLegacySentinel =
            UINT64_C(0x1e6ac90000000002);

        ResolvedTraversalCompositeSlotsContractModel model;
        model.set_opaque_callback(
            ResolvedTraversalContractBase::OpaqueCallback::
                CallbackYieldPostFn);
        model.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::
                LayerProducerYield);
        model.set_persistent_preload(true);
        model.prepare_cpu();
        const auto profile = model.resolve(request_a);
        const auto legacy_before =
            model.exchange_composite_input_slot(kLegacySentinel);

        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 8006;
        signature.segment_key =
            profile.profile_hash() ^ UINT64_C(0x36);
        if (signature.segment_key == 0) {
            signature.segment_key = UINT64_C(0x36);
        }
        SpmCompositeTraceCoordinator coordinator(
            /*expected_occurrence_count=*/2,
            /*expected_producer_yield_count=*/6,
            kPolicy);
        graph.begin(signature);
        coordinator.begin_build(graph);
        {
            auto first =
                coordinator.begin_build_occurrence(model, profile);
            (void)model.exchange_composite_input_slot(
                UINT64_C(0xd000000000000000));
            (void)model.consume_composite_preload_role();
            model.drive_trace_cpu(profile);
            first.finish();
        }
        bool rejected_before_submit = false;
        {
            auto second =
                coordinator.begin_build_occurrence(model, profile);
            (void)model.exchange_composite_input_slot(
                UINT64_C(0xd000000000000001));
            (void)model.consume_composite_preload_role();
            model.drive_trace_cpu(profile);
            rejected_before_submit = throws_c10_containing(
                [&] { second.finish(); },
                "exactly three producer occurrences");
        }
        const bool graph_uncommitted =
            graph.state() == RpuKernelGraph::State::RECORDING &&
            !graph.has_built_signature();
        graph.abort();
        const auto legacy_after =
            model.exchange_composite_input_slot(kLegacySentinel);

        EXPECT(
            rejected_before_submit && graph_uncommitted &&
                legacy_before.first == legacy_after.first &&
                legacy_after.second == kLegacySentinel &&
                graph.state() == RpuKernelGraph::State::PASSTHROUGH &&
                !RpuKernelGraph::has_active(),
            "stable-three producer count fails before Graph submission "
            "and cancellation restores the legacy mutable-input slot");
    }

    {
        constexpr uint64_t kPolicy =
            UINT64_C(0x636f6d706f736975);

        ResolvedTraversalCallbackYieldContractModel model;
        model.set_opaque_callback(
            ResolvedTraversalContractBase::OpaqueCallback::
                CallbackYieldPostFn);
        model.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::
                LayerProducerYield);
        model.prepare_cpu();
        const auto profile = model.resolve(request_a);

        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 8004;
        signature.segment_key =
            profile.profile_hash() ^ UINT64_C(1);
        if (signature.segment_key == 0) {
            signature.segment_key = UINT64_C(2);
        }

        SpmCompositeTraceCoordinator coordinator(
            /*expected_occurrence_count=*/1,
            /*expected_producer_yield_count=*/3,
            kPolicy);
        graph.begin(signature);
        coordinator.begin_build(graph);
        {
            auto occurrence =
                coordinator.begin_build_occurrence(model, profile);
            model.drive_trace_cpu(profile);
            occurrence.finish();
        }
        graph.record_branch(
            UINT64_C(0x80040001), GraphSignature{},
            "spm_composite_unclaimed_suffix");
        graph.capture_data(BranchNodeData{});

        const bool rejected = throws_c10_containing(
            [&] { graph.end(); }, "unclaimed Graph suffix");
        const bool rejected_before_commit =
            graph.state() == RpuKernelGraph::State::RECORDING &&
            !graph.has_built_signature();
        coordinator.cancel();
        graph.abort();

        EXPECT(
            rejected && rejected_before_commit &&
                graph.state() == RpuKernelGraph::State::PASSTHROUGH &&
                !RpuKernelGraph::has_active(),
            "composite Graph rejects an unclaimed suffix before build "
            "commit or device submission and cleans up through cancel plus "
            "abort");
    }

    {
        constexpr uint64_t kPolicy =
            UINT64_C(0x696e7374616e7401);

        ResolvedTraversalCallbackYieldContractModel model;
        model.set_opaque_callback(
            ResolvedTraversalContractBase::OpaqueCallback::
                CallbackYieldPostFn);
        model.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::
                LayerProducerYield);
        model.prepare_cpu();
        const auto profile = model.resolve(request_a);

        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 8007;
        signature.segment_key =
            profile.profile_hash() ^ UINT64_C(0x37);
        if (signature.segment_key == 0) {
            signature.segment_key = UINT64_C(0x37);
        }
        SpmCompositeTraceCoordinator coordinator(
            /*expected_occurrence_count=*/1,
            /*expected_producer_yield_count=*/3,
            kPolicy);
        graph.begin(signature);
        coordinator.begin_build(graph);

        bool sync_rejected = false;
        bool raw_fallback_rejected = true;
        bool unchanged_before_cancel = false;
        {
            auto occurrence =
                coordinator.begin_build_occurrence(model, profile);
            graph.capture_data(BranchNodeData{});
            const GraphOpStreamStamp before = graph.op_stream_stamp();
            const size_t nodes_before = graph.graph_size();

            sync_rejected = throws_c10_containing(
                [&] { graph.sync_point(); },
                "sync_point is forbidden while a composite FMB "
                "invocation is open");
#if RHINO_LAUNCH_HAS_DEV_PROGRAM_API
            std::vector<char> instr(64, 0);
            ::rhino_lkn::Kernel_t raw_kernel(
                instr.data(), instr.size());
            raw_fallback_rejected = throws_c10_containing(
                [&] {
                    graph.enqueue(raw_kernel,
                                  /*grid_dims=*/{1, 1, 1},
                                  /*core_ids=*/{0});
                },
                "raw-kernel downgrade is forbidden while a composite FMB "
                "invocation is open");
#endif

            const GraphOpStreamStamp after = graph.op_stream_stamp();
            unchanged_before_cancel =
                graph.state() == RpuKernelGraph::State::RECORDING &&
                graph.graph_size() == nodes_before &&
                !graph.has_built_signature() &&
                before.graph == after.graph &&
                before.state == after.state &&
                before.build_generation == after.build_generation &&
                before.position == after.position &&
                before.extent == after.extent;
        }
        coordinator.cancel();
        graph.abort();

        EXPECT(
            sync_rejected && raw_fallback_rejected &&
                unchanged_before_cancel &&
                graph.state() == RpuKernelGraph::State::PASSTHROUGH &&
                graph.graph_size() == 0 &&
                !RpuKernelGraph::has_active(),
            "an open composite invocation rejects sync_point and raw-kernel "
            "downgrade before node, cursor, or SDK mutation");
    }

    {
        ResolvedTraversalContractModel model;
        model.prepare_cpu();
        EXPECT(throws_c10_containing(
                   [&] { model.resolve_live(request_a); },
                   "real allocation snapshot"),
               "live profile resolver rejects a CPU-dry allocation snapshot");
        const auto profile = model.resolve(request_a);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 801;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const SpmFmbSealedBuildTrace trace =
            model.seal_trace(profile, graph);
        model.validate_trace(trace, graph);
        EXPECT(trace.cpu_dry() && trace.trace_hash() != 0 &&
                   trace.profile_hash() == profile.profile_hash() &&
                   trace.callback_count() == 4 &&
                   trace.active_callback_count() == 2 &&
                   trace.noop_callback_count() == 2 &&
                   trace.access_count() == 4 &&
                   trace.graph_topology_hash() ==
                       graph.build_topology_hash(),
               "schema-v8 records exact active/no-op Branch callback "
               "callback windows and facade access counts");

        const uint64_t build_generation = graph.build_generation();
        const uint64_t build_topology = graph.build_topology_hash();
        graph.begin(signature);
        for (int layer = 0; layer < 2; ++layer) {
            graph.record_branch(
                (static_cast<uint64_t>(layer + 1) << 32) | 1,
                GraphSignature{}, "spm_schema_v8_cpu_fixture");
            graph.capture_data(BranchNodeData{});
        }
        graph.end();
        model.validate_trace(trace, graph);
        EXPECT(graph.build_generation() == build_generation &&
                   graph.build_topology_hash() == build_topology,
               "schema-v8 token remains valid after stable Graph REPLAY");

        GraphSignature replacement = signature;
        replacement.op_id = 802;
        graph.begin(replacement);
        graph.record_branch(0x802, {}, "spm_schema_v8_generation_drift");
        graph.capture_data(BranchNodeData{});
        graph.end();
        EXPECT(throws_c10_containing(
                   [&] { model.validate_trace(trace, graph); },
                   "Graph identity is stale"),
               "fresh Graph BUILD generation stales an old schema-v8 token");
    }

    {
        ResolvedTraversalContractModel model;
        model.set_trace_per_layer(true);
        model.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::AllFacades);
        model.prepare_cpu();
        const auto profile = model.resolve(request_a);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 8011;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        EXPECT(trace.callback_count() == 4 &&
                   trace.active_callback_count() == 4 &&
                   trace.access_count() == 16 &&
                   trace.trace_hash() != 0,
               "schema-v8 positively records Addr, LayerAddr, AddrOffset, "
               "and LayerAddrOffset");
    }

    {
        ResolvedTraversalContractModel model;
        model.prepare_cpu();
        const auto capture_hash =
            [&](ResolvedTraversalContractBase::Traversal traversal) {
                model.set_traversal(traversal);
                const auto profile = model.resolve(request_a);
                RpuKernelGraph graph;
                GraphSignature signature;
                signature.op_id = 803;
                signature.segment_key = profile.profile_hash();
                graph.begin(signature);
                model.begin_trace(profile);
                model.drive_trace_cpu(profile);
                graph.end();
                return model.seal_trace(profile, graph).trace_hash();
            };
        const uint64_t trace_a = capture_hash(
            ResolvedTraversalContractBase::Traversal::Sequential);
        const uint64_t trace_b = capture_hash(
            ResolvedTraversalContractBase::Traversal::ChunkOuter);
        const uint64_t trace_a_again = capture_hash(
            ResolvedTraversalContractBase::Traversal::Sequential);
        EXPECT(trace_a != trace_b && trace_a == trace_a_again,
               "schema-v8 BUILD trace identity is deterministic under A/B/A");
    }

    const auto trace_mode_rejected =
        [&](ResolvedTraversalContractBase::TraceCallback mode,
            const char* reason) {
            ResolvedTraversalContractModel model;
            model.set_trace_callback(mode);
            model.prepare_cpu();
            const auto profile = model.resolve(request_a);
            RpuKernelGraph graph;
            GraphSignature signature;
            signature.op_id = 804;
            signature.segment_key = profile.profile_hash();
            graph.begin(signature);
            model.begin_trace(profile);
            const bool rejected = throws_c10_containing(
                [&] { model.drive_trace_cpu(profile); }, reason);
            graph.abort();
            return rejected;
        };
    EXPECT(trace_mode_rejected(
               ResolvedTraversalContractBase::TraceCallback::AccessWithoutNode,
               "accessed owned SPM") &&
               trace_mode_rejected(
                   ResolvedTraversalContractBase::TraceCallback::NodeWithoutAccess,
                   "without an owned SPM access"),
           "schema-v8 rejects access-without-node and node-without-access");
    {
        ResolvedTraversalContractModel model;
        model.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::AccessWithoutNode);
        model.prepare_cpu();
        const auto profile = model.resolve(request_a);
        RpuKernelGraph failed_graph;
        GraphSignature signature;
        signature.op_id = 8041;
        signature.segment_key = profile.profile_hash();
        failed_graph.begin(signature);
        model.begin_trace(profile);
        EXPECT(throws_c10(
                   [&] { model.drive_trace_cpu(profile); }),
               "schema-v8 owner-thread failure clears its candidate");
        failed_graph.abort();

        model.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::
                ActiveChunk0NoopFollowers);
        RpuKernelGraph recovered_graph;
        recovered_graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        recovered_graph.end();
        const auto recovered =
            model.seal_trace(profile, recovered_graph);
        EXPECT(recovered.trace_hash() != 0,
               "schema-v8 can re-arm after fail-closed owner-thread cleanup");
    }
    EXPECT(trace_mode_rejected(
               ResolvedTraversalContractBase::TraceCallback::ChildGraphNode,
               "opaque ChildGraph") &&
               trace_mode_rejected(
                   ResolvedTraversalContractBase::TraceCallback::HostCallbackNode,
                   "opaque ChildGraph") &&
               trace_mode_rejected(
                   ResolvedTraversalContractBase::TraceCallback::Tier3Node,
                   "opaque ChildGraph"),
           "schema-v8 rejects ChildGraph, HostCallback, and Tier3 windows");

    {
        std::array<uint8_t, 16> child_src{};
        std::array<uint8_t, 16> child_dst{};
        auto child = std::make_shared<RpuKernelGraph>();
        GraphSignature child_signature;
        child_signature.op_id = 8042;
        child->begin(child_signature);
        MemcpyNodeData child_copy;
        child_copy.dst = child_dst.data();
        child_copy.src = child_src.data();
        child_copy.bytes = child_src.size();
        child_copy.kind = CopyKind::HOST_MEMCPY;
        child->capture_data(std::move(child_copy));
        child->end();

        ResolvedTraversalContractModel model;
        model.prepare_cpu();
        const auto profile = model.resolve(request_a);
        RpuKernelGraph outer;
        GraphSignature outer_signature;
        outer_signature.op_id = 8043;
        outer_signature.segment_key = profile.profile_hash();
        outer.begin(outer_signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        outer.register_child(
            child, {}, child_signature,
            {
                DataPatch{0, 0,
                    reinterpret_cast<uint64_t>(child_dst.data()), 0},
                DataPatch{0, 1,
                    reinterpret_cast<uint64_t>(child_src.data()), 0},
            });
        outer.end();
        EXPECT(throws_c10_containing(
                   [&] { (void)model.seal_trace(profile, outer); },
                   "outer Graph contains an opaque ChildGraph"),
               "schema-v8 rejects a legal ChildGraph in the outer BUILD "
               "suffix without recursively inspecting child provenance");
        model.cancel_trace();
    }

    {
        ResolvedTraversalContractModel model(
            ResolvedTraversalContractBase::Traversal::KvFirst);
        model.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::KvScopeDrift);
        model.prepare_cpu(16, 32, 8);
        const auto profile = model.resolve(resolved_request(1, true));
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 805;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        EXPECT(throws_c10_containing(
                   [&] { model.drive_trace_cpu(profile); },
                   "scope drift"),
               "schema-v8 rejects callback scope drift");
        graph.abort();
    }
    {
        ResolvedTraversalContractModel model;
        model.set_trace_per_layer(true);
        model.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::PerLayerDrift);
        model.prepare_cpu();
        const auto profile = model.resolve(request_a);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 806;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        EXPECT(throws_c10_containing(
                   [&] { model.drive_trace_cpu(profile); },
                   "does not match callback layer"),
               "schema-v8 rejects TempPerLayer callback-layer drift");
        graph.abort();
    }
    {
        ResolvedTraversalContractModel model;
        model.prepare_cpu();
        const auto profile = model.resolve(request_a);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 807;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        EXPECT(throws_c10_containing(
                   [&] { model.touch_unscoped_for_contract(); },
                   "unscoped"),
               "schema-v8 rejects facade access outside a callback window");
        model.cancel_trace();
        graph.abort();
    }
    {
        ResolvedTraversalPersistentPreloadContractModel model;
        model.set_persistent_preload(true);
        model.set_preload_access(
            ResolvedTraversalContractBase::PreloadAccess::Persistent);
        model.prepare_live();
        const auto profile = model.resolve_live(request_a);
        GraphSignature signature;
        signature.op_id = 8073;
        signature.segment_key = profile.profile_hash();

        RpuKernelGraph accepted_graph;
        accepted_graph.begin(signature);
        model.begin_trace(profile);
        model.drive_preload_for_contract();
        EXPECT(throws_c10_containing(
                   [&] { model.touch_persistent_unscoped_for_contract(); },
                   "unscoped"),
               "schema-v8 closes persistent preload authority when its "
               "callback returns");
        model.cancel_trace();
        accepted_graph.abort();

        model.set_preload_access(
            ResolvedTraversalContractBase::PreloadAccess::Temporary);
        RpuKernelGraph rejected_graph;
        rejected_graph.begin(signature);
        model.begin_trace(profile);
        EXPECT(throws_c10_containing(
                   [&] { model.drive_preload_for_contract(); },
                   "accessed non-persistent"),
               "schema-v8 persistent preload lexical scope admits owned "
               "persistent facades and rejects temporary SPM access");
        EXPECT(throws_c10_containing(
                   [&] { model.touch_persistent_unscoped_for_contract(); },
                   "unscoped"),
               "schema-v8 closes persistent preload authority after a "
               "callback exception");
        model.cancel_trace();
        rejected_graph.abort();
    }
    {
        ResolvedTraversalContractModel model;
        model.prepare_cpu();
        const auto profile = model.resolve(request_a);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 8071;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        EXPECT(throws_c10_containing(
                   [&] { model.touch_unscoped_for_contract(); },
                   "unscoped"),
               "schema-v8 keeps its TLS guard through the BUILD suffix");
        model.cancel_trace();
        graph.abort();
    }
    {
        ResolvedTraversalContractModel model;
        model.prepare_cpu();
        const auto profile = model.resolve(request_a);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 8072;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        std::atomic<bool> facade_rejected{false};
        std::atomic<bool> driver_rejected{false};
        std::thread wrong_thread([&] {
            facade_rejected.store(throws_c10_containing(
                [&] { model.touch_unscoped_for_contract(); },
                "owner thread"));
            driver_rejected.store(throws_c10_containing(
                [&] { model.drive_trace_cpu(profile); },
                "owner thread"));
        });
        wrong_thread.join();
        EXPECT(facade_rejected.load() && driver_rejected.load(),
               "schema-v8 facade and driver reject a wrong owner thread");
        model.cancel_trace();
        graph.abort();
    }
    {
        ResolvedTraversalContractModel model;
        model.prepare_cpu();
        const auto profile = model.resolve(request_a);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 808;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        graph.record_branch(0x808, {}, "spm_schema_v8_unowned_gap");
        EXPECT(throws_c10_containing(
                   [&] { model.drive_trace_cpu(profile); },
                   "unowned Graph nodes"),
               "schema-v8 rejects Graph nodes outside canonical callback "
               "windows");
        graph.abort();
    }
    {
        ResolvedTraversalContractModel model;
        model.prepare_cpu();
        const auto profile = model.resolve(request_a);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 809;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        model.invalidate_for_contract();
        EXPECT(throws_c10_containing(
                   [&] { model.validate_trace(trace, graph); },
                   "owner is stale"),
               "schema-v8 owner generation stales a sealed BUILD trace");
    }

    // Schema-v9 converts only a model-declared canonical active-first chunk
    // run into opaque groups.  The positive fixture uses the Wall-like shape:
    // per body/layer, C0 is active and C1/C2 are true no-ops.
    {
        ResolvedTraversalContractModel model;
        model.set_body_iterations(2);
        model.prepare_cpu(16, 48);
        const auto profile = model.resolve(resolved_three_chunk_request());
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 810;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        const SpmFmbSealedActiveGroups groups =
            model.seal_active_groups(trace, graph);
        model.validate_active_groups(groups, graph);
        EXPECT(groups.cpu_dry() && groups.group_hash() != 0 &&
                   groups.source_trace_hash() == trace.trace_hash() &&
                   groups.profile_hash() == profile.profile_hash() &&
                   groups.group_count() == 4 &&
                   groups.member_callback_count() == 12 &&
                   groups.active_callback_count() == 4 &&
                   groups.noop_callback_count() == 8 &&
                   groups.component_node_count() == 8,
               "schema-v9 seals four exact C0-active/C1-C2-no-op groups "
               "across two bodies and two layers");

        const uint64_t build_generation =
            groups.graph_build_generation();
        const uint64_t group_hash = groups.group_hash();
        graph.begin(signature);
        for (int body = 0; body < 2; ++body) {
            (void)body;
            for (int layer = 0; layer < 2; ++layer) {
                graph.record_branch(
                    (static_cast<uint64_t>(layer + 1) << 32) | 1,
                    GraphSignature{}, "spm_schema_v8_cpu_fixture");
                graph.capture_data(BranchNodeData{});
            }
        }
        graph.end();
        model.validate_active_groups(groups, graph);
        EXPECT(graph.build_generation() == build_generation &&
                   groups.group_hash() == group_hash,
               "schema-v9 groups remain valid through stable ordinary "
               "BUILD-to-REPLAY lifecycle");

        model.invalidate_for_contract();
        EXPECT(throws_c10_containing(
                   [&] { model.validate_active_groups(groups, graph); },
                   "owner is stale"),
               "schema-v9 group validation preserves source owner-stale "
               "rejection");
    }

    {
        ResolvedTraversalContractModel grouped_model;
        ResolvedTraversalUngroupedContractModel ungrouped_model;
        grouped_model.prepare_cpu(16, 48);
        ungrouped_model.prepare_cpu(16, 48);
        const auto request = resolved_three_chunk_request(10);
        const auto grouped_profile = grouped_model.resolve(request);
        const auto ungrouped_profile = ungrouped_model.resolve(request);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 811;
        signature.segment_key = ungrouped_profile.profile_hash();
        graph.begin(signature);
        ungrouped_model.begin_trace(ungrouped_profile);
        ungrouped_model.drive_trace_cpu(ungrouped_profile);
        graph.end();
        const auto trace =
            ungrouped_model.seal_trace(ungrouped_profile, graph);
        EXPECT(grouped_profile.profile_hash() !=
                   ungrouped_profile.profile_hash() &&
                   throws_c10_containing(
                       [&] {
                           (void)ungrouped_model.seal_active_groups(
                               trace, graph);
                       },
                       "has not opted"),
               "schema-v9 policy is explicit, signature-bound, and defaults "
               "to fail-closed");
    }

    {
        ResolvedTraversalContractModel model;
        model.set_trace_per_layer(true);
        model.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::AllFacades);
        model.prepare_cpu(16, 48);
        const auto profile = model.resolve(resolved_three_chunk_request(11));
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 812;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        EXPECT(throws_c10_containing(
                   [&] { (void)model.seal_active_groups(trace, graph); },
                   "active/no-op evidence drift"),
               "schema-v9 rejects a second active callback inside a declared "
               "active-first run");
    }

    {
        ResolvedTraversalContractModel model(
            ResolvedTraversalContractBase::Traversal::ChunkOuter);
        model.prepare_cpu(16, 48);
        const auto profile = model.resolve(resolved_three_chunk_request(12));
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 813;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        EXPECT(throws_c10_containing(
                   [&] { (void)model.seal_active_groups(trace, graph); },
                   "at least one no-op follower"),
               "schema-v9 rejects non-consecutive layer members instead of "
               "inventing cross-window groups");
    }

    {
        ResolvedTraversalContractModel model;
        model.prepare_cpu(16, 48);
        const auto profile = model.resolve(resolved_three_chunk_request(13));
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 814;
        signature.segment_key = profile.profile_hash();
        const auto capture = [&]() {
            graph.begin(signature);
            model.begin_trace(profile);
            model.drive_trace_cpu(profile);
            graph.end();
            const auto trace = model.seal_trace(profile, graph);
            return model.seal_active_groups(trace, graph);
        };

        const SpmFmbSealedActiveGroups groups_a = capture();
        graph.invalidate();
        model.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::
                ActiveChunk0NoopFollowersAlternateTopology);
        const SpmFmbSealedActiveGroups groups_b = capture();
        const bool old_a_stale = throws_c10_containing(
            [&] { model.validate_active_groups(groups_a, graph); },
            "Graph identity is stale");
        graph.invalidate();
        model.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::
                ActiveChunk0NoopFollowers);
        const SpmFmbSealedActiveGroups groups_a_again = capture();
        model.validate_active_groups(groups_a_again, graph);
        EXPECT(old_a_stale &&
                   groups_a.group_hash() != groups_b.group_hash() &&
                   groups_a.group_hash() == groups_a_again.group_hash() &&
                   groups_a.graph_build_generation() !=
                       groups_a_again.graph_build_generation() &&
                   throws_c10_containing(
                       [&] {
                           model.validate_active_groups(groups_b, graph);
                       },
                       "Graph identity is stale"),
               "schema-v9 A/B/A keeps semantic hashes deterministic while "
               "fresh BUILD nonces stale old tokens");
    }

    // Schema-v10 derives a pointer-free dense DDR member universe from the
    // sealed v9 groups.  It proves geometry/arena roles only; no existing Graph
    // DMA node receives endpoint authority from this token.
    {
        ResolvedTraversalDenseDdrContractModel model;
        model.prepare_cpu(16, 48);
        const auto profile = model.resolve(resolved_three_chunk_request(20));
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 820;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        const auto groups = model.seal_active_groups(trace, graph);
        const SpmFmbSealedDenseDdrMembers members =
            model.seal_dense_ddr_members(groups, graph);
        model.validate_dense_ddr_members(members, graph);
        EXPECT(members.cpu_dry() && members.semantic_hash() != 0 &&
                   members.source_group_hash() == groups.group_hash() &&
                   members.profile_hash() == profile.profile_hash() &&
                   members.layer_group_count() == 2 &&
                   members.members_per_group() == 3 &&
                   members.total_member_count() == 6 &&
                   members.row_bytes() == 32 &&
                   members.rows_per_group() == 48 &&
                   members.dense_bytes_per_group() == 1536,
               "schema-v10 seals two layer-local, three-member dense DDR "
               "universes with exact independent-local geometry");

        const uint64_t build_generation =
            members.graph_build_generation();
        graph.begin(signature);
        for (int layer = 0; layer < 2; ++layer) {
            graph.record_branch(
                (static_cast<uint64_t>(layer + 1) << 32) | 1,
                GraphSignature{}, "spm_schema_v8_cpu_fixture");
            graph.capture_data(BranchNodeData{});
        }
        graph.end();
        model.validate_dense_ddr_members(members, graph);
        EXPECT(graph.build_generation() == build_generation,
               "schema-v10 dense DDR members remain valid after stable "
               "ordinary REPLAY");

        model.invalidate_for_contract();
        EXPECT(throws_c10_containing(
                   [&] {
                       model.validate_dense_ddr_members(members, graph);
                   },
                   "owner is stale"),
               "schema-v10 inherits source owner-stale rejection");
    }

    {
        ResolvedTraversalContractModel grouped_model;
        ResolvedTraversalDenseDdrContractModel dense_model;
        grouped_model.prepare_cpu(16, 48);
        dense_model.prepare_cpu(16, 48);
        const auto request = resolved_three_chunk_request(21);
        const auto grouped_profile = grouped_model.resolve(request);
        const auto dense_profile = dense_model.resolve(request);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 821;
        signature.segment_key = grouped_profile.profile_hash();
        graph.begin(signature);
        grouped_model.begin_trace(grouped_profile);
        grouped_model.drive_trace_cpu(grouped_profile);
        graph.end();
        const auto trace = grouped_model.seal_trace(grouped_profile, graph);
        const auto groups = grouped_model.seal_active_groups(trace, graph);
        EXPECT(grouped_profile.profile_hash() != dense_profile.profile_hash() &&
                   throws_c10_containing(
                       [&] {
                           (void)grouped_model.seal_dense_ddr_members(
                               groups, graph);
                       },
                       "has not opted"),
               "schema-v10 capability is signature-bound and defaults to "
               "fail-closed without perturbing v9-only profiles");
    }

    {
        ResolvedTraversalDenseDdrContractModel model;
        model.set_body_iterations(2);
        model.prepare_cpu(16, 48);
        const auto profile = model.resolve(resolved_three_chunk_request(22));
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 822;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        const auto groups = model.seal_active_groups(trace, graph);
        EXPECT(throws_c10_containing(
                   [&] {
                       (void)model.seal_dense_ddr_members(groups, graph);
                   },
                   "resolved traversal is not exact"),
               "schema-v10 initial member universe rejects repeated bodies "
               "instead of guessing a post-member set");
    }

    {
        ResolvedTraversalDenseDdrContractModel model(
            ResolvedTraversalContractBase::Traversal::
                SpmResidentSingleLayerGroups);
        model.prepare_cpu(16, 48);
        const auto profile = model.resolve(
            resolved_three_chunk_request(24));
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 824;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        const auto groups = model.seal_active_groups(trace, graph);
        EXPECT(throws_c10_containing(
                   [&] {
                       (void)model.seal_dense_ddr_members(groups, graph);
                   },
                   "resolved traversal is not exact"),
               "schema-v10 rejects SPM_RESIDENT single-layer groups even "
               "when their callback I/O flags mimic DDR ping-pong");
    }

    {
        ResolvedTraversalDenseDdrContractModel model;
        model.prepare_cpu(16, 48);
        const auto profile = model.resolve(
            resolved_three_chunk_request(25));
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 825;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        const auto groups = model.seal_active_groups(trace, graph);
        model.set_hidden_size_for_contract(32);
        EXPECT(throws_c10_containing(
                   [&] {
                       (void)model.seal_dense_ddr_members(groups, graph);
                   },
                   "resolved policy/shape drift"),
               "schema-v10 rejects hidden-size drift after BUILD instead of "
               "deriving new row geometry from mutable model state");
    }

    {
        ResolvedTraversalDenseDdrContractModel model;
        model.prepare_cpu(16, 48);
        const auto profile = model.resolve(resolved_three_chunk_request(23));
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 823;
        signature.segment_key = profile.profile_hash();
        const auto capture = [&]() {
            graph.begin(signature);
            model.begin_trace(profile);
            model.drive_trace_cpu(profile);
            graph.end();
            const auto trace = model.seal_trace(profile, graph);
            const auto groups = model.seal_active_groups(trace, graph);
            return model.seal_dense_ddr_members(groups, graph);
        };
        const SpmFmbSealedDenseDdrMembers members_a = capture();
        graph.invalidate();
        model.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::
                ActiveChunk0NoopFollowersAlternateTopology);
        const SpmFmbSealedDenseDdrMembers members_b = capture();
        const bool old_a_stale = throws_c10_containing(
            [&] { model.validate_dense_ddr_members(members_a, graph); },
            "Graph identity is stale");
        graph.invalidate();
        model.set_trace_callback(
            ResolvedTraversalContractBase::TraceCallback::
                ActiveChunk0NoopFollowers);
        const SpmFmbSealedDenseDdrMembers members_a_again = capture();
        model.validate_dense_ddr_members(members_a_again, graph);
        EXPECT(old_a_stale &&
                   members_a.semantic_hash() != members_b.semantic_hash() &&
                   members_a.semantic_hash() ==
                       members_a_again.semantic_hash() &&
                   members_a.graph_build_generation() !=
                       members_a_again.graph_build_generation() &&
                   throws_c10_containing(
                       [&] {
                           model.validate_dense_ddr_members(members_b, graph);
                       },
                       "Graph identity is stale"),
               "schema-v10 A/B/A keeps semantic identity deterministic while "
               "fresh BUILD nonces stale old member tokens");
    }

    // Schema-v11 binds FMB-minted member/role IDs to ordered Graph DMA
    // occurrences.  The fixture uses two ingress lanes per logical endpoint
    // and one egress lane backed by live managed buffers.
    {
        ResolvedTraversalDmaEndpointContractModel model;
        model.prepare_cpu(16, 48);
        const auto request = resolved_three_chunk_request(30);
        const auto profile = model.resolve(request);
        model.configure_dma_endpoints(profile, request.chunks);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 830;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        const auto groups = model.seal_active_groups(trace, graph);
        const auto members = model.seal_dense_ddr_members(groups, graph);
        const SpmFmbSealedDmaEndpointBindings bindings =
            model.seal_dma_endpoint_bindings(members, graph);
        model.validate_dma_endpoint_bindings(bindings, graph);
        EXPECT(bindings.cpu_dry() && bindings.binding_hash() != 0 &&
                   bindings.source_member_hash() ==
                       members.semantic_hash() &&
                   bindings.profile_hash() == profile.profile_hash() &&
                   bindings.logical_endpoint_count() == 12 &&
                   bindings.dma_node_count() == 18 &&
                   bindings.fixed_dma_node_count() == 12 &&
                   bindings.mutable_dma_node_count() == 6 &&
                   bindings.legacy_dma_node_count() == 0,
               "schema-v11 seals every member ingress/egress with ordered "
               "multi-DMA multiplicity and no public endpoint payload");

        const uint64_t build_generation =
            bindings.graph_build_generation();
        model.set_live_input_base(model.replay_live_input_base());
        graph.begin(signature);
        const bool unarmed_skip_rejected = throws_c10_containing(
            [&] { graph.skip_op_stream_for_fast_replay(); },
            "requires an armed owning FMB token");
        model.arm_dma_endpoint_replay(bindings, graph);
        model.replay_dma_fixture();
        graph.end();
        model.validate_dma_endpoint_bindings(bindings, graph);
        EXPECT(unarmed_skip_rejected &&
                   graph.build_generation() == build_generation,
               "schema-v11 ordinary REPLAY permits only the live value "
               "behind one stable mutable slot to change and rejects an "
               "unarmed skip without side effects");

        bool all_probe_rejections = true;
        for (const auto probe : {
                 ResolvedTraversalDmaEndpointContractModel::ReplayProbe::
                     MutableSlot,
                 ResolvedTraversalDmaEndpointContractModel::ReplayProbe::
                     MutableOffset,
                 ResolvedTraversalDmaEndpointContractModel::ReplayProbe::Bytes,
                 ResolvedTraversalDmaEndpointContractModel::ReplayProbe::Channel,
                 ResolvedTraversalDmaEndpointContractModel::ReplayProbe::
                     EndpointId,
                 ResolvedTraversalDmaEndpointContractModel::ReplayProbe::
                     MutableFixedPeer,
                 ResolvedTraversalDmaEndpointContractModel::ReplayProbe::
                     FixedPeer}) {
            graph.begin(signature);
            model.arm_dma_endpoint_replay(bindings, graph);
            all_probe_rejections =
                model.replay_dma_fixture_with_probe(probe) &&
                all_probe_rejections;
            graph.end();
        }
        model.validate_dma_endpoint_bindings(bindings, graph);
        EXPECT(all_probe_rejections,
               "schema-v11 rejects slot, offset, bytes, channel, ID, and "
               "mutable/fixed-peer drift without consuming the replay "
               "cursor");

        graph.begin(signature);
        model.arm_dma_endpoint_replay(bindings, graph);
        graph.skip_op_stream_for_fast_replay();
        graph.end();
        model.validate_dma_endpoint_bindings(bindings, graph);
        EXPECT(graph.build_generation() == build_generation,
               "schema-v11 sealed endpoint metadata survives a stable full "
               "fast REPLAY without host endpoint re-emission");

        graph.begin(signature);
        model.set_hidden_size_for_contract(32);
        const bool model_shape_arm_rejected = throws_c10_containing(
            [&] { model.arm_dma_endpoint_replay(bindings, graph); },
            "sealed model/owner/Graph identity is stale");
        model.set_hidden_size_for_contract(16);
        model.arm_dma_endpoint_replay(bindings, graph);
        graph.skip_op_stream_for_fast_replay();
        graph.end();
        EXPECT(model_shape_arm_rejected,
               "schema-v11 rejects current model-shape drift before fast "
               "REPLAY and permits the restored sealed owner in the same "
               "scope");

        graph.begin(signature);
        model.arm_dma_endpoint_replay(bindings, graph);
        graph.skip_op_stream_for_fast_replay();
        model.invalidate_for_contract();
        const bool stale_execution_rejected = throws_c10_containing(
            [&] { graph.end(); },
            "owner is stale or unarmed before Graph execution");
        graph.abort();
        const bool stale_token_rejected = throws_c10_containing(
            [&] { model.validate_dma_endpoint_bindings(bindings, graph); },
            "owner is stale");
        EXPECT(stale_execution_rejected && stale_token_rejected,
               "schema-v11 rejects owner generation drift before fast "
               "REPLAY execution and stales the sealed token");
    }

    {
        ResolvedTraversalDenseDdrContractModel legacy_model;
        ResolvedTraversalDmaEndpointContractModel endpoint_model;
        legacy_model.prepare_cpu(16, 48);
        endpoint_model.prepare_cpu(16, 48);
        const auto request = resolved_three_chunk_request(31);
        const auto legacy_profile = legacy_model.resolve(request);
        const auto endpoint_profile = endpoint_model.resolve(request);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 831;
        signature.segment_key = legacy_profile.profile_hash();
        graph.begin(signature);
        legacy_model.begin_trace(legacy_profile);
        legacy_model.drive_trace_cpu(legacy_profile);
        graph.end();
        const auto trace = legacy_model.seal_trace(legacy_profile, graph);
        const auto groups = legacy_model.seal_active_groups(trace, graph);
        const auto members =
            legacy_model.seal_dense_ddr_members(groups, graph);
        EXPECT(legacy_profile.profile_hash() !=
                       endpoint_profile.profile_hash() &&
                   throws_c10_containing(
                       [&] {
                           (void)legacy_model.seal_dma_endpoint_bindings(
                               members, graph);
                       },
                       "has not opted"),
               "schema-v11 capability is signature-bound and ID-0/v10-only "
               "graphs default to fail-closed");
    }

    {
        ResolvedTraversalDmaEndpointContractModel model;
        model.prepare_cpu(16, 48);
        const auto request = resolved_three_chunk_request(32);
        const auto profile = model.resolve(request);
        model.configure_dma_endpoints(profile, request.chunks);
        model.set_omit_last_egress(true);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 832;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        const auto groups = model.seal_active_groups(trace, graph);
        const auto members = model.seal_dense_ddr_members(groups, graph);
        EXPECT(throws_c10_containing(
                   [&] {
                       (void)model.seal_dma_endpoint_bindings(
                           members, graph);
                   },
                   "missing, misordered"),
               "schema-v11 rejects a missing member/role endpoint instead "
               "of accepting a partial semantic DMA universe");
    }

    {
        ResolvedTraversalDmaEndpointContractModel model;
        model.prepare_cpu(16, 48);
        const auto request = resolved_three_chunk_request(34);
        const auto profile = model.resolve(request);
        model.configure_dma_endpoints(profile, request.chunks);
        model.set_inject_legacy_dma(true);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 834;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        const auto groups = model.seal_active_groups(trace, graph);
        const auto members = model.seal_dense_ddr_members(groups, graph);
        EXPECT(throws_c10_containing(
                   [&] {
                       (void)model.seal_dma_endpoint_bindings(
                           members, graph);
                   },
                   "legacy ID-0 DMA"),
               "schema-v11 rejects legacy ID-0 DMA mixed into a claimed "
               "carrier");
    }

    {
        ResolvedTraversalDmaEndpointContractModel model;
        model.prepare_cpu(16, 48);
        const auto request = resolved_three_chunk_request(35);
        const auto profile = model.resolve(request);
        model.configure_dma_endpoints(
            profile, request.chunks, /*ingress_lanes=*/3);
        model.set_emitted_ingress_lanes(2);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 835;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        const auto groups = model.seal_active_groups(trace, graph);
        const auto members = model.seal_dense_ddr_members(groups, graph);
        EXPECT(throws_c10_containing(
                   [&] {
                       (void)model.seal_dma_endpoint_bindings(
                           members, graph);
                   },
                   "observed physical DMA multiplicity"),
               "schema-v11 rejects an omitted lane against the "
               "ID-declared physical multiplicity");
    }

    {
        ResolvedTraversalDmaEndpointContractModel model;
        model.prepare_cpu(16, 48);
        const auto request = resolved_three_chunk_request(36);
        const auto profile = model.resolve(request);
        model.configure_dma_endpoints(
            profile, request.chunks, /*ingress_lanes=*/3);
        model.set_emitted_ingress_lanes(2);
        model.set_channel_gap(true);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 836;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        const auto groups = model.seal_active_groups(trace, graph);
        const auto members = model.seal_dense_ddr_members(groups, graph);
        EXPECT(throws_c10_containing(
                   [&] {
                       (void)model.seal_dma_endpoint_bindings(
                           members, graph);
                   },
                   "channel drift"),
               "schema-v11 rejects an ingress lane gap 0,2");
    }

    {
        ResolvedTraversalDmaEndpointContractModel model;
        model.prepare_cpu(16, 48);
        const auto request = resolved_three_chunk_request(37);
        const auto profile = model.resolve(request);
        model.configure_dma_endpoints(profile, request.chunks);
        model.set_misorder_first_endpoint(true);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 837;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        const auto groups = model.seal_active_groups(trace, graph);
        const auto members = model.seal_dense_ddr_members(groups, graph);
        EXPECT(throws_c10_containing(
                   [&] {
                       (void)model.seal_dma_endpoint_bindings(
                           members, graph);
                   },
                   "misordered"),
               "schema-v11 rejects role misorder/duplicate before the "
               "canonical member ingress");
    }

    {
        ResolvedTraversalDmaEndpointContractModel model;
        model.prepare_cpu(16, 48);
        const auto request = resolved_three_chunk_request(38);
        const auto profile = model.resolve(request);
        model.configure_dma_endpoints(profile, request.chunks);
        model.set_split_mutable_slots(true);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 838;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        const auto groups = model.seal_active_groups(trace, graph);
        const auto members = model.seal_dense_ddr_members(groups, graph);
        EXPECT(throws_c10_containing(
                   [&] {
                       (void)model.seal_dma_endpoint_bindings(
                           members, graph);
                   },
                   "multiple live slots"),
               "schema-v11 rejects multiple mutable slots behind one "
               "logical endpoint");
    }

    {
        ResolvedTraversalDmaEndpointContractModel model;
        model.prepare_cpu(16, 48);
        const auto request = resolved_three_chunk_request(33);
        const auto profile = model.resolve(request);
        model.configure_dma_endpoints(profile, request.chunks);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 833;
        signature.segment_key = profile.profile_hash();
        const auto capture = [&]() {
            graph.begin(signature);
            model.begin_trace(profile);
            model.drive_trace_cpu(profile);
            graph.end();
            const auto trace = model.seal_trace(profile, graph);
            const auto groups = model.seal_active_groups(trace, graph);
            const auto members = model.seal_dense_ddr_members(groups, graph);
            return model.seal_dma_endpoint_bindings(members, graph);
        };
        const SpmFmbSealedDmaEndpointBindings bindings_a = capture();
        graph.invalidate();
        model.set_alternate_topology(true);
        const SpmFmbSealedDmaEndpointBindings bindings_b = capture();
        const bool old_a_stale = throws_c10_containing(
            [&] { model.validate_dma_endpoint_bindings(bindings_a, graph); },
            "Graph identity is stale");
        graph.invalidate();
        model.set_alternate_topology(false);
        const SpmFmbSealedDmaEndpointBindings bindings_a_again = capture();
        model.validate_dma_endpoint_bindings(bindings_a_again, graph);
        EXPECT(old_a_stale &&
                   bindings_a.binding_hash() != bindings_b.binding_hash() &&
                   bindings_a.binding_hash() ==
                       bindings_a_again.binding_hash() &&
                   bindings_a.graph_build_generation() !=
                       bindings_a_again.graph_build_generation() &&
                   throws_c10_containing(
                       [&] {
                           model.validate_dma_endpoint_bindings(
                               bindings_b, graph);
                       },
                       "Graph identity is stale"),
               "schema-v11 A/B/A keeps semantic identity deterministic "
               "while fresh BUILD nonces stale old bindings");
    }

    {
        ResolvedTraversalDmaEndpointContractModel model;
        model.prepare_cpu(16, 48);
        const auto request = resolved_three_chunk_request(53);
        const auto profile = model.resolve(request);
        model.configure_dma_endpoints(profile, request.chunks);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 849;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        model.emit_first_ingress_outside_callback();

        const GraphOpStreamStamp before = graph.op_stream_stamp();
        const size_t nodes_before = graph.graph_size();
        const bool direct_dma_rejected = throws_c10_containing(
            [&] {
                rpu_launch_ddr_spm_bytes_immediate(
                    /*ddr_ptr=*/nullptr, /*num_bytes=*/16,
                    /*spm_addr_unified=*/0);
            },
            "direct/immediate DMA is forbidden with an active Graph");

#if RHINO_LAUNCH_HAS_DEV_PROGRAM_API
        std::vector<char> instr(64, 0);
        ::rhino_lkn::Kernel_t raw_kernel(instr.data(), instr.size());
        const bool raw_fallback_rejected = throws_c10_containing(
            [&] {
                graph.enqueue(raw_kernel,
                              /*grid_dims=*/{1, 1, 1},
                              /*core_ids=*/{0});
            },
            "raw-kernel fallback is forbidden after semantic/canonical DMA "
            "authority has been established");
#else
        const bool raw_fallback_rejected = true;
#endif
        const GraphOpStreamStamp after = graph.op_stream_stamp();
        const bool pre_side_effect_rejection =
            direct_dma_rejected && raw_fallback_rejected &&
            graph.state() == RpuKernelGraph::State::RECORDING &&
            graph.graph_size() == nodes_before &&
            before.graph == after.graph && before.state == after.state &&
            before.build_generation == after.build_generation &&
            before.position == after.position &&
            before.extent == after.extent;
        graph.abort();
        EXPECT(pre_side_effect_rejection &&
                   graph.state() == RpuKernelGraph::State::PASSTHROUGH &&
                   graph.graph_size() == 0 &&
                   !RpuKernelGraph::has_active(),
               "schema-v11 semantic/canonical authority rejects direct DMA "
               "and raw-kernel downgrade before node/cursor/SDK mutation, "
               "then abort cleans the scope");
    }

    {
        ResolvedTraversalDmaEndpointContractModel model;
        model.prepare_cpu(16, 48);
        const auto request = resolved_three_chunk_request(39);
        const auto profile = model.resolve(request);
        model.configure_dma_endpoints(profile, request.chunks);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 839;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        model.emit_first_ingress_outside_callback();
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        const auto groups = model.seal_active_groups(trace, graph);
        const auto members = model.seal_dense_ddr_members(groups, graph);
        EXPECT(throws_c10_containing(
                   [&] {
                       (void)model.seal_dma_endpoint_bindings(
                           members, graph);
                   },
                   "outside the claimed FMB callback interval"),
               "schema-v11 rejects semantic DMA outside the claimed FMB "
               "callback interval");
    }

    {
        ResolvedTraversalDmaEndpointContractModel foreign_model;
        ResolvedTraversalDmaEndpointContractModel owning_model;
        foreign_model.prepare_cpu(16, 48);
        owning_model.prepare_cpu(16, 48);
        const auto request = resolved_three_chunk_request(40);
        const auto foreign_profile = foreign_model.resolve(request);
        const auto owning_profile = owning_model.resolve(request);
        foreign_model.configure_dma_endpoints(
            foreign_profile, request.chunks);
        owning_model.configure_dma_endpoints(
            owning_profile, request.chunks);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 840;
        signature.segment_key = owning_profile.profile_hash();
        graph.begin(signature);
        owning_model.begin_trace(owning_profile);
        const bool foreign_owner_rejected = throws_c10_containing(
            [&] { foreign_model.emit_first_ingress_outside_callback(); },
            "differs from the armed FMB trace");
        owning_model.drive_trace_cpu(owning_profile);
        graph.end();
        const auto trace = owning_model.seal_trace(
            owning_profile, graph);
        const auto groups = owning_model.seal_active_groups(trace, graph);
        const auto members = owning_model.seal_dense_ddr_members(
            groups, graph);
        const auto bindings = owning_model.seal_dma_endpoint_bindings(
            members, graph);
        owning_model.validate_dma_endpoint_bindings(bindings, graph);
        graph.begin(signature);
        const bool foreign_replay_arm_rejected = throws_c10_containing(
            [&] { foreign_model.arm_dma_endpoint_replay(bindings, graph); },
            "sealed model/owner/Graph identity is stale");
        owning_model.arm_dma_endpoint_replay(bindings, graph);
        graph.skip_op_stream_for_fast_replay();
        graph.end();
        owning_model.validate_dma_endpoint_bindings(bindings, graph);
        EXPECT(foreign_profile.profile_hash() ==
                       owning_profile.profile_hash() &&
                   foreign_owner_rejected &&
                   foreign_replay_arm_rejected &&
                   bindings.logical_endpoint_count() == 12,
               "schema-v11 rejects a same-ID endpoint from another FMB "
               "owner before BUILD and fast REPLAY execution");
    }

    // Schema-v12 removes caller-authored endpoint/range authority.  The
    // subclass sees only callback-epoch-bound member refs and fixed-role FMB
    // wrappers; Graph rejects every raw helper before node/cursor mutation.
    static_assert(!std::is_default_constructible_v<
                  v3::SpmFmbRuntimeDenseDdrMemberRef>);
    static_assert(!std::is_aggregate_v<
                  v3::SpmFmbRuntimeDenseDdrMemberRef>);
    static_assert(!std::is_constructible_v<
                  v3::SpmFmbRuntimeDenseDdrMemberRef, uint64_t>);
    static_assert(!std::is_convertible_v<
                  v3::SpmFmbRuntimeDenseDdrMemberRef,
                  GraphDmaSemanticEndpoint>);
    EXPECT(true,
           "schema-v12 runtime member refs are opaque and have no raw "
           "endpoint/numeric conversion");

    {
        ResolvedTraversalRuntimeMemberContractModel model;
        model.prepare_cpu(16, 48);
        const auto request = resolved_three_chunk_request(41);
        const auto profile = model.resolve(request);
        EXPECT(model.count_outside_callback_rejected() &&
                   throws_c10_containing(
                       [&] {
                           (void)model.mint_dma_endpoint(
                               profile, /*layer=*/0, /*member=*/0,
                               v3::SpmFmbDmaEndpointRole::Ingress,
                               /*occurrences=*/8);
                       },
                       "raw semantic endpoint mint is forbidden"),
               "schema-v12 rejects callback-external member access and the "
               "old raw endpoint mint");
        at::Tensor empty_hidden;
        std::vector<at::Tensor> empty_k;
        std::vector<at::Tensor> empty_v;
        EXPECT(throws_c10_containing(
                   [&] {
                       (void)model.run_all_layers(
                           empty_hidden, empty_k, empty_v, std::nullopt,
                           /*position=*/0, /*is_causal=*/true);
                   },
                   "does not authorize production run_all_layers"),
               "schema-v12 CPU capability fails closed at the production "
               "driver entry");

        ResolvedTraversalDmaEndpointContractModel v11_model;
        v11_model.prepare_cpu(16, 48);
        const auto v11_profile = v11_model.resolve(request);
        EXPECT(profile.profile_hash() != v11_profile.profile_hash(),
               "schema-v12 appends independent domain-21 policy identity "
               "without changing v11");

        {
            RpuKernelGraph staged_kernel_graph;
            GraphSignature staged_signature;
            staged_signature.op_id = 840;
            staged_signature.segment_key = profile.profile_hash();
            staged_kernel_graph.begin(staged_signature);
            RpuKernelGraphContractTestAccess::stage_pending_kernel(
                staged_kernel_graph, KernelId::BINARY_SAMESHAPE);
            const GraphOpStreamStamp before =
                staged_kernel_graph.op_stream_stamp();
            const bool rejected = throws_c10_containing(
                [&] { model.begin_trace(profile); },
                "has a pending kernel");
            const GraphOpStreamStamp after =
                staged_kernel_graph.op_stream_stamp();
            staged_kernel_graph.abort();
            EXPECT(rejected && before.graph == after.graph &&
                       before.state == after.state &&
                       before.build_generation == after.build_generation &&
                       before.position == after.position &&
                       before.extent == after.extent,
                   "schema-v12 rejects a kernel staged before the "
                   "canonical BUILD trace without recording a Graph node");
        }

        RpuKernelGraph bypass_probe_child;
        GraphSignature child_signature;
        child_signature.op_id = 839;
        bypass_probe_child.begin(child_signature);
        bypass_probe_child.capture_data(BranchNodeData{});
        bypass_probe_child.end();
        model.set_bypass_probe_child(&bypass_probe_child);

        RpuKernelGraph foreign_probe_graph;
#if RHINO_LAUNCH_HAS_DEV_PROGRAM_API
        alignas(8) char dummy_instructions[64] = {};
        ::rhino_lkn::Kernel_t foreign_probe_kernel(
            dummy_instructions, sizeof(dummy_instructions),
            "schema_v12_foreign_probe");
        model.set_bypass_probe_foreign_graph(
            &foreign_probe_graph, &foreign_probe_kernel);
#endif

        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 841;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        RpuKernelGraphContractTestAccess::stage_pending_kernel_name(
            graph, "ddr2spm_multi_core_v2");
        const GraphOpStreamStamp build_scope_before =
            graph.op_stream_stamp();
        const InactiveGraphStamp foreign_build_before =
            inactive_graph_stamp(foreign_probe_graph);
        const bool build_foreign_lookup_rejected = throws_c10_containing(
            [&] {
                (void)foreign_probe_graph.get_kernel_reset(
                    KernelId::BINARY_SAMESHAPE);
            },
            "foreign Graph kernel lookup");
#if RHINO_LAUNCH_HAS_DEV_PROGRAM_API
        const bool build_foreign_enqueue_rejected = throws_c10_containing(
            [&] {
                foreign_probe_graph.enqueue(
                    foreign_probe_kernel, {1}, {0});
            },
            "foreign Graph kernel enqueue");
#else
        const bool build_foreign_enqueue_rejected = true;
#endif
        const bool build_foreign_sync_rejected = throws_c10_containing(
            [&] { foreign_probe_graph.sync_point(); },
            "foreign Graph sync_point");
#if RHINO_LAUNCH_HAS_DEV_PROGRAM_API
        const bool build_staged_transport_rejected = throws_c10_containing(
            [&] { graph.enqueue(foreign_probe_kernel, {1}, {0}); },
            "pre-staged legacy DDR/SPM transport kernel is forbidden");
#else
        const bool build_staged_transport_rejected = true;
#endif
        RpuKernelGraphContractTestAccess::clear_pending_kernel(graph);
        const GraphOpStreamStamp build_scope_after =
            graph.op_stream_stamp();
        const InactiveGraphStamp foreign_build_after =
            inactive_graph_stamp(foreign_probe_graph);
        const bool build_pre_callback_foreign_guards_ok =
            build_foreign_lookup_rejected &&
            build_foreign_enqueue_rejected && build_foreign_sync_rejected &&
            build_staged_transport_rejected &&
            build_scope_before.graph == build_scope_after.graph &&
            build_scope_before.state == build_scope_after.state &&
            build_scope_before.build_generation ==
                build_scope_after.build_generation &&
            build_scope_before.position == build_scope_after.position &&
            build_scope_before.extent == build_scope_after.extent &&
            foreign_build_before == foreign_build_after;
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        const auto groups = model.seal_active_groups(trace, graph);
        const auto members = model.seal_dense_ddr_members(groups, graph);
        const auto bindings = model.seal_dma_endpoint_bindings(
            members, graph);
        model.validate_dma_endpoint_bindings(bindings, graph);
        EXPECT(bindings.logical_endpoint_count() == 12 &&
                   bindings.dma_node_count() == 54 &&
                   bindings.fixed_dma_node_count() == 30 &&
                   bindings.mutable_dma_node_count() == 24 &&
                   bindings.legacy_dma_node_count() == 0 &&
                   graph.graph_size() == 142 &&
                   model.safe_build_probes_ok() &&
                   model.escaped_follower_rejected() &&
                   model.escaped_context_outside_callback_rejected() &&
                   model.local_context_ok() &&
                   build_pre_callback_foreign_guards_ok,
               "schema-v12 BUILD seals 3-member exact covers while raw "
               "helpers and escaped refs fail before Graph mutation");

        const uint64_t build_generation = graph.build_generation();
        const uint64_t topology_hash = graph.build_topology_hash();
        graph.begin(signature);
        const GraphOpStreamStamp replay_scope_before =
            graph.op_stream_stamp();
        const InactiveGraphStamp child_before =
            inactive_graph_stamp(bypass_probe_child);
        const InactiveGraphStamp foreign_replay_before =
            inactive_graph_stamp(foreign_probe_graph);
        const bool pre_callback_invalidate_rejected =
            throws_c10_containing(
                [&] { graph.invalidate(); },
                "Graph invalidation is forbidden while a Graph scope is "
                "active");
        RpuKernelGraph replay_nested;
        GraphSignature replay_nested_signature;
        replay_nested_signature.op_id = 0xbeef;
        const bool pre_callback_nested_rejected = throws_c10_containing(
            [&] { replay_nested.begin(replay_nested_signature); },
            "nested Graph begin is forbidden");
        const bool pre_callback_child_rejected = throws_c10_containing(
            [&] { bypass_probe_child.replay_prepared_child({}); },
            "direct child replay is forbidden");
        const bool pre_callback_foreign_lookup_rejected =
            throws_c10_containing(
                [&] {
                    (void)foreign_probe_graph.get_kernel_reset(
                        KernelId::BINARY_SAMESHAPE);
                },
                "foreign Graph kernel lookup");
#if RHINO_LAUNCH_HAS_DEV_PROGRAM_API
        const bool pre_callback_foreign_enqueue_rejected =
            throws_c10_containing(
                [&] {
                    foreign_probe_graph.enqueue(
                        foreign_probe_kernel, {1}, {0});
                },
                "foreign Graph kernel enqueue");
#else
        const bool pre_callback_foreign_enqueue_rejected = true;
#endif
        const bool pre_callback_foreign_sync_rejected =
            throws_c10_containing(
                [&] { foreign_probe_graph.sync_point(); },
                "foreign Graph sync_point");
        const GraphOpStreamStamp replay_scope_after =
            graph.op_stream_stamp();
        const InactiveGraphStamp child_after =
            inactive_graph_stamp(bypass_probe_child);
        const InactiveGraphStamp foreign_replay_after =
            inactive_graph_stamp(foreign_probe_graph);
        const bool pre_callback_guards_ok =
            pre_callback_invalidate_rejected &&
            pre_callback_nested_rejected && pre_callback_child_rejected &&
            pre_callback_foreign_lookup_rejected &&
            pre_callback_foreign_enqueue_rejected &&
            pre_callback_foreign_sync_rejected &&
            replay_scope_before.graph == replay_scope_after.graph &&
            replay_scope_before.state == replay_scope_after.state &&
            replay_scope_before.build_generation ==
                replay_scope_after.build_generation &&
            replay_scope_before.position == replay_scope_after.position &&
            replay_scope_before.extent == replay_scope_after.extent &&
            child_before == child_after &&
            foreign_replay_before == foreign_replay_after;
        model.arm_dma_endpoint_replay(bindings, graph);
        model.replay_runtime_members(bindings);
        graph.end();
        model.validate_dma_endpoint_bindings(bindings, graph);
        EXPECT(graph.build_generation() == build_generation &&
                   graph.build_topology_hash() == topology_hash &&
                   model.safe_replay_probes_ok() &&
                   model.escaped_replay_rejected() &&
                   pre_callback_guards_ok,
               "schema-v12 ordinary REPLAY re-enters frozen FMB callback "
               "scopes and preserves cursor/topology identity");

        graph.begin(signature);
        model.arm_dma_endpoint_replay(bindings, graph);
        graph.skip_op_stream_for_fast_replay();
        graph.end();
        model.validate_dma_endpoint_bindings(bindings, graph);
        EXPECT(graph.build_generation() == build_generation &&
                   graph.build_topology_hash() == topology_hash,
               "schema-v12 fast REPLAY needs only the sealed owner arm and "
               "does not recreate runtime refs");
    }

    {
        ResolvedTraversalRuntimeMemberContractModel model;
        model.set_omit_first_egress(true);
        model.prepare_cpu(16, 48);
        const auto profile = model.resolve(
            resolved_three_chunk_request(42));
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 842;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        const bool exact_cover_rejected = throws_c10_containing(
            [&] { model.drive_trace_cpu(profile); },
            "incomplete member/role exact cover");
        const GraphOpStreamStamp poisoned_before = graph.op_stream_stamp();
        const bool poisoned_invalidate_rejected = throws_c10_containing(
            [&] { graph.invalidate(); },
            "Graph invalidation is forbidden while a Graph scope is active");
        const GraphOpStreamStamp poisoned_after = graph.op_stream_stamp();
        const bool poisoned_end_rejected = throws_c10_containing(
            [&] { graph.end(); },
            "scope is poisoned");
        graph.abort();
        EXPECT(exact_cover_rejected && poisoned_invalidate_rejected &&
                   poisoned_before.graph == poisoned_after.graph &&
                   poisoned_before.state == poisoned_after.state &&
                   poisoned_before.build_generation ==
                       poisoned_after.build_generation &&
                   poisoned_before.position == poisoned_after.position &&
                   poisoned_before.extent == poisoned_after.extent &&
                   poisoned_end_rejected,
               "schema-v12 incomplete callback cover poisons the Graph and "
               "fails before BUILD execution");
    }

    {
        ResolvedTraversalRuntimeMemberContractModel model;
        model.set_inject_opaque_after_cover(true);
        model.prepare_cpu(16, 48);
        const auto profile = model.resolve(
            resolved_three_chunk_request(44));
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 844;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        const bool opaque_rejected = throws_c10_containing(
            [&] { model.drive_trace_cpu(profile); },
            "opaque ChildGraph/HostCallback/Tier3Oneshot node");
        const bool incomplete_trace_rejected = throws_c10_containing(
            [&] { graph.end(); },
            "BUILD trace is incomplete");
        graph.abort();
        EXPECT(opaque_rejected && incomplete_trace_rejected,
               "schema-v12 post-callback trace failure cannot execute before "
               "the FMB seal");
    }

    {
        ResolvedTraversalRuntimeMemberContractModel model;
        model.prepare_cpu(16, 48);
        const auto profile = model.resolve(
            resolved_three_chunk_request(43));
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 843;
        signature.segment_key = profile.profile_hash();
        const auto capture = [&]() {
            graph.begin(signature);
            model.begin_trace(profile);
            model.drive_trace_cpu(profile);
            graph.end();
            const auto trace = model.seal_trace(profile, graph);
            const auto groups = model.seal_active_groups(trace, graph);
            const auto members = model.seal_dense_ddr_members(groups, graph);
            return model.seal_dma_endpoint_bindings(members, graph);
        };
        const auto bindings_a = capture();
        graph.invalidate();
        model.set_alternate_topology(true);
        const auto bindings_b = capture();
        const bool old_a_stale = throws_c10_containing(
            [&] { model.validate_dma_endpoint_bindings(bindings_a, graph); },
            "Graph identity is stale");
        graph.invalidate();
        model.set_alternate_topology(false);
        const auto bindings_a_again = capture();
        model.validate_dma_endpoint_bindings(bindings_a_again, graph);
        EXPECT(old_a_stale &&
                   bindings_a.binding_hash() != bindings_b.binding_hash() &&
                   bindings_a.binding_hash() ==
                       bindings_a_again.binding_hash() &&
                   bindings_a.graph_build_generation() !=
                       bindings_a_again.graph_build_generation() &&
                   throws_c10_containing(
                       [&] {
                           model.validate_dma_endpoint_bindings(
                               bindings_b, graph);
                       },
                       "Graph identity is stale"),
               "schema-v12 A/B/A keeps semantic identity deterministic and "
               "rejects stale runtime bindings");
    }

    // Schema-v13 derives the fixed SPM side from the sealed allocation table.
    // The caller still selects only a buffer name; shared/packed kind, slice,
    // core cover, and opaque peer identity are framework-owned.
    static_assert(!std::is_default_constructible_v<
                  v3::SpmFmbSealedSpmPeerBindings>);
    static_assert(!std::is_aggregate_v<
                  v3::SpmFmbSealedSpmPeerBindings>);
    static_assert(!std::is_constructible_v<
                  v3::SpmFmbSealedSpmPeerBindings, uint64_t>);
    static_assert(!std::is_convertible_v<
                  v3::SpmFmbSealedSpmPeerBindings,
                  v3::SpmPipelineLease>);
    static_assert(!std::is_constructible_v<
                  v3::SpmPipelineLease,
                  v3::SpmFmbSealedSpmPeerBindings>);
    static_assert(!std::is_constructible_v<
                  v3::SpmFmbSealedSpmPeerBindings,
                  v3::SpmPipelineLease>);
    EXPECT(true,
           "schema-v13 typed SPM peer token exposes no raw peer/range or "
           "physical lease conversion");

    {
        ResolvedTraversalTypedSpmPeerContractModel v12_same_layout;
        v12_same_layout.set_peer_capability_enabled(false);
        v12_same_layout.prepare_cpu(16, 48);
        const auto request = resolved_three_chunk_request(45);
        const auto v12_profile = v12_same_layout.resolve(request);
        RpuKernelGraph v12_graph;
        GraphSignature v12_signature;
        v12_signature.op_id = 845;
        v12_signature.segment_key = v12_profile.profile_hash();
        v12_graph.begin(v12_signature);
        v12_same_layout.begin_trace(v12_profile);
        v12_same_layout.drive_trace_cpu(v12_profile);
        v12_graph.end();
        const auto v12_trace = v12_same_layout.seal_trace(
            v12_profile, v12_graph);
        const auto v12_groups = v12_same_layout.seal_active_groups(
            v12_trace, v12_graph);
        const auto v12_members = v12_same_layout.seal_dense_ddr_members(
            v12_groups, v12_graph);
        const auto v12_dma = v12_same_layout.seal_dma_endpoint_bindings(
            v12_members, v12_graph);
        const bool unsealed_peer_rejected = throws_c10_containing(
            [&] {
                (void)v12_same_layout.seal_spm_peer_bindings(
                    v12_dma, v12_graph);
            },
            "not one CPU-dry schema-v13 contract");

        ResolvedTraversalTypedSpmPeerContractModel v13_same_layout;
        v13_same_layout.prepare_cpu(16, 48);
        const auto v13_profile = v13_same_layout.resolve(request);
        at::Tensor empty_hidden;
        std::vector<at::Tensor> empty_k;
        std::vector<at::Tensor> empty_v;
        const bool production_rejected = throws_c10_containing(
            [&] {
                (void)v13_same_layout.run_all_layers(
                    empty_hidden, empty_k, empty_v, std::nullopt,
                    /*position=*/0, /*is_causal=*/true);
            },
            "does not authorize production run_all_layers");
        EXPECT(unsealed_peer_rejected &&
                   v12_profile.profile_hash() != v13_profile.profile_hash() &&
                   production_rejected,
               "schema-v13 defaults fail closed, appends independent "
               "domain-22 identity, and remains CPU-dry in production");
    }

    for (const auto pattern : {
             ResolvedTraversalTypedSpmPeerContractModel::LayoutPattern::
                 Undersized,
             ResolvedTraversalTypedSpmPeerContractModel::LayoutPattern::
                 Unsupported}) {
        ResolvedTraversalTypedSpmPeerContractModel model;
        model.set_layout_pattern(pattern);
        model.prepare_cpu(16, 48);
        const auto profile = model.resolve(
            resolved_three_chunk_request(
                pattern == ResolvedTraversalTypedSpmPeerContractModel::
                               LayoutPattern::Undersized
                    ? 46 : 47));
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 845;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        const bool rejected = throws_c10_containing(
            [&] { model.drive_trace_cpu(profile); },
            "capacity is undersized, unsupported, or ambiguous");
        const bool no_dma_recorded = graph.graph_size() == 2;
        const bool execution_blocked = throws_c10_containing(
            [&] { graph.end(); }, "scope is poisoned");
        graph.abort();
        EXPECT(rejected && no_dma_recorded && execution_blocked,
               "schema-v13 rejects undersized/noncanonical capacity before "
               "the first DMA node and prevents BUILD execution");
    }

    {
        ResolvedTraversalTypedSpmPeerContractModel model;
        model.set_hidden_size_for_contract(15);
        model.set_layout_pattern(
            ResolvedTraversalTypedSpmPeerContractModel::LayoutPattern::
                UnalignedSlack);
        model.prepare_cpu(16, 48);
        const auto profile = model.resolve(
            resolved_three_chunk_request(51));
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 845;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        const bool exact_logical_rejected = throws_c10_containing(
            [&] { model.drive_trace_cpu(profile); },
            "logical capacity is undersized, unsupported, or ambiguous");
        const bool no_dma_recorded = graph.graph_size() == 2;
        const bool execution_blocked = throws_c10_containing(
            [&] { graph.end(); }, "scope is poisoned");
        graph.abort();
        EXPECT(exact_logical_rejected && no_dma_recorded &&
                   execution_blocked,
               "schema-v13 rejects 496 logical bytes in the same aligned "
               "bucket as an exact 480-byte shared peer");
    }

    {
        using PeerModel = ResolvedTraversalTypedSpmPeerContractModel;
        PeerModel model;
        model.prepare_cpu(16, 48);
        const auto request = resolved_three_chunk_request(48);
        const auto profile = model.resolve(request);
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 846;
        signature.segment_key = profile.profile_hash();

        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        const auto groups = model.seal_active_groups(trace, graph);
        const auto members = model.seal_dense_ddr_members(groups, graph);
        const auto dma_bindings = model.seal_dma_endpoint_bindings(
            members, graph);
        const auto peers_a = model.seal_spm_peer_bindings(
            dma_bindings, graph);
        model.validate_spm_peer_bindings(peers_a, graph);
        const uint64_t generation_a = graph.build_generation();
        const uint64_t topology_a = graph.build_topology_hash();
        const auto dma_snapshot = debug_dma_snapshots(graph);
        bool address_oracle_ok = dma_snapshot.size() == 54;
        const uint64_t core0_spm_base = managed_spm_root_address(0);
        const uint64_t shared_offset = address_oracle_ok
            ? dma_snapshot[0].dst_addr - core0_spm_base : 0;
        const uint64_t packed_offset = address_oracle_ok
            ? dma_snapshot[8].src_addr - core0_spm_base : 0;
        size_t dma_index = 0;
        for (int layer = 0; layer < 2 && address_oracle_ok; ++layer) {
            const bool ingress_shared = layer == 0;
            const bool egress_shared = layer != 0;
            for (size_t member = 0; member < 3; ++member) {
                const uint64_t ingress_offset =
                    (ingress_shared ? shared_offset : packed_offset) +
                    (ingress_shared ? 0 : member * 512);
                const uint64_t egress_offset =
                    (egress_shared ? shared_offset : packed_offset) +
                    (egress_shared ? 0 : member * 512);
                for (uint8_t core = 0; core < 8; ++core) {
                    const auto& dma = dma_snapshot.at(dma_index++);
                    const uint64_t expected_spm =
                        managed_spm_root_address(core) +
                        ingress_offset;
                    address_oracle_ok = address_oracle_ok &&
                        dma.dst_addr == expected_spm &&
                        dma.bytes == 512 && dma.channel == core &&
                        dma.semantic_endpoint && dma.canonical_member &&
                        dma.typed_spm_peer &&
                        dma.variant == (layer == 0 ? "msrc" : "fixed");
                }
                const auto& egress = dma_snapshot.at(dma_index++);
                const uint64_t expected_spm = core0_spm_base +
                    egress_offset;
                address_oracle_ok = address_oracle_ok &&
                    egress.src_addr == expected_spm &&
                    egress.bytes == 512 && egress.channel == 0 &&
                    egress.semantic_endpoint && egress.canonical_member &&
                    egress.typed_spm_peer &&
                    egress.variant == "fixed";
            }
        }
        address_oracle_ok = address_oracle_ok && dma_index == 54;
        EXPECT(peers_a.logical_peer_count() == 12 &&
                   peers_a.binding_hash() != 0 && peers_a.cpu_dry() &&
                   peers_a.shared_stage_peer_count() == 6 &&
                   peers_a.packed_global_peer_count() == 6 &&
                   peers_a.source_dma_binding_hash() ==
                       dma_bindings.binding_hash() &&
                   dma_bindings.dma_node_count() == 54 &&
                   graph.graph_size() == 142 && address_oracle_ok,
               "schema-v13 seals the mixed shared-stage/packed-global "
               "12-peer/54-DMA cover with independent +0/+512/+1024 SPM "
               "address evidence");

        graph.begin(signature);
        const GraphOpStreamStamp legacy_before = graph.op_stream_stamp();
        const bool old_arm_rejected = throws_c10_containing(
            [&] { model.arm_dma_endpoint_replay(dma_bindings, graph); },
            "legacy schema-v11/v12 arm cannot authorize");
        const bool old_driver_rejected = throws_c10_containing(
            [&] { model.replay_runtime_members(dma_bindings); },
            "legacy schema-v12 driver cannot authorize");
        const GraphOpStreamStamp legacy_after = graph.op_stream_stamp();
        model.set_alias_replay_probe(true);
        model.arm_spm_peer_replay(peers_a, graph);
        model.replay_spm_peers(peers_a);
        graph.end();
        model.validate_spm_peer_bindings(peers_a, graph);
        EXPECT(old_arm_rejected && old_driver_rejected &&
                   legacy_before.position == legacy_after.position &&
                   legacy_before.extent == legacy_after.extent &&
                   model.alias_probe_ok() &&
                   graph.build_generation() == generation_a &&
                   graph.build_topology_hash() == topology_a,
               "schema-v13 ordinary REPLAY rejects legacy authority and a "
               "same-address alias before cursor mutation, then consumes "
               "the canonical peer token");

        graph.begin(signature);
        const GraphOpStreamStamp fast_before = graph.op_stream_stamp();
        model.set_layout_pattern(PeerModel::LayoutPattern::MixedB);
        const bool stale_policy_arm_rejected = throws_c10_containing(
            [&] { model.arm_spm_peer_replay(peers_a, graph); },
            "identity is stale");
        const GraphOpStreamStamp policy_after = graph.op_stream_stamp();
        model.set_layout_pattern(PeerModel::LayoutPattern::MixedA);
        model.set_hidden_size_for_contract(32);
        const bool stale_shape_arm_rejected = throws_c10_containing(
            [&] { model.arm_spm_peer_replay(peers_a, graph); },
            "identity is stale");
        const GraphOpStreamStamp shape_after = graph.op_stream_stamp();
        model.set_hidden_size_for_contract(16);
        const bool unarmed_fast_rejected = throws_c10_containing(
            [&] { graph.skip_op_stream_for_fast_replay(); },
            "requires an armed schema-v13 token");
        const bool unarmed_partial_rejected = throws_c10_containing(
            [&] { graph.skip_layer_body_for_fast_replay(); },
            "requires an armed schema-v13 token");
        const bool unarmed_patch_rejected = throws_c10_containing(
            [&] { graph.skip_op_stream_with_patches({}, {}); },
            "requires an armed schema-v13 token");
        const GraphOpStreamStamp fast_after = graph.op_stream_stamp();
        model.arm_spm_peer_replay(peers_a, graph);
        graph.skip_op_stream_for_fast_replay();
        graph.end();
        EXPECT(stale_policy_arm_rejected && stale_shape_arm_rejected &&
                   unarmed_fast_rejected && unarmed_partial_rejected &&
                   unarmed_patch_rejected &&
                   fast_before.position == policy_after.position &&
                   fast_before.extent == policy_after.extent &&
                   fast_before.position == shape_after.position &&
                   fast_before.extent == shape_after.extent &&
                   fast_before.position == fast_after.position &&
                   fast_before.extent == fast_after.extent &&
                   graph.build_generation() == generation_a &&
                   graph.build_topology_hash() == topology_a,
               "schema-v13 fast REPLAY requires the independent peer arm "
               "before any skip mutation");

        const auto capture = [&](const auto& capture_profile) {
            signature.segment_key = capture_profile.profile_hash();
            graph.begin(signature);
            model.begin_trace(capture_profile);
            model.drive_trace_cpu(capture_profile);
            graph.end();
            const auto next_trace = model.seal_trace(
                capture_profile, graph);
            const auto next_groups = model.seal_active_groups(
                next_trace, graph);
            const auto next_members = model.seal_dense_ddr_members(
                next_groups, graph);
            const auto next_dma = model.seal_dma_endpoint_bindings(
                next_members, graph);
            return model.seal_spm_peer_bindings(next_dma, graph);
        };

        model.set_alias_replay_probe(false);
        graph.invalidate();
        model.set_layout_pattern(PeerModel::LayoutPattern::MixedB);
        const auto profile_b = model.resolve(request);
        const auto peers_b = capture(profile_b);
        const bool old_a_stale = throws_c10(
            [&] { model.validate_spm_peer_bindings(peers_a, graph); });
        graph.begin(signature);
        const GraphOpStreamStamp stale_arm_before =
            graph.op_stream_stamp();
        const bool old_a_arm_stale = throws_c10_containing(
            [&] { model.arm_spm_peer_replay(peers_a, graph); },
            "identity is stale");
        const GraphOpStreamStamp stale_arm_after =
            graph.op_stream_stamp();
        model.arm_spm_peer_replay(peers_b, graph);
        graph.skip_op_stream_for_fast_replay();
        graph.end();
        graph.invalidate();
        model.set_layout_pattern(PeerModel::LayoutPattern::MixedA);
        const auto profile_a_again = model.resolve(request);
        const auto peers_a_again = capture(profile_a_again);
        model.validate_spm_peer_bindings(peers_a_again, graph);
        EXPECT(old_a_stale && old_a_arm_stale &&
                   stale_arm_before.position == stale_arm_after.position &&
                   stale_arm_before.extent == stale_arm_after.extent &&
                   peers_a.binding_hash() != peers_b.binding_hash() &&
                   peers_a.binding_hash() == peers_a_again.binding_hash() &&
                   profile.profile_hash() != profile_b.profile_hash() &&
                   profile.profile_hash() ==
                       profile_a_again.profile_hash() &&
                   generation_a != graph.build_generation() &&
                   throws_c10(
                       [&] {
                           model.validate_spm_peer_bindings(peers_b, graph);
                       }),
               "schema-v13 A/B/A keeps peer identity deterministic while "
               "old owner/Graph generations remain stale");
    }

    for (const auto item : {
             std::pair{
                 ResolvedTraversalTypedSpmPeerContractModel::LayoutPattern::
                     AllShared,
                 true},
             std::pair{
                 ResolvedTraversalTypedSpmPeerContractModel::LayoutPattern::
                     AllPacked,
                 false},
             std::pair{
                 ResolvedTraversalTypedSpmPeerContractModel::LayoutPattern::
                     UnalignedShared,
                 true},
             std::pair{
                 ResolvedTraversalTypedSpmPeerContractModel::LayoutPattern::
                     UnalignedPacked,
                 false}}) {
        ResolvedTraversalTypedSpmPeerContractModel model;
        model.set_layout_pattern(item.first);
        if (item.first ==
                ResolvedTraversalTypedSpmPeerContractModel::LayoutPattern::
                    UnalignedShared ||
            item.first ==
                ResolvedTraversalTypedSpmPeerContractModel::LayoutPattern::
                    UnalignedPacked) {
            model.set_hidden_size_for_contract(15);
        }
        model.prepare_cpu(16, 48);
        const auto profile = model.resolve(
            resolved_three_chunk_request(item.second ? 49 : 50));
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 847;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        const auto groups = model.seal_active_groups(trace, graph);
        const auto members = model.seal_dense_ddr_members(groups, graph);
        const auto dma = model.seal_dma_endpoint_bindings(members, graph);
        const auto peers = model.seal_spm_peer_bindings(dma, graph);
        EXPECT(peers.logical_peer_count() == 12 &&
                   peers.shared_stage_peer_count() ==
                       (item.second ? 12 : 0) &&
                   peers.packed_global_peer_count() ==
                       (item.second ? 0 : 12),
               "schema-v13 independently distinguishes all-shared from "
               "all-packed peer populations");
    }

    {
        ResolvedTraversalTypedSpmPeerContractModel model;
        model.set_layout_pattern(
            ResolvedTraversalTypedSpmPeerContractModel::LayoutPattern::
                TailPacked);
        model.prepare_cpu(16, 40);
        const auto profile = model.resolve(
            resolved_tail_chunk_request(52));
        RpuKernelGraph graph;
        GraphSignature signature;
        signature.op_id = 848;
        signature.segment_key = profile.profile_hash();
        graph.begin(signature);
        model.begin_trace(profile);
        model.drive_trace_cpu(profile);
        graph.end();
        const auto trace = model.seal_trace(profile, graph);
        const auto groups = model.seal_active_groups(trace, graph);
        const auto members = model.seal_dense_ddr_members(groups, graph);
        const auto dma = model.seal_dma_endpoint_bindings(members, graph);
        const auto peers = model.seal_spm_peer_bindings(dma, graph);
        const auto snapshot = debug_dma_snapshots(graph);
        bool tail_oracle_ok = snapshot.size() == 54;
        const uint64_t core0_spm_base = managed_spm_root_address(0);
        const uint64_t ingress_base = tail_oracle_ok
            ? snapshot[0].dst_addr - core0_spm_base : 0;
        const uint64_t egress_base = tail_oracle_ok
            ? snapshot[8].src_addr - core0_spm_base : 1;
        tail_oracle_ok = tail_oracle_ok && ingress_base == egress_base;
        for (int layer = 0; layer < 2 && tail_oracle_ok; ++layer) {
            for (size_t member = 0; member < 3; ++member) {
                const size_t bytes = member == 2 ? 256 : 512;
                const uint64_t slice_begin = member * 512;
                const size_t first = static_cast<size_t>(layer) * 27 +
                    member * 9;
                for (uint8_t core = 0; core < 8; ++core) {
                    const auto& ingress = snapshot[first + core];
                    tail_oracle_ok = tail_oracle_ok &&
                        ingress.bytes == bytes &&
                        ingress.dst_addr ==
                            managed_spm_root_address(core) +
                            ingress_base + slice_begin;
                }
                const auto& egress = snapshot[first + 8];
                tail_oracle_ok = tail_oracle_ok &&
                    egress.bytes == bytes &&
                    egress.src_addr == core0_spm_base +
                        egress_base + slice_begin;
            }
        }
        EXPECT(peers.logical_peer_count() == 12 &&
                   peers.shared_stage_peer_count() == 0 &&
                   peers.packed_global_peer_count() == 12 &&
                   tail_oracle_ok,
               "schema-v13 packed peers exact-cover a 512/512/256-byte "
               "tail at +0/+512/+1024 without full-chunk padding");
    }

    const auto phase_slice = [](uint32_t arena,
                                uint32_t offset,
                                size_t bytes,
                                uint32_t first,
                                uint32_t last) {
        return SpmPhaseSlice(SpmScratchId(arena), offset, bytes,
                             SpmPipelineEvent(first),
                             SpmPipelineEvent(last));
    };
    constexpr size_t kPhaseExtent = 8303616;
    constexpr size_t kRowBytes2048 = 4096;
    constexpr size_t kSourceChunkBytes = 144 * kRowBytes2048;
    constexpr size_t kConsumerChunkBytes = 320 * kRowBytes2048;
    constexpr size_t kVisionDeadBytes = 2200064;
    constexpr uint32_t kSourceOffset = 6500096;
    constexpr uint32_t kPayload0Offset = 1310720;
    constexpr uint32_t kPayload1Offset = 2306048;

    const SpmPhaseSlice vision_dead =
        phase_slice(100, 0, kVisionDeadBytes, 0, 1);
    const SpmPhaseSlice source0 =
        phase_slice(100, kSourceOffset, kSourceChunkBytes, 0, 1);
    const SpmPhaseSlice source1 =
        phase_slice(100, kSourceOffset, kSourceChunkBytes, 2, 3);
    const SpmPhaseSlice source2 =
        phase_slice(100, kSourceOffset, kSourceChunkBytes, 4, 5);
    const SpmPhaseSlice consumer0 =
        phase_slice(101, 0, kConsumerChunkBytes, 8, 9);
    const SpmPhaseSlice consumer1 =
        phase_slice(101, 0, kConsumerChunkBytes, 10, 11);
    const SpmPhaseSlice payload0 =
        phase_slice(101, kPayload0Offset, 243 * kRowBytes2048, 2, 7);
    const SpmPhaseSlice payload1 =
        phase_slice(101, kPayload1Offset, 189 * kRowBytes2048, 4, 9);

    const std::vector<SpmComponentPhaseManifest> phase_manifests = {
        {SpmScratchId(100), 0, kPhaseExtent, 0x1001,
         {vision_dead, source0, source1, source2}},
        {SpmScratchId(101), 0, kPhaseExtent, 0x1002,
         {consumer0, consumer1}},
    };
    const std::vector<SpmSourceChunkRef> production_sources = {
        {SpmChunkKey(SpmPortId(40), 0), dense_fp16(144, 2048), source0},
        {SpmChunkKey(SpmPortId(40), 1), dense_fp16(144, 2048), source1},
        {SpmChunkKey(SpmPortId(40), 2), dense_fp16(144, 2048), source2},
    };
    const std::vector<SpmBorrowedConsumerChunkRef> production_consumers = {
        {SpmChunkKey(SpmPortId(41), 0), dense_fp16(320, 2048), consumer0},
        {SpmChunkKey(SpmPortId(41), 1), dense_fp16(320, 2048), consumer1},
    };
    const std::vector<SpmRetainedPayloadChunkRef> production_payloads = {
        {SpmChunkKey(SpmPortId(41), 0), dense_fp16(243, 2048), payload0},
        {SpmChunkKey(SpmPortId(41), 1), dense_fp16(189, 2048), payload1},
    };
    std::vector<int32_t> production_permutation(432);
    for (int32_t row = 0; row < 432; ++row) {
        production_permutation[static_cast<size_t>(row)] = row;
    }
    const std::vector<SpmDstRowRun> production_runs = {
        {70, 144}, {221, 144}, {372, 144},
    };
    const auto bind_production_route =
        [&](const std::vector<SpmSourceChunkRef>& sources,
            const std::vector<SpmRetainedPayloadChunkRef>& payloads,
            uint64_t policy_version) {
            return SpmChunkedPermutationRoute::bind(
                SpmPortId(40), SpmPortId(41), sources,
                production_consumers, payloads, production_permutation,
                production_runs, policy_version);
        };
    const SpmChunkedPermutationRoute production_route =
        bind_production_route(production_sources, production_payloads, 7);
    const SpmPipelinePlan phase_plan =
        SpmPipelinePlan::compile_phase_overlay(
            phase_manifests, {production_route});

    EXPECT(phase_plan.phase_manifest_count() == 2 &&
               phase_plan.contains_phase_arena(SpmScratchId(100)) &&
               phase_plan.contains_phase_arena(SpmScratchId(101)) &&
               phase_plan.region_count() == 0,
           "phase overlay seals fixed manifests without coarse regions");
    EXPECT(phase_plan.peak_bytes() == kPhaseExtent &&
               phase_plan.total_bytes() == kPhaseExtent,
           "phase-overlay peak is max arena base plus extent, not their sum");
    EXPECT(throws_c10_containing([&] {
               auto physical =
                   SpmPipelineLease::acquire_physical(phase_plan);
           }, "physical admission requires an opaque FusedModelBase-sealed exporter"),
           "public synthetic phase plan fails before physical allocator admission");
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile_phase_overlay(
                   phase_manifests, {production_route}, 1);
           }),
           "synthetic zero-headroom geometry rejects one aligned persistent "
           "byte; real FMB manifests remain unproven");
    EXPECT(phase_plan.chunked_permutation_hash(
               SpmPortId(40), SpmPortId(41)) == production_route.hash(),
           "schema-v5 plan exposes its exact chunked route hash");
    EXPECT(production_route.payload_chunks()[0].spec.rows == 243 &&
               production_route.payload_chunks()[1].spec.rows == 189,
           "retained payload stores exactly 432 image rows split 243 plus 189");
    EXPECT(payload0.offset >= kConsumerChunkBytes &&
               payload0.offset + payload0.bytes <= payload1.offset,
           "retained payload chunks are physically disjoint from residual1 and each other");
    auto empty_live_manifest = phase_manifests;
    empty_live_manifest[0].live_ranges.clear();
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile_phase_overlay(
                   empty_live_manifest, {production_route});
           }),
           "component manifest must provide non-empty complete live ranges");

    const auto& payload_to_consumer =
        production_route.payload_consumer_runs();
    EXPECT(payload_to_consumer.size() == 4 &&
               payload_to_consumer[0].consumer_chunk.ordinal() == 0 &&
               payload_to_consumer[0].transfer_event.value() == 8 &&
               payload_to_consumer[0].consumer_row_begin == 70 &&
               payload_to_consumer[0].row_count == 144 &&
               payload_to_consumer[1].consumer_chunk.ordinal() == 0 &&
               payload_to_consumer[1].transfer_event.value() == 8 &&
               payload_to_consumer[1].consumer_row_begin == 221 &&
               payload_to_consumer[1].row_count == 99 &&
               payload_to_consumer[2].consumer_chunk.ordinal() == 1 &&
               payload_to_consumer[2].transfer_event.value() == 10 &&
               payload_to_consumer[2].consumer_row_begin == 0 &&
               payload_to_consumer[2].row_count == 45 &&
               payload_to_consumer[3].consumer_chunk.ordinal() == 1 &&
               payload_to_consumer[3].transfer_event.value() == 10 &&
               payload_to_consumer[3].consumer_row_begin == 52 &&
               payload_to_consumer[3].row_count == 144,
           "global 144-row run crossing C320 is split into 99 and 45 rows");
    auto reversed_consumers = production_consumers;
    auto reversed_payloads = production_payloads;
    reversed_consumers[0].backing =
        phase_slice(101, 0, kConsumerChunkBytes, 10, 11);
    reversed_consumers[1].backing =
        phase_slice(101, 0, kConsumerChunkBytes, 8, 9);
    reversed_payloads[0].backing = phase_slice(
        101, kPayload0Offset, 243 * kRowBytes2048, 2, 9);
    reversed_payloads[1].backing = phase_slice(
        101, kPayload1Offset, 189 * kRowBytes2048, 4, 7);
    auto reversed_manifests = phase_manifests;
    reversed_manifests[1].live_ranges[0] =
        reversed_consumers[0].backing;
    reversed_manifests[1].live_ranges[1] =
        reversed_consumers[1].backing;
    const auto reversed_consumer_route = SpmChunkedPermutationRoute::bind(
        SpmPortId(40), SpmPortId(41), production_sources,
        reversed_consumers, reversed_payloads, production_permutation,
        production_runs, 7);
    const auto reversed_consumer_plan =
        SpmPipelinePlan::compile_phase_overlay(
            reversed_manifests, {reversed_consumer_route});
    (void)reversed_consumer_plan;
    const auto& reversed_consumer_runs =
        reversed_consumer_route.payload_consumer_runs();
    bool consumer_transfer_events_monotonic = true;
    for (size_t i = 1; i < reversed_consumer_runs.size(); ++i) {
        consumer_transfer_events_monotonic =
            consumer_transfer_events_monotonic &&
            reversed_consumer_runs[i - 1].transfer_event.value() <=
                reversed_consumer_runs[i].transfer_event.value();
    }
    EXPECT(reversed_consumer_runs.size() == 4 &&
               consumer_transfer_events_monotonic &&
               reversed_consumer_runs.front().consumer_chunk.ordinal() == 1 &&
               reversed_consumer_runs.front().transfer_event.value() == 8 &&
               reversed_consumer_runs.back().consumer_chunk.ordinal() == 0 &&
               reversed_consumer_runs.back().transfer_event.value() == 10,
           "payload scatter follows event8 then event10 rather than consumer ordinal");
    const auto& source_to_payload = production_route.source_payload_runs();
    EXPECT(source_to_payload.size() == 4 &&
               source_to_payload[0].source_chunk.ordinal() == 0 &&
               source_to_payload[0].transfer_event.value() == 2 &&
               source_to_payload[0].row_count == 144 &&
               source_to_payload[1].source_chunk.ordinal() == 1 &&
               source_to_payload[1].transfer_event.value() == 4 &&
               source_to_payload[1].row_count == 99 &&
               source_to_payload[2].source_chunk.ordinal() == 1 &&
               source_to_payload[2].transfer_event.value() == 4 &&
               source_to_payload[2].source_row_begin == 99 &&
               source_to_payload[2].row_count == 45 &&
               source_to_payload[3].source_chunk.ordinal() == 2 &&
               source_to_payload[3].transfer_event.value() == 6 &&
               source_to_payload[3].row_count == 144,
           "source lowering preserves source chunk boundaries across payload split");

    // Schema-v5 route v2 decouples retained-payload cardinality from Text
    // consumer chunks and seals RunAdd as a semantic operation.  Two
    // sequential producers feed one consumer through two independent payloads;
    // runtime lowering will use the existing eltwise ADD launcher, not a new
    // compute kernel.
    {
        const SpmPhaseSlice split_source0 =
            phase_slice(102, 0, 256, 0, 0);
        const SpmPhaseSlice split_source1 =
            phase_slice(102, 0, 256, 2, 2);
        const SpmPhaseSlice split_consumer =
            phase_slice(103, 0, 512, 10, 11);
        const SpmPhaseSlice split_payload0 =
            phase_slice(103, 512, 256, 1, 9);
        const SpmPhaseSlice split_payload1 =
            phase_slice(103, 768, 256, 3, 9);
        const std::vector<SpmComponentPhaseManifest> split_manifests = {
            {SpmScratchId(102), 0, 2048, 0x1021,
             {split_source0, split_source1}},
            {SpmScratchId(103), 0, 2048, 0x1031,
             {split_consumer}},
        };
        const std::vector<SpmSourceChunkRef> split_sources = {
            {SpmChunkKey(SpmPortId(70), 0), dense_fp16(2, 64),
             split_source0},
            {SpmChunkKey(SpmPortId(70), 1), dense_fp16(2, 64),
             split_source1},
        };
        const std::vector<SpmBorrowedConsumerChunkRef> split_consumers = {
            {SpmChunkKey(SpmPortId(71), 0), dense_fp16(4, 64),
             split_consumer},
        };
        const std::vector<SpmRetainedPayloadChunkRef> split_payloads = {
            {SpmChunkKey(SpmPortId(71), 0), dense_fp16(2, 64),
             split_payload0},
            {SpmChunkKey(SpmPortId(71), 1), dense_fp16(2, 64),
             split_payload1},
        };
        const auto split_run_add = SpmChunkedPermutationRoute::bind(
            SpmPortId(70), SpmPortId(71), split_sources, split_consumers,
            split_payloads, {0, 1, 2, 3}, {{0, 4}},
            /*policy_version=*/8, SpmConsumerTransform::RunAdd,
            SpmPayloadPartition::SourceConsumerSplit);
        const auto split_plan = SpmPipelinePlan::compile_phase_overlay(
            split_manifests, {split_run_add});
        EXPECT(
            split_run_add.route_schema_version() == 2 &&
                split_run_add.consumer_transform() ==
                    SpmConsumerTransform::RunAdd &&
                split_run_add.payload_partition() ==
                    SpmPayloadPartition::SourceConsumerSplit,
            "route v2 seals RunAdd and source-consumer partition identity");
        EXPECT(split_run_add.payload_chunks().size() == 2 &&
                   split_run_add.source_payload_runs().size() == 2,
               "route v2 preserves two independently produced payloads");
        EXPECT(
            split_run_add.payload_consumer_runs().size() == 2 &&
                split_run_add.payload_consumer_runs()[0]
                        .consumer_chunk.ordinal() == 0 &&
                split_run_add.payload_consumer_runs()[1]
                        .consumer_chunk.ordinal() == 0,
            "route v2 lowers two retained payloads into one RunAdd consumer");
        {
            auto split_lease = SpmPipelineLease::acquire(split_plan, 90);
            const auto checked = split_lease.chunked_permutation(
                SpmPortId(70), SpmPortId(71));
            EXPECT(
                checked.route_schema_version() == 2 &&
                    checked.consumer_transform() ==
                        SpmConsumerTransform::RunAdd &&
                    checked.payload_partition() ==
                        SpmPayloadPartition::SourceConsumerSplit &&
                    checked.payload_chunks().size() == 2,
                "checked route view retains RunAdd and split-payload identity");
        }
        EXPECT(
            throws_c10([&] {
                SpmChunkedPermutationRoute::bind(
                    SpmPortId(70), SpmPortId(71), split_sources,
                    split_consumers,
                    std::vector<SpmRetainedPayloadChunkRef>{split_payloads[0]},
                    {0, 1, 2, 3}, {{0, 4}}, 8,
                    SpmConsumerTransform::RunAdd,
                    SpmPayloadPartition::SourceConsumerSplit);
            }),
            "split-payload route rejects a missing source/consumer payload");
    }

    // SourceConsumerSplit payload ordinals are source/consumer-pair ordinals,
    // not consumer ordinals.  Two payloads feed C0 at event 10 and the third
    // feeds C1 at event 14; lowering must read each payload's own lifetime.
    {
        const SpmPhaseSlice source0 = phase_slice(106, 0, 256, 0, 0);
        const SpmPhaseSlice source1 = phase_slice(106, 0, 256, 2, 2);
        const SpmPhaseSlice source2 = phase_slice(106, 0, 256, 4, 4);
        const SpmPhaseSlice consumer0 = phase_slice(107, 0, 512, 10, 11);
        const SpmPhaseSlice consumer1 = phase_slice(107, 512, 256, 14, 15);
        const SpmPhaseSlice payload0 = phase_slice(107, 768, 256, 1, 9);
        const SpmPhaseSlice payload1 = phase_slice(107, 1024, 256, 3, 9);
        const SpmPhaseSlice payload2 = phase_slice(107, 1280, 256, 5, 13);
        const std::vector<SpmComponentPhaseManifest> manifests = {
            {SpmScratchId(106), 0, 256, 0x1061,
             {source0, source1, source2}},
            {SpmScratchId(107), 0, 1536, 0x1071,
             {consumer0, consumer1}},
        };
        const std::vector<SpmSourceChunkRef> sources = {
            {SpmChunkKey(SpmPortId(74), 0), dense_fp16(2, 64), source0},
            {SpmChunkKey(SpmPortId(74), 1), dense_fp16(2, 64), source1},
            {SpmChunkKey(SpmPortId(74), 2), dense_fp16(2, 64), source2},
        };
        const std::vector<SpmBorrowedConsumerChunkRef> consumers = {
            {SpmChunkKey(SpmPortId(75), 0), dense_fp16(4, 64), consumer0},
            {SpmChunkKey(SpmPortId(75), 1), dense_fp16(2, 64), consumer1},
        };
        const std::vector<SpmRetainedPayloadChunkRef> payloads = {
            {SpmChunkKey(SpmPortId(75), 0), dense_fp16(2, 64), payload0},
            {SpmChunkKey(SpmPortId(75), 1), dense_fp16(2, 64), payload1},
            {SpmChunkKey(SpmPortId(75), 2), dense_fp16(2, 64), payload2},
        };
        const auto route = SpmChunkedPermutationRoute::bind(
            SpmPortId(74), SpmPortId(75), sources, consumers, payloads,
            {0, 1, 2, 3, 4, 5}, {{0, 6}}, /*policy_version=*/8,
            SpmConsumerTransform::RunAdd,
            SpmPayloadPartition::SourceConsumerSplit);
        const auto plan = SpmPipelinePlan::compile_phase_overlay(
            manifests, {route});
        const auto& lowered = route.payload_consumer_runs();
        EXPECT(lowered.size() == 3 &&
                   lowered[0].payload_chunk.ordinal() == 0 &&
                   lowered[0].consumer_chunk.ordinal() == 0 &&
                   lowered[0].transfer_event.value() == 10 &&
                   lowered[1].payload_chunk.ordinal() == 1 &&
                   lowered[1].consumer_chunk.ordinal() == 0 &&
                   lowered[1].transfer_event.value() == 10 &&
                   lowered[2].payload_chunk.ordinal() == 2 &&
                   lowered[2].consumer_chunk.ordinal() == 1 &&
                   lowered[2].transfer_event.value() == 14 &&
                   plan.hash() != 0,
               "split payload lowering uses payload ordinal rather than consumer ordinal");
    }

    // Generic direct-produced row-runs have no source or retained payload
    // backing.  Two disjoint producer chunks may share one logical event and
    // lower into different consumer chunks, but one output may not cross a
    // consumer boundary because the producer receives one final_out_addr.
    {
        const SpmPhaseSlice consumer0 = phase_slice(110, 0, 256, 0, 4);
        const SpmPhaseSlice consumer1 = phase_slice(110, 256, 256, 0, 4);
        const std::vector<SpmComponentPhaseManifest> manifests = {
            {SpmScratchId(110), 0, 512, UINT64_C(0x1101),
             {consumer0, consumer1}},
        };
        const std::vector<SpmDirectProducedChunkRef> produced = {
            {SpmChunkKey(SpmPortId(82), 0), dense_fp16(1, 64),
             SpmPipelineEvent(1)},
            {SpmChunkKey(SpmPortId(82), 1), dense_fp16(1, 64),
             SpmPipelineEvent(1)},
        };
        const std::vector<SpmBorrowedConsumerChunkRef> consumers = {
            {SpmChunkKey(SpmPortId(83), 0), dense_fp16(2, 64),
             consumer0},
            {SpmChunkKey(SpmPortId(83), 1), dense_fp16(2, 64),
             consumer1},
        };
        const auto route = SpmDirectProducedRowRunsRoute::bind(
            SpmPortId(82), SpmPortId(83), produced, consumers,
            {{1, 1}, {2, 1}}, /*policy_version=*/1);
        const auto route_again = SpmDirectProducedRowRunsRoute::bind(
            SpmPortId(82), SpmPortId(83), produced, consumers,
            {{1, 1}, {2, 1}}, /*policy_version=*/1);
        const auto plan = SpmPipelinePlan::compile_phase_overlay(
            manifests, std::vector<SpmChunkedPermutationRoute>{},
            std::vector<SpmDirectProducedRowRunsRoute>{route});
        const auto& lowered = route.lowered_runs();
        EXPECT(route.route_schema_version() == 1 &&
                   route.hash() == route_again.hash() &&
                   route.hash() == UINT64_C(0x96f09287e7a44169) &&
                   lowered.size() == 2 &&
                   lowered[0].consumer_chunk.ordinal() == 0 &&
                   lowered[0].consumer_row_begin == 1 &&
                   lowered[1].consumer_chunk.ordinal() == 1 &&
                   lowered[1].consumer_row_begin == 0 &&
                   plan.peak_bytes() == 512 &&
                   plan.hash() == UINT64_C(0xd648a7f03adf8eb1),
               "direct-produced route lowers shared-event outputs without source or payload storage");
        {
            auto lease = SpmPipelineLease::acquire(plan, 91);
            const auto checked = lease.direct_produced_row_runs(
                SpmPortId(82), SpmPortId(83));
            EXPECT(checked.targets().size() == 2 &&
                       checked.targets()[0].resolve_relative(lease, 91) ==
                           128 &&
                       checked.targets()[1].resolve_relative(lease, 91) ==
                           256 &&
                       checked.targets()[0].size_bytes() == 128,
                   "checked direct-produced targets resolve exact consumer row addresses");
            EXPECT(throws_c10([&] {
                       checked.targets()[0].resolve_relative(lease, 92);
                   }),
                   "direct-produced target rejects allocator-generation drift");
            const auto stale = checked.targets()[0];
            lease.release();
            EXPECT(throws_c10([&] {
                       stale.resolve_relative(lease, 91);
                   }),
                   "direct-produced target rejects a released lease");
        }
        auto changed_event = produced;
        changed_event[1].production_event = SpmPipelineEvent(2);
        EXPECT(SpmDirectProducedRowRunsRoute::bind(
                   SpmPortId(82), SpmPortId(83), changed_event, consumers,
                   {{1, 1}, {2, 1}}, 1).hash() != route.hash(),
               "direct-produced route hash seals production events");
        EXPECT(
            throws_c10([&] {
                SpmDirectProducedRowRunsRoute::bind(
                    SpmPortId(82), SpmPortId(83),
                    std::vector<SpmDirectProducedChunkRef>{
                        {SpmChunkKey(SpmPortId(82), 0),
                         dense_fp16(2, 64), SpmPipelineEvent(1)}},
                    consumers, {{1, 2}}, 1);
            }),
            "direct-produced route rejects an output crossing a consumer chunk boundary");
        EXPECT(
            throws_c10([&] {
                auto outside = produced;
                outside[0].production_event = SpmPipelineEvent(5);
                SpmDirectProducedRowRunsRoute::bind(
                    SpmPortId(82), SpmPortId(83), outside, consumers,
                    {{1, 1}, {2, 1}}, 1);
            }) &&
                throws_c10([&] {
                    SpmDirectProducedRowRunsRoute::bind(
                        SpmPortId(82), SpmPortId(83), produced, consumers,
                        {{1, 1}, {1, 1}}, 1);
                }),
            "direct-produced route rejects an out-of-lifetime write and overlapping targets");
    }

    // Independent direct-produced payloads retain producer output in place
    // until one alpha=1 RunAdd.  There is no nullable source, permutation, or
    // source-to-payload lowering in this schema.
    {
        const SpmPhaseSlice consumer =
            phase_slice(111, 0, 768, 4, 5);
        const std::vector<SpmComponentPhaseManifest> manifests = {
            {SpmScratchId(111), 0, 1280, UINT64_C(0x1111), {consumer}},
        };
        const std::vector<SpmDirectProducedPayloadRef> payloads = {
            {SpmChunkKey(SpmPortId(92), 0), dense_fp16(2, 64),
             phase_slice(111, 768, 256, 1, 3), SpmPipelineEvent(1)},
            {SpmChunkKey(SpmPortId(92), 1), dense_fp16(2, 64),
             phase_slice(111, 1024, 256, 2, 3), SpmPipelineEvent(2)},
        };
        const std::vector<SpmBorrowedConsumerChunkRef> consumers = {
            {SpmChunkKey(SpmPortId(93), 0), dense_fp16(6, 64),
             consumer},
        };
        const auto route = SpmDirectProducedPayloadRunAddRoute::bind(
            SpmPortId(92), SpmPortId(93), payloads, consumers,
            {{0, 2}, {4, 2}}, /*policy_version=*/3);
        const auto route_again =
            SpmDirectProducedPayloadRunAddRoute::bind(
                SpmPortId(92), SpmPortId(93), payloads, consumers,
                {{0, 2}, {4, 2}}, /*policy_version=*/3);
        const auto plan = SpmPipelinePlan::compile_phase_overlay(
            manifests, std::vector<SpmChunkedPermutationRoute>{},
            std::vector<SpmDirectProducedRowRunsRoute>{},
            std::vector<SpmDirectProducedPayloadRunAddRoute>{route});
        const auto plan_again = SpmPipelinePlan::compile_phase_overlay(
            manifests, std::vector<SpmChunkedPermutationRoute>{},
            std::vector<SpmDirectProducedRowRunsRoute>{},
            std::vector<SpmDirectProducedPayloadRunAddRoute>{route_again});
        EXPECT(route.route_schema_version() == 1 && route.alpha() == 1 &&
                   route.hash() == route_again.hash() &&
                   route.run_adds().size() == 2 &&
                   route.run_adds()[0].consumer_row_begin == 0 &&
                   route.run_adds()[1].consumer_row_begin == 4 &&
                   route.run_adds()[0].production_event.value() == 1 &&
                   route.run_adds()[1].production_event.value() == 2 &&
                   route.run_adds()[0].add_event.value() == 4 &&
                   route.run_adds()[1].add_event.value() == 4 &&
                   plan.peak_bytes() == 1280 &&
                   plan.hash() == plan_again.hash(),
               "direct-produced payload schema seals alpha=1 RunAdd without source lowering");
        EXPECT(plan.direct_produced_payload_run_add_hash(
                   SpmPortId(92), SpmPortId(93)) == route.hash(),
               "schema-v15 plan exposes its exact direct payload route hash");
        {
            auto lease = SpmPipelineLease::acquire(plan, 93);
            const auto checked = lease.direct_produced_payload_run_add(
                SpmPortId(92), SpmPortId(93));
            EXPECT(checked.alpha() == 1 && checked.payloads().size() == 2 &&
                       checked.consumer_chunks().size() == 1 &&
                       checked.run_adds().size() == 2 &&
                       checked.payloads()[0].resolve_relative(lease, 93) ==
                           768 &&
                       checked.payloads()[1].resolve_relative(lease, 93) ==
                           1024 &&
                       checked.consumer_chunks()[0].resolve_relative(
                           lease, 93) == 0 &&
                       checked.route_hash() == route.hash(),
                   "checked direct payload view retains both endpoint roles and RunAdd lowering");
            EXPECT(throws_c10([&] {
                       checked.payloads()[0].resolve_relative(lease, 94);
                   }),
                   "direct-produced payload view rejects allocator-generation drift");
            lease.release();
        }

        auto wrong_first = payloads;
        wrong_first[0].production_event = SpmPipelineEvent(0);
        auto late_consumer = consumers;
        late_consumer[0].backing = phase_slice(111, 0, 768, 5, 6);
        EXPECT(
            throws_c10([&] {
                SpmDirectProducedPayloadRunAddRoute::bind(
                    SpmPortId(92), SpmPortId(93), wrong_first, consumers,
                    {{0, 2}, {4, 2}}, 3);
            }) &&
                throws_c10([&] {
                    SpmDirectProducedPayloadRunAddRoute::bind(
                        SpmPortId(92), SpmPortId(93), payloads,
                        late_consumer, {{0, 2}, {4, 2}}, 3);
                }),
            "direct-produced payload route rejects forged production and add boundaries");
        EXPECT(
            throws_c10([&] {
                SpmDirectProducedPayloadRunAddRoute::bind(
                    SpmPortId(92), SpmPortId(93), payloads, consumers,
                    {{0, 2}, {1, 2}}, 3);
            }) &&
                throws_c10([&] {
                    const std::vector<SpmBorrowedConsumerChunkRef>
                        split_consumers = {
                            {SpmChunkKey(SpmPortId(93), 0),
                             dense_fp16(2, 64),
                             phase_slice(111, 0, 256, 4, 5)},
                            {SpmChunkKey(SpmPortId(93), 1),
                             dense_fp16(2, 64),
                             phase_slice(111, 256, 256, 4, 5)},
                        };
                    SpmDirectProducedPayloadRunAddRoute::bind(
                        SpmPortId(92), SpmPortId(93),
                        std::vector<SpmDirectProducedPayloadRef>{payloads[0]},
                        split_consumers, {{1, 2}}, 3);
                }),
            "direct-produced payload route rejects overlap and consumer-boundary crossing");
        EXPECT(throws_c10([&] {
                   auto bad_cols = payloads;
                   bad_cols[0].spec = dense_fp16(2, 24);
                   SpmDirectProducedPayloadRunAddRoute::bind(
                       SpmPortId(92), SpmPortId(93), bad_cols, consumers,
                       {{0, 2}, {4, 2}}, 3);
               }),
               "direct-produced payload route requires 16-column FP16 replicated storage");
    }

    // Exact LingBot2 DS template: three sequential image-local [64,2560]
    // sources remain three payloads even though S225 resolves to one Text
    // consumer chunk.  The three model instances for Text layers 0/1/2 will
    // carry distinct event windows; this fixture proves one typed RunAdd edge
    // only and remains synthetic CPU admission.
    {
        constexpr size_t kLingbotRowBytes = 2560 * sizeof(c10::Half);
        constexpr size_t kLingbotImageBytes = 64 * kLingbotRowBytes;
        constexpr size_t kLingbotTextBytes = 225 * kLingbotRowBytes;
        const SpmPhaseSlice source_image0 =
            phase_slice(104, 0, kLingbotImageBytes, 0, 0);
        const SpmPhaseSlice source_image1 =
            phase_slice(104, 0, kLingbotImageBytes, 8, 8);
        const SpmPhaseSlice source_image2 =
            phase_slice(104, 0, kLingbotImageBytes, 16, 16);
        const SpmPhaseSlice text_layer0 =
            phase_slice(105, 0, kLingbotTextBytes, 26, 27);
        const SpmPhaseSlice payload_image0 = phase_slice(
            105, static_cast<uint32_t>(kLingbotTextBytes),
            kLingbotImageBytes, 1, 25);
        const SpmPhaseSlice payload_image1 = phase_slice(
            105, static_cast<uint32_t>(
                     kLingbotTextBytes + kLingbotImageBytes),
            kLingbotImageBytes, 9, 25);
        const SpmPhaseSlice payload_image2 = phase_slice(
            105, static_cast<uint32_t>(
                     kLingbotTextBytes + 2 * kLingbotImageBytes),
            kLingbotImageBytes, 17, 25);
        const size_t exact_extent =
            kLingbotTextBytes + 3 * kLingbotImageBytes;
        const std::vector<SpmComponentPhaseManifest> manifests = {
            {SpmScratchId(104), 0, kLingbotImageBytes, 0x1041,
             {source_image0, source_image1, source_image2}},
            {SpmScratchId(105), 0, exact_extent, 0x1051,
             {text_layer0}},
        };
        const std::vector<SpmSourceChunkRef> sources = {
            {SpmChunkKey(SpmPortId(72), 0), dense_fp16(64, 2560),
             source_image0},
            {SpmChunkKey(SpmPortId(72), 1), dense_fp16(64, 2560),
             source_image1},
            {SpmChunkKey(SpmPortId(72), 2), dense_fp16(64, 2560),
             source_image2},
        };
        const std::vector<SpmBorrowedConsumerChunkRef> consumers = {
            {SpmChunkKey(SpmPortId(73), 0), dense_fp16(225, 2560),
             text_layer0},
        };
        const std::vector<SpmRetainedPayloadChunkRef> payloads = {
            {SpmChunkKey(SpmPortId(73), 0), dense_fp16(64, 2560),
             payload_image0},
            {SpmChunkKey(SpmPortId(73), 1), dense_fp16(64, 2560),
             payload_image1},
            {SpmChunkKey(SpmPortId(73), 2), dense_fp16(64, 2560),
             payload_image2},
        };
        std::vector<int32_t> identity(192);
        std::iota(identity.begin(), identity.end(), int32_t{0});
        const auto run_add = SpmChunkedPermutationRoute::bind(
            SpmPortId(72), SpmPortId(73), sources, consumers, payloads,
            identity, {{1, 64}, {67, 64}, {133, 64}},
            /*policy_version=*/9, SpmConsumerTransform::RunAdd,
            SpmPayloadPartition::SourceConsumerSplit);
        const auto plan = SpmPipelinePlan::compile_phase_overlay(
            manifests, {run_add});
        const auto& lowered = run_add.payload_consumer_runs();
        EXPECT(run_add.payload_chunks().size() == 3 &&
                   run_add.source_payload_runs().size() == 3 &&
                   lowered.size() == 3 &&
                   lowered[0].consumer_row_begin == 1 &&
                   lowered[1].consumer_row_begin == 67 &&
                   lowered[2].consumer_row_begin == 133 &&
                   lowered[0].row_count == 64 &&
                   lowered[1].row_count == 64 &&
                   lowered[2].row_count == 64 &&
                   plan.peak_bytes() == exact_extent,
               "LingBot2 DS RunAdd keeps three N256 image payloads for one C225 consumer");
        const auto overwrite = SpmChunkedPermutationRoute::bind(
            SpmPortId(72), SpmPortId(73), sources, consumers, payloads,
            identity, {{1, 64}, {67, 64}, {133, 64}},
            /*policy_version=*/9, SpmConsumerTransform::Overwrite,
            SpmPayloadPartition::SourceConsumerSplit);
        EXPECT(overwrite.hash() != run_add.hash(),
               "LingBot2 pooler overwrite and DeepStack RunAdd have distinct route identity");
    }

    // Exact sequential LingBot2 arithmetic gate.  The fixed profile uses one
    // reusable N256 Vision arena, one S225 Text arena without the ordinary
    // deepstack_scratch, and nine image-local DeepStack payloads.  Pooler
    // output is direct-produced into Text residual row-runs, so it has no
    // retained payload.  This remains a synthetic CPU plan until FMB-sealed
    // manifests and direct-produced route authority reach physical admission.
    {
        constexpr size_t kImageCount = 3;
        constexpr size_t kYieldCountPerImage = 4;
        constexpr size_t kDirectYieldCount = kImageCount;
        constexpr size_t kRetainedTapCount = kYieldCountPerImage - 1;
        constexpr size_t kRetainedYieldCount =
            kRetainedTapCount * kImageCount;
        constexpr size_t kRowBytes = 2560 * sizeof(c10::Half);
        constexpr size_t kImagePayloadBytes = 64 * kRowBytes;
        constexpr size_t kTextResidualBytes = 225 * kRowBytes;
        constexpr size_t kVisionTemporaryBytes = 1'966'080;
        constexpr size_t kTextTemporaryBytes = 4'163'072;
        constexpr size_t kDeepstackPayloadBytes =
            9 * kImagePayloadBytes;
        constexpr size_t kDynamicPeakBytes =
            kTextTemporaryBytes + kDeepstackPayloadBytes;
        // Text: 2*36*[2560] + 2*36*[128] + final [2560].
        constexpr size_t kTextPersistentBytes = 392'192;
        // Vision: 24 layers of norm/bias slots, 2*[48,16] RoPE tables,
        // and shared merger norm/m0/m2 bias slots.
        constexpr size_t kVisionPersistentBytes = 381'952;
        constexpr size_t kPersistentTopBytes =
            kTextPersistentBytes + kVisionPersistentBytes;
        constexpr size_t kTotalPeakBytes =
            kDynamicPeakBytes + kPersistentTopBytes;
        constexpr size_t kExpectedHeadroomBytes = 417'280;
        // Production co-residency adds only the exact cold-primed LingBot2
        // expert Persistent* roots; the generic V/T dynamic plan is unchanged.
        constexpr size_t kExpertPersistentBytes = 318'976;
        constexpr size_t kProductionPersistentTopBytes =
            kPersistentTopBytes + kExpertPersistentBytes;
        constexpr size_t kProductionTotalPeakBytes =
            kDynamicPeakBytes + kProductionPersistentTopBytes;
        constexpr size_t kAlignedBudgetBytes =
            SpmAllocator::SPM_PLANNING_BUDGET &
            ~(SpmAllocator::ALIGN - 1);
        constexpr size_t kProductionHeadroomBytes =
            kAlignedBudgetBytes - kProductionTotalPeakBytes;

        bool canonical_yield_cover =
            kDirectYieldCount + kRetainedYieldCount == 12;
        std::array<bool, kRetainedYieldCount> canonical_payloads{};
        for (size_t tap = 0; tap < kRetainedTapCount; ++tap) {
            for (size_t image = 0; image < kImageCount; ++image) {
                const size_t payload_index = tap * kImageCount + image;
                canonical_yield_cover = canonical_yield_cover &&
                    payload_index < canonical_payloads.size() &&
                    !canonical_payloads[payload_index];
                canonical_payloads[payload_index] = true;
            }
        }
        canonical_yield_cover = canonical_yield_cover &&
            std::all_of(canonical_payloads.begin(), canonical_payloads.end(),
                        [](bool present) { return present; });

        const std::array<uint32_t, 9> payload_production_events = {
            2, 10, 18, 4, 12, 20, 6, 14, 22};
        const std::array<SpmPhaseSlice, 3> consumers = {
            phase_slice(109, 0, kTextResidualBytes, 27, 29),
            phase_slice(109, 0, kTextResidualBytes, 30, 32),
            phase_slice(109, 0, kTextResidualBytes, 33, 35),
        };
        const SpmPhaseSlice pooler_consumer =
            phase_slice(109, 0, kTextResidualBytes, 7, 26);

        std::vector<SpmPhaseSlice> vision_ranges = {
            phase_slice(
                108, 0, kVisionTemporaryBytes, 0, 23),
        };
        const std::vector<SpmComponentPhaseManifest> manifests = {
            {SpmScratchId(108),
             static_cast<uint32_t>(kTextResidualBytes),
             kVisionTemporaryBytes, UINT64_C(0x1081), vision_ranges},
            {SpmScratchId(109), 0, kDynamicPeakBytes, UINT64_C(0x1091),
             {
                 phase_slice(
                     109, static_cast<uint32_t>(kTextResidualBytes),
                     kTextTemporaryBytes - kTextResidualBytes, 24, 35),
                 pooler_consumer,
                 consumers[0], consumers[1], consumers[2],
             }},
        };

        // The three merger invocations write the final visual prefix rows at
        // their Vision post_fn events.  A later Text ingress may initialize
        // only the complement of these row-runs; a full residual overwrite at
        // event 24 would invalidate this route and must be rejected by the
        // later FMB/Graph access-trace gate.
        const std::vector<SpmDirectProducedChunkRef> pooler_produced = {
            {SpmChunkKey(SpmPortId(84), 0), dense_fp16(64, 2560),
             SpmPipelineEvent(7)},
            {SpmChunkKey(SpmPortId(84), 1), dense_fp16(64, 2560),
             SpmPipelineEvent(15)},
            {SpmChunkKey(SpmPortId(84), 2), dense_fp16(64, 2560),
             SpmPipelineEvent(23)},
        };
        const auto pooler_route = SpmDirectProducedRowRunsRoute::bind(
            SpmPortId(84), SpmPortId(85), pooler_produced,
            std::vector<SpmBorrowedConsumerChunkRef>{
                {SpmChunkKey(SpmPortId(85), 0),
                 dense_fp16(225, 2560), pooler_consumer}},
            {{1, 64}, {67, 64}, {133, 64}},
            /*policy_version=*/1);

        std::vector<SpmDirectProducedPayloadRunAddRoute>
            deepstack_routes;
        deepstack_routes.reserve(3);
        for (size_t tap = 0; tap < 3; ++tap) {
            const SpmPortId payload_port(
                static_cast<uint32_t>(76 + tap * 2));
            const SpmPortId destination_port(
                static_cast<uint32_t>(77 + tap * 2));
            std::vector<SpmDirectProducedPayloadRef> payloads;
            payloads.reserve(3);
            for (size_t image = 0; image < 3; ++image) {
                const size_t ordinal = tap * 3 + image;
                const size_t payload_offset =
                    kTextTemporaryBytes + ordinal * kImagePayloadBytes;
                payloads.push_back({
                    SpmChunkKey(
                        payload_port, static_cast<uint32_t>(image)),
                    dense_fp16(64, 2560),
                    phase_slice(
                        109, static_cast<uint32_t>(payload_offset),
                        kImagePayloadBytes,
                        payload_production_events[ordinal],
                        consumers[tap].first_event.value() - 1),
                    SpmPipelineEvent(
                        payload_production_events[ordinal])});
            }
            deepstack_routes.push_back(
                SpmDirectProducedPayloadRunAddRoute::bind(
                payload_port, destination_port, payloads,
                std::vector<SpmBorrowedConsumerChunkRef>{
                    {SpmChunkKey(destination_port, 0),
                     dense_fp16(225, 2560), consumers[tap]}},
                {{1, 64}, {67, 64}, {133, 64}},
                /*policy_version=*/10));
        }

        const auto plan = SpmPipelinePlan::compile_phase_overlay(
            manifests, std::vector<SpmChunkedPermutationRoute>{},
            std::vector<SpmDirectProducedRowRunsRoute>{pooler_route},
            deepstack_routes,
            kPersistentTopBytes);
        const auto plan_again = SpmPipelinePlan::compile_phase_overlay(
            manifests, std::vector<SpmChunkedPermutationRoute>{},
            std::vector<SpmDirectProducedRowRunsRoute>{pooler_route},
            deepstack_routes,
            kPersistentTopBytes);
        EXPECT(kVisionTemporaryBytes + kTextResidualBytes <
                   kTextTemporaryBytes &&
                   kDeepstackPayloadBytes == 2'949'120 &&
                   kDynamicPeakBytes == 7'112'192 &&
                   plan.peak_bytes() == kDynamicPeakBytes &&
                   plan.reserved_top_bytes() == kPersistentTopBytes &&
                   plan.total_bytes() == kTotalPeakBytes &&
                   plan.effective_budget_bytes() == 8'303'616 &&
                   plan.effective_budget_bytes() - plan.total_bytes() ==
                       kExpectedHeadroomBytes &&
                   kProductionPersistentTopBytes == 1'093'120 &&
                   kProductionTotalPeakBytes == 8'205'312 &&
                   kProductionHeadroomBytes == 98'304 &&
                   plan.hash() == plan_again.hash() &&
                   canonical_yield_cover,
               "LingBot2 sequential N256 synthetic plan fits with 417,280-byte headroom");
        {
            auto lease = SpmPipelineLease::acquire(plan, 92);
            const auto direct = lease.direct_produced_row_runs(
                SpmPortId(84), SpmPortId(85));
            size_t deepstack_payload_count = 0;
            bool deepstack_routes_exact = true;
            for (size_t tap = 0; tap < deepstack_routes.size(); ++tap) {
                const SpmPortId payload_port(
                    static_cast<uint32_t>(76 + tap * 2));
                const SpmPortId destination_port(
                    static_cast<uint32_t>(77 + tap * 2));
                const auto checked =
                    lease.direct_produced_payload_run_add(
                        payload_port, destination_port);
                deepstack_payload_count += checked.payloads().size();
                deepstack_routes_exact = deepstack_routes_exact &&
                    checked.alpha() == 1 &&
                    checked.payloads().size() == 3 &&
                    checked.run_adds().size() == 3 &&
                    checked.run_adds()[0].consumer_row_begin == 1 &&
                    checked.run_adds()[1].consumer_row_begin == 67 &&
                    checked.run_adds()[2].consumer_row_begin == 133 &&
                    checked.run_adds()[0].add_event.value() ==
                        consumers[tap].first_event.value() &&
                    plan.direct_produced_payload_run_add_hash(
                        payload_port, destination_port) ==
                        deepstack_routes[tap].hash();
                for (size_t image = 0; image < 3; ++image) {
                    const size_t ordinal = tap * 3 + image;
                    deepstack_routes_exact = deepstack_routes_exact &&
                        checked.payloads()[image].resolve_relative(
                            lease, 92) ==
                            kTextTemporaryBytes +
                                ordinal * kImagePayloadBytes &&
                        checked.run_adds()[image]
                                .production_event.value() ==
                            payload_production_events[ordinal];
                }
            }
            EXPECT(direct.targets().size() == 3 &&
                       direct.targets()[0].resolve_relative(lease, 92) ==
                           5'120 &&
                       direct.targets()[1].resolve_relative(lease, 92) ==
                           343'040 &&
                       direct.targets()[2].resolve_relative(lease, 92) ==
                           680'960 &&
                       direct.targets()[0].size_bytes() ==
                           kImagePayloadBytes &&
                       direct.route_hash() == pooler_route.hash() &&
                       plan.direct_produced_row_runs_hash(
                           SpmPortId(84), SpmPortId(85)) ==
                           pooler_route.hash() &&
                       deepstack_payload_count == 9 &&
                       deepstack_routes_exact,
                   "LingBot2 pooler and nine DeepStack payloads expose exact checked direct addresses");
            EXPECT(throws_c10([&] {
                       lease.direct_produced_row_runs(
                           SpmPortId(84), SpmPortId(86));
                   }),
                   "LingBot2 direct-produced view rejects an unsealed endpoint pair");
            lease.release();
        }
        EXPECT(
            plan.total_bytes() + 3 * kImagePayloadBytes >
                plan.effective_budget_bytes(),
            "LingBot2 cannot retain three extra pooler payloads inside the canonical budget");
    }

    // The old coarse geometry allocates full source and P640 storage alongside
    // live scratch, which is over budget; v5 only validates fixed overlays.
    const auto coarse_production_route = permutation_row_runs_route(
        50, dense_fp16(432, 2048), 51, dense_fp16(640, 2048),
        production_permutation, production_runs,
        SpmPermutationRowRunsEvents{0, 0, 1, 2});
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile(
                   std::vector<SpmScratchDecl>{
                       typed_scratch(50, 6000000, 0, 2)},
                   std::vector<SpmPermutationRowRunsRoute>{
                       coarse_production_route});
           }) && phase_plan.peak_bytes() <=
                    SpmAllocator::SPM_PLANNING_BUDGET,
           "production-like coarse full-source/P640 plan is over budget while overlay fits");

    {
        auto lease = SpmPipelineLease::acquire(phase_plan, 77);
        const auto checked = lease.chunked_permutation(
            SpmPortId(40), SpmPortId(41));
        EXPECT(checked.source_chunks().size() == 3 &&
                   checked.payload_chunks().size() == 2 &&
                   checked.consumer_chunks().size() == 2 &&
                   checked.source_chunks()[0].resolve_relative(lease, 77) ==
                       kSourceOffset &&
                   checked.source_chunks()[1].resolve_relative(lease, 77) ==
                       kSourceOffset,
               "checked v5 source views preserve one reused fixed backing");
        EXPECT(checked.payload_chunks()[0].resolve_relative(lease, 77) ==
                   kPayload0Offset &&
                   checked.payload_chunks()[1].resolve_relative(lease, 77) ==
                       kPayload1Offset &&
                   checked.consumer_chunks()[0].resolve_relative(lease, 77) ==
                       0 &&
                   checked.consumer_chunks()[1].resolve_relative(lease, 77) ==
                       0,
               "checked payload and borrowed residual views preserve fixed placement");
        EXPECT(checked.route_hash() == production_route.hash() &&
                   checked.policy_version() == 7 &&
                   checked.payload_consumer_runs().size() == 4,
               "checked route exports sealed policy and lowering metadata");
        EXPECT(throws_c10([&] {
                   checked.payload_chunks()[0].resolve_relative(lease, 78);
               }),
               "checked v5 view rejects allocator-generation drift");

        const auto stale = checked.consumer_chunks()[0];
        lease.release();
        auto next = SpmPipelineLease::acquire(phase_plan, 77);
        EXPECT(throws_c10([&] { stale.resolve_relative(next, 77); }),
               "checked v5 view rejects a stale lease epoch");
        next.release();
    }
    EXPECT(throws_c10([&] {
               auto lease = SpmPipelineLease::acquire(phase_plan, 79);
               lease.chunked_permutation(SpmPortId(41), SpmPortId(40));
           }),
           "checked v5 route rejects an unsealed endpoint pair");

    const SpmPipelinePlan phase_same =
        SpmPipelinePlan::compile_phase_overlay(
            phase_manifests, {production_route});
    EXPECT(phase_plan.hash() == phase_same.hash() &&
               phase_plan.hash() != permutation_plan.hash(),
           "schema-v5 hash is deterministic and distinct from schema-v4");
    EXPECT(production_route.hash() == 0xe45c9537bb3f0e63ULL &&
               phase_plan.hash() == 0x66f5f3dd87cc9205ULL,
           "chunked route and schema-v5 plan hashes are stable across processes");
    auto cross_source_permutation = production_permutation;
    cross_source_permutation[0] = 144;
    cross_source_permutation[144] = 0;
    const auto cross_source_route = SpmChunkedPermutationRoute::bind(
        SpmPortId(40), SpmPortId(41), production_sources,
        production_consumers, production_payloads,
        cross_source_permutation, production_runs, 7);
    bool transfer_events_monotonic = true;
    bool saw_chunk1_event4 = false;
    bool chunk1_events_are4 = true;
    const auto& cross_source_runs =
        cross_source_route.source_payload_runs();
    for (size_t i = 0; i < cross_source_runs.size(); ++i) {
        if (i > 0) {
            transfer_events_monotonic = transfer_events_monotonic &&
                cross_source_runs[i - 1].transfer_event.value() <=
                    cross_source_runs[i].transfer_event.value();
        }
        if (cross_source_runs[i].source_chunk.ordinal() == 1) {
            saw_chunk1_event4 = true;
            chunk1_events_are4 = chunk1_events_are4 &&
                cross_source_runs[i].transfer_event.value() == 4;
        }
    }
    EXPECT(!cross_source_runs.empty() && transfer_events_monotonic &&
               cross_source_runs.front().source_chunk.ordinal() == 0 &&
               cross_source_runs.front().transfer_event.value() == 2 &&
               saw_chunk1_event4 && chunk1_events_are4 &&
               cross_source_route.hash() != production_route.hash(),
           "global permutation schedules chunk0 event2 before overwritten chunk1 event4");
    constexpr uint32_t kIndependentSourceOffset = 3500032;
    constexpr uint32_t kIndependentConsumerOffset = 4000000;
    constexpr uint32_t kIndependentPayload0Offset = 5500160;
    constexpr uint32_t kIndependentPayload1Offset = 6495488;
    const SpmPhaseSlice independent_source0 = phase_slice(
        100, kIndependentSourceOffset, kSourceChunkBytes, 12, 13);
    const SpmPhaseSlice independent_source1 = phase_slice(
        100, kIndependentSourceOffset, kSourceChunkBytes, 14, 15);
    const SpmPhaseSlice independent_source2 = phase_slice(
        100, kIndependentSourceOffset, kSourceChunkBytes, 16, 17);
    const SpmPhaseSlice independent_consumer0 = phase_slice(
        101, kIndependentConsumerOffset, kConsumerChunkBytes, 20, 21);
    const SpmPhaseSlice independent_consumer1 = phase_slice(
        101, kIndependentConsumerOffset, kConsumerChunkBytes, 22, 23);
    const SpmPhaseSlice independent_payload0 = phase_slice(
        101, kIndependentPayload0Offset, 243 * kRowBytes2048, 14, 19);
    const SpmPhaseSlice independent_payload1 = phase_slice(
        101, kIndependentPayload1Offset, 189 * kRowBytes2048, 16, 21);
    auto independent_manifests = phase_manifests;
    independent_manifests[0].live_ranges.insert(
        independent_manifests[0].live_ranges.end(),
        {independent_source0, independent_source1, independent_source2});
    independent_manifests[1].live_ranges.insert(
        independent_manifests[1].live_ranges.end(),
        {independent_consumer0, independent_consumer1});
    const std::vector<SpmSourceChunkRef> independent_sources40 = {
        {SpmChunkKey(SpmPortId(40), 0), dense_fp16(144, 2048),
         independent_source0},
        {SpmChunkKey(SpmPortId(40), 1), dense_fp16(144, 2048),
         independent_source1},
        {SpmChunkKey(SpmPortId(40), 2), dense_fp16(144, 2048),
         independent_source2},
    };
    const std::vector<SpmBorrowedConsumerChunkRef>
        independent_consumers42 = {
            {SpmChunkKey(SpmPortId(42), 0), dense_fp16(320, 2048),
             independent_consumer0},
            {SpmChunkKey(SpmPortId(42), 1), dense_fp16(320, 2048),
             independent_consumer1},
        };
    const std::vector<SpmRetainedPayloadChunkRef>
        independent_payloads42 = {
            {SpmChunkKey(SpmPortId(42), 0), dense_fp16(243, 2048),
             independent_payload0},
            {SpmChunkKey(SpmPortId(42), 1), dense_fp16(189, 2048),
             independent_payload1},
        };
    const auto reused_source_route = SpmChunkedPermutationRoute::bind(
        SpmPortId(40), SpmPortId(42), independent_sources40,
        independent_consumers42, independent_payloads42,
        production_permutation, production_runs, 7);
    auto independent_sources42 = independent_sources40;
    for (size_t i = 0; i < independent_sources42.size(); ++i) {
        independent_sources42[i].key =
            SpmChunkKey(SpmPortId(42), static_cast<uint32_t>(i));
    }
    auto independent_consumers41 = independent_consumers42;
    auto independent_payloads41 = independent_payloads42;
    for (size_t i = 0; i < independent_consumers41.size(); ++i) {
        independent_consumers41[i].key =
            SpmChunkKey(SpmPortId(41), static_cast<uint32_t>(i));
        independent_payloads41[i].key =
            SpmChunkKey(SpmPortId(41), static_cast<uint32_t>(i));
    }
    const auto reused_destination_route = SpmChunkedPermutationRoute::bind(
        SpmPortId(42), SpmPortId(41), independent_sources42,
        independent_consumers41, independent_payloads41,
        production_permutation, production_runs, 7);
    const auto independent_source_plan =
        SpmPipelinePlan::compile_phase_overlay(
            independent_manifests, {reused_source_route});
    const auto independent_destination_plan =
        SpmPipelinePlan::compile_phase_overlay(
            independent_manifests, {reused_destination_route});
    EXPECT(independent_source_plan.hash() != 0 &&
               independent_destination_plan.hash() != 0,
           "port-ID reuse fixtures are independently valid physical routes");
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile_phase_overlay(
                   independent_manifests,
                   {production_route, reused_source_route});
           }) &&
               throws_c10([&] {
                   SpmPipelinePlan::compile_phase_overlay(
                       independent_manifests,
                       {production_route, reused_destination_route});
               }),
           "schema-v5 rejects plan-wide source or destination port-ID reuse");
    const auto policy_changed =
        bind_production_route(production_sources, production_payloads, 8);
    EXPECT(production_route.hash() != policy_changed.hash() &&
               phase_plan.hash() !=
                   SpmPipelinePlan::compile_phase_overlay(
                       phase_manifests, {policy_changed})
                       .hash(),
           "policy version participates in route and plan hashes");
    auto changed_event_sources = production_sources;
    changed_event_sources[2].backing = phase_slice(
        100, kSourceOffset, kSourceChunkBytes, 4, 6);
    auto changed_event_manifests = phase_manifests;
    changed_event_manifests[0].live_ranges[3] =
        changed_event_sources[2].backing;
    const auto changed_event_route =
        bind_production_route(changed_event_sources,
                              production_payloads, 7);
    EXPECT(production_route.hash() != changed_event_route.hash() &&
               phase_plan.hash() !=
                   SpmPipelinePlan::compile_phase_overlay(
                       changed_event_manifests, {changed_event_route})
                       .hash(),
           "global inclusive events participate in route and plan hashes");
    auto changed_layout_manifests = phase_manifests;
    changed_layout_manifests[1].layout_hash = 0x2002;
    EXPECT(phase_plan.hash() !=
               SpmPipelinePlan::compile_phase_overlay(
                   changed_layout_manifests, {production_route})
                   .hash(),
           "component layout hash participates in schema-v5 plan hash");
    auto changed_base_manifests = phase_manifests;
    changed_base_manifests[1].arena_base = 256;
    changed_base_manifests[1].arena_extent = kPhaseExtent - 256;
    EXPECT(phase_plan.hash() !=
               SpmPipelinePlan::compile_phase_overlay(
                   changed_base_manifests, {production_route})
                   .hash(),
           "fixed arena base and extent participate in schema-v5 plan hash");
    auto budget_hash_manifests = phase_manifests;
    for (auto& manifest : budget_hash_manifests) {
        manifest.arena_extent = kPhaseExtent - 256;
    }
    EXPECT(SpmPipelinePlan::compile_phase_overlay(
               budget_hash_manifests, {production_route}).hash() !=
               SpmPipelinePlan::compile_phase_overlay(
                   budget_hash_manifests, {production_route}, 0,
                   kPhaseExtent - 256).hash(),
           "phase-overlay budget participates in schema-v5 plan hash");
    auto smaller_extent_manifests = phase_manifests;
    for (auto& manifest : smaller_extent_manifests) {
        manifest.arena_extent = kPhaseExtent - 512;
    }
    EXPECT(phase_plan.hash() !=
               SpmPipelinePlan::compile_phase_overlay(
                   smaller_extent_manifests, {production_route}, 256)
                   .hash(),
           "phase-overlay reserved top and fixed peak participate in its hash");
    auto moved_payloads = production_payloads;
    moved_payloads[1].backing = phase_slice(
        101, kPayload1Offset + 256, 189 * kRowBytes2048, 4, 9);
    const auto moved_payload_route =
        bind_production_route(production_sources, moved_payloads, 7);
    EXPECT(production_route.hash() != moved_payload_route.hash() &&
               phase_plan.hash() !=
                   SpmPipelinePlan::compile_phase_overlay(
                       phase_manifests, {moved_payload_route})
                       .hash(),
           "explicit retained-payload placement participates in v5 hashes");

    auto same_event_sources = production_sources;
    same_event_sources[1].backing = phase_slice(
        100, kSourceOffset, kSourceChunkBytes, 1, 3);
    auto same_event_manifests = phase_manifests;
    same_event_manifests[0].live_ranges[2] = same_event_sources[1].backing;
    const auto same_event_route =
        bind_production_route(same_event_sources, production_payloads, 7);
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile_phase_overlay(
                   same_event_manifests, {same_event_route});
           }),
           "inclusive events reject same-backing overwrite at predecessor last event");

    auto aliased_payloads = production_payloads;
    aliased_payloads[0].backing =
        phase_slice(101, 0, 243 * kRowBytes2048, 2, 7);
    const auto aliased_payload_route =
        bind_production_route(production_sources, aliased_payloads, 7);
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile_phase_overlay(
                   phase_manifests, {aliased_payload_route});
           }),
           "transfer endpoints reject in-place payload-to-residual byte overlap");

    auto source_aliased_payloads = production_payloads;
    source_aliased_payloads[0].backing = phase_slice(
        101, kSourceOffset, 243 * kRowBytes2048, 2, 7);
    const auto source_aliased_route =
        bind_production_route(production_sources,
                              source_aliased_payloads, 7);
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile_phase_overlay(
                   phase_manifests, {source_aliased_route});
           }),
           "transfer endpoints reject direct source-to-payload byte alias");

    auto unaligned_payloads = production_payloads;
    unaligned_payloads[1].backing = phase_slice(
        101, kPayload1Offset + 1, 189 * kRowBytes2048, 4, 9);
    const auto unaligned_payload_route =
        bind_production_route(production_sources, unaligned_payloads, 7);
    auto unaligned_bytes_manifests = phase_manifests;
    unaligned_bytes_manifests[0].live_ranges.push_back(
        phase_slice(100, 3200000, 257, 12, 12));
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile_phase_overlay(
                   phase_manifests, {unaligned_payload_route});
           }) &&
               throws_c10([&] {
                   SpmPipelinePlan::compile_phase_overlay(
                       unaligned_bytes_manifests, {production_route});
               }),
           "phase seal rejects non-256-byte slice offsets and byte counts");

    auto colliding_manifests = phase_manifests;
    colliding_manifests[0].live_ranges.push_back(phase_slice(
        100, kPayload0Offset, 243 * kRowBytes2048, 2, 2));
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile_phase_overlay(
                   colliding_manifests, {production_route});
           }),
           "fixed byte plus inclusive-event collision is rejected across arenas");

    auto unsealed_sources = production_sources;
    unsealed_sources[0].backing = phase_slice(
        100, kSourceOffset, kSourceChunkBytes, 1, 1);
    const auto unsealed_source_route =
        bind_production_route(unsealed_sources, production_payloads, 7);
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile_phase_overlay(
                   phase_manifests, {unsealed_source_route});
           }),
           "source endpoint must exactly match one complete manifest live range");

    auto out_of_arena_payloads = production_payloads;
    out_of_arena_payloads[0].backing = phase_slice(
        101, static_cast<uint32_t>(kPhaseExtent),
        243 * kRowBytes2048, 2, 7);
    const auto out_of_arena_route =
        bind_production_route(production_sources,
                              out_of_arena_payloads, 7);
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile_phase_overlay(
                   phase_manifests, {out_of_arena_route});
           }),
           "retained payload placement must stay within its sealed arena extent");

    auto bad_payload_rows = production_payloads;
    bad_payload_rows[0].spec = dense_fp16(242, 2048);
    bad_payload_rows[0].backing =
        phase_slice(101, kPayload0Offset, 242 * kRowBytes2048, 2, 7);
    EXPECT(throws_c10([&] {
               bind_production_route(production_sources,
                                     bad_payload_rows, 7);
           }),
           "binder rejects payload rows that do not equal routed image rows");
    auto bad_ordinals = production_sources;
    bad_ordinals[1].key = SpmChunkKey(SpmPortId(40), 2);
    EXPECT(throws_c10([&] {
               bind_production_route(bad_ordinals,
                                     production_payloads, 7);
           }),
           "binder rejects non-contiguous source chunk ordinals");
    auto duplicate_global_permutation = production_permutation;
    duplicate_global_permutation[431] = 430;
    EXPECT(throws_c10([&] {
               SpmChunkedPermutationRoute::bind(
                   SpmPortId(40), SpmPortId(41), production_sources,
                   production_consumers, production_payloads,
                   duplicate_global_permutation, production_runs, 7);
           }),
           "binder requires one strict permutation across all source chunks");
    EXPECT(throws_c10([&] {
               SpmChunkedPermutationRoute::bind(
                   SpmPortId(40), SpmPortId(41), production_sources,
                   production_consumers, production_payloads,
                   production_permutation, production_runs, 0);
           }),
           "binder rejects unversioned lowering policy");
    auto early_payloads = production_payloads;
    early_payloads[0].backing = phase_slice(
        101, kPayload0Offset, 243 * kRowBytes2048, 1, 7);
    auto short_payloads = production_payloads;
    short_payloads[0].backing = phase_slice(
        101, kPayload0Offset, 243 * kRowBytes2048, 2, 6);
    EXPECT(throws_c10([&] {
               bind_production_route(production_sources,
                                     early_payloads, 7);
           }) &&
               throws_c10([&] {
                   bind_production_route(production_sources,
                                         short_payloads, 7);
               }),
           "binder seals earliest incoming and payload-to-consumer transfer boundaries");
    auto overflow_sources = production_sources;
    overflow_sources[2].backing = phase_slice(
        100, kSourceOffset, kSourceChunkBytes, 4,
        std::numeric_limits<uint32_t>::max());
    auto overflow_payloads = production_payloads;
    overflow_payloads[1].backing = phase_slice(
        101, kPayload1Offset, 189 * kRowBytes2048, 4,
        std::numeric_limits<uint32_t>::max());
    EXPECT(throws_c10([&] {
               bind_production_route(overflow_sources,
                                     production_payloads, 7);
           }) &&
               throws_c10([&] {
                   bind_production_route(production_sources,
                                         overflow_payloads, 7);
               }),
           "binder rejects source and payload transfer-event overflow");

    // Lock the first-fit fragmentation behavior inherited from SpmAllocator.
    const std::vector<v3::BufferDecl> fragmented = {
        temp_region("short_0", 1024, 2, 2),
        temp_region("late", 1280, 4, 6),
        temp_region("short_1", 1792, 3, 3),
        temp_region("long", 1024, 2, 5),
    };
    EXPECT(SpmPipelinePlan::compile(fragmented).peak_bytes() == 3840,
           "pipeline plan reuses allocator exact first-fit behavior");

    const std::vector<v3::BufferDecl> boundary = {
        temp_region("boundary", 8303616, 0, 0),
    };
    EXPECT(SpmPipelinePlan::compile(boundary).peak_bytes() == 8303616,
           "largest aligned plan at the canonical budget is admitted");
    const std::vector<v3::BufferDecl> over_boundary = {
        temp_region("over_boundary", 8303617, 0, 0),
    };
    EXPECT(throws_c10(
               [&] { SpmPipelinePlan::compile(over_boundary); }),
           "next 256-byte allocation above effective ceiling is rejected");
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile(regions, 0,
                                        SpmAllocator::SPM_PLANNING_BUDGET + 1);
           }),
           "caller cannot widen the canonical planning ceiling");
    const std::vector<v3::BufferDecl> reserved_boundary = {
        temp_region("reserved_boundary", 8303360, 0, 0),
    };
    const SpmPipelinePlan reserved_aligned =
        SpmPipelinePlan::compile(reserved_boundary, 255);
    EXPECT(reserved_aligned.reserved_top_bytes() == 256 &&
               reserved_aligned.total_bytes() == 8303616,
           "top reservation is aligned before budget admission");
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile(reserved_boundary, 257);
           }),
           "next aligned top reservation above effective ceiling is rejected");
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile(
                   {temp_region("small", 256, 0, 0)},
                   SpmAllocator::SPM_PLANNING_BUDGET);
           }),
           "top reservation equal to raw budget is rejected after alignment");

    EXPECT(throws_c10([&] { SpmPipelinePlan::compile({}); }),
           "empty region list is rejected");

    v3::BufferDecl invalid = temp_region("invalid", 256, 0, 0);
    invalid.storage = StorageClass::Persistent;
    EXPECT(throws_c10(
               [&] { SpmPipelinePlan::compile({invalid}); }),
           "persistent BufferDecl is rejected in V1");
    invalid = temp_region("invalid", 256, 0, 0);
    invalid.scope = BufferScope::Compute;
    EXPECT(throws_c10(
               [&] { SpmPipelinePlan::compile({invalid}); }),
           "non-LayerWide BufferDecl is rejected in V1");
    invalid = temp_region("invalid", 256, 0, 0);
    invalid.per_layer = 2;
    EXPECT(throws_c10(
               [&] { SpmPipelinePlan::compile({invalid}); }),
           "per-layer expansion is rejected in V1");
    invalid = temp_region("invalid", 256, 0, 0);
    invalid.alias_of = "other";
    EXPECT(throws_c10(
               [&] { SpmPipelinePlan::compile({invalid}); }),
           "explicit alias is rejected in V1");
    invalid = temp_region("invalid", 256, 0, 0);
    invalid.preload_callback =
        [](v3::FusedModelBase&, int, uint32_t) {};
    EXPECT(throws_c10(
               [&] { SpmPipelinePlan::compile({invalid}); }),
           "preload callback is rejected in V1");
    invalid = temp_region("invalid", 256, 0, 0);
    invalid.reverse_layer_alloc = true;
    EXPECT(throws_c10(
               [&] { SpmPipelinePlan::compile({invalid}); }),
           "reverse allocation is rejected in V1");
    invalid = temp_region("invalid", 0, 0, 0);
    EXPECT(throws_c10(
               [&] { SpmPipelinePlan::compile({invalid}); }),
           "zero-sized region is rejected");
    invalid = temp_region("invalid", -1, 0, 0);
    EXPECT(throws_c10(
               [&] { SpmPipelinePlan::compile({invalid}); }),
           "negative-sized region is rejected before planning");
    invalid = temp_region("invalid", 256, -1, 0);
    EXPECT(throws_c10(
               [&] { SpmPipelinePlan::compile({invalid}); }),
           "negative first event is rejected");
    invalid = temp_region("invalid", std::numeric_limits<int64_t>::max(),
                          0, 0);
    EXPECT(throws_c10(
               [&] { SpmPipelinePlan::compile({invalid}); }),
           "oversized aligned region is rejected before legacy planner");
    invalid = temp_region("invalid", 256, 2, 1);
    EXPECT(throws_c10(
               [&] { SpmPipelinePlan::compile({invalid}); }),
           "invalid lifetime is rejected");
    EXPECT(throws_c10([&] {
               SpmPipelinePlan::compile({temp_region("dup", 256, 0, 0),
                                         temp_region("dup", 256, 1, 1)});
           }),
           "duplicate region name is rejected");

    std::printf("\n%d passed, %d failed\n", g_passed, g_failed);
    return g_failed == 0 ? 0 : 1;
}

int main() {
    const int result = run_contracts();
    // All test-local Graph, model and Tensor owners have been destroyed.
    rpu::empty_cache();
    return result;
}
