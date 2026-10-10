"""CPU-only checks of native-owned KV domain provenance (no backend import)."""

from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BASE = (ROOT / "src/core/fused_model_base.cpp").read_text()




def _body(source, name):
    match = re.search(re.escape(name) + r"\([^;{}]*\)\s*(?:const\s*)?\{", source)
    assert match, name
    start = match.end() - 1
    depth = 0
    for end in range(start, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start:end + 1]
    raise AssertionError(name)


def test_query_is_leaf_lookup_only_and_observations_publish_after_admission():
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    bindings = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    count = 0
    for path in (ROOT / "src/fused").glob("*.cpp"):
        source = path.read_text()
        for match in re.finditer(r"void rpu_(\w+)_bind_kvinsert_costs\(", source):
            prefix = match[1]
            binder = _body(source, f"rpu_{prefix}_bind_kvinsert_costs")
            query = _body(source, f"rpu_{prefix}_kvinsert_cost_domain")
            registry = re.search(r"(\w+Registry)::get", binder)[1]
            assert f'{registry}::get(handle, "rpu_{prefix}_kvinsert_cost_domain")' in query
            assert f'->kvinsert_cost_domain("{prefix}", descriptor)' in query
            assert f"rpu_{prefix}_kvinsert_cost_domain(" in decls
            assert f'{prefix}_kvinsert_cost_domain(int handle, int[] descriptor) -> (str, str, int[], int[][])' in bindings
            assert f"TORCH_FN(rpu_{prefix}_kvinsert_cost_domain)" in bindings
            mint = _body(source, f"rpu_{prefix}_kvinsert_exact_candidate")
            assert f'{registry}::get(handle, "rpu_{prefix}_kvinsert_exact_candidate")' in mint
            target = "mint_vision_kvinsert_exact_candidate" if prefix == "qwen3_5_vision" else "mint_kvinsert_exact_candidate"
            assert f"->{target}(descriptor, site_id, invocation, route)" in mint
            assert f"rpu_{prefix}_kvinsert_exact_candidate(" in decls
            assert f"TORCH_FN(rpu_{prefix}_kvinsert_exact_candidate)" in bindings
            count += 1
    assert count == 24
    query = _body(BASE, "FusedModelBase::kvinsert_cost_domain")
    assert "std::equal(candidate.descriptor.begin()" in query
    assert "decode_" not in query and "resolve_kvinsert" not in query
    assert "get_chunk_size_override" not in query
    assert "subclass_layout_hash" not in query
    assert "kv_cost_precision_identity_" in query
    chunk_setter = _body(BASE, "FusedModelBase::set_chunk_size_override")
    assert "kv_cost_candidates_.clear()" not in chunk_setter
    navdp = (ROOT / "src/fused/rpu_navdp_model.cpp").read_text()
    assert "set_chunk_size_override(saved_override)" in navdp
    oracle = _body(BASE, "FusedModelBase::resolve_prefill_stage_domain_for_shape_impl")
    assert oracle.index("begin_kvinsert_cost_domain_oracle()") < oracle.index("validate_fmb_planning_shape")
    assert oracle.index("observe_kvinsert_cost_candidates(") < oracle.index("physical_manifest_domain_for_candidate(")
    assert oracle.index("final_chunk_layout_fits(") < oracle.index("admitted.push_back(")
    assert oracle.index("!result.empty()") < oracle.index("finish_kvinsert_cost_domain_oracle(result)")
    observer = _body(BASE, "FusedModelBase::observe_kvinsert_cost_candidates")
    assert "kv_cost_candidates_.size() == 4096" in observer
    assert "1048576 - pimpl_->kv_cost_snapshot_words_" in observer
    vision = _body((ROOT / "src/fused/rpu_qwen3_5_vision_model.cpp").read_text(),
                   "Qwen3_5VisionModel::resolve_prefill_domain")
    assert vision.index("begin_kvinsert_cost_domain_oracle()") < vision.index("configure_execution_geometry(")
    assert "resolve_prefill_stage_domain_for_shape(" in vision
    assert "plan.stages.input.chunks, plan.stages.spans," in vision
    assert "for (const auto& candidate : physical_candidates)" in vision
    assert vision.index("spm_pipeline_component_layout_fits_for_cpu_contract(") < vision.index("returned_candidates.push_back(")
    assert "finish_kvinsert_cost_domain_oracle(returned_candidates)" in vision
    assert ".front()" not in vision
    auto = _body(BASE, "FusedModelBase::resolve_kvinsert_plan_auto")
    assert "rpu_kvinsert_route_arguments(plan, num_cores, num_kv_heads, head_dim)" in auto
    assert "observations.size() < 4096" in auto
    assert "kKvInsertRouteArgumentWords = 22" in (ROOT / "src/ops/rpu_kvinsert_segment_plan.h").read_text()


def test_every_competing_leaf_exports_installed_precision_not_a_weight_label():
    count = 0
    for path in (ROOT / "src/fused").glob("*.cpp"):
        source = path.read_text()
        if not re.search(r"void rpu_\w+_bind_kvinsert_costs\(", source):
            continue
        header = path.with_suffix(".h")
        if header.exists():
            source += header.read_text()
        assert "kvinsert_cost_weight_identity() const override" in source
        assert "append_kvinsert_cost_tensor_identity(identity," in source
        if "public CausalDecoderModel" in source:
            assert "auto identity = causal_kvinsert_cost_weight_identity()" in source
        count += 1
    assert count == 24
    tensor = _body(BASE, "FusedModelBase::append_kvinsert_cost_tensor_identity")
    for field in ("defined()", "scalar_type()", "element_size()", "sizes()", "strides()"):
        assert field in tensor
    assert "data_ptr" not in tensor and "device()" not in tensor
    hyvit = (ROOT / "src/fused/rpu_hyvit2_vision_model.cpp").read_text()
    # q_inplace_ is a hot geometry-derived result; only its actual cold opt-in
    # belongs in profile identity. Exact physical effects are in the descriptor.
    start = hyvit.index("kvinsert_cost_weight_identity() const override")
    end = hyvit.index("return identity;", start)
    assert "static_cast<int64_t>(q_inplace_)" not in hyvit[start:end]
    assert "cold_config_.q_inplace" in hyvit[start:end]


def test_fixed_v2_gemma_owners_export_actual_lookup_and_precision_without_binders():
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    bindings = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    for prefix, registry in (
        ("gemma", "GemmaRegistry"),
        ("gemma2", "Gemma2Registry"),
        ("gemma4", "Gemma4Registry"),
    ):
        source = (ROOT / f"src/fused/rpu_{prefix}_model.cpp").read_text()
        query = _body(source, f"rpu_{prefix}_kvinsert_cost_domain")
        getter = _body(source, f"rpu_{prefix}_kvinsert_cost_catalog_sha256")
        assert f'{registry}::get(handle, "rpu_{prefix}_kvinsert_cost_domain")' in query
        assert f'->kvinsert_cost_domain("{prefix}", descriptor)' in query
        assert f'{registry}::get(handle, "rpu_{prefix}_kvinsert_cost_catalog_sha256")' in getter
        assert "->kvinsert_cost_catalog_sha256()" in getter
        for suffix in ("kvinsert_cost_domain", "kvinsert_cost_catalog_sha256"):
            assert f"rpu_{prefix}_{suffix}(" in decls
            assert f"TORCH_FN(rpu_{prefix}_{suffix})" in bindings
        assert f"rpu_{prefix}_bind_kvinsert_costs" not in source + decls + bindings
        assert "resolve_prefill_stage_domain_for_shape(" in source
        assert "KV_INSERT_CAP_V2);" in source
        start = source.index("kvinsert_cost_weight_identity() const override")
        precision = source[start:source.index("return identity;", start)]
        assert "layer_weights_.empty()" in precision
        assert "append_kvinsert_cost_tensor_identity(identity, *tensor)" in precision
        assert "&weights.q_w" in precision and "&weights.v_w" in precision
        assert "append_kvinsert_cost_scalar_identity(identity, eps_)" in precision
        assert "get_chunk_size_override" not in precision
        assert "seq_len_" not in precision
        assert "subclass_layout_hash" not in precision
        if prefix == "gemma":
            assert "&weights.q_ws" in precision and "&weights.down_ws" in precision
        elif prefix == "gemma2":
            assert "layer_type_ids_" in precision and "query_pre_attn_scalar_" in precision
        else:
            assert "geometry.head_dim" in precision and "geometry.num_kv_heads" in precision
            assert "geometry.kv_source_layer" in precision and "weights.layer_scalar" in precision


def test_exact_diagnostic_mint_revalidates_layout_before_publishing():
    body = _body(BASE, "FusedModelBase::mint_kvinsert_exact_candidate")
    assert body.index("kvinsert_cost_domain(") < body.index("decode_fmb_prefill_stage_candidate(")
    assert "(*row)[10] & (int64_t{1} << route_id)" in body
    assert "static_cast<uint32_t>((*row)[9])" in body
    assert "KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION" in body
    assert "rpu_resolve_kvinsert_segment_plan(" in body
    assert "rpu_resolve_kvinsert_segment_plan_auto(" not in body
    assert "kv_cost_planning_context_" in body and "subclass_layout_identity" in body
    assert body.index("resolve_spm_pipeline_component_layout_for_cpu_contract(") < body.index(
        "kv_cost_candidates_.push_back(metadata)")
    assert "kv_cost_candidates_.size() < 4096" in body
    assert "1048576 - pimpl_->kv_cost_snapshot_words_" in body
    assert "static_cast<int64_t>(layout.layout_hash >> 32)" in body
    source = (ROOT / "src/fused/rpu_qwen3_model.cpp").read_text()
    wrapper = _body(source, "rpu_causal_decoder_kvinsert_exact_candidate")
    assert 'CausalDecoderRegistry::get(handle, "rpu_causal_decoder_kvinsert_exact_candidate")' in wrapper
    assert "->mint_kvinsert_exact_candidate(descriptor, site_id, invocation, route)" in wrapper
    decls = (ROOT / "src/core/rpu_kernel_decls.h").read_text()
    bindings = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text()
    assert "rpu_causal_decoder_kvinsert_exact_candidate(" in decls
    assert "causal_decoder_kvinsert_exact_candidate(int handle, int[] descriptor, int site_id, int invocation, int route) -> (int[], int, int)" in bindings


def test_actual_temporary_geometry_and_custom_prefix_are_native_owned():
    vision = (ROOT / "src/fused/rpu_qwen3_5_vision_model.cpp").read_text()
    mint = _body(vision, "Qwen3_5VisionModel::mint_vision_kvinsert_exact_candidate")
    assert mint.index("kvinsert_cost_owner_prefix(descriptor)") < mint.index("mint_kvinsert_exact_candidate(descriptor")
    assert "descriptor[" not in mint
    assert "prefix[9] = std::get<1>(minted)" in mint and "prefix[10] = std::get<2>(minted)" in mint
    assert "configure_execution_geometry" not in mint  # caller labels cannot set geometry
    assert "bind_kvinsert_cost_owner_prefix(stage_descriptor, descriptor)" in vision
    for name in ("set_chunk_envelope", "bind_kvinsert_costs", "invalidate_model_state",
                 "begin_kvinsert_cost_domain_oracle", "observe_kvinsert_cost_candidates"):
        body = _body(BASE, "FusedModelBase::" + name)
        assert "kv_cost_layout_scope_ = {}" in body
    header = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    context = _body(BASE, "FusedModelBase::remember_kvinsert_cost_planning_context")
    for borrowed in ("hidden_states", "k_caches", "v_caches"):
        assert f"kv_cost_planning_context_->{borrowed} = nullptr" in context
    producer = _body(header, "encode_causal_physical_stage_descriptor")
    assert producer.index("observe_kvinsert_cost_candidates(") < producer.index("physical_manifest_for_candidate(")
    assert "finish_kvinsert_cost_domain_oracle(candidates)" in producer
    getter = _body(header, "multiview_text_stage_descriptor")
    assert "qwen3vl_multiview_text_composite_inputs_primed_" in getter
    assert "kvinsert_cost_domain(\"causal_decoder\", qwen3vl_multiview_text_stage_descriptor_)" in getter
    assert "return qwen3vl_multiview_text_stage_descriptor_" in getter
    assert "resolve_" not in getter


def test_actual_cpp_domain_rows_profile_and_stale_lookup(tmp_path):
    compiler = shutil.which("g++")
    assert compiler
    implementation = (ROOT / "src/core/fused_model_base_impl.h").read_text()
    structs = implementation[implementation.index("    struct KvCostObservation {"):
                             implementation.index("    // A scoped pointer", implementation.index("    struct KvCostObservation {"))]
    harness = r'''
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#define TORCH_CHECK(c, ...) do { if (!(c)) throw std::runtime_error("rejected"); } while(0)
namespace at {
struct IntArrayRef: std::vector<int64_t> {
    using std::vector<int64_t>::vector;
    IntArrayRef(const std::vector<int64_t>& value):std::vector<int64_t>(value){}
    IntArrayRef slice(size_t start) const {return {begin()+start,end()};}
};
struct Tensor {
    bool valid = true; int64_t dtype = 5, bytes = 2;
    std::vector<int64_t> shape{64,128}, stride{128,1};
    bool defined() const { return valid; }
    int64_t scalar_type() const { return dtype; }
    int64_t element_size() const { return bytes; }
    int64_t dim() const { return shape.size(); }
    const auto& sizes() const { return shape; }
    const auto& strides() const { return stride; }
};
}
using KvInsertRouteArguments = std::array<int64_t,22>;
constexpr size_t kKvInsertRouteArgumentWords=22;
constexpr int64_t KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION=1LL<<16;
using KvInsertCostDomainQuery = std::tuple<std::string,std::string,
    std::vector<int64_t>,std::vector<std::vector<int64_t>>>;
enum class FmbPhysicalManifestState { INCOMPLETE, COMPLETE };
enum class FmbRouteFamily { KV_INSERT, OTHER };
enum class FmbGraphLifecycle { RETAINED_CACHE=1, BOUNDED_ONESHOT=2, COMPOSITE_CHILD=4 };
struct Route {
    FmbRouteFamily family=FmbRouteFamily::KV_INSERT;
    int64_t site_id=11, invocation=0, flags=0, selector=1;
    std::vector<int64_t> arguments;
};
struct Manifest {
    FmbPhysicalManifestState state=FmbPhysicalManifestState::COMPLETE;
    FmbGraphLifecycle graph_lifecycle=FmbGraphLifecycle::RETAINED_CACHE;
    int64_t physical_length=32;
    std::vector<Route> routes;
};
using FmbPhysicalExecutionManifest = Manifest;
enum class FmbForwardOperandResidency { UNSPECIFIED, PER_LAYER, FORWARD };
struct LayoutContext {
    uint64_t physical_manifest_fingerprint=0;
    int64_t rope_table_residency=0, attention_policy=0;
    FmbForwardOperandResidency forward_operand_residency=FmbForwardOperandResidency::UNSPECIFIED;
};
struct InferenceContext {
    int64_t position=0;
    void cancel_external_physical_manifest_prologue() {}
    void bind_physical_manifest(const Manifest&) {}
};
struct FmbPrefillStageCandidate { std::vector<int64_t> descriptor{101,102}; Manifest physical_manifest; int stage_plan=1; int64_t qkv_chunk_size=32; };
std::map<std::vector<int64_t>, FmbPrefillStageCandidate> encoded;
std::vector<int64_t> encode_fmb_prefill_stage_candidate(const FmbPrefillStageCandidate& c) {
    auto wire=c.descriptor;
    for(const auto& route:c.physical_manifest.routes) {
        if(route.arguments.size()==22 && route.arguments[17]==1) {
            wire.push_back(route.site_id);wire.push_back(route.selector);wire.push_back(1);
        }
    }
    encoded[wire]=c;return wire;
}
FmbPrefillStageCandidate decode_fmb_prefill_stage_candidate(at::IntArrayRef wire) { return encoded.at(wire); }
uint64_t fmb_physical_manifest_fingerprint(const Manifest& manifest) {
    uint64_t hash=123;
    for(const auto& route:manifest.routes) hash=hash*31+route.selector;
    return hash;
}
int64_t fmb_rope_table_residency(const Manifest&) { return 0; }
struct FmbPhysicalManifestConsumer {
    explicit FmbPhysicalManifestConsumer(const Manifest&) {}
    std::optional<int64_t> attention_layout_policy() const { return 0; }
};
void validate_fmb_physical_manifest(const Manifest& manifest,int64_t length,int64_t position) {
    assert(manifest.physical_length==length && position==64);
}
void validate_fmb_physical_manifest_forward_capability(const Manifest&, bool capability) { assert(capability); }
enum class KvInsertRoute { V2=1, ALIGNED_V16=2 };
struct KvInsertSegmentPlan {
    KvInsertRoute selected; KvInsertRouteArguments arguments{};
    KvInsertRoute route() const {return selected;}
};
KvInsertSegmentPlan rpu_resolve_kvinsert_segment_plan(
    int64_t position,int64_t logical,int64_t physical,int cores,
    int64_t heads,int64_t dim,uint32_t caps,KvInsertRoute route) {
    assert(position==64 && logical==32 && physical==32 && cores==8 && heads==16 && dim==128 && caps>0);
    KvInsertSegmentPlan plan{route};
    plan.arguments[0]=2;plan.arguments[1]=static_cast<int64_t>(route);
    plan.arguments[2]=logical;plan.arguments[3]=physical;plan.arguments[4]=1;
    plan.arguments[5]=1;plan.arguments[6]=position;plan.arguments[8]=logical;
    plan.arguments[17]=1;plan.arguments[18]=int64_t{1}<<static_cast<int64_t>(route);
    return plan;
}
KvInsertRouteArguments rpu_kvinsert_route_arguments(const KvInsertSegmentPlan& plan,int cores,int64_t heads,int64_t dim) {
    assert(cores==8 && heads==16 && dim==128); return plan.arguments;
}
struct RpuKernelGraph { static bool has_active() { return false; } };
struct RpuExecutionCleanupGuard { explicit RpuExecutionCleanupGuard(const char*) {} };
namespace c10 {
struct TestScopeExit { std::function<void()> action; ~TestScopeExit() { action(); } };
template<class F> TestScopeExit make_scope_exit(F action) { return {action}; }
}
struct FusedModelBase {
    using KvCostLayoutScope=std::function<void(const std::function<void()>&)>;
    template<typename... Fields>
    static KvCostLayoutScope capture_kvinsert_cost_layout_fields(Fields&... fields) __CAPTURE_SCOPE__
    struct Impl {
        __STRUCTS__
        std::vector<KvCostCandidate> kv_cost_candidates_;
        std::optional<InferenceContext> kv_cost_planning_context_;
        KvCostLayoutScope kv_cost_layout_scope_;
        InferenceContext ctx_;
        std::vector<int64_t> kv_cost_precision_identity_;
        std::vector<KvCostObservation>* kv_cost_observations_=nullptr;
        size_t kv_cost_snapshot_words_=0;
        uint64_t kv_cost_model_state_gen_=1, model_state_gen_=1;
        std::string kv_cost_snapshot_reason_="READY";
    };
    std::unique_ptr<Impl> pimpl_=std::make_unique<Impl>();
    struct Envelope { int64_t max_kv_len=8192, chunk=640; } envelope;
    at::Tensor weight;
    int64_t hidden=2048, cold_policy=0, hot_chunk=32;
    bool precision_known=true;
    int64_t subclass_hash=0; int layout_calls=0; bool spm_fits=true;
    int64_t temporary_geometry=7;
    int64_t subclass_layout_hash() const { return subclass_hash; }
    bool physical_manifest_forward_capability(const Manifest&) const { return true; }
    FmbForwardOperandResidency physical_forward_operand_residency(const Manifest&) const {
        return FmbForwardOperandResidency::UNSPECIFIED;
    }
    void rebind_kvinsert_exact_candidate(
        FmbPrefillStageCandidate&, const LayoutContext&, int64_t, int64_t,
        const KvInsertSegmentPlan&) const __REBIND__
    LayoutContext bind_physical_layout_context(LayoutContext, const Manifest&) const;
    struct Layout {uint64_t layout_hash;};
    Layout resolve_spm_pipeline_component_layout_for_cpu_contract(const LayoutContext& layout,int stage) {
        ++layout_calls;
        assert(temporary_geometry==7);
        TORCH_CHECK(spm_fits,"SPM rejected");
        assert(pimpl_->ctx_.position==64 && stage==1 && layout.physical_manifest_fingerprint!=0);
        return {0xabcdef1200000000ULL ^ layout.physical_manifest_fingerprint};
    }
    int64_t num_layers() const { return 28; }
    int64_t hidden_size() const { return hidden; }
    int64_t intermediate_size() const { return 6144; }
    int64_t num_q_heads() const { return 16; }
    int64_t num_kv_heads() const { return 8; }
    int64_t head_dim() const { return 128; }
    int64_t attn_tp() const { return 8; }
    const Envelope& chunk_envelope() const { return envelope; }
    static void append_kvinsert_cost_tensor_identity(std::vector<int64_t>&, const at::Tensor&);
    std::vector<int64_t> kvinsert_cost_weight_identity() const {
        if (!precision_known) return {};
        std::vector<int64_t> result{cold_policy};
        append_kvinsert_cost_tensor_identity(result,weight); return result;
    }
    KvInsertCostDomainQuery kvinsert_cost_domain(const char*,at::IntArrayRef) const;
    std::vector<int64_t> installed_profile_identity() const;
    void begin_kvinsert_cost_domain_oracle();
    std::tuple<std::vector<int64_t>,int64_t,int64_t>
        mint_kvinsert_exact_candidate(at::IntArrayRef,int64_t,int64_t,int64_t);
    std::vector<FmbPrefillStageCandidate> observe_kvinsert_cost_candidates(
        const std::function<std::vector<FmbPrefillStageCandidate>()>& producer,
        const LayoutContext* actual_layout=nullptr);
    void finish_kvinsert_cost_domain_oracle(const std::vector<FmbPrefillStageCandidate>& returned_candidates);
    void bind_kvinsert_cost_owner_prefix(at::IntArrayRef,at::IntArrayRef);
    std::vector<int64_t> kvinsert_cost_owner_prefix(at::IntArrayRef) const;
};
struct Qwen3_5VisionModel: FusedModelBase {
    std::tuple<std::vector<int64_t>,int64_t,int64_t> mint_vision_kvinsert_exact_candidate(
        at::IntArrayRef,int64_t,int64_t,int64_t);
};
void FusedModelBase::append_kvinsert_cost_tensor_identity(std::vector<int64_t>& identity,const at::Tensor& tensor) __TENSOR__
LayoutContext FusedModelBase::bind_physical_layout_context(LayoutContext layout,const Manifest& manifest) const __BIND_LAYOUT__
std::vector<int64_t> FusedModelBase::installed_profile_identity() const __PROFILE__
KvInsertCostDomainQuery FusedModelBase::kvinsert_cost_domain(const char* owner_kind,at::IntArrayRef descriptor) const __QUERY__
static FusedModelBase::Impl::KvCostCandidate make_kvinsert_cost_candidate(
    const FmbPrefillStageCandidate& candidate,
    const std::vector<FusedModelBase::Impl::KvCostObservation>& observations) __CANDIDATE__
void FusedModelBase::begin_kvinsert_cost_domain_oracle() __BEGIN__
std::vector<FmbPrefillStageCandidate> FusedModelBase::observe_kvinsert_cost_candidates(
    const std::function<std::vector<FmbPrefillStageCandidate>()>& producer,
    const LayoutContext* actual_layout) __OBSERVE__
void FusedModelBase::finish_kvinsert_cost_domain_oracle(
    const std::vector<FmbPrefillStageCandidate>& returned_candidates) __FINISH__
std::tuple<std::vector<int64_t>,int64_t,int64_t> FusedModelBase::mint_kvinsert_exact_candidate(
    at::IntArrayRef descriptor,int64_t site_id,int64_t invocation,int64_t route_id) __MINT__
void FusedModelBase::bind_kvinsert_cost_owner_prefix(at::IntArrayRef descriptor,at::IntArrayRef native_prefix) __BIND_PREFIX__
std::vector<int64_t> FusedModelBase::kvinsert_cost_owner_prefix(at::IntArrayRef descriptor) const __GET_PREFIX__
std::tuple<std::vector<int64_t>,int64_t,int64_t> Qwen3_5VisionModel::mint_vision_kvinsert_exact_candidate(
    at::IntArrayRef descriptor,int64_t site_id,int64_t invocation,int64_t route) __VISION_MINT__
template<class F> void rejects(F action) { bool failed=false;try{action();}catch(const std::runtime_error&){failed=true;}assert(failed); }
__HY_CONSTANTS__
FmbPrefillStageCandidate hy_candidate(bool vision, bool padding) {
    // Only native/SDK edges are doubles. The two append statements below are
    // extracted verbatim from each real HY physical_manifest_for_candidate.
    FmbPrefillStageCandidate candidate;
    auto& manifest=candidate.physical_manifest;
    manifest.graph_lifecycle=FmbGraphLifecycle::COMPOSITE_CHILD;
    const int64_t invocation=1, tp=8, nq=16, nkv=16, hd=128;
    struct {int64_t len=18;} kv_chunk;
    struct {bool kvpad16;} cold_config_{padding};
    const bool cold_kvpad16_=padding;
    const bool cold_kv_spm_only_=false;
    const int64_t kv_storage_rows=padding?candidate.qkv_chunk_size:0;
    const int64_t kv_physical_rows=padding?kv_storage_rows:kv_chunk.len;
    KvInsertSegmentPlan kv_plan{KvInsertRoute::V2};
    auto& kv_arguments=kv_plan.arguments;
    kv_arguments[0]=2;kv_arguments[1]=1;kv_arguments[2]=kv_chunk.len;
    kv_arguments[3]=kv_physical_rows;kv_arguments[4]=1;
    kv_arguments[5]=1;kv_arguments[6]=64;kv_arguments[8]=kv_physical_rows;
    kv_arguments[17]=2;kv_arguments[18]=padding?10:18;
    const auto append=[&](FmbRouteFamily family,int64_t site,int64_t selector,
                          std::vector<int64_t> arguments={},int64_t flags=0,
                          int64_t ordinal=-1) {
        manifest.routes.push_back({family,site,ordinal<0?invocation:ordinal,flags,selector,std::move(arguments)});
    };
    if(vision) { __HYVIT2_APPEND__ }
    else { __HYEXPERT_APPEND__ }
    return candidate;
}
void test_hy_auxiliary_producers() {
    using Observation=FusedModelBase::Impl::KvCostObservation;
    for(bool vision:{true,false}) for(bool padding:{true,false}) {
        const auto candidate=hy_candidate(vision,padding);
        const auto& kv=candidate.physical_manifest.routes[1];
        Observation observed{kv.site_id,4,8,16,128,31,{}};
        std::copy(kv.arguments.begin(),kv.arguments.end(),observed.arguments.begin());
        auto query=[&](const FmbPrefillStageCandidate& value,const std::vector<Observation>& facts) {
            FusedModelBase owner;
            owner.begin_kvinsert_cost_domain_oracle();
            owner.observe_kvinsert_cost_candidates([&] {
                *owner.pimpl_->kv_cost_observations_=facts;
                return std::vector<FmbPrefillStageCandidate>{value};
            });
            owner.finish_kvinsert_cost_domain_oracle({value});
            return owner.kvinsert_cost_domain(vision?"hyvit2":"hyvla_expert",encode_fmb_prefill_stage_candidate(value));
        };
        const auto good=query(candidate,{observed,observed});
        assert(std::get<1>(good)=="READY" && std::get<3>(good).size()==1);
        assert(std::get<3>(good)[0][0]==kv.site_id && std::get<3>(good)[0][2]==1);
        auto reversed=candidate;
        std::reverse(reversed.physical_manifest.routes.begin(),reversed.physical_manifest.routes.end());
        assert(std::get<3>(query(reversed,{observed}))==std::get<3>(good));
        // Even a non-competing actual KV counterpart must prove auxiliary
        // geometry, without emitting a spurious cost row for the auxiliary.
        auto singleton=candidate;
        auto fixed=observed;
        singleton.physical_manifest.routes[1].arguments[17]=fixed.arguments[17]=1;
        singleton.physical_manifest.routes[1].arguments[18]=fixed.arguments[18]=2;
        auto one=query(singleton,{fixed});
        assert(std::get<1>(one)=="NO_COMPETING_KV_DOMAIN" && std::get<3>(one).empty());
        for(int fault=0;fault<21;++fault) {
            auto bad=candidate;std::vector<Observation> facts{observed};
            auto& aux=bad.physical_manifest.routes[0];
            auto& counter=bad.physical_manifest.routes[1];
            switch(fault) {
                case 0: aux.site_id=99;break;
                case 1: counter.site_id=99;break;
                case 2: ++counter.invocation;break;
                case 3: bad.physical_manifest.routes.pop_back();break;
                case 4: bad.physical_manifest.routes.push_back(counter);break;
                case 5: aux.flags=KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION;break;
                case 6: counter.flags|=KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION;break;
                case 7: ++counter.flags;break;
                case 8: ++aux.selector;break;
                case 9: ++counter.selector;break;
                case 10: aux.arguments[4]=4;break;
                case 11: aux.arguments[5]=8;break;
                case 12: aux.arguments[6]=64;break;
                case 13: ++aux.arguments[2];break;
                case 14: ++aux.arguments[3];break;
                case 15: bad.qkv_chunk_size=16;break;
                case 16: bad.physical_manifest.graph_lifecycle=FmbGraphLifecycle::RETAINED_CACHE;break;
                case 17: facts.clear();break;
                case 18: ++facts[0].arguments[6];break;
                case 19: facts[0].head_dim=64;break;
                case 20: aux.site_id=99;aux.flags=KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION;break;
            }
            const auto rejected=query(bad,facts);
            assert(std::get<1>(rejected)=="NATIVE_KV_DOMAIN_PRODUCER_UNAVAILABLE");
            assert(std::get<3>(rejected).empty());
        }
        auto conflict=observed;conflict.capabilities=7;
        auto ambiguous=query(candidate,{observed,conflict});
        assert(std::get<1>(ambiguous)=="NATIVE_KV_DOMAIN_AMBIGUOUS" && std::get<3>(ambiguous).empty());
        if(padding) {
            auto wrong_capacity=candidate;wrong_capacity.qkv_chunk_size=64;
            assert(std::get<1>(query(wrong_capacity,{observed}))=="NATIVE_KV_DOMAIN_PRODUCER_UNAVAILABLE");
        }
    }
}
int main() {
    test_hy_auxiliary_producers();
    KvInsertRouteArguments args{};
    args[0]=2;args[2]=32;args[3]=32;args[6]=64;args[17]=2;args[18]=3;
    Route route;route.arguments.assign(args.begin(),args.end());
    FmbPrefillStageCandidate candidate;candidate.physical_manifest.routes={route};
    using Observation=FusedModelBase::Impl::KvCostObservation;
    Observation spatial{11,1,8,16,128,7,args};
    auto metadata=make_kvinsert_cost_candidate(candidate,{spatial});
    assert(metadata.reason=="READY");
    assert((metadata.rows[0]==std::vector<int64_t>{11,1,0,64,32,32,8,16,128,7,3,2}));
    // Same 22word arguments do not encode heads: producer facts stay per site.
    auto temporal=spatial;temporal.site_id=12;temporal.num_kv_heads=64;
    auto temporal_route=route;temporal_route.site_id=12;
    candidate.physical_manifest.routes.push_back(temporal_route);
    auto both=make_kvinsert_cost_candidate(candidate,{spatial,temporal});
    assert(both.reason=="READY" && both.rows[0][7]==16 && both.rows[1][7]==64);
    auto wrong=spatial;wrong.num_kv_heads=32;
    assert(make_kvinsert_cost_candidate(candidate,{spatial,wrong,temporal}).reason=="NATIVE_KV_DOMAIN_AMBIGUOUS");
    wrong=spatial;wrong.graph_lifecycle=2;
    assert(make_kvinsert_cost_candidate(candidate,{wrong,temporal}).rows.empty());
    wrong=spatial;++wrong.arguments[3];
    assert(make_kvinsert_cost_candidate(candidate,{wrong,temporal}).rows.empty());
    candidate.physical_manifest.routes={route};
    candidate.physical_manifest.state=FmbPhysicalManifestState::INCOMPLETE;
    assert(make_kvinsert_cost_candidate(candidate,{spatial}).reason=="INCOMPLETE_PHYSICAL_MANIFEST");
    candidate.physical_manifest.state=FmbPhysicalManifestState::COMPLETE;
    candidate.physical_manifest.routes[0].flags=KV_INSERT_ROUTE_FLAG_DYNAMIC_POSITION;
    assert(make_kvinsert_cost_candidate(candidate,{spatial}).rows.empty());
    candidate.physical_manifest.routes[0].flags=0;
    candidate.physical_manifest.routes[0].arguments[18]=1;
    assert(make_kvinsert_cost_candidate(candidate,{spatial}).reason=="NO_COMPETING_KV_DOMAIN");
    Qwen3_5VisionModel model;
    model.pimpl_->kv_cost_candidates_={metadata};
    model.pimpl_->kv_cost_precision_identity_=model.kvinsert_cost_weight_identity();
    auto query=[&](){return model.kvinsert_cost_domain("actual_leaf",metadata.descriptor);};
    const auto profile=std::get<2>(query());
    assert(model.installed_profile_identity()==profile);
    model.hot_chunk=64;
    assert(std::get<2>(query())==profile);
    // The current profile has no model generation or caller-selected chunk.
    ++model.pimpl_->model_state_gen_;rejects(query);
    model.pimpl_->kv_cost_model_state_gen_=model.pimpl_->model_state_gen_;
    assert(std::get<2>(query())==profile);
    rejects([&](){model.kvinsert_cost_domain("actual_leaf",{101,103});});
    model.cold_policy=1;rejects(query);
    model.pimpl_->kv_cost_precision_identity_=model.kvinsert_cost_weight_identity();
    assert(std::get<2>(query())!=profile);
    model.cold_policy=0;model.weight.dtype=1;model.weight.bytes=1;rejects(query);
    model.pimpl_->kv_cost_precision_identity_=model.kvinsert_cost_weight_identity();
    assert(std::get<2>(query())!=profile);
    model.weight.dtype=5;model.weight.bytes=2;model.hidden=1024;
    model.pimpl_->kv_cost_precision_identity_=model.kvinsert_cost_weight_identity();
    assert(std::get<2>(query())!=profile);
    model.pimpl_->kv_cost_snapshot_reason_="NATIVE_STAGE_ORACLE_FAILED";rejects(query);
    // The special Vision oracle uses exactly the same scoped observer and
    // bounded publication as generic FMB, retaining only returned/admitted IDs.
    candidate.physical_manifest.routes={route};
    auto not_returned=candidate;not_returned.descriptor={201,202};
    model.begin_kvinsert_cost_domain_oracle();
    auto admitted=model.observe_kvinsert_cost_candidates([&] {
        model.pimpl_->kv_cost_observations_->push_back(spatial);
        return std::vector<FmbPrefillStageCandidate>{candidate,not_returned};
    });
    assert(model.pimpl_->kv_cost_observations_==nullptr);rejects(query);
    model.finish_kvinsert_cost_domain_oracle({candidate});
    assert(std::get<1>(query())=="READY");
    rejects([&](){model.kvinsert_cost_domain("actual_leaf",not_returned.descriptor);});
    // Rejected SPM candidates are not returned by the actual producer.
    model.begin_kvinsert_cost_domain_oracle();
    model.observe_kvinsert_cost_candidates([&] {
        model.pimpl_->kv_cost_observations_->push_back(spatial);
        return std::vector<FmbPrefillStageCandidate>{};
    });
    model.finish_kvinsert_cost_domain_oracle({});rejects(query);
    model.begin_kvinsert_cost_domain_oracle();
    rejects([&](){model.observe_kvinsert_cost_candidates([&]() -> std::vector<FmbPrefillStageCandidate> {
        model.pimpl_->kv_cost_observations_->push_back(spatial);
        throw std::runtime_error("producer failed");
    });});
    assert(model.pimpl_->kv_cost_observations_==nullptr);
    assert(model.pimpl_->kv_cost_candidates_.empty());rejects(query);
    // Fixed-V2 Gemma variants use this same actual admitted-manifest path.
    // They need neither a competing observation nor an invented parent catalog.
    candidate.physical_manifest.routes[0].arguments[18]=1;
    model.begin_kvinsert_cost_domain_oracle();
    model.observe_kvinsert_cost_candidates([&] {
        return std::vector<FmbPrefillStageCandidate>{candidate};
    });
    model.finish_kvinsert_cost_domain_oracle({candidate});
    auto fixed=model.kvinsert_cost_domain("gemma",candidate.descriptor);
    assert(std::get<0>(fixed)=="gemma");
    assert(std::get<1>(fixed)=="NO_COMPETING_KV_DOMAIN");
    assert(std::get<3>(fixed).empty());
    assert(std::get<2>(fixed).size()>11);
    model.begin_kvinsert_cost_domain_oracle();
    rejects([&](){model.kvinsert_cost_domain("gemma",candidate.descriptor);});
    // Actual mint method: latest native geometry/mask, not requested wire or
    // a forged low-cost certificate, owns the exact route and its layout.
    args[18]=6;spatial.arguments=args;
    route.arguments.assign(args.begin(),args.end());
    candidate.physical_manifest.routes={route};
    LayoutContext actual_layout;
    model.pimpl_->ctx_.position=999;
    model.pimpl_->kv_cost_planning_context_=InferenceContext{64};
    model.pimpl_->kv_cost_layout_scope_=model.capture_kvinsert_cost_layout_fields(model.temporary_geometry);
    model.temporary_geometry=99;
    model.observe_kvinsert_cost_candidates([&] {
        model.pimpl_->kv_cost_observations_->push_back(spatial);
        return std::vector<FmbPrefillStageCandidate>{candidate};
    }, &actual_layout);
    model.finish_kvinsert_cost_domain_oracle({candidate});
    auto mint=[&](){return model.mint_kvinsert_exact_candidate(candidate.descriptor,11,0,2);};
    auto minted=mint();
    assert(model.layout_calls==1 && model.pimpl_->ctx_.position==999 && model.temporary_geometry==99);
    assert(std::get<1>(minted)==0xabcdef12 && std::get<2>(minted)!=0);
    auto exact=decode_fmb_prefill_stage_candidate(std::get<0>(minted));
    assert(exact.physical_manifest.routes[0].selector==2);
    const auto& exactargs=exact.physical_manifest.routes[0].arguments;
    assert(exactargs[17]==1 && exactargs[18]==4 && exactargs[21]==0);
    assert(std::get<1>(model.kvinsert_cost_domain("leaf",std::get<0>(minted)))=="NO_COMPETING_KV_DOMAIN");
    assert(std::get<1>(model.kvinsert_cost_domain("leaf",candidate.descriptor))=="READY");
    assert(mint()==minted && model.pimpl_->kv_cost_candidates_.size()==2);
    rejects([&](){model.mint_kvinsert_exact_candidate(candidate.descriptor,11,0,3);});
    rejects([&](){model.mint_kvinsert_exact_candidate(candidate.descriptor,99,0,2);});
    rejects([&](){model.mint_kvinsert_exact_candidate(candidate.descriptor,11,1,2);});
    model.subclass_hash=1;rejects(mint);model.subclass_hash=0;
    model.spm_fits=false;rejects(mint);model.spm_fits=true;
    assert(model.temporary_geometry==99);
    assert(model.pimpl_->ctx_.position==999 && model.pimpl_->kv_cost_candidates_.size()==2);
    std::vector<int64_t> prefix(19,0);
    prefix[0]=3;prefix[1]=32;prefix[11]=1;prefix[14]=4;prefix[15]=3;prefix[18]=32;
    model.bind_kvinsert_cost_owner_prefix(candidate.descriptor,prefix);
    auto vision=model.mint_vision_kvinsert_exact_candidate(candidate.descriptor,11,0,2);
    auto exact_prefix=model.kvinsert_cost_owner_prefix(std::get<0>(vision));
    assert(exact_prefix[9]==std::get<1>(vision));
    assert(exact_prefix[10]==std::get<2>(vision));
    assert(model.kvinsert_cost_owner_prefix(candidate.descriptor)==prefix);
    rejects([&] {model.mint_vision_kvinsert_exact_candidate({2,3,4},11,0,2);});
    ++model.cold_policy;rejects(mint);--model.cold_policy;
    model.precision_known=false;model.pimpl_->kv_cost_precision_identity_.clear();
    rejects(mint);model.precision_known=true;
    model.begin_kvinsert_cost_domain_oracle();rejects(mint);
    assert(!model.pimpl_->kv_cost_layout_scope_);
    // Same actual typed helper for a second owner geometry, including its
    // bounded native vector. Saved values survive, never borrowed stack data.
    int64_t frames=4,cameras=3;std::vector<int64_t> windows{0,16,32};
    auto scope=model.capture_kvinsert_cost_layout_fields(frames,cameras,windows);
    frames=1;cameras=1;windows={0,64};
    rejects([&] { scope([&] {
        assert(frames==4 && cameras==3 && windows==std::vector<int64_t>({0,16,32}));
        throw std::runtime_error("SPM failed");
    }); });
    assert(frames==1 && cameras==1 && windows==std::vector<int64_t>({0,64}));
}
'''
    constants = []
    for marker, path, prefix, enum, flags in (
        ("__HYVIT2_APPEND__", "src/fused/rpu_hyvit2_vision_model.cpp", "HYVIT2", "HyViT2KvPaddingRoute", "HYVIT2_KV_FLAG_DDR_MIRROR"),
        ("__HYEXPERT_APPEND__", "src/fused/rpu_hyvla_expert_model.cpp", "HYVLA_EXPERT", "HyVlaExpertKvPaddingRoute", "HYVLA_EXPERT_KV_REASON_PREFIX_HISTORY_DDR_REQUIRED"),
    ):
        source = (ROOT / path).read_text()
        for name in (prefix + "_KV_PADDING_SITE", prefix + "_KV_SITE", flags):
            constants.append(re.search(r"constexpr int64_t " + name + r"\s*=\s*\d+(?:LL)?;", source)[0])
        if prefix == "HYVIT2":
            constants.append(re.search(r"constexpr int64_t HYVIT2_KV_FLAG_RAW_SPM_ONLY\s*=\s*\d+(?:LL)?;", source)[0])
        constants.append(re.search(r"enum class " + enum + r"[^;]+?};", source, re.S)[0])
        appends = [re.search(r"append\(\s*FmbRouteFamily::KV_INSERT, " + name + r",.*?;", source, re.S)[0]
                   for name in (prefix + "_KV_PADDING_SITE", prefix + "_KV_SITE")]
        harness = harness.replace(marker, "\n".join(appends))
    harness = harness.replace("__HY_CONSTANTS__", "\n".join(constants))
    harness = harness.replace("__STRUCTS__", structs)
    harness = harness.replace("__CAPTURE_SCOPE__", _body(
        (ROOT / "src/core/fused_model_base.h").read_text(), "capture_kvinsert_cost_layout_fields"))
    harness = harness.replace("__REBIND__", _body(
        (ROOT / "src/core/fused_model_base.h").read_text(), "rebind_kvinsert_exact_candidate"))
    harness = harness.replace("__VISION_MINT__", _body(
        (ROOT / "src/fused/rpu_qwen3_5_vision_model.cpp").read_text(),
        "Qwen3_5VisionModel::mint_vision_kvinsert_exact_candidate"))
    for marker, name in (("__TENSOR__", "FusedModelBase::append_kvinsert_cost_tensor_identity"),
                         ("__BIND_LAYOUT__", "FusedModelBase::bind_physical_layout_context"),
                         ("__PROFILE__", "FusedModelBase::installed_profile_identity"),
                         ("__QUERY__", "FusedModelBase::kvinsert_cost_domain"),
                         ("__CANDIDATE__", "make_kvinsert_cost_candidate"),
                         ("__BEGIN__", "FusedModelBase::begin_kvinsert_cost_domain_oracle"),
                         ("__OBSERVE__", "FusedModelBase::observe_kvinsert_cost_candidates"),
                         ("__MINT__", "FusedModelBase::mint_kvinsert_exact_candidate"),
                         ("__BIND_PREFIX__", "FusedModelBase::bind_kvinsert_cost_owner_prefix"),
                         ("__GET_PREFIX__", "FusedModelBase::kvinsert_cost_owner_prefix"),
                         ("__FINISH__", "FusedModelBase::finish_kvinsert_cost_domain_oracle")):
        harness = harness.replace(marker, _body(BASE, name))
    executable = tmp_path / "native_kv_domain"
    subprocess.run([compiler, "-std=c++17", "-O0", "-x", "c++", "-", "-o", str(executable)],
                   input=harness, text=True, check=True, timeout=60)
    subprocess.run([str(executable)], check=True, timeout=30)


def test_actual_cpp_multiview_text_getter_requires_primed_current_authority(tmp_path):
    compiler = shutil.which("g++")
    assert compiler
    getter = _body((ROOT / "src/fused/rpu_qwen3_model.h").read_text(),
                   "multiview_text_stage_descriptor")
    source = r'''
#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#define TORCH_CHECK(c,...) do {if(!(c)) throw std::runtime_error("rejected");}while(0)
struct Owner {
    bool prepared=true, qwen3vl_multiview_text_composite_inputs_primed_=true;
    bool qwen3vl_multiview_text_composite_dispatch_=false, current_query=true;
    mutable int queries=0;
    std::vector<int64_t> qwen3vl_multiview_text_stage_descriptor_{3,90,225};
    void validate_qwen3vl_multiview_text_contract() const {TORCH_CHECK(prepared);}
    int kvinsert_cost_domain(const char* kind,const std::vector<int64_t>& value) const {
        ++queries;TORCH_CHECK(current_query);
        assert(std::string(kind)=="causal_decoder");
        assert(&value==&qwen3vl_multiview_text_stage_descriptor_);return 0;
    }
    const std::vector<int64_t>& multiview_text_stage_descriptor() const __BODY__
};
template<class F> void rejects(F f) {bool failed=false;try{f();}catch(const std::runtime_error&){failed=true;}assert(failed);}
int main() {
    Owner owner;
    assert(&owner.multiview_text_stage_descriptor()==&owner.qwen3vl_multiview_text_stage_descriptor_);
    assert(owner.queries==1);
    owner.prepared=false;rejects([&]{owner.multiview_text_stage_descriptor();});owner.prepared=true;
    owner.qwen3vl_multiview_text_composite_inputs_primed_=false;
    rejects([&]{owner.multiview_text_stage_descriptor();});
    owner.qwen3vl_multiview_text_composite_inputs_primed_=true;
    owner.qwen3vl_multiview_text_composite_dispatch_=true;
    rejects([&]{owner.multiview_text_stage_descriptor();});
    owner.qwen3vl_multiview_text_composite_dispatch_=false;
    owner.current_query=false;rejects([&]{owner.multiview_text_stage_descriptor();});
    assert(owner.queries==2);
    owner.current_query=true;owner.qwen3vl_multiview_text_stage_descriptor_.clear();
    rejects([&]{owner.multiview_text_stage_descriptor();});assert(owner.queries==2);
}
'''.replace("__BODY__", getter)
    binary = tmp_path / "native_text_getter"
    subprocess.run([compiler, "-std=c++17", "-x", "c++", "-", "-o", str(binary)],
                   input=source, text=True, check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=30)
