"""Host-only contracts for mask preparation before Graph receipts/work."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _block(source: str, marker: str) -> str:
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 1
    cursor = opening + 1
    while depth:
        depth += (source[cursor] == "{") - (source[cursor] == "}")
        cursor += 1
    return source[start:cursor]


def test_fmb_normalizes_dynamic_inputs_before_internal_manifest_branch() -> None:
    source = (ROOT / "src/core/fused_model_base.cpp").read_text(encoding="utf-8")
    run = _block(source, "at::Tensor FusedModelBase::run_all_layers_impl(")

    dynamic = run.index("auto dyn_cfg = dynamic_config(plan);")
    deferred_branch = run.index("if (defer_physical_manifest_branch)", dynamic)
    receipt = run.index("physical_graph.record_physical_manifest_branch(", deferred_branch)
    first_route_validation = run.index(
        "manifest_attention_layout_policy =", receipt
    )

    assert "physical_graph.record_physical_manifest_branch(" not in run[:dynamic]
    assert dynamic < deferred_branch < receipt < first_route_validation


def test_hyvla_reuses_the_causal_decoder_prepared_mask() -> None:
    for relative, marker in (
        (
            "src/fused/rpu_hyvla_vlm_model.cpp",
            "ModelDynamicConfig HyVlaVlmModel::dynamic_config(",
        ),
        (
            "src/fused/rpu_hyvla_expert_model.cpp",
            "ModelDynamicConfig HyVlaExpertModel::dynamic_config(",
        ),
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        dynamic = _block(source, marker)
        assert "CausalDecoderModel::dynamic_config(plan)" in dynamic
        assert "sdpa_prepare_mask(" not in dynamic
        assert "prepared_attn_mask_" in source

    headers = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8")
        for relative in (
            "src/fused/rpu_hyvla_vlm_model.h",
            "src/fused/rpu_hyvla_expert_model.h",
        )
    )
    assert "prepared_mask_hyvla_" not in headers
    assert "PreparedMask prepared_mask_;" not in headers


def test_causal_mask_cache_uses_live_identity_and_mutation_version() -> None:
    source = (ROOT / "src/fused/rpu_qwen3_model.h").read_text(encoding="utf-8")
    dynamic = _block(
        source, "ModelDynamicConfig dynamic_config(const ChunkPlan& plan) override"
    )

    assert "last_attn_mask_ref_.is_same(input_mask)" in dynamic
    assert "version_counter.current_version()" in dynamic
    assert "version_enabled &&" in dynamic
    assert "attention mask mutated while its" in dynamic
    assert "last_attn_mask_ref_ = input_mask;" in dynamic
    assert "last_attn_mask_ptr_" not in source


def test_causal_mask_cache_invalidates_stamp_before_slot_publication() -> None:
    source = (ROOT / "src/fused/rpu_qwen3_model.h").read_text(encoding="utf-8")
    dynamic = _block(
        source, "ModelDynamicConfig dynamic_config(const ChunkPlan& plan) override"
    )
    miss = _block(dynamic, "if (!cache_hit)")

    invalidate_ref = miss.index("last_attn_mask_ref_ = at::Tensor{};")
    invalidate_version = miss.index("last_attn_mask_version_ = -1;")
    prepare = miss.index(
        "PreparedMask next_prepared_attn_mask = sdpa_prepare_mask("
    )
    mutation_check = miss.index("version_counter.current_version()", prepare)
    commit_prepared = miss.index(
        "prepared_attn_mask_ = std::move(next_prepared_attn_mask);"
    )
    commit_ref = miss.index("last_attn_mask_ref_ = input_mask;")

    assert invalidate_ref < invalidate_version < prepare
    assert prepare < mutation_check < commit_prepared < commit_ref


def test_hyvit2_packed_mask_stays_host_side_until_sdpa_preparation() -> None:
    source = (ROOT / "src/fused/rpu_hyvit2_vision_model.cpp").read_text(
        encoding="utf-8"
    )
    packed = _block(
        source,
        "const PreparedMask& packed_block_mask(int64_t seq_len, int64_t n_img)",
    )
    created = packed.index("at::Tensor m = at::full(")
    prepared = packed.index("sdpa_prepare_mask(", created)
    normalization = packed[created:prepared]

    assert "device(at::kPrivateUse1)" not in normalization
    assert ".to(at::kPrivateUse1)" not in normalization
    assert "m = m.contiguous();" in normalization


def test_shared_mask_cache_reuses_and_refreshes_actual_tensor_versions(tmp_path):
    """Execute the base mask cache used by HALO with CPU publication doubles."""
    import torch
    from test_siglip_attention_dry_live_contract import _run_native

    source = (ROOT / "src/fused/rpu_qwen3_model.h").read_text()
    dynamic = _block(source, "ModelDynamicConfig dynamic_config(const ChunkPlan& plan) override")
    mask = _block(dynamic, "if (ctx().attention_mask.has_value())")
    program = r'''
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <cassert>
#include <optional>
struct PreparedMask { at::Tensor tensor; };
struct Cache { int uploads=0; bool mutate=false; };
PreparedMask sdpa_prepare_mask(const std::optional<at::Tensor>& input, bool,
                              int64_t, int64_t, Cache& cache) {
    ++cache.uploads;
    auto copy=input->clone();
    if(cache.mutate) { cache.mutate=false; input->add_(1); }
    return {copy};
}
struct Model {
    struct Context { std::optional<at::Tensor> attention_mask; int64_t seq_len=2,position=2; } context;
    struct Plan { int num_chunks=1; } plan;
    bool planning_domain_query_=false;
    at::Tensor last_attn_mask_ref_;
    int64_t last_attn_mask_version_=-1,last_attn_seq_q_=-1,last_attn_seq_k_=-1;
    PreparedMask prepared_attn_mask_;
    Cache cache;
    Context& ctx() { return context; }
    Cache& sdpa_stable_mask_cache() { return cache; }
    int run() { int cfg=0; __MASK__; return cfg; }
};
int main() {
    for(bool inference:{false,true}) {
        c10::InferenceMode guard(inference);
        Model m;
        auto input=at::zeros({2,4},at::kFloat);
        m.context.attention_mask=input;
        m.run(); assert(m.cache.uploads==1);
        auto old=m.prepared_attn_mask_.tensor;
        m.run(); assert(m.cache.uploads==(inference?2:1));
        assert(m.prepared_attn_mask_.tensor.equal(input));
        input.add_(3);
        int before=m.cache.uploads;
        m.run(); assert(m.cache.uploads==before+1);
        assert(m.prepared_attn_mask_.tensor.equal(input));
        assert(old.sum().item<float>()==0);
        // An equal replacement has a distinct identity and cannot reuse a stamp.
        m.context.attention_mask=input.clone(); before=m.cache.uploads;
        m.run(); assert(m.cache.uploads==before+1);
        ++m.context.position; before=m.cache.uploads;
        m.run(); assert(m.cache.uploads==before+1);
    }
    // A failed publication cannot leave a reusable identity/version receipt.
    Model m; auto input=at::zeros({2,4},at::kFloat);
    m.context.attention_mask=input; m.run(); input.add_(1); m.cache.mutate=true;
    bool rejected=false; try { m.run(); } catch(const c10::Error&) { rejected=true; }
    assert(rejected&&!m.last_attn_mask_ref_.defined());
    assert(m.last_attn_mask_version_==-1);
    int before=m.cache.uploads; m.run(); assert(m.cache.uploads==before+1);
    assert(m.prepared_attn_mask_.tensor.equal(input));
    before=m.cache.uploads; m.run(); assert(m.cache.uploads==before);
}
'''.replace("__MASK__", mask)
    _run_native(tmp_path, program, torch)
