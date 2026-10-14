"""CPU/source contracts for LingBot2's finite variable-prompt envelope."""
from __future__ import annotations

import ast
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = (
    ROOT / "python" / "rpu_backend" / "adapters" / "lingbot_vla_v2"
    / "runtime.py"
)
API = ROOT / "python" / "rpu_backend" / "api" / "lingbot2.py"
QWEN3_MODEL = ROOT / "src" / "fused" / "rpu_qwen3_model.h"
QWEN3_MODEL_CPP = ROOT / "src" / "fused" / "rpu_qwen3_model.cpp"


def _class_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text())
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )


def _source(node: ast.AST, path: Path = RUNTIME) -> str:
    return ast.get_source_segment(path.read_text(), node)


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    node = next(
        item for item in tree.body
        if (
            isinstance(item, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name
                    for target in item.targets)
        ) or (
            isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and item.target.id == name
        )
    )
    return ast.literal_eval(node.value)


def test_generic_runtime_uses_16_row_execution_tail_padding():
    source = _source(
        _class_method(RUNTIME, "LingbotVlaV2Policy", "_prefix_execution_plan")
    )
    assert "plan_bounded_prefill_execution(" in source
    assert "alignment=PREFIX_PADDING_MULTIPLE" in source
    assert "causal_decoder_resolve_prefill_stage_domain" in source
    assert "MULTIVIEW_SPM_Z2_EXECUTION_LEN" in source


def test_generic_same_bucket_position_and_partial_mrope_refresh_use_versions():
    prefix = _source(
        _class_method(RUNTIME, "LingbotVlaV2Policy", "_build_prefix")
    )
    assert "with torch.inference_mode(False):" in prefix
    assert "stable_position_ids_rpu = position_ids_for_decoder.to" in prefix
    assert "torch.is_inference(tensor)" in prefix
    assert '"position_ids_rpu": stable_position_ids_rpu' in prefix
    for token in (
        "_build_prefill_mrope_inputs(",
        "stable_rope_cos_il = rope_cos_il_cpu.to",
        "stable_rope_sin_il = rope_sin_il_cpu.to",
        '"rope_cos_il": stable_rope_cos_il',
        '"rope_sin_il": stable_rope_sin_il',
        "self._pfx_memos[execution_key] = memo",
    ):
        assert token in prefix

    refresh = ast.get_source_segment(
        RUNTIME.read_text(),
        next(
            node for node in ast.parse(RUNTIME.read_text()).body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_refresh_prefix_memo_"
        ),
    )
    assert "with torch.inference_mode(False):" in refresh
    assert 'memo["position_ids"].squeeze(1).transpose(0, 1)' in refresh
    assert '.to(dtype=torch.int32).contiguous()' in refresh
    assert 'memo["position_ids_rpu"].copy_(position_ids_for_decoder)' in refresh
    assert 'memo["rope_cos_il"].copy_(rope_cos_il)' in refresh
    assert 'memo["rope_sin_il"].copy_(rope_sin_il)' in refresh

    vlm_fill = _source(
        _class_method(RUNTIME, "LingbotVlaV2Policy", "_vlm_fill")
    )
    assert "rope_cos_il=rope_cos_il" in vlm_fill
    assert "rope_sin_il=rope_sin_il" in vlm_fill
    assert "prefill_plan=(S, chunk_size, a6_plan)" in vlm_fill
    assert "*a6_plan.graph_key_words()" in vlm_fill

    native = QWEN3_MODEL.read_text()
    for token in (
        "ka_last_pos_src_version_",
        "pos.unsafeGetTensorImpl()->version_counter()",
        "pos_version_counter.enabled()",
        "pos_version_counter.current_version()",
        "pos_content_changed",
        "pos_src_version != ka_last_pos_src_version_",
        "pos.data_ptr() != ka_last_pos_src_ ||",
        "ka_last_pos_src_version_ = pos_src_version",
        "ka_last_cos_src_version_",
        "cos_version_counter.current_version()",
        "cos_content_changed",
        "cos_src_version != ka_last_cos_src_version_",
        "ka_last_cos_src_version_ = cos_src_version",
        "ka_last_sin_src_version_",
        "sin_version_counter.current_version()",
        "sin_content_changed",
        "sin_src_version != ka_last_sin_src_version_",
        "ka_last_sin_src_version_ = sin_src_version",
    ):
        assert token in native
    native_cpp = QWEN3_MODEL_CPP.read_text()
    assert native_cpp.count("ka_last_cos_src_version_ = -1") >= 6
    assert native_cpp.count("ka_last_sin_src_version_ = -1") >= 6


def test_ready_admission_precedes_any_graph_or_rpu_input_conversion():
    source = _source(
        _class_method(RUNTIME, "LingbotVlaV2Policy", "get_action")
    )
    admit = source.index("self._validate_prepared_request")
    assert admit < source.index("state = state.float()")
    assert admit < source.index("self._vgc = rpu_backend.graph.GraphCache(")
    assert admit < source.index("self._fill_prefix_multiview_spm_z2")
    assert admit < source.index("self._build_prefix(")


def test_ready_cpu_preflight_rejects_device_metadata():
    source = _source(
        _class_method(RUNTIME, "LingbotVlaV2Policy", "_request_prefix_geometry")
    )
    for name in (
        "images.device.type", "img_masks.device.type", "grid_thw.device.type",
        "lang_tokens.device.type", "lang_masks.device.type",
    ):
        assert name in source


def test_suffix_padding_semantics_are_in_cache_key_and_call():
    source = _source(
        _class_method(RUNTIME, "LingbotVlaV2Policy", "get_action")
    )
    assert "mask_key = (S, real_S)" in source
    assert "real_prefix_len=real_S" in source
    assert "state, noise, mask_r, S, real_S, graph_rope_offset" in source


def test_ready_denoise_uses_stable_exact_rope_content_per_bucket():
    refresh = _source(
        _class_method(RUNTIME, "LingbotVlaV2Policy", "_prepare_ready_suffix_rope")
    )
    unroll = _source(
        _class_method(RUNTIME, "LingbotVlaV2Policy", "_denoise_unroll_run")
    )
    assert ".narrow(0, 0, N_ACTION + 1).copy_(cos_cpu)" in refresh
    assert ".narrow(0, 0, N_ACTION + 1).copy_(sin_cpu)" in refresh
    assert "return 0" in refresh
    assert "[EXP_LAYERS, S, NUM_STEPS, PREFIX_POLICY_VERSION]" in unroll


def test_suffix_rope_window_matches_full_table_slice_exactly():
    tree = ast.parse(RUNTIME.read_text())
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_suffix_rope_window_cpu"
    )
    namespace = {"torch": torch, "math": math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(RUNTIME), "exec"), namespace)
    build = namespace["build_suffix_rope_window_cpu"]
    cos, sin = build(42, seq_len=51, head_dim=128, rope_theta=5_000_000.0)

    inv = 1.0 / (
        5_000_000.0 ** (torch.arange(0, 128, 2, dtype=torch.float64) / 128)
    )
    freqs = torch.arange(1024, dtype=torch.float64)[:, None] * inv[None, :]
    assert torch.equal(cos, freqs.cos().float().half()[42:93])
    assert torch.equal(sin, freqs.sin().float().half()[42:93])


def test_public_api_calls_max_lang_tokens_a_cap_not_semantic_padding():
    api = API.read_text()
    assert "tokenizer truncation/admission cap" in api
    assert "attention_mask`` and preserves the resulting variable REAL prefix" in api
    assert "def prepare_graphs(" in api
    assert "prefix_length_range: Sequence[int] | None = None" in api
    assert "hi = metadata[\"fixed_real_rows\"] + self._max_lang_tokens" in api


def test_prepare_graphs_is_atomic_with_execution_reconfigure():
    runtime = RUNTIME.read_text()
    prepare = runtime[runtime.index("    @execution_serialized\n    @torch.no_grad()\n    def prepare_graphs(") :]
    assert prepare.index("self.get_action(") < prepare.index("return self.graph_profile")


def test_all_generic_defaults_use_only_ordinary_graph_optimizations():
    bundle = _literal_assignment(API, "_GENERIC_OPT_ENV")
    assert bundle == {
        "RPU_FASTREPLAY_SKIP_SYNC": "1",
        "RPU_KVINSERT_HYBRID_V16": "1",
        "RPU_ADARMS_FUSED_BCAST": "1",
        "RPU_QWEN3VL_VISION_ROPE_SPM": "1",
        "RPU_QWEN3VL_VISION_BATCH": "1",
        "RPU_QWEN3VL_VISION_FUSED_MERGER": "1",
        "RPU_QWEN3VL_VISION_PATCH_EMBED_DEVICE": "1",
        "RPU_LINGBOT2_PREFILL_FAST_REPLAY": "1",
        "RPU_LINGBOT2_VISION_FAST_REPLAY": "1",
        "RPU_LINGBOT2_DENOISE_UNROLL": "1",
    }
    for z2_only in (
        "RPU_WALL_OSS_FAST_REPLAY",
        "RPU_DEEP_FAST_REPLAY",
        "RPU_LINGBOT2_VISION_PACKED_PREP",
        "RPU_LINGBOT2_MULTIVIEW_SPM_Z2",
    ):
        assert z2_only not in bundle
    api = API.read_text()
    assert api.count("**_GENERIC_OPT_ENV") == 3
    assert 'denoise_unroll = env.get("RPU_LINGBOT2_DENOISE_UNROLL") == "1"' in api
    assert 'env.get("RPU_LINGBOT2_GROUPED_EXPERTS") == "1"' in api
    assert "_DENSE_FP16_DENOISE_UNROLL_CAPACITY_DEFAULTS" in api


def test_unrolled_expert_uses_handle_scoped_full_body_replay():
    source = (
        ROOT / "src" / "fused" / "rpu_lingbot_v2_moe_model.cpp"
    ).read_text()
    begin = source.index(
        "ModelStaticConfig LingbotV2MoeExpertModel::static_config()")
    section = source[begin:source.index(
        "void LingbotV2MoeExpertModel::emit_denoise_pre_layers()", begin)]
    assert "if (denoise_unroll_)" in section
    assert "cfg.fast_replay_skip_layer_loop = true;" in section
    assert "cfg.fast_replay_skip_full_body = true;" in section


def test_sealed_z2_overlay_wins_over_generic_quantized_defaults():
    source = _source(
        _class_method(API, "Lingbot2Policy", "to"), path=API
    )
    assert source.index(
        "env.update(_PROFILE_ENV[self._dtype])"
    ) < source.index("env.update(required_env)")


def test_direct_prefix_ready_requires_single_profile_and_real_length():
    source = _source(
        _class_method(RUNTIME, "LingbotVlaV2Policy", "prepare_graphs")
    )
    assert "if self._vision_direct_prefix and (len(profiles) != 1 or lo != hi):" in source
    assert "one fixed REAL prefix" in source
    assert source.index("one fixed REAL prefix") < source.index("self.get_action(")


def test_ready_is_published_only_after_freeze_and_real_replay():
    method = _class_method(
        RUNTIME, "LingbotVlaV2Policy", "prepare_graphs")
    source = _source(method)
    freeze = source.index("cache.freeze()")
    replay = source.index("ready_out = self.get_action")
    publish = source.index("self._graphs_ready = True")
    assert freeze < replay < publish
    assert 'profile["phase"] = "READY"' in source
    assert "self._graphs_ready = False" in source


def test_z2_ready_proves_one_retained_build_and_stable_replay():
    source = _source(
        _class_method(RUNTIME, "LingbotVlaV2Policy", "prepare_graphs")
    )
    assert source.count("lingbot2_multiview_spm_z2_stats") == 2
    assert "z2_ready_stats[9] != 1" in source
    assert "z2_ready_stats[10] <= z2_warm_stats[10]" in source
    assert "z2_ready_stats[20:24] != [1, 1, 0, 0]" in source
