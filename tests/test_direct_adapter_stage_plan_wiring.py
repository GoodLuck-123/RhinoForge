"""Source contract for direct A6 planner owners outside decoder.py."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "op_name", "forward_calls", "digest_calls"),
    [
        ("python/rpu_backend/adapters/halo/text_cache.py", "causal_decoder_forward", 1, 1),
        ("python/rpu_backend/adapters/halo/vit_cache.py", "causal_decoder_forward", 1, 2),
        ("python/rpu_backend/adapters/internvla_n1/backbone.py", "causal_decoder_forward", 1, 1),
        ("python/rpu_backend/adapters/pi05/gemma.py", "gemma_forward", 2, 1),
        ("python/rpu_backend/adapters/wall_oss/action.py", "causal_decoder_forward", 2, 6),
        ("python/rpu_backend/adapters/wall_oss/llm.py", "causal_decoder_forward", 1, 1),
    ],
)
def test_direct_owner_forwards_immutable_winner_and_keys_full_digest(
    relative_path, op_name, forward_calls, digest_calls
):
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    if op_name == "gemma_forward":
        # Both public entrypoints share the same planner/GraphCache body;
        # their cold mode selects a hidden-producing or cache-only native op.
        bindings = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "native_forward"
                    for target in node.targets)
        ]
        assert len(bindings) == 1
        assert ast.unparse(bindings[0].value) == (
            "torch.ops.rpu.gemma_prefill_kv_only if _kv_only "
            "else torch.ops.rpu.gemma_forward"
        )
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and ((isinstance(node.func, ast.Attribute) and node.func.attr == op_name)
             or (op_name == "gemma_forward" and isinstance(node.func, ast.Name)
                 and node.func.id == "native_forward"))
    ]
    assert len(calls) == forward_calls
    for call in calls:
        rendered = ast.unparse(call)
        assert "planned_stage_descriptor" in rendered
        if op_name == "gemma_forward":
            assert isinstance(call.args[7], ast.Constant) and call.args[7].value == 0
        elif len(call.args) > 14:
            assert isinstance(call.args[14], ast.Constant) and call.args[14].value == 0
        else:
            planned = next(
                keyword for keyword in call.keywords
                if keyword.arg == "planned_chunk_size"
            )
            assert isinstance(planned.value, ast.Constant) and planned.value.value == 0
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "graph_key_words"
        for node in ast.walk(tree)
    ) == digest_calls


def test_halo_vit_and_lm_children_key_their_independent_plan_identities():
    tree = ast.parse(
        (ROOT / "python/rpu_backend/adapters/halo/vit_cache.py").read_text(
            encoding="utf-8"
        )
    )
    runner = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HaloVitCacheUpdate"
    )
    methods = {
        node.name: ast.unparse(node)
        for node in runner.body if isinstance(node, ast.FunctionDef)
    }
    assert "component=HALO_VISION_COMPONENT" in methods["_vision_execution_plan"]
    assert "self._vision_execution_plan(n_vit)" in methods["_run_vit"]
    assert "*plan.graph_key_words()" in methods["_run_vit"]
    assert "component=HALO_VIT_LM_COMPONENT" in methods["_lm_execution_plan"]
    assert "self._lm_execution_plan(" in methods["_lm_forward"]
    assert "*a6_plan.graph_key_words()" in methods["_lm_forward"]


def test_pi05_hands_off_one_plan_and_always_removes_it():
    runtime = (
        ROOT / "python/rpu_backend/adapters/pi05/runtime.py"
    ).read_text(encoding="utf-8")
    assert runtime.count("vlm._rpu_planned_prefill_plan = a6_plan") == 1
    assert runtime.count('vars(vlm).pop("_rpu_planned_prefill_plan", None)') == 1
    assert "_rpu_planned_prefill_chunk_size" not in runtime


def test_lingbot2_generic_text_and_action_share_a6_authority():
    path = ROOT / "python/rpu_backend/adapters/lingbot_vla_v2/runtime.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    policy = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LingbotVlaV2Policy"
    )
    methods = {
        node.name: ast.unparse(node)
        for node in policy.body if isinstance(node, ast.FunctionDef)
    }
    assert "plan_bounded_prefill_execution" in methods["_prefix_execution_plan"]
    assert "causal_decoder_resolve_prefill_stage_domain" in methods["_prefix_execution_plan"]
    assert "prefill_plan=(S, chunk_size, a6_plan)" in methods["_vlm_fill"]
    assert "a6_plan.graph_key_words()" in methods["_vlm_fill"]
    source = path.read_text(encoding="utf-8")
    assert "plan_bounded_prefill_execution" in methods["_action_execution_plan"]
    assert "lingbot_v2_moe_resolve_action_stage_domain" in methods[
        "_action_execution_plan"
    ]
    assert "_fixed_composite_child_plan" not in source
    assert methods["get_action"].count("action_plan.graph_key_words()") == 1
    assert "action_plan.graph_key_words()" in methods["_denoise_unroll_run"]
