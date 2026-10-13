"""Board-free source contracts for RhinoVLA component-plan authority."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def test_action_native_descriptor_carries_and_consumes_ddr_attention_route():
    source = _read("src/fused/rpu_rhino_vla_model.cpp")

    assert "resolve_action_stage_domain(" in source
    assert "FmbGraphLifecycle::COMPOSITE_CHILD" in source
    assert "FmbRouteFamily::ATTENTION" in source
    assert "AttentionExecutionPolicy::DDR_KV" in source
    assert "ctx().consume_physical_attention_route(" in source
    assert "planned_stage_descriptor" in source


def test_all_three_production_children_publish_descriptor_receipts():
    action = _read("python/rpu_backend/adapters/rhinovla/fused.py")
    vision = _read("python/rpu_backend/adapters/rhinovla/vision.py")
    pipeline = _read("python/rpu_backend/adapters/rhinovla/pipeline.py")

    assert '"authority": "NATIVE_A6_STAGE_DESCRIPTOR"' in action
    assert '"attention_policy": "DDR_REQUIRED"' in action
    assert "selected_action_plan.stage_tuple.physical_descriptor" in action
    assert '"authority": "NATIVE_A6_STAGE_DESCRIPTOR"' in vision
    assert "selected.stage_tuple.physical_descriptor" in vision
    assert "prefill_plan=(execution_len, chunk_size, prefill_plan)" in pipeline
    assert "*action_plan.graph_key_words()" in pipeline
    assert "selected_action_plan.stage_tuple.physical_descriptor" in pipeline


def test_component_session_has_stable_children_and_transactional_hot_controls():
    runtime = _read("python/rpu_backend/adapters/rhinovla/runtime.py")

    for component in ("language_model", "vision_encoder", "action_expert"):
        assert component in runtime
    assert "bind_execution_session(" in runtime
    assert "with native_execution_reconfigure(" in runtime
    assert "component_generations[component] += 1" in runtime
    assert "journal=self._journal" in runtime
    assert 'not journal["mutation_started"]' in runtime


def test_vision_direct_setter_is_cold_only_and_has_transaction_guard_wrapper():
    source = _read("src/fused/rpu_qwen3vl_vision_model.cpp")
    dispatch = _read("src/core/rpu_dispatch_registrations.inc")

    assert 'get_last_resolved_chunk_size() == 0' in source
    assert "rpu_qwen3vl_vision_enable_execution_reconfigure" in source
    assert "->enable_execution_reconfigure_guard();" in source
    assert "qwen3vl_vision_enable_execution_reconfigure(int handle)" in dispatch


def test_text_padding_remains_a_masked_physical_prefix_for_action():
    pipeline = _read("python/rpu_backend/adapters/rhinovla/pipeline.py")

    assert 'prep["prefix_mask"].new_zeros' in pipeline
    assert "physical_prefix_len = int(self._text_cache.position)" in pipeline
    assert "prefix_len=int(self._text_cache.position)" in pipeline
    assert "exp_max_seq = int(math.ceil((max_seq + suffix_len + 16) / 16) * 16)" in pipeline
    assert '"exp_max_seq": exp_max_seq' in pipeline
    assert 'self.exp_max_seq = cold["exp_max_seq"]' in pipeline
    assert "max_seq_len=self.exp_max_seq" in pipeline
