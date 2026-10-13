import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(relative):
    return (ROOT / relative).read_text(encoding="utf-8")


def test_dispatched_control_setters_route_or_fail_closed():
    causal = _source("src/fused/rpu_qwen3_model.cpp")
    text = _source("src/fused/rpu_qwen3_5_model.cpp")
    q35_vision = _source("src/fused/rpu_qwen3_5_vision_model.cpp")
    vl_vision = _source("src/fused/rpu_qwen3vl_vision_model.cpp")

    assert "->set_control_chunk_size_override(" in causal
    assert "->enable_execution_reconfigure_guard();" in causal
    assert "use the execution-reconfigure transaction for a live handle" in text
    assert "Qwen3.5 vision chunk size must be set before the first forward" in q35_vision
    assert "Qwen3-VL vision chunk size must be configured before" in vl_vision


def test_exact_domain_ignores_auto_cap_but_keeps_native_validation():
    base = _source("src/core/fused_model_base.cpp")
    text = _source("src/fused/rpu_qwen3_5_model.cpp")
    vision = _source("src/fused/rpu_qwen3_5_vision_model.cpp")

    assert (
        "/*apply_auto_cap=*/requested_chunk_size == 0 &&\n"
        "                effective_chunk_override == 0"
    ) in base
    assert (
        "const int64_t effective_chunk_override = planning_chunk_size_override < 0\n"
        "        ? pimpl_->chunk_size_override_ : planning_chunk_size_override;"
    ) in base
    assert "detail::validate_fmb_exact_chunk_size(" in base
    assert "if (!chunk_within_envelope(cs)) return false;" in text
    assert "return chunk_size_cap_ <= 0 || cs <= chunk_size_cap_;" not in text
    assert "Qwen3_5VisionModel::subclass_chunk_size_cap" in vision


def test_z2_reconfigure_quiesces_native_owner_before_staging():
    coordinator = _source("src/graph/graph_runtime.cpp")
    z2 = _source("src/fused/rpu_qwen3vl_multiview_spm_z2.cpp")
    runtime = _source("python/rpu_backend/adapters/lingbot_vla_v2/runtime.py")

    assert "begin_reconfigure_with_quiesce" in coordinator
    assert "lingbot2_multiview_spm_z2_reconfigure_begin" in z2
    assert "_execution_reconfigure_journal" in runtime
    assert "self._reset_execution_runtime_state()" in runtime
    rollback = next(node for node in ast.walk(ast.parse(runtime))
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "rollback_execution_reconfigure")
    restore = next(node.args[0] for node in ast.walk(rollback)
                   if isinstance(node, ast.Call)
                   and ast.unparse(node.func) == "z2.update"
                   and node.args and isinstance(node.args[0], ast.IfExp))
    # A retired Z2 generation must requery its native oracle, not restore a
    # selected descriptor from the old generation. Ordinary mode retains undo.
    assert ast.unparse(restore) == (
        "{key: None for key in journal['plan']} "
        "if self._multiview_spm_z2 else journal['plan']"
    )
    assert '"prepared": False' in runtime


def test_native_reconfigure_begin_has_exact_attempt_recovery():
    coordinator = _source("src/graph/graph_runtime.cpp")
    declarations = _source("src/core/rpu_kernel_decls.h")
    dispatcher = _source("src/core/rpu_dispatch_registrations.inc")
    execution = _source("python/rpu_backend/api/_execution.py")

    assert "state.reconfigure_attempt_token = attempt_token" in coordinator
    assert "bool RpuExecutionCoordinator::abort_reconfigure_attempt" in coordinator
    assert "if (!state.reconfigure_active) return false;" in coordinator
    assert "state.reconfigure_attempt_token == attempt_token" in coordinator
    assert "bool rpu_execution_reconfigure_abort_attempt" in declarations
    assert 'm.def("execution_reconfigure_abort_attempt"' in dispatcher
    assert "ops.execution_reconfigure_begin(attempt_token)" in execution
    assert "ops.execution_reconfigure_abort_attempt(attempt_token)" in execution


def test_every_native_reconfigure_participant_rejects_destroy_until_retired():
    sources = {
        "rpu_causal_decoder_destroy": _source("src/fused/rpu_qwen3_model.cpp"),
        "rpu_qwen3_5_destroy": _source("src/fused/rpu_qwen3_5_model.cpp"),
        "rpu_qwen3_5_vision_destroy": _source(
            "src/fused/rpu_qwen3_5_vision_model.cpp"
        ),
        "rpu_qwen3vl_vision_destroy": _source(
            "src/fused/rpu_qwen3vl_vision_model.cpp"
        ),
    }
    for operation, source in sources.items():
        assert (
            f'->check_execution_reconfigure_destroy_allowed(\n'
            f'            "{operation}");'
        ) in source
