"""CPU-only cold configuration contracts; never import the native bootstrap."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from _module_stubs import real_rpu_env_bool


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python/rpu_backend"


@pytest.fixture
def modules(monkeypatch):
    for name in ("rpu_backend", "rpu_backend.api", "rpu_backend.adapters",
                 "rpu_backend.adapters.rhinovla", "rpu_backend.runtime"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    sys.modules["rpu_backend.runtime"].rpu_env_bool = real_rpu_env_bool()
    planner = ModuleType("rpu_backend.runtime.execution_planner")
    planner.GRAPH_COMPOSITE_CHILD = "COMPOSITE_CHILD"
    monkeypatch.setitem(sys.modules, planner.__name__, planner)

    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, PYTHON / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    execution = load("rpu_backend.api._execution", "api/_execution.py")
    runtime = load("rpu_backend.adapters.rhinovla.runtime", "adapters/rhinovla/runtime.py")
    facade = load("rpu_backend.api.rhinovla", "api/rhinovla.py")
    for fields in runtime._PIPELINE_COLD_ENV.values():
        for env in fields.values():
            monkeypatch.delenv(env, raising=False)
    return execution, runtime, facade


@pytest.mark.parametrize("bad", [0, 1, "true", "auto", None, []])
def test_normalizer_requires_actual_bool(modules, bad):
    execution, runtime, _ = modules
    with pytest.raises(TypeError, match="must be bool"):
        execution.normalize_rpu_execution(
            {"action": {"denoise_unroll": bad}}, entry_point="test",
            supported_components=runtime.RHINOVLA_EXECUTION_COMPONENTS,
        )


def test_bool_controls_require_declared_component_and_stage(modules):
    execution, runtime, _ = modules
    with pytest.raises(ValueError, match="declaration"):
        execution.normalize_rpu_execution({"vision": {"fast_replay": True}}, entry_point="test")
    with pytest.raises(ValueError, match="not supported"):
        execution.normalize_rpu_execution(
            {"vision": {"shared_cache": True}}, entry_point="test",
            supported_components=runtime.RHINOVLA_EXECUTION_COMPONENTS,
        )
    with pytest.raises(ValueError, match="unknown component"):
        runtime.resolve_rhinovla_execution(
            {"components": {"unknown": {"action": {"fast_replay": True}}}},
            entry_point="test",
        )


def test_explicit_child_then_stage_then_env_then_default_and_no_reread(modules, monkeypatch):
    _, runtime, _ = modules
    monkeypatch.setenv("RPU_RHINOVLA_DENOISE_UNROLL", "1")
    monkeypatch.setenv("RPU_RHINOVLA_VISION_FAST_REPLAY", "true")
    request = {
        "action": {"denoise_unroll": True},
        "components": {"action_expert": {"action": {"denoise_unroll": False}}},
    }
    snapshot, effective = runtime.RhinoVLAPipelineColdConfig.resolve(request)
    assert snapshot.stages["action"]["denoise_unroll"] is False
    assert snapshot.stages["vision"]["fast_replay"] is True
    assert snapshot.stages["prefill"]["fast_replay"] is False
    assert effective["action"]["denoise_unroll"] is True
    assert effective["components"] == request["components"]
    assert "chunk_size" not in effective["vision"]
    monkeypatch.setenv("RPU_RHINOVLA_VISION_FAST_REPLAY", "nonsense")
    assert snapshot.bind_defaults(request) == effective
    with pytest.raises(TypeError):
        snapshot.stages["action"]["denoise_unroll"] = True


def test_explicit_false_does_not_even_parse_invalid_shadowed_env(modules, monkeypatch):
    _, runtime, _ = modules
    monkeypatch.setenv("RPU_RHINOVLA_DENOISE_UNROLL", "invalid")
    snapshot, _ = runtime.RhinoVLAPipelineColdConfig.resolve({"action": {"denoise_unroll": False}})
    assert snapshot.stages["action"]["denoise_unroll"] is False


@pytest.mark.parametrize("config", [
    {"vision": {"rpu_mergers_streaming": True}},
    {"prefill": {"prefix_prep_rpu": True}},
    {"prefill": {"partial_mrope": True}},
    {"prefill": {"merger_scatter": True}},
    {"vision": {"bake_merger": True}},
    {"action": {"prefix_alias": True}},
])
def test_invalid_dependencies_fail_before_installation(modules, config):
    _, runtime, _ = modules
    with pytest.raises(ValueError, match="requires"):
        runtime.RhinoVLAPipelineColdConfig.resolve(config)


def make_owner(modules, request=None):
    execution, runtime, _ = modules
    snapshot, effective = runtime.RhinoVLAPipelineColdConfig.resolve(request)
    owner = SimpleNamespace(_pipeline_cold_config=snapshot, _rpu_execution=effective,
                            predict=lambda **kw: kw)
    applied = []
    session = execution.bind_execution_session(
        owner, effective, entry_point="test",
        supported_components=runtime.RHINOVLA_EXECUTION_COMPONENTS,
        validate=snapshot.validate_bound,
        apply=lambda old, new, generation: applied.append(generation),
        graph_mode="COMPOSITE_CHILD",
    )
    owner._rhinovla_execution_controller = SimpleNamespace(runtime=owner, session=session)
    return owner, session, applied


def test_hot_chunk_patch_preserves_cold_and_rejects_real_change_before_apply(modules):
    owner, session, applied = make_owner(modules, {"action": {"denoise_unroll": True, "chunk_size": 32}})
    result = session.reconfigure({"action": {"chunk_size": 64}})
    assert result["action"]["denoise_unroll"] is True
    assert result["action"]["chunk_size"] == 64
    assert applied == [1]
    before = owner._rpu_execution
    with pytest.raises(ValueError, match="cold-bound.*rebuild"):
        session.reconfigure({"action": {"denoise_unroll": False}})
    assert owner._rpu_execution is before and session.generation == 1
    assert applied == [1]
    assert session.reconfigure({"action": {}})["action"]["denoise_unroll"] is True
    assert "chunk_size" not in owner._rpu_execution["action"]


@pytest.mark.parametrize("patch", [
    {"components": {"action_expert": {"action": {"chunk_size": 64}}}},
    {"components": {"action_expert": {"action": {}}}},
    {"components": {"action_expert": {}}},
])
def test_child_hot_or_empty_patch_preserves_explicit_cold_override(modules, patch):
    _, session, _ = make_owner(modules, {
        "action": {"denoise_unroll": True},
        "components": {"action_expert": {"action": {"denoise_unroll": False, "chunk_size": 32}}},
    })
    result = session.reconfigure(patch)
    assert result["components"]["action_expert"]["action"]["denoise_unroll"] is False


def test_custom_runtime_cannot_claim_unconsumed_physical_controls(modules):
    _, runtime, _ = modules
    with pytest.raises(ValueError, match="bind a pipeline cold snapshot"):
        runtime._validate_pipeline_cold(SimpleNamespace(), {"vision": {"fast_replay": True}})
    runtime._validate_pipeline_cold(SimpleNamespace(), {"vision": {"chunk_size": 64}})


def test_facade_factory_compares_materialized_defaults_without_masking_conflict(modules, monkeypatch):
    _, _, facade = modules
    request = {"action": {"chunk_size": 32}}
    owner, session, _ = make_owner(modules, request)
    monkeypatch.setenv("RPU_RHINOVLA_DENOISE_UNROLL", "invalid")
    policy = facade.RhinoVLAPolicy.from_factory(lambda config, **kw: owner, {}, rpu_execution=request)
    assert facade.RhinoVLAPolicy.from_runtime(policy, rpu_execution=request) is policy
    with pytest.raises(ValueError, match="conflicts"):
        facade.RhinoVLAPolicy.from_runtime(owner, rpu_execution={"action": {"chunk_size": 64}})
    with pytest.raises(ValueError, match="conflicts"):
        facade.RhinoVLAPolicy.from_runtime(owner, rpu_execution={"action": {"chunk_size": 32, "denoise_unroll": True}})
    closed = []
    owner.close = lambda: closed.append(True)
    policy.close()
    policy.close()
    assert closed == [True] and session.stats()["state"] == "CLOSED"


def test_pipeline_consumes_snapshot_and_prepares_native_schedule_before_planning():
    source = (PYTHON / "adapters/rhinovla/pipeline.py").read_text()
    assert 'vision_cold["fast_replay"]' in source
    assert 'action_cold["denoise_unroll"]' in source
    assert '_env_truthy("RPU_RHINOVLA_' not in source
    assert 'os.environ.get("RPU_RHINOVLA_VISION_' not in source
    assert source.index("loop_setter = (") < source.index("loop_setter(") < source.index("rhino_vla_prepare_denoise_loop_schedule(")
    assert "self.er._rpu_handle, self.steps, self._denoise_dt" in source
