"""Production vision planners register the actual leaf and owning Session."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters import dinov3, siglip
from rpu_backend.adapters.qwen3_vl import vision as qwen3vl
from rpu_backend.api._execution import bind_execution_session
from rpu_backend.runtime import execution_planner


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("prefix,lifecycle", [
    ("dinov3_vision", 1), ("qwen3vl_vision", 4), ("siglip", 4),
])
def test_actual_leaf_native_binding_and_descriptor_observation(monkeypatch, prefix, lifecycle):
    helpers = ast.parse((ROOT / "tests/test_execution_planner.py").read_text())
    native = {"_MODULE": execution_planner}
    exec(compile(ast.Module(body=[node for node in helpers.body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("_native_candidate", "_native_candidate_v2")], type_ignores=[]),
        "native-domain-fixture", "exec"), native)
    descriptor = native["_native_candidate_v2"](
        64, (64, 64, 64), graph_lifecycle=lifecycle, descriptor_version=3)
    calls = []
    def resolve(*args):
        calls.append(args)
        return [1, 1, len(descriptor), *descriptor]
    monkeypatch.setattr(torch.ops.rpu, prefix + "_resolve_stage_domain", resolve, raising=False)
    parent, other_parent, leaf = SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
    capabilities = {"vision_encoder": {"vision": ("chunk_size",)}}
    session = bind_execution_session(parent, {}, entry_point="vision-parent",
        supported_components=capabilities, graph_mode="COMPOSITE_CHILD")
    other_session = bind_execution_session(other_parent, {}, entry_point="unrelated-parent",
        supported_components=capabilities, graph_mode="COMPOSITE_CHILD")
    session._bind_planner_owner(leaf, "vision_encoder")
    observed = []
    with session._collect_planner_costs(lambda *args: observed.append(args)):
        if prefix == "dinov3_vision":
            result = dinov3._plan_dinov3_vision_execution(17, 64, None, execution_owner=leaf)
        elif prefix == "qwen3vl_vision":
            result = qwen3vl._plan_qwen3_vl_vision_execution(17, 64, 1, None, execution_owner=leaf)
        else:
            result = siglip._siglip_component_execution_plan(
                leaf, 17, 64, 1, external_patch_prologue=True)
    assert calls[0][:2] == (17, 64)
    assert session._planner_native == {("vision_encoder", "vision", prefix): (leaf, 17)}
    assert not other_session._planner_native
    assert [item[4] for item in observed] == ["domain", "result"]
    assert all(item[:4] == (leaf, "vision_encoder", "vision", (prefix, 17)) for item in observed)
    assert observed[0][5][1][0].physical_descriptor == tuple(descriptor)
    assert observed[1][5] == (64, 0, result)
    assert not session._started  # This is a cold plan, not a hidden forward.


def test_parent_constructor_binds_actual_qwen_children_not_the_composite_as_leaf():
    q3vl = (ROOT / "python/rpu_backend/adapters/qwen3_vl/__init__.py").read_text()
    assert "model.model.language_model, _QWEN3_VL_TEXT_COMPONENT" in q3vl
    assert "model.model.visual, _QWEN3_VL_VISION_COMPONENT" in q3vl
    q35 = (ROOT / "python/rpu_backend/adapters/qwen3_5/__init__.py").read_text()
    assert 'getattr(fusion, "language_model", fusion)' in q35
    assert 'vision = getattr(fusion, "visual", None)' in q35
    assert "self._execution_session._bind_planner_owner(vision)" in q35


def test_raw_rhino_vision_reuses_owner_plans_without_unused_retained_ready_cache(monkeypatch):
    from rpu_backend.adapters.rhinovla.vision import _plan_rhinovla_vision_execution
    from rpu_backend.api._execution import prepared_execution_plan_cache
    from rpu_backend.runtime.hw_attrs import validate_postinstall
    from test_execution_planner import _native_candidate_v2
    from test_graph_cache_freeze import _cache

    cache = _cache()
    cache._execution_plans.freeze()
    owner = SimpleNamespace(_rpu_vision_handle=17, _rpu_vision_graph_cache=cache,
                            _rpu_vision_graph_disable=True)
    queries = []
    monkeypatch.setattr(torch.ops.rpu, 'qwen3vl_vision_planner_cache_identity',
                        lambda handle: (1, 1), raising=False)
    def resolve(handle, length, batch):
        queries.append((handle, length, batch))
        words = _native_candidate_v2(length, (32, 32, 32), descriptor_version=3,
                                    graph_lifecycle=4, routes=((1, 101, 2, 0, (), 0),))
        return [1, 1, len(words), *words]
    monkeypatch.setattr(torch.ops.rpu, 'qwen3vl_vision_resolve_stage_domain', resolve, raising=False)
    plan = _plan_rhinovla_vision_execution(owner, 64, 1)
    assert _plan_rhinovla_vision_execution(owner, 64, 1) is plan
    assert queries == [(17, 64, 1)]
    owner._rpu_vision_graph_disable = False
    with pytest.raises(RuntimeError, match='READY miss'):
        _plan_rhinovla_vision_execution(owner, 64, 1)
    assert queries == [(17, 64, 1)]
    # Raw owners can also be validated torch modules. The result cache remains
    # internal warm-inference state and must not trip the existing attr guard.
    first, other = torch.nn.Module(), torch.nn.Module()
    prepared = prepared_execution_plan_cache(first, 'vision', None, 'qwen3vl_vision')
    assert prepared_execution_plan_cache(first, 'vision', None, 'qwen3vl_vision') is prepared
    assert prepared_execution_plan_cache(other, 'vision', None, 'qwen3vl_vision') is not prepared
    validate_postinstall(first)
