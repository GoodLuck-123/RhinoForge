"""Same-config P7 retirement through real adapter callbacks, without a device."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.api._execution import bind_execution_session
from rpu_backend.adapters import pi05, qwen3_vl, qwen3_5
from test_qwen3_5_execution_reconfigure import _Cache


class _Model(SimpleNamespace):
    __hash__ = object.__hash__


@pytest.mark.parametrize("kind", ["qwen3_vl", "pi05"])
@pytest.mark.parametrize("fail_clear", [False, True])
def test_real_adapter_same_config_retires_and_compensates(monkeypatch, kind, fail_clear):
    events = []
    for name in ("execution_reconfigure_commit", "execution_reconfigure_abort",
                 "execution_reconfigure_abort_attempt", "causal_decoder_stage_chunk_size_override",
                 "qwen3vl_vision_stage_chunk_size"):
        monkeypatch.setattr(torch.ops.rpu, name, lambda *args: None, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "execution_reconfigure_begin", lambda attempt: attempt,
                        raising=False)
    if kind == "qwen3_vl":
        owner = qwen3_vl.Qwen3VLAdapter.__new__(qwen3_vl.Qwen3VLAdapter)
        owner.model = _Model(model=SimpleNamespace(
            language_model=SimpleNamespace(
                _rpu_decoder_handle=11,
                _rpu_decoder_graph_cache=_Cache(events, "text"),
            ),
            visual=SimpleNamespace(
                _rpu_vision_handle=12,
                _rpu_vision_graph_cache=_Cache(events, "vision", fail_clear_once=fail_clear),
            ),
        ))
        components = owner._rpu_execution_components
        resolved = qwen3_vl._resolve_qwen3_vl_execution({}, entry_point="test")
        cache_names = {"text", "vision"}
    else:
        owner = pi05.Pi05Adapter.__new__(pi05.Pi05Adapter)
        owner._legacy_vlm_chunk_size = 0
        owner._lerobot_policy = SimpleNamespace(model=_Model(
            paligemma_with_expert=SimpleNamespace(
                paligemma=SimpleNamespace(model=SimpleNamespace(
                    language_model=SimpleNamespace(_rpu_gemma_graph_cache=_Cache(events, "text")),
                    vision_tower=SimpleNamespace(_rpu_siglip_graph_cache=_Cache(
                        events, "vision", fail_clear_once=fail_clear)),
                )),
                gemma_expert=SimpleNamespace(model=SimpleNamespace(
                    _rpu_adarms_graph_cache=_Cache(events, "expert"))),
            ),
            _rpu_fused_denoise_graph_cache=_Cache(events, "denoise"),
        ))
        components = pi05.PI05_EXECUTION_COMPONENTS
        resolved = owner._resolve_execution({}, entry_point="test")
        cache_names = {"text", "vision", "expert", "denoise"}
    owner._execution_generation = 0
    owner._component_generations = {name: 0 for name in components}
    owner._execution_reconfigure_journal = None
    owner._publish_execution_views(
        resolved, generation=0, component_generations=owner._component_generations,
    )
    session = bind_execution_session(
        owner, {}, entry_point="actual adapter force retirement",
        supported_components=components, apply=owner.apply_execution_reconfigure,
        rollback=owner.rollback_execution_reconfigure, graph_mode="COMPOSITE_CHILD",
    )
    if fail_clear:
        with pytest.raises(RuntimeError, match="graph cleanup failure"):
            session._reconfigure({}, force_rebuild=True)
        assert session.generation == 0
        assert set(owner._component_generations.values()) == {0}
        assert session.stats()["state"] == "QUIESCENT"
    else:
        session._reconfigure({}, force_rebuild=True)
        assert session.generation == 1
        assert set(owner._component_generations.values()) == {1}
    assert session.config == {}
    assert {name for name, event in events if event == "clear"} == cache_names


def test_qwen35_actual_constructor_callback_forwards_force_to_native(monkeypatch):
    path = Path(qwen3_5.__file__)
    constructor = next(
        item for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5Adapter"
        for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"
    )
    callback = next(
        value for node in ast.walk(constructor) if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "apply"
    )
    calls = []
    owner = object()
    apply = eval(compile(ast.Expression(callback), str(path), "eval"), {
        "self": owner, "_apply_qwen3_5_execution_config": lambda *args, **kwargs: calls.append((args, kwargs)),
    })
    apply({}, {}, 1, force_rebuild=True)
    assert calls == [((owner, {}, 1), {"force_native": True})]
