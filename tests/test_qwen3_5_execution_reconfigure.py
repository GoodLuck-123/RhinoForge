from __future__ import annotations

from types import SimpleNamespace

import pytest

import rpu_backend.adapters.qwen3_5 as qwen3_5
from rpu_backend.api._execution import (
    bind_execution_session,
    reconfigure_rpu_execution,
)


class _Cache:
    def __init__(self, events, name, *, fail_clear_once=False):
        self.events = events
        self.name = name
        self.fail_clear_once = fail_clear_once

    def begin_warmup(self):
        self.events.append((self.name, "warm"))
        return self

    def clear(self):
        self.events.append((self.name, "clear"))
        if self.fail_clear_once:
            self.fail_clear_once = False
            raise RuntimeError("injected graph cleanup failure")

    def cache_invariant_ok(self):
        return True


class _Graph:
    def __init__(self, events, name):
        self.events = events
        self.name = name

    def invalidate(self):
        self.events.append((self.name, "invalidate"))


def _native_ops(events):
    return SimpleNamespace(
        execution_reconfigure_begin=(
            lambda _attempt: events.append(("txn", "begin")) or 77
        ),
        execution_reconfigure_commit=lambda token: events.append(
            ("txn", "commit", token)
        ),
        execution_reconfigure_abort=lambda token: events.append(
            ("txn", "abort", token)
        ),
        execution_reconfigure_abort_attempt=lambda attempt: events.append(
            ("txn", "abort_attempt", attempt)
        ) or False,
        qwen3_5_stage_prefill_execution_controls=(
            lambda handle, token, cap, chunk, max_kv, envelope: events.append(
                ("text_stage", handle, token, cap, chunk, max_kv, envelope)
            )
        ),
        qwen3_5_vision_stage_prefill_execution_controls=(
            lambda handle, token, cap, chunk: events.append(
                ("vision_stage", handle, token, cap, chunk)
            )
        ),
    )


def test_live_qwen35_reconfigure_invalidates_both_lifecycles(monkeypatch):
    events = []
    ops = _native_ops(events)
    monkeypatch.setattr(
        qwen3_5, "torch", SimpleNamespace(ops=SimpleNamespace(rpu=ops))
    )
    monkeypatch.setattr(
        qwen3_5,
        "_resolve_text_install_options",
        lambda *_args, **_kwargs: (8192, 0, 12, {"text": "new"}),
    )
    monkeypatch.setattr(
        qwen3_5,
        "_resolve_vision_install_options",
        lambda **_kwargs: (256, 256, "auto", 0, {"vision": "new"}),
    )

    text_cache = _Cache(events, "text_cache")
    retained_prefill = _Cache(events, "retained_prefill")
    vision_cache = _Cache(events, "vision_cache")
    state = SimpleNamespace(
        handle=11,
        max_seq_len=8192,
        graph_cache=text_cache,
        prefill_graph_cache=retained_prefill,
        prefill_graph=_Graph(events, "oneshot"),
        padding_budget=0,
        padding_rows="auto",
        execution_config={},
        control_snapshot={},
        exact_chunk_size=None,
        chunk_envelope=(8192, 128),
        execution_generation=0,
        prefill_plan_key=(1,),
        prefill_plan=(1, 1),
        last_a6_plan={"old": True},
        vision_control_options=None,
    )
    vision = SimpleNamespace(
        _rpu_vision_handle=22,
        _rpu_vision_graph_cache=vision_cache,
        _rpu_vision_debug_graph=_Graph(events, "vision_debug"),
        _rpu_vision_chunk_size_cap=0,
        _rpu_vision_exact_chunk_size=None,
        _rpu_vision_has_dispatched=False,
    )
    model = SimpleNamespace(model=SimpleNamespace(visual=vision))
    adapter = SimpleNamespace(model=model, _rpu_execution={})
    monkeypatch.setattr(
        qwen3_5, "_qwen3_5_text_runtime_state", lambda _model: state
    )
    config = {
        "prefill": {"chunk_size": 128, "padding_budget": 12},
        "vision": {"chunk_size": 256},
    }

    qwen3_5._apply_qwen3_5_execution_config(adapter, config, 3)

    assert state.execution_config is config
    assert state.execution_generation == 3
    assert state.prefill_plan_key is None
    assert state.prefill_plan is None
    assert state.last_a6_plan is None
    assert adapter._rpu_execution is config
    assert vision._execution_generation == 3
    assert ("text_cache", "clear") in events
    assert ("retained_prefill", "clear") in events
    assert ("oneshot", "invalidate") in events
    assert ("vision_cache", "clear") in events
    assert ("vision_debug", "invalidate") in events
    assert ("text_stage", 11, 77, 0, 128, 8192, 128) in events
    assert ("vision_stage", 22, 77, 256, 256) in events
    assert events.index(("txn", "commit", 77)) < events.index(
        ("text_cache", "clear")
    )


@pytest.mark.parametrize("force_rebuild", [False, True])
def test_live_qwen35_reconfigure_skips_unchanged_cold_setters(monkeypatch, force_rebuild):
    events = []
    ops = _native_ops(events)
    monkeypatch.setattr(
        qwen3_5, "torch", SimpleNamespace(ops=SimpleNamespace(rpu=ops))
    )
    monkeypatch.setattr(
        qwen3_5,
        "_resolve_text_install_options",
        lambda *_args, **_kwargs: (
            8192, 0, 16, {"chunk_effective_cap": 0}
        ),
    )
    monkeypatch.setattr(
        qwen3_5,
        "_resolve_vision_install_options",
        lambda **_kwargs: (256, 256, "auto", 0, {}),
    )
    state = SimpleNamespace(
        handle=11,
        max_seq_len=8192,
        graph_cache=_Cache(events, "text_cache"),
        prefill_graph=_Graph(events, "oneshot"),
        padding_budget=0,
        padding_rows="auto",
        execution_config={},
        control_snapshot={"chunk_effective_cap": 0},
        exact_chunk_size=None,
        chunk_envelope=(8192, 128),
        last_a6_plan={"dispatched": True},
    )
    vision = SimpleNamespace(
        _rpu_vision_handle=22,
        _rpu_vision_graph_cache=_Cache(events, "vision_cache"),
        _rpu_vision_debug_graph=_Graph(events, "vision_debug"),
        _rpu_vision_chunk_size_cap=256,
        _rpu_vision_exact_chunk_size=256,
        _rpu_vision_has_dispatched=True,
    )
    adapter = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(visual=vision)),
        _rpu_execution={},
    )
    monkeypatch.setattr(
        qwen3_5, "_qwen3_5_text_runtime_state", lambda _model: state
    )

    qwen3_5._apply_qwen3_5_execution_config(
        adapter, {} if force_rebuild else {"prefill": {"chunk_size": 128}}, 1,
        force_native=force_rebuild,
    )

    assert ("text_stage", 11, 77, 0, 0 if force_rebuild else 128, 8192, 128) in events
    assert any(event[0] == "vision_stage" for event in events) is force_rebuild
    if force_rebuild:
        assert ("text_cache", "clear") in events and ("vision_cache", "clear") in events
        assert state.execution_generation == 1 and state.last_a6_plan is None


def test_live_qwen35_begin_failure_does_not_start_rollback(monkeypatch):
    events = []

    def fail_begin(attempt):
        events.append(("txn", "begin", attempt))
        raise RuntimeError("injected begin failure")

    ops = _native_ops(events)
    ops.execution_reconfigure_begin = fail_begin
    monkeypatch.setattr(
        qwen3_5, "torch", SimpleNamespace(ops=SimpleNamespace(rpu=ops))
    )
    monkeypatch.setattr(
        qwen3_5,
        "_resolve_text_install_options",
        lambda *_args, **_kwargs: (
            8192, 0, 0, {"chunk_effective_cap": 0}
        ),
    )
    monkeypatch.setattr(
        qwen3_5,
        "_resolve_vision_install_options",
        lambda **_kwargs: (0, None, "auto", 0, {}),
    )
    state = SimpleNamespace(
        handle=11,
        max_seq_len=8192,
        control_snapshot={"chunk_effective_cap": 0},
        exact_chunk_size=None,
        chunk_envelope=(8192, 128),
        vision_control_options=None,
    )
    adapter = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace()),
        _rpu_execution={},
    )
    monkeypatch.setattr(
        qwen3_5, "_qwen3_5_text_runtime_state", lambda _model: state
    )
    session = bind_execution_session(
        adapter,
        {},
        entry_point="qwen35-begin-failure-test",
        supported={
            "prefill": ("chunk_size", "padding_rows", "padding_budget")
        },
        apply=lambda _old, new, generation: (
            qwen3_5._apply_qwen3_5_execution_config(adapter, new, generation)
        ),
        rollback=lambda old, _new, generation: (
            qwen3_5._rollback_qwen3_5_execution_config(
                adapter, old, generation
            )
        ),
        graph_mode="BOUNDED_ONESHOT",
    )

    with pytest.raises(RuntimeError, match="injected begin failure"):
        reconfigure_rpu_execution(
            adapter, {"prefill": {"chunk_size": 128}}
        )

    assert [event[1] for event in events] == ["begin", "abort_attempt"]
    assert session.config == {}
    assert session.generation == 0
    assert session.stats()["state"] == "QUIESCENT"


def test_live_qwen35_cleanup_failure_restores_committed_native_controls(
    monkeypatch,
):
    events = []
    ops = _native_ops(events)
    monkeypatch.setattr(
        qwen3_5, "torch", SimpleNamespace(ops=SimpleNamespace(rpu=ops))
    )

    def text_options(_max_seq_len, *, execution_config, cold_snapshot=None):
        chunk = execution_config.get("prefill", {}).get("chunk_size", "auto")
        exact = 0 if chunk == "auto" else int(chunk)
        return (8192, 0, 0, {"chunk_effective_cap": 0, "exact": exact})

    monkeypatch.setattr(qwen3_5, "_resolve_text_install_options", text_options)
    monkeypatch.setattr(
        qwen3_5,
        "_resolve_vision_install_options",
        lambda **_kwargs: (0, None, "auto", 0, {}),
    )
    state = SimpleNamespace(
        handle=11,
        max_seq_len=8192,
        graph_cache=_Cache(
            events, "text_cache", fail_clear_once=True
        ),
        prefill_graph=_Graph(events, "oneshot"),
        padding_budget=0,
        padding_rows="auto",
        execution_config={},
        control_snapshot={"chunk_effective_cap": 0, "exact": 0},
        exact_chunk_size=None,
        chunk_envelope=(8192, 128),
        last_a6_plan=None,
    )
    adapter = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace()),
        _rpu_execution={},
    )
    monkeypatch.setattr(
        qwen3_5, "_qwen3_5_text_runtime_state", lambda _model: state
    )
    bind_execution_session(
        adapter,
        {},
        entry_point="qwen35-rollback-test",
        supported={
            "prefill": ("chunk_size", "padding_rows", "padding_budget")
        },
        apply=lambda _old, new, generation: (
            qwen3_5._apply_qwen3_5_execution_config(adapter, new, generation)
        ),
        rollback=lambda old, _new, generation: (
            qwen3_5._apply_qwen3_5_execution_config(
                adapter, old, generation, force_native=True
            )
        ),
        graph_mode="BOUNDED_ONESHOT",
    )

    with pytest.raises(RuntimeError, match="injected graph cleanup failure"):
        reconfigure_rpu_execution(
            adapter, {"prefill": {"chunk_size": 128}}
        )

    assert [event[4] for event in events if event[0] == "text_stage"] == [128, 0]
    assert state.exact_chunk_size is None


def test_live_qwen35_commit_interrupt_forces_old_native_controls(monkeypatch):
    events = []
    native = {"active": False, "text": 0, "vision": 0, "staged": {}}
    fail_commit = [True]

    def begin(_attempt):
        assert not native["active"]
        native["active"] = True
        native["staged"] = {}
        return 77

    def commit(token):
        assert token == 77 and native["active"]
        native.update(native["staged"])
        native["active"] = False
        if fail_commit[0]:
            fail_commit[0] = False
            raise KeyboardInterrupt

    def abort(token):
        if not native["active"]:
            raise RuntimeError(f"stale token {token}")
        native["active"] = False

    ops = SimpleNamespace(
        execution_reconfigure_begin=begin,
        execution_reconfigure_commit=commit,
        execution_reconfigure_abort=abort,
        execution_reconfigure_abort_attempt=lambda _attempt: False,
        qwen3_5_stage_prefill_execution_controls=(
            lambda _handle, _token, _cap, chunk, _max_kv, _envelope: (
                native["staged"].update(text=chunk),
                events.append(("text", chunk)),
            )
        ),
        qwen3_5_vision_stage_prefill_execution_controls=(
            lambda _handle, _token, _cap, chunk: (
                native["staged"].update(vision=chunk),
                events.append(("vision", chunk)),
            )
        ),
    )
    monkeypatch.setattr(
        qwen3_5, "torch", SimpleNamespace(ops=SimpleNamespace(rpu=ops))
    )

    def text_options(_max_seq_len, *, execution_config, cold_snapshot=None):
        chunk = execution_config.get("prefill", {}).get("chunk_size", "auto")
        exact = 0 if chunk == "auto" else int(chunk)
        return (8192, 0, 0, {"chunk_effective_cap": 0, "exact": exact})

    def vision_options(*, execution_config=None, cold_snapshot=None):
        chunk = (execution_config or {}).get("vision", {}).get(
            "chunk_size", "auto"
        )
        exact = None if chunk == "auto" else int(chunk)
        return (0, exact, "auto", 0, {})

    monkeypatch.setattr(qwen3_5, "_resolve_text_install_options", text_options)
    monkeypatch.setattr(qwen3_5, "_resolve_vision_install_options", vision_options)
    state = SimpleNamespace(
        handle=11,
        max_seq_len=8192,
        graph_cache=_Cache(events, "text_cache"),
        prefill_graph=_Graph(events, "oneshot"),
        padding_budget=0,
        padding_rows="auto",
        execution_config={},
        control_snapshot={"chunk_effective_cap": 0},
        exact_chunk_size=None,
        chunk_envelope=(8192, 128),
        last_a6_plan=None,
        vision_control_options=(0, None, "auto", 0, {}),
    )
    vision = SimpleNamespace(
        _rpu_vision_handle=22,
        _rpu_vision_graph_cache=_Cache(events, "vision_cache"),
        _rpu_vision_debug_graph=_Graph(events, "vision_debug"),
        _rpu_vision_chunk_size_cap=0,
        _rpu_vision_exact_chunk_size=None,
    )
    adapter = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace(visual=vision)),
        _rpu_execution={},
    )
    monkeypatch.setattr(
        qwen3_5, "_qwen3_5_text_runtime_state", lambda _model: state
    )
    session = bind_execution_session(
        adapter,
        {},
        entry_point="qwen35-commit-interrupt-test",
        supported={
            "prefill": ("chunk_size", "padding_rows", "padding_budget"),
            "vision": ("chunk_size", "padding_rows", "padding_budget"),
        },
        apply=lambda _old, new, generation: (
            qwen3_5._apply_qwen3_5_execution_config(adapter, new, generation)
        ),
        rollback=lambda old, _new, generation: (
            qwen3_5._apply_qwen3_5_execution_config(
                adapter, old, generation, force_native=True
            )
        ),
        graph_mode="BOUNDED_ONESHOT",
    )

    with pytest.raises(KeyboardInterrupt):
        reconfigure_rpu_execution(
            adapter,
            {
                "prefill": {"chunk_size": 128},
                "vision": {"chunk_size": 256},
            },
        )

    assert native == {
        "active": False,
        "text": 0,
        "vision": 0,
        "staged": {"text": 0, "vision": 0},
    }
    assert events[:4] == [
        ("text", 128), ("vision", 256), ("text", 0), ("vision", 0)
    ]
    assert session.generation == 0
    assert session.stats()["state"] == "QUIESCENT"


def test_live_qwen35_rollback_uses_cold_env_snapshot(monkeypatch):
    events = []
    ops = _native_ops(events)
    monkeypatch.setattr(
        qwen3_5, "torch", SimpleNamespace(ops=SimpleNamespace(rpu=ops))
    )
    monkeypatch.delenv("QWEN3_5_TEXT_CHUNK", raising=False)
    monkeypatch.setenv("QWEN3_5_TEXT_PADDING_BUDGET", "17")
    monkeypatch.delenv("QWEN3_5_VISION_CHUNK", raising=False)
    text_cold = qwen3_5._resolve_text_install_options(8192)
    vision_cold = qwen3_5._resolve_vision_install_options()

    state = SimpleNamespace(
        handle=11,
        max_seq_len=8192,
        graph_cache=_Cache(events, "text_cache", fail_clear_once=True),
        prefill_graph=_Graph(events, "oneshot"),
        padding_budget=17,
        padding_rows="auto",
        execution_config={},
        control_snapshot=text_cold[3],
        exact_chunk_size=None,
        chunk_envelope=(8192, 128),
        last_a6_plan=None,
        vision_control_options=vision_cold,
    )
    adapter = SimpleNamespace(
        model=SimpleNamespace(model=SimpleNamespace()),
        _rpu_execution={},
    )
    monkeypatch.setattr(
        qwen3_5, "_qwen3_5_text_runtime_state", lambda _model: state
    )
    bind_execution_session(
        adapter,
        {},
        entry_point="qwen35-cold-env-rollback-test",
        supported={
            "prefill": ("chunk_size", "padding_rows", "padding_budget")
        },
        apply=lambda _old, new, generation: (
            qwen3_5._apply_qwen3_5_execution_config(adapter, new, generation)
        ),
        rollback=lambda old, _new, generation: (
            qwen3_5._apply_qwen3_5_execution_config(
                adapter, old, generation, force_native=True
            )
        ),
        graph_mode="BOUNDED_ONESHOT",
    )

    # These legacy controls are cold-only. A later process-env mutation must
    # not affect either the hot apply or its rollback.
    monkeypatch.setenv("QWEN3_5_TEXT_CHUNK", "invalid-after-install")
    monkeypatch.setenv("QWEN3_5_TEXT_PADDING_BUDGET", "invalid-after-install")
    monkeypatch.setenv("QWEN3_5_VISION_CHUNK", "invalid-after-install")

    with pytest.raises(RuntimeError, match="injected graph cleanup failure"):
        reconfigure_rpu_execution(
            adapter, {"prefill": {"chunk_size": 128}}
        )

    assert [event[4] for event in events if event[0] == "text_stage"] == [128, 0]
    assert state.padding_budget == 17
    assert state.exact_chunk_size is None


def test_live_qwen35_reconfigure_admits_dispatched_vision_with_native_txn(monkeypatch):
    monkeypatch.setattr(
        qwen3_5,
        "_resolve_text_install_options",
        lambda *_args, **_kwargs: (
            8192, 0, 64, {"chunk_effective_cap": 0}
        ),
    )
    monkeypatch.setattr(
        qwen3_5,
        "_resolve_vision_install_options",
        lambda **_kwargs: (256, 256, "auto", 0, {}),
    )
    state = SimpleNamespace(
        max_seq_len=8192,
        control_snapshot={"chunk_effective_cap": 0},
        has_dispatched=True,
        last_a6_plan={"dispatched": True},
    )
    vision = SimpleNamespace(
        _rpu_vision_handle=22,
        _rpu_vision_chunk_size_cap=128,
        _rpu_vision_exact_chunk_size=128,
        _rpu_vision_has_dispatched=True,
    )
    adapter = SimpleNamespace(
        model=SimpleNamespace(
            config=object(), model=SimpleNamespace(visual=vision)
        ),
        _rpu_execution={},
        preflight_execution=lambda *_args: None,
    )
    monkeypatch.setattr(
        qwen3_5, "_qwen3_5_text_runtime_state", lambda _model: state
    )
    monkeypatch.setattr(
        qwen3_5,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(rpu=_native_ops([]))),
    )

    qwen3_5._validate_qwen3_5_execution_reconfigure(adapter, {})


def test_live_qwen35_reconfigure_admits_dispatched_text_with_native_txn(monkeypatch):
    monkeypatch.setattr(
        qwen3_5,
        "_resolve_text_install_options",
        lambda *_args, **_kwargs: (
            8192, 256, 64, {"chunk_effective_cap": 256}
        ),
    )
    state = SimpleNamespace(
        max_seq_len=8192,
        control_snapshot={"chunk_effective_cap": 128},
        has_dispatched=True,
        last_a6_plan=None,
    )
    adapter = SimpleNamespace(
        model=SimpleNamespace(config=object(), model=SimpleNamespace()),
        _rpu_execution={},
        preflight_execution=lambda *_args: None,
    )
    monkeypatch.setattr(
        qwen3_5, "_qwen3_5_text_runtime_state", lambda _model: state
    )
    monkeypatch.setattr(
        qwen3_5,
        "torch",
        SimpleNamespace(ops=SimpleNamespace(rpu=_native_ops([]))),
    )

    qwen3_5._validate_qwen3_5_execution_reconfigure(adapter, {})
