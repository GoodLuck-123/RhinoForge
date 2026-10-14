"""CPU-only graph capture observability contracts."""

from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import weakref

import pytest


# 2026-08-01 namespace move: the capture scope left _bootstrap.py / graph/__init__.py
# for graph/_runtime.py, and the two duplicate implementations were consolidated into
# one. This test AST-parsed the two old homes and silently StopIteration'd on all 8
# cases until 2026-08-09. Keep this path pointed at wherever the class actually lives.
_RUNTIME = (
    Path(__file__).resolve().parents[1]
    / "python"
    / "rpu_backend"
    / "graph"
    / "_runtime.py"
)
_GRAPH_INIT = _RUNTIME.with_name("__init__.py")


def _load_scope_class(source, class_name):
    """Load only the class under test without importing the hardware backend."""
    tree = ast.parse(source.read_text(), filename=str(source))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == class_name
    )
    nodes = [node]
    if class_name == "_GraphCaptureScope":
        helper = next(
            item
            for item in tree.body
            if isinstance(item, ast.FunctionDef)
            and item.name == "_skip_idle_record_function"
        )
        cache = next(
            item
            for item in tree.body
            if isinstance(item, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_SKIP_IDLE_RF_ENV"
                for target in item.targets
            )
        )
        nodes = [helper, cache, node]
    namespace = {}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[class_name]


class _FakeGraph:
    def __init__(self, states, events, *, post_states=None, end_error=None):
        states = list(states)
        self._states = iter(states)
        self._post_states = iter(post_states or [2] * len(states))
        self._events = events
        self._end_error = end_error
        self._state = None

    def begin(self, sig):
        self._state = next(self._states)
        self._events.append(("begin", sig))

    def state(self):
        return self._state

    def end(self):
        self._events.append(("end",))
        if self._end_error is not None:
            raise self._end_error
        self._state = next(self._post_states)

    def abort(self):
        self._events.append(("abort",))


class _FakeCacheImpl:
    def __init__(self, graph, events):
        self._graph = graph
        self._events = events

    def get_or_create(self, sig):
        return self._graph

    def touch_entry(self, sig, replay):
        self._events.append(("touch", sig, replay))


class _Box:
    def _graph_for_capture(self, sig):
        return self._impl.get_or_create(sig._impl)


@pytest.fixture(
    params=[
        (_RUNTIME, "_GraphCaptureScope"),
        (_GRAPH_INIT, "_GraphScope"),
    ],
    ids=["runtime", "raw"],
)
def scope_implementation(request):
    return request.param


def _scope_fixture(implementation, states, *, post_states=None, end_error=None):
    from test_execution_planner import _MODULE as planner

    source, class_name = implementation
    events = []
    graph = _FakeGraph(
        states, events, post_states=post_states, end_error=end_error
    )
    cache = _Box()
    cache._execution_plans = planner.PreparedExecutionPlans()
    cache._impl = _FakeCacheImpl(graph, events)
    sig = _Box()
    sig._impl = object()
    sig.op_id = "cpu_mock"
    scope_cls = _load_scope_class(source, class_name)
    if class_name == "_GraphScope":
        return lambda: scope_cls(graph, sig._impl), sig, events
    return lambda: scope_cls(cache, sig), sig, events


def test_graph_capture_scope_counts_build_then_replay_after_end():
    make_scope, sig, events = _scope_fixture(
        (_RUNTIME, "_GraphCaptureScope"), [1, 3]
    )

    with patch("torch.profiler.record_function", side_effect=lambda _: nullcontext()):
        with make_scope():
            pass
        with make_scope():
            pass

    assert events == [
        ("begin", sig._impl),
        ("end",),
        ("touch", sig._impl, False),
        ("begin", sig._impl),
        ("end",),
        ("touch", sig._impl, True),
    ]


@pytest.mark.parametrize("_capture_kind", ["empty", "non_replayable"])
def test_graph_capture_scope_does_not_count_passthrough_post_state(
    _capture_kind,
):
    make_scope, sig, events = _scope_fixture(
        (_RUNTIME, "_GraphCaptureScope"), [1], post_states=[0]
    )

    with patch("torch.profiler.record_function", side_effect=lambda _: nullcontext()):
        with make_scope():
            pass

    assert events == [("begin", sig._impl), ("end",)]


def test_graph_capture_scope_does_not_touch_on_body_or_end_failure(
    scope_implementation,
):
    make_scope, sig, events = _scope_fixture(scope_implementation, [3])

    with patch("torch.profiler.record_function", side_effect=lambda _: nullcontext()):
        with pytest.raises(ValueError, match="body"):
            with make_scope():
                raise ValueError("body")

    assert events == [("begin", sig._impl), ("abort",)]

    end_error = RuntimeError("end")
    make_scope, sig, events = _scope_fixture(
        scope_implementation, [3], end_error=end_error
    )
    with patch("torch.profiler.record_function", side_effect=lambda _: nullcontext()):
        with pytest.raises(RuntimeError, match="end"):
            with make_scope():
                pass

    assert events == [("begin", sig._impl), ("end",), ("abort",)]


@pytest.fixture
def retained_capture(monkeypatch):
    from rpu_backend.graph import _runtime as runtime

    events = []
    faults = {}

    class Entry:
        def __init__(self):
            self.borrowed = weakref.WeakSet()
            self.status = 0

        def begin(self, sig):
            events.append("begin")
            if "begin" in faults:
                raise faults["begin"]
            self.status = 1

        def end(self):
            events.append("end")
            if "end" in faults:
                raise faults["end"]
            self.status = 2

        def abort(self):
            events.append("abort")
            self.status = 0
            if "abort" in faults:
                raise faults["abort"]

        def state(self):
            if "state" in faults:
                raise faults["state"]
            return self.status

    class BorrowedGraph:
        def __init__(self, entry):
            entry.borrowed.add(self)
            # Like a native method, a failing operation's Python traceback must
            # not itself own this external shared_ptr wrapper.
            for name in ("begin", "end", "abort", "state"):
                setattr(self, name, getattr(entry, name))

    class NativeCache:
        def __init__(self):
            self.entries = {}

        def get_or_create(self, sig):
            return BorrowedGraph(self.entries.setdefault(sig, Entry()))

        def touch_entry(self, sig, replay):
            events.append("touch")

        def evict(self, sig):
            if self.entries[sig].status == 1:
                raise RuntimeError("cannot evict an active graph")
            if self.entries[sig].borrowed:
                raise RuntimeError("cannot evict a graph with external references")
            del self.entries[sig]
            events.append("evict")

        def clear(self):
            if any(entry.status == 1 for entry in self.entries.values()):
                raise RuntimeError("cannot clear an active graph")
            if any(entry.borrowed for entry in self.entries.values()):
                raise RuntimeError("cannot clear a graph with external references")
            self.entries.clear()
            events.append("clear")

    class Profiler:
        def __enter__(self):
            if "profiler-enter" in faults:
                raise faults["profiler-enter"]
            return self

        def __exit__(self, *args):
            events.append("profiler-exit")
            if "profiler" in faults:
                raise faults["profiler"]

    def skip_idle():
        if "skip-idle" in faults:
            raise faults["skip-idle"]
        return False

    monkeypatch.setattr(runtime, "_skip_idle_record_function", skip_idle)
    monkeypatch.setattr("torch.profiler.record_function", lambda _: Profiler())
    cache = runtime.GraphCache()
    cache._impl = NativeCache()
    sig = SimpleNamespace(_impl="retained-failure", op_id="cpu-reference-lifetime")
    return SimpleNamespace(cache=cache, sig=sig, faults=faults, events=events)


@pytest.mark.parametrize("phase,abort_failure,profiler_failure", [
    ("begin", False, False),
    ("body", False, False), ("body", True, False),
    ("body", False, True), ("body", True, True),
    ("end", False, False), ("end", True, False),
    ("end", False, True), ("end", True, True),
    ("profiler", False, False),
    ("state", False, False), ("state", True, False),
    ("skip-idle", False, False), ("skip-idle", True, False),
    ("profiler-enter", False, False), ("profiler-enter", True, False),
])
def test_failed_capture_releases_scope_reference_while_primary_traceback_is_alive(
    retained_capture, phase, abort_failure, profiler_failure,
):
    case = retained_capture
    if phase in {"state", "skip-idle", "profiler-enter"}:
        primary = KeyboardInterrupt(phase + " interrupted")
    else:
        primary = RuntimeError("build_batch SDK rc=5" if phase == "end" else phase + " failed")
    if phase != "body":
        case.faults[phase] = primary
    if abort_failure:
        case.faults["abort"] = RuntimeError("abort failed")
    if profiler_failure:
        case.faults["profiler"] = RuntimeError("profiler failed")
    scope = case.cache.capture(case.sig)
    try:
        with scope:
            if phase == "body":
                raise primary
    except BaseException as error:
        assert error is primary
        assert error.__traceback__ is not None
        notes = getattr(error, "__notes__", ())
        assert any("abort" in note.lower() for note in notes) == abort_failure
        assert any("profiler" in note.lower() for note in notes) == profiler_failure
        # The retained traceback and scope are intentionally alive here. These
        # are the real wrapper calls made by Vision's except and final close.
        case.cache.evict(case.sig)
        case.cache.clear()
    else:
        pytest.fail("capture should fail")
    assert not case.cache._impl.entries


def test_profiler_entry_exception_keeps_existing_nonfatal_capture_behavior(retained_capture):
    case = retained_capture
    case.faults["profiler-enter"] = RuntimeError("optional profiler unavailable")
    scope = case.cache.capture(case.sig)
    with scope:
        pass
    case.cache.evict(case.sig)
    case.cache.clear()
    assert case.events == ["begin", "end", "touch", "evict", "clear"]


def test_capture_releases_only_its_own_reference(retained_capture):
    case = retained_capture
    scope = case.cache.capture(case.sig)
    with scope as borrowed:
        pass
    # A caller that retains the returned graph remains a real external owner.
    with pytest.raises(RuntimeError, match="external references"):
        case.cache.evict(case.sig)
    with pytest.raises(RuntimeError, match="external references"):
        case.cache.clear()
    del borrowed
    case.cache.evict(case.sig)
    case.cache.clear()
    assert case.events == ["begin", "end", "touch", "profiler-exit", "evict", "clear"]
