"""Session publication checks, not artifact verification or hardware evidence."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def modules(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "python/rpu_backend"
    for name in ("rpu_backend", "rpu_backend.api", "rpu_backend.runtime"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    class BackendError(Exception):
        pass
    sys.modules["rpu_backend.runtime"].RPUBackendError = BackendError
    result = []
    for suffix in ("api._execution", "runtime.execution_planner"):
        name = "rpu_backend." + suffix
        spec = importlib.util.spec_from_file_location(name, root / (suffix.replace(".", "/") + ".py"))
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        result.append(module)
    return result


def make_session(api, **kwargs):
    return api.bind_execution_session(SimpleNamespace(), {}, entry_point="cost-test",
                                      graph_mode="RETAINED_CACHE", **kwargs)


def binding(planner, *, sealed=True):
    scope = planner.PlannerCostScope("a" * 64, "b" * 64, "c" * 64)
    return scope, (planner.PlannerCostCertificate(
        scope, "d" * 64, (("e" * 64, 42),), "f" * 64, sealed),)


def test_costs_are_cold_owner_local_and_survive_chunk_generation(modules):
    api, planner = modules
    components = {"text": {"prefill": ("chunk_size",)}, "vision": {"vision": ("chunk_size",)}}
    session = make_session(api, supported_components=components, apply=lambda *_: None)
    other = make_session(api, supported_components=components)
    installed = binding(planner)
    session._install_planner_costs("text", "prefill", lambda: installed)
    assert session.planner_costs("text", "prefill") == installed
    assert session.planner_costs("vision", "vision") == (None, ())
    assert other.planner_costs("text", "prefill") == (None, ())
    with pytest.raises(RuntimeError, match="immutable"):
        session._install_planner_costs("text", "prefill", lambda: installed)
    session.reconfigure({"components": {"text": {"prefill": {"chunk_size": 32}}}})
    assert session.generation == 1 and session.planner_costs("text", "prefill") == installed
    with session.execute():
        pass
    with pytest.raises(RuntimeError, match="before the first forward"):
        session._install_planner_costs("vision", "vision", lambda: installed)


@pytest.mark.parametrize("domain_scoped", [False, True])
def test_calibration_uses_existing_retirement_transaction_and_exact_owner(modules, domain_scoped):
    api, planner = modules
    events = []
    def apply(old, new, generation, *, force_rebuild=False):
        events.append((old, new, generation))
        assert force_rebuild or old != new
    session = make_session(api, supported={"prefill": ("chunk_size",)},
                           apply=apply)
    owner = session._owner
    api.record_planner_native(owner, "prefill", "", ("causal_decoder", 7))
    sibling_session = make_session(api, supported={"prefill": ("chunk_size",)})
    sibling = sibling_session._owner
    api.record_planner_native(sibling, "prefill", "", ("causal_decoder", 9))
    selection = planner.PlannerCalibrationSelection(
        binding(planner)[0], "d" * 64, "e" * 64, domain_scoped=domain_scoped)
    def select(value):
        session._select_planner_calibration(owner, "prefill", value, native_prefix="causal_decoder")
    select(selection)
    assert session.generation == 1 and session.config == {}
    assert events == [({}, {}, 1)]
    assert api.planner_calibration_selection(owner, "prefill", "", ("causal_decoder", 7)) is selection
    assert api.planner_calibration_selection(owner, "prefill", "", ("other", 7)) is None
    assert api.planner_calibration_selection(sibling, "prefill", "", ("causal_decoder", 9)) is None
    with session.execute():
        with pytest.raises(RuntimeError, match="quiescent"):
            select(None)
    with pytest.raises(RuntimeError, match="clear P7"):
        session.reconfigure({"prefill": {"chunk_size": 32}})
    select(None)
    assert session.generation == 2 and session.config == {}
    assert api.planner_calibration_selection(owner, "prefill", "", ("causal_decoder", 7)) is None
    session.reconfigure({"prefill": {"chunk_size": 32}})
    assert session.generation == 3


def test_native_exact_calibration_transaction_keeps_cold_cap_config_and_memo(modules):
    from test_execution_planner import _native_candidate_v2

    api, planner = modules
    events = []
    owner = SimpleNamespace(cold_cap=64, prefill_memo=object(), cache_position=0)
    session = api.bind_execution_session(owner, {"prefill": {"chunk_size": 128, "padding_rows": 0}},
        entry_point="exact-native-calibration-test", graph_mode="RETAINED_CACHE",
        supported={"prefill": ("chunk_size", "padding_rows")},
        apply=lambda old, new, generation, **kwargs: events.append((old, new, generation)))
    api.record_planner_native(owner, "prefill", "", ("causal_decoder", 7))
    session.require_cold()
    original = session.config
    memo = owner.prefill_memo
    scope = planner.PlannerCostScope("a" * 64, "b" * 64, "c" * 64)
    words = _native_candidate_v2(128, (128,) * 3, descriptor_version=3)
    def plan():
        selection = api.planner_calibration_selection(owner, "prefill", "", ("causal_decoder", 7))
        return planner.plan_prefill(128, 128, 0, exact_chunk_size=session.config["prefill"]["chunk_size"],
            padding_rows=session.config["prefill"]["padding_rows"], cost_scope=scope,
            calibration_selection=selection, calibration_mint=lambda length, row, route: row,
            resolve_stage_domain=lambda _length: (1, 1, len(words), *words))
    baseline = plan()
    selection = planner.PlannerCalibrationSelection(scope, baseline.cost_domain_digest,
        baseline.selected_cost_candidate_digest, (501, 0, 2))
    session._select_planner_calibration(owner, "prefill", selection, native_prefix="causal_decoder")
    actual = plan()
    assert actual.request == baseline.request and actual.request.mode == "EXACT"
    assert actual.optimality == "P7_CALIBRATION_ONLY" and actual.cost_certificate is None
    assert session.config == original and events == [(original, original, 1)]
    assert dict(original["prefill"]) == {"chunk_size": 128, "padding_rows": 0}
    assert owner.cold_cap == 64 and owner.prefill_memo is memo and owner.cache_position == 0
    with pytest.raises(RuntimeError, match="before the first forward"):
        session.require_cold()
    session._select_planner_calibration(owner, "prefill", None, native_prefix="causal_decoder")
    assert session.config == original and session.generation == 2
    assert plan().request == baseline.request


@pytest.mark.parametrize("domain_scoped", [False, True])
def test_calibration_failure_rolls_back_selection_without_borrowing_an_owner(modules, domain_scoped):
    api, planner = modules
    events = []
    def fail(*_, force_rebuild=False):
        assert force_rebuild
        raise RuntimeError("retire failed")
    session = make_session(api, apply=fail, rollback=lambda *_: events.append("rollback"))
    owner = session._owner
    selection = planner.PlannerCalibrationSelection(
        binding(planner)[0], "d" * 64, "e" * 64, domain_scoped=domain_scoped)
    with pytest.raises(ValueError, match="native producer"):
        session._select_planner_calibration(owner, "prefill", selection, native_prefix="causal_decoder")
    api.record_planner_native(owner, "prefill", "", ("causal_decoder", 7))
    with pytest.raises(RuntimeError, match="retire failed"):
        session._select_planner_calibration(owner, "prefill", selection, native_prefix="causal_decoder")
    assert events == ["rollback"] and session.generation == 0
    assert session._planner_calibration is None
    assert session.stats()["state"] == "QUIESCENT"


def test_calibration_cannot_silently_use_an_unadopted_same_config_callback(modules):
    api, planner = modules
    session = make_session(api, apply=lambda old, new, generation: None,
                           rollback=lambda *_: None)
    owner = session._owner
    api.record_planner_native(owner, "prefill", "", ("causal_decoder", 7))
    selection = planner.PlannerCalibrationSelection(binding(planner)[0], "d" * 64, "e" * 64)
    with pytest.raises(TypeError, match="force_rebuild"):
        session._select_planner_calibration(owner, "prefill", selection, native_prefix="causal_decoder")
    assert session.generation == 0 and session._planner_calibration is None


@pytest.mark.parametrize("fault", [None, "layout_bool", "layout_zero", "topology", "manifest"])
def test_shared_mint_decodes_actual_native_manifest_and_preserves_adapter_metadata(modules, fault):
    _api, planner = modules
    from test_execution_planner import _native_candidate_v2

    def descriptor(chunk, attention):
        return _native_candidate_v2(
            64, (chunk, chunk, chunk), descriptor_version=3,
            routes=((1, 101, attention, 0, (), 0),))
    original = descriptor(64, 1)
    row, = planner._decode_native_stage_domain(
        (1, 1, len(original), *original), execution_len=64, logical_len=64,
        position=0, graph_mode="RETAINED_CACHE")
    row = planner.StageTuple.from_value(planner.StageTuple(
        *row.as_tuple(), (*row.physical_metadata, ("layout_hash_hi", 1),
                          ("layout_hash_lo", 2), ("adapter_geometry", 7)),
        row.physical_descriptor))
    changed = descriptor(32 if fault == "topology" else 64, 2)
    if fault == "manifest":
        changed[-4] += 1
    layout = (True, 4) if fault == "layout_bool" else (
        (0, 0) if fault == "layout_zero" else (3, 4))
    def mint(handle, raw, site, invocation, route):
        assert (handle, raw, site, invocation, route) == (17, original, 501, 0, 2)
        return changed, *layout
    path = Path(__file__).resolve().parents[1] / "python/rpu_backend/runtime/decoder.py"
    node = next(item for item in ast.parse(path.read_text()).body
                if isinstance(item, ast.FunctionDef)
                and item.name == "_mint_planner_calibration_candidate")
    namespace = {"torch": SimpleNamespace(ops=SimpleNamespace(rpu=SimpleNamespace(
        causal_decoder_kvinsert_exact_candidate=mint)))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    def call():
        return namespace[node.name](("causal_decoder", 17), 64, 64, 0,
                                     "RETAINED_CACHE", row, (501, 0, 2))
    if fault:
        with pytest.raises(ValueError):
            call()
    else:
        result = call()
        assert result.physical_descriptor == tuple(changed)
        assert result.as_tuple() == row.as_tuple()
        metadata = dict(result.physical_metadata)
        assert (metadata["layout_hash_hi"], metadata["layout_hash_lo"]) == (3, 4)
        assert metadata["adapter_geometry"] == 7
        assert metadata["manifest_fingerprint"] != dict(row.physical_metadata)["manifest_fingerprint"]


def test_invalid_target_is_rejected_before_installer(modules):
    api, _ = modules
    session = make_session(api, supported={"prefill": ("chunk_size",)})
    def unexpected():
        pytest.fail("installer must not run")
    for component, stage in (("unknown", "prefill"), ("", "vision")):
        with pytest.raises(ValueError):
            session._install_planner_costs(component, stage, unexpected)
    assert session.stats()["state"] == "QUIESCENT"


@pytest.mark.parametrize("fault", ["unsealed", "interrupted", "executed"])
def test_partial_native_installation_poison_prevents_forward(modules, fault):
    api, planner = modules
    session = make_session(api)
    def install():
        if fault == "interrupted":
            raise KeyboardInterrupt("after possible native mutation")
        if fault == "executed":
            with session.execute():
                pass
        return binding(planner, sealed=fault != "unsealed")
    with pytest.raises((KeyboardInterrupt, RuntimeError, ValueError)):
        session._install_planner_costs("", "prefill", install)
    assert session.stats()["state"] == "POISONED"
    with pytest.raises(RuntimeError, match="poisoned"):
        with session.execute():
            pass
    with pytest.raises(RuntimeError, match="poisoned"):
        session.planner_costs("", "prefill")


def test_closed_session_cannot_install_or_serve_costs(modules):
    api, planner = modules
    session = make_session(api)
    session.close()
    with pytest.raises(RuntimeError):
        session._install_planner_costs("", "prefill", lambda: binding(planner))
    with pytest.raises(RuntimeError, match="closed"):
        session.planner_costs("", "prefill")


def test_production_child_lookup_is_explicit_isolated_and_cold(modules):
    api, planner = modules
    components = {"text": {"prefill": ("chunk_size",)}, "vision": {"vision": ("chunk_size",)}}
    session = make_session(api, supported_components=components)
    other = make_session(api, supported_components=components)
    text, vision = SimpleNamespace(), SimpleNamespace()
    session._bind_planner_owner(text, "text")
    session._bind_planner_owner(vision, "vision")
    installed = binding(planner)
    session._install_planner_costs("text", "prefill", lambda: installed)
    assert api.planner_cost_binding(text, "prefill") == installed
    assert api.planner_cost_binding(vision, "vision") == (None, ())
    with pytest.raises(ValueError, match="different session"):
        other._bind_planner_owner(text, "text")
    with pytest.raises(ValueError, match="undeclared stage"):
        api.planner_cost_binding(text, "vision")
    with pytest.raises(ValueError, match="explicitly bound child"):
        api.planner_cost_binding(session._owner, "prefill")
    with session.execute():
        pass
    session._bind_planner_owner(text, "text")
    with pytest.raises(RuntimeError, match="before the first forward"):
        session._bind_planner_owner(text, "vision")
    with pytest.raises(RuntimeError, match="before the first forward"):
        session._bind_planner_owner(SimpleNamespace(), "text")
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        api.planner_cost_binding(text, "prefill")


def test_shared_production_planner_reads_owner_binding_without_a_second_cost_authority(modules):
    api, planner = modules
    session = make_session(api, supported={"prefill": ("chunk_size",)})
    installed = binding(planner)
    session._install_planner_costs("", "prefill", lambda: installed)
    path = Path(__file__).resolve().parents[1] / "python/rpu_backend/runtime/decoder.py"
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "plan_bounded_prefill_execution")
    received = []
    def plan(*args, **kwargs):
        received.append(kwargs)
        return SimpleNamespace(selected=SimpleNamespace(
            execution_len=64, stage_tuple=SimpleNamespace(compute_chunk=32)))
    namespace = {"os": os, "plan_prefill": plan}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    call = namespace[node.name]
    assert call(64, 64, 0, resolve_stage_domain=lambda _n: (),
                execution_owner=session._owner) == (64, 32)
    assert received[0]["cost_scope"] is installed[0]
    assert received[0]["cost_certificates"] is installed[1]
    with pytest.raises(ValueError, match="cannot override"):
        call(64, 64, 0, resolve_stage_domain=lambda _n: (),
             execution_owner=session._owner, cost_scope=installed[0])


def test_one_owner_with_two_physical_handles_requires_exact_component(modules):
    api, planner = modules
    session = make_session(api, supported_components={
        "first": {"prefill": ("chunk_size",)},
        "second": {"prefill": ("chunk_size",)},
    })
    owner = SimpleNamespace()
    session._bind_planner_owner(owner, "first")
    session._bind_planner_owner(owner, "second")
    installed = binding(planner)
    session._install_planner_costs("first", "prefill", lambda: installed)
    assert api.planner_cost_binding(owner, "prefill", "first") == installed
    assert api.planner_cost_binding(owner, "prefill", "second") == (None, ())
    with pytest.raises(ValueError, match="explicit component"):
        api.planner_cost_binding(owner, "prefill")
    with pytest.raises(ValueError, match="not bound"):
        api.planner_cost_binding(owner, "prefill", "unknown")


def test_native_facts_and_costs_do_not_cross_producer_prefixes(modules):
    api, planner = modules
    session = make_session(api, supported={"action": ("chunk_size",)})
    owner = session._owner
    api.record_planner_native(owner, "action", None, ("adarms", 7))
    api.record_planner_native(owner, "action", None, ("pi05_denoise_step", 9))
    assert session._planner_native[("", "action", "adarms")] == (owner, 7)
    installed = binding(planner)
    session._install_planner_costs("", "action", lambda: installed, native_prefix="adarms")
    assert api.planner_cost_binding(owner, "action", native_prefix="adarms") == installed
    assert api.planner_cost_binding(owner, "action", native_prefix="pi05_denoise_step") == (None, ())
    api.record_planner_native(owner, "action", None, ("adarms", 7))
    with pytest.raises(RuntimeError, match="revalidation"):
        api.record_planner_native(owner, "action", None, ("adarms", 8))
    for invalid in (("adarms", True), ("adarms", 0), ("../adarms", 1)):
        with pytest.raises(ValueError, match="literal prefix"):
            api.record_planner_native(owner, "action", None, invalid)


def test_cost_probe_is_scoped_cold_and_cleans_up_after_failure(modules):
    api, _ = modules
    session = make_session(api)
    owner = session._owner
    observed = []
    with session._collect_planner_costs(lambda *row: observed.append(row)):
        api.record_planner_native(owner, "prefill", None, ("causal_decoder", 7))
        observer = api.planner_cost_observer(owner, "prefill", None, ("causal_decoder", 7))
        observer("domain", (32, ()))
        assert observed == [(owner, "", "prefill", ("causal_decoder", 7), "domain", (32, ()))]
        with pytest.raises(ValueError, match="non-nested"):
            with session._collect_planner_costs(lambda *_: None):
                pass
    assert session._planner_cost_observer is None
    with pytest.raises(RuntimeError, match="prepare failed"):
        with session._collect_planner_costs(lambda *_: None):
            raise RuntimeError("prepare failed")
    assert session._planner_cost_observer is None
    with session.execute():
        pass
    with pytest.raises(RuntimeError):
        with session._collect_planner_costs(lambda *_: None):
            pass




@pytest.mark.parametrize("fault", ["forward", "reconfigure", "close", "native", "install",
    "cost_drift", "generation_drift", "closed", "poisoned", "active", "unsafe", "exception"])
def test_live_cost_read_rejects_mutation_and_clears_observer(modules, monkeypatch, fault):
    api, planner = modules
    session = make_session(api, apply=lambda *_: None)
    owner = session._owner
    api.record_planner_native(owner, "prefill", None, ("causal_decoder", 7))
    with session.execute():
        pass
    if fault in {"closed", "poisoned"}:
        setattr(session, "_" + fault, True)
    elif fault == "active":
        session._active = 1
    elif fault == "unsafe":
        monkeypatch.setattr(api, "_UNSAFE_PROCESS_REASON", "unsafe read")
    with pytest.raises((ValueError, RuntimeError)):
        with session._collect_planner_costs(lambda *_: None, read_only=True):
            if fault == "forward":
                with session.execute():
                    pytest.fail("read-only collection executed a forward")
            elif fault == "reconfigure":
                session.reconfigure({"prefill": {"chunk_size": 32}})
            elif fault == "close":
                session.shutdown(lambda: pytest.fail("read-only collection retired the owner"))
            elif fault == "native":
                api.record_planner_native(owner, "prefill", None, ("causal_decoder", 8))
            elif fault == "install":
                session._install_planner_costs("", "prefill", lambda: binding(planner))
            elif fault == "cost_drift":
                session._planner_costs[("", "prefill", "causal_decoder")] = binding(planner)
            elif fault == "generation_drift":
                session._generation += 1
            elif fault == "exception":
                raise RuntimeError("native query failed")
    assert session._planner_cost_observer is None
    if fault not in {"cost_drift", "generation_drift", "closed", "poisoned", "active", "unsafe"}:
        assert session.stats()["state"] == "QUIESCENT" and session.generation == 0
        assert session._planner_native[("", "prefill", "causal_decoder")][1] == 7


def _prepared_caller(planner):
    """Exercise the real text caller and shared facade without a device import."""
    import torch
    path = Path(__file__).resolve().parents[1] / 'python/rpu_backend/runtime/decoder.py'
    names = {'plan_bounded_prefill_execution', '_text_prefill_execution_plan'}
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(os=os, torch=torch, plan_prefill=planner.plan_prefill,
                     PlannerRejectError=planner.PlannerRejectError,
                     native_chunk_reject=planner.native_chunk_reject)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def test_prepared_text_caller_reuses_shapes_and_stops_ready_misses_before_search(modules, monkeypatch):
    import torch
    from test_execution_planner import _native_domain
    from test_graph_cache_freeze import _cache, _sig

    api, planner = modules
    graph_cache = _cache()
    owner = SimpleNamespace(_rpu_decoder_graph_cache=graph_cache)
    session = api.bind_execution_session(
        owner, {'prefill': {'padding_budget': 0}}, entry_point='prepared-text',
        supported={'prefill': ('chunk_size', 'padding_budget')},
        apply=lambda *_: graph_cache.clear(), graph_mode='RETAINED_CACHE')
    epoch, queries = [1], []
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_planner_cache_identity',
                        lambda handle: (1, epoch[0]), raising=False)
    def domain(handle, length, position, *args):
        queries.append((handle, length, position))
        return _native_domain(length, (32, 32, 32), position=position)
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_resolve_prefill_stage_domain', domain, raising=False)
    namespace = _prepared_caller(planner)
    owner_bindings = []
    resolve_owner = api._planner_owner_binding
    monkeypatch.setattr(api, '_planner_owner_binding', lambda *args:
                        owner_bindings.append(args) or resolve_owner(*args))
    call = namespace['_text_prefill_execution_plan']
    kv = SimpleNamespace(position=0, max_seq_len=128)
    def forward(length):
        with session.execute():
            before = len(owner_bindings)
            plan = call(owner, 7, kv, length)[2]
            assert len(owner_bindings) == before + 1
            with graph_cache.capture(_sig((length, plan.graph_key_words()))):
                pass
            assert owner._rpu_last_a6_plan == plan.as_dict(include_candidates=False)
            return plan
    first, second = forward(64), forward(80)
    searches = len(queries)
    assert forward(64) is first and forward(80) is second
    assert len(queries) == searches
    graph_cache.freeze()
    assert forward(64) is first
    with pytest.raises(RuntimeError, match='plan READY miss'):
        forward(96)
    assert len(queries) == searches
    for missing_owner, signature in ((owner, None), (None, ())):
        with pytest.raises(RuntimeError, match='READY requires'):
            namespace['plan_bounded_prefill_execution'](
                64, 128, 0, execution_owner=missing_owner,
                plan_signature=signature, graph_cache=graph_cache,
                resolve_stage_domain=lambda length: queries.append(length))
    assert len(queries) == searches
    epoch[0] += 1
    with pytest.raises(RuntimeError, match='plan READY miss'):
        forward(64)
    assert len(queries) == searches
    graph_cache.begin_warmup()
    assert forward(64) is not first
    session.reconfigure({'prefill': {'chunk_size': 32, 'padding_budget': 0}})
    assert len(graph_cache._execution_plans) == 0
    assert forward(64).request.chunk_size == 32


def test_prepared_cost_bound_plan_reuses_selection_but_observer_queries_and_failures_retry(modules, monkeypatch):
    import torch
    api, planner = modules
    session = make_session(api, supported={'prefill': ('chunk_size',)}, apply=lambda *_: None)
    session._install_planner_costs('', 'prefill', lambda: binding(planner), native_prefix='causal_decoder')
    namespace = _prepared_caller(planner)
    queries, fail = [], [False]
    def select(*args, **kwargs):
        queries.append(kwargs)
        if fail[0]:
            raise ValueError('invalid native domain')
        observer = kwargs['cost_domain_observer']
        if observer is not None:
            observer(64, ())
        return SimpleNamespace(selected=SimpleNamespace(
            execution_len=64, stage_tuple=SimpleNamespace(compute_chunk=32)))
    namespace['plan_prefill'] = select
    monkeypatch.setattr(torch.ops.rpu, 'causal_decoder_planner_cache_identity',
                        lambda handle: (1, 1), raising=False)
    def call(signature=()):
        return namespace['plan_bounded_prefill_execution'](
            64, 64, 0, execution_owner=session._owner,
            execution_native=('causal_decoder', 7), plan_signature=signature,
            resolve_stage_domain=lambda length: ())
    assert call() == call() == (64, 32)
    assert len(queries) == 1
    assert queries[0]['cost_scope'] == binding(planner)[0]
    observed = []
    with session._collect_planner_costs(lambda *event: observed.append(event)):
        assert call() == (64, 32)
    assert len(queries) == 2 and observed
    assert call() == (64, 32) and len(queries) == 2
    fail[0] = True
    with pytest.raises(ValueError, match='invalid native domain'):
        call(('new-mask',))
    fail[0] = False
    assert call(('new-mask',)) == (64, 32) and len(queries) == 4
    session.reconfigure({'prefill': {'chunk_size': 32}})
    assert call() == (64, 32) and len(queries) == 5
    session.close()
    with pytest.raises(RuntimeError, match='closed'):
        call()


def test_planner_context_reads_current_cost_observer_calibration_and_binding(modules):
    api, planner = modules
    session = make_session(api, supported={"prefill": ("chunk_size",)}, apply=lambda *a, **kw: None)
    owner, native = session._owner, ("causal_decoder", 7)
    context = lambda: api.planner_execution_context(owner, "prefill", None, native)
    assert context() == (None, (), None, None)
    installed = binding(planner)
    session._install_planner_costs("", "prefill", lambda: installed, native_prefix=native[0])
    assert context() == (*installed, None, None)
    events = []
    with session._collect_planner_costs(lambda *event: events.append(event)):
        scope, certificates, observer, calibration = context()
        assert (scope, certificates) == installed and calibration is None
        observer("domain", (64, ()))
        assert events == [(owner, "", "prefill", native, "domain", (64, ()))]
        with pytest.raises(ValueError, match="actual planner's native binding"):
            api.planner_execution_context(owner, "prefill", None, None)
    assert context() == (*installed, None, None)
    selection = planner.PlannerCalibrationSelection(installed[0], "d" * 64, "e" * 64)
    session._select_planner_calibration(owner, "prefill", selection, native_prefix=native[0])
    assert context() == (*installed, None, selection)
    assert api.planner_execution_context(owner, "prefill", None, ("other", 7)) == (None, (), None, None)
    session._select_planner_calibration(owner, "prefill", None, native_prefix=native[0])
    assert context() == (*installed, None, None)
    # A changed owner session cannot inherit the former owner's cached authority.
    other = make_session(api, supported={"prefill": ("chunk_size",)})
    owner._execution_session = other
    assert context() == (None, (), None, None)
    owner._execution_session = session
    with pytest.raises(ValueError, match="undeclared stage"):
        api.planner_execution_context(owner, "vision", None, native)
    session.poison()
    with pytest.raises(RuntimeError, match="poisoned"):
        context()


def test_planner_context_stable_native_still_revalidates_replacement_and_rejects_alias(modules):
    api, planner = modules
    session = make_session(api, supported={"prefill": ("chunk_size",)})
    owner = session._owner
    call = lambda handle: api.planner_execution_context(owner, "prefill", None, ("causal_decoder", handle))
    call(7)
    installed = binding(planner)
    session._install_planner_costs("", "prefill", lambda: installed, native_prefix="causal_decoder")
    key = ("", "causal_decoder")
    session._planner_native_catalogs[key] = (7, "exact-native-catalog")
    revalidated = []
    def revalidate(native):
        revalidated.append(native)
        # Recursive cost inspection must not consume the unverified parent's costs.
        assert call(native[1]) == (None, (), None, None)
        return (native[1], "exact-native-catalog")
    session._planner_native_revalidators[key] = revalidate
    assert call(7) == (*installed, None, None) and revalidated == []
    assert call(8) == (*installed, None, None)
    assert revalidated == [("causal_decoder", 8)]
    assert session._planner_native_catalogs[key][0] == 8
    assert call(8) == (*installed, None, None) and len(revalidated) == 1
    # Same recorded handle is insufficient if another stage replaced its proof.
    session._planner_native_catalogs[key] = (9, "exact-native-catalog")
    assert call(8) == (*installed, None, None) and len(revalidated) == 2
    with pytest.raises(ValueError, match="literal prefix"):
        call(True)
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        call(8)
