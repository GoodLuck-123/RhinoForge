"""Fixed decode planning/collection on host stubs; never loads the device SDK."""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_execution_session_costs import modules, binding  # noqa: F401
from test_execution_planner import _native_candidate_v2


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "python/rpu_backend/runtime/decoder.py"


@pytest.fixture
def decode(modules):
    api, planner = modules
    owner = SimpleNamespace()
    session = api.bind_execution_session(
        owner, {}, entry_point="host-decode", supported={"prefill": ("chunk_size",)},
        graph_mode="RETAINED_CACHE")
    descriptor = _native_candidate_v2(
        1, (16,) * 3, descriptor_version=3, routes=((1, 101, 2, 0, (), 0),))
    calls = []
    def native(handle, position=0):
        calls.append((handle, position) if position else handle)
        return (_native_candidate_v2(1, (16, 16, 16), position=position,
                    descriptor_version=3, routes=((1, 101, 2, 0, (), 0),))
                if position else descriptor)
    scope = dict(os=os, plan_prefill=planner.plan_prefill,
                 GRAPH_RETAINED_CACHE="RETAINED_CACHE",
                 PlannerRejectError=planner.PlannerRejectError,
                 torch=SimpleNamespace(ops=SimpleNamespace(rpu=SimpleNamespace(
                     causal_decoder_resolve_decode_stage_descriptor=native,
                     causal_decoder_planner_cache_identity=lambda handle: (1, 1)))))
    names = {"plan_bounded_prefill_execution", "plan_native_component_execution",
             "_text_decode_execution_plan"}
    nodes = [node for node in ast.parse(SOURCE.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), scope)
    return SimpleNamespace(api=api, planner=planner, owner=owner, session=session,
                           descriptor=descriptor, calls=calls, plan=scope["_text_decode_execution_plan"])


def test_fixed_decode_native_winner_is_memoized_and_cost_scope_invalidates_it(decode):
    first = decode.plan(decode.owner, 7)
    assert first.search_complete and not first.truncated
    assert first.selected.stage_tuple.physical_descriptor == tuple(decode.descriptor)
    assert decode.owner._rpu_decode_stage_descriptor == tuple(decode.descriptor)
    assert decode.plan(decode.owner, 7) is first
    assert decode.calls == [7]
    decode.session._install_planner_costs(
        "", "decode", lambda: binding(decode.planner), native_prefix="causal_decoder")
    second = decode.plan(decode.owner, 7)
    assert second is not first and second.cost_scope is not None
    assert decode.calls == [7, 7]
    # Ordinary cost-bound decode reuses the immutable selection. Explicit
    # observation below still asks the current oracle for a fresh cost domain.
    third = decode.plan(decode.owner, 7)
    assert third is second
    assert decode.calls == [7, 7]
    assert second.optimality != "HARDWARE_COST_CERTIFIED"  # Unmatched host fixture certificate.


def test_cold_observer_always_runs_actual_decode_oracle_and_decode_stays_nonconfigurable(decode):
    decode.plan(decode.owner, 7)
    observations = []
    with decode.session._collect_planner_costs(lambda *row: observations.append(row)):
        result = decode.plan(decode.owner, 7)
    assert decode.calls == [7, 7]
    assert [row[-2] for row in observations] == ["domain", "result"]
    assert all(row[2:4] == ("decode", ("causal_decoder", 7)) for row in observations)
    assert observations[-1][-1] == (1, 0, result)
    with pytest.raises(ValueError, match="unknown stage"):
        decode.session.reconfigure({"decode": {"chunk_size": 16}})


def test_decode_oracle_is_outside_production_capture():
    tree = ast.parse(SOURCE.read_text())
    forward = next(node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == "rpu_decoder_model_forward")
    calls = [node for node in ast.walk(forward) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_text_decode_execution_plan"]
    capture = next(node for node in ast.walk(forward)
                   if isinstance(node, ast.FunctionDef) and node.name == "_capture")
    assert len(calls) == 1 and calls[0].lineno < capture.lineno


def test_scalar_position_decode_plans_are_distinct_and_preserve_default(decode):
    default = decode.plan(decode.owner, 7)
    first = decode.plan(decode.owner, 7, position=576)
    second = decode.plan(decode.owner, 7, position=577)
    assert first.selected.stage_tuple.physical_descriptor != second.selected.stage_tuple.physical_descriptor
    assert decode.plan(decode.owner, 7, position=576) is first
    assert decode.calls == [7, (7, 576), (7, 577)]
    assert decode.owner._rpu_decode_stage_descriptor == default.selected.stage_tuple.physical_descriptor


def test_real_generic_decode_capture_uses_full_plan_digest_and_retains_native_descriptor(decode, monkeypatch):
    from contextlib import nullcontext
    import torch

    class Cache:
        position = 7
        max_seq_len = 128
    cache = Cache()
    captures, dispatched = [], []
    prepared = decode.planner.PreparedExecutionPlans()
    owner = decode.owner
    owner._rpu_decoder_handle = 7
    owner._rpu_decoder_hidden_size = 4
    owner._rpu_decoder_num_layers = 1
    owner._rpu_decoder_deepstack_hash = 0
    owner._rpu_batch_decode_enabled = False
    owner.config = SimpleNamespace(hidden_size=4)
    owner._rpu_decoder_graph_cache = SimpleNamespace(
        is_frozen=lambda: False, prepare_plan=prepared.prepare,
        capture=lambda signature: captures.append(signature) or nullcontext())
    def run(model, handle, **kwargs):
        dispatched.append(model._rpu_decode_stage_descriptor)
        cache.position += 1
        return torch.zeros((1, 1, 4)), cache
    forward = next(node for node in ast.walk(ast.parse(SOURCE.read_text()))
                   if isinstance(node, ast.FunctionDef) and node.name == "rpu_decoder_model_forward")
    namespace = dict(torch=torch, topology=None, arch="qwen3", _RPUCache_for_sig=Cache,
        validate_decoder_cache_topology=lambda *a, **kw: None,
        _is_batched_prefill=lambda *a: False,
        _canonicalize_plain_text_controls=lambda *a, **kw: (None, None, None),
        _reject_unconsumed_text_padding=lambda *a: None,
        _text_decode_execution_plan=decode.plan,
        _GraphSignature=lambda **kw: kw, chunk_policy_key=lambda handle: 0,
        prefill_position_key=lambda length, position: 0,
        _run_causal_decoder_forward=run, _validate_decoder_base_output=lambda *a: None,
        BaseModelOutputWithPast=SimpleNamespace)
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(SOURCE), "exec"), namespace)
    call = lambda: namespace[forward.name](owner, input_ids=torch.zeros((1, 1), dtype=torch.long),
                                           past_key_values=cache)
    call()
    first = owner._decoder_decode_plan[1]
    call()
    assert len(decode.calls) == 1 and captures[0] == captures[1]
    assert tuple(captures[0]["dyn_dims"][-8:]) == first.graph_key_words()
    assert len(captures[0]["dyn_dims"]) == 14
    assert dispatched == [first.selected.stage_tuple.physical_descriptor] * 2
    assert len(dispatched[0]) > 8  # Native dispatch retains every descriptor word.
    # A changed native epoch requires a fresh plan; its changed physical route
    # must change the graph signature while dispatch still receives full words.
    changed = _native_candidate_v2(1, (16,) * 3, descriptor_version=3,
                                   routes=((1, 101, 1, 0, (), 0),))
    decode.descriptor[:] = changed
    monkeypatch.setattr(decode.plan.__globals__["torch"].ops.rpu,
                        "causal_decoder_planner_cache_identity", lambda handle: (1, 2))
    call()
    replacement = owner._decoder_decode_plan[1]
    assert captures[2] != captures[0]
    assert tuple(captures[2]["dyn_dims"][-8:]) == replacement.graph_key_words()
    assert dispatched[-1] == tuple(changed)
