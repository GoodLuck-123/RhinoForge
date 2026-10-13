"""Actual HALO adapter/planner contracts with CPU-only device-operation doubles."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.halo import text_decode
from rpu_backend.adapters.halo.runtime import (
    HALO_TEXT_DECODE_COMPONENT,
    HaloExecutionController,
)
from test_execution_planner import _native_candidate, _native_candidate_v2


def _descriptor(*, length=1, chunks=(16, 16, 16), lifecycle=1, selector=1):
    # Codec fixture only: production obtains these words from the installed
    # native handle. This does not exercise or certify its kernel geometry.
    return tuple(_native_candidate_v2(
        length, chunks, descriptor_version=3, graph_lifecycle=lifecycle,
        routes=((1, 101, selector, 0, (), 0),),
    ))


@pytest.fixture
def runtime(monkeypatch):
    events = []

    class GraphCache:
        def is_frozen(self):
            return False

        @contextmanager
        def capture(self, signature):
            events.append(("capture", signature))
            yield
            events.append(("executed",))

        def prepare_plan(self, key, prepare):
            return prepare()

        def clear(self):
            events.append(("clear",))

    def update_position(rows):
        events.append(("position", rows))

    cache = SimpleNamespace(k_caches=[torch.zeros(1)], v_caches=[torch.zeros(1)],
                            update_position=update_position)
    runner = text_decode.HaloTextDecodeStep(
        cfg=SimpleNamespace(num_layers=1, hidden_size=2), text=None,
        host=SimpleNamespace(lm_head_w=torch.eye(2)), cache=cache, max_seq_len=32,
        _handle=17, _graph_cache=GraphCache(), _cos=None, _sin=None,
        _decode_descriptor=_descriptor(),
    )
    controller = HaloExecutionController()
    controller.attach(runner, {HALO_TEXT_DECODE_COMPONENT: (runner._graph_cache,)})
    native = SimpleNamespace(descriptor=runner._decode_descriptor)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_planner_cache_identity",
                        lambda handle: (handle, native.descriptor), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_resolve_decode_stage_descriptor",
                        lambda handle: native.descriptor, raising=False)

    def inject(keys, values, position):
        events.append(("inject", position))
        runner._prefix_epoch += 1

    def embed(_token):
        events.append(("embed",))
        return torch.tensor([[[0.25, 0.75]]])

    def forward(*args, **kwargs):
        assert controller._execution_session.stats()["state"] == "EXECUTING"
        assert HALO_TEXT_DECODE_COMPONENT not in vars(runner).get("_rpu_last_execution_receipts", {})
        events.append(("native", args, kwargs))
        return args[1]

    monkeypatch.setattr(runner, "_inject_prefix", inject)
    monkeypatch.setattr(runner, "_embed_token", embed)
    monkeypatch.setattr(text_decode.torch.ops.rpu, "causal_decoder_forward", forward, raising=False)
    prefix = ([torch.zeros(3, 1, 2)], [torch.zeros(3, 1, 2)])
    return SimpleNamespace(runner=runner, controller=controller, events=events, prefix=prefix, native=native)


def test_native_decode_template_drives_the_fixed_plan_and_actual_forward(runtime):
    runner = runtime.runner
    plan = runner._execution_plan(3)
    assert plan.graph_mode == "RETAINED_CACHE"
    assert plan.selection_scope == "STAGE_DOMAIN_ROUTE_CAPABILITY_SELECTED"
    assert plan.candidate_count == plan.feasible_count == 1
    assert plan.selected.stage_tuple.physical_descriptor == runner._decode_descriptor
    metadata = dict(plan.selected.stage_tuple.physical_metadata)
    assert metadata["stage:decode"] == 1 and "stage:prefill" not in metadata
    assert metadata["graph_lifecycle"] == metadata["manifest_state"] == 1
    assert metadata["position"] == 3
    # The native decode template is position-zero; the real KV/RoPE offsets
    # remain independent forward arguments, not rewritten descriptor words.
    assert metadata["kv_logical_length"] == 1

    hidden, logits, token = runner.decode_step(torch.tensor([7]), torch.tensor([2]), runtime.prefix)
    assert torch.equal(hidden, torch.tensor([0.25, 0.75]))
    assert torch.equal(logits, hidden) and token == 1
    names = [event[0] for event in runtime.events]
    assert names == ["inject", "embed", "capture", "native", "executed", "position"]
    _, args, kwargs = next(event for event in runtime.events if event[0] == "native")
    assert args[0] == runner._handle and args[2] is runner.cache.k_caches
    assert args[3] is runner.cache.v_caches and args[4:9] == (None, 3, True, None, [])
    assert kwargs == {"cos_sin_offset": 2, "planned_stage_descriptor": runner._decode_descriptor}
    receipt = runner._rpu_last_execution_receipts[HALO_TEXT_DECODE_COMPONENT]
    assert receipt["stage"] == "decode" and receipt["graph_mode"] == "RETAINED_CACHE"
    assert receipt["authority"] == "NATIVE_FMB_DESCRIPTOR"
    assert receipt["selected"]["physical_descriptor"] == kwargs["planned_stage_descriptor"]
    assert receipt["kv_route"] == "DDR_REQUIRED"
    assert runtime.controller._execution_session.stats()["state"] == "QUIESCENT"


@pytest.mark.parametrize("fault", ["child_lifecycle", "prefill_shape", "asymmetric", "legacy", "fingerprint", "boolean"])
def test_bad_native_template_fails_before_any_device_boundary(runtime, fault):
    runner = runtime.runner
    if fault == "child_lifecycle":
        descriptor = _descriptor(lifecycle=4)
    elif fault == "prefill_shape":
        descriptor = _descriptor(length=2)
    elif fault == "asymmetric":
        descriptor = _descriptor(chunks=(1, 16, 16))
    elif fault == "legacy":
        descriptor = tuple(_native_candidate(1, (16,) * 3))
    else:
        descriptor = list(_descriptor())
        descriptor[-1 if fault == "fingerprint" else 0] = 1 if fault == "fingerprint" else True
        descriptor = tuple(descriptor)
    runtime.native.descriptor = descriptor
    with pytest.raises(ValueError):
        runner.decode_step(torch.tensor([7]), torch.tensor([2]), runtime.prefix)
    assert runtime.events == []
    assert not hasattr(runner, "_rpu_last_execution_receipts")


def test_fixed_plan_identity_tracks_actual_descriptor_position_and_controller_generation(runtime):
    runner = runtime.runner
    first = runner._execution_plan(3)
    assert runner._execution_plan(4).physical_plan_digest != first.physical_plan_digest
    runtime.native.descriptor = _descriptor(selector=2)
    second = runner._execution_plan(3)
    assert second.physical_plan_digest != first.physical_plan_digest
    runtime.controller._component_generations[HALO_TEXT_DECODE_COMPONENT] += 1
    third = runner._execution_plan(3)
    assert third.physical_plan_digest != second.physical_plan_digest
    with pytest.raises(ValueError):
        runtime.controller.reconfigure_rpu_execution({"components": {"text_decode": {"decode": {"chunk_size": 16}}}})


def test_failed_native_forward_does_not_publish_a_decode_receipt(runtime, monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("native rejected descriptor")

    monkeypatch.setattr(text_decode.torch.ops.rpu, "causal_decoder_forward", fail)
    with pytest.raises(RuntimeError, match="native rejected descriptor"):
        runtime.runner.decode_step(torch.tensor([7]), torch.tensor([2]), runtime.prefix)
    assert not hasattr(runtime.runner, "_rpu_last_execution_receipts")
    assert "position" not in [event[0] for event in runtime.events]


def test_cold_collection_observes_the_live_fixed_decode_owner(runtime, monkeypatch):
    runner = runtime.runner
    current = _descriptor(selector=2)
    calls, observations = [], []

    def native(handle):
        calls.append(handle)
        return current

    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_resolve_decode_stage_descriptor",
                        native, raising=False)
    session = runtime.controller._execution_session
    with session._collect_planner_costs(lambda *row: observations.append(row)):
        plan = runner._execution_plan(3)
    assert calls == [runner._handle]
    assert plan.selected.stage_tuple.physical_descriptor == current
    assert [row[-2] for row in observations] == ["domain", "result"]
    assert all(row[0] is runner and row[1:4] == (
        HALO_TEXT_DECODE_COMPONENT, "decode", ("causal_decoder", runner._handle))
        for row in observations)
    assert observations[-1][-1] == (1, 0, plan)
    assert session._planner_native[(HALO_TEXT_DECODE_COMPONENT, "decode", "causal_decoder")] == (
        runner, runner._handle)
    assert runtime.events == [] and not hasattr(runner, "_rpu_last_execution_receipts")


def test_saved_descriptor_is_only_a_mirror_of_the_live_native_plan(runtime):
    runner = runtime.runner
    runner._decode_descriptor = (True,)
    plan = runner._execution_plan(3)
    assert plan.selected.stage_tuple.physical_descriptor == runtime.native.descriptor
    assert runner._decode_descriptor == runtime.native.descriptor


@pytest.mark.parametrize("position", [-1, True, 3.0])
def test_invalid_actual_position_does_not_query_native(runtime, monkeypatch, position):
    monkeypatch.setattr(torch.ops.rpu, "causal_decoder_resolve_decode_stage_descriptor",
                        lambda handle: pytest.fail("invalid position reached native oracle"))
    with pytest.raises(ValueError, match="position"):
        runtime.runner._execution_plan(position)
