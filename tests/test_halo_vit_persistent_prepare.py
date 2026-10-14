"""Cold HALO builder ordering; native allocation correctness has its own gate."""
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.halo import runtime, vit_cache
from rpu_backend.api.cache import RPUCache


@pytest.fixture
def cold_builder(monkeypatch):
    from rpu_backend.api import _execution, causal_lm

    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setattr(runtime, "_FAILED_RETIREMENTS", [])
    controller = runtime.HaloExecutionController()
    state = SimpleNamespace(events=[], caches=[], graphs=[], installed=set(),
                            prepare_failure=None, destroy_failure=None,
                            envelope_failure=None, controller=controller)
    cfg = vit_cache.HaloConfig(num_layers=1, hidden_size=16, num_q_heads=1,
                              num_kv_heads=1, head_dim=16, intermediate_size=128,
                              attn_tp=1)
    vcfg = vit_cache.VitCacheConfig(vit_hidden=16, vit_layers=1, vit_heads=1,
                                  vit_orig_head_dim=16, vit_intermediate=128)
    tensor = torch.zeros(1, dtype=torch.float16)
    text = SimpleNamespace(**{name: [tensor] for name in (
        "q_w", "k_w", "v_w", "o_w", "q_norm", "k_norm", "input_norm",
        "post_norm", "gate_w", "up_w", "down_w", "q_bias", "k_bias", "v_bias"
    )}, final_norm_w=tensor)
    monkeypatch.setattr(vit_cache, "_HaloTensorStore", lambda path: object())
    monkeypatch.setattr(vit_cache, "load_halo_branches", lambda *args: (text, None, None))
    monkeypatch.setattr(vit_cache, "_load_vit_host_weights", lambda *args: None)
    monkeypatch.setattr(vit_cache, "_build_siglip_skeleton", lambda *args: object())
    monkeypatch.setattr(vit_cache, "_convert_siglip_weights_for_rpu", lambda obj: None)
    monkeypatch.setattr(vit_cache, "_convert_siglip_projector_for_rpu", lambda obj: None)
    monkeypatch.setattr(vit_cache, "_gather_siglip_weights", lambda *args: (tensor,))
    monkeypatch.setattr(vit_cache, "build_frame_rope_tables", lambda *args: (tensor, tensor))

    def create(handle):
        state.events.append(("create", handle))
        return handle

    def install(handle, *args):
        state.events.append(("weights", handle))
        state.installed.add(handle)

    def allocate(**kwargs):
        cache = RPUCache(device="cpu", **kwargs)
        state.caches.append(cache)
        state.events.append(("cache", len(state.caches)))
        return cache

    def envelope(handle, capacity, chunk):
        assert state.events[-1] == ("weights", handle)
        state.events.append(("envelope", handle, capacity, chunk))
        if state.envelope_failure == handle:
            raise RuntimeError(f"envelope {handle} failed")

    def forbidden(*args, **kwargs):
        pytest.fail("cold persistent preparation must not query, forward, or BUILD")

    def prepare(handle, *args):
        assert state.installed == {41, 42} and len(state.caches) == 2
        assert not state.graphs and not controller._owners
        assert all(cache.position == 0 for cache in state.caches)
        assert not controller._execution_session._started
        controller._execution_session.require_cold()
        state.events.append(("prepare", handle, *args))
        if state.prepare_failure == handle:
            raise RuntimeError(f"prepare {handle} failed")

    graph_type = vit_cache.rpu_backend.graph.GraphCache

    def graph():
        cache = graph_type()
        index = len(state.graphs) + 1
        state.graphs.append(cache)
        state.events.append(("graph", index))
        clear = cache.clear
        monkeypatch.setattr(cache, "capture", forbidden)

        def clear_graph():
            state.events.append(("clear", index))
            clear()

        monkeypatch.setattr(cache, "clear", clear_graph)
        return cache

    def destroy(handle):
        state.events.append(("destroy", handle))
        if state.destroy_failure == handle:
            raise RuntimeError(f"destroy {handle} failed")

    for family, handle in (("siglip", 41), ("causal_decoder", 42)):
        monkeypatch.setattr(torch.ops.rpu, family + "_create",
                            lambda h=handle: create(h), raising=False)
        monkeypatch.setattr(torch.ops.rpu, family + "_set_weights", install, raising=False)
        monkeypatch.setattr(torch.ops.rpu, family + "_set_chunk_envelope", envelope, raising=False)
        monkeypatch.setattr(torch.ops.rpu, family + "_prepare_persistent_spm", prepare, raising=False)
        monkeypatch.setattr(torch.ops.rpu, family + "_destroy", destroy, raising=False)
        query = "siglip_resolve_stage_domain" if family == "siglip" else "causal_decoder_resolve_prefill_stage_domain"
        monkeypatch.setattr(torch.ops.rpu, query, forbidden, raising=False)
        monkeypatch.setattr(torch.ops.rpu, family + "_forward", forbidden, raising=False)
    monkeypatch.setattr(vit_cache, "RPUCache", allocate)
    monkeypatch.setattr(vit_cache.rpu_backend.graph, "GraphCache", graph)
    state.build = lambda **kwargs: vit_cache.build_halo_vit_cache(
        "unused-checkpoint", cfg=cfg, vcfg=vcfg, execution_controller=controller, **kwargs)
    return state


def test_both_persistent_owners_prepare_before_graphs_and_keep_session_cold(cold_builder):
    f = cold_builder
    runner = f.build()
    try:
        assert f.events == [
            ("create", 41), ("weights", 41), ("envelope", 41, 1792, 0), ("cache", 1),
            ("create", 42), ("weights", 42), ("envelope", 42, 4736, 0), ("cache", 2),
            ("prepare", 41, 1564, 1), ("prepare", 42, 1566, 0, False, 0),
            ("graph", 1), ("graph", 2),
        ]
        assert set(f.controller._owners) == {
            runtime.HALO_VISION_COMPONENT, runtime.HALO_VIT_LM_COMPONENT}
        assert runner._execution_session is f.controller._execution_session
        assert not runner._execution_session._started
        runner._execution_session.require_cold()
        assert all(cache.position == 0 for cache in f.caches)
        assert all(graph.size() == 0 and graph.cache_invariant_ok() for graph in f.graphs)
    finally:
        runner.close()
    assert f.events[-4:] == [("clear", 2), ("clear", 1), ("destroy", 42), ("destroy", 41)]
    assert runner._halo_resources.closed and not f.controller._owners


def test_vision_envelope_tracks_nondefault_cache_capacity(cold_builder):
    f = cold_builder
    runner = f.build(vit_max_seq=2048)
    try:
        assert ("envelope", 41, 2048, 0) in f.events
        assert runner.vit_cache.max_seq_len == runner.vit_max_seq == 2048
        f.controller._execution_session.require_cold()
    finally:
        runner.close()


def test_vision_envelope_failure_retires_before_second_owner(cold_builder):
    f = cold_builder
    f.envelope_failure = 41
    with pytest.raises(RuntimeError, match="envelope 41 failed"):
        f.build()
    assert f.events == [
        ("create", 41), ("weights", 41), ("envelope", 41, 1792, 0), ("destroy", 41)]
    assert not f.caches and not f.graphs and not f.controller._owners
    f.controller._execution_session.require_cold()


@pytest.mark.parametrize("prepare_handle", [41, 42])
@pytest.mark.parametrize("destroy_handle", [None, 41, 42])
def test_prepare_failure_uses_strict_builder_retirement(cold_builder, prepare_handle, destroy_handle):
    f = cold_builder
    f.prepare_failure, f.destroy_failure = prepare_handle, destroy_handle
    with pytest.raises(RuntimeError, match=f"^prepare {prepare_handle} failed") as error:
        f.build()
    assert str(error.value) == f"prepare {prepare_handle} failed"
    assert [event for event in f.events if event[0] == "prepare"] == (
        [("prepare", 41, 1564, 1)] +
        ([("prepare", 42, 1566, 0, False, 0)] if prepare_handle == 42 else []))
    assert not f.graphs and not f.controller._owners
    assert f.controller._execution_session._started is (destroy_handle is not None)
    assert all(cache.position == 0 for cache in f.caches)
    assert [event for event in f.events if event[0] == "destroy"] == (
        [("destroy", 42)] if destroy_handle == 42 else [("destroy", 42), ("destroy", 41)])
    if destroy_handle is None:
        assert not getattr(error.value, "__notes__", ())
        assert not runtime._FAILED_RETIREMENTS
        f.controller._execution_session.require_cold()
    else:
        assert error.value.__notes__ == [
            f"HALO builder cleanup also failed: RuntimeError('destroy {destroy_handle} failed')"]
        resources, = runtime._FAILED_RETIREMENTS
        assert not resources.closed and resources.failed
        assert set(resources.handles) == (
            {"_lm_handle", "_vit_handle"} if destroy_handle == 42 else {"_vit_handle"})
        assert f.controller._execution_session._poisoned
        with pytest.raises(RuntimeError, match="already failed"):
            resources.retire()


@pytest.mark.parametrize("capacity, component", [("vit_max_seq", "vision"), ("lm_max_seq", "language")])
def test_oversized_prepare_rows_fail_before_asset_loading(cold_builder, monkeypatch, capacity, component):
    def load_forbidden(*args):
        pytest.fail("invalid prepare rows must fail before opening checkpoint assets")

    monkeypatch.setattr(vit_cache, "_HaloTensorStore", load_forbidden)
    with pytest.raises(ValueError, match=f"HALO {component} rows .* exceed cache capacity 16"):
        cold_builder.build(**{capacity: 16})
    assert not cold_builder.events
    cold_builder.controller._execution_session.require_cold()
