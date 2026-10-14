"""Actual RGBD/Former construction and retirement with CPU-only device doubles."""
import gc
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
import weakref

import pytest
import torch

from rpu_backend.adapters.internvla_n1 import rgbd_rpu
from rpu_backend.api import _execution, causal_lm
from rpu_backend.runtime import _native_retirement
from test_graph_cache_freeze import _cache, _sig


def _tensor(*shape):
    return torch.ones(1).expand(shape)


class Cache:
    def __init__(self, **kwargs):
        self.max_seq_len = kwargs["max_seq_len"]
        self.k_caches = []
        self.v_caches = []


def _tower():
    def linear(out, inp):
        return SimpleNamespace(weight=_tensor(out, inp), bias=_tensor(out))
    def norm():
        return SimpleNamespace(weight=_tensor(384), bias=_tensor(384))
    block = SimpleNamespace(
        attn=SimpleNamespace(qkv=linear(1152, 384), proj=linear(384, 384)),
        mlp=SimpleNamespace(fc1=linear(1536, 384), fc2=linear(384, 1536)),
        norm1=norm(), norm2=norm(), ls1=SimpleNamespace(gamma=_tensor(384)),
        ls2=SimpleNamespace(gamma=_tensor(384)))
    return SimpleNamespace(blocks=[block] * 12, norm=norm(), num_register_tokens=1)


def _former_weights():
    weights = {}
    for layer in range(2):
        prefix = f"rgbd_encoder.former_net.layers.{layer}."
        for name in ("self_attn", "multihead_attn"):
            for suffix, shape in (("in_proj_weight", (1152, 384)), ("in_proj_bias", (1152,)),
                                  ("out_proj.weight", (384, 384)), ("out_proj.bias", (384,))):
                weights[prefix + name + "." + suffix] = _tensor(*shape)
        for name, out, inp in (("linear1", 2048, 384), ("linear2", 384, 2048)):
            weights[prefix + name + ".weight"] = _tensor(out, inp)
            weights[prefix + name + ".bias"] = _tensor(out)
        for name in ("norm1", "norm2", "norm3"):
            for slot in ("weight", "bias"):
                weights[prefix + name + "." + slot] = _tensor(384)
    return weights


@pytest.fixture
def native(monkeypatch):
    gc.collect()
    monkeypatch.setattr(_execution, "_UNSAFE_PROCESS_REASON", None)
    monkeypatch.setattr(_native_retirement, "_FAILED_RETIREMENTS", [])
    monkeypatch.setattr(causal_lm, "_LIVE_REF", None)
    monkeypatch.setattr(causal_lm, "_LIVE_TERMINAL_REASON", None)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    state = dict(events=[], resources=[], fault=None, destroy_error=None, next_handle=41, envelopes=[])
    monkeypatch.setattr(rgbd_rpu, "require_controlled_evaluation", lambda _: None)
    monkeypatch.setattr(rgbd_rpu, "verify_asset_manifest", lambda *args: None)
    monkeypatch.setattr(rgbd_rpu, "_hr", lambda value: value)
    monkeypatch.setattr(rgbd_rpu, "tp_col_swizzle_mc_weight", lambda value, *args: value)
    monkeypatch.setattr(rgbd_rpu, "tp_row_swizzle_mc_weight", lambda value, *args: value)
    monkeypatch.setattr(rgbd_rpu, "RPUCache", Cache)
    def graph():
        cache = _cache()
        cache._impl.cache_invariant_ok = lambda: True
        return cache
    monkeypatch.setattr(rgbd_rpu.rpu_backend.graph, "GraphCache", graph)
    original_to = torch.Tensor.to
    def to(value, *args, **kwargs):
        if args and args[0] == "rpu":
            args = ("cpu", *args[1:])
        return original_to(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", to)
    resource_type = _native_retirement._InstalledNativeResource
    def resource(*args, **kwargs):
        value = resource_type(*args, **kwargs)
        state["resources"].append(value)
        return value
    monkeypatch.setattr(rgbd_rpu, "_InstalledNativeResource", resource)
    def create(*args):
        handle = state["next_handle"]
        state["next_handle"] += 1
        return handle
    for op in ("dinov3_vision_create", "navdp_create"):
        monkeypatch.setattr(torch.ops.rpu, op, create, raising=False)
    def destroy(handle):
        state["events"].append(("destroy", handle))
        if state["destroy_error"]:
            raise state["destroy_error"]
    for op in ("dinov3_vision_destroy", "navdp_destroy"):
        monkeypatch.setattr(torch.ops.rpu, op, destroy, raising=False)
    def setter(name):
        def invoke(*args):
            state["events"].append((name, args[0]))
            if state["fault"] == name:
                raise KeyboardInterrupt("original setter failure")
        return invoke
    for op in ("dinov3_vision_set_weights", "dinov3_vision_set_rope",
               "dinov3_vision_set_identity_rope", "dinov3_vision_set_final_norm",
               "dinov3_vision_set_acc32", "dinov3_vision_set_16b_sdpa",
               "dinov3_vision_set_output_mlp", "navdp_set_former_mode", "navdp_set_weights"):
        monkeypatch.setattr(torch.ops.rpu, op, setter(op), raising=False)
    def dino_envelope(handle, capacity, chunk):
        state["envelopes"].append((handle, capacity, chunk))
        if state["fault"] == "dinov3_vision_set_chunk_envelope":
            raise RuntimeError("DINO RGBD envelope failure")
    monkeypatch.setattr(torch.ops.rpu, "dinov3_vision_set_chunk_envelope", dino_envelope, raising=False)
    def former_envelope(handle, capacity, chunk):
        state["envelopes"].append((handle, capacity, chunk))
        if state["fault"] == "navdp_set_chunk_envelope":
            raise RuntimeError("Former envelope failure")
    monkeypatch.setattr(torch.ops.rpu, "navdp_set_chunk_envelope", former_envelope, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "dinov3_vision_position_idx_keepalive", lambda _: torch.zeros(512, 2), raising=False)
    import safetensors.torch
    monkeypatch.setattr(safetensors.torch, "load_file", lambda _: _former_weights())
    yield state
    for resource in state["resources"] + _native_retirement._FAILED_RETIREMENTS:
        resource.finalizer.detach()
    torch.set_num_threads(threads)


def _build(kind):
    if kind == "former":
        return rgbd_rpu.FormerRPU("unused")
    return rgbd_rpu.RGBDTowerRPU(_tower(), asset_paths=("unused",), component_id=kind)


KINDS = (rgbd_rpu.RGB_COMPONENT, rgbd_rpu.DEPTH_COMPONENT, "former")


@pytest.mark.parametrize("failed_retirement", (False, True))
def test_former_graph_inputs_survive_capture_reuse_and_retirement(
        native, monkeypatch, failed_retirement):
    owner = _build("former")
    # Device allocation doubles own independent storage, like CPU -> device upload.
    monkeypatch.setattr(rgbd_rpu, "_hr", lambda value: value.half().clone())
    plan = SimpleNamespace(
        selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=(32,))),
        graph_key_words=lambda: (32,))
    monkeypatch.setattr(rgbd_rpu, "plan_navdp_component", lambda *a, **k: (32, plan))
    monkeypatch.setattr(rgbd_rpu, "publish_execution_receipt", lambda *a, **k: None)
    monkeypatch.setattr(torch.ops.rpu, "navdp_get_resolved_chunk_size", lambda _: 32, raising=False)
    calls = []
    def forward(handle, query, memory, *args):
        calls.append((weakref.ref(query), weakref.ref(memory)))
        return query.clone() + memory.mean(dim=1, keepdim=True)
    monkeypatch.setattr(torch.ops.rpu, "navdp_forward", forward, raising=False)
    @contextmanager
    def capture(signature):
        yield
        gc.collect()
        # Graph.end reads the allocations after the native call returned.
        assert all(ref() is not None for ref in calls[-1])
    monkeypatch.setattr(owner.gc, "capture", capture)
    query = torch.stack([torch.full((32, 384), n) for n in (2., 5.)])
    memory = torch.stack([torch.full((1024, 384), n) for n in (3., 7.)])
    first = owner(query, memory)
    second = owner(query + 10, memory + 20)
    torch.testing.assert_close(first, query + memory.mean(dim=1, keepdim=True))
    torch.testing.assert_close(second, first + 30)
    # Both batches and later calls feed the same captured addresses new data.
    assert len(calls) == 4
    for slot in (0, 1):
        assert all(call[slot]() is calls[0][slot]() for call in calls)
        assert calls[0][slot]() is not None
    if failed_retirement:
        native["destroy_error"] = RuntimeError("Former input retirement failed")
        with pytest.raises(RuntimeError, match="Former input retirement failed"):
            owner.destroy()
    else:
        owner.destroy()
    del owner
    gc.collect()
    assert all((ref() is not None) is failed_retirement for call in calls for ref in call)


def test_former_cold_envelope_uses_own_cache_not_base_navdp_capacity(native):
    owner = _build("former")
    try:
        assert owner.cache.max_seq_len == 1152
        assert native["envelopes"] == [(owner.handle, owner.cache.max_seq_len, 0)]
    finally:
        owner.destroy()


def test_former_cold_envelope_failure_retires_pending_handle(native):
    native["fault"] = "navdp_set_chunk_envelope"
    with pytest.raises(RuntimeError, match="Former envelope failure"):
        _build("former")
    resource, = native["resources"]
    assert resource.handle is None and not resource.keepalive and not resource.finalizer.alive
    assert native["events"].count(("destroy", 41)) == 1


def test_rgbd_dino_cold_envelope_uses_cache_not_2d_rope_rows(native):
    owner = _build(rgbd_rpu.RGB_COMPONENT)
    try:
        assert owner.cache.max_seq_len == 512
        assert native["envelopes"] == [(owner.handle, owner.cache.max_seq_len, 0)]
    finally:
        owner.destroy()


def test_rgbd_dino_envelope_failure_retires_pending_handle(native):
    native["fault"] = "dinov3_vision_set_chunk_envelope"
    with pytest.raises(RuntimeError, match="DINO RGBD envelope failure"):
        _build(rgbd_rpu.RGB_COMPONENT)
    assert native["events"].count(("destroy", 41)) == 1


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("fault", (None, "clear", "invariant", "destroy"))
def test_actual_components_retire_graph_first_with_no_failed_retry(native, kind, fault):
    owner = _build(kind)
    state = owner._native_resource
    graph, = state.graphs
    graph.begin_warmup()
    with graph.capture(_sig(kind)):
        pass
    graph.freeze()
    native["events"].clear()
    clear = graph._impl.clear
    def clear_graph():
        native["events"].append("clear")
        if fault == "clear":
            raise ReferenceError("graph clear failure")
        clear()
    graph._impl.clear = clear_graph
    graph._impl.cache_invariant_ok = lambda: fault != "invariant"
    if fault == "destroy":
        native["destroy_error"] = KeyboardInterrupt("native destroy failure")
    if fault is None:
        owner.destroy()
        owner.destroy()
        assert native["events"] == ["clear", ("destroy", 41)]
        assert state.handle is None and not state.finalizer.alive and not state.keepalive
        with pytest.raises(RuntimeError, match="destroyed"):
            owner.forward(None, None) if kind == "former" else owner.forward(None)
    else:
        with pytest.raises((ReferenceError, RuntimeError, KeyboardInterrupt)):
            owner.destroy()
        assert state.failed and state.keepalive and owner.handle == 41
        before = list(native["events"])
        with pytest.raises(RuntimeError):
            owner.destroy()
        state.finalizer()
        assert native["events"] == before and _execution._UNSAFE_PROCESS_REASON


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("failed", (False, True))
def test_actual_gc_preserves_native_weights_and_cache_until_success(native, kind, failed):
    owner = _build(kind)
    resource = owner._native_resource
    cached = weakref.ref(owner.cache)
    # CPU nn.Module's patched parent may form an actual owner cycle.
    owner.cycle = owner
    owner_ref = weakref.ref(owner)
    if failed:
        native["destroy_error"] = RuntimeError("GC destroy failure")
    native["events"].clear()
    del owner
    gc.collect()
    assert owner_ref() is None
    assert native["events"] == [("destroy", 41)]
    assert bool(resource.keepalive) is failed
    assert (cached() is not None) is failed
    gc.collect()
    assert native["events"] == [("destroy", 41)]


@pytest.mark.parametrize("kind,op", [(rgbd_rpu.RGB_COMPONENT, "dinov3_vision_set_weights"),
                                    (rgbd_rpu.DEPTH_COMPONENT, "dinov3_vision_set_final_norm"),
                                    ("former", "navdp_set_weights")])
@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_actual_constructor_keeps_primary_error_and_pending_weights(native, kind, op, cleanup_fails):
    native["fault"] = op
    if cleanup_fails:
        native["destroy_error"] = ReferenceError("cleanup failure")
    with pytest.raises(KeyboardInterrupt, match="original setter failure") as caught:
        _build(kind)
    resource, = native["resources"]
    assert (resource.handle is not None) is cleanup_fails
    assert bool(resource.keepalive) is cleanup_fails
    assert bool(getattr(caught.value, "__notes__", ())) is cleanup_fails
    assert [event for event in native["events"] if event[0] == "destroy"] == [("destroy", 41)]
    before = list(native["events"])
    resource.finalizer()
    assert native["events"] == before


@pytest.mark.parametrize("kind", KINDS)
def test_actual_component_respects_parent_active_session(native, kind):
    owner = _build(kind)
    parent = SimpleNamespace()
    session = _execution.bind_execution_session(parent, {}, entry_point="RGBD-parent", graph_mode="COMPOSITE_CHILD")
    owner._execution_session = session
    native["events"].clear()
    with session.execute():
        with pytest.raises(RuntimeError, match="during a forward"):
            owner.destroy()
    assert native["events"] == [] and _execution._UNSAFE_PROCESS_REASON is None
    owner.destroy()
    assert session.stats()["state"] == "QUIESCENT"


@pytest.mark.parametrize("failed", (False, True))
def test_actual_output_mlp_pending_binding_retains_all_addresses(native, failed):
    owner = _build(rgbd_rpu.RGB_COMPONENT)
    native["fault"] = "dinov3_vision_set_output_mlp"
    if failed:
        native["destroy_error"] = RuntimeError("cleanup failure")
    with pytest.raises(KeyboardInterrupt, match="original setter failure"):
        owner.set_output_mlp(_tensor(384, 384), _tensor(384), _tensor(384, 384), _tensor(384))
    assert bool(owner._native_resource.keepalive) is failed
    assert (owner.handle is not None) is failed


@pytest.mark.parametrize("fault", (None, "setter", "cleanup"))
def test_actual_rgbd_builder_closures_and_failed_composite_cleanup(native, monkeypatch, tmp_path, fault):
    class Encoder:
        def __init__(self):
            self.rgb_model = _tower()
            self.depth_model = _tower()
        def float(self):
            return self
    cpu_module = ModuleType("rpu_backend.adapters.internvla_n1.rgbd")
    cpu_module.build_rgbd_encoder = lambda *args, **kwargs: Encoder()
    monkeypatch.setitem(sys.modules, cpu_module.__name__, cpu_module)
    monkeypatch.setattr(rgbd_rpu, "_verify_dinov2_source_tree", lambda *args: tmp_path)
    if fault is not None:
        native["fault"] = "navdp_set_weights"
        if fault == "cleanup":
            native["destroy_error"] = RuntimeError("Former native cleanup failure")
        with pytest.raises(KeyboardInterrupt, match="original setter failure") as caught:
            rgbd_rpu.build_rgbd_encoder_rpu("unused", tmp_path)
        destroyed = [event for event in native["events"] if event[0] == "destroy"]
        assert destroyed == ([("destroy", 43)] if fault == "cleanup" else
                             [("destroy", 43), ("destroy", 42), ("destroy", 41)])
        assert bool(getattr(caught.value, "__notes__", ())) is (fault == "cleanup")
        before = list(native["events"])
        for resource in native["resources"]:
            resource.finalizer()
        assert native["events"] == before
        if fault == "cleanup":
            assert all(resource.keepalive for resource in native["resources"])
    else:
        encoder = rgbd_rpu.build_rgbd_encoder_rpu("unused", tmp_path)
        owners = [weakref.ref(child) for child in (*encoder._rpu_towers, encoder._rpu_former)]
        native["events"].clear()
        del encoder
        gc.collect()
        assert all(owner() is None for owner in owners)
        assert sorted(native["events"]) == [("destroy", 41), ("destroy", 42), ("destroy", 43)]
        assert all(not resource.keepalive for resource in native["resources"])
