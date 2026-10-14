from __future__ import annotations
from contextlib import contextmanager
from types import SimpleNamespace
import pytest
import torch
from rpu_backend.adapters.rhinovla.runtime import RHINOVLA_ACTION_COMPONENT, RHINOVLA_TEXT_COMPONENT, RHINOVLA_VISION_COMPONENT, bind_rhinovla_execution_runtime
from rpu_backend.api import RhinoVLAPolicy

class _Cache:

    def __init__(self):
        self.clears = 0

    def begin_warmup(self):
        pass

    def clear(self):
        self.clears += 1

    def cache_invariant_ok(self):
        return True

class _KvCache:

    def __init__(self):
        self.position = 17
        self.resets = []

    def reset_to_position(self, position):
        self.position = int(position)
        self.resets.append(self.position)

class _Child:
    pass

class _Runtime:

    def __init__(self):
        from rpu_backend.runtime._native_retirement import _InstalledNativeResource
        self.text = _Child()
        self.text._rpu_decoder_handle = 11
        self.text._rpu_text_graph_cache = _Cache()
        self.text._rpu_decoder_graph_cache = _Cache()
        self.vision = _Child()
        self.vision._rpu_vision_handle = 22
        self.vision._rpu_vision_graph_cache = _Cache()
        self.action = _Child()
        self.action._rpu_handle = 33
        self.action._rpu_graph_cache = _Cache()
        self.action._rpu_kv_cache = _KvCache()
        self.action._rpu_prefix_source_snapshot = object()
        for (child, attribute, handle_name, graphs) in ((self.text, '_rpu_decoder_retirement_state', '_rpu_decoder_handle', (self.text._rpu_text_graph_cache, self.text._rpu_decoder_graph_cache)), (self.vision, '_rpu_vision_retirement_state', '_rpu_vision_handle', (self.vision._rpu_vision_graph_cache,)), (self.action, '_rpu_retirement_state', '_rpu_handle', (self.action._rpu_graph_cache,))):
            setattr(child, attribute, _InstalledNativeResource(child, getattr(child, handle_name), lambda _handle: None, graphs=graphs, keepalive=(), label='RhinoVLA fixture', handle_name=handle_name))
            setattr(child, handle_name + '_finalizer', getattr(child, attribute).finalizer)

    def _publish_receipts(self):
        for (child, component, stage) in ((self.text, RHINOVLA_TEXT_COMPONENT, 'prefill'), (self.vision, RHINOVLA_VISION_COMPONENT, 'vision'), (self.action, RHINOVLA_ACTION_COMPONENT, 'action')):
            child._rpu_last_execution_plan = {'component': component, 'stage': stage, 'generation': child._fmb_execution_generation, 'authority': 'NATIVE_A6_STAGE_DESCRIPTOR', 'dry_forward_agreement': True}

    def predict(self, **request):
        self._publish_receipts()
        return request

    def prepare_graphs(self, **request):
        self._publish_receipts()
        return request

@pytest.fixture
def native_ops(monkeypatch):
    import gc
    gc.collect()
    from rpu_backend.api import _execution, causal_lm
    from rpu_backend.runtime import _native_retirement
    import rpu_backend.adapters.rhinovla.runtime as runtime_module
    monkeypatch.setattr(_execution, '_UNSAFE_PROCESS_REASON', None)
    monkeypatch.setattr(causal_lm, '_LIVE_REF', None)
    monkeypatch.setattr(_native_retirement, '_FAILED_RETIREMENTS', [])
    monkeypatch.setattr(runtime_module, '_FAILED_RETIREMENTS', [])
    events = []
    names = ('causal_decoder_set_chunk_size_override', 'causal_decoder_enable_execution_reconfigure', 'qwen3vl_vision_set_chunk_size', 'qwen3vl_vision_enable_execution_reconfigure', 'rhino_vla_set_chunk_size', 'rhino_vla_enable_execution_reconfigure', 'execution_reconfigure_begin', 'execution_reconfigure_commit', 'execution_reconfigure_abort', 'execution_reconfigure_abort_attempt', 'causal_decoder_stage_chunk_size_override', 'qwen3vl_vision_stage_chunk_size', 'rhino_vla_stage_chunk_size')
    for name in names:
        monkeypatch.setattr(torch.ops.rpu, name, lambda *args, _name=name: events.append((_name, args)), raising=False)
    for name in ('causal_decoder_get_resolved_chunk_size', 'qwen3vl_vision_get_resolved_chunk_size', 'rhino_vla_get_resolved_chunk_size'):
        monkeypatch.setattr(torch.ops.rpu, name, lambda _handle: 0, raising=False)
    import rpu_backend.adapters.rhinovla.runtime as runtime_module

    @contextmanager
    def transaction(_ops, *, journal=None):
        events.append(('begin', ()))
        yield 71
        if journal is not None:
            journal['mutation_started'] = True
        events.append(('commit', ()))
    monkeypatch.setattr(runtime_module, 'native_execution_reconfigure', transaction)
    return events

def _bind(runtime, value=None):
    return bind_rhinovla_execution_runtime(runtime, text_model=runtime.text, vision_model=runtime.vision, action_expert=runtime.action, rpu_execution=value)

def test_private_same_config_rebuild_retires_all_actual_children(native_ops):
    runtime = _Runtime()
    controller = _bind(runtime)
    runtime._publish_receipts()
    caches = (runtime.text._rpu_text_graph_cache, runtime.vision._rpu_vision_graph_cache, runtime.action._rpu_graph_cache)
    controller.session.reconfigure({})
    assert [cache.clears for cache in caches] == [0, 0, 0]
    controller.session._reconfigure({}, force_rebuild=True)
    assert [cache.clears for cache in caches] == [1, 1, 1]
    assert set(controller._component_generations.values()) == {1}
    assert runtime.action._rpu_kv_cache.position == 0
    assert runtime.action._rpu_prefix_source_snapshot is None
    for child in (runtime.text, runtime.vision, runtime.action):
        assert not hasattr(child, '_rpu_last_execution_plan')
        assert child._fmb_execution_generation == 1
    assert {name for (name, _) in native_ops if '_stage_chunk' in name} == {'causal_decoder_stage_chunk_size_override', 'qwen3vl_vision_stage_chunk_size', 'rhino_vla_stage_chunk_size'}
    runtime._publish_receipts()
    assert all((row['generation'] == 1 for row in controller.collect_receipts().values()))

def test_factory_rejects_unknown_component_before_external_load():
    called = False

    def factory(_config, *, rpu_execution):
        nonlocal called
        called = True
        return SimpleNamespace(rpu_execution=rpu_execution)
    with pytest.raises(ValueError, match='unknown component'):
        RhinoVLAPolicy.from_factory(factory, {}, rpu_execution={'components': {'typo': {}}})
    assert called is False

def test_configured_external_runtime_requires_explicit_child_binding():
    runtime = SimpleNamespace(predict=lambda **request: request, _rpu_execution={'action': {'chunk_size': 48}})
    with pytest.raises(ValueError, match='bind_rhinovla_execution_runtime'):
        RhinoVLAPolicy.from_runtime(runtime)

def test_bind_resolves_components_and_predict_collects_standard_receipts(native_ops):
    runtime = _Runtime()
    controller = _bind(runtime, {'prefill': {'chunk_size': 64}, 'components': {RHINOVLA_VISION_COMPONENT: {'vision': {'chunk_size': 144}}, RHINOVLA_ACTION_COMPONENT: {'action': {'chunk_size': 48}}}})
    policy = RhinoVLAPolicy.from_runtime(runtime)
    assert controller.session is runtime._execution_session
    assert runtime.text._rpu_execution['prefill']['chunk_size'] == 64
    assert runtime.vision._rpu_execution['vision']['chunk_size'] == 144
    assert runtime.action._rpu_execution['action']['chunk_size'] == 48
    assert policy.predict(state=1) == {'state': 1}
    assert set(policy.last_rpu_execution_plan) == {'prefill', 'vision', 'action'}
    assert all((receipt['authority'] == 'NATIVE_A6_STAGE_DESCRIPTOR' for receipt in policy.last_rpu_execution_plan.values()))
    assert {name for (name, _args) in native_ops if name.endswith('enable_execution_reconfigure')} == {'causal_decoder_enable_execution_reconfigure', 'qwen3vl_vision_enable_execution_reconfigure', 'rhino_vla_enable_execution_reconfigure'}

def test_bind_rejects_any_child_that_already_forwarded_before_mutation(native_ops, monkeypatch):
    runtime = _Runtime()
    monkeypatch.setattr(torch.ops.rpu, 'rhino_vla_get_resolved_chunk_size', lambda _handle: 48)
    with pytest.raises(RuntimeError, match='bound before their first forward'):
        _bind(runtime)
    assert not any(('set_chunk_size' in name for (name, _args) in native_ops))

def test_hot_reconfigure_invalidates_only_changed_child_and_changes_generation(native_ops):
    runtime = _Runtime()
    controller = _bind(runtime)
    policy = RhinoVLAPolicy.from_runtime(runtime)
    policy.reconfigure({'components': {RHINOVLA_ACTION_COMPONENT: {'action': {'chunk_size': 48}}}})
    assert runtime.text._rpu_text_graph_cache.clears == 0
    assert runtime.vision._rpu_vision_graph_cache.clears == 0
    assert runtime.action._rpu_graph_cache.clears == 1
    assert runtime.action._rpu_kv_cache.position == 0
    assert runtime.action._rpu_kv_cache.resets == [0]
    assert runtime.action._rpu_prefix_source_snapshot is None
    assert controller._component_generations == {RHINOVLA_TEXT_COMPONENT: 0, RHINOVLA_VISION_COMPONENT: 0, RHINOVLA_ACTION_COMPONENT: 1}
    assert runtime.action._fmb_execution_generation == 1
    assert ('rhino_vla_stage_chunk_size', (33, 71, 48)) in native_ops

def test_native_begin_failure_keeps_config_generations_and_graphs(native_ops, monkeypatch):
    del native_ops
    runtime = _Runtime()
    controller = _bind(runtime)
    policy = RhinoVLAPolicy.from_runtime(runtime)
    old_config = controller.session.config
    old_generations = dict(controller._component_generations)
    import rpu_backend.adapters.rhinovla.runtime as runtime_module

    @contextmanager
    def fail_before_yield(_ops, *, journal=None):
        raise RuntimeError('begin failed')
        yield
    monkeypatch.setattr(runtime_module, 'native_execution_reconfigure', fail_before_yield)
    with pytest.raises(RuntimeError, match='begin failed'):
        policy.reconfigure({'vision': {'chunk_size': 144}})
    assert controller.session.config == old_config
    assert controller._component_generations == old_generations
    assert runtime.text._rpu_text_graph_cache.clears == 0
    assert runtime.vision._rpu_vision_graph_cache.clears == 0
    assert runtime.action._rpu_graph_cache.clears == 0
    assert controller.session._poisoned is False

def test_stale_child_receipt_is_rejected(native_ops):
    runtime = _Runtime()
    _bind(runtime)
    runtime.predict = lambda **request: request
    runtime.text._rpu_last_execution_plan = {'component': RHINOVLA_TEXT_COMPONENT, 'stage': 'prefill', 'generation': 0, 'authority': 'NATIVE_A6_STAGE_DESCRIPTOR', 'dry_forward_agreement': True}
    policy = RhinoVLAPolicy.from_runtime(runtime)
    with pytest.raises(RuntimeError, match='did not publish'):
        policy.predict(state=1)

def test_unconfigured_runtime_cannot_bypass_component_planner():
    runtime = SimpleNamespace(predict=lambda **request: request)
    with pytest.raises(ValueError, match='bind_rhinovla_execution_runtime'):
        RhinoVLAPolicy.from_runtime(runtime)
