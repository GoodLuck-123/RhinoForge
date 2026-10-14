from __future__ import annotations
from types import SimpleNamespace
from dataclasses import replace
import gc
import weakref
import pytest
import torch
import rpu_backend
from rpu_backend.adapters import halo
from rpu_backend.adapters.halo import image_flow, runtime, text_cache, text_decode
from rpu_backend.adapters.halo import vit_cache
from rpu_backend.adapters.halo.weights import HaloConfig

class _FakeTensor:
    pass

class _FakeCache:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

class _GraphCache:
    _next_id = 0

    def __init__(self, **kwargs):
        type(self)._next_id += 1
        self.name = f'graph-{type(self)._next_id}'
        self.kwargs = kwargs

    def clear(self):
        _STATE['events'].append(('clear', self.name))
        if _STATE.get('graph_error') == self.name:
            raise RuntimeError('synthetic graph retirement failure')

    def cache_invariant_ok(self):
        return _STATE.get('bad_invariant') != self.name
_STATE = None

def _branch():
    tensor = _FakeTensor()
    return SimpleNamespace(q_w=tensor, k_w=tensor, v_w=tensor, o_w=tensor, q_norm=tensor, k_norm=tensor, input_norm=tensor, post_norm=tensor, gate_w=tensor, up_w=tensor, down_w=tensor, q_bias=tensor, k_bias=tensor, v_bias=tensor, final_norm_w=tensor)

@pytest.fixture
def native_runtime(monkeypatch):
    from rpu_backend.api import causal_lm
    from rpu_backend.api import _execution
    monkeypatch.setattr(causal_lm, '_LIVE_REF', None)
    monkeypatch.setattr(causal_lm, '_LIVE_TERMINAL_REASON', None)
    monkeypatch.setattr(_execution, '_UNSAFE_PROCESS_REASON', None)
    monkeypatch.setattr(runtime, '_FAILED_RETIREMENTS', [])
    global _STATE
    state = {'next_handle': 1, 'alive': {}, 'events': [], 'error_op': None, 'error': None}
    _STATE = state
    _GraphCache._next_id = 0

    def create(component):

        def inner():
            handle = state['next_handle']
            state['next_handle'] += 1
            state['alive'][handle] = component
            state['events'].append(('create', component, handle))
            return handle
        return inner

    def destroy(component):

        def inner(handle):
            state['events'].append(('destroy', component, handle))
            if state.get('destroy_error') == component:
                raise RuntimeError('synthetic native retirement failure')
            assert state['alive'].pop(handle) == component
        return inner

    def setter(name):

        def inner(handle, *args, **kwargs):
            state['events'].append(('set', name, handle))
            if state['error_op'] == name:
                raise state['error']
        return inner
    ops = {'halo_action_expert_create': create('action'), 'halo_action_expert_destroy': destroy('action'), 'halo_action_expert_set_weights': setter('action_weights'), 'halo_action_expert_set_moe_weights': setter('action_moe'), 'halo_action_expert_set_chunk_envelope': setter('action_envelope'), 'causal_decoder_create': create('decoder'), 'causal_decoder_destroy': destroy('decoder'), 'causal_decoder_set_weights': setter('decoder_weights'), 'causal_decoder_set_chunk_envelope': setter('decoder_envelope'), 'causal_decoder_prepare_persistent_spm': setter('decoder_prepare'), 'causal_decoder_resolve_decode_stage_descriptor': lambda handle: (3, handle, 1), 'halo_image_flow_create': create('image'), 'halo_image_flow_destroy': destroy('image'), 'halo_image_flow_set_weights': setter('image_weights'), 'halo_image_flow_set_gen_weights': setter('image_gen'), 'halo_image_flow_set_chunk_envelope': setter('image_envelope'), 'siglip_create': create('vit'), 'siglip_destroy': destroy('vit'), 'siglip_set_weights': setter('vit_weights'), 'siglip_set_chunk_envelope': setter('vit_envelope'), 'siglip_prepare_persistent_spm': setter('vit_prepare')}
    for (name, function) in ops.items():
        monkeypatch.setattr(torch.ops.rpu, name, function, raising=False)
    for module in (runtime, text_cache, text_decode, image_flow, vit_cache):
        monkeypatch.setattr(module, 'RPUCache', _FakeCache)
    monkeypatch.setattr(rpu_backend.graph, 'GraphCache', _GraphCache)
    branch = _branch()
    host = SimpleNamespace(lm_head_w=_FakeTensor())
    monkeypatch.setattr(runtime, 'load_halo_branches', lambda *args, **kwargs: (branch, branch, host))
    monkeypatch.setattr(runtime, 'build_rope_tables', lambda *args, **kwargs: (_FakeTensor(), _FakeTensor()))
    monkeypatch.setattr(runtime, 'build_text_row_mask', lambda cfg: _FakeTensor())
    monkeypatch.setattr(runtime.HaloStepRunner, '__post_init__', lambda self: setattr(self, '_x_emb', _FakeTensor()))
    for module in (text_cache, text_decode):
        monkeypatch.setattr(module, 'load_halo_branches', lambda *args, **kwargs: (branch, branch, host))
        monkeypatch.setattr(module, 'build_seq_rope_tables', lambda *args, **kwargs: (_FakeTensor(), _FakeTensor()))
    monkeypatch.setattr(image_flow, 'load_halo_gen_branch', lambda *args, **kwargs: (branch, branch, host))
    monkeypatch.setattr(image_flow, 'build_rope_tables', lambda *args, **kwargs: (_FakeTensor(), _FakeTensor()))
    monkeypatch.setattr(image_flow, 'build_image_text_row_mask', lambda cfg: _FakeTensor())
    monkeypatch.setattr(vit_cache, '_HaloTensorStore', lambda path: object())
    monkeypatch.setattr(vit_cache, 'load_halo_branches', lambda *args, **kwargs: (branch, branch, host))
    monkeypatch.setattr(vit_cache, '_load_vit_host_weights', lambda *args, **kwargs: object())
    monkeypatch.setattr(vit_cache, '_build_siglip_skeleton', lambda *args, **kwargs: object())
    monkeypatch.setattr(vit_cache, '_convert_siglip_weights_for_rpu', lambda *args, **kwargs: None)
    monkeypatch.setattr(vit_cache, '_convert_siglip_projector_for_rpu', lambda *args, **kwargs: None)
    monkeypatch.setattr(vit_cache, '_gather_siglip_weights', lambda *args, **kwargs: ())
    monkeypatch.setattr(vit_cache, 'build_frame_rope_tables', lambda *args, **kwargs: (_FakeTensor(), _FakeTensor()))
    return state

def _single_builders(cfg=None):
    cfg = HaloConfig() if cfg is None else cfg
    return (('action', 'action_weights', '_handle_finalizer', lambda : runtime.build_halo_action_expert('unused', rope_position=11, cfg=cfg)), ('text_cache', 'decoder_weights', '_handle_finalizer', lambda : text_cache.build_halo_text_cache('unused', cfg=cfg)), ('text_decode', 'decoder_weights', '_handle_finalizer', lambda : text_decode.build_halo_text_decode('unused', cfg=cfg)), ('image_flow', 'image_weights', '_handle_finalizer', lambda : image_flow.build_halo_image_flow('unused', cfg=cfg)))

@pytest.mark.parametrize('name,error_op,finalizer_attr,builder', _single_builders(), ids=lambda value: value if isinstance(value, str) else None)
def test_single_handle_builder_cleans_base_exception(native_runtime, name, error_op, finalizer_attr, builder):

    class BuildInterrupted(BaseException):
        pass
    native_runtime['error_op'] = error_op
    native_runtime['error'] = BuildInterrupted(f'synthetic {name} interruption')
    with pytest.raises(BuildInterrupted, match=name):
        builder()
    assert native_runtime['alive'] == {}
    assert native_runtime['events'][-1][0] == 'destroy'

@pytest.mark.parametrize('name,error_op,finalizer_attr,builder', _single_builders(), ids=lambda value: value if isinstance(value, str) else None)
def test_single_handle_runner_tracks_finalizer_and_closes_graph_first(native_runtime, name, error_op, finalizer_attr, builder):
    runner = builder()
    finalizer = getattr(runner, finalizer_attr)
    assert finalizer.alive
    event_count = len(native_runtime['events'])
    runner.close()
    close_events = native_runtime['events'][event_count:]
    assert close_events[0][0] == 'clear'
    assert close_events[-1][0] == 'destroy'
    assert not finalizer.alive
    assert native_runtime['alive'] == {}
    runner.close()
    assert native_runtime['events'][event_count:] == close_events

def test_action_builder_mask_is_versioned_under_inference_mode(native_runtime):
    with torch.inference_mode():
        runner = runtime.build_halo_action_expert('unused', rope_position=11)
        try:
            mask = runner._attn_mask
            assert mask.device.type == 'cpu' and (not mask.is_inference())
            assert tuple(mask.shape) == (1, 1, runner.cfg.chunk_size, runner.cfg.total_len)
            assert torch.count_nonzero(mask) == 0
            version = mask._version
            mask[0, 0, 0, 0] = -1
            assert mask._version > version
        finally:
            runner.close()

def test_single_handle_finalizer_registration_failure_cleans_raw_owner(native_runtime, monkeypatch):

    class RegistrationInterrupted(BaseException):
        pass

    def fail_registration(*args, **kwargs):
        raise RegistrationInterrupted('synthetic finalizer interruption')
    monkeypatch.setattr(runtime.weakref, 'finalize', fail_registration)
    with pytest.raises(RegistrationInterrupted, match='finalizer interruption'):
        runtime.build_halo_action_expert('unused', rope_position=11)
    assert native_runtime['alive'] == {}
    assert [event[0] for event in native_runtime['events'][-2:]] == ['clear', 'destroy']

def _vit_builder(cfg=None):
    cfg = HaloConfig() if cfg is None else cfg
    vcfg = vit_cache.VitCacheConfig(vit_hidden=8, vit_layers=1, vit_heads=8, vit_orig_head_dim=16, vit_intermediate=16, patch_in_dim=4, connector_out=cfg.hidden_size)
    return vit_cache.build_halo_vit_cache('unused', cfg=cfg, vcfg=vcfg, vit_max_seq=1792, lm_max_seq=1792)

def test_vit_builder_cleans_both_raw_handles_on_base_exception(native_runtime):

    class BuildInterrupted(BaseException):
        pass
    native_runtime['error_op'] = 'decoder_weights'
    native_runtime['error'] = BuildInterrupted('synthetic vit interruption')
    with pytest.raises(BuildInterrupted, match='vit interruption'):
        _vit_builder()
    assert native_runtime['alive'] == {}
    assert native_runtime['events'][-2:] == [('destroy', 'decoder', 2), ('destroy', 'vit', 1)]

def test_vit_runner_tracks_both_finalizers_and_closes_graphs_first(native_runtime):
    runner = _vit_builder()
    assert runner._vit_handle_finalizer.alive
    assert runner._lm_handle_finalizer.alive
    event_count = len(native_runtime['events'])
    runner.close()
    close_events = native_runtime['events'][event_count:]
    assert [event[0] for event in close_events] == ['clear', 'clear', 'destroy', 'destroy']
    assert [event[1] for event in close_events[-2:]] == ['decoder', 'vit']
    assert native_runtime['alive'] == {}
    runner.close()
    assert native_runtime['events'][event_count:] == close_events

def test_vit_second_finalizer_failure_cleans_both_handles(native_runtime, monkeypatch):

    class RegistrationInterrupted(BaseException):
        pass
    calls = 0

    class Finalizer:

        def __init__(self, callback, args):
            self._callback = callback
            self._args = args
            self.alive = True

        def __call__(self):
            if self.alive:
                self.alive = False
                self._callback(*self._args)

        def detach(self):
            self.alive = False

    def finalize(owner, callback, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RegistrationInterrupted('synthetic second finalizer failure')
        return Finalizer(callback, args)
    monkeypatch.setattr(runtime.weakref, 'finalize', finalize)
    with pytest.raises(RegistrationInterrupted, match='second finalizer failure'):
        _vit_builder()
    assert native_runtime['alive'] == {}
    assert [event[0] for event in native_runtime['events'][-4:]] == ['clear', 'clear', 'destroy', 'destroy']
    assert [event[1] for event in native_runtime['events'][-2:]] == ['decoder', 'vit']

def test_internal_native_prefix_runner_is_closed_deterministically(monkeypatch):
    state = {'closed': 0}

    class Runner:

        def prefill(self, *args, **kwargs):
            return {0: (_FakeTensor(), _FakeTensor())}

        def close(self):
            state['closed'] += 1
    monkeypatch.setattr(halo, 'build_halo_text_cache', lambda *args, **kwargs: Runner())
    result = halo.build_native_prefix_banks('unused', img_uncond_text_pack={'packed_text_ids': _FakeTensor(), 'packed_text_position_ids': _FakeTensor()})
    assert 0 in result['cfg_img']
    assert state['closed'] == 1

def test_image_flow_default_binds_real_production_envelope(native_runtime, monkeypatch):
    calls = []
    monkeypatch.setattr(torch.ops.rpu, 'halo_image_flow_set_chunk_envelope', lambda handle, max_seq_len, chunk: calls.append((handle, max_seq_len, chunk)), raising=False)
    runner = image_flow.build_halo_image_flow('unused')
    try:
        assert calls == [(runner._handle, 10976, 0)]
        assert runner.max_seq_len == 10976
        assert runner._graph_cache.kwargs == {'max_entries': 3}
    finally:
        runner.close()

@pytest.mark.parametrize('builder,config,match', ((lambda : runtime.build_halo_action_expert('unused', rope_position=11, rpu_execution={'components': {'action_expert': {'action': {'chunk_size': 48}}}}), 'action_expert', 'fixed 18-row'), (lambda : image_flow.build_halo_image_flow('unused', rpu_execution={'components': {'image_flow': {'action': {'chunk_size': 17}}}}), 'image_flow', 'positive multiple of 16')))
def test_exact_rejection_precedes_weight_loading(native_runtime, monkeypatch, builder, config, match):
    loaded = []
    monkeypatch.setattr(runtime, 'load_halo_branches', lambda *args, **kwargs: loaded.append(True))
    monkeypatch.setattr(image_flow, 'load_halo_gen_branch', lambda *args, **kwargs: loaded.append(True))
    with pytest.raises(ValueError, match=match):
        builder()
    assert loaded == []
    assert native_runtime['alive'] == {}

def test_shared_controller_invalidates_only_changed_child_graph():
    events = []

    class Graph:

        def __init__(self, name):
            self.name = name

        def begin_warmup(self):
            events.append(('warmup', self.name))

        def clear(self):
            events.append(('clear', self.name))

        def cache_invariant_ok(self):
            return True

    class Owner:
        _closed = False
    controller = runtime.HaloExecutionController()
    (action_owner, image_owner) = (Owner(), Owner())
    controller.attach(action_owner, {runtime.HALO_ACTION_COMPONENT: (Graph('action'),)})
    controller.attach(image_owner, {runtime.HALO_IMAGE_FLOW_COMPONENT: (Graph('image'),)})
    controller.reconfigure_rpu_execution({'components': {'action_expert': {'action': {'chunk_size': 32}}}})
    assert events == [('warmup', 'action'), ('clear', 'action')]
    assert controller.component_generation('action_expert') == 1
    assert controller.component_generation('image_flow') == 0
    assert action_owner._execution_session is image_owner._execution_session
    assert action_owner._rpu_execution == controller.rpu_execution
    from rpu_backend.api._execution import record_planner_native
    from rpu_backend.runtime.execution_planner import PlannerCalibrationSelection, PlannerCostScope
    record_planner_native(image_owner, 'action', 'image_flow', ('halo_image_flow', 17))
    selection = PlannerCalibrationSelection(PlannerCostScope('a' * 64, 'b' * 64, 'c' * 64), 'd' * 64, 'e' * 64)
    events.clear()
    old = controller.rpu_execution
    controller._execution_session._select_planner_calibration(image_owner, 'action', selection, component='image_flow', native_prefix='halo_image_flow')
    assert controller.rpu_execution == old
    assert sorted(events) == sorted([('warmup', 'action'), ('clear', 'action'), ('warmup', 'image'), ('clear', 'image')])
    assert controller.component_generation('action_expert') == 2
    assert controller.component_generation('image_flow') == 1

def test_text_decode_fixed_receipt_keeps_prefix_history_in_ddr():

    class Owner:
        _closed = False
    controller = runtime.HaloExecutionController()
    owner = Owner()
    controller.attach(owner, {runtime.HALO_TEXT_DECODE_COMPONENT: ()})
    plan = runtime._plan_halo_fixed_execution(owner, component=runtime.HALO_TEXT_DECODE_COMPONENT, stage='decode', logical_len=1, execution_len=1, chunk_size=16, position=4706, handle=91, kv_route='DDR_REQUIRED')
    receipt = owner._rpu_last_execution_receipts['text_decode']
    assert plan.selected.stage_tuple.compute_chunk == 16
    assert receipt['authority'] == 'FIXED_ABI_SINGLETON'
    assert receipt['semantic_spans'] == 'INPUT_OWNED'
    assert receipt['kv_route'] == 'DDR_REQUIRED'
    assert receipt['kv_route_reason'] == 'PREFIX_HISTORY_DDR_REQUIRED_NO_RAW_RESIDENCY_ABI'
    assert receipt['attention_reason'] == receipt['kv_route_reason']

def test_production_fmb_helper_rejects_empty_native_descriptor(monkeypatch):
    empty_plan = SimpleNamespace(selected=SimpleNamespace(stage_tuple=SimpleNamespace(physical_descriptor=())))

    def return_empty_plan(*args, plan_result_sink, **kwargs):
        plan_result_sink(empty_plan)
    monkeypatch.setattr(runtime, 'plan_bounded_prefill_execution', return_empty_plan)

    class Owner:
        pass
    owner = Owner()
    controller = runtime.HaloExecutionController()
    controller.attach(owner, {runtime.HALO_ACTION_COMPONENT: ()})
    with pytest.raises(RuntimeError, match='A6 winner has no native descriptor'):
        runtime._plan_halo_fmb_execution(owner, component=runtime.HALO_ACTION_COMPONENT, stage='action', logical_len=18, position=4706, handle=91, native_prefix='halo_action_expert', resolve_stage_domain=lambda _length: (), kv_route='DDR_REQUIRED')
    assert not hasattr(owner, '_rpu_last_execution_receipts')
_RUNNER_NAMES = ('action', 'text_cache', 'text_decode', 'image_flow', 'vit')

def _build_runner(name, controller=None):
    kwargs = {} if controller is None else {'execution_controller': controller}
    if name == 'action':
        return runtime.build_halo_action_expert('unused', rope_position=11, **kwargs)
    if name == 'text_cache':
        return text_cache.build_halo_text_cache('unused', **kwargs)
    if name == 'text_decode':
        return text_decode.build_halo_text_decode('unused', **kwargs)
    if name == 'image_flow':
        return image_flow.build_halo_image_flow('unused', **kwargs)
    return vit_cache.build_halo_vit_cache('unused', vcfg=vit_cache.VitCacheConfig(vit_hidden=8), **kwargs)

@pytest.mark.parametrize('name', _RUNNER_NAMES)
@pytest.mark.parametrize('failure', ('clear', 'invariant', 'destroy', 'dead_finalizer'))
def test_explicit_retirement_failure_is_terminal_without_gc_retry(native_runtime, name, failure):
    from rpu_backend.api import causal_lm
    runner = _build_runner(name)
    resources = runner._halo_resources
    controller = runner._halo_execution_controller
    handles = dict(resources.handles)
    graph = resources.graphs[-1]
    if failure == 'clear':
        native_runtime['graph_error'] = graph.name
    elif failure == 'invariant':
        native_runtime['bad_invariant'] = graph.name
    elif failure == 'destroy':
        native_runtime['destroy_error'] = native_runtime['alive'][next(iter(handles.values()))[0]]
    else:
        next(iter(resources.finalizers.values())).detach()
    start = len(native_runtime['events'])
    with pytest.raises(RuntimeError, match='retirement|invariant|dead finalizer'):
        runner.close()
    assert not runner._closed
    assert resources.handles == handles
    assert resources.failed
    assert controller._halo_failed_retirement is resources
    assert controller._execution_session.stats()['state'] == 'POISONED'
    assert not controller._execution_session._closed
    assert causal_lm._LIVE_TERMINAL_REASON is not None
    if failure != 'destroy':
        assert not any((event[0] == 'destroy' for event in native_runtime['events'][start:]))
    events = list(native_runtime['events'])
    with pytest.raises(RuntimeError, match='already failed'):
        runner.close()
    for finalizer in resources.finalizers.values():
        finalizer()
    assert native_runtime['events'] == events
    for new_name in _RUNNER_NAMES:
        with pytest.raises(RuntimeError, match='unsafe process'):
            _build_runner(new_name)
    assert native_runtime['events'] == events

def test_partial_dual_destroy_detaches_only_confirmed_handle(native_runtime):
    runner = _build_runner('vit')
    (lm_handle, vit_handle) = (runner._lm_handle, runner._vit_handle)
    native_runtime['destroy_error'] = 'vit'
    start = len(native_runtime['events'])
    with pytest.raises(RuntimeError, match='native retirement'):
        runner.close()
    assert [event[0] for event in native_runtime['events'][start:]] == ['clear', 'clear', 'destroy', 'destroy']
    assert lm_handle not in native_runtime['alive']
    assert native_runtime['alive'] == {vit_handle: 'vit'}
    assert runner._lm_handle is None
    assert runner._vit_handle == vit_handle
    assert not runner._lm_handle_finalizer.alive
    assert runner._vit_handle_finalizer.alive
    assert not runner._closed
    events = list(native_runtime['events'])
    runner._vit_handle_finalizer()
    with pytest.raises(RuntimeError, match='already failed'):
        runner.close()
    assert native_runtime['events'] == events

@pytest.mark.parametrize('name', _RUNNER_NAMES)
def test_close_during_forward_rejects_before_touching_resources(native_runtime, name):
    from rpu_backend.api import causal_lm
    runner = _build_runner(name)
    session = runner._execution_session
    events = list(native_runtime['events'])
    with session.execute():
        with pytest.raises(RuntimeError, match='active forward'):
            runner.close()
        assert native_runtime['events'] == events
        assert session.stats()['state'] == 'EXECUTING'
        assert not runner._closed
        assert causal_lm._LIVE_TERMINAL_REASON is None
    runner.close()
    assert native_runtime['alive'] == {}

@pytest.mark.parametrize('name', _RUNNER_NAMES)
def test_closed_child_rejects_real_public_entries_without_closing_sibling(native_runtime, name):
    controller = runtime.HaloExecutionController()
    runner = _build_runner(name, controller)
    sibling_name = 'image_flow' if name == 'action' else 'action'
    sibling = _build_runner(sibling_name, controller)
    session = controller._execution_session
    sibling_entries = {key: value for (key, value) in controller._owners.items() if value[0]() is sibling}
    with session.execute():
        pass
    runner.close()
    closed_events = list(native_runtime['events'])
    assert controller._owners == sibling_entries
    assert session.stats()['generation'] == 0
    assert session.stats()['state'] == 'QUIESCENT'
    entries = {'action': ('step', 'rollout_cond', 'rollout_cfg', 'insert_prefix', 'make_cfg_branch'), 'text_cache': ('prefill',), 'text_decode': ('decode_step',), 'image_flow': ('denoise_step', 'rollout'), 'vit': ('run', 'prefill_frames', 'get_lm_kv')}
    for method in entries[name]:
        with pytest.raises(RuntimeError, match='component is closed'):
            getattr(runner, method)()
    with pytest.raises(RuntimeError, match='closed HALO'):
        controller.attach(runner, {})
    with session.execute():
        assert not sibling._closed
    assert native_runtime['events'] == closed_events
    sibling.close()
    assert controller._owners == {}
    assert native_runtime['alive'] == {}

def test_cold_child_can_be_replaced_but_hot_attach_remains_atomic_rejection(native_runtime):
    controller = runtime.HaloExecutionController()
    old = _build_runner('action', controller)
    old.close()
    controller._execution_session.require_cold()
    new = _build_runner('action', controller)
    assert controller._owners[runtime.HALO_ACTION_COMPONENT][0]() is new
    with controller._execution_session.execute():
        pass
    new.close()
    events = list(native_runtime['events'])
    with pytest.raises(RuntimeError, match='cold|first forward'):
        _build_runner('action', controller)
    assert native_runtime['events'] == events
    assert controller._owners == {}

def test_failed_multi_component_attach_does_not_publish_partial_owner(native_runtime):
    controller = runtime.HaloExecutionController()
    existing = _build_runner('action', controller)

    class Owner:
        _closed = False
    new = Owner()
    before = dict(controller._owners)
    with pytest.raises(RuntimeError, match='live owner'):
        controller.attach(new, {runtime.HALO_IMAGE_FLOW_COMPONENT: (), runtime.HALO_ACTION_COMPONENT: ()})
    assert controller._owners == before
    assert not hasattr(new, '_planner_cost_session')
    existing.close()

@pytest.mark.parametrize('name', _RUNNER_NAMES)
def test_actual_gc_retires_all_graphs_before_handles(native_runtime, name):
    runner = _build_runner(name)
    controller = runner._halo_execution_controller
    runner_ref = weakref.ref(runner)
    finalizers = tuple(runner._halo_resources.finalizers.values())
    graph_count = len(runner._halo_resources.graphs)
    handle_count = len(runner._halo_resources.handles)
    start = len(native_runtime['events'])
    del runner
    gc.collect()
    assert runner_ref() is None
    assert [event[0] for event in native_runtime['events'][start:]] == ['clear'] * graph_count + ['destroy'] * handle_count
    assert native_runtime['alive'] == {}
    assert controller._owners == {}
    assert all((not finalizer.alive for finalizer in finalizers))

@pytest.mark.parametrize('name', _RUNNER_NAMES)
def test_builder_failure_keeps_original_error_when_raw_cleanup_fails(native_runtime, name):

    class BuildInterrupted(BaseException):
        pass
    (operation, component) = {'action': ('action_weights', 'action'), 'text_cache': ('decoder_weights', 'decoder'), 'text_decode': ('decoder_weights', 'decoder'), 'image_flow': ('image_weights', 'image'), 'vit': ('decoder_weights', 'decoder')}[name]
    original = BuildInterrupted('original builder failure')
    controller = runtime.HaloExecutionController()
    native_runtime.update(error_op=operation, error=original, destroy_error=component)
    with pytest.raises(BuildInterrupted) as caught:
        _build_runner(name, controller)
    assert caught.value is original
    assert any(('cleanup also failed' in note for note in original.__notes__))
    resources = controller._halo_failed_retirement
    assert resources.failed
    assert len(resources.handles) == (2 if name == 'vit' else 1)
    assert controller._owners == {}
    assert controller._execution_session.stats()['state'] == 'POISONED'
    events = list(native_runtime['events'])
    runtime._finalize_halo_resources(resources)
    assert native_runtime['events'] == events

def test_second_graph_constructor_failure_retires_pending_first_graph(native_runtime, monkeypatch):
    calls = 0

    def graph(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('second graph constructor failed')
        return _GraphCache(**kwargs)
    monkeypatch.setattr(rpu_backend.graph, 'GraphCache', graph)
    with pytest.raises(RuntimeError, match='second graph constructor'):
        _build_runner('vit')
    assert [event[0] for event in native_runtime['events'][-3:]] == ['clear', 'destroy', 'destroy']
    assert native_runtime['alive'] == {}

@pytest.mark.parametrize('cost_kind', ('native_only', 'scope_only', 'outer_only'))
@pytest.mark.parametrize('fail_close', (False, True))
def test_retirement_drops_only_target_native_owner_and_never_transfers_costs(native_runtime, cost_kind, fail_close):
    from rpu_backend.runtime.execution_planner import PlannerCostScope
    controller = runtime.HaloExecutionController()
    owner = _build_runner('image_flow', controller)
    sibling = _build_runner('action', controller)
    session = controller._execution_session
    (component, prefix) = (runtime.HALO_IMAGE_FLOW_COMPONENT, 'halo_image_flow')
    (key, proof_key) = ((component, 'action', prefix), (component, prefix))
    sibling_key = (runtime.HALO_ACTION_COMPONENT, 'action', 'halo_action_expert')
    session._record_planner_native(owner, component, 'action', (prefix, owner._handle))
    session._record_planner_native(sibling, *sibling_key[:2], (sibling_key[2], sibling._handle))
    scope = PlannerCostScope('a' * 64, 'b' * 64, 'c' * 64)
    sibling_proof = (sibling_key[0], sibling_key[2])
    session._planner_native_catalogs[sibling_proof] = (sibling._handle, scope)
    session._planner_native_revalidators[sibling_proof] = lambda captured=sibling: captured
    session._install_planner_costs(sibling_key[0], 'action', lambda : (scope, ()), native_prefix=sibling_key[2], allow_singleton=True)
    if cost_kind != 'native_only':
        session._install_planner_costs(component, 'action', lambda : (scope, ()), native_prefix=prefix, allow_singleton=True)
    if cost_kind != 'outer_only':
        session._planner_native_catalogs[proof_key] = (owner._handle, scope)
        session._planner_native_revalidators[proof_key] = lambda captured=owner: captured
    native_before = dict(session._planner_native)
    costs_before = dict(session._planner_costs)
    catalogs_before = dict(session._planner_native_catalogs)
    revalidators_before = dict(session._planner_native_revalidators)
    session.require_cold()
    if fail_close:
        native_runtime['graph_error'] = owner._graph_cache.name
        with pytest.raises(RuntimeError, match='graph retirement'):
            owner.close()
        assert session._planner_native == native_before
        assert session._planner_native_revalidators == revalidators_before
    else:
        owner.close()
        session.require_cold()
        assert key not in session._planner_native
        assert session._planner_native[sibling_key] == native_before[sibling_key]
        assert proof_key not in session._planner_native_revalidators
        assert session._planner_native_revalidators[sibling_proof] is revalidators_before[sibling_proof]
        events = list(native_runtime['events'])
        with pytest.raises(RuntimeError, match='installed costs'):
            _build_runner('image_flow', controller)
        assert native_runtime['events'] == events
        sibling.close()
    assert session._planner_costs == costs_before
    assert session._planner_native_catalogs == catalogs_before

@pytest.mark.parametrize('name', _RUNNER_NAMES)
def test_graph_constructor_failure_retires_all_pending_handles(native_runtime, monkeypatch, name):

    def fail_graph(**kwargs):
        raise RuntimeError('first graph constructor failed')
    monkeypatch.setattr(rpu_backend.graph, 'GraphCache', fail_graph)
    with pytest.raises(RuntimeError, match='first graph constructor'):
        _build_runner(name)
    assert native_runtime['alive'] == {}

@pytest.mark.parametrize('name', _RUNNER_NAMES)
def test_gc_is_not_rooted_by_finalizer_through_native_planner_owner(native_runtime, name):
    runner = _build_runner(name)
    controller = runner._halo_execution_controller
    component = next(iter(controller._owners))
    handle = next(iter(runner._halo_resources.handles.values()))[0]
    controller._execution_session._record_planner_native(runner, component, 'prefill', ('halo_fixture', handle))
    (runner_ref, controller_ref) = (weakref.ref(runner), weakref.ref(controller))
    resources = runner._halo_resources
    del runner, controller
    gc.collect()
    assert runner_ref() is None
    assert controller_ref() is None
    assert resources.closed and (not resources.handles)
    assert native_runtime['alive'] == {}

@pytest.mark.parametrize('name', _RUNNER_NAMES)
def test_failed_gc_also_latches_process_and_preserves_unretired_handle(native_runtime, name):
    from rpu_backend.api import causal_lm
    runner = _build_runner(name)
    controller = runner._halo_execution_controller
    resources = runner._halo_resources
    native_runtime['graph_error'] = resources.graphs[-1].name
    handles = dict(resources.handles)
    start = len(native_runtime['events'])
    del runner
    gc.collect()
    assert resources.failed and resources.handles == handles
    assert controller._halo_failed_retirement is resources
    assert controller._execution_session.stats()['state'] == 'POISONED'
    assert causal_lm._LIVE_TERMINAL_REASON is not None
    assert not any((event[0] == 'destroy' for event in native_runtime['events'][start:]))
    events = list(native_runtime['events'])
    runtime._finalize_halo_resources(resources)
    assert native_runtime['events'] == events

def test_last_owner_failed_gc_retains_resources_without_an_external_controller(native_runtime):
    runner = _build_runner('text_cache')
    controller = runner._halo_execution_controller
    component = next(iter(controller._owners))
    resources = runner._halo_resources
    handle = next(iter(resources.handles.values()))[0]
    controller._execution_session._record_planner_native(runner, component, 'prefill', ('halo_fixture', handle))
    native_runtime['graph_error'] = resources.graphs[0].name
    resource_ref = weakref.ref(resources)
    del runner, controller, resources
    gc.collect()
    gc.collect()
    retained = resource_ref()
    assert retained is not None and retained.failed and (not retained.closed)
    assert runtime._FAILED_RETIREMENTS == [retained]
    events = list(native_runtime['events'])
    runtime._finalize_halo_resources(retained)
    gc.collect()
    assert native_runtime['events'] == events
    assert any((value[0] == handle for value in retained.handles.values()))

def test_child_gc_during_sibling_forward_is_terminal_without_address_changes(native_runtime):
    from rpu_backend.api import causal_lm
    controller = runtime.HaloExecutionController()
    child = _build_runner('text_cache', controller)
    sibling = _build_runner('action', controller)
    resources = child._halo_resources
    events = list(native_runtime['events'])
    with controller._execution_session.execute():
        del child
        gc.collect()
        assert resources.failed and (not resources.closed)
        assert native_runtime['events'] == events
        assert controller._execution_session.stats()['state'] == 'POISONED'
        assert causal_lm._LIVE_TERMINAL_REASON is not None
    with pytest.raises(RuntimeError, match='unsafe'):
        sibling.step()
    sibling._handle_finalizer()
    assert native_runtime['events'] == events

@pytest.mark.parametrize('owned', (False, True))
@pytest.mark.parametrize('forward_fails', (False, True))
@pytest.mark.parametrize('cleanup_fails', (False, True))
def test_actual_prefix_bank_helper_preserves_forward_and_cleanup_errors(native_runtime, monkeypatch, owned, forward_fails, cleanup_fails):
    original = ValueError('original prefix prefill failure')
    expected = {0: (_FakeTensor(), _FakeTensor())}
    observed = []

    @runtime._halo_execution_serialized
    def prefill(owner, *args, **kwargs):
        observed.append(owner)
        if forward_fails:
            raise original
        return expected
    monkeypatch.setattr(text_cache.HaloTextCacheUpdate, 'prefill', prefill)
    supplied = None if owned else _build_runner('text_cache')
    native_runtime['destroy_error'] = 'decoder' if cleanup_fails else None

    def call():
        return halo.build_native_prefix_banks('unused', img_uncond_text_pack={'packed_text_ids': _FakeTensor(), 'packed_text_position_ids': _FakeTensor()}, text_cache=supplied)
    if forward_fails:
        with pytest.raises(ValueError) as caught:
            call()
        assert caught.value is original
        notes = getattr(original, '__notes__', ())
        assert bool(notes) is (owned and cleanup_fails)
        if notes:
            assert 'cleanup also failed' in notes[0]
    elif owned and cleanup_fails:
        with pytest.raises(RuntimeError, match='native retirement'):
            call()
    else:
        assert call() == {'cfg_img': expected}
    runner = observed[0]
    if owned:
        assert runner._closed is not cleanup_fails
        if cleanup_fails:
            assert runner._halo_resources.failed
            assert runner._execution_session.stats()['state'] == 'POISONED'
    else:
        assert not runner._closed
        assert not any((event[0] == 'destroy' for event in native_runtime['events']))
        native_runtime['destroy_error'] = None
        runner.close()

@pytest.mark.parametrize('field,value', [('attn_tp', 4), ('attn_tp', 6), ('mlp_cores', 4), ('mlp_cores', 6)])
@pytest.mark.parametrize('entry', ['action', 'text_cache', 'text_decode', 'image_flow', 'vit_cache', 'prefix_banks', 'cfg_branch', 'load_branches', 'load_gen_branch'])
def test_actual_halo_entries_reject_native_layout_mismatch_before_resources(native_runtime, monkeypatch, entry, field, value):
    from rpu_backend.adapters.halo import weights
    cfg = replace(HaloConfig(), **{field: value})
    touched = []

    def forbidden(*args, **kwargs):
        touched.append('load/swizzle/allocation')
        raise AssertionError('invalid HALO topology reached a resource boundary')
    for module in (runtime, text_cache, text_decode, vit_cache):
        monkeypatch.setattr(module, 'load_halo_branches', forbidden)
    monkeypatch.setattr(image_flow, 'load_halo_gen_branch', forbidden)
    for module in (weights, vit_cache):
        monkeypatch.setattr(module, '_HaloTensorStore', forbidden)
    for module in (runtime, text_cache, text_decode, image_flow, vit_cache):
        monkeypatch.setattr(module, 'RPUCache', forbidden)
    monkeypatch.setattr(torch, 'empty', forbidden)
    monkeypatch.setattr(torch, 'zeros', forbidden)
    calls = {name: builder for (name, _, _, builder) in _single_builders(cfg)}
    calls.update({'vit_cache': lambda : _vit_builder(cfg), 'prefix_banks': lambda : halo.build_native_prefix_banks('unused', cfg=cfg, img_uncond_text_pack={}, text_cache=SimpleNamespace(prefill=forbidden)), 'cfg_branch': lambda : runtime.build_cfg_branch('test', 0, [None] * cfg.num_layers, [None] * cfg.num_layers, rope_position=0, prefix_len=0, epoch=1, cfg=cfg), 'load_branches': lambda : weights.load_halo_branches('unused', cfg), 'load_gen_branch': lambda : weights.load_halo_gen_branch('unused', cfg)})
    with pytest.raises(ValueError, match=f'HALO {field}=.*expected'):
        calls[entry]()
    assert touched == []
    assert native_runtime['events'] == []
    assert native_runtime['alive'] == {}

@pytest.mark.parametrize('field,value', [('attn_tp', True), ('attn_tp', 2.0), ('mlp_cores', 8.0)])
def test_halo_native_core_layout_does_not_coerce_nonintegers(native_runtime, field, value):
    cfg = replace(HaloConfig(), **{field: value})
    with pytest.raises(ValueError, match=f'HALO {field}='):
        runtime.build_halo_action_expert('unused', rope_position=0, cfg=cfg)
    assert native_runtime['events'] == []

def test_direct_halo_step_runner_rejects_layout_before_input_allocation(monkeypatch):
    runner = object.__new__(runtime.HaloStepRunner)
    runner.cfg = replace(HaloConfig(), mlp_cores=4)

    def forbidden(*args, **kwargs):
        raise AssertionError('invalid HALO topology allocated its input buffer')
    monkeypatch.setattr(torch, 'empty', forbidden)
    with pytest.raises(ValueError, match='HALO mlp_cores='):
        runner.__post_init__()

@pytest.mark.parametrize('entry', ['action', 'text_cache', 'text_decode', 'image_flow'])
def test_halo_core_guard_preserves_valid_noncore_configuration(native_runtime, entry):
    cfg = replace(HaloConfig(), prefix_len=16)
    calls = {name: builder for (name, _, _, builder) in _single_builders(cfg)}
    runner = calls[entry]()
    assert runner.cfg is cfg
    assert any((event[0] == 'set' for event in native_runtime['events']))
    runner.close()
    assert native_runtime['alive'] == {}
