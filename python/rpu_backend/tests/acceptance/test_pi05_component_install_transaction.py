from __future__ import annotations
import gc
import weakref
from types import SimpleNamespace
import pytest
import torch
import rpu_backend
from rpu_backend.adapters.pi05 import adarms, gemma, w8a16

class _FakeTensor:

    def __init__(self, *shape):
        self.shape = shape
        self.dtype = torch.float16

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def dim(self):
        return len(self.shape)

    def unsqueeze(self, dim):
        return _FakeTensor(1, *self.shape)

class _Rotary:
    max_position_embeddings = 32

    def __call__(self, hidden, positions):
        return (_FakeTensor(32, 8), _FakeTensor(32, 8))

class _PublishInterrupted(BaseException):
    pass

class _Model:

    def arm_assignment_failure(self, name):
        object.__setattr__(self, '_fail_assignment', name)

    def __setattr__(self, name, value):
        if self.__dict__.get('_fail_assignment') == name:
            object.__setattr__(self, '_fail_assignment', None)
            raise _PublishInterrupted(f'synthetic publication interruption for {name}')
        object.__setattr__(self, name, value)

def _linear(out_features, in_features):
    return SimpleNamespace(weight=_FakeTensor(out_features, in_features), out_features=out_features)

def _model():
    hidden = 128
    intermediate = 256
    layer = SimpleNamespace(self_attn=SimpleNamespace(q_proj=_linear(hidden, hidden), k_proj=_linear(64, hidden), v_proj=_linear(64, hidden), o_proj=_linear(hidden, hidden)), mlp=SimpleNamespace(gate_proj=_linear(intermediate, hidden), up_proj=_linear(intermediate, hidden), down_proj=_linear(hidden, intermediate)), input_layernorm=SimpleNamespace(weight=_FakeTensor(hidden)), post_attention_layernorm=SimpleNamespace(weight=_FakeTensor(hidden)))
    model = _Model()
    model.layers = [layer]
    model.config = SimpleNamespace(head_dim=16, num_attention_heads=8, num_key_value_heads=4, hidden_size=hidden, intermediate_size=intermediate, rms_norm_eps=1e-06, num_hidden_layers=1)
    model.rotary_emb = _Rotary()
    model.norm = SimpleNamespace(weight=_FakeTensor(hidden))
    model.forward = lambda *args, **kwargs: 'original'
    return model

@pytest.fixture
def native_install(monkeypatch):
    from rpu_backend.api import _execution, causal_lm
    from rpu_backend.runtime import _native_retirement
    gc.collect()
    monkeypatch.setattr(_execution, '_UNSAFE_PROCESS_REASON', None)
    monkeypatch.setattr(causal_lm, '_LIVE_REF', None)
    monkeypatch.setattr(causal_lm, '_LIVE_TERMINAL_REASON', None)
    monkeypatch.setattr(_native_retirement, '_FAILED_RETIREMENTS', [])
    state = {kind: {'next_handle': 1, 'alive': set(), 'events': [], 'fail_set': set(), 'fail_destroy': set()} for kind in ('gemma', 'adarms')}

    def install_native(kind, module):
        component = state[kind]

        def create(linear_acc32=False):
            handle = component['next_handle']
            component['next_handle'] += 1
            component['alive'].add(handle)
            component.setdefault('accumulation', {})[handle] = linear_acc32
            component['events'].append(('create', handle))
            return handle

        def set_weights(handle, *args):
            component['events'].append(('set_weights', handle))
            if handle in component['fail_set']:
                raise RuntimeError(f'synthetic {kind} set_weights failure for {handle}')

        def destroy(handle):
            component['events'].append(('destroy', handle))
            if handle in component['fail_destroy']:
                raise RuntimeError(f'synthetic {kind} destroy failure for {handle}')
            if handle not in component['alive']:
                raise RuntimeError(f'double destroy for {kind} handle {handle}')
            component['alive'].remove(handle)
        monkeypatch.setattr(module.torch.ops.rpu, f'{kind}_create', create, raising=False)
        monkeypatch.setattr(module.torch.ops.rpu, f'{kind}_set_weights', set_weights, raising=False)
        monkeypatch.setattr(module.torch.ops.rpu, f'{kind}_set_chunk_envelope', lambda handle, max_kv_len, chunk: None, raising=False)
        if kind == 'gemma':
            monkeypatch.setattr(module.torch.ops.rpu, 'gemma_set_prefill_pair_rows', lambda handle, rows: None, raising=False)
        monkeypatch.setattr(module.torch.ops.rpu, f'{kind}_destroy', destroy, raising=False)
    install_native('gemma', gemma)
    install_native('adarms', adarms)

    class RPUCache:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class GraphCache:

        def __init__(self):
            self.clear_calls = 0

        def clear(self):
            self.clear_calls += 1

        def cache_invariant_ok(self):
            return True
    monkeypatch.setattr(gemma, 'RPUCache', RPUCache)
    monkeypatch.setattr(adarms, 'RPUCache', RPUCache)
    monkeypatch.setattr(rpu_backend.graph, 'GraphCache', GraphCache)
    monkeypatch.setattr(gemma.torch, 'zeros', lambda *a, **k: _FakeTensor(1, 1, 128))
    monkeypatch.setattr(gemma.torch, 'arange', lambda *a, **k: _FakeTensor(32))
    monkeypatch.setattr(adarms, '_prepare_adarms_dense_weights', lambda model: ([_FakeTensor(384, 128)], [_FakeTensor(384)], [_FakeTensor(384, 128)], [_FakeTensor(384)]))
    empty_scales = lambda *a, **k: ([], [], [], [], [], [], [])
    monkeypatch.setattr(w8a16, 'gemma_decoder_scale_lists', empty_scales)
    monkeypatch.setattr(w8a16, 'pi05_expert_scale_lists', empty_scales)
    from rpu_backend.graph import lazy_init_guard
    monkeypatch.setattr(lazy_init_guard, '_verify_lazy_init', lambda model: None)
    return state
_COMPONENTS = {'gemma': {'module': gemma, 'install': gemma.patch_gemma_model_for_rpu_all_layers_once, 'handle': '_rpu_vlm_decoder_handle', 'finalizer': '_rpu_vlm_decoder_handle_finalizer', 'cache': '_rpu_gemma_graph_cache'}, 'adarms': {'module': adarms, 'install': adarms.patch_adarms_model_for_rpu_all_layers_once, 'handle': '_rpu_action_handle', 'finalizer': '_rpu_action_handle_finalizer', 'cache': '_rpu_adarms_graph_cache'}}

def _finalize(model, component):
    finalizer = getattr(model, component['finalizer'], None)
    if finalizer is not None and finalizer.alive:
        finalizer()

@pytest.mark.parametrize('kind', ('gemma', 'adarms', 'siglip', 'fused'))
def test_leaf_set_weights_and_destroy_failure_retains_actual_owner(monkeypatch, kind):
    from rpu_backend.api import _execution
    from rpu_backend.runtime import _native_retirement
    if kind == 'siglip':
        from rpu_backend.tests.acceptance import test_siglip_install_transaction as leaf
        native = leaf.native_install.__wrapped__(monkeypatch)
        (model, projector) = leaf._model()
        install = lambda : leaf._install(model, projector)
        resource_name = '_rpu_vision_retirement_state'
    elif kind == 'fused':
        from rpu_backend.tests.acceptance import test_pi05_fused_install_transaction as leaf
        native = leaf.native_install.__wrapped__(monkeypatch)
        (adapter, model) = leaf._adapter()
        install = adapter._install_fused_denoise_handle
        resource_name = '_rpu_fused_denoise_retirement_state'
    else:
        native = native_install.__wrapped__(monkeypatch)[kind]
        model = _model()
        component = _COMPONENTS[kind]
        install = lambda : component['install'](model)
        resource_name = component['handle'].replace('_handle', '_retirement_state')
    native['fail_set'].add(1)
    native['fail_destroy'].add(1)
    owner_ref = weakref.ref(model)
    with pytest.raises(RuntimeError, match='set_weights failure') as caught:
        install()
    resource = getattr(model, resource_name)
    assert resource in _native_retirement._FAILED_RETIREMENTS
    assert resource.owner() is model and resource.handle == 1 and (resource.failed is not None)
    assert resource.keepalive and resource.graphs
    assert all((graph.clear_calls == 1 for graph in resource.graphs))
    assert getattr(model, resource.handle_name) == 1 and native['alive'] == {1}
    assert 'cleanup failed' in ' '.join(caught.value.__notes__)
    assert _execution._UNSAFE_PROCESS_REASON
    before = list(native['events'])
    native['fail_destroy'].clear()
    with pytest.raises(RuntimeError, match='unsafe|restart|Restart|retirement'):
        install()
    if resource.finalizer.alive:
        resource.finalizer()
    resource._gc_retire()
    del model, install, caught
    if kind == 'fused':
        del adapter
    gc.collect()
    assert owner_ref() is resource.owner() and owner_ref() is not None
    assert native['events'] == before

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_missing_layers_fails_before_native_create(native_install, kind):
    component = _COMPONENTS[kind]
    model = _model()
    del model.layers
    with pytest.raises(RuntimeError, match='no \\.layers'):
        component['install'](model)
    assert native_install[kind]['events'] == []
    assert not hasattr(model, component['handle'])

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_prepare_failure_does_not_create_or_publish(native_install, monkeypatch, kind):
    component = _COMPONENTS[kind]
    model = _model()
    original_forward = model.forward

    class FailingCache:

        def __init__(self, **kwargs):
            raise RuntimeError('synthetic cache preparation failure')
    monkeypatch.setattr(component['module'], 'RPUCache', FailingCache)
    with pytest.raises(RuntimeError, match='cache preparation failure'):
        component['install'](model)
    assert native_install[kind]['events'] == []
    assert not hasattr(model, component['handle'])
    assert model.forward is original_forward

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_set_weights_failure_cleans_pending_handle(native_install, kind):
    component = _COMPONENTS[kind]
    model = _model()
    original_forward = model.forward
    native_install[kind]['fail_set'].add(1)
    with pytest.raises(RuntimeError, match='set_weights failure'):
        component['install'](model)
    assert native_install[kind]['events'] == [('create', 1), ('set_weights', 1), ('destroy', 1)]
    assert native_install[kind]['alive'] == set()
    assert not hasattr(model, component['handle'])
    assert model.forward is original_forward

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_failed_replacement_preserves_published_install(native_install, kind):
    component = _COMPONENTS[kind]
    model = _model()
    assert component['install'](model) == 1
    old_forward = model.forward
    old_cache = getattr(model, component['cache'])
    old_finalizer = getattr(model, component['finalizer'])
    old_rope = (getattr(model, '_gemma_rope_cos', None), getattr(model, '_gemma_rope_sin', None))
    native_install[kind]['events'].clear()
    native_install[kind]['fail_set'].add(2)
    with pytest.raises(RuntimeError, match='set_weights failure'):
        component['install'](model)
    assert native_install[kind]['events'] == [('create', 2), ('set_weights', 2), ('destroy', 2)]
    assert native_install[kind]['alive'] == {1}
    assert getattr(model, component['handle']) == 1
    assert getattr(model, component['finalizer']) is old_finalizer
    assert old_finalizer.alive
    assert getattr(model, component['cache']) is old_cache
    assert model.forward is old_forward
    if kind == 'adarms':
        assert (model._gemma_rope_cos, model._gemma_rope_sin) == old_rope
    _finalize(model, component)

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_old_destroy_failure_aborts_replacement(native_install, kind):
    from rpu_backend.api import _execution
    from rpu_backend.runtime import _native_retirement
    component = _COMPONENTS[kind]
    model = _model()
    assert component['install'](model) == 1
    old_finalizer = getattr(model, component['finalizer'])
    native_install[kind]['events'].clear()
    native_install[kind]['fail_destroy'].add(1)
    with pytest.raises(RuntimeError, match='destroy failure'):
        component['install'](model)
    assert native_install[kind]['events'] == [('create', 2), ('set_weights', 2), ('destroy', 1)]
    assert native_install[kind]['alive'] == {1, 2}
    pending = getattr(model, component['handle'].replace('_handle', '_retirement_state'))
    assert getattr(model, component['handle']) == pending.handle == 2
    assert pending.failed is not None and pending.keepalive and pending.graphs
    assert {resource.handle for resource in _native_retirement._FAILED_RETIREMENTS} == {1, 2}
    assert _execution._UNSAFE_PROCESS_REASON
    before = list(native_install[kind]['events'])
    native_install[kind]['fail_destroy'].clear()
    with pytest.raises(RuntimeError, match='unsafe|restart|Restart|retirement'):
        component['install'](model)
    old_finalizer()
    _finalize(model, component)
    assert native_install[kind]['events'] == before

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_first_install_publication_baseexception_rolls_back(native_install, kind):
    component = _COMPONENTS[kind]
    model = _model()
    original_forward = model.forward
    model.arm_assignment_failure(component['finalizer'])
    with pytest.raises(_PublishInterrupted, match='publication interruption'):
        component['install'](model)
    assert native_install[kind]['events'] == [('create', 1), ('set_weights', 1), ('destroy', 1)]
    assert native_install[kind]['alive'] == set()
    assert not hasattr(model, component['handle'])
    assert not hasattr(model, component['cache'])
    assert model.forward is original_forward
    if kind == 'adarms':
        assert not hasattr(model, '_gemma_rope_cos')
        assert not hasattr(model, '_gemma_rope_sin')

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_replacement_publication_baseexception_preserves_old_install(native_install, kind):
    component = _COMPONENTS[kind]
    model = _model()
    assert component['install'](model) == 1
    old_forward = model.forward
    old_cache = getattr(model, component['cache'])
    old_rpu_cache = model._rpu_cache
    old_finalizer = getattr(model, component['finalizer'])
    old_rope = (getattr(model, '_gemma_rope_cos', None), getattr(model, '_gemma_rope_sin', None))
    native_install[kind]['events'].clear()
    model.arm_assignment_failure('forward')
    with pytest.raises(_PublishInterrupted, match='publication interruption'):
        component['install'](model)
    assert native_install[kind]['events'] == [('create', 2), ('set_weights', 2), ('destroy', 2)]
    assert native_install[kind]['alive'] == {1}
    assert getattr(model, component['handle']) == 1
    assert getattr(model, component['finalizer']) is old_finalizer
    assert old_finalizer.alive
    assert getattr(model, component['cache']) is old_cache
    assert model._rpu_cache is old_rpu_cache
    assert model.forward is old_forward
    if kind == 'adarms':
        assert (model._gemma_rope_cos, model._gemma_rope_sin) == old_rope
    _finalize(model, component)

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_freeze_baseexception_rolls_back_before_old_retirement(native_install, monkeypatch, kind):
    component = _COMPONENTS[kind]
    model = _model()
    assert component['install'](model) == 1
    old_forward = model.forward
    old_cache = getattr(model, component['cache'])
    old_finalizer = getattr(model, component['finalizer'])
    native_install[kind]['events'].clear()
    from rpu_backend.graph import lazy_init_guard

    def fail_freeze(_model):
        raise _PublishInterrupted('synthetic freeze interruption')
    monkeypatch.setattr(lazy_init_guard, '_verify_lazy_init', fail_freeze)
    with pytest.raises(_PublishInterrupted, match='freeze interruption'):
        component['install'](model)
    assert native_install[kind]['events'] == [('create', 2), ('set_weights', 2), ('destroy', 2)]
    assert native_install[kind]['alive'] == {1}
    assert getattr(model, component['handle']) == 1
    assert getattr(model, component['finalizer']) is old_finalizer
    assert old_finalizer.alive
    assert getattr(model, component['cache']) is old_cache
    assert model.forward is old_forward
    _finalize(model, component)

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_successful_replacement_retires_old_finalizer(native_install, kind):
    component = _COMPONENTS[kind]
    model = _model()
    assert component['install'](model) == 1
    old_finalizer = getattr(model, component['finalizer'])
    old_cache = getattr(model, component['cache'])
    native_install[kind]['events'].clear()
    assert component['install'](model) == 2
    assert native_install[kind]['events'] == [('create', 2), ('set_weights', 2), ('destroy', 1)]
    assert native_install[kind]['alive'] == {2}
    assert getattr(model, component['handle']) == 2
    assert getattr(model, component['finalizer']).alive
    assert not old_finalizer.alive
    assert getattr(model, component['cache']) is not old_cache
    _finalize(model, component)

@pytest.mark.parametrize('kind', tuple(_COMPONENTS))
def test_replacement_does_not_double_destroy_dead_finalizer(native_install, kind):
    component = _COMPONENTS[kind]
    model = _model()
    assert component['install'](model) == 1
    getattr(model, component['finalizer'])()
    native_install[kind]['events'].clear()
    assert component['install'](model) == 2
    assert native_install[kind]['events'] == [('create', 2), ('set_weights', 2)]
    assert native_install[kind]['alive'] == {2}
    assert getattr(model, component['handle']) == 2
    assert getattr(model, component['finalizer']).alive
    _finalize(model, component)

@pytest.mark.parametrize('kind,stage', (('gemma', 'prefill'), ('adarms', 'action')))
@pytest.mark.parametrize('acc32', (False, True))
def test_component_accumulation_is_bound_before_weights(native_install, kind, stage, acc32):
    model = _model()
    model._fmb_execution_component_config = {stage: {'linear_acc32': acc32}}
    component = _COMPONENTS[kind]
    try:
        handle = component['install'](model)
        assert native_install[kind]['accumulation'] == {handle: acc32}
        assert native_install[kind]['events'][:2] == [('create', handle), ('set_weights', handle)]
    finally:
        _finalize(model, component)
