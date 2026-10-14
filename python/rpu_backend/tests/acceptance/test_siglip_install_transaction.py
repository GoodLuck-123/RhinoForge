from __future__ import annotations
import gc
from types import SimpleNamespace
import pytest
import torch
import rpu_backend
from rpu_backend.adapters import siglip

class _FakeTensor:

    def __init__(self, *shape):
        self.shape = shape
        self.dtype = torch.float16

    def to(self, *args, **kwargs):
        return self

    def contiguous(self):
        return self

class _PublishInterrupted(BaseException):
    pass

class _Model:

    def arm_assignment_failure(self, name):
        object.__setattr__(self, '_fail_assignment', name)

    def __setattr__(self, name, value):
        if self.__dict__.get('_fail_assignment') == name:
            object.__setattr__(self, '_fail_assignment', None)
            raise _PublishInterrupted(f'synthetic publication failure: {name}')
        object.__setattr__(self, name, value)

def _linear(out_features, in_features):
    return SimpleNamespace(weight=_FakeTensor(out_features, in_features), bias=_FakeTensor(out_features), out_features=out_features)

def _model():
    hidden = 128
    intermediate = 256
    attention = SimpleNamespace(q_proj=_linear(hidden, hidden), k_proj=_linear(hidden, hidden), v_proj=_linear(hidden, hidden), out_proj=_linear(hidden, hidden))
    mlp = SimpleNamespace(fc1=_linear(intermediate, hidden), fc2=_linear(hidden, intermediate))
    layer = SimpleNamespace(self_attn=attention, mlp=mlp, layer_norm1=SimpleNamespace(weight=_FakeTensor(hidden), bias=_FakeTensor(hidden)), layer_norm2=SimpleNamespace(weight=_FakeTensor(hidden), bias=_FakeTensor(hidden)))
    embeddings = SimpleNamespace(patch_embedding=SimpleNamespace(kernel_size=(14, 14), stride=(14, 14)), _rpu_gemm_weight=_FakeTensor(hidden, hidden), _rpu_pos_emb_fused=_FakeTensor(256, hidden))
    model = _Model()
    model.encoder = SimpleNamespace(layers=[layer])
    model.config = SimpleNamespace(num_attention_heads=8, hidden_size=hidden, intermediate_size=intermediate, layer_norm_eps=1e-06)
    model.post_layernorm = SimpleNamespace(weight=_FakeTensor(hidden), bias=_FakeTensor(hidden))
    model.embeddings = embeddings
    model.forward = lambda *args, **kwargs: 'original'
    projector = _linear(hidden, hidden)
    return (model, projector)

@pytest.fixture
def native_install(monkeypatch):
    from rpu_backend.api import _execution, causal_lm
    from rpu_backend.runtime import _native_retirement
    gc.collect()
    monkeypatch.setattr(_execution, '_UNSAFE_PROCESS_REASON', None)
    monkeypatch.setattr(causal_lm, '_LIVE_REF', None)
    monkeypatch.setattr(causal_lm, '_LIVE_TERMINAL_REASON', None)
    monkeypatch.setattr(_native_retirement, '_FAILED_RETIREMENTS', [])
    state = {'next_handle': 1, 'alive': set(), 'events': [], 'fail_set': set(), 'fail_patch': set(), 'envelopes': {}, 'fail_envelope': set(), 'fail_destroy': set()}

    def create(linear_acc32=False):
        handle = state['next_handle']
        state['next_handle'] += 1
        state['alive'].add(handle)
        state.setdefault('accumulation', {})[handle] = linear_acc32
        state['events'].append(('create', handle))
        return handle

    def set_weights(handle, *args):
        state['events'].append(('set_weights', handle))
        if handle in state['fail_set']:
            raise RuntimeError(f'synthetic set_weights failure for {handle}')

    def set_patch(handle, *args):
        state['events'].append(('set_patch', handle))
        if handle in state['fail_patch']:
            raise RuntimeError(f'synthetic patch failure for {handle}')

    def set_envelope(handle, capacity, chunk):
        assert state['events'][-1] == ('set_weights', handle)
        if handle in state['fail_envelope']:
            raise RuntimeError(f'synthetic envelope failure for {handle}')
        state['envelopes'][handle] = (capacity, chunk)

    def destroy(handle):
        state['events'].append(('destroy', handle))
        if handle in state['fail_destroy']:
            raise RuntimeError(f'synthetic destroy failure for {handle}')
        if handle not in state['alive']:
            raise RuntimeError(f'double destroy for {handle}')
        state['alive'].remove(handle)

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
    monkeypatch.setattr(siglip, '_convert_siglip_weights_for_rpu', lambda *a, **k: None)
    monkeypatch.setattr(siglip, '_convert_siglip_projector_for_rpu', lambda *a, **k: None)
    monkeypatch.setattr(siglip, '_detect_and_validate_siglip_encoder_w8a16', lambda *a: False)
    monkeypatch.setattr(siglip, '_siglip_weight_to_rpu', lambda module: module.weight)
    monkeypatch.setattr(siglip, 'RPUCache', RPUCache)
    monkeypatch.setattr(rpu_backend.graph, 'GraphCache', GraphCache)
    from rpu_backend.graph import lazy_init_guard
    monkeypatch.setattr(lazy_init_guard, '_verify_lazy_init', lambda model: None)
    monkeypatch.setattr(siglip.torch.ops.rpu, 'siglip_create', create, raising=False)
    monkeypatch.setattr(siglip.torch.ops.rpu, 'siglip_set_weights', set_weights, raising=False)
    monkeypatch.setattr(siglip.torch.ops.rpu, 'siglip_set_weights_w8a16', set_weights, raising=False)
    monkeypatch.setattr(siglip.torch.ops.rpu, 'siglip_set_chunk_envelope', set_envelope, raising=False)
    monkeypatch.setattr(siglip.torch.ops.rpu, 'siglip_model_set_patch_emb', set_patch, raising=False)
    monkeypatch.setattr(siglip.torch.ops.rpu, 'siglip_destroy', destroy, raising=False)
    return state

def _install(model, projector):
    return siglip.patch_siglip_model_for_rpu_all_layers_once(model, projector)

def _finalize(model):
    finalizer = getattr(model, '_rpu_vision_handle_finalizer', None)
    if finalizer is not None and finalizer.alive:
        finalizer()

@pytest.mark.parametrize('num_images', [1, 3])
@pytest.mark.parametrize('w8a16', [False, True])
def test_install_binds_actual_cache_envelope(native_install, monkeypatch, num_images, w8a16):
    (model, projector) = _model()
    monkeypatch.setattr(siglip, '_detect_and_validate_siglip_encoder_w8a16', lambda *args: w8a16)
    try:
        handle = siglip.patch_siglip_model_for_rpu_all_layers_once(model, projector, num_images=num_images)
        assert native_install['envelopes'] == {handle: (256 * num_images, 0)}
        assert model._rpu_cache.kwargs['max_seq_len'] == 256 * num_images
    finally:
        _finalize(model)

def test_envelope_failure_retires_pending_handle_without_publication(native_install):
    (model, projector) = _model()
    original_forward = model.forward
    native_install['fail_envelope'].add(1)
    with pytest.raises(RuntimeError, match='envelope failure'):
        _install(model, projector)
    assert native_install['events'] == [('create', 1), ('set_weights', 1), ('destroy', 1)]
    assert not native_install['alive'] and (not native_install['envelopes'])
    assert not hasattr(model, '_rpu_vision_handle')
    assert model.forward is original_forward

def test_prepare_failure_does_not_create_or_publish_handle(native_install, monkeypatch):
    (model, projector) = _model()
    original_forward = model.forward

    class FailingCache:

        def __init__(self, **kwargs):
            raise RuntimeError('synthetic cache preparation failure')
    monkeypatch.setattr(siglip, 'RPUCache', FailingCache)
    with pytest.raises(RuntimeError, match='cache preparation failure'):
        _install(model, projector)
    assert native_install['events'] == []
    assert not hasattr(model, '_rpu_vision_handle')
    assert model.forward is original_forward

def test_set_weights_failure_cleans_pending_handle_immediately(native_install):
    (model, projector) = _model()
    original_forward = model.forward
    native_install['fail_set'].add(1)
    with pytest.raises(RuntimeError, match='set_weights failure'):
        _install(model, projector)
    assert native_install['events'] == [('create', 1), ('set_weights', 1), ('destroy', 1)]
    assert native_install['alive'] == set()
    assert not hasattr(model, '_rpu_vision_handle')
    assert model.forward is original_forward

def test_failed_replacement_preserves_published_install(native_install):
    (model, projector) = _model()
    assert _install(model, projector) == 1
    old_forward = model.forward
    old_cache = model._rpu_cache
    old_finalizer = model._rpu_vision_handle_finalizer
    native_install['events'].clear()
    native_install['fail_patch'].add(2)
    with pytest.raises(RuntimeError, match='patch failure'):
        _install(model, projector)
    assert native_install['events'] == [('create', 2), ('set_weights', 2), ('set_patch', 2), ('destroy', 2)]
    assert native_install['alive'] == {1}
    assert model._rpu_vision_handle == 1
    assert model._rpu_vision_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_cache is old_cache
    assert model.forward is old_forward
    _finalize(model)

def test_old_destroy_failure_aborts_replacement(native_install):
    from rpu_backend.api import _execution
    from rpu_backend.runtime import _native_retirement
    (model, projector) = _model()
    assert _install(model, projector) == 1
    old_finalizer = model._rpu_vision_handle_finalizer
    native_install['events'].clear()
    native_install['fail_destroy'].add(1)
    with pytest.raises(RuntimeError, match='destroy failure'):
        _install(model, projector)
    assert native_install['events'] == [('create', 2), ('set_weights', 2), ('set_patch', 2), ('destroy', 1)]
    assert native_install['alive'] == {1, 2}
    pending = model._rpu_vision_retirement_state
    assert model._rpu_vision_handle == pending.handle == 2
    assert pending.failed is not None and pending.keepalive and pending.graphs
    assert {resource.handle for resource in _native_retirement._FAILED_RETIREMENTS} == {1, 2}
    assert _execution._UNSAFE_PROCESS_REASON
    before = list(native_install['events'])
    native_install['fail_destroy'].clear()
    with pytest.raises(RuntimeError, match='unsafe|restart|Restart|retirement'):
        _install(model, projector)
    old_finalizer()
    _finalize(model)
    assert native_install['events'] == before

def test_first_install_publication_baseexception_rolls_back(native_install):
    (model, projector) = _model()
    original_forward = model.forward
    model.arm_assignment_failure('_rpu_vision_handle_finalizer')
    with pytest.raises(_PublishInterrupted, match='publication failure'):
        _install(model, projector)
    assert native_install['events'] == [('create', 1), ('set_weights', 1), ('set_patch', 1), ('destroy', 1)]
    assert native_install['alive'] == set()
    assert not hasattr(model, '_rpu_vision_handle')
    assert not hasattr(model, '_rpu_cache')
    assert model.forward is original_forward

def test_replacement_publication_baseexception_preserves_old_install(native_install):
    (model, projector) = _model()
    assert _install(model, projector) == 1
    old_forward = model.forward
    old_cache = model._rpu_cache
    old_graph_cache = model._rpu_siglip_graph_cache
    old_finalizer = model._rpu_vision_handle_finalizer
    native_install['events'].clear()
    model.arm_assignment_failure('forward')
    with pytest.raises(_PublishInterrupted, match='publication failure'):
        _install(model, projector)
    assert native_install['events'] == [('create', 2), ('set_weights', 2), ('set_patch', 2), ('destroy', 2)]
    assert native_install['alive'] == {1}
    assert model._rpu_vision_handle == 1
    assert model._rpu_vision_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_cache is old_cache
    assert model._rpu_siglip_graph_cache is old_graph_cache
    assert model.forward is old_forward
    _finalize(model)

def test_freeze_baseexception_rolls_back_before_old_retirement(native_install, monkeypatch):
    (model, projector) = _model()
    assert _install(model, projector) == 1
    old_forward = model.forward
    old_cache = model._rpu_cache
    old_finalizer = model._rpu_vision_handle_finalizer
    native_install['events'].clear()
    from rpu_backend.graph import lazy_init_guard

    def fail_freeze(_model):
        raise _PublishInterrupted('synthetic freeze failure')
    monkeypatch.setattr(lazy_init_guard, '_verify_lazy_init', fail_freeze)
    with pytest.raises(_PublishInterrupted, match='freeze failure'):
        _install(model, projector)
    assert native_install['events'] == [('create', 2), ('set_weights', 2), ('set_patch', 2), ('destroy', 2)]
    assert native_install['alive'] == {1}
    assert model._rpu_vision_handle == 1
    assert model._rpu_vision_handle_finalizer is old_finalizer
    assert old_finalizer.alive
    assert model._rpu_cache is old_cache
    assert model.forward is old_forward
    _finalize(model)

def test_successful_replacement_retires_old_finalizer(native_install):
    (model, projector) = _model()
    assert _install(model, projector) == 1
    old_finalizer = model._rpu_vision_handle_finalizer
    old_cache = model._rpu_cache
    native_install['events'].clear()
    assert _install(model, projector) == 2
    assert native_install['events'] == [('create', 2), ('set_weights', 2), ('set_patch', 2), ('destroy', 1)]
    assert native_install['alive'] == {2}
    assert model._rpu_vision_handle == 2
    assert model._rpu_vision_handle_finalizer.alive
    assert not old_finalizer.alive
    assert model._rpu_cache is not old_cache
    _finalize(model)

def test_replacement_does_not_double_destroy_dead_finalizer(native_install):
    (model, projector) = _model()
    assert _install(model, projector) == 1
    model._rpu_vision_handle_finalizer()
    native_install['events'].clear()
    assert _install(model, projector) == 2
    assert native_install['events'] == [('create', 2), ('set_weights', 2), ('set_patch', 2)]
    assert native_install['alive'] == {2}
    assert model._rpu_vision_handle == 2
    assert model._rpu_vision_handle_finalizer.alive
    _finalize(model)

@pytest.mark.parametrize('acc32', (False, True))
def test_vision_accumulation_is_bound_before_weights(native_install, acc32):
    (model, projector) = _model()
    model._fmb_execution_component_config = {'vision': {'linear_acc32': acc32}}
    try:
        handle = siglip.patch_siglip_model_for_rpu_all_layers_once(model, projector, linear_acc32=acc32)
        assert native_install['accumulation'] == {handle: acc32}
        assert native_install['events'][:2] == [('create', handle), ('set_weights', handle)]
    finally:
        _finalize(model)

def test_standalone_vision_preserves_implicit_projector_accumulation(native_install):
    (model, projector) = _model()
    try:
        handle = _install(model, projector)
        assert native_install['accumulation'] == {handle: None}
    finally:
        _finalize(model)
