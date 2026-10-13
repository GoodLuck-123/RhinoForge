from __future__ import annotations
from types import MethodType, SimpleNamespace
import pytest
import torch
from rpu_backend.adapters import paligemma
from rpu_backend.adapters.paligemma import gemma2
from rpu_backend.api.errors import RPUBackendError

class _InstallInterrupted(BaseException):
    pass

class _Finalizer:

    def __init__(self, events, *, alive=True):
        self._events = events
        self.alive = alive

    def __call__(self):
        if self.alive:
            self._events.append(('finalize', 17))
            self.alive = False
        return True

@pytest.mark.parametrize('failure_type', [RuntimeError, _InstallInterrupted])
def test_failed_weight_configuration_destroys_raw_handle(monkeypatch, failure_type):
    events = []
    monkeypatch.setattr(torch.ops.rpu, 'gemma2_create', lambda : events.append(('create', 17)) or 17, raising=False)

    def fail_set_weights(handle, *args):
        events.append(('set_weights', handle))
        raise failure_type('synthetic configuration failure')
    monkeypatch.setattr(torch.ops.rpu, 'gemma2_set_weights', fail_set_weights, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'gemma2_destroy', lambda handle: events.append(('destroy', handle)), raising=False)
    with pytest.raises(failure_type, match='configuration failure'):
        gemma2._configure_gemma2_handle((object(),), 64)
    assert events == [('create', 17), ('set_weights', 17), ('destroy', 17)]

def test_successful_weight_configuration_leaves_handle_owned(monkeypatch):
    events = []
    monkeypatch.setattr(torch.ops.rpu, 'gemma2_create', lambda : 23, raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'gemma2_set_weights', lambda handle, *args: events.append(('set_weights', handle)), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'gemma2_destroy', lambda handle: events.append(('destroy', handle)), raising=False)
    monkeypatch.setattr(torch.ops.rpu, 'gemma2_set_chunk_envelope', lambda handle, capacity, chunk: events.append(('envelope', handle, capacity, chunk)), raising=False)
    assert gemma2._configure_gemma2_handle((object(),), 64) == 23
    assert events == [('set_weights', 23), ('envelope', 23, 64, 0)]

def test_composite_cleanup_uses_finalizer_and_restores_instance_forward(monkeypatch):
    events = []
    direct_destroys = []
    original_forward = lambda : 'original'
    decoder = SimpleNamespace(forward=lambda : 'patched', _rpu_gemma2_handle=17, _rpu_gemma2_handle_finalizer=_Finalizer(events), _rpu_cache=object(), _rpu_last_execution_plan={'generation': 7}, _rpu_attn_tp=4, _rpu_lazy_init_checked=True, _rpu_required_attrs=('_rpu_cache',), _rpu_effective_num_kv_heads=4, _rpu_gemma2_had_instance_forward=True, _rpu_gemma2_original_instance_forward=original_forward, _rpu_gemma2_install_started=True, _rpu_gemma2_install_ready=True)
    monkeypatch.setattr(torch.ops.rpu, 'gemma2_destroy', lambda handle: direct_destroys.append(handle), raising=False)
    paligemma._cleanup_gemma2_handle(decoder)
    assert events == [('finalize', 17)]
    assert direct_destroys == []
    assert decoder.forward is original_forward
    assert not hasattr(decoder, '_rpu_last_execution_plan')
    assert not hasattr(decoder, '_rpu_attn_tp')
    assert decoder._rpu_gemma2_install_started is True
    for attr in gemma2._INSTALL_STATE_ATTRS:
        assert not hasattr(decoder, attr)

def test_composite_cleanup_does_not_double_destroy_dead_finalizer(monkeypatch):
    direct_destroys = []
    decoder = SimpleNamespace(_rpu_gemma2_handle=29, _rpu_gemma2_handle_finalizer=_Finalizer([], alive=False))
    monkeypatch.setattr(torch.ops.rpu, 'gemma2_destroy', lambda handle: direct_destroys.append(handle), raising=False)
    paligemma._cleanup_gemma2_handle(decoder)
    assert direct_destroys == []
    assert not hasattr(decoder, '_rpu_gemma2_handle')
    assert not hasattr(decoder, '_rpu_gemma2_handle_finalizer')

def test_cleanup_restores_class_forward_by_removing_instance_override():

    class Decoder:

        def forward(self):
            return 'class-forward'
    decoder = Decoder()
    decoder.forward = lambda : 'patched'
    decoder._rpu_gemma2_had_instance_forward = False
    decoder._rpu_gemma2_original_instance_forward = None
    paligemma._cleanup_gemma2_handle(decoder)
    assert 'forward' not in vars(decoder)
    assert decoder.forward() == 'class-forward'

def test_restore_helpers_bypass_rejecting_attribute_hooks():

    class Decoder:

        def forward(self):
            return 'class-forward'

        def __setattr__(self, name, value):
            if self.__dict__.get('_reject_runtime_attrs', False):
                raise AssertionError('rollback re-entered __setattr__')
            super().__setattr__(name, value)

        def __delattr__(self, name):
            if self.__dict__.get('_reject_runtime_attrs', False):
                raise AssertionError('rollback re-entered __delattr__')
            super().__delattr__(name)
    decoder = Decoder()
    original_forward = lambda : 'original-forward'
    decoder_vars = vars(decoder)
    decoder_vars.update({'forward': lambda : 'patched-forward', '_rpu_gemma2_handle': 41, '_rpu_cache': object(), '_reject_runtime_attrs': True})
    gemma2._restore_instance_forward(decoder, had_instance_forward=True, original_forward=original_forward)
    gemma2._restore_attrs(decoder, {'_rpu_gemma2_handle': 17, '_rpu_cache': gemma2._MISSING})
    assert decoder.forward is original_forward
    assert decoder._rpu_gemma2_handle == 17
    assert not hasattr(decoder, '_rpu_cache')
    gemma2._restore_instance_forward(decoder, had_instance_forward=False, original_forward=None)
    assert 'forward' not in vars(decoder)
    assert decoder.forward() == 'class-forward'

def test_composite_cleanup_bypasses_rejecting_attribute_hooks():

    class RejectingOwner:

        def forward(self):
            return 'class-forward'

        def __setattr__(self, name, value):
            if self.__dict__.get('_reject_runtime_attrs', False):
                raise AssertionError('cleanup re-entered __setattr__')
            super().__setattr__(name, value)

        def __delattr__(self, name):
            if self.__dict__.get('_reject_runtime_attrs', False):
                raise AssertionError('cleanup re-entered __delattr__')
            super().__delattr__(name)
    events = []

    class Cache:

        def clear(self):
            events.append(('clear', 'siglip'))
    vision = RejectingOwner()
    vars(vision).update({'forward': lambda : 'patched-vision-forward', '_rpu_cache': object(), '_rpu_siglip_graph_cache': Cache(), '_rpu_lazy_init_checked': True, '_rpu_required_attrs': ('_rpu_cache',), '_rpu_vision_handle': 17, '_rpu_vision_handle_finalizer': _Finalizer(events), '_reject_runtime_attrs': True})
    paligemma._cleanup_siglip_handle(vision, had_instance_forward=False, original_forward=None)
    assert events == [('clear', 'siglip'), ('finalize', 17)]
    assert 'forward' not in vars(vision)
    assert vision.forward() == 'class-forward'
    for attr in ('_rpu_cache', '_rpu_siglip_graph_cache', '_rpu_lazy_init_checked', '_rpu_required_attrs', '_rpu_vision_handle', '_rpu_vision_handle_finalizer'):
        assert not hasattr(vision, attr)
    decoder = RejectingOwner()
    original_forward = lambda : 'original-decoder-forward'
    vars(decoder).update({'forward': lambda : 'patched-decoder-forward', '_rpu_gemma2_handle': 17, '_rpu_gemma2_handle_finalizer': _Finalizer([]), '_rpu_gemma2_had_instance_forward': True, '_rpu_gemma2_original_instance_forward': original_forward, '_rpu_gemma2_install_ready': True, '_reject_runtime_attrs': True})
    paligemma._cleanup_gemma2_handle(decoder)
    assert decoder.forward is original_forward
    for attr in gemma2._INSTALL_STATE_ATTRS:
        assert not hasattr(decoder, attr)

def test_complete_ready_install_is_idempotent_without_transformer_reentry():

    def rpu_gemma2_model_forward(self):
        return 'patched'
    decoder = SimpleNamespace()
    decoder.forward = MethodType(rpu_gemma2_model_forward, decoder)
    decoder._rpu_gemma2_handle = 31
    decoder._rpu_gemma2_handle_finalizer = _Finalizer([])
    decoder._rpu_cache = object()
    decoder._rpu_gemma2_graph_cache = object()
    decoder._execution_session = object()
    decoder._rpu_lazy_init_checked = True
    decoder._rpu_gemma2_install_ready = True
    assert gemma2.patch_gemma2_model_for_rpu_all_layers_once(decoder) == 31

@pytest.mark.parametrize('state', ['broken_ready', 'partial_install'])
def test_incomplete_install_requires_model_reload(state):
    if state == 'broken_ready':
        decoder = SimpleNamespace(forward=lambda : 'patched', _rpu_gemma2_handle=37, _rpu_gemma2_handle_finalizer=_Finalizer([], alive=False), _rpu_cache=object(), _rpu_lazy_init_checked=True, _rpu_gemma2_install_ready=True)
        expected = 'ready marker is incomplete'
    else:
        decoder = SimpleNamespace(_rpu_gemma2_install_started=True)
        expected = 'previously started'
    with pytest.raises(RPUBackendError, match=expected):
        gemma2.patch_gemma2_model_for_rpu_all_layers_once(decoder)
