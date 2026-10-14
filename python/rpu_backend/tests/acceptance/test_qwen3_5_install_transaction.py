from __future__ import annotations
from types import SimpleNamespace
from rpu_backend.adapters import qwen3_5

def test_failed_text_cleanup_bypasses_rejecting_attribute_hooks(monkeypatch):

    class RejectingOwner:

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
            events.append('clear')

        def cache_invariant_ok(self):
            return True

    class Finalizer:
        alive = True

        def detach(self):
            self.alive = False
            events.append('detach')
            return True
    import torch
    monkeypatch.setattr(torch.ops.rpu, 'qwen3_5_destroy', lambda _h: events.append('destroy'), raising=False)
    runtime_state = SimpleNamespace(handle=31, graph_cache=Cache(), handle_finalizer=Finalizer())
    inner = RejectingOwner()
    vars(inner).update({'_rpu_qwen3_5': runtime_state, '_reject_runtime_attrs': True})
    model = RejectingOwner()
    original_forward = lambda : 'original-forward'
    vars(model).update({'forward': lambda : 'patched-forward', '_lm_head_w_rpu_keepalive': object(), '_reject_runtime_attrs': True})
    adapter = SimpleNamespace(_handle=31, _graph_cache=runtime_state.graph_cache, _rpu_is_ready=True)
    qwen3_5._cleanup_failed_text_install(adapter, model, inner, True, original_forward)
    assert events == ['clear', 'destroy', 'detach']
    assert not hasattr(inner, '_rpu_qwen3_5')
    assert not hasattr(model, '_lm_head_w_rpu_keepalive')
    assert model.forward is original_forward
    assert adapter._handle is None
    assert adapter._graph_cache is None
    assert adapter._rpu_is_ready is False
