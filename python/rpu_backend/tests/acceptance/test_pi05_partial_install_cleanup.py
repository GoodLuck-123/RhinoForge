from __future__ import annotations
from types import SimpleNamespace
from rpu_backend.adapters import pi05

class _Finalizer:

    def __init__(self, events):
        self.alive = True
        self._events = events

    def __call__(self):
        self._events.append('finalize')
        self.alive = False

class _Cache:

    def __init__(self):
        self.cleared = False

    def clear(self):
        self.cleared = True

def test_component_cleanup_prefers_tracked_finalizer_and_deletes_state():
    events = []
    finalizer = _Finalizer(events)
    owner = SimpleNamespace(_handle=17, _handle_finalizer=finalizer)
    pi05._cleanup_component_handle(owner, handle_attr='_handle', finalizer_attr='_handle_finalizer', destroy=lambda handle: events.append(('destroy', handle)))
    assert events == ['finalize']
    assert not finalizer.alive
    assert not hasattr(owner, '_handle')
    assert not hasattr(owner, '_handle_finalizer')

def test_component_cleanup_falls_back_to_direct_destroy_without_finalizer():
    events = []
    owner = SimpleNamespace(_handle=23)
    pi05._cleanup_component_handle(owner, handle_attr='_handle', finalizer_attr='_handle_finalizer', destroy=lambda handle: events.append(('destroy', handle)))
    assert events == [('destroy', 23)]
    assert not hasattr(owner, '_handle')

def test_component_cleanup_does_not_double_destroy_dead_finalizer():
    events = []
    finalizer = _Finalizer(events)
    finalizer.alive = False
    owner = SimpleNamespace(_handle=29, _handle_finalizer=finalizer)
    pi05._cleanup_component_handle(owner, handle_attr='_handle', finalizer_attr='_handle_finalizer', destroy=lambda handle: events.append(('destroy', handle)))
    assert events == []
    assert not hasattr(owner, '_handle')
    assert not hasattr(owner, '_handle_finalizer')

def test_component_graph_cache_is_cleared_before_attribute_removal():
    cache = _Cache()
    owner = SimpleNamespace(_graph_cache=cache)
    pi05._clear_component_graph_cache(owner, '_graph_cache')
    assert cache.cleared
    assert not hasattr(owner, '_graph_cache')

def test_pre_swizzle_failure_releases_process_owner(monkeypatch):
    owner = object()
    released = []
    monkeypatch.setattr(pi05, '_release_live_instance', lambda value: released.append(value))
    pi05._release_after_failed_install(owner, pre_swizzle=True)
    assert released == [owner]

def test_post_swizzle_failure_keeps_process_owner_until_gc(monkeypatch):
    owner = object()
    released = []
    monkeypatch.setattr(pi05, '_release_live_instance', lambda value: released.append(value))
    pi05._release_after_failed_install(owner, pre_swizzle=False)
    assert released == []
