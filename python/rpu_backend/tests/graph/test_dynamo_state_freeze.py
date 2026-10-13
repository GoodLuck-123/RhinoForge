from __future__ import annotations
import pytest
import torch
import torch.nn as nn
import rpu_backend

def test_t1_preflight_catches_missing_required_attr():
    from rpu_backend.graph.lazy_init_guard import _verify_lazy_init, DynamoUnsafeLazyStateError
    fake = nn.Module()
    fake._rpu_lazy_init_checked = True
    fake._rpu_required_attrs = ('_rpu_cache',)
    with pytest.raises(DynamoUnsafeLazyStateError) as exc_info:
        _verify_lazy_init(fake)
    assert '_rpu_cache' in str(exc_info.value)
    assert 'Module' in str(exc_info.value)

def test_t4_no_marker_module_not_checked():
    from rpu_backend.graph.lazy_init_guard import _verify_lazy_init
    vanilla = nn.Linear(4, 8)
    assert not hasattr(vanilla, '_rpu_lazy_init_checked')
    _verify_lazy_init(vanilla)

def test_t1b_top_level_api_export():
    import rpu_backend
    assert hasattr(rpu_backend, 'freeze_for_dynamo'), 'rpu_backend.freeze_for_dynamo missing from top-level export'
    assert hasattr(rpu_backend, 'DynamoUnsafeLazyStateError'), 'rpu_backend.DynamoUnsafeLazyStateError missing from top-level export'
    assert callable(rpu_backend.freeze_for_dynamo)
    assert issubclass(rpu_backend.DynamoUnsafeLazyStateError, RuntimeError)

def test_t1c_best_effort_no_modules_no_raise():
    from rpu_backend.graph.lazy_init_guard import _verify_lazy_init_best_effort

    class _FakeGM:
        _modules = {'sub_0': nn.Linear(4, 8)}
    _verify_lazy_init_best_effort(_FakeGM(), example_inputs=[])

def test_t1d_best_effort_catches_unsafe_module():
    from rpu_backend.graph.lazy_init_guard import _verify_lazy_init_best_effort, DynamoUnsafeLazyStateError
    unsafe = nn.Module()
    unsafe._rpu_lazy_init_checked = True
    unsafe._rpu_required_attrs = ('_rpu_cache',)

    class _FakeGM:
        _modules = {'sub_0': unsafe}
    with pytest.raises(DynamoUnsafeLazyStateError):
        _verify_lazy_init_best_effort(_FakeGM(), example_inputs=[])

def test_t3_recompile_sentinel_fires_on_lazy_attr_guard(caplog, monkeypatch):
    import logging
    import rpu_backend

    class _LazyAttrModule(nn.Module):

        def forward(self, x):
            if not hasattr(self, '_flag'):
                self._flag = True
            return x + 1
    mod = _LazyAttrModule()
    from types import SimpleNamespace
    monkeypatch.setattr(torch.rpu, 'GraphCache', lambda: SimpleNamespace(_impl=object()), raising=False)
    from rpu_backend.graph.dynamo_backend import RpuDynamoBackend
    fresh = RpuDynamoBackend(max_entries=4, debug=False)
    graph = SimpleNamespace(begin=lambda signature: None, state=lambda: None,
                            end=lambda: None, abort=lambda: None)
    fresh._planner = SimpleNamespace(make_plan=lambda *args, **kwargs: SimpleNamespace(signature=SimpleNamespace(_impl=object())))
    fresh._registry = SimpleNamespace(admit_for_call=lambda plan: graph,
                                     touch_after_call=lambda *args, **kwargs: None)
    compiled = torch.compile(mod, backend=fresh)
    x = torch.randn(2, 3)
    monkeypatch.setattr(logging.getLogger('rpu_backend'), 'propagate', True)
    with caplog.at_level(logging.WARNING, logger='rpu_backend.graph.dynamo_backend'):
        _ = compiled(x)
        _ = compiled(x)
    msgs = [r.message for r in caplog.records]
    has_warn = any(('recompile detected' in m for m in msgs))
    assert has_warn, f"Expected 'recompile detected' warning,got messages: {msgs}"

def test_t5_load_time_default_setter_unaffected():
    from rpu_backend.graph.lazy_init_guard import _verify_lazy_init
    model = nn.Module()
    if not hasattr(model, '_rpu_chunk_size'):
        model._rpu_chunk_size = 0
    _verify_lazy_init(model)
if __name__ == '__main__':
    test_t1_preflight_catches_missing_required_attr()
    test_t4_no_marker_module_not_checked()
    test_t1b_top_level_api_export()
    print('T1 + T4 + smoke PASS')
