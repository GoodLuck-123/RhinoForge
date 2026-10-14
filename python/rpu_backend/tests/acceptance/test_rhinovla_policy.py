from __future__ import annotations
import sys
import types
from contextlib import contextmanager
from pathlib import Path
import pytest
from rpu_backend.api import RhinoVLAPolicy
from rpu_backend.runtime import control

class _Runtime:

    def __init__(self):
        self.calls = []
        self._rpu_execution = {}
        self._execution_session = _Session()
        self._rhinovla_execution_controller = _Controller(self, self._execution_session)

    def prepare_graphs(self, **request):
        self.calls.append(('prepare', request))
        return {'ready': True}

    def predict(self, **request):
        self.calls.append(('predict', request))
        return request['state'] + 1

class _Session:
    config = {}

    @contextmanager
    def execute(self):
        yield 0

    def register_config_view(self, view):
        view._rpu_execution = self.config

class _Controller:

    def __init__(self, runtime, session):
        self.runtime = runtime
        self.session = session

    def clear_receipts(self):
        pass

    def collect_receipts(self):
        return {}

def test_runtime_facade_delegates_graph_and_prediction_calls():
    runtime = _Runtime()
    policy = RhinoVLAPolicy.from_runtime(runtime)
    assert policy.runtime is runtime
    assert policy.prepare_graphs(state=4) == {'ready': True}
    assert policy.predict(state=4) == 5
    assert policy.predict_action_chunk(state=8) == 9
    assert runtime.calls == [('prepare', {'state': 4}), ('predict', {'state': 4}), ('predict', {'state': 8})]

def test_delivery_factory_target_is_resolved_by_public_api():
    module = types.ModuleType('_customer_rhinovla_runtime')
    seen = []

    def build_runtime(config, *, rpu_execution):
        seen.append((config, rpu_execution))
        return _Runtime()
    module.build_runtime = build_runtime
    sys.modules[module.__name__] = module
    try:
        config = {'model': {'checkpoint': '/model/checkpoint.pt'}}
        policy = RhinoVLAPolicy.from_factory(f'{module.__name__}:build_runtime', config)
    finally:
        sys.modules.pop(module.__name__, None)
    assert isinstance(policy.runtime, _Runtime)
    assert seen == [(config, {})]

def test_pretrained_factory_receives_explicit_model_owned_inputs():
    seen = []

    def build_runtime(path, **kwargs):
        seen.append((path, kwargs))
        return _Runtime()
    config = {'input_profile': 'three-camera'}
    policy = RhinoVLAPolicy.from_pretrained('/models/rhino-vla/model.pt', runtime_factory=build_runtime, config=config, instance_id=2)
    assert isinstance(policy.runtime, _Runtime)
    assert seen == [('/models/rhino-vla/model.pt', {'config': config, 'instance_id': 2, 'rpu_execution': {}})]
    path = Path('/models/rhino-vla/model.pt')
    RhinoVLAPolicy.from_pretrained(path, runtime_factory=build_runtime)
    assert seen[-1] == (path, {'rpu_execution': {}})

@pytest.mark.parametrize('path', [None, 7, b'checkpoint'])
def test_pretrained_factory_rejects_non_path_inputs_before_factory(path):
    called = False

    def build_runtime(*args, **kwargs):
        nonlocal called
        called = True
    with pytest.raises(TypeError, match='must be str or PathLike'):
        RhinoVLAPolicy.from_pretrained(path, runtime_factory=build_runtime)
    assert called is False

def test_pretrained_factory_rejects_empty_path_before_factory():
    with pytest.raises(ValueError, match='must not be empty'):
        RhinoVLAPolicy.from_pretrained('', runtime_factory=lambda path: object())

def test_invalid_runtime_contract_fails_at_construction():
    with pytest.raises(TypeError, match='predict'):
        RhinoVLAPolicy.from_runtime(object())
    with pytest.raises(ValueError, match='module:callable'):
        RhinoVLAPolicy.from_factory('missing_separator', {})

def test_rpu_availability_tracks_native_extension_state(monkeypatch):
    import rpu_backend
    monkeypatch.setattr(control, '_device_nodes_accessible', lambda : True)
    monkeypatch.setattr(control, '_cpp', lambda : types.SimpleNamespace(rpu_runtime_is_available=lambda : True))
    monkeypatch.setattr(rpu_backend, '_cpp_loaded', True)
    assert control.is_available()
    monkeypatch.setattr(rpu_backend, '_cpp_loaded', False)
    assert not control.is_available()
