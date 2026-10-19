"""Explicit device requests must fail before checkpoint loading or swizzling."""
from __future__ import annotations

import errno

import pytest

import rpu_backend
from rpu_backend import _native_loader
from rpu_backend.api import causal_lm
from rpu_backend.api.errors import RPUBackendError
from rpu_backend.tests.acceptance.test_causal_lm_adapter_install_lifecycle import (
    _Outer, _llama_config, _qwen3_config,
)


@pytest.fixture(params=[('Qwen3ForCausalLM', _qwen3_config), ('LlamaForCausalLM', _llama_config)])
def cpu_checkpoint(request, monkeypatch):
    architecture, config_factory = request.param
    config = config_factory()
    config.architectures = [architecture]
    model = _Outer(config)
    loads = []

    def load(*args, **kwargs):
        loads.append(kwargs)
        return model

    monkeypatch.setattr(causal_lm.AutoConfig, 'from_pretrained', lambda *a, **k: config)
    monkeypatch.setattr(causal_lm.AutoModelForCausalLM, 'from_pretrained', load)
    return model, loads


def _deny_mem():
    raise PermissionError(errno.EACCES, 'Permission denied', '/dev/mem')


@pytest.mark.parametrize('device', [None, 'cpu'])
def test_cpu_load_does_not_require_devices(monkeypatch, cpu_checkpoint, device):
    model, loads = cpu_checkpoint
    monkeypatch.setattr(_native_loader, '_check_device_nodes',
                        lambda: pytest.fail('CPU load probed device permissions'))
    monkeypatch.setattr(rpu_backend, '_cpp_loaded', False)
    assert causal_lm.RPUModelForCausalLM.from_pretrained('fixture', device=device) is model
    assert len(loads) == 1 and loads[0]['device_map'] == 'cpu'
    assert not getattr(model, '_rpu_swizzle_started', False)


@pytest.mark.parametrize('failure', ['permissions', 'native_missing'])
def test_rpu_load_fails_before_checkpoint_materialization(monkeypatch, cpu_checkpoint, failure):
    model, loads = cpu_checkpoint
    monkeypatch.setattr(_native_loader, '_check_device_nodes',
                        _deny_mem if failure == 'permissions' else lambda: None)
    monkeypatch.setattr(rpu_backend, '_cpp_loaded', False)
    match = r'Errno 13.*?/dev/mem' if failure == 'permissions' else 'native backend is not loaded'
    with pytest.raises(RPUBackendError, match=match):
        causal_lm.RPUModelForCausalLM.from_pretrained('fixture', device='rpu')
    assert loads == []
    assert not getattr(model, '_rpu_swizzle_started', False)
    assert causal_lm._LIVE_REF is None


def test_two_step_load_rechecks_access_before_claim_or_weight_mutation(monkeypatch, cpu_checkpoint):
    model, loads = cpu_checkpoint
    monkeypatch.setattr(rpu_backend, '_cpp_loaded', True)
    monkeypatch.setattr(_native_loader, '_check_device_nodes', lambda: None)
    loaded = causal_lm.RPUModelForCausalLM.from_pretrained('fixture')
    assert loaded is model and len(loads) == 1
    monkeypatch.setattr(_native_loader, '_check_device_nodes', _deny_mem)

    with pytest.raises(RPUBackendError, match=r'Errno 13.*?/dev/mem'):
        loaded.to('rpu')
    assert not getattr(model, '_rpu_swizzle_started', False)
    assert causal_lm._LIVE_REF is None
    assert not any(isinstance(event, tuple) and event[0] == 'move' for event in model._events)
