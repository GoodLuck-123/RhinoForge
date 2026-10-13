from __future__ import annotations
from pathlib import Path
import pytest
import torch
from rpu_backend.runtime.device import extract_to_device_target, is_rpu_device_target
from rpu_backend.api.errors import RPUBackendError
from rpu_backend.api.policy import Pi05Policy
_ADAPTER_ROUTERS = ('python/rpu_backend/adapters/qwen3.py', 'python/rpu_backend/adapters/llama.py', 'python/rpu_backend/adapters/qwen3_vl/__init__.py', 'python/rpu_backend/adapters/qwen3_5/__init__.py', 'python/rpu_backend/adapters/dinov3.py', 'python/rpu_backend/adapters/gemma4/__init__.py', 'python/rpu_backend/adapters/_template/minimal_causal_lm.py')

@pytest.mark.parametrize('target,expected', [('rpu', True), ('rpu:0', True), (torch.device('rpu'), True), ('privateuseone', True), ('cpu', False), ('cuda', False), (torch.float16, False)])
def test_rpu_device_target_classification(target, expected):
    assert is_rpu_device_target(target, entry_point='test.to') is expected

@pytest.mark.parametrize('target', ['rpu:1', torch.device('rpu:1'), 'rpu-ish', 0, True])
def test_ambiguous_or_unsupported_rpu_targets_fail(target):
    with pytest.raises(ValueError, match='rpu|device|integer'):
        is_rpu_device_target(target, entry_point='test.to')

def test_to_target_extraction_distinguishes_dtype_and_integer_device():
    assert extract_to_device_target(('rpu',), {}) == 'rpu'
    assert extract_to_device_target((0,), {}) == 0
    assert extract_to_device_target((torch.float16,), {}) is None
    assert extract_to_device_target((), {'device': 'rpu:0'}) == 'rpu:0'

def test_every_builtin_adapter_router_uses_structured_device_parser():
    for path in _ADAPTER_ROUTERS:
        source = Path(path).read_text(encoding='utf-8')
        assert 'startswith("rpu")' not in source, path
        assert 'is_rpu_device_target(' in source, path
        assert 'extract_to_device_target(' in source, path

class _FakePiAdapter:

    def __init__(self):
        self.calls = 0

    def to_rpu(self):
        self.calls += 1

class _FakeLeRobotPolicy:

    def __init__(self):
        self.targets = []

    def to(self, target):
        self.targets.append(target)
        return self

def _fake_pi_policy() -> Pi05Policy:
    policy = object.__new__(Pi05Policy)
    policy._adapter = _FakePiAdapter()
    policy._lerobot_policy = _FakeLeRobotPolicy()
    policy._rpu_ready = False
    return policy

def test_pi05_policy_routes_only_valid_rpu_zero_to_adapter():
    policy = _fake_pi_policy()
    assert policy.to('cpu') is policy
    assert policy._lerobot_policy.targets == ['cpu']
    assert policy.to('rpu:0') is policy
    assert policy._adapter.calls == 1
    assert policy._rpu_ready is True
    assert policy.to(torch.device('rpu')) is policy
    assert policy._adapter.calls == 2

@pytest.mark.parametrize('target', ['rpu:1', 'rpu-anything', 0])
def test_pi05_invalid_rpu_target_does_not_reach_policy_or_adapter(target):
    policy = _fake_pi_policy()
    with pytest.raises(ValueError):
        policy.to(target)
    assert policy._adapter.calls == 0
    assert policy._lerobot_policy.targets == []

def test_pi05_cannot_move_away_after_irreversible_rpu_install():
    policy = _fake_pi_policy()
    policy.to('rpu')
    with pytest.raises(RPUBackendError, match='irreversible'):
        policy.to('cpu')
    assert policy._lerobot_policy.targets == []
