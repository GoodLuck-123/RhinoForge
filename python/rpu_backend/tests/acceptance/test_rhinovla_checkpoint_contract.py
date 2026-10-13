from __future__ import annotations
import sys
import types
import warnings
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import pytest
import torch
import rpu_backend.adapters.rhinovla.checkpoint as checkpoint_loader

class _FakeConfig:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class _FakeStateful:

    def __init__(self, *args, **kwargs):
        self.loaded_state = None
        self.eval_called = False

    def load_state_dict(self, state, *, strict):
        assert strict is True
        self.loaded_state = state

    def eval(self):
        self.eval_called = True
        return self

def _fake_model_module():
    return types.SimpleNamespace(RhinoActionExpertConfig=_FakeConfig, RhinoActionExpert=_FakeStateful, RhinoActionIO=_FakeStateful)

def _valid_state(*, rank: int=2, instances: int=2) -> dict:
    width = 4
    head_dim = 2
    action_dim = 3
    state_dim = 5
    mlp_dim = 6
    parent = 'action_expert.layers.0.mlp.gate_proj'
    lora_a = torch.zeros(instances, width, rank)
    lora_b = torch.zeros(instances, rank, mlp_dim)
    lora_a[1].fill_(2)
    lora_b[1].fill_(3)
    return {'action_expert.layers.0.self_attn.q_proj.weight': torch.zeros(2 * head_dim, width), 'action_expert.layers.0.self_attn.k_proj.weight': torch.zeros(head_dim, width), 'action_expert.layers.0.self_attn.q_norm.weight': torch.ones(head_dim), f'{parent}.base.weight': torch.ones(mlp_dim, width), f'{parent}.lora_a': lora_a, f'{parent}.lora_b': lora_b, 'action_io.action_in_proj.weight': torch.zeros(width, action_dim), 'action_io.state_proj.weight': torch.zeros(width, state_dim), 'state_mask_proj.weight': torch.zeros(width, state_dim), 'state_mask_proj.bias': torch.zeros(width), 'action_mask_proj.weight': torch.zeros(width, action_dim), 'action_mask_proj.bias': torch.zeros(width)}

def _loader_paths(tmp_path):
    checkpoint = tmp_path / 'model.pt'
    checkpoint.touch()
    model_root = tmp_path / 'model_root'
    source = model_root / 'rhinovla/model/modules/action_model/rhino_action_expert.py'
    source.parent.mkdir(parents=True)
    source.write_text('# reviewed test source\n', encoding='utf-8')
    return (checkpoint, model_root)

def test_valid_bundle_merges_declared_instance_lora(monkeypatch, tmp_path):
    (checkpoint, model_root) = _loader_paths(tmp_path)
    state = _valid_state()
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: state)
    monkeypatch.setattr(checkpoint_loader, '_load_expert_module', lambda root: _fake_model_module())
    monkeypatch.setattr(checkpoint_loader, '_create_rotary', lambda **kwargs: object())
    with pytest.warns(UserWarning, match='executes Python source'):
        (expert, action_io, mask_proj, config) = checkpoint_loader.load_action_bundle(checkpoint, model_root=model_root, action_horizon=30, instance_id=1, lora_alpha=4, lora_rank=2, trust_model_code=True)
    merged = expert.loaded_state['layers.0.mlp.gate_proj.weight']
    expected = torch.ones(6, 4) + 24
    assert torch.equal(merged, expected)
    assert config.action_horizon == 30
    assert config.depth == 1
    assert config.action_dim == 3
    assert config.state_dim == 5
    assert action_io.eval_called
    assert not mask_proj.training

@pytest.mark.parametrize(('override', 'match'), [({'instance_id': True}, 'instance_id.*integer'), ({'instance_id': 0.5}, 'instance_id.*integer'), ({'instance_id': -1}, 'instance_id.*>= 0'), ({'action_horizon': True}, 'action_horizon.*integer'), ({'action_horizon': 30.5}, 'action_horizon.*integer'), ({'lora_rank': '32'}, 'lora_rank.*integer'), ({'lora_rank': 0}, 'lora_rank.*>= 1'), ({'lora_alpha': True}, 'lora_alpha.*finite'), ({'lora_alpha': float('nan')}, 'lora_alpha.*finite'), ({'trust_model_code': 'false'}, 'trust_model_code must be bool')])
def test_profile_arguments_reject_coercion_before_io(override, match):
    kwargs = {'model_root': 'unused', 'action_horizon': 30, 'instance_id': 0, 'lora_alpha': 32, 'lora_rank': 32, 'trust_model_code': True}
    kwargs.update(override)
    with pytest.raises((TypeError, ValueError), match=match):
        checkpoint_loader.load_action_bundle('unused', **kwargs)

def test_model_source_requires_explicit_boolean_opt_in(monkeypatch):
    load_called = False

    def forbidden_load(*args, **kwargs):
        nonlocal load_called
        load_called = True
    monkeypatch.setattr(torch, 'load', forbidden_load)
    with pytest.raises(PermissionError, match='trust_model_code=True'):
        checkpoint_loader.load_action_bundle('unused', model_root='unused', action_horizon=30)
    assert load_called is False

@pytest.mark.parametrize('failure', ['rank', 'instance', 'pair', 'layers', 'mask'])
def test_checkpoint_structure_fails_before_model_code(monkeypatch, tmp_path, failure):
    (checkpoint, model_root) = _loader_paths(tmp_path)
    state = _valid_state()
    expected = ''
    if failure == 'rank':
        state['action_expert.layers.0.mlp.gate_proj.lora_a'] = torch.zeros(2, 4, 3)
        expected = 'LoRA shapes'
    elif failure == 'instance':
        expected = 'instance_id.*outside'
    elif failure == 'pair':
        del state['action_expert.layers.0.mlp.gate_proj.lora_b']
        expected = 'matching lora_a and lora_b'
    elif failure == 'layers':
        state['action_expert.layers.2.extra.weight'] = torch.ones(1)
        expected = 'contiguous'
    else:
        state['state_mask_proj.weight'] = torch.zeros(4, 4)
        expected = 'state_mask_proj.weight.*shape'
    model_code_called = False

    def forbidden_model_code(root):
        nonlocal model_code_called
        model_code_called = True
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: state)
    monkeypatch.setattr(checkpoint_loader, '_load_expert_module', forbidden_model_code)
    call_instance = 2 if failure == 'instance' else 0
    with warnings.catch_warnings(record=True) as caught:
        with pytest.raises((ValueError, IndexError), match=expected):
            checkpoint_loader.load_action_bundle(checkpoint, model_root=model_root, action_horizon=30, instance_id=call_instance, lora_rank=2, trust_model_code=True)
    assert caught == []
    assert model_code_called is False

def test_model_module_cache_key_tracks_source_content(tmp_path):
    (_, model_root) = _loader_paths(tmp_path)
    source = checkpoint_loader._expert_source(model_root)
    source.write_text('VALUE = 1\n', encoding='utf-8')
    first = checkpoint_loader._load_expert_module(model_root)
    source.write_text('VALUE = 2\n', encoding='utf-8')
    second = checkpoint_loader._load_expert_module(model_root)
    try:
        assert first.VALUE == 1
        assert second.VALUE == 2
        assert first is not second
    finally:
        sys.modules.pop(first.__name__, None)
        sys.modules.pop(second.__name__, None)

def test_concurrent_model_module_load_never_returns_partial_module(tmp_path):
    (_, model_root) = _loader_paths(tmp_path)
    source = checkpoint_loader._expert_source(model_root)
    source.write_text('import time\ntime.sleep(0.05)\nVALUE = 7\n', encoding='utf-8')
    barrier = Barrier(2)

    def load():
        barrier.wait()
        return checkpoint_loader._load_expert_module(model_root)
    with ThreadPoolExecutor(max_workers=2) as pool:
        (first, second) = pool.map(lambda _: load(), range(2))
    try:
        assert first is second
        assert first.VALUE == 7
    finally:
        sys.modules.pop(first.__name__, None)
