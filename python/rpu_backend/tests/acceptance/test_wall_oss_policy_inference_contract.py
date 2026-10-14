from __future__ import annotations
import ast
import math
from pathlib import Path
import pytest
import torch
from rpu_backend.api import WallOssPolicy
from rpu_backend.api.errors import RPUBackendError

class _FakeVLA:
    action_dim = 4

    def __init__(self, output=None, action_dim=4):
        self._closed = False
        self.action_dim = action_dim
        self.output = {'action': torch.zeros(1, 2, self.action_dim)} if output is None else output
        self.predict_calls = []
        self.prepare_calls = []

    def _runtime_complete(self):
        return not self._closed

    def close(self):
        self._closed = True

    def predict(self, *args, **kwargs):
        self.predict_calls.append((args, kwargs))
        return self.output

    def prepare_graphs(self, **kwargs):
        self.prepare_calls.append(kwargs)
        return {'prepared': True, 'prefix_length_range': kwargs['prefix_length_range']}

def _ready_policy(*, output=None) -> tuple[WallOssPolicy, _FakeVLA]:
    policy = WallOssPolicy.from_checkpoint('unused/checkpoint', camera_names=('face_view',), action_horizon=2)
    vla = _FakeVLA(output=output)
    policy._vla = vla
    policy._rpu_build_started = True
    policy._rpu_ready = True
    return (policy, vla)

def _ready_rtc_policy() -> tuple[WallOssPolicy, _FakeVLA]:
    policy = WallOssPolicy.from_checkpoint('unused/checkpoint', rtc=True, dataset_key='ex_normal', camera_names=('face_view', 'left_wrist_view', 'right_wrist_view'), delta_action=True, state_bins=512, action_horizon=20, num_steps=6)
    vla = _FakeVLA(output={'action': torch.zeros(1, 20, 26)}, action_dim=26)
    policy._vla = vla
    policy._rpu_build_started = True
    policy._rpu_ready = True
    return (policy, vla)

def _valid_infer_kwargs() -> dict:
    return {'images': [object()], 'instruction': 'pick up the cube', 'proprioception': [0.1, 0.2, 0.3], 'state_mask': [1, 0, 1], 'noise': torch.zeros(2, 4), 'dof_mask': torch.ones(1, 2, 4), 'num_steps': 3, 'noise_seed': 7}

def test_infer_normalizes_valid_inputs_and_output_shape():
    (policy, vla) = _ready_policy()
    result = policy.infer(**_valid_infer_kwargs())
    assert tuple(result.actions.shape) == (2, 4)
    assert result.actions.device.type == 'cpu'
    assert result.extra['num_steps'] == 3
    assert result.extra['noise_seed'] == 7
    (args, kwargs) = vla.predict_calls[0]
    assert len(args[0]) == 1
    assert args[1] == 'pick up the cube'
    assert tuple(args[2].shape) == (1, 1, 3)
    assert tuple(args[3].shape) == (1, 1, 3)
    assert tuple(args[4].shape) == (1, 2, 4)
    assert tuple(args[5].shape) == (1, 2, 4)
    assert kwargs == {'num_steps': 3}

def test_infer_forwards_valid_model_space_rtc_prefix():
    (policy, vla) = _ready_rtc_policy()
    result = policy.infer(images=[object(), object(), object()], instruction='sort parcel', proprioception=torch.zeros(26), noise=torch.zeros(20, 26), rtc_prefix_actions=torch.arange(78, dtype=torch.float32).reshape(3, 26), inference_delay=2)
    assert result.extra['inference_delay'] == 2
    (_, kwargs) = vla.predict_calls[0]
    assert kwargs['num_steps'] == 6
    assert kwargs['use_rtc'] is True
    assert kwargs['inference_delay'] == 2
    assert tuple(kwargs['rtc_prefix_actions'].shape) == (1, 3, 26)

def test_infer_forwards_explicit_rtc_d0_without_a_prefix():
    (policy, vla) = _ready_rtc_policy()
    result = policy.infer(images=[object(), object(), object()], instruction='sort parcel', proprioception=torch.zeros(26), noise=torch.zeros(20, 26), use_rtc=True, inference_delay=0)
    assert result.extra['use_rtc'] is True
    assert result.extra['inference_delay'] == 0
    (_, kwargs) = vla.predict_calls[0]
    assert kwargs == {'num_steps': 6, 'use_rtc': True}

def test_rtc_rejects_non_six_step_infer_and_prepare_before_runtime():
    (policy, vla) = _ready_rtc_policy()
    common = {'images': [object(), object(), object()], 'instruction': 'sort parcel', 'proprioception': torch.zeros(26)}
    with pytest.raises(ValueError, match='requires num_steps=6'):
        policy.infer(**common, num_steps=5)
    with pytest.raises(ValueError, match='requires num_steps=6'):
        policy.prepare_graphs(**common, prefix_length_range=(639, 639), num_steps=5)
    assert vla.predict_calls == []
    assert vla.prepare_calls == []

@pytest.mark.parametrize('prefix,delay,match', [(None, 1, 'required'), (torch.zeros(1, 26), 0, 'requires inference_delay'), (torch.zeros(1, 26), 11, 'must be <= 10'), (torch.zeros(1, 3), 1, 'action_dim=26')])
def test_infer_rejects_invalid_rtc_prefix_before_runtime(prefix, delay, match):
    policy = WallOssPolicy.from_checkpoint('unused/checkpoint', rtc=True, dataset_key='ex_normal', camera_names=('face_view', 'left_wrist_view', 'right_wrist_view'), delta_action=True, state_bins=512, action_horizon=20, num_steps=6)
    vla = _FakeVLA(output={'action': torch.zeros(1, 20, 26)}, action_dim=26)
    policy._vla = vla
    policy._rpu_build_started = True
    policy._rpu_ready = True
    with pytest.raises(ValueError, match=match):
        policy.infer(images=[object(), object(), object()], instruction='task', proprioception=torch.zeros(26), noise=torch.zeros(20, 26), rtc_prefix_actions=prefix, inference_delay=delay)
    assert vla.predict_calls == []

@pytest.mark.parametrize('updates,match', [({'images': []}, 'expected 1 images'), ({'images': [None]}, 'must not contain None'), ({'instruction': '  '}, 'instruction'), ({'proprioception': []}, 'proprioception.*empty'), ({'proprioception': [0.0, math.nan]}, 'proprioception.*finite'), ({'state_mask': [1, 0]}, 'state_mask has 2 entries'), ({'state_mask': [1, 0.5, 0]}, 'state_mask.*0 or 1'), ({'noise': torch.zeros(3, 4)}, 'noise shape'), ({'noise': torch.full((2, 4), math.inf)}, 'noise.*finite'), ({'dof_mask': torch.full((2, 4), 0.5)}, 'dof_mask.*0 or 1'), ({'num_steps': 0}, 'num_steps'), ({'num_steps': 2.5}, 'num_steps'), ({'num_steps': True}, 'num_steps'), ({'noise_seed': 1.5}, 'noise_seed'), ({'use_rtc': 'true'}, 'use_rtc'), ({'use_rtc': True}, 'requires WallOssPolicy rtc=True')])
def test_infer_rejects_invalid_request_before_runtime(updates, match):
    (policy, vla) = _ready_policy()
    kwargs = _valid_infer_kwargs()
    kwargs.update(updates)
    with pytest.raises(ValueError, match=match):
        policy.infer(**kwargs)
    assert vla.predict_calls == []

@pytest.mark.parametrize('output,match', [({}, "mapping containing 'action'"), ({'action': [[0.0] * 4] * 2}, 'torch.Tensor'), ({'action': torch.zeros(3, 4)}, 'action shape'), ({'action': torch.full((2, 4), math.nan)}, 'non-finite')])
def test_infer_rejects_invalid_runtime_output(output, match):
    (policy, _) = _ready_policy(output=output)
    with pytest.raises(RPUBackendError, match=match):
        policy.infer(**_valid_infer_kwargs())

def test_infer_rejects_non_integral_runtime_action_dim():
    (policy, vla) = _ready_policy()
    vla.action_dim = 4.5
    with pytest.raises(ValueError, match='runtime action_dim'):
        policy.infer(**_valid_infer_kwargs())
    assert vla.predict_calls == []

def test_infer_and_prepare_reject_closed_runtime_before_dispatch():
    (policy, vla) = _ready_policy()
    policy._graph_profile = {'stale': True}
    vla.close()
    with pytest.raises(RPUBackendError, match='complete live runtime'):
        policy.infer(**_valid_infer_kwargs())
    with pytest.raises(RPUBackendError, match='complete live runtime'):
        policy.prepare_graphs(images=[object()], instruction='task', proprioception=[0.0, 0.0, 0.0], prefix_length_range=(545, 640))
    assert vla.predict_calls == []
    assert vla.prepare_calls == []
    assert policy.graphs_ready is False
    assert policy.graph_profile is None

def test_prepare_graphs_validates_range_and_integer_controls():
    (policy, vla) = _ready_policy()
    policy.prepare_graphs(images=[object()], instruction='task', proprioception=[0.0, 0.0, 0.0], state_mask=[1, 1, 1], prefix_length_range=(545, 640), num_steps=3, noise_seed=-1)
    call = vla.prepare_calls[0]
    assert call['prefix_length_range'] == (545, 640)
    assert call['num_steps'] == 3
    assert tuple(call['noise'].shape) == (1, 2, 4)

@pytest.mark.parametrize('prefix_range,num_steps,match', [((1.5, 2), 3, 'prefix_length_range minimum'), ((0, 2), 3, 'prefix_length_range minimum'), ((3, 2), 3, 'min <= max'), ((1,), 3, 'two-item'), ('1,2', 3, 'two-item'), ((1, 2), 0, 'num_steps')])
def test_prepare_graphs_rejects_invalid_controls_before_runtime(prefix_range, num_steps, match):
    (policy, vla) = _ready_policy()
    with pytest.raises(ValueError, match=match):
        policy.prepare_graphs(images=[object()], instruction='task', proprioception=[0.0, 0.0, 0.0], prefix_length_range=prefix_range, num_steps=num_steps)
    assert vla.prepare_calls == []

def test_wall_oss_runtime_invariants_survive_python_optimized_mode():
    root = Path(__file__).resolve().parents[4]
    runtime = root / 'python/rpu_backend/adapters/wall_oss/runtime.py'
    tree = ast.parse(runtime.read_text(encoding='utf-8'))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]
