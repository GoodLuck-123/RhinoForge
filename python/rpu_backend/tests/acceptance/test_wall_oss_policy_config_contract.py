from __future__ import annotations
import ast
import math
from collections.abc import Mapping
from pathlib import Path
import pytest
import torch
from rpu_backend.api import WallOssPolicy
from rpu_backend.api import wall_oss as wall_oss_api

def _make_policy(**kwargs) -> WallOssPolicy:
    return WallOssPolicy.from_checkpoint('unused/checkpoint', **kwargs)

def test_valid_constructor_normalizes_public_configuration():
    policy = _make_policy(camera_names=('face_view', 'left_wrist_view'), state_bins=512, action_hz=10, action_horizon=10, num_steps=5, max_seq_len=640, active_slots=(0, 3), runtime_env={'HF_HUB_OFFLINE': 1, 'RPU_WARMUP': True})
    assert policy.camera_names == ['face_view', 'left_wrist_view']
    assert policy.action_horizon == 10
    assert policy._state_bins == 512
    assert policy._action_hz == 10.0
    assert policy._active_slots == [0, 3]
    assert policy._runtime_env == {'HF_HUB_OFFLINE': '1', 'RPU_WARMUP': 'True'}

@pytest.mark.parametrize('name,value', [('state_bins', 1), ('state_bins', 256.5), ('state_bins', True), ('action_horizon', 0), ('action_horizon', 1.5), ('num_steps', -1), ('num_steps', '10'), ('max_seq_len', 0), ('max_seq_len', False)])
def test_integer_configuration_rejects_truncation_and_nonpositive_values(name, value):
    with pytest.raises(ValueError, match=name):
        _make_policy(**{name: value})

def test_shared_cache_must_leave_room_beyond_action_horizon():
    with pytest.raises(ValueError, match='greater than action_horizon'):
        _make_policy(max_seq_len=32, action_horizon=32)

@pytest.mark.parametrize('value', [0, -1.0, math.nan, math.inf, True, '32'])
def test_action_hz_must_be_positive_and_finite(value):
    with pytest.raises(ValueError, match='action_hz'):
        _make_policy(action_hz=value)

@pytest.mark.parametrize('name', ['delta_action', 'w8a16', 'w4a16', 'nvfp4a16', 'rtc'])
def test_boolean_configuration_does_not_use_truthiness(name):
    with pytest.raises(ValueError, match=name):
        _make_policy(**{name: 'false'})

@pytest.mark.parametrize('camera_names', ['face_view', (), ('',), (1,), ('unknown_view',)])
def test_camera_roster_fails_before_model_loading(camera_names):
    with pytest.raises(ValueError, match='camera_names'):
        _make_policy(camera_names=camera_names)

def test_public_camera_roster_stays_in_sync_with_prompt_mapping():
    root = Path(__file__).resolve().parents[4]
    runtime = root / 'python/rpu_backend/adapters/wall_oss/runtime.py'
    tree = ast.parse(runtime.read_text(encoding='utf-8'))
    assignment = next((node for node in tree.body if isinstance(node, ast.Assign) and any((isinstance(target, ast.Name) and target.id == '_CAM_NAME_MAPPING' for target in node.targets))))
    mapping = ast.literal_eval(assignment.value)
    assert wall_oss_api._SUPPORTED_CAMERA_NAMES == frozenset(mapping)

def _normalizer_validator():
    root = Path(__file__).resolve().parents[4]
    runtime = root / 'python/rpu_backend/adapters/wall_oss/runtime.py'
    tree = ast.parse(runtime.read_text(encoding='utf-8'))
    function = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_validate_normalizer_dataset'))
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {'Mapping': Mapping}
    exec(compile(module, str(runtime), 'exec'), namespace)
    return namespace['_validate_normalizer_dataset']

def test_normalizer_dataset_requires_both_sidecars_and_both_fields():
    validate = _normalizer_validator()
    action = {'min.robot': torch.zeros(2), 'delta.robot': torch.ones(2)}
    proprio = {'min.robot': torch.zeros(3), 'delta.robot': torch.ones(3)}
    validate(action, proprio, 'robot')
    with pytest.raises(ValueError, match='normalizer_action.*available datasets'):
        validate({'min.other': torch.zeros(2)}, proprio, 'robot')
    with pytest.raises(ValueError, match='normalizer_propri.*available datasets'):
        validate(action, {'min.other': torch.zeros(3)}, 'robot')

def test_normalizer_preflight_precedes_irreversible_model_build():
    root = Path(__file__).resolve().parents[4]
    runtime = root / 'python/rpu_backend/adapters/wall_oss/runtime.py'
    source = runtime.read_text(encoding='utf-8')
    tree = ast.parse(source)
    build = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_wall_oss_vla'))
    body = ast.get_source_segment(source, build)
    assert body is not None
    preflight = body.index('_validate_normalizer_dataset(')
    assert preflight < body.index('build_wall_oss_action(')
    assert preflight < body.index('build_wall_oss_vision(')
    assert 'weights_only=True' in body
    assert 'dataset_key = [' not in body

@pytest.mark.parametrize('dataset_key', [None, ''])
def test_dataset_key_must_be_nonempty_string(dataset_key):
    with pytest.raises(ValueError, match='dataset_key'):
        _make_policy(dataset_key=dataset_key)

def test_rtc_constructor_locks_the_wrc_v728_profile():
    policy = _make_policy(rtc=True, dataset_key='ex_normal', camera_names=('face_view', 'left_wrist_view', 'right_wrist_view'), delta_action=True, state_bins=512, action_horizon=20, num_steps=6)
    assert policy._rtc is True
    assert policy._rtc_max_delay == 10
    assert policy._action_hz == 32.0
    assert policy._active_slots == list(range(20))
    with pytest.raises(ValueError, match='action_horizon=20'):
        _make_policy(rtc=True, dataset_key='ex_normal', camera_names=('face_view', 'left_wrist_view', 'right_wrist_view'), delta_action=True, state_bins=512, action_horizon=32, num_steps=6)

def test_rtc_accepts_only_overflow_safe_hybrid_w8a16_quantization():
    common = {'rtc': True, 'dataset_key': 'ex_normal', 'camera_names': ('face_view', 'left_wrist_view', 'right_wrist_view'), 'delta_action': True, 'state_bins': 512, 'action_horizon': 20, 'num_steps': 6, 'fp16_ckpt_dir': '/models/wrc-sf-v728-fp16'}
    policy = _make_policy(w8a16=True, **common)
    assert policy.precision == 'w8a16-hybrid'
    for mode in ('w4a16', 'nvfp4a16'):
        with pytest.raises(ValueError, match='hybrid W8A16'):
            _make_policy(**common, **{mode: True})

@pytest.mark.parametrize('update,match', [({'action_hz': 10}, 'action_hz=32'), ({'max_seq_len': 671}, 'max_seq_len>=672'), ({'camera_names': ('face_view',)}, 'camera_names'), ({'active_slots': tuple(range(26))}, 'active_slots=range')])
def test_rtc_rejects_training_profile_drift(update, match):
    kwargs = {'rtc': True, 'dataset_key': 'ex_normal', 'camera_names': ('face_view', 'left_wrist_view', 'right_wrist_view'), 'delta_action': True, 'state_bins': 512, 'action_horizon': 20, 'num_steps': 6}
    kwargs.update(update)
    with pytest.raises(ValueError, match=match):
        _make_policy(**kwargs)

def test_rtc_locks_trained_desktop_state_and_action_masks():
    policy = _make_policy(rtc=True, dataset_key='ex_normal', camera_names=('face_view', 'left_wrist_view', 'right_wrist_view'), delta_action=True, state_bins=512, action_horizon=20, num_steps=6)
    (_, _, _, state_mask) = policy._observation_inputs(images=[object(), object(), object()], instruction='sort parcel', proprioception=torch.zeros(26), state_mask=None)
    expected_state = torch.tensor([1.0] * 20 + [0.0] * 6)
    expected_dof = torch.tensor([1.0] * 20 + [0.0] * 6)
    assert torch.equal(state_mask.reshape(-1), expected_state)
    assert torch.equal(policy._dof_mask_input(None, horizon=20, action_dim=26)[0, 0], expected_dof)
    customer_state_mask = torch.tensor([1.0] * 20 + [0.0] * 6)
    (_, _, _, state_mask) = policy._observation_inputs(images=[object(), object(), object()], instruction='sort parcel', proprioception=torch.zeros(26), state_mask=customer_state_mask)
    assert torch.equal(state_mask.reshape(-1), customer_state_mask)
    with pytest.raises(ValueError, match='state_mask'):
        policy._observation_inputs(images=[object(), object(), object()], instruction='sort parcel', proprioception=torch.zeros(26), state_mask=torch.ones(26))
    with pytest.raises(ValueError, match='dof_mask'):
        policy._dof_mask_input(torch.ones(1, 20, 26), horizon=20, action_dim=26)

def test_rtc_disables_unvalidated_model_scoped_precision_and_vision_schedules():
    policy = _make_policy(rtc=True, dataset_key='ex_normal', camera_names=('face_view', 'left_wrist_view', 'right_wrist_view'), delta_action=True, state_bins=512, action_horizon=20, num_steps=6, runtime_env={'RPU_WALL_OSS_ACTION_FP32_TAIL': '1', 'RPU_WALL_OSS_VISION_WEIGHT_OUTER': '1', 'RPU_WALL_OSS_VISION_LAYER_GROUP': '32'})
    assert 'RPU_LINEAR_ACC32' not in policy._runtime_env
    assert 'RPU_RMSNORM_NEWTON' not in policy._runtime_env
    assert policy._runtime_env['RPU_WALL_OSS_ACTION_FP32_TAIL'] == '0'
    assert policy._runtime_env['RPU_WALL_OSS_FUSED_DENOISE'] == '1'
    assert policy._runtime_env['RPU_WALL_OSS_DENOISE_UNROLL'] == '1'
    assert policy._runtime_env['RPU_WALL_OSS_ATTN_TP8'] == '1'
    assert policy._runtime_env['RPU_WALL_OSS_VISION_FUSED_MERGER'] == '1'
    assert policy._runtime_env['RPU_WALL_OSS_VISION_WEIGHT_OUTER'] == '0'
    assert policy._runtime_env['RPU_WALL_OSS_VISION_LAYER_GROUP'] == '0'
    assert policy._runtime_env['RPU_WALL_OSS_VISION_MIXED_PRECISION'] == '1'
    assert policy._runtime_env['RPU_WALL_OSS_PREFILL_MIXED_PRECISION'] == '1'

def test_runtime_env_rejects_retired_global_linear_acc32():
    with pytest.raises(ValueError, match='RPU_LINEAR_ACC32 is retired'):
        _make_policy(runtime_env={'RPU_LINEAR_ACC32': '1'})

@pytest.mark.parametrize('name', ['RPU_RMSNORM_NEWTON', 'RPU_RMSNORM_VWARP'])
def test_runtime_env_rejects_non_authoritative_rmsnorm_selectors(name):
    with pytest.raises(ValueError, match='not a Wall-OSS production selector'):
        _make_policy(runtime_env={name: '1'})

def test_rtc_w8_preserves_explicit_normal_vision_schedule():
    common = {'rtc': True, 'w8a16': True, 'fp16_ckpt_dir': 'unused/fp16', 'dataset_key': 'ex_normal', 'camera_names': ('face_view', 'left_wrist_view', 'right_wrist_view'), 'delta_action': True, 'state_bins': 512, 'action_horizon': 20, 'num_steps': 6}
    default = _make_policy(**common)
    normal = _make_policy(**common, runtime_env={'RPU_WALL_OSS_VISION_WEIGHT_OUTER': '0'})
    assert 'RPU_WALL_OSS_VISION_WEIGHT_OUTER' not in default._runtime_env
    assert 'RPU_RMSNORM_VWARP' not in default._runtime_env
    assert normal._runtime_env['RPU_WALL_OSS_VISION_WEIGHT_OUTER'] == '0'
    assert 'RPU_RMSNORM_VWARP' not in normal._runtime_env

@pytest.mark.parametrize('active_slots', ['0', (True,), (1.5,), (-1,)])
def test_active_slots_reject_ambiguous_or_negative_entries(active_slots):
    with pytest.raises(ValueError, match='active_slots'):
        _make_policy(active_slots=active_slots)

def test_active_slot_above_loaded_action_dim_is_not_silently_ignored():
    policy = _make_policy(active_slots=(0, 4))
    with pytest.raises(ValueError, match='action_dim=4'):
        policy._default_dof_mask(horizon=2, action_dim=4)
    policy = _make_policy(active_slots=(0, 3))
    mask = policy._default_dof_mask(horizon=2, action_dim=4)
    assert tuple(mask.shape) == (1, 2, 4)
    assert torch.equal(mask[0, 0], torch.tensor([1.0, 0.0, 0.0, 1.0]))

@pytest.mark.parametrize('runtime_env', [[], {1: 'x'}, {'': 'x'}, {'A=B': 'x'}, {'A': 'x\x00y'}])
def test_runtime_env_rejects_values_os_environ_cannot_represent(runtime_env):
    with pytest.raises(ValueError, match='runtime_env'):
        _make_policy(runtime_env=runtime_env)

def test_public_execution_config_accepts_all_wall_stages():
    requested = {'prefill': {'chunk_size': 320, 'padding_rows': 'auto', 'padding_budget': 64}, 'vision': {'chunk_size': 576}, 'action': {'chunk_size': 32}}
    policy = _make_policy(action_horizon=32, rpu_execution=requested)
    assert {stage: dict(fields) for (stage, fields) in policy._rpu_execution.items()} == requested

@pytest.mark.parametrize('stage,field', [('vision', 'padding_rows'), ('vision', 'padding_budget'), ('action', 'padding_rows'), ('action', 'padding_budget')])
def test_wall_vision_and_action_reject_unsupported_padding(stage, field):
    with pytest.raises(ValueError, match='not supported'):
        _make_policy(rpu_execution={stage: {field: 0}})

def test_exact_action_chunk_must_equal_policy_horizon():
    with pytest.raises(ValueError, match='ceil16\\(action_horizon\\)'):
        _make_policy(action_horizon=16, rpu_execution={'action': {'chunk_size': 32}})
    policy = _make_policy(action_horizon=10, rpu_execution={'action': {'chunk_size': 16}})
    assert policy._rpu_execution['action']['chunk_size'] == 16

def test_vla_builder_propagates_execution_config_to_every_component_branch():
    root = Path(__file__).resolve().parents[4]
    runtime_path = root / 'python/rpu_backend/adapters/wall_oss/runtime.py'
    tree = ast.parse(runtime_path.read_text(encoding='utf-8'))
    build = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'build_wall_oss_vla'))
    for builder in ('build_wall_oss_action', 'build_wall_oss_vision'):
        calls = [node for node in ast.walk(build) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and (node.func.id == builder)]
        assert len(calls) == 4
        assert all((any((keyword.arg == 'execution_config' for keyword in call.keywords)) for call in calls))

def test_vla_horizon_and_planner_ordering_contracts():
    root = Path(__file__).resolve().parents[4]
    runtime_path = root / 'python/rpu_backend/adapters/wall_oss/runtime.py'
    source = runtime_path.read_text(encoding='utf-8')
    tree = ast.parse(source)
    runtime_class = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'WallOssVLA'))
    methods = {node.name: ast.get_source_segment(source, node) for node in runtime_class.body if isinstance(node, ast.FunctionDef)}
    predict = methods['predict']
    prepare = methods['prepare_graphs']
    assert predict.index('validate_execution_horizon') < predict.index('self.vision.forward')
    assert prepare.index('validate_execution_horizon') < prepare.index('_prefix_execution_profiles')
    assert predict.index('self.vision.forward') < predict.index('padded_len, _ = self._prefill_execution_plan')
    materialize = prepare.index('self.predict(')
    clear_throwaway = prepare.index('cache.clear()', materialize)
    stable_profiles = prepare.index('profiles = self._prefix_execution_profiles')
    assert materialize < clear_throwaway < stable_profiles
