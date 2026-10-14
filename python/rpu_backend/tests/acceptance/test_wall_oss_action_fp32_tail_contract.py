from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
ACTION_PY = ROOT / 'python' / 'rpu_backend' / 'adapters' / 'wall_oss' / 'action.py'
DISPATCH_REGISTRATIONS = ROOT / 'src' / 'core' / 'rpu_dispatch_registrations.inc'
KERNEL_DECLS = ROOT / 'src' / 'core' / 'rpu_kernel_decls.h'
STEP_MODEL = ROOT / 'src' / 'fused' / 'rpu_wall_oss_action_step_model.cpp'

def _fused_denoise_body() -> str:
    source = ACTION_PY.read_text()
    start = source.index('    def _denoise_with_mask_fused(')
    end = source.index('    def denoise_with_mask(', start)
    return source[start:end]

def test_action_routes_are_cold_snapshotted_and_force_stepwise_execution():
    source = ACTION_PY.read_text()
    body = _fused_denoise_body()
    assert 'self._action_fp32_tail = rpu_env_bool(' in source
    assert 'self._denoise_unroll = rpu_env_bool(' in source
    assert 'rpu_env_bool(' not in body
    assert 'fp32_tail = self._action_fp32_tail' in body
    assert 'if not fp32_tail and self._denoise_unroll:' in body
    assert 'self._denoise_with_mask_loop(' in body

def test_off_path_does_not_create_a_cpu_fp32_noise_copy():
    body = _fused_denoise_body()
    assert 'if fp32_tail else None' in body
    assert 'x_fp32 if fp32_tail else noise' in body

def test_stepwise_fallback_collects_trajectories_only_in_debug_mode():
    body = _fused_denoise_body()
    assert 'if debug:\n                    vs_rpu.append(' in body
    assert 'else:\n                xs = vs = None' in body

def test_step_op_return_schema_matches_cpp_declarations():
    registrations = DISPATCH_REGISTRATIONS.read_text()
    schema_start = registrations.index('m.def("wall_oss_action_step_forward(')
    schema_end = registrations.index('m.def("wall_oss_action_denoise_loop_forward(', schema_start)
    schema = registrations[schema_start:schema_end]
    assert ') -> Tensor' in schema
    assert KERNEL_DECLS.read_text().count('at::Tensor rpu_wall_oss_action_step_forward(') == 1
    step_model = STEP_MODEL.read_text()
    assert step_model.count('at::Tensor rpu_wall_oss_action_step_forward(') == 1
    assert 'at::Tensor WallOssActionStepModel::step_forward(' in step_model
    assert 'return CausalDecoderModel::forward(' in step_model
