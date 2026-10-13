from __future__ import annotations
import os
from pathlib import Path
import pytest
import torch
from rpu_backend.adapters.hy_vla import preprocess as P
_INSTRUCTION = 'pick up the bottle'
_TAG = '<｜hy_Assistant｜>'

def test_prepare_images_accepts_the_shapes_a_real_client_sends() -> None:
    import numpy as np
    u8_chw = torch.randint(0, 256, (3, 48, 64), dtype=torch.uint8)
    ref = P.prepare_images([u8_chw.float() / 255.0])[0]
    cases = {'CHW uint8 tensor': u8_chw, 'NCHW uint8 tensor': u8_chw[None], 'HWC uint8 tensor': u8_chw.permute(1, 2, 0), 'NHWC uint8 tensor': u8_chw.permute(1, 2, 0)[None], 'HWC uint8 numpy': u8_chw.permute(1, 2, 0).numpy(), 'CHW fp64 tensor': u8_chw.double() / 255.0, 'NCHW fp32 numpy': (u8_chw[None].float() / 255.0).numpy()}
    for (name, frame) in cases.items():
        assert _same(P.prepare_images([frame])[0], ref), f'{name} 与规范入参不逐位相同'
    pil = pytest.importorskip('PIL.Image')
    got = P.prepare_images([pil.fromarray(u8_chw.permute(1, 2, 0).numpy())])[0]
    assert _same(got, ref), 'PIL.Image 与规范入参不逐位相同'

def test_prepare_images_refuses_ambiguous_input_instead_of_guessing() -> None:
    import numpy as np
    u8 = torch.randint(0, 256, (3, 48, 64), dtype=torch.uint8)
    with pytest.raises(ValueError, match='像素范围'):
        P.prepare_images([u8.float()])
    with pytest.raises(ValueError, match='像素范围'):
        P.prepare_images([u8.float() / 255.0 * 2.0 - 1.0])
    with pytest.raises(ValueError, match='3 通道'):
        P.prepare_images([torch.rand(4, 48, 64)])
    with pytest.raises(ValueError, match='batch'):
        P.prepare_images([torch.rand(2, 3, 48, 64)])
    with pytest.raises(ValueError, match='3 维或 4 维'):
        P.prepare_images([torch.rand(48, 64)])
    with pytest.raises(TypeError, match='不认识的帧类型'):
        P.prepare_images([[[0.0] * 64] * 48])
    with pytest.raises(ValueError, match='第 1 台相机'):
        P.prepare_images([torch.rand(3, 48, 64), np.zeros((4, 8, 8), dtype=np.uint8)])

def test_sample_noise_is_seed_reproducible_and_leaves_global_rng_alone() -> None:
    assert torch.equal(P.sample_noise(seed=0), P.sample_noise(seed=0))
    assert not torch.equal(P.sample_noise(seed=0), P.sample_noise(seed=1))
    torch.manual_seed(1234)
    before = torch.randn(4)
    torch.manual_seed(1234)
    P.sample_noise(seed=7)
    assert torch.equal(torch.randn(4), before), 'sample_noise(seed=…) 污染了全局 RNG'

def test_visual_segment_isolation_true_is_refused_not_silently_wrong() -> None:
    m = torch.ones(1, 8, 8, dtype=torch.bool)
    with pytest.raises(NotImplementedError, match='visual_segment_isolation'):
        P.apply_visual_segment_mask(m, [(1, 3)], [(1, 3)], isolation=True)

def test_actions_property_refuses_to_return_normalized_values() -> None:
    from rpu_backend.api.errors import RPUBackendError
    from rpu_backend.api.hy_embodied import HyEmbodiedActionOutput
    out = HyEmbodiedActionOutput(actions_normalized=torch.zeros(50, 20), actions_full=torch.zeros(50, 32), latency_ms=1.0, phase_ms={})
    assert out.physical_action_available is False
    with pytest.raises(RPUBackendError, match='归一化'):
        _ = out.actions
    assert out.actions_normalized.shape == (50, 20)

def test_action_decoder_shapes_and_first_step_is_near_the_start_pose() -> None:
    np = pytest.importorskip('numpy')
    pytest.importorskip('scipy')
    from rpu_backend.api.hy_action_decode import HyVlaActionDecoder
    h = 50
    mean = np.zeros((h, 20))
    mean[:, 3:9] = [1, 0, 0, 0, 1, 0]
    mean[:, 13:19] = [1, 0, 0, 0, 1, 0]
    dec = HyVlaActionDecoder(act_mean=mean, act_std=np.ones((h, 20)), umi_coord_frame=False, drop_first=False)
    start = np.array([0.3, 0.2, 0.5, 1, 0, 0, 0, 0.7, -0.3, -0.2, 0.5, 1, 0, 0, 0, 0.4])
    out = dec.decode(np.zeros((h, 20)), start)
    assert out.shape == (h, 16)
    assert np.isfinite(out).all()
    assert np.allclose(out[0, :3], start[:3], atol=1e-09), '零增量应解回起始位置'
    assert np.allclose(out[0, 8:11], start[8:11], atol=1e-09)
    assert HyVlaActionDecoder(act_mean=mean, act_std=np.ones((h, 20)), umi_coord_frame=False).decode(np.zeros((h, 20)), start).shape == (h - 1, 16)

def test_decoder_rejects_absolute_branch_checkpoints(tmp_path) -> None:
    import pickle
    np = pytest.importorskip('numpy')
    pytest.importorskip('scipy')
    from rpu_backend.api.hy_action_decode import HyVlaActionDecoder
    p = tmp_path / 'norm_stats.pkl'
    with open(p, 'wb') as f:
        pickle.dump({'action_mean': np.zeros((50, 20)), 'action_std': np.ones((50, 20)), 'action_mean_abs': np.zeros((50, 20)), 'action_std_abs': np.ones((50, 20))}, f)
    with pytest.raises(NotImplementedError, match='with_absolute'):
        HyVlaActionDecoder.from_norm_stats(p)

def _stub_policy(ckpt_dir, *, norm_stats_path=None, decoder=None, raw=None):
    import types
    from rpu_backend.api.hy_embodied import HyEmbodiedPolicy
    pol = HyEmbodiedPolicy.from_checkpoint(ckpt_dir, prefix_len=240, norm_stats_path=norm_stats_path)
    pol._rpu_ready = True
    pol._decoder = decoder
    pol._runner = types.SimpleNamespace(get_action=lambda **kw: torch.zeros(1, 50, 32) if raw is None else raw)
    return pol

def _stub_obs(*, n_valid=1):
    prefill_mask = torch.zeros(240, 240, dtype=torch.bool)
    if isinstance(n_valid, int) and (not isinstance(n_valid, bool)):
        prefill_mask[:min(max(n_valid, 0), 240), 0] = True
    return {'images': [torch.zeros(1, 3, 224, 224) for _ in range(3)], 'lang_tokens': torch.zeros(64, dtype=torch.long), 'prefill_mask': prefill_mask, 'prefill_pos': torch.zeros(240, dtype=torch.long), 'modality_mask': torch.zeros(240, dtype=torch.bool), 'denoise_mask': torch.zeros(51, 291, dtype=torch.bool), 'suffix_pos': torch.zeros(51, dtype=torch.long), 'noise': torch.zeros(1, 50, 32), 'state': torch.zeros(1, 32), 'n_valid': n_valid}

class _StubDecoder:

    def __init__(self, decoded):
        self._decoded = decoded

    def encode_state(self, ee_pose):
        import numpy as np
        return np.zeros(20, dtype=np.float32)

    def decode(self, norm20, ee_pose):
        return self._decoded

def _raw_path(monkeypatch, pol):
    pol._tok = object()
    monkeypatch.setattr(P, 'build_observation', lambda *a, **k: _stub_obs(n_valid=240))

def test_model_horizon_and_action_horizon_are_different_numbers(tmp_path) -> None:
    pol = _stub_policy(tmp_path)
    assert pol.model_horizon == 50
    assert pol.action_horizon == 49
    assert pol.action_horizon == pol.model_horizon - 1

def test_norm_stats_is_never_auto_enabled_from_the_ckpt_dir(tmp_path) -> None:
    from rpu_backend.api.hy_embodied import HyEmbodiedPolicy
    (tmp_path / 'norm_stats.pkl').write_bytes(b'not-really-a-pickle')
    pol = HyEmbodiedPolicy.from_checkpoint(tmp_path)
    assert pol._norm_stats_path is None
    assert pol.action_decode_configured is False

@pytest.mark.parametrize('kw', [{'images': [1, 2, 3]}, {'instruction': 'x'}, {'ee_pose': [0.0] * 16}, {'state': [0.0] * 20}, {'noise': torch.zeros(1)}, {'noise_seed': 0}])
def test_obs_is_mutually_exclusive_with_raw_observation_inputs(tmp_path, kw) -> None:
    pol = _stub_policy(tmp_path)
    with pytest.raises(ValueError, match='互斥'):
        pol.infer(obs={}, **kw)

def test_decoder_configured_but_no_ee_pose_yields_no_physical_action(tmp_path) -> None:
    from rpu_backend.api.errors import RPUBackendError
    pol = _stub_policy(tmp_path, decoder=_StubDecoder(None))
    assert pol.action_decode_configured is True
    out = pol.infer(obs=_stub_obs())
    assert out.physical_action_available is False
    assert out.extra['physical_action_available'] is False
    with pytest.raises(RPUBackendError, match='归一化'):
        _ = out.actions
    assert out.actions_normalized.shape == (50, 20)

def test_prebuilt_obs_requires_n_valid_and_rejects_prefix_truncation(tmp_path) -> None:
    pol = _stub_policy(tmp_path)
    missing = _stub_obs()
    missing.pop('n_valid')
    with pytest.raises(ValueError, match='n_valid'):
        pol.infer(obs=missing)
    with pytest.raises(ValueError, match='正整数'):
        pol.infer(obs=_stub_obs(n_valid=1.0))
    with pytest.raises(ValueError, match='prefix_len'):
        pol.infer(obs=_stub_obs(n_valid=241))
    inconsistent = _stub_obs(n_valid=2)
    inconsistent['prefill_mask'][1].zero_()
    with pytest.raises(ValueError, match='prefill_mask'):
        pol.infer(obs=inconsistent)
    noncontiguous = _stub_obs(n_valid=2)
    noncontiguous['prefill_mask'][1].zero_()
    noncontiguous['prefill_mask'][2, 0] = True
    with pytest.raises(ValueError, match='连续有效前缀'):
        pol.infer(obs=noncontiguous)

def test_prebuilt_obs_validates_runner_tensor_shapes_before_call(tmp_path) -> None:
    called = []
    pol = _stub_policy(tmp_path)
    pol._runner.get_action = lambda **kwargs: called.append(kwargs)
    malformed = _stub_obs()
    malformed['prefill_mask'] = torch.zeros(239, 240, dtype=torch.bool)
    with pytest.raises(ValueError, match='prefill_mask'):
        pol.infer(obs=malformed)
    assert called == []

def test_physical_action_available_is_true_only_with_a_real_decoded_chunk(tmp_path, monkeypatch) -> None:
    np = pytest.importorskip('numpy')
    pol = _stub_policy(tmp_path, norm_stats_path=tmp_path / 'n.pkl', decoder=_StubDecoder(np.zeros((49, 16))))
    _raw_path(monkeypatch, pol)
    out = pol.infer(images=[1, 2, 3], instruction='x', ee_pose=[0.0] * 16)
    assert out.physical_action_available is True
    assert out.extra['physical_action_available'] is True
    assert out.extra['action_horizon'] == 49 and out.extra['model_horizon'] == 50
    assert out.actions.shape == (49, 16)
    assert out.extra['coord_convention_source'] == 'vendor-default (unconfirmed)'

@pytest.mark.parametrize('bad,match', [('wrong_steps', '期望 49 步'), ('wrong_width', '意外形状'), ('nan', '非有限值')])
def test_decode_output_is_rejected_on_bad_shape_or_nonfinite(tmp_path, monkeypatch, bad, match) -> None:
    np = pytest.importorskip('numpy')
    from rpu_backend.api.errors import RPUBackendError
    decoded = {'wrong_steps': np.zeros((50, 16)), 'wrong_width': np.zeros((49, 7)), 'nan': np.full((49, 16), np.nan)}[bad]
    pol = _stub_policy(tmp_path, norm_stats_path=tmp_path / 'n.pkl', decoder=_StubDecoder(decoded))
    _raw_path(monkeypatch, pol)
    with pytest.raises(RPUBackendError, match=match):
        pol.infer(images=[1, 2, 3], instruction='x', ee_pose=[0.0] * 16)

def _same(got: torch.Tensor, ref: torch.Tensor) -> bool:
    if ref.dtype != got.dtype:
        ref = ref.bool() if got.dtype == torch.bool else ref.to(got.dtype)
    return got.shape == ref.shape and torch.equal(got, ref)
