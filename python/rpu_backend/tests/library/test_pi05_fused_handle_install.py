import os
import pytest
import torch
from rpu_backend.tests.fixtures.pi05_loader import load_pi05_model
pytestmark = pytest.mark.skipif(os.environ.get('PI05_HARDWARE') != '1', reason='Requires hardware + Pi0.5 weights — set PI05_HARDWARE=1.')

def _load_and_swizzle():
    import gc
    from rpu_backend.api.policy import Pi05Policy
    from rpu_backend.model_registry import model_path as registry_model_path
    model_path = os.environ.get('PI05_MODEL_PATH', str(registry_model_path('pi05-libero-finetuned-w8a16-vlm-expert')))
    (lerobot_policy, _) = load_pi05_model(model_path, dtype='fp16')
    wrapper = Pi05Policy.from_lerobot_policy(lerobot_policy)
    wrapper.to('rpu')
    gc.collect()
    return (wrapper, lerobot_policy.model)

def _release(wrapper, pi05_pytorch):
    import gc
    from rpu_backend.api.causal_lm import _release_live_instance
    handle = getattr(pi05_pytorch, '_rpu_fused_denoise_handle', None)
    if handle is not None:
        try:
            torch.ops.rpu.pi05_denoise_step_destroy(handle)
        except Exception:
            pass
        pi05_pytorch._rpu_fused_denoise_handle = None
    _release_live_instance(wrapper._lerobot_policy)
    del pi05_pytorch
    del wrapper
    gc.collect()

def test_fused_handle_installed_by_default(monkeypatch):
    monkeypatch.delenv('RPU_PI05_FUSED_DENOISE', raising=False)
    (wrapper, pi05_pytorch) = _load_and_swizzle()
    try:
        assert getattr(pi05_pytorch, '_rpu_fused_denoise_handle', None) is not None
    finally:
        _release(wrapper, pi05_pytorch)

def test_env_optout_skips_handle(monkeypatch):
    monkeypatch.setenv('RPU_PI05_FUSED_DENOISE', '0')
    (wrapper, pi05_pytorch) = _load_and_swizzle()
    try:
        assert pi05_pytorch._rpu_fused_denoise_handle is None
    finally:
        _release(wrapper, pi05_pytorch)
