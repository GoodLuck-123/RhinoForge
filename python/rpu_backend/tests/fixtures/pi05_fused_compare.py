from __future__ import annotations
import gc
import os
import torch

def _model_path() -> str:
    from rpu_backend.model_registry import model_path
    return os.environ.get('PI05_MODEL_PATH', str(model_path('pi05-libero-finetuned-w8a16-vlm-expert')))

def _release(wrapper, lerobot_policy):
    from rpu_backend.api.causal_lm import _release_live_instance
    pi05_pytorch = lerobot_policy.model
    handle = getattr(pi05_pytorch, '_rpu_fused_denoise_handle', None)
    if handle is not None:
        finalizers = getattr(getattr(wrapper, '_adapter', None), '_finalizers', [])
        live_finalizers = [finalizer for finalizer in finalizers if finalizer.alive]
        if live_finalizers:
            for finalizer in live_finalizers:
                finalizer()
            finalizers.clear()
        else:
            try:
                torch.ops.rpu.pi05_denoise_step_destroy(handle)
            except Exception:
                pass
        pi05_pytorch._rpu_fused_denoise_handle = None
    pi05_pytorch._rpu_fused_denoise_graph_cache = None
    try:
        _release_live_instance(lerobot_policy)
    except Exception:
        pass
    del wrapper, pi05_pytorch, lerobot_policy
    gc.collect()

def load_pi05_on_rpu(env_value: str):
    from rpu_backend.tests.fixtures.pi05_loader import load_pi05_model
    from rpu_backend.api.policy import Pi05Policy
    os.environ['RPU_PI05_FUSED_DENOISE'] = env_value
    (lerobot_policy, cfg) = load_pi05_model(_model_path(), dtype='fp16')
    wrapper = Pi05Policy.from_lerobot_policy(lerobot_policy)
    wrapper.to('rpu')
    gc.collect()
    return (wrapper, lerobot_policy, cfg)

def run_sample_actions(lerobot_policy, num_steps: int, seed: int=42, bsize: int=1, num_text_tokens: int=32):
    from rpu_backend.tests.fixtures.pi05_fuse_fixture import _make_libero_dummy_inputs
    (images, img_masks, tokens, masks) = _make_libero_dummy_inputs(bsize, num_text_tokens)
    cfg = lerobot_policy.model.config
    g = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(bsize, cfg.chunk_size, cfg.max_action_dim, dtype=torch.float32, generator=g)
    return lerobot_policy.model.sample_actions(images, img_masks, tokens, masks, noise=noise, num_steps=num_steps)

def run_fresh_pi05_sample(env_value: str, *, num_steps: int, seed: int=42, bsize: int=1, num_text_tokens: int=32):

    def _run_once():
        (wrapper, lerobot_policy, _) = load_pi05_on_rpu(env_value)
        try:
            output = run_sample_actions(lerobot_policy, num_steps=num_steps, seed=seed, bsize=bsize, num_text_tokens=num_text_tokens)
            return output.detach().cpu().clone()
        finally:
            _release(wrapper, lerobot_policy)
    result = _run_once()
    gc.collect()
    return result
