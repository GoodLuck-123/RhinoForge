import os
import gc
import pytest
import torch
pytestmark = pytest.mark.skipif(os.environ.get('PI05_HARDWARE') != '1', reason='Requires hardware + Pi0.5 weights; set PI05_HARDWARE=1 and run with sudo.')

def _make_libero_dummy_inputs(bsize=1, num_text_tokens=32):
    images = [torch.zeros(bsize, 3, 224, 224, dtype=torch.float16)]
    img_masks = [torch.ones(bsize, dtype=torch.bool)]
    tokens = torch.zeros(bsize, num_text_tokens, dtype=torch.int64)
    masks = torch.ones(bsize, num_text_tokens, dtype=torch.bool)
    return (images, img_masks, tokens, masks)

def prepare_pi05_step_fixture(wrapper):
    from rpu_backend.adapters.pi05.weights import _precompute_adarms_cond_all
    from rpu_backend.adapters.pi05.runtime import _resolve_rpu_device, _prefill_prefix_embs, _apply_kv_cache, _build_attention_mask_4d, _plan_pi05_denoise_action_execution, _set_denoise_rope_position
    from rpu_backend.graph import GraphSignature
    lerobot_policy = wrapper._lerobot_policy
    cfg = lerobot_policy.config
    pi05_pytorch = lerobot_policy.model
    handle = getattr(pi05_pytorch, '_rpu_fused_denoise_handle', None)
    cache = getattr(pi05_pytorch, '_rpu_fused_denoise_graph_cache', None)
    assert handle is not None, 'Fixture: _install_fused_denoise_handle did not set the fused handle'
    assert cache is not None, 'Fixture: _install_fused_denoise_handle did not set the graph cache'
    expert = pi05_pytorch.paligemma_with_expert.gemma_expert.model
    ec = expert.config
    H = pi05_pytorch.action_in_proj.out_features
    num_steps = 1
    L = len(expert.layers)
    (images, img_masks, tokens, masks) = _make_libero_dummy_inputs(bsize=1)
    rpu_device = _resolve_rpu_device(pi05_pytorch)
    (past_key_values, prefix_pad_masks, prefix_len) = _prefill_prefix_embs(pi05_pytorch, images, img_masks, tokens, masks, rpu_device)
    _apply_kv_cache(pi05_pytorch, past_key_values)
    mask_4d = _build_attention_mask_4d(pi05_pytorch, prefix_pad_masks, prefix_len, bsize=1, chunk_size=cfg.chunk_size, dtype=torch.float16, device='rpu')
    adarms_cond_all = _precompute_adarms_cond_all(pi05_pytorch, num_steps=num_steps)
    rope_pos = _set_denoise_rope_position(handle, prefix_pad_masks)
    cache_capacity = past_key_values.k_caches[0].size(1) * past_key_values.k_caches[0].size(5)
    action_plan = _plan_pi05_denoise_action_execution(pi05_pytorch, handle=int(handle), logical_len=int(cfg.chunk_size), position=int(prefix_len), kv_len=int(prefix_len + cfg.chunk_size), cache_capacity=int(cache_capacity), loop_mode=False, num_steps=1)
    sig = GraphSignature(op_id='pi05_denoise_step', shapes=[1, cfg.chunk_size, cfg.max_action_dim], dyn_dims=[L, ec.num_attention_heads, int(getattr(expert, '_rpu_effective_num_kv_heads', ec.num_key_value_heads)), ec.head_dim, prefix_len, num_steps, H, rope_pos, *action_plan.graph_key_words()], dtypes=[torch.float16])

    def step(x_t_rpu, cond_step, v_t_buf, step_sig=sig):
        with cache.capture(step_sig):
            torch.ops.rpu.pi05_denoise_step_forward(handle, x_t_rpu, past_key_values.k_caches, past_key_values.v_caches, cond_step, mask_4d, v_t_buf, prefix_len, action_plan.selected.stage_tuple.physical_descriptor)
    return {'handle': handle, 'cache': cache, 'sig': sig, 'wrapper': wrapper, 'pi05_pytorch': pi05_pytorch, 'lerobot_policy': lerobot_policy, 'config': cfg, 'past_key_values': past_key_values, 'mask_4d': mask_4d, 'prefix_len': prefix_len, 'adarms_cond_all': adarms_cond_all, 'H': H, 'L': L, 'step': step, 'action_plan': action_plan}

@pytest.fixture(scope='module')
def pi05_fuse_fixture():
    from rpu_backend.tests.fixtures.pi05_loader import load_pi05_model
    from rpu_backend.api.policy import Pi05Policy
    from rpu_backend.model_registry import model_path as registry_model_path
    model_path = os.environ.get('PI05_MODEL_PATH', str(registry_model_path('pi05-libero-finetuned-w8a16-vlm-expert')))
    (lerobot_policy, _) = load_pi05_model(model_path, dtype='fp16')
    wrapper = Pi05Policy.from_lerobot_policy(lerobot_policy)
    wrapper.to('rpu')
    gc.collect()
    try:
        yield prepare_pi05_step_fixture(wrapper)
    finally:
        wrapper.close()
        del wrapper
        del lerobot_policy
        gc.collect()
