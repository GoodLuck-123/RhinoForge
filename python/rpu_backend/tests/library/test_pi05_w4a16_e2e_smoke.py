from __future__ import annotations
import os
from pathlib import Path
from rpu_backend.model_registry import model_path
import pytest
import torch
pytestmark = pytest.mark.skipif(os.environ.get('RPU_HW_AVAILABLE') != '1', reason='requires device hardware (set RPU_HW_AVAILABLE=1 and run with sudo)')
W4_DIR = Path(os.environ.get('RPU_PI05_W4A16_MODEL_DIR', str(model_path('pi05-libero-finetuned-w4real-kvint8'))))

def test_pi05_w4a16_loads_and_forwards():
    if not (W4_DIR / 'config.json').is_file():
        pytest.skip(f'real-w4 checkpoint missing: {W4_DIR} (run convert with --real-w4)')
    import rpu_backend
    from rpu_backend.api import Pi05Policy
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.configs.policies import PreTrainedConfig
    from rpu_backend.tests.library.test_pi05_e2e import _prepare_dummy_inputs
    torch.rpu.set_caching_allocator(True)
    torch.rpu.set_ddr_flush(True)
    config = PreTrainedConfig.from_pretrained(str(W4_DIR))
    config.dtype = 'float32'
    config.device = 'cpu'
    batch = _prepare_dummy_inputs(config, 32)
    policy = Pi05Policy.from_pretrained(str(W4_DIR), dtype=torch.float16).to('rpu')
    vlm = policy._lerobot_policy.model.paligemma_with_expert.paligemma.model.language_model
    q0 = vlm.layers[0].self_attn.q_proj.weight
    k0 = vlm.layers[0].self_attn.k_proj.weight
    assert q0.dtype == torch.uint8, f'q_proj should be packed uint8, got {q0.dtype}'
    assert k0.dtype == torch.int8, f'k_proj (KV) should stay int8, got {k0.dtype}'
    actions = policy._forward_rpu(batch, return_chunk=True, num_steps=5)
    a = actions.cpu().float()
    assert a.dim() == 3 and a.shape[0] == 1, f'unexpected action shape {tuple(a.shape)}'
    assert torch.isfinite(a).all(), 'action contains NaN/Inf'
