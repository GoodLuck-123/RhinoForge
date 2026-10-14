from __future__ import annotations
import pytest
import torch
from rpu_backend.adapters import dinov3
from rpu_backend.runtime.registry import ADAPTERS, get_adapter
from rpu_backend.api.errors import UnsupportedModelError

def _cfg(hidden, heads, layers, *, use_gated_mlp=False, num_register_tokens=4):
    from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTConfig
    return DINOv3ViTConfig(hidden_size=hidden, num_attention_heads=heads, num_hidden_layers=layers, patch_size=16, image_size=224, num_register_tokens=num_register_tokens, use_gated_mlp=use_gated_mlp, rope_theta=100.0, layer_norm_eps=1e-05)

def test_module_imports_and_registers():
    assert dinov3.DINOV3_VIT_ARCH == 'DINOv3ViTModel'
    assert 'DINOv3ViTModel' in ADAPTERS
    assert get_adapter('DINOv3ViTModel') is dinov3.DINOv3Adapter

def test_preflight_accepts_vit_b():
    dinov3.DINOv3Adapter.preflight(_cfg(768, 12, 12))

@pytest.mark.parametrize('hidden,heads,layers,reason', [(384, 6, 12, 'q·k'), (1024, 16, 24, 'LayerScale')])
def test_preflight_rejects_numeric_blocked_profiles(hidden, heads, layers, reason):
    with pytest.raises(UnsupportedModelError, match=f'numerical-blocked.*{reason}'):
        dinov3.DINOv3Adapter.preflight(_cfg(hidden, heads, layers))

def test_preflight_rejects_swiglu():
    with pytest.raises(UnsupportedModelError, match='SwiGLU'):
        dinov3.DINOv3Adapter.preflight(_cfg(768, 12, 12, use_gated_mlp=True))

def test_preflight_rejects_vit_7b():
    with pytest.raises(UnsupportedModelError, match='profile'):
        dinov3.DINOv3Adapter.preflight(_cfg(4096, 32, 40))

def test_preflight_rejects_non_default_register_count():
    with pytest.raises(UnsupportedModelError, match='num_register_tokens'):
        dinov3.DINOv3Adapter.preflight(_cfg(768, 12, 12, num_register_tokens=8))

def test_build_dinov3_rope_tables_shape_and_theta_default():
    (cos, sin) = dinov3.build_dinov3_rope_tables(head_dim=64, max_hw=14, device='cpu', dtype=torch.float32)
    assert cos.shape == (14, 16)
    assert sin.shape == (14, 16)
    assert torch.allclose(cos[0, 0], torch.tensor(0.9009687, dtype=torch.float32), atol=0.0001)

def test_build_position_idx_row_major_layout():
    pos = dinov3.build_position_idx(num_patches_h=14, num_patches_w=14)
    assert pos.shape == (196, 2)
    assert pos.dtype == torch.int16
    assert pos[0].tolist() == [0, 0]
    assert pos[1].tolist() == [0, 1]
    assert pos[14].tolist() == [1, 0]
    assert pos[-1].tolist() == [13, 13]
