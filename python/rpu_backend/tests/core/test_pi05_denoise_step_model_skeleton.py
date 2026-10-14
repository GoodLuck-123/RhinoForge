import pytest
import torch
import rpu_backend

def test_pi05_denoise_step_ops_registered():
    for op in ('pi05_denoise_step_create', 'pi05_denoise_step_destroy', 'pi05_denoise_step_set_weights', 'pi05_denoise_step_forward'):
        assert hasattr(torch.ops.rpu, op), f'torch.ops.rpu.{op} missing'

def test_create_destroy_roundtrip():
    h = torch.ops.rpu.pi05_denoise_step_create()
    assert isinstance(h, int) and h >= 0, f'create returned {h!r}'
    torch.ops.rpu.pi05_denoise_step_destroy(h)
