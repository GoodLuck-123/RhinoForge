from __future__ import annotations
import os
import pytest
import torch
import rpu_backend

def _rpu_zeros(*shape, dtype=torch.float16):
    return torch.zeros(*shape, dtype=dtype, device='rpu')

@pytest.fixture
def handle():
    h = torch.ops.rpu.causal_decoder_create()
    yield h
    torch.ops.rpu.causal_decoder_destroy(h)

def test_set_weights_empty_defaults_preserve_existing_behavior(handle):
    with pytest.raises(RuntimeError, match='empty weight lists'):
        torch.ops.rpu.causal_decoder_set_weights(handle, [], [], [], [], [], [], [], [], [], [], [], _rpu_zeros(1, 64), _rpu_zeros(1, 64), _rpu_zeros(2048), 16, 8, 128, 2048, 6144, 1e-06, True)

def test_forward_default_args_preserve_existing_behavior(handle):
    h = _rpu_zeros(1, 4, 2048)
    with pytest.raises(RuntimeError, match='called before set_weights'):
        torch.ops.rpu.causal_decoder_forward(handle, h, [], [], None, 0, True)
if __name__ == '__main__':
    os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    h = torch.ops.rpu.causal_decoder_create()
    try:
        for fn in [test_set_weights_empty_defaults_preserve_existing_behavior, test_forward_default_args_preserve_existing_behavior]:
            fn(h)
            print(f'[OK] {fn.__name__}')
    finally:
        torch.ops.rpu.causal_decoder_destroy(h)
    print('\n[PASS] R-Phase 0.5 schema bw-compat — 2/2 cases')
