import sys
import pytest
import torch.nn as nn

def test_internal_debug_calls():
    from rpu_backend.runtime.debug import get_debug, get_profile
    val_d = get_debug()
    val_p = get_profile()
    assert isinstance(val_d, bool), f'get_debug returned non-bool: {type(val_d)}'
    assert isinstance(val_p, bool), f'get_profile returned non-bool: {type(val_p)}'

def test_internal_runtime_calls():
    from rpu_backend.runtime.control import get_spm_mode
    _ = get_spm_mode()

def test_internal_weights_calls():
    from rpu_backend.runtime.weights import get_linear_partition
    result = get_linear_partition(128, 128)
    assert result in (-1, 0, 1), f'unexpected partition code: {result}'

def test_internal_patches_calls():
    import torch
    from rpu_backend.runtime.decoder import _install_rmsnorm_class_swap as patch_rmsnorm_class

    class SyntheticRMSNorm(nn.Module):

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(16))
            self.variance_epsilon = 1e-06

        def forward(self, x):
            return x
    patch_rmsnorm_class(SyntheticRMSNorm)
    m = SyntheticRMSNorm()
    _ = m.forward(m.weight.new_ones(1, 16))

def main() -> int:
    test_internal_debug_calls()
    test_internal_runtime_calls()
    test_internal_weights_calls()
    test_internal_patches_calls()
    print('PASS: _internal.* call-level smoke (M2 iter-3; M1 cross-module path; ops gated behind hardware)')
    return 0
if __name__ == '__main__':
    sys.exit(main())

@pytest.mark.hardware
def test_internal_ops_calls():
    import torch
    rms_norm = torch.ops.rpu.rms_norm
    x = torch.ones(1, 16, dtype=torch.float16, device='rpu')
    w = torch.ones(16, dtype=torch.float16, device='rpu')
    y = rms_norm(x, (16,), w, 1e-06)
    assert y.shape == x.shape
