from __future__ import annotations
import warnings
import pytest
import torch
import rpu_backend
_WARNING_FRAGMENT = 'does not take effect'

@pytest.fixture(autouse=True)
def _preserve_global_rng():
    state = torch.get_rng_state()
    yield
    torch.set_rng_state(state)

def test_rng_protocol_hooks_are_registered():
    assert callable(getattr(torch.rpu, '_is_in_bad_fork', None))
    assert callable(getattr(torch.rpu, 'manual_seed_all', None))
    assert torch._C._get_privateuse1_backend_name() == 'rpu'
    assert getattr(torch, 'rpu') is torch.rpu

def test_manual_seed_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        torch.manual_seed(1234)
    offenders = [str(w.message) for w in caught if _WARNING_FRAGMENT in str(w.message)]
    assert not offenders, f'torch.manual_seed still warns about the rpu device: {offenders}'

def test_hooks_report_the_truth_and_touch_nothing():
    assert torch.rpu._is_in_bad_fork() is False
    torch.manual_seed(7)
    before = torch.get_rng_state()
    assert torch.rpu.manual_seed_all(99) is None
    after = torch.get_rng_state()
    assert torch.equal(before, after), 'manual_seed_all must be a no-op; it ran before default_generator.manual_seed and would only race the real seeding'

def test_there_is_still_no_device_generator():
    with pytest.raises(RuntimeError, match='PrivateUse1HooksInterface'):
        torch.Generator(device='rpu')

@pytest.mark.hardware
def test_seeding_still_reaches_device_tensors():
    torch.manual_seed(1234)
    on_device = torch.randn(64, device='rpu').cpu()
    torch.manual_seed(1234)
    on_host = torch.randn(64)
    assert torch.equal(on_device, on_host), 'seeded device draw diverged from the CPU stream'
    torch.manual_seed(1234)
    again = torch.randn(64, device='rpu').cpu()
    assert torch.equal(on_device, again), 'same seed did not reproduce'
    torch.manual_seed(4321)
    other = torch.randn(64, device='rpu').cpu()
    assert not torch.equal(on_device, other), 'different seeds produced the same draw'
