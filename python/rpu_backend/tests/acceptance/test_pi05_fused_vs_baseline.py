import os
import pytest
from rpu_backend.adapters.pi05.runtime import _validate_fused_parity
from rpu_backend.tests.fixtures.pi05_fused_compare import run_fresh_pi05_sample
pytestmark = pytest.mark.skipif(os.environ.get('PI05_HARDWARE') != '1', reason='Requires hardware + Pi0.5 weights — set PI05_HARDWARE=1 with sudo.')

@pytest.mark.parametrize('seed', [0, 1, 7, 42, 1234])
def test_fused_vs_baseline_per_seed(monkeypatch, seed):
    monkeypatch.setenv('RPU_PI05_FUSED_DENOISE', '0')
    out_b = run_fresh_pi05_sample('0', num_steps=10, seed=seed)
    monkeypatch.setenv('RPU_PI05_FUSED_DENOISE', '1')
    out_f = run_fresh_pi05_sample('1', num_steps=10, seed=seed)
    metrics = _validate_fused_parity(out_f, out_b, where=f'seed={seed}')
    print(f'seed={seed} fused parity: {metrics}', flush=True)
