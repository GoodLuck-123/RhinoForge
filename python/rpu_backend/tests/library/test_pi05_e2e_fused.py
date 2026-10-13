import os
import pytest
from rpu_backend.tests.fixtures.pi05_fused_compare import load_pi05_on_rpu, run_sample_actions, _release
pytestmark = pytest.mark.skipif(os.environ.get('PI05_HARDWARE') != '1', reason='Requires hardware + Pi0.5 weights — set PI05_HARDWARE=1 with sudo.')

def test_e2e_fused_smoke(monkeypatch):
    monkeypatch.setenv('RPU_PI05_FUSED_DENOISE', '1')
    (wrapper, policy, cfg) = load_pi05_on_rpu('1')
    try:
        out = run_sample_actions(policy, num_steps=10, seed=0)
        assert out.shape == (1, cfg.chunk_size, cfg.max_action_dim), f'unexpected shape {tuple(out.shape)}'
        assert out.isfinite().all(), 'output contains non-finite values'
    finally:
        _release(wrapper, policy)
