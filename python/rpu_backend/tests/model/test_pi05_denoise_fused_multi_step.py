'test pi05 denoise fused multi step regression.'
import os
import pytest

from rpu_backend.adapters.pi05.runtime import _validate_fused_parity
from rpu_backend.tests.fixtures.pi05_fused_compare import (
    run_fresh_pi05_sample,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("PI05_HARDWARE") != "1",
    reason="Requires hardware + Pi0.5 weights — set PI05_HARDWARE=1 with sudo.")


def test_multi_step_fused_parity(monkeypatch):
    monkeypatch.setenv("RPU_PI05_FUSED_DENOISE", "0")
    out_b = run_fresh_pi05_sample("0", num_steps=10, seed=42)

    monkeypatch.setenv("RPU_PI05_FUSED_DENOISE", "1")
    out_f = run_fresh_pi05_sample("1", num_steps=10, seed=42)

    metrics = _validate_fused_parity(out_f, out_b, where="num_steps=10")
    print(f"num_steps=10 fused parity: {metrics}", flush=True)
