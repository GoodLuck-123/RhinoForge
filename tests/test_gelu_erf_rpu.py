"""Focused board checks for the FP16 eager ERF GELU bridge."""
import os

import pytest
import torch
import rpu_backend


pytestmark = pytest.mark.skipif(
    not os.access("/dev/rpu", os.R_OK | os.W_OK), reason="requires the RPU board")


def test_gelu_erf_values_and_independent_outputs():
    for count in (8, 55 * 896):
        inputs = torch.linspace(-5, 5, count, dtype=torch.float16)
        device = inputs.to("rpu")
        result = torch.ops.rpu.gelu_erf(device)
        observed = result.cpu()
        assert observed.shape == inputs.shape and observed.dtype == torch.float16
        assert torch.isfinite(observed).all()
        torch.testing.assert_close(observed, torch.nn.functional.gelu(inputs), atol=0.008, rtol=0.008)
        changed = torch.ops.rpu.gelu_erf((-inputs).to("rpu")).cpu()
        assert not torch.equal(observed, changed)
        assert torch.equal(result.cpu(), observed)
        assert torch.equal(torch.ops.rpu.gelu_erf(device).cpu(), observed)


def test_gelu_erf_rejects_unsupported_inputs_and_capture():
    for value in (torch.zeros(8), torch.zeros(8, dtype=torch.float16),
                  torch.zeros(8, device="rpu", dtype=torch.float32),
                  torch.zeros(7, device="rpu", dtype=torch.float16),
                  torch.zeros(0, device="rpu", dtype=torch.float16),
                  torch.zeros(8, 8, device="rpu", dtype=torch.float16).T):
        with pytest.raises(RuntimeError, match="gelu_erf requires"):
            torch.ops.rpu.gelu_erf(value)
    cache = torch.rpu.GraphCache(max_entries=1)
    signature = torch.rpu.GraphSignature(op_id="test_gelu_erf", shapes=[8], dtypes=[torch.float16])
    value = torch.zeros(8, device="rpu", dtype=torch.float16)
    try:
        with pytest.raises(RuntimeError, match="PASSTHROUGH only"):
            with cache.capture(signature):
                torch.ops.rpu.gelu_erf(value)
    finally:
        cache.clear()
