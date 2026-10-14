#!/usr/bin/env python
'Gemma2 all layers once device regression.'

import os

import pytest
import torch


@pytest.mark.skipif(
    os.environ.get("RPU_HW_AVAILABLE") != "1",
    reason="requires device hardware (set RPU_HW_AVAILABLE=1)",
)
def test_gemma2_ops_registered_and_handle_lifecycle():
    import rpu_backend  # noqa: F401

    for op_name in (
        "gemma2_create",
        "gemma2_destroy",
        "gemma2_set_weights",
        "gemma2_forward",
    ):
        assert hasattr(torch.ops.rpu, op_name), (
            f"torch.ops.rpu.{op_name} not registered"
        )

    handle = torch.ops.rpu.gemma2_create()
    try:
        assert isinstance(handle, int)
        assert handle >= 0
    finally:
        torch.ops.rpu.gemma2_destroy(handle)
