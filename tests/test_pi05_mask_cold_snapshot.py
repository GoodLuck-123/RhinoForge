"""Pi0.5 mask preparation keeps its environment request local to one owner."""

from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.pi05 import Pi05Adapter


def _policy():
    policy = torch.nn.Module()
    policy.model = torch.nn.Module()
    policy.model.config = SimpleNamespace(
        chunk_size=50, image_features=("camera0", "camera1", "camera2")
    )
    return policy


@pytest.mark.parametrize("cold_request,expected", [(None, False), ("0", False), ("1", True)])
def test_prefill_mask_dtype_request_is_cold_and_default_off(monkeypatch, cold_request, expected):
    name = "RPU_PI05_PREFILL_MASK_FP16"
    if cold_request is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, cold_request)
    owner = _policy()
    Pi05Adapter(owner)
    assert owner.model._pi05_prefill_mask_fp16 is expected

    monkeypatch.setenv(name, str(int(not expected)))
    Pi05Adapter(owner)
    assert owner.model._pi05_prefill_mask_fp16 is expected
    replacement = _policy()
    Pi05Adapter(replacement)
    assert replacement.model._pi05_prefill_mask_fp16 is not expected


def test_invalid_prefill_mask_dtype_request_is_rejected(monkeypatch):
    monkeypatch.setenv("RPU_PI05_PREFILL_MASK_FP16", "invalid")
    owner = _policy()
    with pytest.raises(ValueError, match="RPU_PI05_PREFILL_MASK_FP16"):
        Pi05Adapter(owner)
    assert "_pi05_prefill_mask_fp16" not in vars(owner.model)
