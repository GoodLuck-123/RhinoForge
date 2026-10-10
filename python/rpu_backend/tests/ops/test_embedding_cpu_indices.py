"""Host token IDs avoid a device round trip without changing embedding results."""
import ctypes

import pytest
import torch
import torch.nn.functional as F

import rpu_backend  # noqa: F401


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("empty", [False, True])
def test_cpu_ids_and_dirty_rpu_ids_match_embedding_and_retain_outputs(dtype, empty):
    weight = torch.arange(17 * 32, dtype=torch.float16).reshape(17, 32) / 32
    device_weight = weight.to("rpu")
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=dtype)[:, ::2]
    if empty:
        ids = ids[:, :0]
    output = F.embedding(ids, device_weight)
    saved = F.embedding(ids, weight)
    assert output.device.type == "rpu"
    assert torch.equal(output.cpu(), saved)
    ids.add_(7)
    changed = F.embedding(ids, device_weight)
    assert torch.equal(changed.cpu(), F.embedding(ids, weight))
    assert torch.equal(output.cpu(), saved)
    if not empty:
        # Public device IDs can be modified through a shared-DDR CPU alias that
        # bypasses Tensor versions and copy_ publication. Keep that boundary.
        public_ids = ids.contiguous().to("rpu")
        replacement = (ids - 7).contiguous()
        ctypes.memmove(public_ids.data_ptr(), replacement.data_ptr(), replacement.nbytes)
        assert torch.equal(F.embedding(public_ids, device_weight).cpu(), saved)


def test_cpu_embedding_preserves_index_errors():
    weight = torch.ones(4, 32, dtype=torch.float16).to("rpu")
    for value in (-1, 4):
        with pytest.raises((IndexError, RuntimeError), match="index|range"):
            F.embedding(torch.tensor([value]), weight)
    with pytest.raises(RuntimeError, match="Long|Int|indices"):
        F.embedding(torch.tensor([1.0]), weight)
