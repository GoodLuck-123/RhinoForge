"""Full-storage zero and partial views preserve values and in-place versions."""
import pytest
import torch

import rpu_backend  # noqa: F401


VIEWS = {
    "full": lambda x: x,
    "transpose": lambda x: x.transpose(0, 1),
    "offset": lambda x: x.view(-1)[3:99],
    "strided": lambda x: x[:, ::3],
    "empty_view": lambda x: x[1:1],
    "empty_storage": lambda x: x.new_empty((0, 16)),
}


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("view_name", VIEWS)
def test_zero_preserves_storage_neighbours_and_alias_versions(inference, view_name):
    with torch.inference_mode(inference):
        expected = torch.arange(256, dtype=torch.float16).reshape(8, 32)
        actual = expected.to("rpu")
        cpu_view, rpu_view = VIEWS[view_name](expected), VIEWS[view_name](actual)
        version = None if inference else rpu_view._version
        alias = rpu_view.view_as(rpu_view)
        cpu_view.zero_()
        returned = rpu_view.zero_()
        assert returned is rpu_view
        assert torch.equal(actual.cpu(), expected)
        assert torch.equal(rpu_view.cpu(), cpu_view)
        if not inference:
            assert rpu_view._version == version + 1
            assert alias._version == rpu_view._version
