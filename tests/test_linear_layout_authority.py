"""Host checks of converter-bound eager layout, including non-A9 owners."""
import copy
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.runtime import weights


def test_valid_recorded_layout_reaches_native_and_survives_deepcopy(monkeypatch):
    import rpu_backend

    layer = torch.nn.Linear(16, 16, bias=False).half()
    monkeypatch.setattr(weights, "_eager_dispatch_layout", lambda *_: (1, 8))
    weights._record_linear_layout(layer, 0, 8)
    calls = []
    result = torch.ones(1, 16, dtype=torch.float16)
    monkeypatch.setattr(rpu_backend, "_cpp_ext", SimpleNamespace(
        linear_with_partition=lambda *args: calls.append(args) or result), raising=False)
    value = SimpleNamespace(device=SimpleNamespace(type="rpu"), dtype=torch.float16,
                            dim=lambda: 2, size=lambda _: 16, contiguous=lambda: result)
    copied = copy.deepcopy(layer)
    for owner in (layer, copied):
        assert owner(value) is result
        assert calls[-1][1] is owner.weight and calls[-1][3] == 0
    assert len(calls) == 2


@pytest.mark.parametrize("layout", [(weights.SKIPPED_PARTITION, None), (1, 4), (0, 8)])
@pytest.mark.parametrize("fault", ["partition", "cores", "bool", "delete"])
def test_layout_mirror_cannot_change_native_dispatch(monkeypatch, layout, fault):
    import rpu_backend

    layer = torch.nn.Linear(16, 16, bias=False).half()
    original = layer.weight.detach().clone()
    # This boundary is the same converter record used by non-A9 Rhino/QwenPi05.
    monkeypatch.setattr(weights, "_eager_dispatch_layout", lambda *_: (1, 8))
    weights._record_linear_layout(layer, *layout)
    calls = []
    monkeypatch.setattr(rpu_backend, "_cpp_ext", SimpleNamespace(
        linear_with_partition=lambda *args: calls.append(args)), raising=False)
    if fault == "partition":
        layer._rpu_linear_partition = 1 - layout[0]
    elif fault == "cores":
        layer._rpu_linear_num_cores = 8 if layout[1] != 8 else 4
    elif fault == "bool":
        layer._rpu_linear_partition = bool(layout[0])
    else:
        del layer._rpu_linear_num_cores
    with pytest.raises(RuntimeError, match="recorded swizzle layout changed"):
        layer(SimpleNamespace(device=SimpleNamespace(type="rpu")))
    assert not calls and torch.equal(layer.weight, original)
    # CPU use and deepcopy are retained, including companion-owned raw weights.
    x = torch.ones(1, 16, dtype=torch.float16)
    assert torch.equal(layer(x), torch.nn.functional.linear(x, original))
    assert torch.equal(copy.deepcopy(layer)(x), layer(x))
