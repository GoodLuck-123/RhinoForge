"""Cold checkpoint layouts feed the same biased SwiGLU projections."""
import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from rpu_backend.adapters.wall_oss.vision import (
    _SafeTensorStore, _load_vision_gate_up, _pad_1d, _pad_cols, _pad_rows,
    _split_gate_up,
)


PREFIX = "visual.blocks.0.mlp"


def _weights():
    rng = torch.Generator().manual_seed(25)
    return {name: torch.randn(shape, generator=rng).half() for name, shape in {
        "gate_proj.weight": (16, 8), "up_proj.weight": (16, 8),
        "gate_proj.bias": (16,), "up_proj.bias": (16,),
        "down_proj.weight": (8, 16), "down_proj.bias": (8,),
    }.items()}


def _store(tmp_path, values):
    save_file({f"{PREFIX}.{name}": value for name, value in values.items()},
              str(tmp_path / "model.safetensors"))
    return _SafeTensorStore(str(tmp_path))


@pytest.mark.parametrize("layout", ["separate", "fused", "padded_fused"])
def test_original_and_wall_layouts_preserve_swiglu_and_source(tmp_path, layout):
    original = _weights()
    values = dict(original)
    if layout != "separate":
        for suffix in ("weight", "bias"):
            halves = [values.pop(f"{name}_proj.{suffix}") for name in ("gate", "up")]
            if layout == "padded_fused":
                halves = [F.pad(value, (0, 0, 0, 16)) if suffix == "weight"
                          else F.pad(value, (0, 16)) for value in halves]
            values[f"gate_up_proj.{suffix}"] = torch.cat(halves)
    store = _store(tmp_path, values)
    before = (tmp_path / "model.safetensors").read_bytes()
    weight, bias = _load_vision_gate_up(store, PREFIX, 8, 16, w8a16=False)
    gate, up = _split_gate_up(weight, 16, 32, "weight")
    gate_bias, up_bias = _split_gate_up(bias, 16, 32, "bias")
    assert torch.equal(gate[:16], original["gate_proj.weight"])
    assert torch.equal(up[:16], original["up_proj.weight"])
    x = torch.arange(24, dtype=torch.float64).reshape(3, 8) / 24
    expected = F.linear(
        F.silu(F.linear(x, original["gate_proj.weight"].double(), original["gate_proj.bias"].double()))
        * F.linear(x, original["up_proj.weight"].double(), original["up_proj.bias"].double()),
        original["down_proj.weight"].double(), original["down_proj.bias"].double())
    actual = F.linear(
        F.silu(F.linear(x, _pad_rows(gate, 32).double(), _pad_1d(gate_bias, 32).double()))
        * F.linear(x, _pad_rows(up, 32).double(), _pad_1d(up_bias, 32).double()),
        _pad_cols(original["down_proj.weight"], 32).double(), original["down_proj.bias"].double())
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
    assert (tmp_path / "model.safetensors").read_bytes() == before


@pytest.mark.parametrize("fault", ["missing_up", "missing_bias", "weight_shape", "bias_shape", "integer"])
def test_incomplete_or_malformed_separate_layout_is_rejected(tmp_path, fault):
    values = _weights()
    if fault == "missing_up":
        del values["up_proj.weight"]
    elif fault == "missing_bias":
        del values["gate_proj.bias"]
    elif fault == "weight_shape":
        values["up_proj.weight"] = values["up_proj.weight"][:-1]
    elif fault == "bias_shape":
        values["up_proj.bias"] = values["up_proj.bias"][:-1]
    else:
        values["up_proj.weight"] = values["up_proj.weight"].to(torch.int8)
    with pytest.raises((KeyError, ValueError)):
        _load_vision_gate_up(_store(tmp_path, values), PREFIX, 8, 16, w8a16=False)


def test_quantized_layout_keeps_existing_fused_contract(tmp_path):
    with pytest.raises(KeyError, match="gate_up_proj.weight"):
        _load_vision_gate_up(_store(tmp_path, _weights()), PREFIX, 8, 16, w8a16=True)
