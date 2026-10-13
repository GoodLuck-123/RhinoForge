import importlib.util
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CONVERT = (
    ROOT
    / "python"
    / "rpu_backend"
    / "adapters"
    / "lingbot_vla_v2"
    / "convert.py"
)


def _load_convert():
    name = "lingbot2_bias_preflight_convert"
    spec = importlib.util.spec_from_file_location(name, CONVERT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _fixture(prefix, layer_count=36, experts=32):
    values = {
        f"{prefix}layers.{layer}.mlp.e_score_correction_bias":
        torch.zeros(experts, dtype=torch.float32)
        for layer in range(layer_count)
    }
    return values, values.__getitem__


def test_nonzero_correction_bias_is_supported_and_preserved():
    convert = _load_convert()
    values, getter = _fixture(convert.EXP_P)
    key = f"{convert.EXP_P}layers.17.mlp.e_score_correction_bias"
    values[key][5] = 0.125

    biases = convert.load_router_correction_biases(getter)

    assert len(biases) == 36
    assert biases[17][5].item() == pytest.approx(0.125)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_correction_bias_fails_with_layer(bad):
    convert = _load_convert()
    values, getter = _fixture(convert.EXP_P)
    key = f"{convert.EXP_P}layers.35.mlp.e_score_correction_bias"
    values[key][3] = bad

    with pytest.raises(ValueError, match="layer 35 must be finite"):
        convert.load_router_correction_biases(getter)


def test_fp16_overflow_correction_bias_fails_before_upload():
    convert = _load_convert()
    values, getter = _fixture(convert.EXP_P)
    key = f"{convert.EXP_P}layers.0.mlp.e_score_correction_bias"
    values[key][0] = torch.finfo(torch.float32).max

    with pytest.raises(ValueError, match="layer 0 is not representable in FP16"):
        convert.load_router_correction_biases(getter)


def test_nonzero_fp16_underflow_correction_bias_fails_before_upload():
    convert = _load_convert()
    values, getter = _fixture(convert.EXP_P)
    key = f"{convert.EXP_P}layers.4.mlp.e_score_correction_bias"
    values[key][7] = 1.0e-12

    with pytest.raises(ValueError, match="layer 4 contains a nonzero value"):
        convert.load_router_correction_biases(getter)


def test_correction_bias_shape_is_checked_for_every_layer():
    convert = _load_convert()
    values, getter = _fixture(convert.EXP_P)
    key = f"{convert.EXP_P}layers.9.mlp.e_score_correction_bias"
    values[key] = torch.zeros(31, dtype=torch.float32)

    with pytest.raises(ValueError, match=r"layer 9: shape \(31,\) != \(32,\)"):
        convert.load_router_correction_biases(getter)
