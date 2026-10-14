"""Pure-CPU checks for LingBot2 physical-action de-normalization."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch


MODULE = (Path(__file__).resolve().parents[2] / "python" / "rpu_backend"
          / "api" / "action_denorm.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("lingbot2_action_denorm_cpu", MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_horizon_specific_stats_use_last_axis_as_action_dim(tmp_path):
    mod = _load_module()
    stats = {
        "norm_stats": {
            "action.arm": {
                "mean": [[1.0, 2.0], [3.0, 4.0]],
                "std": [[0.5, 1.0], [1.5, 2.0]],
            }
        }
    }
    path = tmp_path / "norm.json"
    path.write_text(json.dumps(stats))

    denorm = mod.ActionDenormalizer.from_norm_json(str(path))
    assert denorm.real_dim == 2
    got = denorm.unapply(torch.zeros(2, 2))
    torch.testing.assert_close(got, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_delta_group_requires_and_adds_physical_state(tmp_path):
    mod = _load_module()
    path = tmp_path / "norm.json"
    path.write_text(json.dumps({"action.arm": {"mean": [0.0, 0.0], "std": [1.0, 1.0]}}))
    group = mod.ActionGroup("action.arm", 2, "identity", subtract_state=True)
    denorm = mod.ActionDenormalizer.from_spec(str(path), [group])

    got = denorm.unapply(torch.tensor([[0.25, -0.5]]), state=torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(got, torch.tensor([[1.25, 1.5]]))


@pytest.mark.parametrize("dim", [0, -1, 1.5, True])
def test_group_dim_must_be_a_positive_integer(tmp_path, dim):
    mod = _load_module()
    path = tmp_path / "norm.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="positive integer"):
        mod.ActionDenormalizer.from_spec(
            str(path), [mod.ActionGroup("action.arm", dim, "identity")]
        )


@pytest.mark.parametrize(
    ("mean", "match"),
    [
        ([3.14], "action dim 1 != configured group dim 16"),
        ([float("nan")] * 16, "contains non-finite values"),
        ([[[0.0] * 16]], "must be \\[dim\\] or \\[horizon, dim\\]"),
    ],
)
def test_stats_cannot_broadcast_or_contain_nonfinite_values(tmp_path, mean, match):
    mod = _load_module()
    path = tmp_path / "norm.json"
    path.write_text(json.dumps({
        "action.arm": {"mean": mean, "std": [1.0] * 16},
    }))
    with pytest.raises(ValueError, match=match):
        mod.ActionDenormalizer.from_spec(
            str(path), [mod.ActionGroup("action.arm", 16, "meanstd")]
        )


def test_float32_physical_overflow_fails_closed(tmp_path):
    mod = _load_module()
    path = tmp_path / "norm.json"
    path.write_text(json.dumps({
        "action.arm": {"mean": [1e100, 1e100], "std": [1.0, 1.0]},
    }))
    denorm = mod.ActionDenormalizer.from_spec(
        str(path), [mod.ActionGroup("action.arm", 2, "meanstd")]
    )
    with pytest.raises(ValueError, match="physical action contains non-finite"):
        denorm.unapply(torch.zeros(1, 2))


def test_training_config_must_name_each_group_norm_type(tmp_path):
    mod = _load_module()
    path = tmp_path / "norm.json"
    path.write_text(json.dumps({"action.arm": {"mean": [0.0], "std": [1.0]}}))
    training = {"data": {"joints": [{"arm": 1}], "norm_type": []}}
    with pytest.raises(ValueError, match="missing explicit norm_type"):
        mod.ActionDenormalizer.from_training_config(str(path), training)
