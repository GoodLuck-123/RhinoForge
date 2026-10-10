"""CPU checks for shared example measurement; never initialize the backend."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    if relative.startswith("examples/"):
        return importlib.import_module(relative[:-3].replace("/", "."))
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("inference_mode", [True, False])
@pytest.mark.parametrize("target", ["qwen3", "internvla_n1", "rhinovla"])
def test_setup_keeps_versioned_storage_and_request_mode(monkeypatch, tmp_path, inference_mode, target):
    runner = load("pi_setup_runner_test", "examples/common/measurement.py")
    performance = load("pi_setup_performance_test", "python/rpu_backend/runtime/performance.py")
    common = importlib.import_module("examples.common.config")
    monkeypatch.setattr(common, "cpu_runtime", lambda: load("cold_example_cpu_test", "examples/common/cpu.py"))
    monkeypatch.setattr(common, "example_profiles", lambda: {
        target + ".test": {"id": target + ".test", "support": {"status": "SUPPORTED"}}
    })
    monkeypatch.setitem(sys.modules, "rpu_backend", ModuleType("rpu_backend"))
    monkeypatch.setitem(sys.modules, "rpu_backend.runtime.performance", performance)
    monkeypatch.setattr(torch, "rpu", SimpleNamespace(
        _is_in_bad_fork=lambda: False, manual_seed_all=lambda seed: None), raising=False)
    modes = []
    owner = SimpleNamespace(close=Mock())

    def prepare(_config):
        assert not torch.is_grad_enabled()
        weight = torch.ones(2)
        assert not weight.is_inference()
        version = weight._version
        weight.add_(1)
        assert weight._version == version + 1
        def infer():
            modes.append(torch.is_inference_mode_enabled())
            assert not torch.is_grad_enabled()
            return weight.clone()
        return infer, owner

    monkeypatch.setattr(runner, "_prepare", prepare)
    config = {"example": {"target": target, "profile_id": target + ".test"}, "input": {},
              "run": {"warmup": 1, "runs": 2, "output_dir": str(tmp_path),
                      "inference_mode": inference_mode}}
    assert runner.run(config) == 0
    assert modes == [inference_mode] * 3
    owner.close.assert_called_once()
