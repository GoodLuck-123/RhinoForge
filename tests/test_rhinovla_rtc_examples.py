"""Public RTC examples reject mismatched requests before loading model inputs."""
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from examples.runners.rhinovla import runner as policies

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def rtc_example(monkeypatch, tmp_path):
    name = "rpu_backend.adapters.rhinovla.rtc"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "python/rpu_backend/adapters/rhinovla/rtc.py")
    rtc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rtc)
    monkeypatch.setitem(sys.modules, name, rtc)
    api = ModuleType("rpu_backend.api")
    monkeypatch.setitem(sys.modules, "rpu_backend.api", api)
    config = {"example": {"dtype": "fp16", "profile_id": policies._RHINOVLA_RTC_PROFILE},
              "input": {"runtime_factory": policies._RHINOVLA_V3_FACTORY,
                        "factory_config": str(tmp_path / "factory.json"),
                        "request": "rtc_request.pt", "checkpoint": "weights.pt",
                        "trust_model_code": True, "num_steps": 10},
              "rpu_execution": {}}
    factory = {"rtc": True, "qin": "qin.pt"}
    request = {"x0": torch.ones(1, 30, 96), "delay": 6,
               "action_prefix": torch.zeros(1, 6, 96)}
    calls = []
    def load_mapping(_config, key):
        calls.append(key)
        return request if key == "request" else {"input_ids": torch.zeros(1, 1)}
    def construct(name, values, **kwargs):
        calls.append("factory")
        assert name == policies._RHINOVLA_V3_FACTORY
        assert values["rtc"] is True
        return SimpleNamespace(predict=lambda **payload: payload, close=lambda: None)
    monkeypatch.setattr(policies, "_load_mapping", load_mapping)
    monkeypatch.setattr(policies, "_checkpoint", lambda _: Path("weights.pt"))
    api.RhinoVLAPolicy = SimpleNamespace(from_factory=construct)
    def prepare():
        Path(config["input"]["factory_config"]).write_text(json.dumps(factory))
        policies.validate_rhinovla_config(config)
        owner = SimpleNamespace(source_config=config)
        return policies._factory(owner), owner
    return config, factory, request, calls, prepare


@pytest.mark.parametrize("delay", range(7))
def test_rtc_example_passes_each_admitted_delay_before_construction(rtc_example, delay):
    _, _, request, calls, prepare = rtc_example
    request.update(delay=delay, action_prefix=torch.zeros(1, delay, 96))
    infer, owner = prepare()
    assert calls == ["request", "qin", "factory"]
    assert owner.owner is not None
    actual = infer()
    assert actual["delay"] == delay
    assert actual["action_prefix"] is request["action_prefix"]


@pytest.mark.parametrize("fault", ["missing_mode", "false_mode", "integer_mode",
    "wrong_profile", "missing_prefix", "missing_delay", "bool_delay", "tensor_delay",
    "delay7", "short_prefix", "nonfinite_prefix", "wrong_factory"])
def test_rtc_example_rejects_before_loading_qin_and_model(rtc_example, fault):
    config, factory, request, calls, prepare = rtc_example
    if fault == "missing_mode":
        del factory["rtc"]
    elif fault == "false_mode":
        factory["rtc"] = False
    elif fault == "integer_mode":
        factory["rtc"] = 1
    elif fault == "wrong_profile":
        config["example"]["profile_id"] = "rhinovla.v3.fp16.trusted-factory"
    elif fault == "missing_prefix":
        del request["action_prefix"]
    elif fault == "missing_delay":
        del request["delay"]
    elif fault == "bool_delay":
        request["delay"] = True
    elif fault == "tensor_delay":
        request["delay"] = torch.tensor(6)
    elif fault == "delay7":
        request["delay"] = 7
    elif fault == "short_prefix":
        request["action_prefix"] = torch.zeros(1, 5, 96)
    elif fault == "nonfinite_prefix":
        request["action_prefix"][0, 0, 0] = float("nan")
    elif fault == "wrong_factory":
        config["input"]["runtime_factory"] = "caller.integration:create_runtime"
    with pytest.raises((TypeError, ValueError)):
        prepare()
    assert "qin" not in calls and "factory" not in calls


def test_nonrtc_example_rejects_rtc_request_before_qin(rtc_example):
    config, factory, _, calls, prepare = rtc_example
    config["example"]["profile_id"] = "rhinovla.v3.fp16.trusted-factory"
    factory["rtc"] = False
    with pytest.raises(ValueError, match="require a v3 RTC runtime"):
        prepare()
    assert calls == ["request"]


def test_rtc_template_has_separate_identity_and_passes_config_validation():
    from examples.common.config import load_config, example_profiles, planner_profiles
    config = load_config(ROOT / "examples/configs/rhinovla/v3/rtc_fp16.toml")
    profile_id = config["example"]["profile_id"]
    assert profile_id == policies._RHINOVLA_RTC_PROFILE
    assert profile_id not in planner_profiles()
    row = example_profiles()[profile_id]
    assert row["example_only"] is True and "support" not in row
    gates = config["example"]["opt_in"]
    assert gates["RPU_RHINOVLA_PRECOMPUTE_ADARMS"] == "1"
    for name in ("EXPERT_FUSIONS", "ADARMS_RESIDENT", "FULL_W8A16",
                 "EXPERT_W8A16", "PACKED_QKV", "SKIP_ADARMS_GEMV"):
        assert gates["RPU_RHINOVLA_" + name] == "0"
