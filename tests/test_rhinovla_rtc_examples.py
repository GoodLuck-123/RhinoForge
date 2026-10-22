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
CONFIG_DIR = ROOT / "examples/configs/rhinovla/v3"
RTC_RECIPES = (
    ("fp16", False, False, False),
    ("w8a16_action", True, False, False),
    ("w8a16_action_adarms", True, True, False),
    ("w8a16_pipeline", True, True, True),
    ("w8a16_pipeline_high_fused_vector_qk", True, True, True),
)


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
    config = {"example": {"dtype": "fp16", "profile_id": "rhinovla.v3.rtc.fp16.trusted-factory"},
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
        for key in ("expert_w8a16", "full_w8a16"):
            if key in config["input"]:
                assert values[key] is config["input"][key]
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


@pytest.mark.parametrize("recipe,expert,full,pipeline", RTC_RECIPES)
def test_rtc_template_preserves_precision_and_cold_configuration(recipe, expert, full, pipeline):
    from examples.common.config import load_config, example_profiles, planner_profiles
    config = load_config(CONFIG_DIR / f"rtc_{recipe}.toml")
    baseline = load_config(CONFIG_DIR / f"{recipe}.toml")
    profile_id = config["example"]["profile_id"]
    assert policies._RHINOVLA_RTC_PROFILES[profile_id] == baseline["example"]["profile_id"]
    assert profile_id not in planner_profiles()
    row = example_profiles()[profile_id]
    assert row["example_only"] is True and "support" not in row
    assert row["input_envelope"]["limits"]["delay_max"] == 6
    assert config["input"]["runtime_factory"] == policies._RHINOVLA_V3_FACTORY
    assert policies._rhinovla_quantization(config) == {
        "expert_w8a16": expert, "full_w8a16": full}
    assert (policies._rhinovla_precision_profile(config)
            in policies._RHINOVLA_FULL_PIPELINE_PROFILES) is pipeline
    assert config["example"]["dtype"] == baseline["example"]["dtype"]
    for key in ("rpu_execution", "cpu", "run"):
        assert config.get(key) == baseline.get(key)
    gates = config["example"]["opt_in"]
    assert gates == baseline["example"]["opt_in"]
    assert gates["RPU_RHINOVLA_PRECOMPUTE_ADARMS"] == "1"
    assert gates["RPU_RHINOVLA_EXPERT_FUSIONS"] == "1"
    assert gates["RPU_RHINOVLA_ADARMS_RESIDENT"] == "1"
    for name in ("PACKED_QKV", "SKIP_ADARMS_GEMV"):
        assert gates["RPU_RHINOVLA_" + name] == "0"


@pytest.mark.parametrize("recipe,expert,full,pipeline", RTC_RECIPES)
def test_each_rtc_profile_binds_quantization_before_installed_storage_check(
        rtc_example, monkeypatch, recipe, expert, full, pipeline):
    from examples.common.config import load_config
    config, _, _, calls, prepare = rtc_example
    factory_path = config["input"]["factory_config"]
    config.clear()
    config.update(load_config(CONFIG_DIR / f"rtc_{recipe}.toml"))
    config["input"]["factory_config"] = factory_path
    inspected = []
    monkeypatch.setattr(policies, "_confirm_rhinovla_quantization",
                        lambda owner, expected, **kwargs: inspected.append((expected, kwargs)))
    infer, _ = prepare()
    assert calls == ["request", "qin", "factory"]
    assert infer()["delay"] == 6
    assert inspected == ([({"expert_w8a16": expert, "full_w8a16": full},
                           {"full_pipeline": pipeline})] if expert else [])


@pytest.mark.parametrize("recipe", [item[0] for item in RTC_RECIPES[1:]])
def test_rtc_w8_scope_cannot_be_changed_by_factory_json(rtc_example, recipe):
    from examples.common.config import load_config
    config, factory, _, calls, prepare = rtc_example
    factory_path = config["input"]["factory_config"]
    config.clear()
    config.update(load_config(CONFIG_DIR / f"rtc_{recipe}.toml"))
    config["input"]["factory_config"] = factory_path
    factory["full_w8a16"] = not config["input"]["full_w8a16"]
    with pytest.raises(ValueError, match="factory_config.full_w8a16 conflicts"):
        prepare()
    assert calls == ["request"]
