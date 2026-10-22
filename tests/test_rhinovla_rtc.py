"""CPU contracts for the bounded RTC request and source timestep semantics."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PATH = Path(__file__).resolve().parents[1] / "python/rpu_backend/adapters/rhinovla/rtc.py"
spec = importlib.util.spec_from_file_location("rhino_rtc_contract", PATH)
rtc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rtc)


@pytest.mark.parametrize("delay", range(7))
def test_each_delay_preserves_masked_owned_fp32_prefix(delay):
    prefix = torch.randn(delay, 96)
    saved = prefix.clone()
    mask = torch.ones(1, 96)
    mask[:, 7::8] = 0
    result = rtc.normalize_rtc_request(
        rtc=True, action_prefix=prefix, delay=delay,
        x0=torch.zeros(1, 30, 96), state=torch.zeros(1, 96),
        state_mask=None, action_mask=mask)
    assert result.shape == (1, 30, 96)
    assert torch.equal(result[:, :delay], saved[None] * mask[:, None])
    assert not result[:, delay:].count_nonzero()
    prefix.fill_(100)
    assert torch.equal(result[:, :delay], saved[None] * mask[:, None])


@pytest.mark.parametrize("delay,error", [(-1, ValueError), (7, ValueError),
    (True, TypeError), (1.0, TypeError), (torch.tensor(1), TypeError), (None, ValueError)])
def test_delay_rejects_unsupported_values(delay, error):
    with pytest.raises(error):
        rtc.validate_rtc_request(rtc=True, action_prefix=torch.zeros(6, 96), delay=delay)


@pytest.mark.parametrize("prefix", [torch.zeros(2, 6, 96), torch.zeros(5, 96),
    torch.zeros(31, 96), torch.zeros(6, 95), torch.full((6, 96), float("nan")),
    torch.full((6, 96), float("inf"))])
def test_prefix_rejects_invalid_shape_or_nonfinite_values(prefix):
    with pytest.raises(ValueError):
        rtc.validate_rtc_request(rtc=True, action_prefix=prefix, delay=6)


def test_cold_mode_and_missing_pair_are_explicit():
    rtc.validate_rtc_request(rtc=False)
    with pytest.raises(ValueError, match="require"):
        rtc.validate_rtc_request(rtc=False, delay=0)
    with pytest.raises(ValueError, match="both"):
        rtc.validate_rtc_request(rtc=True, delay=0)
    with pytest.raises(TypeError, match="floating CPU"):
        rtc.validate_rtc_request(rtc=True, action_prefix=torch.zeros(6, 96, dtype=torch.long), delay=6)


def test_nonuniform_action_masks_and_overflow_reject_before_device_work():
    request = dict(rtc=True, action_prefix=torch.ones(6, 96), delay=6,
                   x0=torch.zeros(1, 30, 96), state=torch.zeros(1, 96),
                   state_mask=None, action_mask=torch.ones(1, 30, 96))
    request["action_mask"][:, 1, 0] = 0
    with pytest.raises(ValueError, match="shared"):
        rtc.normalize_rtc_request(**request)
    request["action_mask"] = None
    request["action_prefix"].fill_(1e6)
    with pytest.raises(ValueError, match="FP16"):
        rtc.normalize_rtc_request(**request)


def test_groups_match_upstream_token_embedding_then_state_mean():
    def embedding(t, width):
        return torch.cat((torch.sin(t[:, None] * torch.arange(1, width // 2 + 1)),
                          torch.cos(t[:, None] * torch.arange(1, width // 2 + 1))), dim=-1)
    times = torch.tensor([1.0, 0.7, 0.1])
    groups = rtc.rtc_condition_groups(times, embedding, 8)
    for delay in range(7):
        for step, time in enumerate(times):
            token_times = torch.full((30,), time)
            token_times[:delay] = 0
            upstream = embedding(token_times, 8)
            reconstructed = torch.cat((groups[delay, step, :1],
                groups[delay, step, 1:2].expand(delay, -1),
                groups[delay, step, 2:3].expand(30 - delay, -1)))
            assert torch.equal(reconstructed, torch.cat((upstream.mean(0, keepdim=True), upstream)))
    assert not torch.allclose(groups[6, 0, 0], embedding(torch.tensor([0.8]), 8)[0])


@pytest.mark.parametrize("expert,full", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("resident,fusions", [(False, False), (True, False), (True, True)])
def test_install_accepts_precision_and_grouped_optimization_paths(expert, full, resident, fusions):
    flags = {name: False for name in ("expert_fusions", "adarms_resident", "packed_qkv",
              "skip_adarms_gemv", "precompute_time_proj", "fold_action_time_in")}
    flags["precompute_adarms"] = True
    flags.update(adarms_resident=resident, expert_fusions=fusions)
    args = dict(rtc=True, steps=10, flow_direction="official_descending",
                expert_w8a16=expert, full_expert_w8a16=full,
                native_cold=SimpleNamespace(**flags), action_cold={"denoise_unroll": True})
    rtc.validate_rtc_install(**args)
    for name, value in (("steps", 5),
                        ("flow_direction", "legacy_ascending"),
                        ("action_cold", {"denoise_unroll": False})):
        with pytest.raises(ValueError):
            rtc.validate_rtc_install(**{**args, name: value})
    for flag in ("precompute_adarms", "packed_qkv", "skip_adarms_gemv",
                 "precompute_time_proj", "fold_action_time_in"):
        changed = {**flags, flag: not flags[flag]}
        with pytest.raises(ValueError):
            rtc.validate_rtc_install(**{**args, "native_cold": SimpleNamespace(**changed)})


@pytest.mark.parametrize("flag", ["HIGH_PRECISION", "HIGH_PRECISION_FUSIONS",
    "PRECOMPUTE_GATE_TANH", "VECTOR_K_NORM", "VECTOR_Q_NORM", "PARTIAL_ROPE"])
def test_factory_admits_rtc_math_options_before_asset_or_weight_loading(monkeypatch, flag):
    from rpu_backend.adapters.rhinovla import factory
    def must_not_load(*args, **kwargs):
        pytest.fail("RTC rejected configuration reached weight/asset loading")
    monkeypatch.setattr(factory, "_validate_official_assets", must_not_load)
    monkeypatch.setattr(factory, "_load_official_model", must_not_load)
    for suffix in ("PRECOMPUTE_ADARMS", "GATED_NO_SUB", "ADARMS_RESIDENT",
                   "EXPERT_FUSIONS", "ALIGNED_KV", "EXPERT_W8A16", "FULL_W8A16"):
        monkeypatch.setenv("RPU_RHINOVLA_" + suffix, "1")
    monkeypatch.setenv("RPU_RHINOVLA_DENOISE_UNROLL", "1")
    for op in ("rhino_vla_set_denoise_loop_rtc_tables", "rhino_vla_prepare_rtc",
               "rhino_vla_denoise_loop_rtc_forward", "rhino_vla_set_denoise_loop_rtc_tables_w8a16"):
        monkeypatch.setattr(torch.ops.rpu, op, lambda *a: None, raising=False)
    for suffix in ("HIGH_PRECISION", "HIGH_PRECISION_FUSIONS", "PRECOMPUTE_GATE_TANH",
                   "VECTOR_K_NORM", "VECTOR_Q_NORM", "PARTIAL_ROPE"):
        monkeypatch.setenv("RPU_RHINOVLA_" + suffix, "1" if suffix == flag else "0")
    # Invalid qin must be caught next, rather than an obsolete FP16/math guard.
    with pytest.raises(ValueError, match="qin is missing"):
        factory.v3_runtime_factory({"rtc": True, "trust_model_code": True,
            "model_root": "/nonexistent/source", "checkpoint": "/nonexistent/model.pt",
            "qin": {}, "steps": 10})


def test_factory_rejects_forced_oneshot_before_loading(monkeypatch):
    from rpu_backend.adapters.rhinovla import factory
    monkeypatch.setenv("RPU_GRAPH_FORCE_ONESHOT_ON_REPLAY", "1")
    with pytest.raises(ValueError, match="forced oneshot"):
        factory.v3_runtime_factory({"rtc": True, "trust_model_code": True,
            "model_root": "/nonexistent/source", "checkpoint": "/nonexistent/model.pt",
            "qin": {}, "steps": 10})


@pytest.mark.parametrize("expert,full", [(False, False), (True, False), (True, True)])
def test_factory_rejects_fp16_only_native_before_loading(monkeypatch, expert, full):
    from rpu_backend.adapters.rhinovla import factory
    for suffix, value in (("PRECOMPUTE_ADARMS", "1"), ("DENOISE_UNROLL", "1"),
                          ("EXPERT_W8A16", "0"), ("FULL_W8A16", "0")):
        monkeypatch.setenv("RPU_RHINOVLA_" + suffix, value)
    for op in ("rhino_vla_set_denoise_loop_rtc_tables", "rhino_vla_prepare_rtc",
               "rhino_vla_denoise_loop_rtc_forward"):
        monkeypatch.setattr(torch.ops.rpu, op, lambda *a: None, raising=False)
    monkeypatch.delattr(torch.ops.rpu, "rhino_vla_set_denoise_loop_rtc_tables_w8a16", raising=False)
    monkeypatch.setattr(factory, "_validate_official_assets", lambda *a, **kw: pytest.fail("loaded assets"))
    with pytest.raises(RuntimeError, match="matching native"):
        factory.v3_runtime_factory({"rtc": True, "trust_model_code": True,
            "model_root": "/nonexistent/source", "checkpoint": "/nonexistent/model.pt",
            "qin": {}, "steps": 10, "expert_w8a16": expert, "full_w8a16": full})
