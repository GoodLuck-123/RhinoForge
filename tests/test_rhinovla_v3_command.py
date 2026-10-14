"""Public v3 command input and replay behavior, without model assets."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import validate_rhinovla_v3 as command


def _payload():
    return {"metadata": command._identity(command._factory()), "cases": [
        {"qwen_inputs": {"input_ids": torch.tensor([[seed, 1]]),
                         "attention_mask": torch.ones(1, 2, dtype=torch.long),
                         "pixel_values": torch.full((1, 4), float(seed)),
                         "image_grid_thw": torch.tensor([[1, 1, 1]])},
         "request": command._request(seed)} for seed in (0, 1)]}


def test_prepared_requests_are_repeatable_and_separate():
    first, second = command._request(0), command._request(1)
    assert all(torch.equal(value, command._request(0)[name]) for name, value in first.items())
    assert not torch.equal(first["x0"], second["x0"])
    assert not bool(first["x0"][..., 16:].any())
    assert not bool(first["state_mask"][..., 16:].any())


@pytest.mark.parametrize("fault", [None, "identity", "shape", "nan", "geometry"])
def test_fixture_rejects_different_identity_and_invalid_requests(tmp_path, fault):
    payload = _payload()
    if fault == "identity":
        payload["metadata"]["source_commit"] = "different"
    elif fault == "shape":
        payload["cases"][0]["request"]["x0"] = torch.zeros(1, 1, 96)
    elif fault == "nan":
        payload["cases"][0]["request"]["state"][0, 0] = float("nan")
    elif fault == "geometry":
        payload["cases"][1]["qwen_inputs"]["input_ids"] = torch.ones(1, 3)
    path = tmp_path / "inputs.pt"
    torch.save(payload, path)
    if fault:
        with pytest.raises(ValueError):
            command._load_fixture(path, command._factory())
    else:
        loaded = command._load_fixture(path, command._factory())
        assert loaded["metadata"] == payload["metadata"]


@pytest.mark.parametrize("fault", [None, "cache_growth", "no_replay", "output_alias", "close"])
def test_run_uses_all_requests_and_closes_before_publishing(monkeypatch, tmp_path, fault):
    from rpu_backend.api import RhinoVLAPolicy

    payload = _payload()
    fixture, output = tmp_path / "inputs.pt", tmp_path / "actions.pt"
    torch.save(payload, fixture)
    entries, caches = {}, {}
    for index, name in enumerate(command._CHILD_CACHES):
        entry = SimpleNamespace(signature=name, kernel_count=3, segment_count=1,
                                replay_count=0, recapture_count=0,
                                graph_lifetime_id=index + 1, build_generation=1)
        entries[name] = entry
        caches[name] = SimpleNamespace(begin_warmup=lambda: None, freeze=lambda: None,
            cache_invariant_ok=lambda: True, size=lambda: 1,
            snapshot=lambda row=entry: [row])
    runtime = SimpleNamespace(qin={}, **{owner: SimpleNamespace(**{cache: caches[name]})
        for name, (owner, cache) in command._CHILD_CACHES.items()})
    state = {"calls": 0, "closed": False}
    buffer = torch.empty(1, 30, 96)
    def predict(**request):
        state["calls"] += 1
        for entry in entries.values():
            entry.replay_count += int(fault != "no_replay")
        if fault == "cache_growth" and state["calls"] > 1:
            entries["text"].signature = "second-build"
        assert torch.equal(runtime.qin["input_ids"], payload["cases"][int(state["calls"] == 2)]["qwen_inputs"]["input_ids"])
        value = request["x0"].clone()
        if fault == "output_alias":
            buffer.copy_(value)
            return buffer
        return value
    def close():
        state["closed"] = True
        if fault == "close":
            raise RuntimeError("close failed")
    def construct(name, config, **kwargs):
        assert name.endswith(":v3_runtime_factory")
        runtime.qin = {key: value.clone() for key, value in config["qin"].items()}
        return SimpleNamespace(runtime=runtime, predict=predict, close=close)
    monkeypatch.setattr(RhinoVLAPolicy, "from_factory", construct)
    args = SimpleNamespace(config=None, fixture=fixture, output=output, source=tmp_path,
                           checkpoint=tmp_path / "weights.pt", runs=1)
    if fault:
        with pytest.raises(RuntimeError):
            command.run(args)
        assert not output.exists()
    else:
        command.run(args)
        saved = torch.load(output, weights_only=True)
        assert len(saved["actions"]) == 4
        assert torch.equal(saved["actions"][0], saved["actions"][2])
        assert saved["metadata"] == payload["metadata"]
    assert state["closed"]


def test_run_requires_output_before_loading_assets():
    with pytest.raises(SystemExit):
        command.main(["run", "--source", ".", "--checkpoint", "missing", "--fixture", "missing"])
