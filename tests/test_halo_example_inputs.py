"""HALO resolves a checkpoint file and validates the single-step profile."""
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from examples.common.config import load_config
from examples.runners.halo import runner as example
from rpu_backend.adapters import halo
from rpu_backend.model_registry import model_path


def _fixture():
    cfg = halo.HaloConfig()
    cache = torch.zeros(1).expand(cfg.prefix_len, cfg.num_kv_heads, cfg.head_dim)
    return {
        "x_t": torch.zeros(len(cfg.action_rows), cfg.action_dim),
        "timestep": torch.ones(len(cfg.action_rows)),
        "packed_action_position_ids": torch.arange(len(cfg.action_rows)),
        "packed_text_ids": torch.tensor([1, 2]),
        "packed_position_ids": torch.full((cfg.chunk_size,), 11),
        "packed_text_indexes": torch.tensor(cfg.text_rows),
        "packed_action_token_indexes": torch.tensor(cfg.action_rows),
        "key_values_lens": torch.tensor([cfg.prefix_len]),
        "past_key_values": {
            name: {i: cache for i in range(cfg.num_layers)}
            for name in ("key_cache", "value_cache")
        },
    }


@pytest.fixture
def assets(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("RPU_MODEL_CACHE", str(tmp_path / "models"))
    checkpoint = Path(model_path("halo_action_expert"))
    checkpoint.parent.mkdir(parents=True)
    save_file({"test_weight": torch.zeros(1)}, checkpoint)
    config = load_config(Path(__file__).resolve().parents[1]
                         / "examples/configs/halo/action_expert/fp16.toml")
    fixture = tmp_path / "step.pt"
    config["input"]["fixture"] = str(fixture)
    calls = []
    policy = SimpleNamespace(
        insert_prefix=lambda *args: calls.append(("prefix", args)),
        step=lambda *args: calls.append(("step", args)) or args[0].clone(),
    )

    def build(path, **kwargs):
        calls.append(("build", path, kwargs))
        return policy

    monkeypatch.setattr(halo, "build_halo_action_expert", build)
    return checkpoint, fixture, SimpleNamespace(source_config=config), calls, policy


@pytest.mark.parametrize("wrapped", [False, True])
def test_registry_file_reaches_builder_without_appending_a_directory(assets, wrapped):
    checkpoint, fixture, runner, calls, policy = assets
    values = _fixture()
    torch.save({"kwargs": values} if wrapped else values, fixture)

    step = example.prepare(runner)

    assert checkpoint.parts[-2:] == ("HALO", "action_expert.safetensors")
    assert calls[0] == ("build", checkpoint, {
        "rope_position": 11, "rpu_execution": runner.source_config["rpu_execution"],
    })
    assert runner.owner is policy
    assert len(calls[1][1][0]) == halo.HaloConfig().num_layers
    assert torch.equal(calls[1][1][0][-1], values["past_key_values"]["key_cache"][27])
    assert torch.equal(step(), values["x_t"])
    assert [call[0] for call in calls] == ["build", "prefix", "step"]


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_checkpoint_must_be_a_file(assets, kind):
    checkpoint, fixture, runner, calls, _ = assets
    checkpoint.unlink()
    if kind == "directory":
        checkpoint.mkdir()
        save_file({"test_weight": torch.zeros(1)}, checkpoint / "ema.safetensors")
    torch.save(_fixture(), fixture)
    with pytest.raises(FileNotFoundError, match="HALO checkpoint must name a safetensors file"):
        example.prepare(runner)
    assert calls == []


@pytest.mark.parametrize("change,match", [
    (lambda v: v.update(packed_position_ids=torch.arange(18)), "constant packed_position_ids"),
    (lambda v: v.update(packed_text_indexes=torch.tensor([0, 1])), "text/action rows"),
    (lambda v: v.update(key_values_lens=torch.tensor([1])), "prefix length"),
    (lambda v: v.update(past_key_values=[]), "key_cache/value_cache mapping"),
])
def test_invalid_fixture_fails_before_weight_loading(assets, change, match):
    _, fixture, runner, calls, _ = assets
    values = _fixture()
    change(values)
    torch.save({"kwargs": values}, fixture)
    with pytest.raises(ValueError, match=match):
        example.prepare(runner)
    assert calls == []
