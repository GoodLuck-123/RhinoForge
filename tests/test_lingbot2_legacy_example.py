"""Legacy policy inputs must reach construction before device work."""

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.common.config import ConfigError, load_config
from examples.common.migration import dumps
from examples.runners.lingbot2 import runner


def template():
    return load_config(ROOT / "examples/configs/lingbot/legacy/exact_z2/6b/w8a16.toml")


@pytest.mark.parametrize("key", ["qwen3vl_base", "training_config"])
@pytest.mark.parametrize("value", [None, "", " ", False, 1])
def test_required_input_paths_are_checked_without_model_loading(tmp_path, key, value):
    config = template()
    if value is None:
        config["input"].pop(key)
    else:
        config["input"][key] = value
    path = tmp_path / "config.toml"
    path.write_text(dumps(config))
    with pytest.raises(ConfigError, match=key):
        load_config(path)


@pytest.mark.parametrize("missing", ["training_config", "processor_config"])
def test_missing_assets_fail_before_loading_observation_or_policy(tmp_path, monkeypatch, missing):
    config = template()
    config["input"].update(qwen3vl_base=str(tmp_path), training_config=str(tmp_path / "training.yaml"))
    if missing != "training_config":
        (tmp_path / "training.yaml").write_text("model: {}")
    if missing != "processor_config":
        (tmp_path / "config.json").write_text("{}")
    observation = Mock(side_effect=AssertionError("input loading must wait for asset preflight"))
    monkeypatch.setattr(runner, "_load_mapping", observation)
    with pytest.raises(FileNotFoundError, match="training_config|qwen3vl_base"):
        runner.prepare(SimpleNamespace(source_config=config))
    observation.assert_not_called()


def test_observation_and_paired_assets_reach_policy_unchanged(tmp_path, monkeypatch):
    config = template()
    config["input"].update(qwen3vl_base=str(tmp_path), training_config=str(tmp_path / "training.yaml"))
    (tmp_path / "training.yaml").write_text("model: {}")
    (tmp_path / "config.json").write_text("{}")
    observation = {"noise": object()}
    monkeypatch.setattr(runner, "_load_mapping", lambda *args: observation)
    monkeypatch.setattr(runner, "_checkpoint", lambda config: tmp_path / "checkpoint")
    policy = Mock()
    policy.to.return_value = policy
    output = object()
    policy.infer.return_value = SimpleNamespace(actions_full=output)
    constructor = Mock(return_value=policy)
    monkeypatch.setitem(sys.modules, "rpu_backend.api", SimpleNamespace(Lingbot2Policy=SimpleNamespace(from_checkpoint=constructor)))
    owner = SimpleNamespace(source_config=config)

    infer = runner.prepare(owner)

    constructor.assert_called_once_with(
        tmp_path / "checkpoint", dtype="w8a16", qwen3vl_base=tmp_path,
        training_config_path=tmp_path / "training.yaml",
        runtime_env=config["example"]["opt_in"], rpu_execution=config["rpu_execution"],
    )
    policy.prepare_graphs.assert_called_once_with(obs=observation, prefix_length_range=[215, 225])
    assert owner.owner is policy
    assert infer() is output
    policy.infer.assert_called_once_with(obs=observation)
