"""Legacy policy inputs must reach construction before device work."""

from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.common.config import load_config
from examples.runners.lingbot2 import runner


def template():
    return load_config(ROOT / "examples/configs/lingbot/legacy/exact_z2/6b/w8a16.toml")


def test_observation_reaches_policy_with_template_defaults(tmp_path, monkeypatch):
    config = template()
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
        tmp_path / "checkpoint", dtype="w8a16",
        runtime_env=config["example"]["opt_in"], rpu_execution=config["rpu_execution"],
    )
    policy.prepare_graphs.assert_called_once_with(obs=observation, prefix_length_range=[215, 225])
    assert owner.owner is policy
    assert infer() is output
    policy.infer.assert_called_once_with(obs=observation)
