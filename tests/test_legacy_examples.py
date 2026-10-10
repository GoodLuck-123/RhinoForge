"""The previous CLI paths retain their dispatch and explicit migration boundaries."""
from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest

from examples.common import cli, legacy
from examples.common.config import ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "examples/configs"
ENTRIES = (
    "run_model", "causal_lm", "qwen3", "llama", "dinov3", "g05", "gemma4",
    "gr00t", "halo", "hy_embodied", "hy_vla", "internvla_navdp", "lingbot2",
    "pi05", "qwen3_5_text", "qwen3_5_vision", "qwen3_vl", "rhinovla", "siglip",
    "wall_oss",
)


def _cli_without_runtime(entry, *args):
    # Exercise the files users invoke, with imports blocked even if the test
    # environment has already loaded Torch in its own parent process.
    script = r'''
import importlib.abc, runpy, sys
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'rpu_backend'}:
            raise AssertionError('configuration command imported ' + fullname)
sys.meta_path.insert(0, NoRuntime())
sys.argv = sys.argv[1:]
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as error:
    code = error.code
else:
    code = 0
assert 'torch' not in sys.modules and 'rpu_backend' not in sys.modules
raise SystemExit(code)
'''
    return subprocess.run([sys.executable, "-S", "-c", script,
                           str(ROOT / f"examples/{entry}.py"), *args],
                          cwd=ROOT, text=True, capture_output=True, timeout=15)


@pytest.mark.parametrize("entry", ENTRIES)
def test_legacy_entry_default_config_or_specific_migration_without_runtime(entry):
    result = _cli_without_runtime(entry, "--check-config")
    if entry in {"gr00t", "internvla_navdp", "siglip"}:
        assert result.returncode == 2
        assert legacy.MIGRATIONS[entry] in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert "configuration OK:" in result.stdout


def test_run_model_lists_original_targets_without_config_or_runtime():
    result = _cli_without_runtime("run_model", "--list-targets")
    assert result.returncode == 0, result.stderr
    assert set(result.stdout.splitlines()) == {
        "qwen3", "llama", "hy_vla", "halo", "dinov3", "g05", "gemma4", "gr00t",
        "internvla_navdp", "lingbot2", "pi05", "qwen3_5_text", "qwen3_5_vision",
        "qwen3_vl", "rhinovla", "siglip", "wall_oss",
    }


@pytest.mark.parametrize("entry,config", [
    ("gr00t", "gr00t/n1_7_droid/fp16.toml"),
    ("internvla_navdp", "internvla/navdp/fp16.toml"),
    ("g05", "g05/3cam/fp16.toml"),
])
def test_explicit_current_config_can_use_legacy_entry(entry, config):
    result = _cli_without_runtime(entry, "--config", str(CONFIGS / config), "--check-config")
    assert result.returncode == 0, result.stderr


def test_legacy_dispatch_preserves_user_config_options(monkeypatch, tmp_path):
    from examples.common import logging

    config = tmp_path / "user.toml"
    config.write_text((CONFIGS / "qwen3/text/0_6b/fp16.toml").read_text().replace(
        "runs = 3", "runs = 7"))
    seen = []
    monkeypatch.setattr(logging, "logging_session", lambda *a, **k: nullcontext())
    monkeypatch.setattr(cli, "_dispatch", lambda config, args: seen.append((config, args)) or 17)
    assert legacy.main("qwen3", ["--config", str(config), "--profile"]) == 17
    parsed, args = seen.pop()
    assert parsed["run"]["runs"] == 7 and parsed["run"]["profile"] is True
    assert args.config == config
    assert parsed["example"]["registry_alias"] == "qwen3-0.6b"
    with pytest.raises(SystemExit) as error:
        legacy.main("qwen3", ["--config", str(CONFIGS / "llama/3_2_1b/fp16.toml")])
    assert error.value.code == 2 and seen == []


@pytest.mark.parametrize("old,new", legacy.PATH_ALIASES.items())
def test_old_paths_preserve_canonical_profile_and_precision(old, new):
    assert tomllib.loads((CONFIGS / old).read_text()) == tomllib.loads((CONFIGS / new).read_text())
    assert load_config(CONFIGS / old) == load_config(CONFIGS / new)


@pytest.mark.parametrize("old_target,required", [
    ("gr00t", ("request.input_pt", "input.observation", "four-image")),
    ("internvla_navdp", ("input.fixture", "input.asset_manifest", "embeddings")),
    ("siglip", ("SigLIP adapter", "complete Pi0.5 batch")),
])
def test_non_equivalent_old_runner_configs_require_explicit_migration(tmp_path, old_target, required):
    path = tmp_path / "copied.toml"
    path.write_text(f'[runner]\ntarget = "{old_target}"\n[model]\ncheckpoint = "/inputs/checkpoint"\n')
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert all(item in str(error.value) for item in required)


@pytest.mark.parametrize("target", ['[]', '{}', 'false', '4', '""'])
def test_invalid_legacy_target_reports_configuration_error(tmp_path, target):
    path = tmp_path / "invalid.toml"
    path.write_text(f"[runner]\ntarget = {target}\n")
    with pytest.raises(ConfigError, match=r"\[runner\].target must be a nonempty string"):
        load_config(path)
    result = _cli_without_runtime("run_model", "--config", str(path), "--check-config")
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
    assert "[runner].target must be a nonempty string" in result.stderr


@pytest.mark.parametrize("config,required", [
    ('[example]\nprofile_id = "lingbot-vla-v2-6b.fp16.source-only"\n',
     ("training_config", "input.case", "action dimensions")),
    ('[example]\narchitecture = "Qwen3_5MoeForConditionalGeneration"\n'
     '[input]\nprefill_tokens = 4096\nprompt_token_ids = [1, 2]\n',
     ("prompt_file", "prompt_sha256", "input_ids_sha256", "forward_argmax")),
])
def test_non_equivalent_profile_and_workload_errors_include_field_mapping(tmp_path, config, required):
    path = tmp_path / "copied.toml"
    path.write_text(config)
    with pytest.raises(ConfigError) as error:
        load_config(path)
    assert all(item in str(error.value) for item in required)


@pytest.mark.parametrize("old_path,match", [
    ("lingbot/v2/6b_fp16.toml", "training_config"),
    ("siglip/pi05_component/fp16.toml", "SigLIP component"),
])
def test_removed_non_equivalent_paths_explain_migration(old_path, match):
    with pytest.raises(ConfigError, match=match):
        load_config(CONFIGS / old_path)


def test_g05_source_integration_checks_identity_and_never_dispatches_action_runner(monkeypatch, tmp_path):
    path = CONFIGS / "g05/base/fp16.toml"
    config = load_config(path)
    assert config["model"]["continuous_action"] is True
    assert config["model"]["discrete_action"] is True
    assert config["model"]["predict_cot"] is True
    monkeypatch.setattr(cli, "_g05_runner", lambda: pytest.fail("source integration chose action-only runner"))
    assert legacy.main("g05", ["--check-config"]) == 0
    with pytest.raises(SystemExit) as error:
        legacy.main("g05", [])
    assert error.value.code == 2
    for old, new in [('predict_cot = true', 'predict_cot = false'),
                     ('OpenGalaxea/G05', 'another/model'),
                     ('RPU_LOG_LEVEL = 3', 'OTHER_ENV = "1"')]:
        bad = tmp_path / "bad.toml"
        bad.write_text(path.read_text().replace(old, new))
        with pytest.raises(ConfigError):
            load_config(bad)


def test_rhinovla_v1_factory_profile_and_official_profile_remain_distinct():
    caller = load_config(CONFIGS / "rhinovla/v1/fp16.toml")
    official = load_config(CONFIGS / "rhinovla/v1/official_fp16.toml")
    assert caller["example"]["profile_id"] == "rhinovla.v1.fp16.trusted-factory"
    assert "runtime_factory" in caller["input"]
    assert official["example"]["profile_id"] == "rhinovla.fp16.official-nonrtc"
    assert "runtime_factory" not in official["input"]


def test_rhinovla_v1_invokes_exact_caller_factory_without_new_optimization_flags(monkeypatch, tmp_path):
    import rpu_backend.api
    from examples.runners.rhinovla import runner

    config = load_config(CONFIGS / "rhinovla/v1/fp16.toml")
    factory_path = tmp_path / "factory.json"
    factory_path.write_text(json.dumps({"custom_setting": 9}))
    config["input"].update(runtime_factory="caller.policy:create_runtime",
                           factory_config=str(factory_path), request="/inputs/request.pt")
    before = deepcopy(config)
    request = {"caller_tensor": object()}
    calls = []
    policy = SimpleNamespace(predict=lambda **kwargs: kwargs)
    class Factory:
        @classmethod
        def from_factory(cls, name, options, **kwargs):
            calls.append((name, options, kwargs))
            return policy
    monkeypatch.setattr(rpu_backend.api, "RhinoVLAPolicy", Factory)
    monkeypatch.setattr(runner, "_load_mapping", lambda inputs, key: request)
    monkeypatch.setattr(runner, "_checkpoint", lambda config: Path("/inputs/checkpoint"))
    monkeypatch.setattr(runner, "_official", lambda _: pytest.fail("changed caller factory to official runtime"))
    context = SimpleNamespace(source_config=config, owner=None)
    infer = runner.prepare(context)
    assert context.owner is policy and infer() == request
    name, options, kwargs = calls.pop()
    assert name == "caller.policy:create_runtime"
    assert options == {"custom_setting": 9, "checkpoint": "/inputs/checkpoint", "steps": 10,
                       "trust_model_code": True}
    assert kwargs == {"rpu_execution": config["rpu_execution"]}
    assert config == before
