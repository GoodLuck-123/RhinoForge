"""Legacy migration preserves requests and requires new input identities."""
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from examples.common.config import ConfigError, load_config
from examples.common.legacy import PATH_ALIASES
from examples.common.migration import dumps, main, migrate, migration_kind

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "examples/configs"
CASES = (
    ("gr00t", "gr00t/n1_7_droid/224/fp16.toml", '[runner]\ntarget="gr00t"\n'
     '[request]\ninput_pt="old-four-images.pt"\nseed=17\nnum_steps=10\n',
     {"observation": "new-three-images.pt"}),
    ("internvla_navdp", "internvla/navdp/w8a16_system2.toml", '[runner]\ntarget="internvla_navdp"\n'
     '[model]\nasset_manifest_json="old-components.json"\n'
     '[request]\ninput_pt="old-embeddings.pt"\nseed=19\nsample_num=3\n',
     {"fixture": "full-chain.pt", "asset_manifest": "full-chain.json"}),
    ("siglip", "pi05/2cam/fp16.toml", '[runner]\ntarget="siglip"\n'
     '[model]\nalias="pi05-libero-finetuned"\n[request]\nimages=["image.ppm"]\n',
     {"batch": "complete-batch.pt", "checkpoint": "policy-checkpoint"}),
    ("lingbot_fp16", "lingbot/v2/3cam_t32/fp16.toml",
     '[example]\nprofile_id="lingbot-vla-v2-6b.fp16.source-only"\n'
     '[input]\nimages=["left.ppm","right.ppm","wrist.ppm"]\n'
     'instruction="Pick up the block."\nproprioception=[0.1,0.2]\nseed=23\n'
     '[run]\nwarmup=2\nruns=7\nseed=0\n',
     {"checkpoint": "original-checkpoint", "training_config": "training.yaml"}),
    ("moe_synthetic", "qwen3_5/text/35b_a3b/fp8_e4m3_experts_attn_gdn_p4096_d4096.toml", None, {}),
)


@pytest.mark.parametrize("old", [old for old, new in PATH_ALIASES.items()
                                 if not new.startswith("gr00t/")])
def test_removed_template_alias_reaches_migration_contract(tmp_path, capsys, old):
    source = tmp_path / "old.toml"
    source.write_text('[runner]\ntarget="gr00t"\n')
    output = tmp_path / "new.toml"
    with pytest.raises(SystemExit) as error:
        main(["--source", str(source), "--template", str(CONFIGS / old), "--output", str(output)])
    assert error.value.code == 2
    assert "gr00t migration requires a current gr00t template" in capsys.readouterr().err
    assert not output.exists()


@pytest.mark.parametrize("old", [old for old, new in PATH_ALIASES.items()
                                 if new.startswith("gr00t/")])
def test_renamed_gr00t_template_is_read_through_migration_command(tmp_path, old):
    source = tmp_path / "old.toml"
    source.write_text('[runner]\ntarget="gr00t"\n[request]\nseed=17\nnum_steps=10\n')
    output = tmp_path / "new.toml"
    assert main(["--source", str(source), "--template", str(CONFIGS / old),
                 "--output", str(output), "--input", "observation=observation.pt"]) == 0
    migrated = load_config(output)
    template = load_config(CONFIGS / PATH_ALIASES[old])
    assert migrated["example"] == template["example"]
    assert migrated["input"]["image_size"] == template["input"]["image_size"]
    assert migrated["input"]["observation"] == "observation.pt"


def old_config(kind, text, template):
    if text is not None:
        return tomllib.loads(text)
    data = deepcopy(template)
    data["input"] = {"prompt_token_ids": [1, 2], "prefill_fill_token_id": 3,
                     "prefill_tokens": 4096, "decode_steps": 4096, "stop_on_eos": False}
    data["run"].update(runs=1, torch_num_threads=24)
    data.pop("cpu", None)
    return data


@pytest.mark.parametrize("kind,path,text,inputs", CASES)
def test_migrated_config_retains_request_and_validates_without_runtime(tmp_path, kind, path, text, inputs):
    template_path = CONFIGS / path
    template = tomllib.loads(template_path.read_text())
    source = old_config(kind, text, template)
    inputs = dict(inputs)
    if kind == "moe_synthetic":
        prompt = tmp_path / "prompt.txt"
        prompt.write_text("A caller-owned complete prompt.\n")
        inputs.update(prompt_file=str(prompt), input_ids_sha256="1" * 64)
    before = deepcopy(source)
    converted = migrate(source, template, inputs)
    assert source == before
    source_path = tmp_path / "old.toml"
    source_path.write_text(dumps(source))
    output = tmp_path / "current.toml"
    block_runtime = '''
import importlib.abc, runpy, sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'rpu_backend'}:
            raise AssertionError('migration imported ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    args = [sys.executable, "-S", "-c", block_runtime, str(ROOT / "examples/migrate_config.py"),
            "--source", str(source_path), "--template", str(template_path), "--output", str(output)]
    for key, value in inputs.items():
        args += ["--input", f"{key}={value}"]
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    actual = tomllib.loads(output.read_text())
    assert actual == converted
    load_config(output)
    assert "request" not in actual and "runner" not in actual
    if kind == "gr00t":
        assert actual["input"]["seed"] == 17 and actual["input"]["num_steps"] == 10
        assert actual["input"]["observation"] != before["request"]["input_pt"]
    elif kind == "internvla_navdp":
        assert actual["input"]["seed"] == 19 and actual["input"]["sample_num"] == 3
        assert actual["input"]["asset_manifest"] != before["model"]["asset_manifest_json"]
    elif kind == "siglip":
        assert actual["input"]["batch"] == "complete-batch.pt"
        assert "images" not in actual["input"]
    elif kind == "lingbot_fp16":
        assert actual["input"]["images"] == before["input"]["images"]
        assert actual["input"]["state"] == [0.1, 0.2]
        assert actual["input"]["instruction"] == before["input"]["instruction"]
        assert actual["run"]["seed"] == 23 and actual["run"]["runs"] == 7
        assert "expected_language_tokens" not in actual["input"]
    else:
        assert actual["input"]["prompt_sha256"] == hashlib.sha256(prompt.read_bytes()).hexdigest()
        assert actual["input"]["input_ids_sha256"] == "1" * 64
        assert actual["input"]["prefill_tokens"] == actual["input"]["decode_steps"] == 4096
        assert actual["run"]["timing_mode"] == "forward_argmax"
        assert actual["run"]["runs"] == 1 and actual["cpu"]["threads"] == 24


@pytest.mark.parametrize("kind,path,text,inputs", CASES)
def test_new_input_requirements_cannot_be_satisfied_by_template_placeholders(kind, path, text, inputs):
    template = tomllib.loads((CONFIGS / path).read_text())
    with pytest.raises(ConfigError, match="provide --input"):
        migrate(old_config(kind, text, template), template, {})


@pytest.mark.parametrize("table", ["runner", "example", "input", "run", "rpu_execution"])
def test_malformed_legacy_sections_are_configuration_errors(table):
    with pytest.raises(ConfigError, match="must be a table"):
        migration_kind({table: []})


def test_failure_leaves_source_and_existing_output_unchanged(tmp_path):
    old = tmp_path / "old.toml"
    old.write_text('[runner]\ntarget="gr00t"\n')
    output = tmp_path / "new.toml"
    args = ["--source", str(old), "--template", str(CONFIGS / "gr00t/n1_7_droid/224/fp16.toml"),
            "--output", str(output)]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2 and not output.exists()
    output.write_text("keep this file\n")
    with pytest.raises(SystemExit) as error:
        main(args + ["--input", "observation=new.pt"])
    assert error.value.code == 2 and output.read_text() == "keep this file\n"
    assert old.read_text() == '[runner]\ntarget="gr00t"\n'


def test_wrong_target_and_changed_moe_identity_are_rejected():
    with pytest.raises(ConfigError, match="current gr00t template"):
        migrate({"runner": {"target": "gr00t"}}, {"example": {"target": "qwen3"}}, {})
    template = tomllib.loads((CONFIGS / CASES[-1][1]).read_text())
    old = old_config("moe_synthetic", None, template)
    old["example"]["registry_alias"] = "another-checkpoint"
    with pytest.raises(ConfigError, match="retain the checkpoint profile"):
        migrate(old, template, {})
