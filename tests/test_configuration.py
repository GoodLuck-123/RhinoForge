from __future__ import annotations

from pathlib import Path
import os
import subprocess
import sys
import tomllib

import pytest

from rpu_backend.api._execution import normalize_rpu_execution


ROOT = Path(__file__).resolve().parents[1]


def test_all_example_configs_validate_without_torch_or_native_import(tmp_path):
    """Exercise the same mandatory-config CLI used by the published guides."""
    guard = tmp_path / "guard"
    guard.mkdir()
    for name in ("torch", "rpu_backend"):
        (guard / f"{name}.py").write_text(
            f"raise AssertionError('{name} imported during configuration check')\n"
        )
    env = dict(os.environ, PYTHONPATH=str(guard))
    script = """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from examples.common.cli import main
configs = sorted(Path('examples/configs').rglob('*.toml'))
assert configs
for path in configs:
    assert main(['--config', str(path), '--check-config']) == 0
assert 'torch' not in sys.modules and 'rpu_backend' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", script], cwd=ROOT, env=env,
        text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("configuration OK:") == len(
        list((ROOT / "examples/configs").rglob("*.toml"))
    )


@pytest.mark.parametrize("extra", [
    '\n[runner.env]\nLD_PRELOAD = "unexpected"\n',
    '\n[env]\nRPU_KERNEL_LIB_PATH = "unexpected.ref"\n',
])
def test_current_example_schema_rejects_arbitrary_environment(tmp_path, extra):
    from examples.common.config import ConfigError, load_config

    template = ROOT / "examples/configs/qwen3/text/0_6b/fp16.toml"
    candidate = tmp_path / "invalid.toml"
    candidate.write_text(template.read_text() + extra)
    with pytest.raises((ConfigError, ValueError)):
        load_config(candidate)


def _join(*parts: str) -> str:
    return "".join(parts)


RETIRED_RUNTIME_OPTIONS = {
    "RPU_ALLREDUCE_CHUNK_BALANCED",
    "RPU_ALLREDUCE_CHUNK_V2",
    "RPU_ALLREDUCE_FUSED",
    "RPU_ALLREDUCE_PARTIAL_TWOSTAGE",
    "RPU_ALLREDUCE_RING",
    "RPU_ALLREDUCE_TWOSTAGE",
    "RPU_ALLREDUCE_TWOSTAGE_TRACE",
    "RPU_ALLREDUCE_V2_CAP",
    "RPU_INTERNVLA_N1_EXACT_TWOSTAGE",
    "RPU_LINEAR_AUTOTILE",
    "RPU_LINEAR_AUTOTILE_LIB",
    "RPU_LINEAR_TILE_FORCE",
    "RPU_LINEAR_TILE_GY1",
    "RPU_LINEAR_TILE_M384N",
    "RPU_LINEAR_TILE_M384W",
    "RPU_LINEAR_TILE_M64",
    "RPU_LINEAR_TILE_VIS",
    "RPU_LINEAR_TILE_W8_FORCE",
    "RPU_LINEAR_TILE_W8_M384W",
    "RPU_LINEAR_TILE_W8_M64",
    "RPU_LINEAR_TILE_W8_M64N",
    "RPU_LINEAR_TILE_W8_VIS",
    "RPU_LINGBOT2_DENOISE_TWOSTAGE",
    "RPU_LINGBOT2_DOWN_ACC32_TILE",
    "RPU_LINGBOT2_OUTER_TWOSTAGE",
}


def test_execution_configuration_is_validated_and_frozen() -> None:
    config = normalize_rpu_execution(
        {
            "prefill": {
                "chunk_size": 128,
                "padding_budget": 64,
            }
        },
        entry_point="test",
        supported={"prefill": ("chunk_size", "padding_budget")},
    )
    assert dict(config["prefill"]) == {
        "chunk_size": 128,
        "padding_budget": 64,
    }
    with pytest.raises(TypeError, match="read-only"):
        config["prefill"].chunk_size = 64

    with pytest.raises(ValueError, match="positive multiple of 16"):
        normalize_rpu_execution(
            {"prefill": {"chunk_size": 17}},
            entry_point="test",
        )
    with pytest.raises(ValueError, match="conflicts"):
        normalize_rpu_execution(
            {"prefill": {"padding_rows": 16, "padding_budget": 16}},
            entry_point="test",
        )


def test_linear_acc32_requires_boolean_and_declared_owner_capability() -> None:
    supported = {
        "prefill": ("chunk_size", "linear_acc32"),
        "vision": ("chunk_size", "linear_acc32"),
    }
    config = normalize_rpu_execution(
        {"prefill": {"linear_acc32": True}, "vision": {"linear_acc32": False}},
        entry_point="test", supported=supported,
    )
    assert config["prefill"]["linear_acc32"] is True
    assert config["vision"]["linear_acc32"] is False

    with pytest.raises(TypeError, match="must be bool"):
        normalize_rpu_execution(
            {"prefill": {"linear_acc32": 1}},
            entry_point="test", supported=supported,
        )
    with pytest.raises(ValueError, match="explicit supported"):
        normalize_rpu_execution(
            {"vision": {"linear_acc32": True}}, entry_point="test",
        )
    with pytest.raises(ValueError, match="not supported"):
        normalize_rpu_execution(
            {"components": {"vision_encoder": {
                "vision": {"linear_acc32": True},
            }}},
            entry_point="test",
            supported_components={"vision_encoder": {"vision": ("chunk_size",)}},
        )


def test_example_toml_files_are_portable() -> None:
    config_dir = ROOT / "examples" / "configs"
    configs = sorted(config_dir.rglob("*.toml"))
    assert configs
    banned = (
        _join("/", "n", "fs"),
        _join("/", "da", "ta", "/"),
        _join("10", ".", "10", "."),
        _join("192", ".", "168", "."),
        _join("h", "f", "_"),
        _join("tok", "en", "="),
    )
    for path in configs:
        text = path.read_text(encoding="utf-8")
        parsed = tomllib.loads(text)
        assert parsed
        assert not any(value in text for value in banned), path


def test_example_configs_exclude_retired_runtime_options() -> None:
    for path in (ROOT / "examples" / "configs").rglob("*.toml"):
        text = path.read_text(encoding="utf-8")
        found = {name for name in RETIRED_RUNTIME_OPTIONS if name in text}
        assert not found, (path, found)


def test_qwen3_8b_profile_uses_auto_prefill_chunk() -> None:
    profile = tomllib.loads(
        (ROOT / "examples" / "configs" / "qwen3/text/8b/fp16.toml").read_text(
            encoding="utf-8"
        )
    )
    assert profile["rpu_execution"]["prefill"]["chunk_size"] == "auto"


def _policy_with_runtime_env(name: str, runtime_env):
    from rpu_backend.api import (
        HyEmbodiedPolicy,
        Lingbot2Policy,
        WallOssPolicy,
    )

    if name == "lingbot2":
        return Lingbot2Policy.from_checkpoint(
            "unused/checkpoint", runtime_env=runtime_env
        )
    if name == "wall_oss":
        return WallOssPolicy.from_checkpoint(
            "unused/checkpoint", runtime_env=runtime_env
        )
    return HyEmbodiedPolicy.from_checkpoint(
        "unused/checkpoint", runtime_env=runtime_env
    )


@pytest.mark.parametrize(
    "loader",
    [
        pytest.param("causal", id="causal"),
        pytest.param("conditional", id="conditional"),
    ],
)
@pytest.mark.parametrize("value", [True, None, 0, "false"])
def test_generic_loaders_reject_custom_model_code(loader: str, value) -> None:
    if loader == "causal":
        from rpu_backend import RPUModelForCausalLM as model_loader
    else:
        from rpu_backend.api import (
            RPUModelForConditionalGeneration as model_loader,
        )

    with pytest.raises(ValueError, match="trust_remote_code=False"):
        model_loader.from_pretrained(
            "unused/checkpoint", trust_remote_code=value
        )
