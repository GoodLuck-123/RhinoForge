"""Compatibility for the previous public example entry points."""
from __future__ import annotations

from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
DEFAULTS = {
    "run_model": "qwen3/text/0_6b/fp16.toml",
    "causal_lm": "qwen3/text/0_6b/fp16.toml",
    "qwen3": "qwen3/text/0_6b/fp16.toml",
    "llama": "llama/3_2_1b/fp16.toml",
    "dinov3": "dinov3/vit_b/fp16.toml",
    "g05": "g05/base/fp16.toml",
    "gemma4": "gemma4/e4b/fp16.toml",
    "gr00t": None,
    "halo": "halo/action_expert/fp16.toml",
    "hy_embodied": "hy_embodied/umi/fp16.toml",
    "hy_vla": "hy_embodied/umi/fp16.toml",
    "internvla_navdp": None,
    "lingbot2": "lingbot/legacy/exact_z2/6b_w8a16.toml",
    "pi05": "pi05/2cam/fp16.toml",
    "qwen3_5_text": "qwen3_5/text/0_8b/fp16.toml",
    "qwen3_5_vision": "qwen3_5/vl/2b/fp16.toml",
    "qwen3_vl": "qwen3_vl/vl/2b/fp16.toml",
    "rhinovla": "rhinovla/v1/fp16.toml",
    "siglip": None,
    "wall_oss": "wall_oss/legacy/fp16.toml",
}
LIST_TARGETS = tuple(sorted(set(DEFAULTS) - {"run_model", "causal_lm", "hy_embodied"}))
TARGET_ALIASES = {
    "causal_lm": {"qwen3", "llama"},
    "hy_embodied": {"hy_vla"},
    "internvla_navdp": {"internvla_n1"},
    "qwen3_5_text": {"qwen3_5"},
    "wall_oss": {"wall_oss", "wall_oss_sorting", "wall_oss_rtc"},
}
PATH_ALIASES = {
    "lingbot/v2/6b_w8a16.toml": "lingbot/legacy/exact_z2/6b_w8a16.toml",
    "rhinovla/v3/w8a16.toml": "rhinovla/v3/expert_w8a16.toml",
    **{f"wall_oss/base/{name}.toml": f"wall_oss/legacy/{name}.toml"
       for name in ("fp16", "w4a16", "w4a16_pgrp", "w8a16")},
}
MIGRATIONS = {
    "gr00t": (
        "The old GR00T DROID/Cosmos request is not the three-camera example. "
        "Follow examples/configs/gr00t/README.md and pass --config explicitly. "
        "For a matching supported checkpoint, map request.input_pt to input.observation "
        "and request.seed/num_steps to input.seed/num_steps; prepare the required camera "
        "and state tensors. Keep the checkpoint and embodiment identity; renaming fields "
        "alone does not convert the old four-image request."
    ),
    "internvla_navdp": (
        "The old NavDP embeddings-only request is not a full-chain input. "
        "Follow examples/configs/internvla/README.md and pass --config explicitly. "
        "Prepare input.fixture and input.asset_manifest for the full chain; preserve "
        "request.sample_num/seed as input.sample_num/seed. Do not rename the old "
        "embeddings file or component manifest as full-chain assets."
    ),
    "siglip": (
        "The standalone SigLIP component configuration has no equivalent full-policy "
        "template. Keep component callers on the SigLIP adapter, or explicitly prepare "
        "a complete Pi0.5 batch using examples/configs/pi05/README.md and "
        "examples/run.py --config PATH. An image list alone is not a policy batch."
    ),
    "lingbot_fp16": (
        "The old LingBot FP16 image/proprioception profile needs explicit input migration. "
        "Use examples/configs/lingbot/v2/README.md to select cameras and language length; "
        "provide input.checkpoint, input.training_config and input.case. Preserve the "
        "checkpoint, camera order, action dimensions and normalization; do not map it "
        "to the separate exact-Z2 W8 profile."
    ),
    "moe_synthetic": (
        "The old MoE token-filler workload needs explicit input migration. Use "
        "examples/configs/qwen3_5/text/35b_a3b/mixed_e4m3_p4096_d4096.toml: "
        "replace prompt_token_ids/prefill_fill_token_id with prompt_file, prompt_sha256 "
        "and input_ids_sha256, and set run.timing_mode='forward_argmax'. Keep the same "
        "checkpoint and explicit prefill/decode lengths; the text content changes."
    ),
}
MIGRATIONS = {name: message + " Migration command: python examples/migrate_config.py --help."
              for name, message in MIGRATIONS.items()}
_G05_IDENTITY = {
    "source": "OpenGalaxea/G05",
    "revision": "e312be81e90c56a55bcb26b57429bd39a335b449",
    "subfolder": "g05-base",
    "implementation": "OpenGalaxea/GalaxeaVLA",
    "implementation_revision": "89f2322b4ad016e192437adc1a2c253b05bab246",
    "config_sha256": "c98af37352c0f600341d2ecbdb49fcdfa87812198654448991615065fa1a4461",
    "license": "G0.5 Community License (non-commercial)",
}
G05_INTEGRATION = (
    "Construct the official CPU G05PolicyQwen35 from OpenGalaxea/GalaxeaVLA at "
    "revision 89f2322b4ad016e192437adc1a2c253b05bab246, then call "
    "rpu_backend.adapters.g05.patch_g05_policy_for_rpu(policy, max_seq_len=2048) "
    "and invoke its simultaneous continuous/discrete/CoT request path. "
    "This source-integration configuration does not select the newer action-only "
    "runner. See examples/configs/g05/README.md."
)


def is_runner_config(config):
    runner = config.get("runner")
    return isinstance(runner, dict) and "target" in runner and "example" not in config


def migration_error(config):
    """Return an actionable error only for a known incompatible old schema."""
    if is_runner_config(config):
        target = config["runner"]["target"]
        if not isinstance(target, str) or not target:
            return "[runner].target must be a nonempty string"
        return MIGRATIONS.get(target)
    example = config.get("example", {})
    if not isinstance(example, dict):
        return None
    if example.get("profile_id") == "lingbot-vla-v2-6b.fp16.source-only":
        return MIGRATIONS["lingbot_fp16"]
    inputs = config.get("input", {})
    if (example.get("architecture") == "Qwen3_5MoeForConditionalGeneration"
            and isinstance(inputs, dict) and "prefill_tokens" in inputs
            and {"prompt_token_ids", "prefill_fill_token_id"} & inputs.keys()):
        return MIGRATIONS["moe_synthetic"]
    return None


def resolve_config_path(path):
    """Resolve removed repository templates without replacing caller files."""
    path = Path(path).expanduser()
    if path.exists() or path.is_symlink():
        return path
    try:
        relative = path.resolve().relative_to(CONFIGS.resolve()).as_posix()
    except ValueError:
        return path
    target = PATH_ALIASES.get(relative)
    return CONFIGS / target if target else path


def missing_config_message(path):
    try:
        relative = Path(path).resolve().relative_to(CONFIGS).as_posix()
    except ValueError:
        return None
    key = {"lingbot/v2/6b_fp16.toml": "lingbot_fp16",
           "siglip/pi05_component/fp16.toml": "siglip"}.get(relative)
    return MIGRATIONS.get(key)


def validate_g05_integration(config):
    """Preserve the former source-only configuration check, without importing Torch."""
    if set(config) - {"runner", "model", "logging"}:
        raise ValueError("G0.5 integration accepts only [runner], [model] and [logging]")
    model, runner = config.get("model"), config.get("runner")
    if not isinstance(model, dict) or not isinstance(runner, dict):
        raise ValueError("G0.5 integration requires [model] and [runner]")
    if runner.get("target") != "g05" or set(runner) - {"target", "env", "torch_profile"}:
        raise ValueError("G0.5 integration requires runner.target='g05'")
    if not any(isinstance(model.get(k), str) and model[k] for k in ("alias", "checkpoint")):
        raise ValueError("G0.5 [model] needs alias or checkpoint")
    for name, value in _G05_IDENTITY.items():
        if model.get(name) != value:
            raise ValueError(f"G0.5 [model].{name} must retain its original integration identity")
    if model.get("max_seq_len", 2048) != 2048 or any(
            model.get(name) is not True for name in
            ("continuous_action", "discrete_action", "predict_cot", "source_only_acknowledged")):
        raise ValueError("G0.5 integration requires max_seq_len=2048 and its original request modes")
    env = runner.get("env", {})
    if not isinstance(env, dict) or set(env) - {"RPU_LOG_LEVEL"}:
        raise ValueError("G0.5 integration [runner.env] only accepts RPU_LOG_LEVEL")
    if "RPU_LOG_LEVEL" in env and (str(env["RPU_LOG_LEVEL"]) not in {str(i) for i in range(6)}):
        raise ValueError("RPU_LOG_LEVEL must be an integer from 0 through 5")
    profile = runner.get("torch_profile", {})
    if not isinstance(profile, dict) or set(profile) - {
            "enabled", "output", "record_shapes", "profile_memory", "with_stack"}:
        raise ValueError("unsupported [runner.torch_profile] setting")
    for name in ("enabled", "record_shapes", "profile_memory", "with_stack"):
        if type(profile.get(name, False)) is not bool:
            raise ValueError(f"[runner.torch_profile].{name} must be boolean")
    if not isinstance(profile.get("output", "profiles/torch_trace.json"), str) or not profile.get(
            "output", "profiles/torch_trace.json"):
        raise ValueError("[runner.torch_profile].output must be a path string")
    return config


def main(entry, argv=None):
    from examples.common.cli import main as run

    default = DEFAULTS[entry]
    return run(argv, default_config=CONFIGS / default if default else None,
               legacy_entry=entry, keep_logging_until_exit=True)
