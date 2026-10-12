"""Public lingbot2 model construction and input binding."""
from __future__ import annotations

from pathlib import Path

from examples.common.inputs import _load_mapping, _required_file, _controlled_images as _images, _controlled_checkpoint as _checkpoint


def validate_legacy_config(config):
    inputs = config["input"]
    for key in ("qwen3vl_base", "training_config"):
        if not isinstance(inputs.get(key), str) or not inputs[key].strip():
            raise ValueError(f"[input].{key} must be a non-empty path")


def _lingbot2(runner):
    config, inputs = runner.source_config, runner.source_config["input"]
    validate_legacy_config(config)
    training_config = _required_file(inputs, "training_config")
    processor = Path(inputs["qwen3vl_base"]).expanduser()
    if not processor.is_dir() or not (processor / "config.json").is_file():
        raise FileNotFoundError(f"[input].qwen3vl_base requires a local processor directory with config.json: {processor}")
    if "observation" in inputs:
        request = {"obs": _load_mapping(inputs, "observation")}
    else:
        request = {"images": _images(inputs), "instruction": inputs["instruction"],
                   "proprioception": inputs["proprioception"],
                   "noise_seed": inputs.get("seed", 0)}
    from rpu_backend.api import Lingbot2Policy

    policy = Lingbot2Policy.from_checkpoint(
        _checkpoint(config), dtype="w8a16",
        qwen3vl_base=processor,
        training_config_path=training_config,
        runtime_env=config["example"].get("opt_in"),
        rpu_execution=config.get("rpu_execution"),
    ).to("rpu")
    runner.owner = policy
    policy.prepare_graphs(
        **request, prefix_length_range=inputs.get("prefix_length_range", [215, 225]))
    return lambda: policy.infer(**request).actions_full


prepare = _lingbot2
