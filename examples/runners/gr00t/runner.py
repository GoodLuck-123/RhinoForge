"""Public gr00t model construction and input binding."""
from __future__ import annotations

import os

from examples.common.inputs import _load_mapping, _policy_checkpoint as _checkpoint, _model_path

def _gr00t(runner):
    from rpu_backend.adapters.gr00t import build_gr00t_vla
    from examples.runners.gr00t.config import validate_config, validate_observation

    config, inputs = runner.source_config, runner.source_config["input"]
    dtype = config["example"].get("dtype")
    if dtype not in {"fp16", "w8a16"}:
        raise ValueError("GR00T example requires explicit fp16 or w8a16 dtype")
    three_camera_config = validate_config(config)
    observation = _load_mapping(inputs, "observation")
    # Existing benchmark archives wrap the same tensor mapping in inputs/meta.
    # Geometry and text length below are checked against tensors, not metadata.
    if isinstance(observation.get("inputs"), dict):
        observation = observation["inputs"]
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "state"}
    missing = sorted(required.difference(observation))
    if missing:
        raise ValueError(f"GR00T observation is missing fields: {missing}")
    prefix_rows = int(observation["input_ids"].shape[-1])
    max_prefix_len = inputs.get("max_prefix_len", max(256, (prefix_rows + 15) // 16 * 16))
    if (type(max_prefix_len) is not int or not 0 < max_prefix_len <= 512
            or max_prefix_len % 16 or prefix_rows > max_prefix_len):
        raise ValueError("GR00T input.max_prefix_len must cover the observation, be 16-aligned, and be in 16..512")
    qwen3vl = _model_path(
        inputs.get("qwen3vl_registry_alias"),
        label="[input].qwen3vl_registry_alias",
    )
    if three_camera_config:
        from transformers import AutoTokenizer
        validate_observation(config, observation, AutoTokenizer.from_pretrained(str(qwen3vl)))
        import torch
        from rpu_backend.graph import GraphRuntimePolicy
        # Four persistent graphs plus one transient patch-embedding queue.
        if not GraphRuntimePolicy.from_environment(graph_arena_count=5).prepare_arenas():
            raise RuntimeError("GR00T could not prepare five Graph arenas")
        torch.rpu.set_ddr_flush(False)
        torch.rpu.set_ddr_flush_force(True)
    # This selector is read only while installing/swizzling weights. Keep the
    # explicit profile authoritative without changing the caller's environment.
    previous = os.environ.get("RPU_GR00T_W8A16")
    os.environ["RPU_GR00T_W8A16"] = "1" if dtype == "w8a16" else "0"
    try:
        runtime = build_gr00t_vla(
            str(_checkpoint(config)),
            str(qwen3vl),
            embodiment_id=inputs.get("embodiment_id", 20),
            num_steps=inputs.get("num_steps", 4),
            max_prefix_len=max_prefix_len,
            rpu_execution=config.get("rpu_execution"),
        )
    finally:
        if previous is None:
            os.environ.pop("RPU_GR00T_W8A16", None)
        else:
            os.environ["RPU_GR00T_W8A16"] = previous
    runner.owner = runtime
    return lambda: runtime.get_action(
        dict(observation),
        seed=inputs.get("seed", 0),
        num_steps=inputs.get("num_steps", 4),
    )


prepare = _gr00t
