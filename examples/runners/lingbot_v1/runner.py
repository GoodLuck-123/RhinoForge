"""Public lingbot_v1 model construction and input binding."""
from __future__ import annotations


from examples.common.inputs import _load_mapping, _policy_checkpoint as _checkpoint, _model_path

def _lingbot(runner):
    from rpu_backend.adapters.lingbot_vla import build_lingbot_vla

    config, inputs = runner.source_config, runner.source_config["input"]
    observation = _load_mapping(inputs, "observation")
    qwen_alias = inputs.get("qwen_base_registry_alias")
    qwen_base = (
        _model_path(qwen_alias, label="[input].qwen_base_registry_alias")
        if qwen_alias is not None
        else None
    )
    runtime = build_lingbot_vla(
        str(_checkpoint(config)),
        qwen_base=str(qwen_base) if qwen_base else None,
        max_seq=inputs.get("max_seq", 256),
        rpu_execution=config.get("rpu_execution"),
    )
    runner.owner = runtime
    return lambda: runtime.get_action(
        dict(observation), num_steps=inputs.get("num_steps", 10)
    )


prepare = _lingbot
