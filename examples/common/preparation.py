"""Dispatch the supported targets to model-owned construction."""
from importlib import import_module
from types import SimpleNamespace


def prepare(config):
    from examples.common.measurement import _close

    target = config["example"]["target"]
    text = target in {"qwen3", "llama", "qwen3_5", "qwen3_vl", "dinov3", "gemma4"}
    if target == "qwen3_5_vision":
        module_name = "qwen3_5.vision"
    elif target == "wall_oss_rtc":
        module_name = "wall_oss.rtc"
    elif target == "internvla_n1":
        module_name = "internvla.runner"
    elif target == "lingbot":
        module_name = "lingbot_v1.runner"
    else:
        module_name = target + ".runner"
    module = import_module("examples.runners." + module_name)
    runner = SimpleNamespace(source_config=config, owner=None)
    policy = target in {"wall_oss", "gr00t", "hy_vla", "lingbot", "internvla_n1"} or (
        target == "rhinovla" and "runtime_factory" in config["input"])
    try:
        if text:
            return module.prepare(config, runner)
        infer = module.prepare(runner)
        return infer, runner.owner
    except BaseException as error:
        try:
            _close(runner.owner, target)
        except BaseException as cleanup_error:
            label = "policy construction" if policy else "model setup"
            error.add_note(f"{label} cleanup failed: {cleanup_error!r}")
        raise
