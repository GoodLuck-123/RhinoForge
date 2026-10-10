"""Caller-owned inputs shared by the model runners; imports stay hardware-lazy."""
from __future__ import annotations

from pathlib import Path

def _required_file(inputs, key):
    value = inputs.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"[input].{key} must name an existing caller-owned file")
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"[input].{key}: {path}")
    return path

def _load_mapping(inputs, key):
    import torch

    path = _required_file(inputs, key)
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"{path} must be a tensor-only mapping loadable with weights_only=True"
        ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a mapping")
    return value

def _images(inputs):
    from PIL import Image

    paths = inputs.get("images")
    if not isinstance(paths, list) or not paths:
        raise ValueError("[input].images must contain at least one image path")
    result = []
    for value in paths:
        path = _required_file({"image": value}, "image")
        with Image.open(path) as image:
            result.append(image.convert("RGB"))
    return result

def _policy_checkpoint(config):
    from rpu_backend.model_registry import model_path

    checkpoint = config["input"].get("checkpoint")
    return Path(checkpoint).expanduser() if checkpoint else Path(model_path(config["example"]["registry_alias"]))

def _model_path(alias, *, label):
    if not isinstance(alias, str) or not alias:
        raise ValueError(f"{label} must be a non-empty model-registry alias")
    from rpu_backend.model_registry import model_path

    return Path(model_path(alias))

def _controlled_images(inputs):
    from PIL import Image

    paths = inputs.get("images")
    if not isinstance(paths, list) or not paths:
        raise ValueError("[input].images must contain at least one image path")
    result = []
    for value in paths:
        path = _required_file({"images": value}, "images")
        with Image.open(path) as image:
            image = image.convert("RGB")
            if "image_size" in inputs:
                image = image.resize((inputs["image_size"], inputs["image_size"]))
            result.append(image)
    return result

def _controlled_checkpoint(config):
    if "_model_override" in config:
        from examples.common.config import planner_profiles

        example = config["example"]
        profile = planner_profiles().get(example.get("profile_id"), {})
        if (example["target"] != "qwen3_5_vision"
                or profile.get("family") != "qwen3-5-moe"
                or profile.get("component") != "multimodal"):
            raise ValueError("model path override requires a Qwen3.5 MoE image profile")
        return Path(config["_model_override"])
    override = config.get("input", {}).get("checkpoint")
    if override is not None:
        if config["example"]["target"] not in {"wall_oss_rtc", "g05"}:
            raise ValueError("checkpoint override is only supported for exact Wall RTC/G0.5 releases")
        if not isinstance(override, str) or not override:
            raise ValueError("[input].checkpoint must name an existing checkpoint directory")
        path = Path(override).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"[input].checkpoint: {path}")
        return path
    from rpu_backend.model_registry import model_path

    return Path(model_path(config["example"]["registry_alias"]))
