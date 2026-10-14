"""Opt-in replay of a hashed, single-image Qwen3-VL preprocessed input."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import MethodType, SimpleNamespace


def load_prepared_input(inputs):
    """Hash and parse the same bytes before checkpoint/model construction."""
    import torch
    from safetensors.torch import load

    path = Path(inputs["prepared_input_safetensors"])
    content = path.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != inputs["prepared_input_sha256"]:
        raise ValueError("prepared Qwen3-VL input SHA256 does not match prepared_input_sha256")
    try:
        tensors = load(content)
    except Exception as error:
        raise ValueError("prepared Qwen3-VL input must be a valid safetensors file") from error
    names = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "position_ids"}
    if set(tensors) != names:
        raise ValueError("prepared Qwen3-VL input requires exactly input_ids, attention_mask, "
                         "pixel_values, image_grid_thw and position_ids")
    for name, value in tensors.items():
        dtype = torch.float32 if name == "pixel_values" else torch.int64
        if value.dtype != dtype:
            raise ValueError(f"prepared Qwen3-VL {name} must have dtype {dtype}")
    ids, mask, grid, pixels, positions = (tensors[name] for name in
                                         ("input_ids", "attention_mask", "image_grid_thw",
                                          "pixel_values", "position_ids"))
    if ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < 1 or bool((ids < 0).any()):
        raise ValueError("prepared Qwen3-VL input_ids must be non-negative B1 INT64 tokens")
    if mask.shape != ids.shape or not bool((mask == 1).all()):
        raise ValueError("prepared Qwen3-VL attention_mask must match input_ids and contain only ones")
    if (grid.shape != (1, 3) or not bool((grid > 0).all()) or int(grid[0, 0]) != 1):
        raise ValueError("prepared Qwen3-VL image_grid_thw requires one positive single-frame image grid")
    if (pixels.ndim != 2 or pixels.shape[0] != int(grid.prod())
            or pixels.shape[1] < 1 or not bool(torch.isfinite(pixels).all())):
        raise ValueError("prepared Qwen3-VL pixel_values must be finite patches matching image_grid_thw")
    if positions.shape != (3, 1, ids.shape[1]) or bool((positions < 0).any()):
        raise ValueError("prepared Qwen3-VL position_ids must be non-negative [3,1,prefill_tokens]")
    metadata = {"format": "single_image_preprocessed_safetensors", "file": str(path),
                "file_sha256": digest, "file_bytes": len(content),
                "tensors": {name: {"shape": list(value.shape), "dtype": str(value.dtype),
                                   "sha256": hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()}
                            for name, value in sorted(tensors.items())}}
    return tensors, metadata


def validate_prepared_positions(tensors, metadata, checkpoint_config):
    """Use HF's normal M-RoPE calculation, without allocating a model's weights.

    position_ids are a saved witness, not a forward override. Forward receives
    modality IDs so its normal prefill path also sets the decode rope_deltas.
    """
    import torch
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel

    vision = checkpoint_config.vision_config
    ids, grid, pixels = (tensors[name] for name in ("input_ids", "image_grid_thw", "pixel_values"))
    if bool((ids >= checkpoint_config.text_config.vocab_size).any()):
        raise ValueError("prepared Qwen3-VL token IDs exceed the checkpoint vocabulary")
    merge = vision.spatial_merge_size
    patch_width = vision.in_channels * vision.temporal_patch_size * vision.patch_size ** 2
    if (pixels.shape[1] != patch_width or any(int(value) % merge for value in grid[0, 1:])):
        raise ValueError("prepared Qwen3-VL image geometry does not match checkpoint patch/merge sizes")
    image_mask = ids == checkpoint_config.image_token_id
    indices = image_mask[0].nonzero().reshape(-1)
    expected = int(grid.prod()) // merge ** 2
    if (indices.numel() != expected or expected < 1
            or int(indices[-1]) - int(indices[0]) + 1 != expected
            or int(indices[0]) == 0 or int(indices[-1]) + 1 >= ids.shape[1]
            or int(ids[0, indices[0] - 1]) != checkpoint_config.vision_start_token_id
            or int(ids[0, indices[-1] + 1]) != checkpoint_config.vision_end_token_id
            or int((ids == checkpoint_config.vision_start_token_id).sum()) != 1
            or int((ids == checkpoint_config.vision_end_token_id).sum()) != 1
            or bool((ids == checkpoint_config.video_token_id).any())):
        raise ValueError("prepared Qwen3-VL requires one complete image placeholder block matching its grid")
    encoded = {name: value for name, value in tensors.items() if name != "position_ids"}
    encoded["mm_token_type_ids"] = image_mask.to(torch.int64)
    # These two HF utilities use only configuration and tensor geometry. Calling
    # them on a metadata owner avoids constructing an uninitialized full model.
    owner = SimpleNamespace(config=checkpoint_config)
    owner.get_vision_position_ids = MethodType(Qwen3VLModel.get_vision_position_ids, owner)
    positions, delta = Qwen3VLModel.get_rope_index(
        owner, ids, mm_token_type_ids=encoded["mm_token_type_ids"],
        image_grid_thw=grid, attention_mask=tensors["attention_mask"])
    if not torch.equal(positions, tensors["position_ids"]):
        raise ValueError("prepared Qwen3-VL position_ids do not match the checkpoint's normal M-RoPE positions")
    metadata["normal_mrope_positions_verified"] = True
    metadata["rope_deltas"] = delta.tolist()
    value = encoded["mm_token_type_ids"]
    metadata["derived_mm_token_type_ids"] = {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()}
    return encoded
