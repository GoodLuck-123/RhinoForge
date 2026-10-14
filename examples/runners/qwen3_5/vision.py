"""Public Qwen3.5 multimodal construction."""
from __future__ import annotations


from examples.common.inputs import _controlled_images as _images, _controlled_checkpoint as _checkpoint

def _qwen35_vision(runner):
    from examples.common.config import validate_generation_options

    validate_generation_options(runner.source_config)
    import torch
    from examples.common.generation import _extend_multimodal_input, _generate
    from transformers import AutoProcessor
    from rpu_backend.api import RPUModelForConditionalGeneration

    config, inputs = runner.source_config, runner.source_config["input"]
    timing_mode = config.get("run", {}).get("timing_mode", "per_call")
    checkpoint = str(_checkpoint(config))
    if "profile_id" in config["example"]:
        from examples.common.config import validate_checkpoint_profile

        validate_checkpoint_profile(config, checkpoint)
    images = _images(inputs)
    prompt = inputs.get("prompt", "Describe the image.")
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    messages = [{"role": "user", "content": [
        *({"type": "image"} for _ in images), {"type": "text", "text": prompt}
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[text], images=images, return_tensors="pt")
    if "mm_token_type_ids" not in encoded:
        raise ValueError("Qwen3.5 processor must return mm_token_type_ids for image inference")
    encoded = _extend_multimodal_input(encoded, inputs, processor.tokenizer)
    decode_steps = inputs.get("decode_steps", 4)
    required_len = int(encoded["input_ids"].shape[1]) + decode_steps
    max_seq_len = (required_len if timing_mode == "legacy_dense_vl"
                   else max(64, ((required_len + 63) // 64) * 64))
    is_moe = config["example"]["architecture"] == (
        "Qwen3_5MoeForConditionalGeneration"
    )
    if is_moe:
        from rpu_backend.adapters.qwen3_5_moe import Qwen3_5MoeAdapter
        from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache

        raw = RPUModelForConditionalGeneration.from_pretrained(
            checkpoint, dtype=torch.float16,
            local_files_only=True, rpu_execution=config.get("rpu_execution"),
        )
        runner.owner = raw
        model = Qwen3_5MoeAdapter(
            raw, rpu_execution=config.get("rpu_execution")
        ).to_rpu(max_seq_len=max_seq_len)
        cache = Qwen3_5MoeCache.from_model(model, max_seq_len=max_seq_len)
        request = {
            key: encoded[key]
            for key in (
                "input_ids", "pixel_values", "image_grid_thw",
                "attention_mask", "mm_token_type_ids",
            )
            if key in encoded
        }
    else:
        from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache

        if timing_mode == "legacy_dense_vl":
            from rpu_backend.adapters.qwen3_5 import Qwen3_5Adapter

            raw = RPUModelForConditionalGeneration.from_pretrained(
                checkpoint, dtype=torch.float16,
                local_files_only=True, rpu_execution=config.get("rpu_execution"),
            )
            runner.owner = raw
            model = Qwen3_5Adapter(
                raw.half(), rpu_execution=config.get("rpu_execution")
            ).to_rpu(max_seq_len=max_seq_len)
        else:
            model = RPUModelForConditionalGeneration.from_pretrained(
                checkpoint, dtype=torch.float16, device="rpu",
                local_files_only=True, rpu_execution=config.get("rpu_execution"),
            )
        runner.owner = model
        cache = Qwen3_5Cache.from_model(model, max_seq_len=max_seq_len)
        request = {
            key: (
                encoded[key].to("rpu")
                if key == "pixel_values" else encoded[key]
            )
            for key in (
                "input_ids", "pixel_values", "image_grid_thw",
                "mm_token_type_ids",
            )
        }
    runner.owner = model

    input_ids = request.pop("input_ids")
    def infer(decode_steps=decode_steps):
        return _generate(model, cache, input_ids, decode_steps, request,
                         processor.tokenizer, stop_on_eos=inputs.get("stop_on_eos", False),
                         input_device="cpu", timing_mode=timing_mode)

    return infer


prepare = _qwen35_vision
