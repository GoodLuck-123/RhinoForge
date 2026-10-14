"""Public qwen3_vl model construction."""
from __future__ import annotations

from examples.common.text import text_context

def prepare(config, runner):
    import torch
    from examples.common.generation import _image_inputs, _text_input_ids, _extend_multimodal_input, _generate
    prepared = None
    if "prepared_input_safetensors" in config["input"]:
        from examples.common.config import validate_generation_options
        from .inputs import load_prepared_input

        validate_generation_options(config)
        prepared = load_prepared_input(config["input"])
    example, inputs, target, checkpoint, execution, decode_steps = text_context(config)
    from examples.common.config import validate_checkpoint_profile
    from transformers import AutoProcessor

    validate_checkpoint_profile(config, checkpoint)
    component = inputs.get("component", "multimodal")
    input_metadata = None
    if prepared is not None:
        from .inputs import validate_prepared_positions
        from transformers import AutoConfig

        tensors, input_metadata = prepared
        checkpoint_config = AutoConfig.from_pretrained(
            checkpoint, local_files_only=True, trust_remote_code=False)
        encoded = validate_prepared_positions(tensors, input_metadata, checkpoint_config)
    images = _image_inputs(inputs) if prepared is None and component != "text" else []
    processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    if images:
        messages = [{"role": "user", "content": [
            *({"type": "image", "image": image} for image in images),
            {"type": "text", "text": inputs["prompt"]},
        ]}]
        encoded = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt")
        encoded = _extend_multimodal_input(encoded, inputs, processor.tokenizer)
    elif prepared is None:
        encoded = {"input_ids": _text_input_ids(processor.tokenizer, inputs)}
    from rpu_backend.api import RPUCache, RPUModelForConditionalGeneration

    model = RPUModelForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.float16, device="rpu", rpu_execution=execution,
        local_files_only=True, trust_remote_code=False)
    runner.owner = model
    if component == "vision":
        return lambda: model.model.visual(
            encoded["pixel_values"].to("rpu", dtype=torch.float16),
            grid_thw=encoded["image_grid_thw"]), model
    cache = RPUCache.from_model(model.model.language_model,
                               input_ids=encoded["input_ids"],
                               max_new_tokens=decode_steps)
    prefill = {name: encoded[name] for name in (
        "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"
    ) if name in encoded}
    input_ids = encoded["input_ids"]
    def infer(decode_steps=decode_steps):
        result = _generate(model, cache, input_ids, decode_steps, prefill, processor.tokenizer,
                           stop_on_eos=inputs.get("stop_on_eos", True), input_device="cpu",
                           reuse_greedy_logits=True)
        if input_metadata is not None:
            result["generation"]["prepared_input"] = dict(input_metadata)
        return result

    return infer, model
