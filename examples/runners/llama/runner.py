"""Public llama text model construction."""
from __future__ import annotations

from examples.common.text import text_context, text_inputs, text_result

def prepare(config, runner):
    import torch
    example, inputs, target, checkpoint, execution, decode_steps = text_context(config)
    tokenizer, input_ids, input_metadata, capacity = text_inputs(config, checkpoint, decode_steps)
    token_selector = None
    from rpu_backend import RPUCache, RPUModelForCausalLM

    model = RPUModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.float16, device="rpu", rpu_execution=execution,
        local_files_only=True, trust_remote_code=False)
    runner.owner = model
    cache = RPUCache.from_model(model, input_ids=input_ids,
                               max_new_tokens=decode_steps)
    return text_result(model, cache, input_ids, decode_steps, tokenizer, inputs, target, token_selector, input_metadata), model
