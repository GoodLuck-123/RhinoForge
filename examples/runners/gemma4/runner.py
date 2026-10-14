"""Public gemma4 text model construction."""
from __future__ import annotations

from examples.common.text import text_context, text_inputs, text_result

def prepare(config, runner):
    import torch
    from examples.common.generation import _cpu_fp32_argmax_token
    example, inputs, target, checkpoint, execution, decode_steps = text_context(config)
    tokenizer, input_ids, input_metadata, capacity = text_inputs(config, checkpoint, decode_steps)
    token_selector = None
    from transformers import AutoModelForImageTextToText
    from rpu_backend.adapters.gemma4 import Gemma4Adapter

    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map="cpu",
        local_files_only=True, low_cpu_mem_usage=True).eval()
    model._rpu_max_seq = max(512, capacity)
    model = Gemma4Adapter(model, rpu_execution=execution).to_rpu()
    runner.owner = model
    cache = model.build_rpu_cache(max_seq_len=max(512, capacity))
    return text_result(model, cache, input_ids, decode_steps, tokenizer, inputs, target, token_selector, input_metadata), model
