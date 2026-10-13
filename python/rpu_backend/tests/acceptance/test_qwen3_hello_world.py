from __future__ import annotations
import os
import pytest
HARDWARE_REQUIRED = pytest.mark.hardware
from rpu_backend.model_registry import model_path as _model_path
_QWEN3_LOCAL = str(_model_path('qwen3-0.6b'))
QWEN3_MODEL = _QWEN3_LOCAL if os.path.isdir(_QWEN3_LOCAL) else 'Qwen/Qwen3-0.6B'

def _hw_available() -> bool:
    return os.environ.get('RPU_HW_AVAILABLE', '0') == '1'

@HARDWARE_REQUIRED
@pytest.mark.skipif(not _hw_available(), reason='device hardware + sudo required')
def test_qwen3_hello_world_after_pi05_import():
    import torch
    from transformers import AutoTokenizer
    import rpu_backend.adapters.pi05
    from rpu_backend import RPUModelForCausalLM, RPUCache
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL)
    model = RPUModelForCausalLM.from_pretrained(QWEN3_MODEL, dtype=torch.float16).to('rpu')
    input_ids = tokenizer('Hello, world!', return_tensors='pt').input_ids
    cache = RPUCache.from_model(model, input_ids=input_ids, max_new_tokens=16)
    output_ids = model.generate(input_ids.to('rpu'), past_key_values=cache, max_new_tokens=16)
    assert output_ids.shape[1] >= input_ids.shape[1] + 16, f'Expected at least {input_ids.shape[1] + 16} tokens, got {output_ids.shape[1]}; Pi05 import broke Qwen3 — F6 regression'
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    assert 'Hello' in decoded or 'hello' in decoded.lower(), f'Decoded output `{decoded[:100]}` does not contain the prompt; generation probably failed silently'

@HARDWARE_REQUIRED
@pytest.mark.skipif(not _hw_available(), reason='device hardware + sudo required')
def test_qwen3_hello_world_without_pi05_import():
    import torch
    from transformers import AutoTokenizer
    from rpu_backend import RPUModelForCausalLM, RPUCache
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL)
    model = RPUModelForCausalLM.from_pretrained(QWEN3_MODEL, dtype=torch.float16).to('rpu')
    input_ids = tokenizer('Hello, world!', return_tensors='pt').input_ids
    cache = RPUCache.from_model(model, input_ids=input_ids, max_new_tokens=16)
    output_ids = model.generate(input_ids.to('rpu'), past_key_values=cache, max_new_tokens=16)
    assert output_ids.shape[1] >= input_ids.shape[1] + 16
