__test__ = False
import sys
import torch
from transformers import AutoTokenizer
from rpu_backend import RPUModelForCausalLM, RPUCache
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
model = RPUModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', dtype=torch.float16).to('rpu')
input_ids = tokenizer('Hello, world!', return_tensors='pt').input_ids
cache = RPUCache.from_model(model, input_ids=input_ids, max_new_tokens=16)
output_ids = model.generate(input_ids.to('rpu'), past_key_values=cache, max_new_tokens=16)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
assert output_ids.shape[1] >= input_ids.shape[1] + 16, f'Expected at least {input_ids.shape[1] + 16} tokens, got {output_ids.shape[1]}'
print('[PASS] docs/api_reference.md Quick Start runnable + produces >= input + max_new_tokens tokens')
sys.exit(0)
