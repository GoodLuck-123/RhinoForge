from __future__ import annotations
import numpy as np
import torch
TOKENIZER_NAME = 'unsloth/gemma-2-2b'

def _load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME, local_files_only=True)

def _prepare_dummy_inputs(config, tokenizer_max_length, batch_size=1):
    img_size = config.image_resolution[0] if hasattr(config, 'image_resolution') else 224
    if hasattr(config, 'image_features') and config.image_features:
        img_keys = list(config.image_features.keys())
    else:
        img_keys = ['observation.images.base_0_rgb', 'observation.images.left_wrist_0_rgb', 'observation.images.right_wrist_0_rgb']
    batch = {}
    for key in img_keys:
        batch[key] = torch.rand(batch_size, 3, img_size, img_size, dtype=torch.float32, device='cpu')
        batch[f'{key}_padding_mask'] = torch.ones(batch_size, dtype=torch.bool)
    state = torch.randn(batch_size, config.max_state_dim, dtype=torch.float32, device='cpu').clamp(-1, 1)
    tokenizer = _load_tokenizer()
    state_np = state.cpu().numpy()
    discretized = np.digitize(state_np, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    state_str = ' '.join(map(str, discretized[0]))
    prompt = f'Task: pick up the red cup, State: {state_str};\nAction: '
    tokenized = tokenizer([prompt] * batch_size, max_length=tokenizer_max_length, padding='max_length', truncation=True, return_tensors='pt')
    batch['observation.state'] = state
    batch['observation.language.tokens'] = tokenized['input_ids']
    batch['observation.language.attention_mask'] = tokenized['attention_mask'].bool()
    return batch
