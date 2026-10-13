from __future__ import annotations
import torch
from rpu_backend.api.policy import Pi05Policy
_DTYPE_MAP = {'fp16': torch.float16, 'fp32': torch.float32, 'bf16': torch.bfloat16}

def load_pi05_model(model_path: str, dtype: str='fp16', fused: bool=False):
    torch_dtype = _DTYPE_MAP.get(dtype)
    if torch_dtype is None:
        raise ValueError(f'Unsupported dtype string: {dtype!r}; expected one of {list(_DTYPE_MAP)}.')
    wrapper = Pi05Policy.from_pretrained(model_path, dtype=torch_dtype)
    lerobot_policy = wrapper._lerobot_policy
    return (lerobot_policy, lerobot_policy.config)
