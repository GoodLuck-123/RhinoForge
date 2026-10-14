from __future__ import annotations
import os
import torch
DEFAULT_SEED = 0
ENV_VAR = 'RPU_TEST_SEED'

def input_seed(default: int=DEFAULT_SEED) -> int:
    raw = os.environ.get(ENV_VAR)
    if raw is None or not raw.strip():
        return int(default)
    try:
        return int(raw.strip(), 0)
    except ValueError:
        raise ValueError(f'{ENV_VAR}={raw!r} is not an integer (accepts decimal or 0x-prefixed)') from None

def input_generator(default: int=DEFAULT_SEED) -> torch.Generator:
    return torch.Generator().manual_seed(input_seed(default))
