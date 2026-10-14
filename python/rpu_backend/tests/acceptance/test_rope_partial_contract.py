from __future__ import annotations
import ast
from pathlib import Path
import pytest
import torch
from rpu_backend.runtime import rope_partial
BUILDERS = (rope_partial.build_interleaved_mrope_cos_sin, rope_partial.build_chunked_mrope_cos_sin)

def _kwargs() -> dict:
    return {'head_dim': 128, 'rope_theta': 1000000.0, 'mrope_section': [16, 24, 24]}

def test_rope_partial_has_no_optimized_away_assertions():
    tree = ast.parse(Path(rope_partial.__file__).read_text(encoding='utf-8'))
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Assert)]

@pytest.mark.parametrize('builder', BUILDERS)
def test_partial_mrope_valid_tables_are_fp16_and_board_free(builder):
    positions = torch.arange(12, dtype=torch.long)
    position_ids = positions[:, None].expand(12, 3).contiguous()
    (cos, sin) = builder(position_ids, **_kwargs())
    assert tuple(cos.shape) == (12, 64)
    assert tuple(sin.shape) == (12, 64)
    assert cos.dtype is torch.float16
    assert sin.dtype is torch.float16

@pytest.mark.parametrize('builder', BUILDERS)
@pytest.mark.parametrize(('position_ids', 'overrides', 'error', 'message'), [(torch.zeros(3, dtype=torch.long), {}, ValueError, '\\[seq, 3\\]'), (torch.zeros(2, 3), {}, TypeError, 'integer dtype'), (torch.tensor([[0, -1, 0]]), {}, ValueError, 'non-negative'), (torch.zeros(1, 3, dtype=torch.long), {'head_dim': 127}, ValueError, 'head_dim'), (torch.zeros(1, 3, dtype=torch.long), {'rotary_dim': 130}, ValueError, 'rotary_dim 130'), (torch.zeros(1, 3, dtype=torch.long), {'rope_theta': float('inf')}, ValueError, 'positive and finite'), (torch.zeros(1, 3, dtype=torch.long), {'mrope_section': [16, 24, 23]}, ValueError, 'must sum')])
def test_partial_mrope_rejects_invalid_contracts(builder, position_ids, overrides, error, message):
    kwargs = _kwargs()
    kwargs.update(overrides)
    with pytest.raises(error, match=message):
        builder(position_ids, **kwargs)
