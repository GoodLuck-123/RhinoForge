from __future__ import annotations
import importlib.util
import os
import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLRotaryEmbedding
_rp_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../runtime/rope_partial.py'))
_rp_spec = importlib.util.spec_from_file_location('wall_oss_rope_partial_cpu', _rp_path)
_rp_mod = importlib.util.module_from_spec(_rp_spec)
_rp_spec.loader.exec_module(_rp_mod)
build_chunked_mrope_cos_sin = _rp_mod.build_chunked_mrope_cos_sin
build_interleaved_mrope_cos_sin = _rp_mod.build_interleaved_mrope_cos_sin
HEAD_DIM = 128
ROPE_THETA = 1000000.0
MROPE_SECTION = [16, 24, 24]

def _transformers_tables(position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    config = Qwen2_5_VLTextConfig(hidden_size=HEAD_DIM, num_attention_heads=1, num_key_value_heads=1, rope_parameters={'rope_type': 'default', 'rope_theta': ROPE_THETA, 'mrope_section': MROPE_SECTION})
    rotary = Qwen2_5_VLRotaryEmbedding(config)
    (cos, sin) = rotary(torch.zeros(1, position_ids.size(0), HEAD_DIM, dtype=torch.float16), position_ids.T[:, None, :])
    sections = MROPE_SECTION * 2

    def select_axes(table: torch.Tensor) -> torch.Tensor:
        selected = torch.cat([chunk[axis % 3] for (axis, chunk) in enumerate(table.split(sections, dim=-1))], dim=-1)
        return selected[0, :, :HEAD_DIM // 2].contiguous()
    return (select_axes(cos), select_axes(sin))

def _positions() -> dict[str, torch.Tensor]:
    text = torch.arange(40, dtype=torch.long)
    mixed = [[i, i, i] for i in range(5)]
    mixed += [[5, 5 + h, 5 + w] for h in range(9) for w in range(16)]
    mixed += [[14 + i, 14 + i, 14 + i] for i in range(5)]
    return {'text_only': text[:, None].expand(40, 3).contiguous(), 'mixed': torch.tensor(mixed, dtype=torch.long)}

def test_chunked_mrope_matches_transformers_qwen25_contract():
    for (name, position_ids) in _positions().items():
        (expected_cos, expected_sin) = _transformers_tables(position_ids)
        (actual_cos, actual_sin) = build_chunked_mrope_cos_sin(position_ids, head_dim=HEAD_DIM, rope_theta=ROPE_THETA, mrope_section=MROPE_SECTION)
        torch.testing.assert_close(actual_cos, expected_cos, rtol=0, atol=4e-06, msg=name)
        torch.testing.assert_close(actual_sin, expected_sin, rtol=0, atol=4e-06, msg=name)
        if name == 'mixed':
            (interleaved_cos, _) = build_interleaved_mrope_cos_sin(position_ids, head_dim=HEAD_DIM, rope_theta=ROPE_THETA, mrope_section=MROPE_SECTION)
            assert not torch.equal(actual_cos, interleaved_cos)
if __name__ == '__main__':
    test_chunked_mrope_matches_transformers_qwen25_contract()
    print('PASS')
