import pytest
from rpu_backend._cpp_ext import sdpa_helpers as h

def test_total_tile_boundary_accepts():
    h.validate_kernel_call(seq_q=128, seq_k=128, tile_m_v16=8, tile_n_v16=16, tile_k_v16=8, tile_m=128, grid_dim_x=1, gqa_group_size=1, hd_q_v16=8, hd_k_v16=8, hd_v_v16=8, attn_mask_type=0)

def test_total_tile_over_rejects():
    with pytest.raises(RuntimeError, match='total_tile'):
        h.validate_kernel_call(seq_q=128, seq_k=128, tile_m_v16=8, tile_n_v16=17, tile_k_v16=8, tile_m=128, grid_dim_x=1, gqa_group_size=1, hd_q_v16=8, hd_k_v16=8, hd_v_v16=8, attn_mask_type=0)

def test_vlm_overflow_rejects():
    with pytest.raises(RuntimeError, match='VLM'):
        h.validate_kernel_call(seq_q=128, seq_k=128, tile_m_v16=8, tile_n_v16=8, tile_k_v16=8, tile_m=128, grid_dim_x=1, gqa_group_size=1, hd_q_v16=24, hd_k_v16=24, hd_v_v16=24, attn_mask_type=2)
