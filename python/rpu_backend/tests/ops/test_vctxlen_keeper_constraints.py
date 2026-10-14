import pytest
from rpu_backend._cpp_ext import sdpa_helpers as h

def test_vctxlen_ltm_tile5_rejected():
    with pytest.raises(RuntimeError, match='LTM'):
        h.validate_kernel_call(seq_q=80, seq_k=80, tile_m_v16=5, tile_n_v16=8, tile_k_v16=8, tile_m=80, grid_dim_x=1, gqa_group_size=1, hd_q_v16=8, hd_k_v16=8, hd_v_v16=8, attn_mask_type=1)
