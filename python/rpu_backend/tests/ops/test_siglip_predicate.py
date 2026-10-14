from rpu_backend._cpp_ext import sdpa_helpers as h

def test_siglip_padded_head_dim_accepts():
    assert h.is_valid_chunk_size(num_q_heads=16, num_kv_heads=16, num_cores=8, head_dim=80, attn_mask_type=2, cs=256, seq_len=256, position=0)

def test_siglip_orig_head_dim_rejected():
    assert not h.is_valid_chunk_size(num_q_heads=16, num_kv_heads=16, num_cores=8, head_dim=72, attn_mask_type=2, cs=256, seq_len=256, position=0)
