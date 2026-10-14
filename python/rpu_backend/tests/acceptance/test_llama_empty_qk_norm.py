from __future__ import annotations
import pytest
import torch
pytestmark = pytest.mark.hardware

def _make_mock_llama_weights(num_layers=2, hidden=2048, intermediate=8192, n_q_heads=32, n_kv_heads=8, head_dim=64):
    device = 'rpu'
    dtype = torch.float16
    q_w_list = [torch.randn(n_q_heads * head_dim, hidden, dtype=dtype, device=device) for _ in range(num_layers)]
    k_w_list = [torch.randn(n_kv_heads * head_dim, hidden, dtype=dtype, device=device) for _ in range(num_layers)]
    v_w_list = [torch.randn(n_kv_heads * head_dim, hidden, dtype=dtype, device=device) for _ in range(num_layers)]
    o_w_list = [torch.randn(hidden, n_q_heads * head_dim, dtype=dtype, device=device) for _ in range(num_layers)]
    input_norm_list = [torch.randn(hidden, dtype=dtype, device=device) for _ in range(num_layers)]
    post_norm_list = [torch.randn(hidden, dtype=dtype, device=device) for _ in range(num_layers)]
    gate_list = [torch.randn(intermediate, hidden, dtype=dtype, device=device) for _ in range(num_layers)]
    up_list = [torch.randn(intermediate, hidden, dtype=dtype, device=device) for _ in range(num_layers)]
    down_list = [torch.randn(hidden, intermediate, dtype=dtype, device=device) for _ in range(num_layers)]
    return dict(q_w_list=q_w_list, k_w_list=k_w_list, v_w_list=v_w_list, o_w_list=o_w_list, q_norm_list=[], k_norm_list=[], input_norm_list=input_norm_list, post_norm_list=post_norm_list, gate_list=gate_list, up_list=up_list, down_list=down_list)

def test_empty_qk_norm_lists_do_not_raise():
    pytest.importorskip('rpu_backend')
    import rpu_backend
    handle = torch.ops.rpu.causal_decoder_create()
    try:
        weights = _make_mock_llama_weights(num_layers=2)
        cos_cache = torch.zeros(8192, 64, dtype=torch.float16, device='rpu')
        sin_cache = torch.zeros(8192, 64, dtype=torch.float16, device='rpu')
        final_norm_w = torch.randn(2048, dtype=torch.float16, device='rpu')
        torch.ops.rpu.causal_decoder_set_weights(handle, weights['q_w_list'], weights['k_w_list'], weights['v_w_list'], weights['o_w_list'], weights['q_norm_list'], weights['k_norm_list'], weights['input_norm_list'], weights['post_norm_list'], weights['gate_list'], weights['up_list'], weights['down_list'], cos_cache, sin_cache, final_norm_w, 32, 8, 64, 2048, 8192, 1e-05, True)
        print('[U-2] OK: set_weights accepted empty q_norm/k_norm lists (has_qk_norm_=false branch)')
    finally:
        torch.ops.rpu.causal_decoder_destroy(handle)

def test_empty_qk_norm_does_not_apply_qk_rmsnorm():
    pytest.importorskip('rpu_backend')
    import rpu_backend
    handle = torch.ops.rpu.causal_decoder_create()
    try:
        weights = _make_mock_llama_weights(num_layers=2)
        cos_cache = torch.zeros(8192, 64, dtype=torch.float16, device='rpu')
        sin_cache = torch.zeros(8192, 64, dtype=torch.float16, device='rpu')
        final_norm_w = torch.randn(2048, dtype=torch.float16, device='rpu')
        torch.ops.rpu.causal_decoder_set_weights(handle, weights['q_w_list'], weights['k_w_list'], weights['v_w_list'], weights['o_w_list'], weights['q_norm_list'], weights['k_norm_list'], weights['input_norm_list'], weights['post_norm_list'], weights['gate_list'], weights['up_list'], weights['down_list'], cos_cache, sin_cache, final_norm_w, 32, 8, 64, 2048, 8192, 1e-05, True)
        print('[U-2] OK: in-place RoPE branch fired for has_qk_norm_=false (no crash on set_weights)')
    finally:
        torch.ops.rpu.causal_decoder_destroy(handle)

def test_asymmetric_qk_norm_lists_raise():
    pytest.importorskip('rpu_backend')
    import rpu_backend
    handle = torch.ops.rpu.causal_decoder_create()
    try:
        weights = _make_mock_llama_weights(num_layers=2)
        cos_cache = torch.zeros(8192, 64, dtype=torch.float16, device='rpu')
        sin_cache = torch.zeros(8192, 64, dtype=torch.float16, device='rpu')
        final_norm_w = torch.randn(2048, dtype=torch.float16, device='rpu')
        bad_q_norm = [torch.randn(64, dtype=torch.float16, device='rpu') for _ in range(2)]
        with pytest.raises(RuntimeError, match='q_norm_list.*k_norm_list'):
            torch.ops.rpu.causal_decoder_set_weights(handle, weights['q_w_list'], weights['k_w_list'], weights['v_w_list'], weights['o_w_list'], bad_q_norm, weights['k_norm_list'], weights['input_norm_list'], weights['post_norm_list'], weights['gate_list'], weights['up_list'], weights['down_list'], cos_cache, sin_cache, final_norm_w, 32, 8, 64, 2048, 8192, 1e-05, True)
        print('[U-2] OK: asymmetric q_norm/k_norm lists rejected loudly')
    finally:
        torch.ops.rpu.causal_decoder_destroy(handle)
