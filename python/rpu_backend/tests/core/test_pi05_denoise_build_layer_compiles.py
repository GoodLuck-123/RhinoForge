from pathlib import Path
import rpu_backend

def test_build_layer_subgraph_phases_present():
    cpp = Path(__file__).resolve().parents[4] / 'src/fused/rpu_pi05_denoise_step_model.cpp'
    text = cpp.read_text()
    must_have = ('Phase A: per-layer GEMV', 'Phase 1: layer input DMA', 'Phase 2: AdaRMSNorm', 'Phase 3: QKV Linear', 'Phase 4: RoPE', 'Phase 5: KV cache insert', 'Phase 6: SDPA', 'Phase 7: O_proj', 'Phase 8: all_reduce_sum_residual', 'Phase 9: GATED attn residual', 'Phase 10: Post-attention AdaRMSNorm', 'Phase 11: MLP pipeline', 'Phase 13: GATED MLP residual', 'Phase 14: Output DMA')
    for p in must_have:
        assert p in text, f'build_layer_subgraph missing: {p}'
    assert 'cond_loaded_this_forward_' not in text, 'Phase 0 cond block must be deleted from build_layer_subgraph (§C2)'
