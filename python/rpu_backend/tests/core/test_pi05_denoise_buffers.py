from pathlib import Path
import rpu_backend

def test_declare_buffers_lists_required_names():
    cpp = Path(__file__).resolve().parents[4] / 'src/fused/rpu_pi05_denoise_step_model.cpp'
    text = cpp.read_text()
    required = ['residual1', 'input_norm', 'residual2', 'q', 'k', 'v', 'output', 'oproj', 'sdpa_tmp', 'sdpa_mask', 'gate', 'up', 'down', 'cond', 'attn_gemv', 'mlp_gemv', 'bias_temp', 'gemv_partial', 'x_t_spm', 'action_emb_core0_spm', 'final_gemv', 'final_out_spm', 'v_t_core0_spm', 'action_in_proj_b_spm', 'action_out_proj_b_spm']
    missing = [n for n in required if f'"{n}"' not in text]
    assert not missing, f'declare_buffers missing: {missing}'
    assert '"action_emb_spm"' not in text, 'action_emb_spm was dropped in Rev 2 (§C5); use FMB layer-0 DMA path instead'
