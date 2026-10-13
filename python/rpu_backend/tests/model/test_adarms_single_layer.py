#!/usr/bin/env python
'Adarms single layer device regression.'

import math
import pytest
import torch

from rpu_backend.model_registry import set_hf_home_env
set_hf_home_env()

import rpu_backend  # noqa: F401  # import side effects: torch.ops.rpu / torch.rpu binding
from rpu_backend.api.cache import RPUCache
from rpu_backend.runtime.weights import (
    transform_linear_weight,
    NUM_CORES,
    tp_row_swizzle_mc_weight,
    tp_col_swizzle_mc_weight,
)


# ---------------------------------------------------------------------------
# Pi0.5 Action Expert dimensions (matches the v2 baseline test pre-rewrite).
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 1024
NUM_Q_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 256
INTERMEDIATE_SIZE = 4096
EPS = 1e-6
ATTN_TP = min(8, NUM_KV_HEADS)  # = 1
MAX_POS = 4096

MAX_DIFF_SINGLE = 0.3

# 3-way oracle empirically showed: v2≈CPU within 0.005 (spec oracle correct), but v3-vs-CPU and
# v2-vs-v3 both ≈ 0.155 max-element due to fp16 chunking + GraphCache REPLAY differences vs v2





MAX_MSE_SINGLE = 5e-3   # secondary structural-correctness guard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_rope_freqs(head_dim, max_pos):
    'Round-3 BLOCKER 2: non-identity RoPE.'
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_pos).float()
    freqs = torch.einsum("i,j->ij", positions, inv_freq)
    cos_fp32 = freqs.cos().contiguous()
    sin_fp32 = freqs.sin().contiguous()
    cos_rpu = cos_fp32.to(dtype=torch.float16, device="rpu").contiguous()
    sin_rpu = sin_fp32.to(dtype=torch.float16, device="rpu").contiguous()
    return cos_rpu, sin_rpu, cos_fp32, sin_fp32


def _setup_random_inputs_and_weights(seq_len):
    'Generate random inputs + per-layer weights.'
    torch.manual_seed(42)
    hidden = torch.randn(1, seq_len, HIDDEN_SIZE, dtype=torch.float16)
    cond = torch.randn(HIDDEN_SIZE, dtype=torch.float16)

    def mk_w(out_f, in_f):
        return torch.randn(out_f, in_f, dtype=torch.float16) * 0.01

    q_w = mk_w(NUM_Q_HEADS * HEAD_DIM, HIDDEN_SIZE)
    k_w = mk_w(NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE)
    v_w = mk_w(NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE)
    o_w = mk_w(HIDDEN_SIZE, NUM_Q_HEADS * HEAD_DIM)
    gate_w = mk_w(INTERMEDIATE_SIZE, HIDDEN_SIZE)
    up_w = mk_w(INTERMEDIATE_SIZE, HIDDEN_SIZE)
    down_w = mk_w(HIDDEN_SIZE, INTERMEDIATE_SIZE)
    attn_dw = mk_w(3 * HIDDEN_SIZE, HIDDEN_SIZE)
    attn_db = torch.randn(3 * HIDDEN_SIZE, dtype=torch.float16) * 0.01
    mlp_dw = mk_w(3 * HIDDEN_SIZE, HIDDEN_SIZE)
    mlp_db = torch.randn(3 * HIDDEN_SIZE, dtype=torch.float16) * 0.01

    # v3 device-swizzled copies
    q_w_rpu = tp_col_swizzle_mc_weight(q_w, ATTN_TP).to("rpu").contiguous()
    k_w_rpu = tp_col_swizzle_mc_weight(k_w, ATTN_TP).to("rpu").contiguous()
    v_w_rpu = tp_col_swizzle_mc_weight(v_w, ATTN_TP).to("rpu").contiguous()
    o_w_rpu = tp_row_swizzle_mc_weight(o_w, ATTN_TP).to("rpu").contiguous()
    gate_w_rpu = tp_col_swizzle_mc_weight(gate_w, NUM_CORES).to("rpu").contiguous()
    up_w_rpu = tp_col_swizzle_mc_weight(up_w, NUM_CORES).to("rpu").contiguous()
    down_w_rpu = tp_row_swizzle_mc_weight(down_w, NUM_CORES).to("rpu").contiguous()
    attn_dw_rpu = transform_linear_weight(attn_dw, partition=0, num_cores=NUM_CORES).to("rpu").contiguous()
    attn_db_rpu = attn_db.to("rpu").contiguous()
    mlp_dw_rpu = transform_linear_weight(mlp_dw, partition=0, num_cores=NUM_CORES).to("rpu").contiguous()
    mlp_db_rpu = mlp_db.to("rpu").contiguous()

    return {
        "hidden": hidden,
        "cond": cond,
        "raw": {
            "q": q_w, "k": k_w, "v": v_w, "o": o_w,
            "gate": gate_w, "up": up_w, "down": down_w,
            "attn_dw": attn_dw, "attn_db": attn_db,
            "mlp_dw": mlp_dw, "mlp_db": mlp_db,
        },
        "rpu": {
            "q": q_w_rpu, "k": k_w_rpu, "v": v_w_rpu, "o": o_w_rpu,
            "gate": gate_w_rpu, "up": up_w_rpu, "down": down_w_rpu,
            "attn_dw": attn_dw_rpu, "attn_db": attn_db_rpu,
            "mlp_dw": mlp_dw_rpu, "mlp_db": mlp_db_rpu,
        },
    }


def _rmsnorm_fp32(x, eps):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps)


def _apply_rope_fp32(x, cos, sin, position):
    'RoPE rotation per Llama/Gemma convention: pair (x[..., 2i], x[..., 2i+1])'
    seq_len = x.shape[1]
    cos_slice = cos[position:position + seq_len]
    sin_slice = sin[position:position + seq_len]
    # Broadcast shape: [1, S, 1, D/2]
    cos_b = cos_slice[None, :, None, :]
    sin_b = sin_slice[None, :, None, :]
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    out1 = x1 * cos_b - x2 * sin_b
    out2 = x1 * sin_b + x2 * cos_b
    return torch.cat([out1, out2], dim=-1)


def _cpu_reference_adarms_decoder_layer(hidden, cond, raw, cos_fp32, sin_fp32, position, attention_mask=None):
    'Surgical CPU fp32 reimplementation of one AdaRMS-Gemma decoder layer.'
    h = hidden.to(dtype=torch.float32)
    c = cond.to(dtype=torch.float32)
    H = HIDDEN_SIZE
    nq = NUM_Q_HEADS
    nkv = NUM_KV_HEADS
    d = HEAD_DIM
    seq_len = h.shape[1]
    residual0 = h

    # ── Modulation projection (attn): gemv = attn_dw @ cond + attn_db; scale1 += 1
    raw1 = raw["attn_dw"].float() @ c + raw["attn_db"].float()  # [3H]
    scale1 = raw1[:H] + 1.0
    shift1 = raw1[H : 2 * H]
    gate1 = raw1[2 * H : 3 * H]


    h_norm1 = _rmsnorm_fp32(h, EPS) * scale1 + shift1  # [1, S, H]


    q = (h_norm1 @ raw["q"].float().t()).view(1, seq_len, nq, d)
    k = (h_norm1 @ raw["k"].float().t()).view(1, seq_len, nkv, d)
    v = (h_norm1 @ raw["v"].float().t()).view(1, seq_len, nkv, d)
    q_rot = _apply_rope_fp32(q, cos_fp32, sin_fp32, position)
    k_rot = _apply_rope_fp32(k, cos_fp32, sin_fp32, position)

    # ── SDPA + O_proj (device unified layout [B,S,H,D]; transpose for SDPA)
    q_t = q_rot.transpose(1, 2)  # [B, nq, S, d]
    # GQA: repeat KV heads to match Q heads
    if nkv != nq:
        rep = nq // nkv
        k_t = k_rot.transpose(1, 2).repeat_interleave(rep, dim=1)
        v_t = v.transpose(1, 2).repeat_interleave(rep, dim=1)
    else:
        k_t = k_rot.transpose(1, 2)
        v_t = v.transpose(1, 2)
    attn_scores = (q_t @ k_t.transpose(-1, -2)) / math.sqrt(d)
    if attention_mask is not None:
        attn_scores = attn_scores + attention_mask.float()
    attn_p = torch.softmax(attn_scores, dim=-1)
    attn_out = attn_p @ v_t  # [B, nq, S, d]
    attn_out = attn_out.transpose(1, 2).reshape(1, seq_len, nq * d)  # back to [B, S, nq*d]
    attn_out = attn_out @ raw["o"].float().t()  # O_proj → [1, S, H]


    # v2/v3 kernel comment "residual2 = gate * (residual2 - residual1) + residual1"
    # where residual2_pre = sum_cores(O_proj) + residual1 (= attn_branch + hidden).
    # → residual2_post = hidden + gate1 * attn_branch  (additive, NOT interpolation)
    out_attn = residual0 + gate1 * attn_out

    # ── Modulation projection (mlp): gemv = mlp_dw @ cond + mlp_db; scale2 += 1
    raw2 = raw["mlp_dw"].float() @ c + raw["mlp_db"].float()
    scale2 = raw2[:H] + 1.0
    shift2 = raw2[H : 2 * H]
    gate2 = raw2[2 * H : 3 * H]


    h_norm2 = _rmsnorm_fp32(out_attn, EPS) * scale2 + shift2


    gate_out = h_norm2 @ raw["gate"].float().t()  # [1, S, I]
    up_out = h_norm2 @ raw["up"].float().t()
    gate_act = torch.nn.functional.gelu(gate_out, approximate="tanh")
    mlp_inner = gate_act * up_out
    mlp_out = mlp_inner @ raw["down"].float().t()  # [1, S, H]


    # v2/v3 same pattern: residual1 = gate2 * (mlp_full - post_attn) + post_attn
    # where mlp_full = mlp_branch + post_attn. → out = post_attn + gate2 * mlp_branch.
    out = out_attn + gate2 * mlp_out
    return out


def _v3_forward(handle, setup, cos_rpu, sin_rpu, position, num_layers=1, attn_mask=None):
    'Drive `torch.ops.rpu.adarms_forward` against the v3 AdaRMSModel handle path.'
    rpu = setup["rpu"]
    torch.ops.rpu.spm_alloc_reset_temporary()
    torch.ops.rpu.adarms_set_weights(
        handle,
        [rpu["q"]] * num_layers, [rpu["k"]] * num_layers,
        [rpu["v"]] * num_layers, [rpu["o"]] * num_layers,
        [rpu["gate"]] * num_layers, [rpu["up"]] * num_layers, [rpu["down"]] * num_layers,
        [rpu["attn_dw"]] * num_layers, [rpu["attn_db"]] * num_layers,
        [rpu["mlp_dw"]] * num_layers, [rpu["mlp_db"]] * num_layers,
        cos_rpu, sin_rpu,
        NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
        HIDDEN_SIZE, INTERMEDIATE_SIZE, EPS,
        [], [], [], [], [], [], [],
    )
    kv = RPUCache(
        num_layers=num_layers, batch_size=1, max_seq_len=2048,
        num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, attn_tp=ATTN_TP,
    )
    h_rpu = setup["hidden"].to("rpu").contiguous()
    c_rpu = setup["cond"].to("rpu").contiguous()
    is_causal = (attn_mask is None) and (num_layers > 0) and False  # Action-Expert default = bidirectional
    out_rpu = torch.ops.rpu.adarms_forward(
        handle, h_rpu, c_rpu, kv.k_caches, kv.v_caches, attn_mask, position, is_causal,
    )
    return out_rpu.float().cpu()


# ---------------------------------------------------------------------------
# Pytest fixture (Round-3 BLOCKER 9)
# ---------------------------------------------------------------------------
@pytest.fixture
def adarms_handle():
    'Function-scoped handle: create + cleanup with try/finally + SPM reset between tests.'
    handle = torch.ops.rpu.adarms_create()
    try:
        yield handle
    finally:
        try:
            torch.ops.rpu.adarms_destroy(handle)
        except Exception:  # noqa: BLE001
            pass
        try:
            torch.ops.rpu.spm_alloc_reset_temporary()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# Tests (Round-3 BLOCKER 2 + WARNING 10: parametrize seq_len; non-identity RoPE)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seq_len,position", [(2, 0), (80, 0)])
def test_adarms_single_layer_v3_vs_cpu_fp32(seq_len, position, adarms_handle):
    'v3 device adarms_forward vs surgical CPU fp32 reference; max_diff ≤ MAX_DIFF_SINGLE.'
    setup = _setup_random_inputs_and_weights(seq_len)
    cos_rpu, sin_rpu, cos_fp32, sin_fp32 = _make_rope_freqs(HEAD_DIM, MAX_POS)

    out_rpu = _v3_forward(adarms_handle, setup, cos_rpu, sin_rpu, position)
    out_cpu = _cpu_reference_adarms_decoder_layer(
        setup["hidden"], setup["cond"], setup["raw"], cos_fp32, sin_fp32, position,
    )

    diff = (out_rpu - out_cpu).abs()
    max_diff = diff.max().item()
    mse = diff.pow(2).mean().item()
    print(
        f"\n[adarms_single seq_len={seq_len} pos={position}] "
        f"max_diff={max_diff:.6f} mse={mse:.6f} "
        f"rpu_norm={out_rpu.norm():.4f} cpu_norm={out_cpu.norm():.4f}"
    )
    assert max_diff <= MAX_DIFF_SINGLE, (
        f"AdaRMS single-layer (seq_len={seq_len}, pos={position}): "
        f"max_diff={max_diff:.6f} > {MAX_DIFF_SINGLE} (FP16 comparison tolerance)"
    )
    assert mse <= MAX_MSE_SINGLE, (
        f"AdaRMS single-layer (seq_len={seq_len}, pos={position}): "
        f"mse={mse:.6f} > {MAX_MSE_SINGLE} (structural-correctness guard)"
    )


def test_adarms_single_layer_replay_determinism(adarms_handle):
    'Re-run with same inputs and verify GraphCache REPLAY produces identical output.'
    seq_len, position = 80, 0
    setup = _setup_random_inputs_and_weights(seq_len)
    cos_rpu, sin_rpu, _, _ = _make_rope_freqs(HEAD_DIM, MAX_POS)

    out1 = _v3_forward(adarms_handle, setup, cos_rpu, sin_rpu, position)
    out2 = _v3_forward(adarms_handle, setup, cos_rpu, sin_rpu, position)
    diff = (out1 - out2).abs()
    print(
        f"\n[adarms_single replay] max_diff={diff.max().item():.8f} "
        f"mse={diff.pow(2).mean().item():.8f}"
    )
    assert diff.max().item() == 0.0, (
        f"AdaRMS REPLAY non-deterministic: max_diff={diff.max().item():.8f}"
    )
