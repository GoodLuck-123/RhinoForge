#!/usr/bin/env python
'Adarms multilayer device regression.'

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

from rpu_backend.tests.model.test_adarms_single_layer import (
    HIDDEN_SIZE, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, INTERMEDIATE_SIZE, EPS, ATTN_TP, MAX_POS,
    MAX_DIFF_SINGLE, MAX_MSE_SINGLE,
    _make_rope_freqs,
    _cpu_reference_adarms_decoder_layer,
)


SEQ_LEN = 50
NUM_LAYERS = 18
# Empirical multi-layer noise: max_diff ~ MAX_DIFF_SINGLE * sqrt(num_layers) per Round-3 WARNING 5


MAX_DIFF_MULTI = MAX_DIFF_SINGLE * math.sqrt(NUM_LAYERS)
MAX_MSE_MULTI = MAX_MSE_SINGLE * NUM_LAYERS


def _setup_multilayer_weights():
    'Generate per-layer weights for NUM_LAYERS layers + a single shared input/cond.'
    torch.manual_seed(42)
    hidden = torch.randn(1, SEQ_LEN, HIDDEN_SIZE, dtype=torch.float16)
    cond = torch.randn(HIDDEN_SIZE, dtype=torch.float16)

    def mk_w(out_f, in_f):
        return torch.randn(out_f, in_f, dtype=torch.float16) * 0.01

    layers = []
    for L in range(NUM_LAYERS):
        torch.manual_seed(100 + L)
        q = mk_w(NUM_Q_HEADS * HEAD_DIM, HIDDEN_SIZE)
        k = mk_w(NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE)
        v = mk_w(NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE)
        o = mk_w(HIDDEN_SIZE, NUM_Q_HEADS * HEAD_DIM)
        gate = mk_w(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        up = mk_w(INTERMEDIATE_SIZE, HIDDEN_SIZE)
        down = mk_w(HIDDEN_SIZE, INTERMEDIATE_SIZE)
        attn_dw = mk_w(3 * HIDDEN_SIZE, HIDDEN_SIZE)
        attn_db = torch.randn(3 * HIDDEN_SIZE, dtype=torch.float16) * 0.01
        mlp_dw = mk_w(3 * HIDDEN_SIZE, HIDDEN_SIZE)
        mlp_db = torch.randn(3 * HIDDEN_SIZE, dtype=torch.float16) * 0.01

        # device swizzles
        q_rpu = tp_col_swizzle_mc_weight(q, ATTN_TP).to("rpu").contiguous()
        k_rpu = tp_col_swizzle_mc_weight(k, ATTN_TP).to("rpu").contiguous()
        v_rpu = tp_col_swizzle_mc_weight(v, ATTN_TP).to("rpu").contiguous()
        o_rpu = tp_row_swizzle_mc_weight(o, ATTN_TP).to("rpu").contiguous()
        gate_rpu = tp_col_swizzle_mc_weight(gate, NUM_CORES).to("rpu").contiguous()
        up_rpu = tp_col_swizzle_mc_weight(up, NUM_CORES).to("rpu").contiguous()
        down_rpu = tp_row_swizzle_mc_weight(down, NUM_CORES).to("rpu").contiguous()
        attn_dw_rpu = transform_linear_weight(attn_dw, partition=0, num_cores=NUM_CORES).to("rpu").contiguous()
        attn_db_rpu = attn_db.to("rpu").contiguous()
        mlp_dw_rpu = transform_linear_weight(mlp_dw, partition=0, num_cores=NUM_CORES).to("rpu").contiguous()
        mlp_db_rpu = mlp_db.to("rpu").contiguous()

        layers.append({
            "raw": {
                "q": q, "k": k, "v": v, "o": o,
                "gate": gate, "up": up, "down": down,
                "attn_dw": attn_dw, "attn_db": attn_db,
                "mlp_dw": mlp_dw, "mlp_db": mlp_db,
            },
            "rpu": {
                "q": q_rpu, "k": k_rpu, "v": v_rpu, "o": o_rpu,
                "gate": gate_rpu, "up": up_rpu, "down": down_rpu,
                "attn_dw": attn_dw_rpu, "attn_db": attn_db_rpu,
                "mlp_dw": mlp_dw_rpu, "mlp_db": mlp_db_rpu,
            },
        })
    return hidden, cond, layers


def _v3_multilayer_forward(handle, hidden, cond, layers, cos_rpu, sin_rpu, position, attn_mask=None):
    'Drive `torch.ops.rpu.adarms_forward` with N-layer set_weights + 1 forward call.'
    torch.ops.rpu.spm_alloc_reset_temporary()
    rpu_lists = {k: [layers[L]["rpu"][k] for L in range(NUM_LAYERS)] for k in layers[0]["rpu"]}
    torch.ops.rpu.adarms_set_weights(
        handle,
        rpu_lists["q"], rpu_lists["k"], rpu_lists["v"], rpu_lists["o"],
        rpu_lists["gate"], rpu_lists["up"], rpu_lists["down"],
        rpu_lists["attn_dw"], rpu_lists["attn_db"],
        rpu_lists["mlp_dw"], rpu_lists["mlp_db"],
        cos_rpu, sin_rpu,
        NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
        HIDDEN_SIZE, INTERMEDIATE_SIZE, EPS,
        [], [], [], [], [], [], [],
    )
    kv = RPUCache(
        num_layers=NUM_LAYERS, batch_size=1, max_seq_len=2048,
        num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, attn_tp=ATTN_TP,
    )
    h_rpu = hidden.to("rpu").contiguous()
    c_rpu = cond.to("rpu").contiguous()
    is_causal = attn_mask is None
    out = torch.ops.rpu.adarms_forward(
        handle, h_rpu, c_rpu, kv.k_caches, kv.v_caches, attn_mask, position, is_causal,
    )
    return out.float().cpu()


def _cpu_multilayer_reference(hidden, cond, layers, cos_fp32, sin_fp32, position, attention_mask=None):
    'Apply the per-layer surgical CPU fp32 reference NUM_LAYERS times sequentially.'
    h = hidden
    for L in range(NUM_LAYERS):
        out = _cpu_reference_adarms_decoder_layer(
            h, cond, layers[L]["raw"], cos_fp32, sin_fp32, position, attention_mask=attention_mask,
        )
        h = out.to(dtype=torch.float16)  # match device per-layer fp16 stream truncation
    return h.to(dtype=torch.float32)


@pytest.fixture
def adarms_handle():
    'Function-scoped handle: create + cleanup with try/finally + SPM reset.'
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


def test_adarms_multilayer_v3_vs_cpu_fp32(adarms_handle):
    'v3 device N-layer vs surgical CPU fp32 reference; max_diff ≤ MAX_DIFF_MULTI.'
    hidden, cond, layers = _setup_multilayer_weights()
    cos_rpu, sin_rpu, cos_fp32, sin_fp32 = _make_rope_freqs(HEAD_DIM, MAX_POS)
    position = 0

    out_rpu = _v3_multilayer_forward(adarms_handle, hidden, cond, layers, cos_rpu, sin_rpu, position)
    out_cpu = _cpu_multilayer_reference(hidden, cond, layers, cos_fp32, sin_fp32, position)

    diff = (out_rpu - out_cpu).abs()
    max_diff = diff.max().item()
    mse = diff.pow(2).mean().item()
    print(
        f"\n[adarms_multi N={NUM_LAYERS} seq_len={SEQ_LEN} pos={position}] "
        f"max_diff={max_diff:.6f} mse={mse:.6f} "
        f"rpu_norm={out_rpu.norm():.4f} cpu_norm={out_cpu.norm():.4f} "
        f"budget={MAX_DIFF_MULTI:.6f} (= {MAX_DIFF_SINGLE} * sqrt({NUM_LAYERS}))"
    )
    assert max_diff <= MAX_DIFF_MULTI, (
        f"AdaRMS multi-layer N={NUM_LAYERS}: max_diff={max_diff:.6f} > "
        f"MAX_DIFF_MULTI={MAX_DIFF_MULTI:.6f} "
        f"(= {MAX_DIFF_SINGLE} * sqrt({NUM_LAYERS}))"
    )
    assert mse <= MAX_MSE_MULTI, (
        f"AdaRMS multi-layer N={NUM_LAYERS}: mse={mse:.6f} > "
        f"MAX_MSE_MULTI={MAX_MSE_MULTI:.6f} "
        f"(= {MAX_MSE_SINGLE} * {NUM_LAYERS}; structural-correctness guard)"
    )


def test_adarms_multilayer_zero_mask_matches_no_mask(adarms_handle):
    'An all-zeros bidirectional mask should be a no-op vs no-mask (both = full attention).'
    hidden, cond, layers = _setup_multilayer_weights()
    cos_rpu, sin_rpu, _, _ = _make_rope_freqs(HEAD_DIM, MAX_POS)

    out_nomask = _v3_multilayer_forward(adarms_handle, hidden, cond, layers, cos_rpu, sin_rpu, 0)

    # Recreate a fresh handle for the mask run (graph cache keys diverge with mask present)
    torch.ops.rpu.adarms_destroy(adarms_handle)
    new_handle = torch.ops.rpu.adarms_create()
    try:
        kv_len = SEQ_LEN
        mask = torch.zeros(1, 1, SEQ_LEN, kv_len, dtype=torch.float16).cpu().contiguous()
        out_mask = _v3_multilayer_forward(new_handle, hidden, cond, layers, cos_rpu, sin_rpu, 0, attn_mask=mask)
    finally:
        torch.ops.rpu.adarms_destroy(new_handle)
        torch.ops.rpu.spm_alloc_reset_temporary()

    diff = (out_nomask - out_mask).abs()
    max_diff = diff.max().item()
    print(
        f"\n[adarms_multi mask-vs-nomask] max_diff={max_diff:.6f} "
        f"nomask_norm={out_nomask.norm():.4f} mask_norm={out_mask.norm():.4f}"
    )
    # All-zero mask should produce identical output up to fp16 numerical reordering
    assert max_diff <= MAX_DIFF_MULTI, (
        f"All-zero mask should be no-op: max_diff={max_diff:.6f} > {MAX_DIFF_MULTI:.6f}"
    )
