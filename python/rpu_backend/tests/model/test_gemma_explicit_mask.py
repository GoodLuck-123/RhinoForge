#!/usr/bin/env python
'Gemma explicit mask device regression.'
from __future__ import annotations

import pytest
import torch

import rpu_backend  # noqa: F401  # side effect: torch.ops.rpu / torch.rpu binding
from rpu_backend.api.cache import RPUCache
from rpu_backend.graph import GraphCache, GraphSignature
from rpu_backend.runtime.weights import (
    NUM_CORES,
    tp_col_swizzle_mc_weight,
    tp_row_swizzle_mc_weight,
)

# Pi0.5 Gemma VLM geometry, shrunk on the layer/intermediate axes only.
HIDDEN = 2048
NUM_Q_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 256
INTERMEDIATE = 4096
NUM_LAYERS = 2
EPS = 1e-6
ATTN_TP = min(8, NUM_KV_HEADS)
SEQ_LEN = 64
MAX_SEQ = 256

NEG_INF = float("-inf")


def _rpu_available() -> bool:
    try:
        torch.zeros(1, dtype=torch.float16).to("rpu")
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _rpu_available(), reason="device device unavailable (needs root + /dev/rpu)"
)


def _rope_tables():
    inv_freq = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    freqs = torch.einsum("i,j->ij", torch.arange(MAX_SEQ).float(), inv_freq)
    cos = freqs.cos().to(torch.float16).to("rpu").contiguous()
    sin = freqs.sin().to(torch.float16).to("rpu").contiguous()
    return cos, sin


def _build_handle():
    torch.manual_seed(1234)

    def mk(out_f, in_f):
        return torch.randn(out_f, in_f, dtype=torch.float16) * 0.02

    def mk_norm():
        return (torch.randn(HIDDEN, dtype=torch.float16) * 0.02).to("rpu").contiguous()

    q_w, k_w, v_w, o_w = [], [], [], []
    gate, up, down = [], [], []
    in_norm, post_norm = [], []
    for _ in range(NUM_LAYERS):
        q_w.append(tp_col_swizzle_mc_weight(
            mk(NUM_Q_HEADS * HEAD_DIM, HIDDEN), ATTN_TP).to("rpu").contiguous())
        k_w.append(tp_col_swizzle_mc_weight(
            mk(NUM_KV_HEADS * HEAD_DIM, HIDDEN), ATTN_TP).to("rpu").contiguous())
        v_w.append(tp_col_swizzle_mc_weight(
            mk(NUM_KV_HEADS * HEAD_DIM, HIDDEN), ATTN_TP).to("rpu").contiguous())
        o_w.append(tp_row_swizzle_mc_weight(
            mk(HIDDEN, NUM_Q_HEADS * HEAD_DIM), ATTN_TP).to("rpu").contiguous())
        gate.append(tp_col_swizzle_mc_weight(
            mk(INTERMEDIATE, HIDDEN), NUM_CORES).to("rpu").contiguous())
        up.append(tp_col_swizzle_mc_weight(
            mk(INTERMEDIATE, HIDDEN), NUM_CORES).to("rpu").contiguous())
        down.append(tp_row_swizzle_mc_weight(
            mk(HIDDEN, INTERMEDIATE), NUM_CORES).to("rpu").contiguous())
        in_norm.append(mk_norm())
        post_norm.append(mk_norm())

    cos, sin = _rope_tables()
    final_norm = mk_norm()

    handle = int(torch.ops.rpu.gemma_create())
    empty: list[torch.Tensor] = []
    torch.ops.rpu.gemma_set_weights(
        handle, q_w, k_w, v_w, o_w, in_norm, post_norm, gate, up, down,
        cos, sin, final_norm,
        NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, HIDDEN, INTERMEDIATE, EPS,
        empty, empty, empty, empty, empty, empty, empty)
    return handle


def _make_cache() -> RPUCache:
    return RPUCache(
        num_layers=NUM_LAYERS, batch_size=1, max_seq_len=MAX_SEQ,
        num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, attn_tp=ATTN_TP,
    )


def _causal_additive_mask() -> torch.Tensor:
    '[SEQ, SEQ] additive mask: 0 where attendable, -inf above the diagonal.'
    m = torch.zeros(SEQ_LEN, SEQ_LEN, dtype=torch.float32)
    m.masked_fill_(torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, dtype=torch.bool), 1), NEG_INF)
    return m.to(torch.float16).contiguous()


def _run_leg(handle, hidden, mask, is_causal, tag):
    'Run one leg twice through its own GraphCache: BUILD then REPLAY.'
    cache = _make_cache()
    gc = GraphCache()
    sig = GraphSignature(
        op_id=f"test_gemma_explicit_mask::{tag}",
        shapes=[SEQ_LEN, NUM_LAYERS, HIDDEN],
        dyn_dims=[NUM_Q_HEADS, 0],
        dtypes=[torch.float16],
    )
    outs = []
    for _ in range(2):
        cache.reset_to_position(0)
        with gc.capture(sig):
            out = torch.ops.rpu.gemma_forward(
                handle, hidden, cache.k_caches, cache.v_caches,
                mask, 0, is_causal, 0)
        # Read AFTER the capture scope exits (the graph runs in end()).
        outs.append(out.detach().to(torch.float32).cpu().clone())
        torch.ops.rpu.spm_alloc_reset_temporary()
    assert torch.equal(outs[0], outs[1]), (
        f"{tag}: BUILD and REPLAY disagree "
        f"(max |delta| {(outs[0] - outs[1]).abs().max().item():.6g})")
    assert torch.isfinite(outs[0]).all(), f"{tag}: non-finite output"
    assert outs[0].abs().max().item() > 0.0, f"{tag}: all-zero output"
    return outs[0]


def test_gemma_honours_explicit_attention_mask():
    handle = _build_handle()
    try:
        torch.manual_seed(7)
        hidden = (torch.randn(1, SEQ_LEN, HIDDEN, dtype=torch.float16) * 0.5)
        hidden = hidden.to("rpu").contiguous()

        causal_mask = _causal_additive_mask()
        full_mask = torch.zeros(SEQ_LEN, SEQ_LEN, dtype=torch.float16).contiguous()

        implicit = _run_leg(handle, hidden, None, True, "implicit_causal")
        expl_causal = _run_leg(handle, hidden, causal_mask, True, "explicit_causal")
        expl_full = _run_leg(handle, hidden, full_mask, True, "explicit_full")
        expl_full_nc = _run_leg(handle, hidden, full_mask, False, "explicit_full_nc")

        def rel(a, b):
            return ((a - b).norm() / (b.norm() + 1e-8)).item()

        # ---- Gate 1 (correctness): an explicit CAUSAL mask must reproduce the
        # implicit causal path. Fails if masked columns land as additive 0
        # instead of -inf.
        rel_causal = rel(expl_causal, implicit)
        assert rel_causal < 1e-3, (
            "explicit causal mask does not reproduce the implicit causal path "
            f"(relative L2 {rel_causal:.6g}) — the mask is honoured but WRONG")

        # ---- Gate 2 (the defect): a full-attend mask must change the answer.
        # Before the fix this was bit-identical to `implicit`.
        rel_full = rel(expl_full, implicit)
        assert rel_full > 1e-2, (
            "explicit full-attend mask produced the causal answer "
            f"(relative L2 {rel_full:.6g}) — GemmaModel::forward is discarding "
            "the caller's attention_mask when is_causal is also set")

        # ---- Gate 3: is_causal must not change what an explicit mask means.
        assert torch.equal(expl_full, expl_full_nc), (
            "explicit mask + is_causal=True differs from the same mask with "
            f"is_causal=False (max |delta| "
            f"{(expl_full - expl_full_nc).abs().max().item():.6g})")

        print(f"[gemma explicit mask] rel(explicit_causal, implicit)={rel_causal:.3e} "
              f"rel(explicit_full, implicit)={rel_full:.3e}")
    finally:
        torch.ops.rpu.gemma_destroy(handle)


if __name__ == "__main__":  # script-mode invocation (CI runs some tests as scripts)
    test_gemma_honours_explicit_attention_mask()
    print("PASS")
