#!/usr/bin/env python
'Halo image flow single chunk device regression.'
from __future__ import annotations

import os

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

# HALO geometry shrunk on every axis the kernels allow. num_kv_heads=1 keeps
# attn_tp=1, so the attention weights need no cross-core swizzle bookkeeping.
HIDDEN = 2048
NUM_Q_HEADS = 8
NUM_KV_HEADS = 1
HEAD_DIM = 256
INTERMEDIATE = 4096
NUM_LAYERS = 2          # >1 is the whole point: layer 1 is what dereferenced null
EPS = 1e-6
ATTN_TP = min(8, NUM_KV_HEADS)
QUERY_LEN = 32
MAX_SEQ = 128
PREFIX_LEN = 0


def _rpu_available() -> bool:
    try:
        torch.zeros(1, dtype=torch.float16).to("rpu")
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _rpu_available(), reason="device device unavailable (needs root + /dev/rpu)"
)


def _h(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float16).to("rpu").contiguous()


def _rope_tables():
    inv_freq = 1.0 / (10000 ** (torch.arange(0, HEAD_DIM, 2).float() / HEAD_DIM))
    freqs = torch.einsum("i,j->ij", torch.arange(MAX_SEQ).float(), inv_freq)
    return _h(freqs.cos()), _h(freqs.sin())


def _twin(seed: int):
    'One und/gen weight twin: every per-layer list the two setters want.'
    g = torch.Generator().manual_seed(seed)

    def mk(out_f, in_f):
        return torch.randn(out_f, in_f, dtype=torch.float32, generator=g).half() * 0.02

    def vec(n):
        return torch.randn(n, dtype=torch.float32, generator=g).half() * 0.02

    w = {k: [] for k in ("q_w", "k_w", "v_w", "o_w", "q_norm", "k_norm",
                         "input_norm", "post_norm", "gate_w", "up_w", "down_w",
                         "q_bias", "k_bias", "v_bias")}
    for _ in range(NUM_LAYERS):
        w["q_w"].append(_h(tp_col_swizzle_mc_weight(
            mk(NUM_Q_HEADS * HEAD_DIM, HIDDEN), ATTN_TP)))
        w["k_w"].append(_h(tp_col_swizzle_mc_weight(
            mk(NUM_KV_HEADS * HEAD_DIM, HIDDEN), ATTN_TP)))
        w["v_w"].append(_h(tp_col_swizzle_mc_weight(
            mk(NUM_KV_HEADS * HEAD_DIM, HIDDEN), ATTN_TP)))
        w["o_w"].append(_h(tp_row_swizzle_mc_weight(
            mk(HIDDEN, NUM_Q_HEADS * HEAD_DIM), ATTN_TP)))
        w["gate_w"].append(_h(tp_col_swizzle_mc_weight(
            mk(INTERMEDIATE, HIDDEN), NUM_CORES)))
        w["up_w"].append(_h(tp_col_swizzle_mc_weight(
            mk(INTERMEDIATE, HIDDEN), NUM_CORES)))
        w["down_w"].append(_h(tp_row_swizzle_mc_weight(
            mk(HIDDEN, INTERMEDIATE), NUM_CORES)))
        w["q_norm"].append(_h(vec(HEAD_DIM)))
        w["k_norm"].append(_h(vec(HEAD_DIM)))
        w["input_norm"].append(_h(vec(HIDDEN)))
        w["post_norm"].append(_h(vec(HIDDEN)))
        w["q_bias"].append(_h(vec(NUM_Q_HEADS * HEAD_DIM)))
        w["k_bias"].append(_h(vec(NUM_KV_HEADS * HEAD_DIM)))
        w["v_bias"].append(_h(vec(NUM_KV_HEADS * HEAD_DIM)))
    w["final_norm_w"] = _h(vec(HIDDEN))
    return w


def _build_handle():
    und = _twin(11)
    gen = _twin(22)
    cos, sin = _rope_tables()

    # 1 = und/base row (HALO: the two start/end_of_image text tokens), 0 = gen.
    row_mask = torch.zeros(QUERY_LEN, dtype=torch.float16)
    row_mask[0] = 1.0
    row_mask[-1] = 1.0
    row_mask = _h(row_mask)

    handle = int(torch.ops.rpu.halo_image_flow_create())
    try:
        torch.ops.rpu.halo_image_flow_set_weights(
            handle,
            und["q_w"], und["k_w"], und["v_w"], und["o_w"],
            und["q_norm"], und["k_norm"],
            und["input_norm"], und["post_norm"],
            und["gate_w"], und["up_w"], und["down_w"],
            cos, sin, und["final_norm_w"],
            NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM, HIDDEN, INTERMEDIATE, EPS,
            True,  # use_silu
            und["q_bias"], und["k_bias"], und["v_bias"])
        torch.ops.rpu.halo_image_flow_set_gen_weights(
            handle,
            gen["q_w"], gen["k_w"], gen["v_w"], gen["o_w"],
            gen["q_norm"], gen["k_norm"],
            gen["input_norm"], gen["post_norm"],
            gen["gate_w"], gen["up_w"], gen["down_w"],
            gen["q_bias"], gen["k_bias"], gen["v_bias"],
            gen["final_norm_w"], row_mask, QUERY_LEN)
        # Production exposes AUTO only for ImageFlow.  This diagnostic opts
        # into FusedModelBase's explicit unsafe-exact seam below, but still
        # needs the real cold max-sequence envelope that production sets.
        torch.ops.rpu.halo_image_flow_set_chunk_envelope(handle, MAX_SEQ, 0)
    except BaseException:
        torch.ops.rpu.halo_image_flow_destroy(handle)
        raise
    return handle, cos, sin


def _make_cache() -> RPUCache:
    return RPUCache(
        num_layers=NUM_LAYERS, batch_size=1, max_seq_len=MAX_SEQ,
        num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM, attn_tp=ATTN_TP,
    )


def _run_leg(handle, x_emb, cos, sin, chunk_size, tag):
    'Run one chunking plan twice through its own GraphCache: BUILD then REPLAY.'
    cache = _make_cache()
    gc = GraphCache()
    sig = GraphSignature(
        op_id=f"test_halo_image_flow_single_chunk::{tag}",
        shapes=[QUERY_LEN, NUM_LAYERS, HIDDEN],
        dyn_dims=[NUM_Q_HEADS, chunk_size],
        dtypes=[torch.float16],
    )
    outs = []

    # forward, so this had to set/restore it around each call. It is this
    # handle's own state now — set it once, no restore to get wrong.
    torch.ops.rpu.halo_image_flow_set_chunk_size_override(handle, chunk_size)
    for _ in range(2):
        with gc.capture(sig):
            out = torch.ops.rpu.halo_image_flow_step_forward(
                handle, x_emb, cache.k_caches, cache.v_caches,
                cos, sin, PREFIX_LEN)
        # Read AFTER the capture scope exits (the graph runs in end()).
        outs.append(out.detach().to(torch.float32).cpu().clone())
        torch.ops.rpu.spm_alloc_reset_temporary()
    assert torch.equal(outs[0], outs[1]), (
        f"{tag}: BUILD and REPLAY disagree "
        f"(max |delta| {(outs[0] - outs[1]).abs().max().item():.6g})")
    assert torch.isfinite(outs[0]).all(), f"{tag}: non-finite output"
    assert outs[0].abs().max().item() > 0.0, f"{tag}: all-zero output"
    return outs[0]


def test_halo_image_flow_single_chunk_spm_resident():
    if os.getenv("RPU_CHUNK_FORCE_UNSAFE") not in {"1", "true"}:
        pytest.skip(
            "diagnostic exact chunks require RPU_CHUNK_FORCE_UNSAFE=1; "
            "production HALO ImageFlow intentionally exposes AUTO only")

    torch.manual_seed(31337)
    x_emb = _h(torch.randn(1, QUERY_LEN, HIDDEN, dtype=torch.float32) * 0.5)

    # The execution envelope is cold and freezes after first dispatch.  Use
    # separate deterministic-weight handles instead of hot-mutating one live
    # Graph owner between physical plans.
    handle, cos, sin = _build_handle()
    try:
        multi = _run_leg(handle, x_emb, cos, sin, 16, "multi_chunk")
    finally:
        torch.ops.rpu.halo_image_flow_destroy(handle)

    handle, cos, sin = _build_handle()
    try:
        single = _run_leg(handle, x_emb, cos, sin, QUERY_LEN, "single_chunk")
    finally:
        torch.ops.rpu.halo_image_flow_destroy(handle)

    rel = ((single - multi).norm() / (multi.norm() + 1e-8)).item()
    assert rel < 1e-2, (
        "single-chunk (SPM_RESIDENT) output diverges from the multi-chunk "
        f"(DDR ping-pong) reference: relative L2 {rel:.6g} — the skipped "
        "layer-input DMA was load-bearing, not redundant")
    print(f"[halo image_flow single-chunk] rel(single, multi)={rel:.3e}")


if __name__ == "__main__":  # script-mode invocation
    test_halo_image_flow_single_chunk_spm_resident()
    print("PASS")
