#!/usr/bin/env python3
'Adarms same sig two position device regression.'

import sys

import torch

import rpu_backend  # noqa: F401  (registers torch.rpu device + GraphCache API)
from rpu_backend.api.cache import RPUCache

from rpu_backend.tests.model.test_adarms_multilayer import (
    NUM_LAYERS, SEQ_LEN, _setup_multilayer_weights,
)
from rpu_backend.tests.model.test_adarms_single_layer import (
    HIDDEN_SIZE, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_DIM,
    INTERMEDIATE_SIZE, EPS, ATTN_TP, MAX_POS,
    _make_rope_freqs,
)


# POS_B must satisfy BOTH LTM-mask SDPA constraints:
#   (1) position % seq_q == 0  (rpu_sdpa.cpp ~ "must be multiple of seq_q")
#   (2) position == 0 OR position % 16 == 0  (v16-alignment)
# With SEQ_LEN=50 (imported from test_adarms_multilayer fixture), this means
# POS_B must be a multiple of LCM(50, 16) = 400. POS_B=400 also crosses
# a v16 SDPA bucket (POS_A=0 → seq_k_v16=CeilDiv(50,16)=4 vs
# POS_B=400 → seq_k_v16=CeilDiv(450,16)=29) so all position-derived
# register paths must mutate correctly for the REPLAY output to match the
# fresh-BUILD reference.
POS_A = 0
POS_B = 400

# device-vs-device comparison threshold. P2 sync_mutable_params writes the same
# register values a fresh BUILD would write, so we expect bit-identical or
# near-bit-identical outputs. 0.9999 leaves headroom for any incidental
# float reordering across BUILD vs REPLAY scheduling.
COS_SIM_THRESHOLD = 0.9999


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().cpu().view(-1)
    b = b.float().cpu().view(-1)
    return (a @ b / (a.norm() * b.norm())).item()


def _setup_handle(layers, cos_rpu, sin_rpu):
    handle = torch.ops.rpu.adarms_create()
    rpu_lists = {k: [layers[L]["rpu"][k] for L in range(NUM_LAYERS)]
                 for k in layers[0]["rpu"]}
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
    return handle


def _build_sig(seq_len, include_position, position):
    'Mirror the sig structure S1 will install in adapters/pi05/adarms.py.'
    if include_position:
        return rpu_backend.graph.GraphSignature(
            op_id="pi05_adarms_gate_with_pos",
            shapes=[seq_len, HIDDEN_SIZE, int(position)],
            dyn_dims=[NUM_LAYERS, NUM_Q_HEADS],
            dtypes=[torch.float16],
        )
    return rpu_backend.graph.GraphSignature(
        op_id="pi05_adarms_gate_drop_pos",
        shapes=[seq_len, HIDDEN_SIZE],
        dyn_dims=[NUM_LAYERS, NUM_Q_HEADS],
        dtypes=[torch.float16],
    )


def _call_once(handle, cache, hidden, cond, position, graph_cache, include_position):
    'Mimics adapter forward post-S1: wrap adarms_forward in cache.capture(sig).'
    cache.reset_to_position(position)
    seq_len = hidden.shape[1]
    h_rpu = hidden.to("rpu").contiguous()
    c_rpu = cond.to("rpu").contiguous()
    sig = _build_sig(seq_len, include_position, position)
    with graph_cache.capture(sig):
        out = torch.ops.rpu.adarms_forward(
            handle, h_rpu, c_rpu, cache.k_caches, cache.v_caches,
            None,        # attention_mask → MASK_LTM (causal, no DMA)
            position,
            True,        # is_causal
        )
    torch.ops.rpu.spm_alloc_reset_temporary()
    cache.update_position(seq_len)
    return out.float().cpu()


def _segment_activity(snap):
    'Sum of (hit + miss) over all entries.'
    return sum(e.prepared_segment_hit_total + e.prepared_segment_miss_total
               for e in snap)


def _two_position_run(handle, cache, hidden, cond, include_position):
    graph_cache = rpu_backend.graph.GraphCache()
    n_initial = len(list(graph_cache.snapshot()))

    out_a = _call_once(handle, cache, hidden, cond, POS_A,
                       graph_cache, include_position)
    snap_a = list(graph_cache.snapshot())
    n_after_a = len(snap_a)
    activity_after_a = _segment_activity(snap_a)
    replay_after_a = sum(e.replay_count for e in snap_a)

    out_b = _call_once(handle, cache, hidden, cond, POS_B,
                       graph_cache, include_position)
    snap_b = list(graph_cache.snapshot())
    n_after_b = len(snap_b)
    activity_after_b = _segment_activity(snap_b)
    replay_after_b = sum(e.replay_count for e in snap_b)

    return {
        "out_a":            out_a,
        "out_b":            out_b,
        "n_initial":        n_initial,
        "n_after_a":        n_after_a,
        "n_after_b":        n_after_b,
        "activity_after_a": activity_after_a,
        "activity_after_b": activity_after_b,
        "activity_delta_b": activity_after_b - activity_after_a,
        "replay_after_a":   replay_after_a,
        "replay_after_b":   replay_after_b,
        "replay_delta_b":   replay_after_b - replay_after_a,
    }


def _run_variant(label, layers, cos_rpu, sin_rpu, hidden, cond, include_position):
    print(f"\n[Variant {label}] sig "
          f"{'INCLUDES' if include_position else 'DROPS'} position ...")
    handle = _setup_handle(layers, cos_rpu, sin_rpu)
    cache = RPUCache(num_layers=NUM_LAYERS, batch_size=1, max_seq_len=2048,
                     num_kv_heads=NUM_KV_HEADS, head_dim=HEAD_DIM,
                     attn_tp=ATTN_TP)
    try:
        r = _two_position_run(handle, cache, hidden, cond, include_position)
    finally:
        try:
            torch.ops.rpu.adarms_destroy(handle)
        except Exception:
            pass
        try:
            torch.ops.rpu.spm_alloc_reset_temporary()
        except Exception:
            pass
    print(f"  entries: initial={r['n_initial']}, "
          f"after-A={r['n_after_a']}, after-B={r['n_after_b']}; "
          f"segment_activity (hit+miss) after-A={r['activity_after_a']}, "
          f"after-B={r['activity_after_b']} "
          f"(delta_B={r['activity_delta_b']}); "
          f"replay_count after-A={r['replay_after_a']}, "
          f"after-B={r['replay_after_b']} "
          f"(delta_B={r['replay_delta_b']})")
    return r


def main() -> int:
    print("=" * 72)
    print("S1 gate: AdaRMS same-sig two-position cache hit")
    print(f"  POS_A={POS_A}, POS_B={POS_B}, SEQ_LEN={SEQ_LEN}, "
          f"NUM_LAYERS={NUM_LAYERS}")
    print(f"  cos_sim threshold (device-vs-device): {COS_SIM_THRESHOLD}")
    print("=" * 72)

    torch.rpu.set_profile(False)
    torch.rpu.set_caching_allocator(True)
    torch.rpu.set_ddr_flush(True)

    hidden, cond, layers = _setup_multilayer_weights()
    cos_rpu, sin_rpu, _, _ = _make_rope_freqs(HEAD_DIM, MAX_POS)

    # Variant A: sig with position — fresh BUILD each position (negative
    # control + ground-truth references for the device-vs-device correctness check).
    ra = _run_variant("A", layers, cos_rpu, sin_rpu, hidden, cond,
                      include_position=True)

    # Variant B: sig without position — BUILD at POS_A + sync-only REPLAY at POS_B.
    rb = _run_variant("B", layers, cos_rpu, sin_rpu, hidden, cond,
                      include_position=False)

    # -------------------- assertions -------------------- #
    fails = []

    # Negative control: Variant A must grow by exactly 1 per call.
    if ra["n_after_a"] != ra["n_initial"] + 1:
        fails.append(
            f"[NEG-CTRL] Variant A first call did NOT add an entry "
            f"(initial={ra['n_initial']}, after-A={ra['n_after_a']}). "
            f"GraphCache wrap mechanism may be broken.")
    if ra["n_after_b"] != ra["n_after_a"] + 1:
        fails.append(
            f"[NEG-CTRL] Variant A second call did NOT add an entry "
            f"(after-A={ra['n_after_a']}, after-B={ra['n_after_b']}). "
            f"sig-includes-position should force cache miss at different "
            f"positions.")
    if ra["activity_delta_b"] != 0:
        fails.append(
            f"[NEG-CTRL] Variant A had unexpected REPLAY activity "
            f"(delta_B={ra['activity_delta_b']}, expected 0). "
            f"Sig with position should never REPLAY across position change.")
    if ra["replay_delta_b"] != 0:
        fails.append(
            f"[NEG-CTRL] Variant A had unexpected replay_count growth "
            f"(delta_B={ra['replay_delta_b']}, expected 0).")

    # Main gate: Variant B second call MUST hit the cached entry.
    if rb["n_after_a"] != rb["n_initial"] + 1:
        fails.append(
            f"[GATE] Variant B first call did NOT add an entry "
            f"(initial={rb['n_initial']}, after-A={rb['n_after_a']}).")
    if rb["n_after_b"] != rb["n_after_a"]:
        fails.append(
            f"[GATE] Variant B GraphCache grew across positions "
            f"(after-A={rb['n_after_a']}, after-B={rb['n_after_b']}). "
            f"sig-without-position should hit cached entry at POS_B.")
    if rb["activity_delta_b"] < 1:
        fails.append(
            f"[GATE] Variant B second call had no REPLAY segment activity "
            f"(delta_B={rb['activity_delta_b']}, expected >= 1). "
            f"launch_segment_for_replay should have run at least once.")
    if rb["replay_delta_b"] < 1:
        fails.append(
            f"[GATE] Variant B REPLAY was not reflected in "
            f"GraphCache.replay_count (delta_B={rb['replay_delta_b']}, "
            f"expected >= 1).")

    # We don't isolate `prepared_segment_hit_total` from `_miss_total`:
    # AdaRMS is a 37-segment graph and only single-segment graphs hit the P2
    # sync-only fast path (`_hit_total`); multi-segment graphs always rotate
    # through `_miss_total` (full per-segment rebuild). Both paths still
    # produce correct REPLAY output (validated by the cos_sim check below).

    # Correctness: P2 sync_mutable_params at POS_B must reproduce Variant A's
    # fresh BUILD output at POS_B. Both variants use identical fake weights
    # and the same RPUCache lifecycle (fresh cache → POS_A write → POS_B
    # reset+write), so K/V cache state at SDPA-read time matches across
    # variants. Output difference can only come from position-derived
    # register desync — which P2 is supposed to fix.
    cs_a = cos_sim(rb["out_a"], ra["out_a"])
    cs_b = cos_sim(rb["out_b"], ra["out_b"])
    print(f"\nRPU-vs-device cos_sim: POS_A (BUILD vs BUILD) = {cs_a:.6f}, "
          f"POS_B (REPLAY vs fresh BUILD) = {cs_b:.6f}")
    if cs_a < COS_SIM_THRESHOLD:
        fails.append(
            f"POS_A cos_sim {cs_a:.6f} < {COS_SIM_THRESHOLD} "
            f"(BUILD-at-POS_A should be variant-invariant)")
    if cs_b < COS_SIM_THRESHOLD:
        fails.append(
            f"POS_B cos_sim {cs_b:.6f} < {COS_SIM_THRESHOLD} "
            f"(P2 REPLAY at POS_B should match fresh BUILD at POS_B)")

    print()
    print("=" * 72)
    if fails:
        print("TEST FAILED:")
        for msg in fails:
            print(f"  [FAIL] {msg}")
        print("=" * 72)
        return 1

    print(f"TEST PASSED  ("
          f"Variant-A 2 entries, "
          f"Variant-B 1 entry + replay_delta={rb['replay_delta_b']} "
          f"+ activity_delta={rb['activity_delta_b']}, "
          f"cos_sim >= {COS_SIM_THRESHOLD})")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
