#!/usr/bin/env python3
'Chunk policy keys the graph device regression.'

import argparse
import sys

import torch

import rpu_backend  # noqa: F401  (registers torch.rpu device)
from rpu_backend import RPUCache
from rpu_backend.adapters.qwen3 import lookup_causal_decoder
from rpu_backend.runtime.decoder import (
    _install_causal_decoder_forward as patch_qwen3_for_rpu,
)

# The loader is the one from the P3 gate rather than a sixth copy of it; that
# file is import-safe (everything runs under __main__).
from test_qwen3_same_sig_two_position import (  # noqa: E402
    convert_and_move_to_rpu,
    load_cpu_model,
    patch_classes,
    setup_rpu,
)

SEQ = 512           # under qwen3-0.6b's certified envelope (adapters/qwen3.py:67, max_kv 1024)
POLICY_A = 256      # both are SDPA-valid on this geometry and both BIND (asserted below)
POLICY_B = 128
COS_THRESHOLD = 0.999


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().cpu().reshape(-1)
    b = b.float().cpu().reshape(-1)
    return (a @ b / (a.norm() * b.norm())).item()


def set_chunk_policy(handle, chunk_size, legacy_global=False):
    "Point this handle's chunk authority at `chunk_size`."
    if legacy_global:


        # this raises ImportError/AttributeError on a current one — which is
        # itself the migration being complete.
        from rpu_backend.runtime import control
        control.set_chunk_size(chunk_size)
        return
    op = getattr(torch.ops.rpu, "causal_decoder_set_chunk_size_override", None)
    if op is None:
        raise RuntimeError(
            "causal_decoder_set_chunk_size_override is missing: this binary "
            "does not provide per-handle chunk selection; this "
            "installs. Re-run with --legacy-global to reproduce the pre-fix "
            "behaviour through runtime.control.set_chunk_size.")
    op(handle, chunk_size)


def resolve(handle, seq_len, position=0):
    return int(torch.ops.rpu.causal_decoder_resolve_prefill_chunk_size(
        handle, seq_len, position))


def prefill(backbone, cache, ids):
    cache.reset()
    with torch.no_grad():
        out = backbone(input_ids=ids, past_key_values=cache, use_cache=True)
    return out.last_hidden_state[0].float().cpu().clone()


def stats(cache_obj):
    snap = list(cache_obj.snapshot())
    return {
        "entries": len(snap),
        "replays": sum(int(getattr(e, "replay_count", 0)) for e in snap),
        "recaptures": sum(int(getattr(e, "recapture_count", 0)) for e in snap),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy-global", action="store_true",
                    help="drive the legacy process-global setting instead of the "
                         "per-handle setter (reproduces the red)")
    args = ap.parse_args()
    legacy = args.legacy_global

    print("=" * 70)
    print("Chunk policy must key the graph signature")
    print(f"route = {'process global (PRE-FIX)' if legacy else 'per-handle setter'}")
    print("=" * 70)

    setup_rpu()
    patch_classes()
    tokenizer, model = load_cpu_model()
    del tokenizer
    model = convert_and_move_to_rpu(model)
    patch_qwen3_for_rpu(
        model.model, arch="qwen3", chunk_envelope_for=lookup_causal_decoder)

    backbone = model.model
    handle = backbone._rpu_decoder_handle
    cache_obj = backbone._rpu_decoder_graph_cache
    config = model.config
    head_dim = getattr(config, "head_dim",
                       config.hidden_size // config.num_attention_heads)
    kv = RPUCache(
        num_layers=config.num_hidden_layers,
        batch_size=1,
        max_seq_len=SEQ + 16,
        num_kv_heads=config.num_key_value_heads,
        head_dim=head_dim,
        device="rpu",
        dtype=torch.float16,
    )
    # Deterministic ids on the CPU, then moved: torch.empty(device='rpu') is
    # seedless, so a tensor built directly on the device differs every run.
    torch.manual_seed(0)
    ids = torch.randint(0, int(config.vocab_size), (1, SEQ)).to("rpu")

    fails = []

    # ── 0. The two policies must actually differ, or nothing below asserts ──
    set_chunk_policy(handle, POLICY_A, legacy)
    got_a = resolve(handle, SEQ)
    set_chunk_policy(handle, POLICY_B, legacy)
    got_b = resolve(handle, SEQ)
    print(f"\nresolved chunk: policy {POLICY_A} -> {got_a}, "
          f"policy {POLICY_B} -> {got_b}")
    if got_a == got_b:
        print(f"[FAIL] both policies resolve to {got_a}; this test would "
              f"assert nothing. Pick two chunk sizes that bind on this geometry.")
        return 1

    # ── 1. Policy A: BUILD then REPLAY must be byte-identical ──────────────
    set_chunk_policy(handle, POLICY_A, legacy)
    out_a1 = prefill(backbone, kv, ids)
    s1 = stats(cache_obj)
    out_a2 = prefill(backbone, kv, ids)
    s2 = stats(cache_obj)
    print(f"\nafter A build : {s1}")
    print(f"after A replay: {s2}")
    if not torch.equal(out_a1, out_a2):
        fails.append("policy A REPLAY did not reproduce its own BUILD "
                     "byte-for-byte (the baseline is already unstable)")
    if s2["entries"] != s1["entries"]:
        fails.append(f"policy A grew the cache on replay: "
                     f"{s1['entries']} -> {s2['entries']}")

    # ── 2. Policy B at the SAME shape must be a SECOND signature ───────────
    set_chunk_policy(handle, POLICY_B, legacy)
    out_b = prefill(backbone, kv, ids)
    s3 = stats(cache_obj)
    print(f"after B build : {s3}")
    if s3["entries"] != s2["entries"] + 1:
        fails.append(
            f"same shape under a different chunk policy did NOT produce a "
            f"second signature: entries {s2['entries']} -> {s3['entries']} "
            f"(expected {s2['entries'] + 1}). The prefill signature does not "
            f"encode the chunk plan, so policy B REPLAYED policy A's graph "
            f"under policy B's SPM layout.")

    # ── 3. Chunking partitions the same math — the answers must agree ──────
    cs_ab = cos_sim(out_a1, out_b)
    print(f"\ncos(policy {POLICY_A}, policy {POLICY_B}) = {cs_ab:.6f}")
    if cs_ab < COS_THRESHOLD:
        fails.append(f"chunk policy changed the answer: cos {cs_ab:.6f} < "
                     f"{COS_THRESHOLD}")

    # ── 4. A -> B -> A must return to A's graph, not rebuild it ────────────
    set_chunk_policy(handle, POLICY_A, legacy)
    out_a3 = prefill(backbone, kv, ids)
    s4 = stats(cache_obj)
    print(f"after A again : {s4}")
    if not torch.equal(out_a1, out_a3):
        fails.append("A->B->A did not reproduce A's first output byte-for-byte "
                     "(the returning replay ran under the wrong SPM layout)")
    if s4["entries"] != s3["entries"]:
        fails.append(f"returning to policy A built a THIRD entry: "
                     f"{s3['entries']} -> {s4['entries']}")
    if s4["recaptures"] != 0:
        fails.append(f"recapture_count = {s4['recaptures']}, expected 0 — the "
                     f"two policies are thrashing one slot instead of holding "
                     f"two")

    print()
    print("=" * 70)
    if fails:
        print("TEST FAILED:")
        for msg in fails:
            print(f"  [FAIL] {msg}")
        print("=" * 70)
        return 1
    print(f"TEST PASSED  (2 signatures at one shape, {s4['recaptures']} "
          f"recaptures, A->B->A byte-identical, cos {cs_ab:.6f})")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
