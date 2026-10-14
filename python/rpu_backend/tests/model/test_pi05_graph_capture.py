#!/usr/bin/env python3
'Pi05 graph capture device regression.'
from __future__ import annotations
import argparse
import os
import sys
import time
import traceback
import types

import torch
import rpu_backend
from rpu_backend.model_registry import model_path


# Stub lerobot 0.4.4 strict transformers-replace check before importing lerobot.
# (modeling_pi05.PI05Pytorch.__init__ demands a patched SigLIP module that we
# don't ship in torch_yqj.) device adapter overrides SigLIP forward via
# _rpu_siglip_forward at .to('rpu') time, so the replacement's runtime behavior
# is irrelevant for this graph capture test.
def _stub_lerobot_transformers_check():
    import transformers.models.siglip as _sl
    stub = types.ModuleType('transformers.models.siglip.check')
    stub.check_whether_transformers_replace_is_installed_correctly = lambda: True
    sys.modules['transformers.models.siglip.check'] = stub
    _sl.check = stub


_stub_lerobot_transformers_check()


def setup_env():
    torch.rpu.set_caching_allocator(True)
    torch.rpu.set_ddr_flush(True)
    torch.rpu.set_debug(False)
    torch.rpu.set_profile(False)


def _build_dummy_batch():
    'Dummy zeros batch matching Pi0.5 hardware-smoke shape (bs=1, 224×224).'
    from lerobot.utils.constants import (
        OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK,
    )
    return {
        "observation.images.image": torch.zeros(1, 3, 224, 224, dtype=torch.float32),
        "observation.images.wrist_image": torch.zeros(1, 3, 224, 224, dtype=torch.float32),
        "observation.state": torch.zeros(1, 8, dtype=torch.float32),
        OBS_LANGUAGE_TOKENS: torch.zeros(1, 32, dtype=torch.int64),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 32, dtype=torch.bool),
        "task": ["pick up the object"],
    }


def load_policy():
    'Load Pi0.5 model on device. Prefer pi05-libero-finetuned, fall back to pi05-base.'
    from rpu_backend.api import Pi05Policy

    for name in ("pi05-libero-finetuned", "pi05-base"):
        p = model_path(name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "model.safetensors")):
            print(f"  loading {name} from {p} ...", flush=True)
            t0 = time.time()
            policy = Pi05Policy.from_pretrained(str(p), dtype=torch.float16)
            print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
            print("  .to('rpu') ...", flush=True)
            t1 = time.time()
            policy.to("rpu")
            print(f"  to('rpu') in {time.time()-t1:.1f}s", flush=True)
            return policy
    raise FileNotFoundError(
        "Neither pi05-libero-finetuned nor pi05-base has model.safetensors.")


def _make_sig(num_steps: int) -> "rpu_backend.graph.GraphSignature":
    'Same sig key across Run 2 and Run 3 so the 2nd capture hits REPLAYING.'
    return rpu_backend.graph.GraphSignature(
        op_id="pi05_predict_action_chunk",
        shapes=[1, 32],           # tokens shape (deterministic across calls)
        dyn_dims=[num_steps],     # disambiguates if user picks different num_steps
        dtypes=[torch.float16],
    )


def _forward_chunk(policy, batch, num_steps):
    'Pi05Policy._forward_rpu(..., return_chunk=True) → device chunk tensor.'
    with torch.no_grad():
        return policy._forward_rpu(batch, return_chunk=True, num_steps=num_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-steps", type=int, default=2,
                    help="denoise steps (low-power default: 2; full default: 10)")
    args = ap.parse_args()

    print("=" * 60, flush=True)
    print("Pi0.5 PASSTHROUGH vs GRAPH (byte-exact, low-power)", flush=True)
    print(f"  num_steps={args.num_steps}", flush=True)
    print("=" * 60, flush=True)

    setup_env()
    policy = load_policy()
    batch = _build_dummy_batch()
    sig = _make_sig(args.num_steps)

    # Run 1: PASSTHROUGH
    print("\n--- Run 1: PASSTHROUGH (no graph) ---", flush=True)
    t0 = time.time()
    out_pt_dev = _forward_chunk(policy, batch, args.num_steps)
    print(f"  forward: {time.time()-t0:.2f}s", flush=True)
    out_pt = out_pt_dev.clone().cpu()
    print(f"  shape={list(out_pt.shape)} dtype={out_pt.dtype} "
          f"sample[0,0,:4]={out_pt[0,0,:4].float().tolist()}", flush=True)

    # Run 2: GRAPH RECORDING
    print("\n--- Run 2: GRAPH RECORDING ---", flush=True)
    cache = rpu_backend.graph.GraphCache()
    t0 = time.time()
    with cache.capture(sig) as g:
        in2 = g.state()
        out_g1_dev = _forward_chunk(policy, batch, args.num_steps)
    end2 = g.state()
    rep2 = g.replayable()
    size2 = g.graph_size()
    out_g1 = out_g1_dev.clone().cpu()
    print(f"  forward: {time.time()-t0:.2f}s", flush=True)
    print(f"  in_state={in2} after_end={end2} graph_size={size2} "
          f"replayable={rep2}", flush=True)
    print(f"  sample[0,0,:4]={out_g1[0,0,:4].float().tolist()}", flush=True)

    # Run 3: GRAPH REPLAYING
    print("\n--- Run 3: GRAPH REPLAYING (same sig) ---", flush=True)
    t0 = time.time()
    with cache.capture(sig) as g:
        in3 = g.state()
        out_g2_dev = _forward_chunk(policy, batch, args.num_steps)
    end3 = g.state()
    rep3 = g.replayable()
    size3 = g.graph_size()
    out_g2 = out_g2_dev.clone().cpu()
    print(f"  forward: {time.time()-t0:.2f}s", flush=True)
    print(f"  in_state={in3} after_end={end3} graph_size={size3} "
          f"replayable={rep3}", flush=True)
    print(f"  sample[0,0,:4]={out_g2[0,0,:4].float().tolist()}", flush=True)

    # GraphCache snapshot
    snap = cache.snapshot()
    print("\n--- cache snapshot ---", flush=True)
    for i, e in enumerate(snap):
        print(f"  entry #{i}: kernel={e.kernel_count} seg={e.segment_count} "
              f"replay={e.replay_count} recapture={e.recapture_count}", flush=True)
        if e.non_replayable_reason:
            print(f"    !!! non_replayable: {e.non_replayable_reason!r}",
                  flush=True)

    # 验收
    STATE = {0: "PASSTHROUGH", 1: "RECORDING", 2: "BUILT", 3: "REPLAYING"}
    chk = lambda b, label: print(f"  [{'PASS' if b else 'FAIL'}] {label}",
                                  flush=True)

    print("\n" + "=" * 60, flush=True)
    chk(in2 == 1, f"Run 2 in_scope=RECORDING (got {STATE.get(in2)})")
    chk(end2 == 2, f"Run 2 after_end=BUILT (got {STATE.get(end2)})")
    chk(rep2, f"Run 2 replayable=True (got {rep2})")
    chk(size2 > 0, f"Run 2 graph_size > 0 (got {size2})")
    chk(in3 == 3, f"Run 3 in_scope=REPLAYING (got {STATE.get(in3)})")
    chk(end3 == 2, f"Run 3 after_end=BUILT (got {STATE.get(end3)})")
    chk(rep3, f"Run 3 replayable=True (got {rep3})")
    chk(size3 == size2, f"Run 3 graph_size {size3} == Run 2 {size2}")
    chk(torch.equal(out_pt, out_g1),
        "Run 1 == Run 2 (PASSTHROUGH == RECORDING) bit-exact")
    chk(torch.equal(out_g1, out_g2),
        "Run 2 == Run 3 (RECORDING == REPLAYING) bit-exact")
    chk(torch.equal(out_pt, out_g2),
        "Run 1 == Run 3 (PASSTHROUGH == REPLAYING) bit-exact")

    all_ok = (
        in2 == 1 and end2 == 2 and rep2 and size2 > 0 and
        in3 == 3 and end3 == 2 and rep3 and size3 == size2 and
        torch.equal(out_pt, out_g1) and
        torch.equal(out_g1, out_g2) and
        torch.equal(out_pt, out_g2)
    )
    print("=" * 60, flush=True)
    print("PASS" if all_ok else "FAIL", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
