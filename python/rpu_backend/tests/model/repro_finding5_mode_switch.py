#!/usr/bin/env python3
'Repro finding5 mode switch device regression.'
import os
import sys
import traceback

import torch

import rpu_backend  # noqa: F401  (registers torch.ops.rpu / device device)
from rpu_backend.adapters.halo import HaloConfig, build_halo_text_cache

SEQ = 128          # small prefill
PINNED_CS = 64     # set_chunk_size override → causal & bidirectional share chunk_size
COS_OK = 0.9999    # C (causal round-trip) vs A — replay-stable layout


def _ckpt_path() -> str:
    override = os.environ.get("RPU_HALO_CKPT")
    if override:
        return override
    from pathlib import Path
    from rpu_backend.model_registry import model_path
    # Match the committed HALO tests' robust pattern (Path / rel), not Path + str.
    return str(Path(model_path("halo_action_expert"))
               / "halo_robotwin/finetune_ckpt/cot_aug_try3/0100000/ema.safetensors")


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().cpu().view(-1)
    b = b.float().cpu().view(-1)
    return (a @ b / (a.norm() * b.norm())).item()


def _forward(handle, cache, hidden, is_causal: bool) -> torch.Tensor:
    cache.reset()
    raw = torch.ops.rpu.causal_decoder_forward(
        handle,
        hidden,
        cache.k_caches,
        cache.v_caches,
        None,                 # attention_mask=None → with is_causal flag picks LTM/NONE
        cache.position,       # 0 after reset
        is_causal,
        None, [], None, None,
    )
    return raw[:, -1, :].squeeze(0).float().cpu()


def main() -> int:
    print("=" * 64)
    print("Attention mode participates in the allocation hash")
    print(f"  seq={SEQ}  pinned chunk_size={PINNED_CS}  (causal↔bidirectional, one HALO LM handle)")
    print("=" * 64)

    cfg = HaloConfig()
    runner = build_halo_text_cache(_ckpt_path(), cfg=cfg)
    handle = runner._handle
    cache = runner.cache

    torch.manual_seed(0)
    hidden = torch.randn(1, SEQ, cfg.hidden_size, dtype=torch.float16).contiguous().to("rpu")

    # PIN chunk_size so causal & bidirectional share it → only is_causal differs.

    torch.ops.rpu.causal_decoder_set_chunk_size_override(handle, PINNED_CS)

    results = {}
    failures = []

    try:
        results["A"] = _forward(handle, cache, hidden, is_causal=True)
        print(f"[A] causal forward         OK  logit_min={results['A'].min():.4f}")
    except Exception as e:  # noqa: BLE001
        failures.append(f"A(causal) raised: {type(e).__name__}: {e}")
        traceback.print_exc()

    try:
        results["B"] = _forward(handle, cache, hidden, is_causal=False)
        finite = torch.isfinite(results["B"]).all().item()
        print(f"[B] bidirectional forward  OK  logit_min={results['B'].min():.4f} finite={finite}")
        if not finite:
            failures.append("B(bidirectional) produced non-finite output")
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        failures.append(f"B(bidirectional) raised: {msg}")
        print(f"[B] bidirectional forward  FAIL — {msg}")
        if "q_kv" in str(e) or "not found" in str(e):
            print("    ^ UNFIXED finding-5 signature (reused causal alloc has no q_kv slot)")
        traceback.print_exc()

    try:
        results["C"] = _forward(handle, cache, hidden, is_causal=True)
        print(f"[C] causal forward (again) OK  logit_min={results['C'].min():.4f}")
    except Exception as e:  # noqa: BLE001
        failures.append(f"C(causal-again) raised: {type(e).__name__}: {e}")
        traceback.print_exc()

    if "A" in results and "C" in results:
        cs = _cos(results["A"], results["C"])
        ok = cs >= COS_OK
        print(f"[C vs A] causal round-trip cos={cs:.7f} (gate ≥ {COS_OK}) -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(f"C vs A causal round-trip cos={cs:.7f} < {COS_OK}")

    print("-" * 64)
    if failures:
        print("FINDING5_REPRO_RESULT: FAIL")
        for f in failures:
            print("  -", f)
        return 1
    print("FINDING5_REPRO_RESULT: PASS (mode switch on one handle no longer "
          "corrupts the SPM allocation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
