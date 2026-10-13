#!/usr/bin/env python
'Siglip batch 3image device regression.'

import json
import os
import sys

from rpu_backend.model_registry import model_path, set_hf_home_env
set_hf_home_env()

import torch  # noqa: E402
import rpu_backend  # noqa: F401, E402
from rpu_backend.adapters.siglip import (  # noqa: E402
    _siglip_forward_images,
    patch_siglip_model_for_rpu_all_layers_once,
)


DEFAULT_MODEL_PATH = str(model_path("pi05-libero-finetuned"))


def _load_fresh_vision_model(model_path):
    from rpu_backend.tests.fixtures.pi05_loader import load_pi05_model
    policy, _ = load_pi05_model(model_path, dtype="fp16", fused=True)
    vt = policy.model.paligemma_with_expert.paligemma.model.vision_tower
    vit = vt.vision_model if hasattr(vt, 'vision_model') else vt
    proj = policy.model.paligemma_with_expert.paligemma.model.multi_modal_projector
    return vit, proj


def main():
    model_path = os.environ.get("SIGLIP_MODEL_PATH", DEFAULT_MODEL_PATH)
    num_images = int(os.environ.get("SIGLIP_BATCH_N", "3"))

    vision_model, projector = _load_fresh_vision_model(model_path)
    # Patch for batched N=3. The N<=num_images robustness lets a single-image
    # forward (N=1) also run on this model (builds a separate N=1 graph entry).
    patch_siglip_model_for_rpu_all_layers_once(
        vision_model, projector, num_images=num_images)

    # Plan 01-01 canonical warmup harness (consume via the binding, not os.environ).
    # Warm up BOTH shapes the test measures — the single-image [1,...] reference
    # AND the batched [N,...] path — so neither measured pass is a cold BUILD
    # (a cold first forward intermittently reads uninitialized SPM → NaN). Scaled
    # to [-1,1] to avoid fp16 overflow in the warmup itself.
    warmup_count = torch.rpu.get_rpu_warmup()
    if warmup_count > 0:
        print(f"Warmup: {warmup_count}× on single [1,3,224,224] + batched "
              f"[{num_images},3,224,224]...")
        with torch.no_grad():
            for i in range(warmup_count):
                for _nb in sorted({1, num_images}):
                    _pv = ((torch.randn(_nb, 3, 224, 224, dtype=torch.float32) * 0.4)
                           .clamp(-1.0, 1.0).to(dtype=torch.float16, device="rpu"))
                    if _nb == 1:
                        _ = vision_model(pixel_values=_pv)
                    else:
                        _ = _siglip_forward_images(
                            vision_model,
                            [_pv[index:index + 1] for index in range(_nb)],
                        )
                    del _, _pv
                print(f"  warmup {i+1}")

    # 3 distinct deterministic images, in the realistic SigLIP input range
    # [-1,1] (pi05 normalizes pixels to [-1,1] before the vision tower). Raw
    # N(0,1) randn has ±4 outliers that overflow fp16 inside SigLIP (known ViT
    # fp16 overflow — LN var / SDPA q·kᵀ) and yield NaN regardless of batching.
    images = []
    for i in range(num_images):
        torch.manual_seed(2000 + i)
        img = (torch.randn(1, 3, 224, 224, dtype=torch.float32) * 0.4).clamp(-1.0, 1.0)
        images.append(img.to(dtype=torch.float16, device="rpu").contiguous())

    # --- GROUND TRUTH: single-image path (N=1, legacy unified SDPA) ---
    print("\nReference pass (single-image, N=1, unified SDPA):")
    ref_outs = []
    with torch.no_grad():
        for i, img in enumerate(images):
            out = vision_model(pixel_values=img).last_hidden_state  # [1,256,2048]
            ref = out.detach().clone()
            ref_outs.append(ref)
            print(f"  img {i}: shape={tuple(ref.shape)} "
                  f"mean={ref.float().mean().item():.6f}")
    seq_per = ref_outs[0].shape[1]  # 256

    # --- BATCHED: production list entry → one forward_multi (N>1) ---
    print(f"\nBatched pass ({num_images} camera images → forward_multi):")
    with torch.no_grad():
        packed1 = _siglip_forward_images(
            vision_model, images).detach().clone()
        graph_cache = vision_model._rpu_siglip_graph_cache
        snapshot1 = list(graph_cache.snapshot())
        cache_size1 = graph_cache.size()
        replay1 = sum(int(entry.replay_count) for entry in snapshot1)
        packed2 = _siglip_forward_images(
            vision_model, images).detach().clone()
        snapshot2 = list(graph_cache.snapshot())
    replay2 = sum(int(entry.replay_count) for entry in snapshot2)
    graph_lifecycle_ok = (
        cache_size1 > 0
        and graph_cache.size() == cache_size1
        and len(snapshot2) == len(snapshot1)
        and replay2 > replay1
        and all(int(entry.recapture_count) == 0 for entry in snapshot2)
        and graph_cache.cache_invariant_ok()
    )
    print(f"  packed shape={tuple(packed1.shape)}")
    print(f"  graph cache: size={cache_size1}->{graph_cache.size()} "
          f"replay={replay1}->{replay2} "
          f"[{'PASS' if graph_lifecycle_ok else 'FAIL'}]")

    # --- B) REPLAY determinism: forward #1 == forward #2 bit-identical ---
    replay_diff = (packed1.cpu().float() - packed2.cpu().float()).abs().max().item()
    replay_ok = (replay_diff == 0.0)
    print(f"\nREPLAY determinism: max_abs_diff(fwd1, fwd2)={replay_diff:.3e} "
          f"[{'PASS' if replay_ok else 'FAIL'}]")

    # --- A) Correctness + isolation: packed slot i vs single-image ref i ---
    packed_cpu = packed1.cpu().float()
    print("\nPer-image comparison (packed slot i vs single-image ref i):")
    all_ok = True
    max_rel = 0.0
    min_cos = 1.0
    for i in range(num_images):
        p = packed_cpu[:, i * seq_per:(i + 1) * seq_per, :]
        r = ref_outs[i].detach().cpu().float()
        diff_t = (p - r).abs()
        diff = diff_t.max().item()
        maxv = r.abs().max().item()
        rel = diff / maxv if maxv > 0 else 0.0
        cos = torch.nn.functional.cosine_similarity(
            p.reshape(-1), r.reshape(-1), dim=0).item()
        ok = (rel < 1e-2 and cos > 0.9999)
        print(f"  img {i}: max_abs_diff={diff:.4e} rel_diff={rel:.4e} "
              f"cos_sim={cos:.8f} [{'PASS' if ok else 'FAIL'}]")
        all_ok = all_ok and ok
        max_rel = max(max_rel, rel)
        min_cos = min(min_cos, cos)

    passed = all_ok and replay_ok and graph_lifecycle_ok
    if passed:
        print(f"\n[PASS] SigLIP {num_images}-image batch: PASS "
              f"(max rel_diff={max_rel:.2e}, min cos_sim={min_cos:.8f})")
    else:
        print(f"\n[FAIL] SigLIP {num_images}-image batch: FAIL "
              f"(max rel_diff={max_rel:.2e}, min cos_sim={min_cos:.8f}, "
              f"replay_ok={replay_ok}, graph_lifecycle_ok={graph_lifecycle_ok})")

    result = {
        "correctness_ok": bool(all_ok),
        "replay_deterministic": bool(replay_ok),
        "graph_lifecycle_ok": bool(graph_lifecycle_ok),
        "num_images": int(num_images),
        "seq_per_image": int(seq_per),
        "max_rel_diff_lt_1e_2": bool(max_rel < 1e-2),
        "min_cos_sim_gt_0_9999": bool(min_cos > 0.9999),
    }
    print(f"RESULT: {json.dumps(result, sort_keys=True)}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
