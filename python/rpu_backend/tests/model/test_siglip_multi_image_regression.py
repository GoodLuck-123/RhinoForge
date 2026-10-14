#!/usr/bin/env python
'Siglip multi image regression device regression.'

import json
import os
import sys

from rpu_backend.model_registry import model_path, set_hf_home_env
set_hf_home_env()

import torch  # noqa: E402
import rpu_backend  # noqa: F401, E402

from rpu_backend.adapters.siglip import (  # noqa: E402
    patch_siglip_model_for_rpu_all_layers_once,
)


DEFAULT_MODEL_PATH = str(model_path("pi05-libero-finetuned"))


def _load_fresh_vision_model(model_path):
    'Identical loader to test_siglip_all_layers_once.py helper (duplicated'
    from rpu_backend.tests.fixtures.pi05_loader import load_pi05_model
    policy, _ = load_pi05_model(model_path, dtype="fp16", fused=True)
    vt = policy.model.paligemma_with_expert.paligemma.model.vision_tower
    vit = vt.vision_model if hasattr(vt, 'vision_model') else vt
    proj = policy.model.paligemma_with_expert.paligemma.model.multi_modal_projector
    return vit, proj


def main():
    model_path = os.environ.get("SIGLIP_MODEL_PATH", DEFAULT_MODEL_PATH)
    num_images = 3

    vision_model, projector = _load_fresh_vision_model(model_path)
    patch_siglip_model_for_rpu_all_layers_once(vision_model, projector)

    # Plan 01-01 canonical warmup harness — consume the env var via the binding,
    # never read os.environ["RPU_WARMUP"] directly.
    warmup_count = torch.rpu.get_rpu_warmup()
    if warmup_count > 0:
        print(f"Warmup: running {warmup_count} forward(s) on a dummy image...")
        with torch.no_grad():
            for i in range(warmup_count):
                _pv = torch.randn(1, 3, 224, 224,
                                  dtype=torch.float16, device="rpu")
                _ = vision_model(pixel_values=_pv)
                del _, _pv
                print(f"  warmup {i+1}")

    # Make 3 distinct deterministic pixel inputs. torch.manual_seed is global;
    # using fixed per-image seeds guarantees the images differ from one
    # another but each run is reproducible.
    images = []
    for i in range(num_images):
        torch.manual_seed(1000 + i)
        images.append(torch.randn(1, 3, 224, 224,
                                  dtype=torch.float16, device="rpu"))

    # Reference pass — run each image independently and CLONE the output
    # before the next forward. `.clone()` snapshots the DDR contents into a
    # fresh storage, so even a stable-DDR regression cannot corrupt these
    # references. This gives us the GROUND TRUTH against which the cat pattern
    # is compared.
    print("\nReference pass (clone per image — immune to Pitfall 4):")
    ref_outs = []
    with torch.no_grad():
        for i, img in enumerate(images):
            out = vision_model(pixel_values=img).last_hidden_state
            ref = out.detach().clone()
            ref_outs.append(ref)
            print(f"  img {i}: shape={tuple(ref.shape)} "
                  f"mean={ref.float().mean().item():.6f}")

    # Probe pass — Pi0.5-style accumulate-then-cat. Append the output
    # WITHOUT .clone() to exercise the exact pattern that triggered the
    # original Pitfall 4 bug. With Pitfall 4 in place (fresh at::empty per
    # forward via allocate_tracked_output), each ref is independent DDR and
    # the cat preserves per-image data. Without it, all three refs would
    # alias the same DDR and the cat would contain 3× the last image.
    print("\nProbe pass (no clone — exercises Pitfall 4 surface):")
    probe_refs = []
    with torch.no_grad():
        for i, img in enumerate(images):
            out = vision_model(pixel_values=img).last_hidden_state
            probe_refs.append(out)  # append WITHOUT clone (the P4 trap)
            print(f"  img {i}: id(storage)={out.data_ptr():#x} "
                  f"shape={tuple(out.shape)}")

    # torch.cat along the seq dimension. For a [1, 256, 2048] per-image
    # output, the cat is [1, 768, 2048] with slot i = probe_refs[i].
    cat = torch.cat(probe_refs, dim=1)
    assert cat.shape[1] == num_images * ref_outs[0].shape[1], \
        f"cat shape {tuple(cat.shape)} unexpected"

    seq_per = ref_outs[0].shape[1]

    # Per-slot comparison against the reference. Move tensors to CPU + fp32
    # first — some reductions (torch.nn.functional.mse_loss) don't fully
    # reduce to scalar on the device PrivateUse1 device path.
    max_rel_diff_overall = 0.0
    all_slots_ok = True
    print("\nPer-slot comparison (cat slot i vs ref image i):")
    for i in range(num_images):
        slot = cat[:, i * seq_per:(i + 1) * seq_per, :].detach().cpu().float()
        ref = ref_outs[i].detach().cpu().float()
        diff_t = (slot - ref).abs()
        diff = diff_t.max().item()
        max_val = ref.abs().max().item()
        rel = diff / max_val if max_val > 0 else 0.0
        mse = (diff_t * diff_t).mean().item()
        tag = "PASS" if rel < 1e-4 else "FAIL"
        print(f"  slot {i}: max_abs_diff={diff:.6e} "
              f"rel_diff={rel:.6e} mse={mse:.6e} [{tag}]")
        if rel >= 1e-4:
            all_slots_ok = False
        max_rel_diff_overall = max(max_rel_diff_overall, rel)

    # Aggregate MSE across the whole cat, for the RESULT payload.
    cat_cpu = cat.detach().cpu().float()
    cat_ref_cpu = torch.cat(
        [r.detach().cpu().float() for r in ref_outs], dim=1)
    _cat_diff = (cat_cpu - cat_ref_cpu).abs()
    cat_mse = (_cat_diff * _cat_diff).mean().item()
    cat_max_abs_diff = _cat_diff.max().item()

    # Final PASS/FAIL line.
    if all_slots_ok:
        print(f"\n[PASS] multi-image accumulation: PASS "
              f"(max rel_diff={max_rel_diff_overall:.2e}, "
              f"cat_mse={cat_mse:.2e})")
    else:
        print(f"\n[FAIL] multi-image accumulation: FAIL "
              f"(max rel_diff={max_rel_diff_overall:.2e} — Pitfall 4 regression)")

    # Canonical RESULT: line — deterministic payload only (no timestamps,
    # no memory addresses, no elapsed-time). Fixed-precision floats are
    # byte-deterministic because device forwards are deterministic REPLAYs of
    # the same graph. Wrap floats through float() to avoid half-precision
    # serialization noise.
    result = {
        "all_slots_ok": bool(all_slots_ok),
        "cat_mse_lt_1e_4": bool(cat_mse < 1e-4),
        "num_images": int(num_images),
        "seq_per_image": int(seq_per),
        "cat_mse_under_1e_1": bool(cat_mse < 1e-1),  # very loose sanity
    }
    print(f"RESULT: {json.dumps(result, sort_keys=True)}")

    sys.exit(0 if all_slots_ok else 1)


if __name__ == "__main__":
    main()
