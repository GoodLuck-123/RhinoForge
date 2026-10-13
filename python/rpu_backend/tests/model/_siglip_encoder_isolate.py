
' siglip encoder isolate regression.'
import os, sys
from rpu_backend.model_registry import model_path, set_hf_home_env
set_hf_home_env()
import torch
import rpu_backend
from rpu_backend.adapters.siglip import (
    _SIGLIP_BRANCH_PRE_EMBEDDED,
    patch_siglip_model_for_rpu_all_layers_once,
)

L = int(os.environ.get("ENC_SEQ", "512"))
CACHE_IMAGE_CAPACITY = max(1, (L + 255) // 256)

def _load():
    from rpu_backend.tests.fixtures.pi05_loader import load_pi05_model
    policy, _ = load_pi05_model(str(model_path("pi05-libero-finetuned")), dtype="fp16", fused=True)
    vt = policy.model.paligemma_with_expert.paligemma.model.vision_tower
    vit = vt.vision_model if hasattr(vt, 'vision_model') else vt
    proj = policy.model.paligemma_with_expert.paligemma.model.multi_modal_projector
    return vit, proj

def main():
    vm, proj = _load()


    patch_siglip_model_for_rpu_all_layers_once(
        vm, proj, num_images=CACHE_IMAGE_CAPACITY)
    h = vm._rpu_vision_handle
    cache = vm._rpu_cache
    nl = cache.num_layers

    torch.manual_seed(4242)
    hidden = (torch.randn(1, L, 1152, dtype=torch.float16) * 0.1).to('rpu').contiguous()

    sig = rpu_backend.graph.GraphSignature(
        op_id="pi05_siglip_compute",
        shapes=[L, nl, cache.head_dim],
        dyn_dims=[cache.num_kv_heads, min(8, cache.num_kv_heads), 1,
                  int(getattr(vm, "_siglip_w8a16", 0))],
        dtypes=[torch.float16],
        branch_key=_SIGLIP_BRANCH_PRE_EMBEDDED,
    )

    outs = []
    for it in range(3):
        cache.reset_to_position(0)
        kc = [cache.k_caches[i] for i in range(nl)]
        vc = [cache.v_caches[i] for i in range(nl)]
        with vm._rpu_siglip_graph_cache.capture(sig):
            o = torch.ops.rpu.siglip_forward(h, hidden, kc, vc)
        torch.ops.rpu.spm_alloc_reset_temporary()
        outs.append(o.detach().cpu().float().clone())
        print(f"  iter {it}: mean={outs[-1].mean().item():.6f} "
              f"nan={bool(torch.isnan(outs[-1]).any())}")

    d01 = (outs[0] - outs[1]).abs().max().item()
    d12 = (outs[1] - outs[2]).abs().max().item()
    graph_snapshot = list(vm._rpu_siglip_graph_cache.snapshot())
    graph_ok = (
        vm._rpu_siglip_graph_cache.size() == 1
        and len(graph_snapshot) == 1
        and int(graph_snapshot[0].replay_count) >= 2
        and int(graph_snapshot[0].recapture_count) == 0
        and vm._rpu_siglip_graph_cache.cache_invariant_ok()
    )
    ok = (d01 == 0.0 and d12 == 0.0 and graph_ok)
    print(f"\nENCODER seq={L} (image_batch_count=1, v2 SDPA): "
          f"replay max_abs_diff (0v1)={d01:.3e} (1v2)={d12:.3e} "
          f"[{'DETERMINISTIC' if ok else 'NON-DETERMINISTIC'}]")
    print(f"  graph size={vm._rpu_siglip_graph_cache.size()} "
          f"replay={graph_snapshot[0].replay_count if graph_snapshot else 0} "
          f"recapture={graph_snapshot[0].recapture_count if graph_snapshot else 0} "
          f"[{'PASS' if graph_ok else 'FAIL'}]")
    print(f"RESULT: {{\"seq\": {L}, \"deterministic\": "
          f"{str(ok).lower()}, \"graph_lifecycle_ok\": "
          f"{str(graph_ok).lower()}}}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
