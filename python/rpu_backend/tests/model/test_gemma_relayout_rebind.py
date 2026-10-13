'test gemma relayout rebind regression.'
import sys
import traceback
import os

import torch
import rpu_backend
from rpu_backend import RPUCache
from rpu_backend.runtime.weights import swizzle_model_inplace

from rpu_backend.adapters.pi05.gemma import patch_gemma_model_for_rpu_all_layers_once


if hasattr(rpu_backend, "set_profile"):
    rpu_backend.set_profile(False)
if hasattr(rpu_backend, "set_debug"):
    rpu_backend.set_debug(False)
torch.rpu.set_caching_allocator(True)
torch.rpu.set_ddr_flush(True)






from rpu_backend.model_registry import model_path
MODEL_PATH = os.environ.get(
    "PI05_MODEL_PATH",
    str(model_path("pi05-libero-finetuned-w8a16-vlm-expert")),
)


def _load_vlm():
    from rpu_backend.tests.fixtures.pi05_loader import load_pi05_model
    policy, _ = load_pi05_model(MODEL_PATH, dtype="fp16", fused=True)
    return policy.model.paligemma_with_expert.paligemma.model.language_model


def _prepare_rpu_instance():
    vlm = _load_vlm()
    cfg = vlm.config
    attn_tp = min(8, cfg.num_key_value_heads)
    swizzle_model_inplace(vlm, attn_num_cores=attn_tp)
    vlm._rpu_attn_tp = attn_tp
    vlm._rpu_max_seq_len = 2048
    vlm.to("rpu")
    vlm.embed_tokens = vlm.embed_tokens.to("cpu")
    patch_gemma_model_for_rpu_all_layers_once(vlm)
    return vlm


def _make_input(vlm, seq_len=128):
    'Choose seq_len long enough to actually subdivide into multiple chunks'
    vocab = vlm.config.vocab_size
    ids = torch.randint(0, vocab, (1, seq_len))
    emb = vlm.embed_tokens(ids)
    return emb.to(dtype=torch.float16, device="rpu")


def _make_cache(vlm):
    cfg = vlm.config
    return RPUCache(
        num_layers=cfg.num_hidden_layers, batch_size=1, max_seq_len=2048,
        num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
        attn_tp=min(8, cfg.num_key_value_heads))


def _forward(vlm, inp):
    cache = _make_cache(vlm)
    with torch.no_grad():
        out = vlm(inputs_embeds=inp.clone(), past_key_values=cache,
                  use_cache=True).last_hidden_state
    return out.detach().float().cpu()


def main():
    print("[test_gemma_relayout_rebind] preparing instance")
    vlm = _prepare_rpu_instance()


    warmup = torch.rpu.get_rpu_warmup()
    inp = _make_input(vlm, seq_len=128)





    vlm._rpu_chunk_size = 0
    for _ in range(max(warmup, 0)):
        rpu_backend.reset_graph_cache()
        _ = _forward(vlm, inp)


    vlm._rpu_chunk_size = 0
    rpu_backend.reset_graph_cache()
    out_A = _forward(vlm, inp)








    vlm._rpu_chunk_size = 64
    rpu_backend.reset_graph_cache()
    out_B = _forward(vlm, inp)


    vlm._rpu_chunk_size = 0
    rpu_backend.reset_graph_cache()
    out_C = _forward(vlm, inp)




    print("[test_gemma_relayout_rebind] preparing reference instance")
    vlm_ref = _prepare_rpu_instance()
    vlm_ref._rpu_chunk_size = 64
    rpu_backend.reset_graph_cache()
    out_ref = _forward(vlm_ref, inp)

    vlm._rpu_chunk_size = 0
    vlm_ref._rpu_chunk_size = 0
    rpu_backend.reset_graph_cache()


    mse_B = ((out_B - out_ref) ** 2).mean().item()
    mse_C_vs_A = ((out_C - out_A) ** 2).mean().item()




    diff_AB = ((out_A - out_B) ** 2).mean().item()

    print(f"  MSE(out_B vs ref@64) = {mse_B:.6e}")
    print(f"  MSE(out_C vs out_A)  = {mse_C_vs_A:.6e} (reversibility)")
    print(f"  MSE(out_A vs out_B)  = {diff_AB:.6e} (layout transition visible)")

    if mse_B >= 1e-3:
        print(f"[FAIL] relayout rebind: MSE(out_B vs ref@64) {mse_B:.6e} >= 1e-3 — "
              f"lambda re-binding failed OR declare_buffers not re-invoked on "
              f"chunk_size change")
        sys.exit(1)
    if mse_C_vs_A >= 1e-3:
        print(f"[FAIL] relayout rebind reversibility: MSE(out_C vs out_A) "
              f"{mse_C_vs_A:.6e} >= 1e-3 — post-relayout state is corrupting "
              f"subsequent forwards after chunk_size returns to auto")
        sys.exit(1)

    print(f"[PASS] Gemma relayout lambda rebind: "
          f"MSE_B={mse_B:.4e}, MSE_C_vs_A={mse_C_vs_A:.4e}")
    print(f"RESULT: {{\"mse_B\": {mse_B:.6e}, "
          f"\"mse_C_vs_A\": {mse_C_vs_A:.6e}, "
          f"\"layout_delta\": {diff_AB:.6e}, \"pass\": true}}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(2)
