'SigLIP handle, weight invalidation and capture branch contracts.'
import os
import pytest
import torch
import rpu_backend
from rpu_backend.model_registry import model_path
from rpu_backend.adapters.siglip import patch_siglip_model_for_rpu_all_layers_once
pytestmark = pytest.mark.skipif(os.environ.get("RPU_HW_AVAILABLE") != "1", reason="requires device access")

@pytest.fixture
def vision_pair():
    return load_fresh_vision_model(str(model_path("pi05-libero-finetuned")))

@pytest.fixture
def projector(vision_pair):
    return vision_pair[1]

@pytest.fixture
def vision_model(vision_pair):
    vision, projection = vision_pair
    patch_siglip_model_for_rpu_all_layers_once(vision, projection)
    return vision


def load_fresh_vision_model(model_path):
    'Load a fresh (unpatched, UNCONVERTED) SigLIP vision model + projector.'
    from rpu_backend.tests.fixtures.pi05_loader import load_pi05_model
    policy, _ = load_pi05_model(model_path, dtype="fp16", fused=True)
    vt = policy.model.paligemma_with_expert.paligemma.model.vision_tower
    vit = vt.vision_model if hasattr(vt, 'vision_model') else vt
    proj = policy.model.paligemma_with_expert.paligemma.model.multi_modal_projector

    return vit, proj

def test_section1_handle_lifecycle():
    print("\n--- Section 1: Handle lifecycle ---")


    h = torch.ops.rpu.siglip_create()
    assert h >= 0, f"siglip_create returned negative handle: {h}"
    torch.ops.rpu.siglip_destroy(h)
    print("  1.1 create/destroy: PASS")


    h1 = torch.ops.rpu.siglip_create()
    h2 = torch.ops.rpu.siglip_create()
    assert h1 != h2, f"handles not unique: h1={h1}, h2={h2}"
    torch.ops.rpu.siglip_destroy(h1)
    torch.ops.rpu.siglip_destroy(h2)
    print("  1.2 unique handles: PASS")







    h3 = torch.ops.rpu.siglip_create()
    torch.ops.rpu.siglip_destroy(h3)
    h4 = torch.ops.rpu.siglip_create()
    assert h4 != h3, f"ABA reuse detected: h3={h3}, h4={h4}"
    torch.ops.rpu.siglip_destroy(h4)
    print("  1.3 ABA safety: PASS")

def test_section2_first_inference(vision_model, projector):
    print("\n--- Section 2: First inference ---")

    handle = patch_siglip_model_for_rpu_all_layers_once(vision_model, projector)


    hidden_states = torch.randn(1, 256, 1152, dtype=torch.float16, device="rpu")

    output = vision_model(pixel_values=hidden_states)
    last_hidden = output.last_hidden_state


    assert last_hidden.shape == (1, 256, 2048), \
        f"Expected [1,256,2048], got {last_hidden.shape}"
    assert torch.isfinite(last_hidden).all(), "Output contains NaN/Inf"





    print(f"  2.1 first inference: PASS (shape={last_hidden.shape}, "
          f"finite=True -> BUILD confirmed)")

    return handle

def test_section3_replay(vision_model):
    print("\n--- Section 3: Replay inference ---")

    hidden_states = torch.randn(1, 256, 1152, dtype=torch.float16, device="rpu")

    output1 = vision_model(pixel_values=hidden_states)



    out1_snapshot = output1.last_hidden_state.clone()

    output2 = vision_model(pixel_values=hidden_states)
    out2_snapshot = output2.last_hidden_state.clone()


    rel_diff = (out1_snapshot.float() - out2_snapshot.float()).abs().max().item()
    max_val = out1_snapshot.float().abs().max().item()
    rel = rel_diff / max_val if max_val > 0 else 0.0
    assert rel < 0.001, f"REPLAY outputs differ: rel_diff={rel:.6f}"
    print(f"  3.1 replay deterministic: PASS (rel_diff={rel:.6f})")

def test_section4_weights_invalidation(vision_model):
    'Test that calling siglip_set_weights on the SAME handle triggers'
    print("\n--- Section 4: set_weights invalidation ---")

    handle = vision_model._rpu_vision_handle


    weight_args_tuple = vision_model._test_siglip_weight_args_tuple

    hidden_states = torch.randn(1, 256, 1152, dtype=torch.float16, device="rpu")


    out_original = vision_model(pixel_values=hidden_states).last_hidden_state.clone()








    PROJ_B_IDX = 19
    modified_args = list(weight_args_tuple)
    modified_args[PROJ_B_IDX] = weight_args_tuple[PROJ_B_IDX] + 1.0

    torch.ops.rpu.siglip_set_weights(handle, *modified_args)


    out_modified = vision_model(pixel_values=hidden_states).last_hidden_state.clone()

    assert torch.isfinite(out_modified).all(), "Post-modification output has NaN/Inf"
    diff_mod = (out_original.float() - out_modified.float()).abs().max().item()
    assert diff_mod > 0.1, \
        f"Output unchanged after weight modification (diff={diff_mod:.6f}) -- " \
        f"weights_dirty_ likely not triggering rebuild"
    print(f"  4.1 modified weights change output: PASS (diff={diff_mod:.4f})")


    torch.ops.rpu.siglip_set_weights(handle, *weight_args_tuple)
    out_restored = vision_model(pixel_values=hidden_states).last_hidden_state.clone()

    rel_diff = (out_original.float() - out_restored.float()).abs().max().item()
    max_val = out_original.float().abs().max().item()
    rel = rel_diff / max_val if max_val > 0 else 0.0
    assert rel < 0.001, f"Output didn't revert after restoring weights: rel={rel:.6f}"
    print(f"  4.2 restored weights revert output: PASS (rel_diff={rel:.6f})")

def test_section6_patch_emb_autodetect(vision_model, projector):
    'Test that siglip_forward auto-detects 4D [1,C,H,W] input, runs'
    print("\n--- Section 6: 4D patch-emb auto-detection ---")


    if not hasattr(vision_model, '_rpu_vision_handle'):
        patch_siglip_model_for_rpu_all_layers_once(vision_model, projector)


    assert hasattr(vision_model, '_rpu_vision_handle'), "Handle not set"
    assert hasattr(vision_model.embeddings, '_rpu_gemm_weight'), \
        "Patch embedding _rpu_gemm_weight not set -- converter failed to set up patch_emb"
    assert hasattr(vision_model.embeddings, '_rpu_pos_emb_fused'), \
        "Patch embedding _rpu_pos_emb_fused not set -- converter failed to set up patch_emb"


    pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.float16, device="rpu")

    output = vision_model(pixel_values=pixel_values)
    last_hidden = output.last_hidden_state


    assert last_hidden.shape == (1, 256, 2048), \
        f"4D path output shape mismatch: expected [1,256,2048], got {last_hidden.shape}"
    assert torch.isfinite(last_hidden).all(), "4D path output contains NaN/Inf"
    print(f"  6.1 4D patch-emb auto-detect: PASS, shape={last_hidden.shape}")


    hidden_3d = torch.randn(1, 256, 1152, dtype=torch.float16, device="rpu")
    output_3d = vision_model(pixel_values=hidden_3d)
    assert output_3d.last_hidden_state.shape == (1, 256, 2048), \
        "3D path broken after 4D"
    assert torch.isfinite(output_3d.last_hidden_state).all(), "3D path NaN after 4D"
    print(f"  6.2 3D still works after 4D: PASS")














    graph_cache = vision_model._rpu_siglip_graph_cache
    assert graph_cache.size() == 2, (
        f"expected 2 cached SigLIP compute graphs (4D patch-embed + 3D "
        f"pre-embedded), got {graph_cache.size()} -- the two branches share a "
        f"GraphSignature again")
    assert graph_cache.cache_invariant_ok(), "SigLIP GraphCache invariant violated"



    before = {s.signature.branch_key: s.replay_count
              for s in graph_cache.snapshot()}
    assert len(before) == 2, \
        f"the 2 entries do not carry distinct branch_keys: {sorted(before)}"
    vision_model(pixel_values=pixel_values)
    vision_model(pixel_values=hidden_3d)
    after = {s.signature.branch_key: s for s in graph_cache.snapshot()}
    assert graph_cache.size() == 2, \
        f"cache grew to {graph_cache.size()} on re-run (recapture?)"
    assert set(after) == set(before), \
        f"branch_key set changed: {sorted(before)} -> {sorted(after)}"
    for bk, snap in after.items():
        assert snap.replay_count > before[bk], (
            f"branch_key={bk} did not REPLAY on re-run "
            f"(replay_count {before[bk]} -> {snap.replay_count})")
        assert snap.recapture_count == 0, \
            f"branch_key={bk} recaptured {snap.recapture_count}x"
    print(f"  6.3 4D/3D graphs are distinct cache entries, both REPLAY: PASS "
          f"(branch_keys={sorted(after)}, "
          f"replays={[after[b].replay_count for b in sorted(after)]})")
