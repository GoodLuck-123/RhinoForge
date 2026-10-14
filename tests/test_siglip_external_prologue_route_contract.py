"""Board-free contracts for SigLIP patch routes and receipt transfer."""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_siglip_patch_launchers_consume_canonical_routes():
    source = _source("src/fused/rpu_siglip_model.cpp")
    sites = {
        "SIGLIP_PATCH_INPUT_DMA_SITE": "1143336462482724752LL",
        "SIGLIP_PATCH_POSITION_DMA_SITE": "1099468962711310446LL",
        "SIGLIP_PATCH_LINEAR_SITE": "6500914803307480014LL",
        "SIGLIP_PATCH_OUTPUT_DMA_SITE": "7261580027961404620LL",
    }
    for name, site_id in sites.items():
        assert f"{name} = {site_id}" in source
        assert source.count(name) >= 3

    helper = source[
        source.index("static void fused_patch_embedding(") :
        source.index("static void check_siglip_w8a16_scale_lists(")
    ]
    launches = tuple(
        re.finditer(r"\brpu_launch_ddr_broadcast_spm_dma_mutable\s*\(", helper)
    )
    assert len(launches) == 2
    assert "SIGLIP_PATCH_INPUT_DMA_SITE" in helper[
        max(0, launches[0].start() - 1000) : launches[0].start()
    ]
    assert "SIGLIP_PATCH_POSITION_DMA_SITE" in helper[
        max(0, launches[1].start() - 1000) : launches[1].start()
    ]
    for launcher, site in (
        ("rpu_launch_linear_spm_to_spm_acc16_kernel", "SIGLIP_PATCH_LINEAR_SITE"),
        ("rpu_launch_spm_copy_ddr_dma_mutable", "SIGLIP_PATCH_OUTPUT_DMA_SITE"),
    ):
        call = re.search(rf"\b{launcher}\s*\(", helper)
        assert call is not None
        assert site in helper[max(0, call.start() - 1000) : call.start()]

    assert "FmbLinearRouteSelector::AUTO_TILE" in helper
    assert helper.count("if (physical_ctx != nullptr)") == 4
    assert "{num_patches, cin_orig, cin_padded, kh, kw," in helper
    assert "{num_patches, cout, K, 1, 1, 0}" in helper
    assert "{num_patches, cout, 1}" in helper
    assert "{seq_off * cout * DWIDTH, num_patches * cout}" in helper


def test_siglip_patch_manifest_is_mode_explicit_and_standalone_safe():
    source = _source("src/fused/rpu_siglip_model.cpp")
    registration = _source("src/core/rpu_dispatch_registrations.inc")
    adapter = _source("python/rpu_backend/adapters/siglip.py")

    assert "bool external_patch_prologue" in source
    assert "if (planning_external_patch_prologue_)" in source
    assert "static_cast<int64_t>(plan.spans.size()) ==" in source
    assert "consume_external_prologue ? &ctx() : nullptr" in source
    assert source.count("/*consume_external_prologue=*/true") == 2
    assert source.count("/*consume_external_prologue=*/false") == 2
    assert "bool external_patch_prologue=False" in registration
    assert "external_patch_prologue=input_tensor.dim() == 4" in adapter
    assert "external_patch_prologue=True" in adapter


def test_siglip_attention_receipt_digests_image_batch_geometry():
    source = _source("src/fused/rpu_siglip_model.cpp")
    assert source.count(
        "{image_batch_count_, per_image_ctx, seq_len, seq_len_,"
    ) == 3
    assert "{image_batch_count_, per_image_ctx, chunk.len, physical_len," in source
    assert "SIGLIP_RAW_SPM_ATTN_SITE" in source
    for site in ("SIGLIP_MINIBATCH_ATTN_SITE", "SIGLIP_UNIFIED_ATTN_SITE"):
        launcher = (
            "rpu_launch_sdpa_spm_minibatch_kernel"
            if "MINIBATCH" in site
            else "rpu_launch_sdpa_spm_unified_kernel_v2"
        )
        call = re.search(rf"\b{launcher}\s*\(", source)
        assert call is not None
        prefix = source[max(0, call.start() - 900) : call.start()]
        assert "consume_physical_route(" in prefix
        assert site in prefix
        assert "image_batch_count_" in prefix


def test_external_prologue_receipts_transfer_only_same_fingerprint():
    header = _source("src/core/fused_model_base.h")
    implementation = _source("src/core/fused_model_base.cpp")
    plan = _source("src/core/fmb_three_stage_chunk_plan.cpp")
    host_test = _source("tests/test_fmb_three_stage_chunk_plan.cpp")

    assert "begin_external_physical_manifest_prologue(" in header
    assert "retain_consumed_if_matching(manifest)" in header
    assert "external_physical_manifest_prologue_pending_" in header
    assert "transferred_external_physical_manifest" in implementation
    assert "external_physical_manifest_prologue_replay()" in implementation
    assert "incoming_fingerprint == fingerprint_" in plan
    assert "same-fingerprint transfer retains prologue receipts" in host_test
    assert "external prologue transfer rejects a different fingerprint" in host_test
    assert "same-fingerprint external REPLAY" in host_test
