"""Board-free source contract for the widened SDPA admission rule.

The native extension is not available on ordinary host runners, so these checks
lock the two layers of the contract without pretending to provide hardware
evidence: the shared generator predicate admits only the structural
multiple-of-eight boundary, while a model capability may retain a narrower
numeric ceiling for a measured-bad geometry.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPERS = (ROOT / "src/core/rpu_helpers.h").read_text(encoding="utf-8")
QWEN3 = (ROOT / "src/fused/rpu_qwen3_model.h").read_text(encoding="utf-8")
QWEN35 = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text(encoding="utf-8")


def _section(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    return source[begin : source.index(end, begin)]


def test_shared_predicate_and_profile_cap_are_distinct():
    """Structural admission is widened; an optional cap remains explicit."""
    assert "int64_t max_sync_product = 0;" in HELPERS
    assert "if (cfg.max_sync_product < 0) return false;" in HELPERS
    assert "if (call_cfg.max_sync_product > 0 && product > call_cfg.max_sync_product)" in HELPERS
    assert "if (product > 8 && product % 8 != 0) return false;" in HELPERS
    assert "product <= 8 || product % 8 == 0 || sync_force_unsafe" in HELPERS


def test_qwen3_06b_numeric_cap_is_geometry_scoped():
    """The known Qwen3-0.6B product-16 failure is not a global rollback."""
    make_cfg = _section(
        QWEN3,
        "SdpaConfig make_sdpa_config(int mask = 1) const",
        "// ── Model state",
    )
    for token in (
        "hidden_size() == 1024 && intermediate_size() == 3072",
        "num_q_heads() == 16 && num_kv_heads() == 8",
        "head_dim() == 128",
        "has_qk_norm_ && !has_qkv_bias_",
        "qwen3_06b_shape ? 8 : 0",
    ):
        assert token in make_cfg
    # Other profiles keep the zero/default capability and therefore use the
    # shared structural rule (the expression has exactly one narrowed branch).
    assert make_cfg.count("qwen3_06b_shape ? 8 : 0") == 1


def test_qwen3_and_qwen35_planner_routes_call_the_shared_validity_helper():
    qwen3_valid = _section(
        QWEN3,
        "bool subclass_chunk_size_valid(int64_t cs, int64_t seq_len, int64_t position)",
        "int64_t subclass_chunk_size_cap",
    )
    qwen35_valid = _section(
        QWEN35,
        "bool Qwen3_5Model::subclass_chunk_size_valid(int64_t cs, int64_t seq_len,",
        "// MR-A: Qwen3.5 had no cap hook",
    )
    assert "sdpa_is_valid_chunk_size(make_sdpa_config" in qwen3_valid
    assert "sdpa_is_valid_chunk_size(make_sdpa_config" in qwen35_valid


def test_unsafe_launcher_bypass_is_diagnostic_only():
    marker = "RPU_CHUNK_FORCE_UNSAFE"
    assert marker in HELPERS
    unsafe = HELPERS[HELPERS.index(marker) : HELPERS.index("if (attn_mask_type == 1)", HELPERS.index(marker))]
    assert "Diagnostic only" in unsafe
    assert "profile-specific certificates" in unsafe
