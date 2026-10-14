'Hy model geometry and native admission contracts.'
from pathlib import Path
import pytest
import torch
_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_cpp_uses_per_handle_chunk_and_stable_mask_state() -> None:
    fused = _REPO_ROOT / "src/fused"
    expert = (fused / "rpu_hyvla_expert_model.cpp").read_text(
        encoding="utf-8")
    vlm = (fused / "rpu_hyvla_vlm_model.cpp").read_text(encoding="utf-8")
    vit = (fused / "rpu_hyvit2_vision_model.cpp").read_text(encoding="utf-8")

    assert "g_chunk_size_override" not in expert + vlm
    for text, registry, component in (
        (expert, "HyVlaExpertRegistry", "hyvla_expert"),
        (vlm, "HyVlaVlmRegistry", "hyvla_vlm"),
    ):
        setter = text.split(f"void rpu_{component}_set_chunk_size_override(", 1)[1].split("\n}", 1)[0]
        assert "int64_t handle, int64_t chunk_size" in setter
        assert f"{registry}::get(" in setter
        assert "handle," in setter
        assert "->set_control_chunk_size_override(" in setter
        assert "CausalDecoderModel::forward(" in text
        assert "std::optional<at::Tensor>(attn_mask_4d)" in text
    decoder = (fused / "rpu_qwen3_model.h").read_text(encoding="utf-8")
    assert "sdpa_stable_mask_cache()" in decoder
    assert "sdpa_stable_mask_cache(),\n                                   n_img" in vit

def test_config_geometry_matches_checkpoint_facts() -> None:
    '`HyVlaConfig` 的派生量必须与 ckpt/config 实测一致。'
    from rpu_backend.adapters.hy_vla import HyVlaConfig
    c = HyVlaConfig()
    assert c.vit_head_dim_padded == 80, "72 → 16 对齐"
    assert c.vit_inter_padded == 4352, "4304 → 8 核 ×128 对齐"




    assert c.kv_heads_rpu in (c.num_kv_heads, 2 * c.num_kv_heads), \
        "kv_heads_rpu 只能是真实值或它的 2 倍（G 的槽位复制）"
    assert c.attn_tp == min(8, c.kv_heads_rpu), "attn_tp = min(8, kv_heads_rpu)"
    assert c.attn_tp in (4, 8) and c.num_q_heads % c.attn_tp == 0


    assert abs(c.rope_base - 1.115884e7) / 1.115884e7 < 1e-5

    assert 2 + 3 * (1 + c.grid * (c.grid + 1) + 1) + 64 == 240

def test_build_hy_vla_rejects_out_of_envelope_prefix() -> None:
    '`prefix_len` 的包络在入口即校验 —— 240 是需求上限也是能力上限。'
    from rpu_backend.adapters.hy_vla import build_hy_vla
    for bad in (0, 15, 100, 256, 272):
        with pytest.raises(ValueError, match="prefix_len"):
            build_hy_vla("/nonexistent.safetensors", torch.zeros(1, 196, 1152),
                         prefix_len=bad)
