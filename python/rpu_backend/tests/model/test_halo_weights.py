from __future__ import annotations
import dataclasses
import os
from pathlib import Path
import pytest
import torch

def _inv_freq(cfg) -> torch.Tensor:
    return 1.0 / cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float64) / cfg.head_dim)

def test_build_rope_tables_constant_position(monkeypatch) -> None:
    from rpu_backend.adapters.halo import HaloConfig
    from rpu_backend.adapters.halo import weights as w
    monkeypatch.setattr(w, '_to_rpu_half', lambda t: t.contiguous())
    cfg = HaloConfig()
    half = cfg.head_dim // 2
    pos = 11
    (cos, sin) = w.build_rope_tables(cfg, pos)
    assert tuple(cos.shape) == (cfg.total_len, half)
    assert cos.dtype == torch.float16
    assert torch.equal(cos[0], cos[100])
    ang = pos * _inv_freq(cfg)
    assert torch.allclose(cos[0].double(), ang.cos().to(torch.float16).double(), atol=0.002)
    assert torch.allclose(sin[0].double(), ang.sin().to(torch.float16).double(), atol=0.002)
    (cos2, _) = w.build_rope_tables(cfg, pos, rows=100)
    assert tuple(cos2.shape) == (100, half)
    with pytest.raises(ValueError):
        w.build_rope_tables(cfg, -1)

def test_timestep_embedding_shape_and_formula() -> None:
    from rpu_backend.adapters.halo import HaloConfig
    from rpu_backend.adapters.halo import weights as w
    cfg = HaloConfig()
    emb = w.timestep_embedding(torch.tensor([0.0, 1.0, 0.5]), cfg)
    assert tuple(emb.shape) == (3, cfg.time_freq_dim)
    half = cfg.time_freq_dim // 2
    assert torch.allclose(emb[0, :half], torch.ones(half))
    assert torch.allclose(emb[0, half:], torch.zeros(half))

def test_sinusoidal_pos_embedding_shape_and_formula() -> None:
    from rpu_backend.adapters.halo import HaloConfig
    from rpu_backend.adapters.halo import weights as w
    cfg = HaloConfig()
    emb = w.sinusoidal_pos_embedding(torch.tensor([0, 1, 5]), cfg)
    assert tuple(emb.shape) == (3, cfg.hidden_size)
    half = cfg.hidden_size // 2
    assert torch.allclose(emb[0, :half], torch.zeros(half))
    assert torch.allclose(emb[0, half:], torch.ones(half))

def test_build_text_row_mask(monkeypatch) -> None:
    from rpu_backend.adapters.halo import HaloConfig
    from rpu_backend.adapters.halo import weights as w
    monkeypatch.setattr(w, '_to_rpu_half', lambda t: t.contiguous())
    cfg = HaloConfig()
    m = w.build_text_row_mask(cfg)
    assert tuple(m.shape) == (cfg.chunk_size,)
    for row in cfg.text_rows:
        assert m[row].item() == 1.0
    assert m.sum().item() == len(cfg.text_rows)
    with pytest.raises(ValueError):
        w.build_text_row_mask(dataclasses.replace(cfg, text_rows=(0, 999)))

def test_build_image_text_row_mask(monkeypatch) -> None:
    from rpu_backend.adapters.halo import HaloConfig
    from rpu_backend.adapters.halo import weights as w
    monkeypatch.setattr(w, '_to_rpu_half', lambda t: t.contiguous())
    cfg = HaloConfig()
    m = w.build_image_text_row_mask(cfg)
    assert tuple(m.shape) == (cfg.num_image_query_tokens,)
    for row in cfg.image_text_rows:
        assert m[row].item() == 1.0
    assert m.sum().item() == len(cfg.image_text_rows)
    with pytest.raises(ValueError):
        w.build_image_text_row_mask(dataclasses.replace(cfg, image_text_rows=(0, 99999)))

def _patch_fake_store(monkeypatch, missing=()):
    from rpu_backend.adapters.halo import weights as w
    requested = []
    missing_set = set(missing)

    class _FakeStore:

        def __init__(self, ckpt_path):
            pass

        def get(self, name):
            requested.append(name)
            if name in missing_set:
                raise KeyError(f'missing tensor {name}')
            return torch.zeros(2, 2, dtype=torch.float32)
    monkeypatch.setattr(w, '_HaloTensorStore', _FakeStore)
    monkeypatch.setattr(w, '_load_branch', lambda store, cfg, suffix, key: object())
    return requested

def test_load_halo_branches_default_skips_lm_head(monkeypatch) -> None:
    from rpu_backend.adapters.halo import HaloConfig
    from rpu_backend.adapters.halo import weights as w
    requested = _patch_fake_store(monkeypatch)
    (_text, _act, host) = w.load_halo_branches(Path('/unused/ckpt'), HaloConfig())
    assert host.lm_head_w is None
    assert 'language_model.lm_head.weight' not in requested

def test_load_halo_branches_loads_lm_head_when_requested(monkeypatch) -> None:
    from rpu_backend.adapters.halo import HaloConfig
    from rpu_backend.adapters.halo import weights as w
    requested = _patch_fake_store(monkeypatch)
    (_text, _act, host) = w.load_halo_branches(Path('/unused/ckpt'), HaloConfig(), load_lm_head=True)
    assert host.lm_head_w is not None
    assert 'language_model.lm_head.weight' in requested

def test_load_halo_branches_tied_embedding_ckpt_loads_by_default(monkeypatch) -> None:
    from rpu_backend.adapters.halo import HaloConfig
    from rpu_backend.adapters.halo import weights as w
    _patch_fake_store(monkeypatch, missing=['language_model.lm_head.weight'])
    (_text, _act, host) = w.load_halo_branches(Path('/unused/ckpt'), HaloConfig())
    assert host.lm_head_w is None
    _patch_fake_store(monkeypatch, missing=['language_model.lm_head.weight'])
    with pytest.raises(KeyError):
        w.load_halo_branches(Path('/unused/ckpt'), HaloConfig(), load_lm_head=True)

def test_build_native_prefix_banks_frames_fail_fast() -> None:
    from rpu_backend.adapters.halo import build_native_prefix_banks
    with pytest.raises(NotImplementedError, match='分阶段'):
        build_native_prefix_banks('/nonexistent/ckpt.safetensors', img_uncond_text_pack={'packed_text_ids': None, 'packed_text_position_ids': None}, frames=[object()])
