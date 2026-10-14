from __future__ import annotations
import types
import pytest
import torch

def _patch_rpu_device_to_cpu(monkeypatch) -> None:
    real_to = torch.Tensor.to

    def patched(self, *args, **kwargs):
        device = kwargs.get('device')
        new_args = list(args)
        if device is None:
            for (i, a) in enumerate(new_args):
                if isinstance(a, str) and a.startswith('rpu'):
                    device = a
                    new_args.pop(i)
                    break
        if isinstance(device, str) and device.startswith('rpu'):
            kwargs = {k: v for (k, v) in kwargs.items() if k != 'device'}
            return real_to(self, *new_args, **kwargs)
        return real_to(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', patched)

def test_check_frame_position_negatives() -> None:
    from rpu_backend.adapters.halo.vit_cache import HaloVitCacheUpdate
    s = types.SimpleNamespace()
    HaloVitCacheUpdate._check_frame_position(s, torch.full((1566,), 2, dtype=torch.long), 2)
    bad = torch.full((1566,), 2, dtype=torch.long)
    bad[5] = 3
    with pytest.raises(ValueError):
        HaloVitCacheUpdate._check_frame_position(s, bad, 2)
    with pytest.raises(ValueError):
        HaloVitCacheUpdate._check_frame_position(s, torch.zeros(10, dtype=torch.long), 1)

def test_rope_position_constant_and_nonconstant() -> None:
    from rpu_backend.adapters.halo.vit_cache import HaloVitCacheUpdate
    s = types.SimpleNamespace()
    assert HaloVitCacheUpdate._rope_position(s, torch.zeros(1566, dtype=torch.long)) == 0
    assert HaloVitCacheUpdate._rope_position(s, torch.full((10,), 5, dtype=torch.long)) == 5
    bad = torch.zeros(10, dtype=torch.long)
    bad[3] = 1
    with pytest.raises(NotImplementedError):
        HaloVitCacheUpdate._rope_position(s, bad)

def test_build_frame_rope_tables_step_structure(monkeypatch) -> None:
    from rpu_backend.adapters.halo import HaloConfig
    from rpu_backend.adapters.halo import vit_cache as vc
    _patch_rpu_device_to_cpu(monkeypatch)
    cfg = HaloConfig()
    frame_len = 4
    (cos, sin) = vc.build_frame_rope_tables(cfg, frame_len, 64)
    half = cfg.head_dim // 2
    assert tuple(cos.shape) == (64, half) and cos.dtype == torch.float16
    inv = 1.0 / cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float64) / cfg.head_dim)
    for i in (0, frame_len - 1, frame_len, 2 * frame_len, 63):
        ang = i // frame_len * inv
        assert torch.allclose(cos[i].double(), ang.cos().to(torch.float16).double(), atol=0.002)
    assert torch.equal(cos[0], cos[frame_len - 1])
    assert not torch.equal(cos[0], cos[frame_len])
    assert torch.allclose(cos[0].double(), torch.ones(half, dtype=torch.float64), atol=0.002)
    with pytest.raises(ValueError):
        vc.build_frame_rope_tables(cfg, 0, 64)
    with pytest.raises(ValueError):
        vc.build_frame_rope_tables(cfg, 4, 60)
    with pytest.raises(ValueError, match='positive multiple'):
        vc.build_frame_rope_tables(cfg, 4, 0)
    with pytest.raises(ValueError, match='head_dim'):
        vc.build_frame_rope_tables(HaloConfig(head_dim=127), 4, 64)

def test_prefill_frames_validation() -> None:
    from rpu_backend.adapters.halo import VitCacheConfig
    from rpu_backend.adapters.halo.vit_cache import HaloVitCacheUpdate
    vcfg = VitCacheConfig()
    s = types.SimpleNamespace(vcfg=vcfg, lm_max_seq=4736, lm_cache=None)
    with pytest.raises(ValueError):
        HaloVitCacheUpdate.prefill_frames(s, [])
    too_many = [{'packed_seqlens': torch.tensor([vcfg.seq_total])} for _ in range(4)]
    with pytest.raises(ValueError):
        HaloVitCacheUpdate.prefill_frames(s, too_many)

    class _Cache:

        def reset(self) -> None:
            pass
    s2 = types.SimpleNamespace(vcfg=vcfg, lm_max_seq=4736, lm_cache=_Cache())
    bad = [{'packed_seqlens': torch.tensor([vcfg.seq_total - 1])}]
    with pytest.raises(ValueError):
        HaloVitCacheUpdate.prefill_frames(s2, bad)

class _PosCache:

    def __init__(self) -> None:
        self.position = 0
        self.reset_called = False
        self.updates = []

    def reset(self) -> None:
        self.position = 0
        self.reset_called = True

    def update_position(self, n) -> None:
        self.updates.append(int(n))
        self.position += int(n)

def test_forward_and_advance_updates_position_by_rows() -> None:
    from rpu_backend.adapters.halo.vit_cache import HaloVitCacheUpdate
    seq = 1566
    packed = torch.zeros(seq, 8, dtype=torch.float16)
    fwd = []
    s = types.SimpleNamespace(lm_cache=_PosCache(), _lm_forward=lambda ps, pos: (fwd.append((pos, int(ps.shape[0]))), torch.zeros(int(ps.shape[0]), 4))[1])
    HaloVitCacheUpdate._forward_and_advance(s, packed, 100)
    assert fwd == [(100, seq)]
    assert s.lm_cache.updates == [seq]
    assert s.lm_cache.position == seq

def test_run_lm_advances_position_single_frame() -> None:
    from rpu_backend.adapters.halo.vit_cache import HaloVitCacheUpdate
    seq = 1566
    packed = torch.zeros(seq, 8, dtype=torch.float16)
    fwd = []
    s = types.SimpleNamespace(lm_cache=_PosCache(), _rope_position=lambda pids: 0, _lm_forward=lambda ps, pos: (fwd.append((pos, int(ps.shape[0]))), torch.zeros(int(ps.shape[0]), 4))[1])
    s._forward_and_advance = lambda ps, pos: HaloVitCacheUpdate._forward_and_advance(s, ps, pos)
    HaloVitCacheUpdate._run_lm(s, packed, torch.zeros(seq, dtype=torch.long))
    assert s.lm_cache.reset_called
    assert fwd == [(0, seq)]
    assert s.lm_cache.position == seq
