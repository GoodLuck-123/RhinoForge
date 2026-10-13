from __future__ import annotations
from typing import Tuple
import pytest
import torch

def test_euler_action_schedule() -> None:
    from rpu_backend.adapters.halo.runtime import euler_action_schedule
    (timesteps, dts) = euler_action_schedule(10)
    assert tuple(timesteps.shape) == tuple(dts.shape) == (9,)
    assert abs(float(timesteps[0]) - 1.0) < 1e-06
    assert float(timesteps[-1]) > 0.0
    assert torch.allclose(dts, torch.full((9,), 1.0 / 9.0), atol=1e-06)
    ref = torch.linspace(1, 0, 10)
    assert torch.equal(timesteps, ref[:-1])
    assert torch.equal(dts, ref[:-1] - ref[1:])
    with pytest.raises(ValueError):
        euler_action_schedule(1)

def test_halo_cfg_branch_signature_no_stale_replay() -> None:
    import types
    from rpu_backend.graph._runtime import _graph_fnv1a_64
    from rpu_backend.adapters.halo import HaloConfig
    from rpu_backend.adapters.halo.runtime import _cfg_branch_dyn_dims
    cfg = HaloConfig()

    def _branch(prefix_len: int, rope: int, branch_id: int, epoch: int):
        return types.SimpleNamespace(prefix_len=prefix_len, rope_position=rope, branch_id=branch_id, epoch=epoch)

    def _key(branch, *, op_id: str='halo_action_expert_step_cfg', shapes=(cfg.chunk_size, cfg.hidden_size), dtype_codes: Tuple[int, ...]=(5,)) -> tuple:
        return (_graph_fnv1a_64(op_id), tuple(shapes), tuple(_cfg_branch_dyn_dims(cfg, branch)), tuple(dtype_codes))
    base = _key(_branch(4706, 11, 0, 1))
    assert _key(_branch(4706, 11, 0, 2)) != base, 'epoch(分配身份)未进 key → 同一 runner 跨 episode 复用会陈旧 replay 旧 DMA 指针(静默污染)'
    matrix = {'branch_id (cond 0→text 1)': _branch(4706, 11, 1, 1), 'rope_position (11→3)': _branch(4706, 3, 0, 1), 'prefix_len/kv_len (4706→4698)': _branch(4698, 11, 0, 1), 'epoch (1→2,新分配)': _branch(4706, 11, 0, 2)}
    for (name, b) in matrix.items():
        assert _key(b) != base, f'sig 未随 {name} 变化 → 会 stale replay(跨分支/跨步/跨 episode 假阳)'
    assert _key(_branch(4706, 11, 0, 1), op_id='halo_action_expert_step') != base, 'op_id 须进 key'
    assert _key(_branch(4706, 11, 0, 1), shapes=(16, cfg.hidden_size)) != base, 'shapes 须进 key'
    assert _key(_branch(4706, 11, 0, 1)) == base, '同 sig 应相等(REPLAY 命中,否则每步重 BUILD)'
    cond_k = _key(_branch(4706, 11, 0, 1))
    text_k = _key(_branch(4698, 3, 1, 2))
    img_k = _key(_branch(8, 8, 2, 3))
    assert len({cond_k, text_k, img_k}) == 3, '三分支 sig 必须两两相异(各自独立 BUILD)'

def test_make_cfg_branch_bumps_allocation_epoch(monkeypatch) -> None:
    import types
    from rpu_backend.adapters.halo import HaloConfig
    from rpu_backend.adapters.halo import runtime as rt
    seen = []

    def _fake_build(name, bid, kl, vl, *, rope_position, prefix_len, epoch, cfg=None, max_seq_len=4736):
        seen.append(epoch)
        return types.SimpleNamespace(name=name, branch_id=bid, epoch=epoch, prefix_len=prefix_len, rope_position=rope_position)
    monkeypatch.setattr(rt, 'build_cfg_branch', _fake_build)
    runner = types.SimpleNamespace(_cfg_branch_epoch=0, cfg=HaloConfig())
    b1 = rt.HaloStepRunner.make_cfg_branch(runner, 'cond', 0, [], [], rope_position=11, prefix_len=4706)
    b2 = rt.HaloStepRunner.make_cfg_branch(runner, 'cond', 0, [], [], rope_position=11, prefix_len=4706)
    assert (b1.epoch, b2.epoch) == (1, 2), 'make_cfg_branch 须单调递增 epoch(新分配新身份)'
    assert runner._cfg_branch_epoch == 2 and seen == [1, 2]

def test_build_cfg_branch_requires_epoch() -> None:
    from rpu_backend.adapters.halo.runtime import build_cfg_branch
    with pytest.raises(TypeError):
        build_cfg_branch('cond', 0, [], [], rope_position=11, prefix_len=4706)
