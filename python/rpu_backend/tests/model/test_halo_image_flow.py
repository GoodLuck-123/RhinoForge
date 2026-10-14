from __future__ import annotations
from typing import Tuple
import pytest
import torch
TIMESTEP_SHIFT = 3.0
NUM_TIMESTEPS = 50

def test_euler_image_schedule() -> None:
    (nt, shift) = (NUM_TIMESTEPS, TIMESTEP_SHIFT)
    ts = torch.linspace(1, 0, nt)
    ts = shift * ts / (1 + (shift - 1) * ts)
    dts = ts[:-1] - ts[1:]
    ts_run = ts[:-1]
    assert tuple(ts_run.shape) == tuple(dts.shape) == (nt - 1,)
    assert ts_run[0].item() == pytest.approx(1.0, abs=1e-06)
    assert ts_run[-1].item() > 0.0
    assert torch.all(dts > 0), 'dt 必须全正（t 单调降）'

def test_halo_image_flow_cfg_branch_signature_no_stale_replay() -> None:
    from rpu_backend.graph._runtime import _graph_fnv1a_64
    from rpu_backend.adapters.halo.weights import HaloConfig
    cfg = HaloConfig()

    def _key(prefix_len: int, epoch: int, *, op_id: str='halo_image_flow_step', num_layers: int=cfg.num_layers, n_query: int=cfg.num_image_query_tokens, hidden: int=cfg.hidden_size, dtype_codes: tuple=(5,)) -> tuple:
        return (_graph_fnv1a_64(op_id), (n_query, hidden), (num_layers, prefix_len, n_query, epoch), tuple(dtype_codes))
    cond = _key(9586, 1)
    text = _key(9579, 2)
    img = _key(7, 3)
    cond_replay = _key(9586, 1)
    assert len({cond, text, img}) == 3, '3 CFG 分支 sig 必须两两相异（否则跨分支 stale replay）'
    assert cond == cond_replay, '同 regime 同 epoch 应可复用（REPLAY 命中，否则每步重 BUILD）'
    base = _key(9586, 1)
    assert _key(9579, 1) != base, 'prefix_len 变须改 key'
    assert _key(9586, 2) != base, 'epoch 变须改 key'
    assert _key(9586, 1, num_layers=27) != base, 'num_layers 变须改 key'
    assert _key(9586, 1, op_id='halo_image_flow_step_other') != base, 'op_id 变须改 key'
