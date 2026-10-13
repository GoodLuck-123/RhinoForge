from __future__ import annotations
from rpu_backend.adapters.wall_oss.vision import _get_window_index, _block_diag_mask, _compute_vision_position_idx_cpu, _pack_multi_pos, _pack_multi_image
import os
import sys
import importlib.util
import torch

def test_multi_image_window_layout():
    grid_i = torch.tensor([[1, 16, 16]])
    grid_group = grid_i.repeat(2, 1)
    pos = _pack_multi_pos(grid_group, sms=2)
    assert pos.shape == (2 * 256, 2)
    assert torch.equal(pos[:256], pos[256:512])
    (win_multi, reverses, mask, cu) = _pack_multi_image(grid_group, window_size=112, sms=2, patch_size=14)
    assert mask.shape == (256, 256)
    assert win_multi.numel() == 2 * (256 // 4)
    assert len(reverses) == 2
    assert cu[0].item() == 0 and cu[-1].item() == 256
    (win1, _, _) = _get_window_index(grid_i, 112, 2, 14)
    assert torch.equal(win_multi[:64], win1)
    assert torch.equal(win_multi[64:], win1 + 64)
