from __future__ import annotations

import torch

from rpu_backend.adapters.lingbot_vla_v2.prefix import (
    AlignConfig,
    apply_suffix_prefix_blocking_,
)


def test_lingbot_alignment_blocks_future_depth_unless_disabled() -> None:
    assert AlignConfig().block_suffix_to_future_video is True
    visible = torch.ones(2, 32, dtype=torch.bool)
    apply_suffix_prefix_blocking_(visible, 32, AlignConfig(block_future_depth_to_action=False))
    assert visible.all()

    blocked = torch.ones_like(visible)
    apply_suffix_prefix_blocking_(
        blocked,
        32,
        AlignConfig(),
    )
    assert blocked[:, :-8].all()
    assert not blocked[:, -8:].any()
