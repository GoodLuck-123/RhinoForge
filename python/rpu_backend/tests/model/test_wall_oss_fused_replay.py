'Wall oss fused replay device regression.'
import torch

import rpu_backend  # noqa: F401
from rpu_backend.adapters.wall_oss.action import build_wall_oss_action
from rpu_backend.adapters.wall_oss.llm import build_wall_oss_llm
from rpu_backend.model_registry import model_path


def test_fused_replay_bit_stable_across_predicts():
    torch.manual_seed(0)
    ckpt = str(model_path("wall-oss-0.5"))
    llm = build_wall_oss_llm(ckpt, expert=0, max_seq_len=512, w8a16=False)
    act = build_wall_oss_action(ckpt, llm=llm, max_seq_len=512, w8a16=False, fused=True)

    P, H = 16, 8
    input_ids = torch.randint(0, 1000, (1, P), dtype=torch.long)
    noise = torch.randn(1, H, act.action_dim)
    dof = torch.ones(1, H, act.action_dim)
    pos = torch.arange(P, P + H, dtype=torch.int32)[:, None].expand(H, 3).contiguous()

    llm.forward(input_ids)               # prefill prefix K/V once
    P = llm.cache.position

    x1 = act.denoise_with_mask(P, noise, dof, pos, num_steps=10)["x_norm"]   # BUILD + 9 REPLAY
    llm.cache.reset_to_position(P)        # restore prefix for the second run
    x2 = act.denoise_with_mask(P, noise, dof, pos, num_steps=10)["x_norm"]   # 10 REPLAY (cached)

    max_abs = (x1 - x2).abs().max().item()
    print(f"replay robustness: max|x1-x2|={max_abs:.3e}")
    assert torch.equal(x1, x2), f"fused denoise not bit-stable across predicts (max|d|={max_abs})"
