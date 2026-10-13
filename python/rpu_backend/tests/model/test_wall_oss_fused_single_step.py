'Wall oss fused single step device regression.'
import torch
import torch.nn.functional as F

import rpu_backend  # noqa: F401
from rpu_backend.adapters.wall_oss.action import build_wall_oss_action
from rpu_backend.adapters.wall_oss.llm import build_wall_oss_llm
from rpu_backend.model_registry import model_path


def test_fused_single_step_matches_python_loop(monkeypatch):
    monkeypatch.setenv("RPU_WALL_OSS_DENOISE_UNROLL", "0")
    torch.manual_seed(0)
    ckpt = str(model_path("wall-oss-0.5"))
    max_seq = 512
    llm = build_wall_oss_llm(ckpt, expert=0, max_seq_len=max_seq, w8a16=False)
    ref = build_wall_oss_action(ckpt, llm=llm, max_seq_len=max_seq, w8a16=False, fused=False)
    fused = build_wall_oss_action(ckpt, llm=llm, max_seq_len=max_seq, w8a16=False, fused=True)

    P, H = 16, 8
    input_ids = torch.randint(0, 1000, (1, P), dtype=torch.long)
    noise = torch.randn(1, H, ref.action_dim)
    dof_mask = torch.ones(1, H, ref.action_dim)        # row-constant -> bias path
    pos = torch.arange(P, P + H, dtype=torch.int32)[:, None].expand(H, 3).contiguous()

    # Shared prefill (prefix K/V at [0, P)).
    llm.forward(input_ids)
    assert llm.cache.position == P

    out_ref = ref.denoise_with_mask(
        P, noise, dof_mask, pos, num_steps=1, debug=True)
    v_ref = out_ref["vs"][0]                            # [1,H,26] cpu fp32

    llm.cache.reset_to_position(P)                      # restore prefix for the fused run
    out_fused = fused.denoise_with_mask(
        P, noise, dof_mask, pos, num_steps=1, debug=True)
    v_fused = out_fused["vs"][0]

    cos = F.cosine_similarity(v_ref.flatten(), v_fused.flatten(), 0).item()
    mse = (v_ref - v_fused).pow(2).mean().item()
    print(f"single-step v: cos={cos:.6f} mse={mse:.3e} "
          f"ref={tuple(v_ref.shape)} fused={tuple(v_fused.shape)}")
    assert cos >= 0.999, f"fused single-step v cos {cos} < 0.999"
