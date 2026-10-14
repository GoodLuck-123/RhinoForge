'Wall oss denoise loop device regression.'
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import rpu_backend  # noqa: F401
from rpu_backend.adapters.wall_oss.action import build_wall_oss_action
from rpu_backend.adapters.wall_oss.llm import build_wall_oss_llm
from rpu_backend.api import RPUCache
from rpu_backend.model_registry import model_path


def _cos(a, b):
    return F.cosine_similarity(a.flatten().float(), b.flatten().float(), 0).item()


def test_unroll_matches_proven_loop(monkeypatch):
    torch.rpu.set_caching_allocator(True)
    torch.manual_seed(0)
    # `_DEFAULT_CKPT` no longer exists — build_wall_oss_llm resolves its default
    # at call time now. Same source the sibling test uses
    # (test_wall_oss_fused_single_step.py), so both gates load one checkpoint.
    ckpt = str(model_path("wall-oss-0.5"))
    max_seq = 512
    llm = build_wall_oss_llm(ckpt, expert=0, max_seq_len=max_seq, w8a16=False)
    m = build_wall_oss_action(ckpt, llm=llm, max_seq_len=max_seq, w8a16=False, fused=True)

    P, H = 16, 8
    input_ids = torch.randint(0, 1000, (1, P), dtype=torch.long)
    noise = torch.randn(1, H, m.action_dim)
    dof_mask = torch.ones(1, H, m.action_dim)          # row-constant -> bias+te path
    pos = torch.arange(P, P + H, dtype=torch.int32)[:, None].expand(H, 3).contiguous()

    llm.forward(input_ids)                              # prefix K/V at [0, P)
    assert llm.cache.position == P

    # Proven per-step fused op: opt OUT of the (now default-on) unroll so this genuinely
    # exercises `wall_oss_action_step` (1 BUILD + 9 REPLAY), not the unroll comparing to itself.
    monkeypatch.setenv("RPU_WALL_OSS_DENOISE_UNROLL", "0")
    proven = m.denoise_with_mask(P, noise, dof_mask, pos, num_steps=10)
    monkeypatch.delenv("RPU_WALL_OSS_DENOISE_UNROLL", raising=False)

    # Unrolled single-graph loop (direct call; env-independent). Restore the prefix first
    # (the proven run overwrote [P,P+H)); also exercises the per-step -> unroll handle switch.
    llm.cache.reset_to_position(P)
    unroll = m._denoise_with_mask_loop(P, noise, dof_mask, pos, num_steps=10)

    cos = _cos(unroll["x_norm"], proven["x_norm"])
    print(f"unroll vs proven final action: cos={cos:.6f} "
          f"proven={tuple(proven['x_norm'].shape)} unroll={tuple(unroll['x_norm'].shape)}")
    assert cos >= 0.999, f"unroll final action cos {cos} < 0.999"

    # Replay bit-stability: a second unrolled predict must be bit-identical.
    llm.cache.reset_to_position(P)
    again = m._denoise_with_mask_loop(P, noise, dof_mask, pos, num_steps=10)
    maxd = (unroll["x_norm"] - again["x_norm"]).abs().max().item()
    print(f"unroll replay robustness: max|x1-x2|={maxd:.3e}")
    assert torch.equal(unroll["x_norm"], again["x_norm"]), f"unroll not bit-stable (max|d|={maxd})"


def test_rtc_hard_pin_mutable_prefix_matches_fresh_build():
    'Runtime contract.'
    torch.rpu.set_caching_allocator(True)
    torch.manual_seed(0)
    ckpt = str(model_path("wall-oss-0.5"))
    cache = RPUCache(
        num_layers=36, batch_size=1, max_seq_len=512,
        num_kv_heads=2, head_dim=128, attn_tp=2,
    )
    llm = SimpleNamespace(num_layers=36, cache=cache, close=lambda: None)
    m = build_wall_oss_action(
        ckpt, llm=llm, max_seq_len=512, w8a16=False, fused=True)
    try:
        # Stock weights are sufficient to gate the C++ RTC execution structure.

        m._causal_action_attention_mask = False
        m._flow_time_scale = 0.999
        m._rtc_max_delay = 10
        m._rtc_rope_gap = 10
        m._masked_dof_velocity = True

        P, H, D, steps = 16, 20, 3, 2
        # This gate covers only RTC action-graph mutable-prefix/Graph mechanics.
        # The cache has the production geometry and position, but its zero prefix

        cache.reset_to_position(P)
        noise = torch.randn(1, H, m.action_dim)
        dof = torch.ones(1, H, m.action_dim)
        pos = (torch.arange(P, P + H, dtype=torch.int32)[:, None]
               .expand(H, 3).contiguous())
        prefix_a = torch.zeros(1, H, m.action_dim)
        prefix_b = torch.full_like(prefix_a, 0.25)

        def run(prefix, delay=D):
            llm.cache.reset_to_position(P)
            return m._denoise_with_mask_loop(
                P, noise, dof, pos, num_steps=steps, debug=True,
                rtc_prefix_actions=prefix, rtc_delay=delay)

        m._graph_cache.clear()
        out_a = run(prefix_a)
        out_b_replay = run(prefix_b)
        out_a_replay = run(prefix_a)
        assert torch.equal(out_a["x_norm"], out_a_replay["x_norm"])
        assert not torch.equal(out_a["x_norm"], out_b_replay["x_norm"])
        assert torch.equal(
            out_b_replay["x_norm"][:, :D], prefix_b[:, :D].half().float())
        assert torch.count_nonzero(out_b_replay["vs"][0, :D]) > 0
        assert torch.count_nonzero(out_b_replay["vs"][1:, :D]) == 0
        assert torch.equal(
            out_b_replay["xs"][:, :D],
            prefix_b[:, :D].half().float().expand(steps + 1, D, m.action_dim))
        snap = m._graph_cache._impl.snapshot()
        assert len(snap) == 1
        assert sum(entry.replay_count for entry in snap) > 0
        assert m._graph_cache._impl.cache_invariant_ok()

        m._graph_cache.clear()
        out_b_fresh = run(prefix_b)
        assert torch.equal(out_b_replay["x_norm"], out_b_fresh["x_norm"])

        out_d1 = run(prefix_b, 1)
        out_d10 = run(prefix_b, 10)
        out_d3_again = run(prefix_b, D)
        assert torch.equal(out_d1["x_norm"][:, :1], prefix_b[:, :1].half().float())
        assert torch.equal(
            out_d10["x_norm"][:, :10], prefix_b[:, :10].half().float())
        assert torch.equal(out_b_fresh["x_norm"], out_d3_again["x_norm"])
        snap = m._graph_cache._impl.snapshot()
        assert len(snap) == 3
        assert m._graph_cache._impl.cache_invariant_ok()
    finally:
        m.close()
        llm.close()
