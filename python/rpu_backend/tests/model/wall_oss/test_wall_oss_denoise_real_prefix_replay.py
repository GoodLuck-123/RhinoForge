'Wall oss denoise real prefix replay device regression.'
import os

import pytest
import torch

from rpu_backend.model_registry import model_path

CK = str(model_path("wall-oss-0.5"))

P_EXEC = 192      # execution prefix (what the signature keys on)
REAL_A = 180      # two REAL prefixes inside the same 16-band -> same P, same key
REAL_B = 172
H = 32
STEPS = 2         # the mask is re-uploaded per step; 2 is enough and keeps this cheap


@pytest.fixture(scope="module")
def denoise():
    'Expert-1 action denoiser with the shared prefix K/V filled at [0, P_EXEC).'
    if not os.path.isdir(CK):
        pytest.skip(f"wall-oss-0.5 checkpoint not found at {CK}")
    from rpu_backend.adapters.wall_oss import build_wall_oss_action

    torch.manual_seed(0)
    model = build_wall_oss_action(CK, max_seq_len=512, fused=False)
    model.llm.forward(torch.randint(0, 151000, (1, P_EXEC)))
    assert model.llm.cache.position == P_EXEC
    noise = torch.randn(1, H, model.action_dim)
    dof = torch.ones(1, H, model.action_dim)
    pos = (torch.arange(P_EXEC, P_EXEC + H, dtype=torch.int32)[:, None]
           .expand(H, 3).contiguous())
    cache, gc = model.llm.cache, model._graph_cache

    @torch.inference_mode()
    def run(real_p):
        cache.reset_to_position(P_EXEC)
        return model.denoise_with_mask(P_EXEC, noise, dof, pos, num_steps=STEPS,
                                       real_prefix_len=real_p)["x_norm"].clone()

    def fresh(real_p):
        gc.clear()
        return run(real_p)

    ns = type("D", (), {})()
    ns.run, ns.fresh, ns.gc = run, fresh, gc
    ns.ref_a, ns.ref_b = fresh(REAL_A), fresh(REAL_B)
    yield ns
    model.close()


def test_real_prefix_actually_changes_the_answer(denoise):
    'Positive control. Without this the three tests below are vacuously green.'
    assert not torch.equal(denoise.ref_a, denoise.ref_b), (
        f"real prefix {REAL_A} and {REAL_B} produced the SAME action at P={P_EXEC}; "
        "the probe is blind, not the adapter correct")


def test_replay_at_the_same_real_prefix_is_byte_identical(denoise):
    'Matched-shape control: a REPLAY that changes nothing must change nothing.'
    denoise.gc.clear()
    build = denoise.run(REAL_A)
    replay = denoise.run(REAL_A)
    assert torch.equal(build, denoise.ref_a)
    assert torch.equal(replay, denoise.ref_a)


def test_replay_under_a_shared_key_sees_the_other_real_prefix(denoise):
    'BUILD at real prefix A, REPLAY at B: must equal a fresh BUILD at B, byte for byte.'
    denoise.gc.clear()
    assert torch.equal(denoise.run(REAL_A), denoise.ref_a)
    stale = denoise.run(REAL_B)
    assert torch.equal(stale, denoise.ref_b), (
        "the graph BUILT at real_prefix_len=%d served real_prefix_len=%d with a stale "
        "mask: max|delta|=%g" % (REAL_A, REAL_B,
                                 (stale - denoise.ref_b).abs().max().item()))


def test_alternating_real_prefixes_share_exactly_one_graph_key(denoise):
    'A->B->A->B (the pattern action.py:382 names) and the cache-growth half.'
    denoise.gc.clear()
    legs = [denoise.run(r) for r in (REAL_A, REAL_B, REAL_A, REAL_B)]
    expect = [denoise.ref_a, denoise.ref_b, denoise.ref_a, denoise.ref_b]
    got = [torch.equal(g, e) for g, e in zip(legs, expect)]
    assert got == [True] * 4, (
        "A->B->A->B legs matching their own fresh BUILD: %s (real prefixes %s); "
        "max|delta| per leg %s" % (
            got, [REAL_A, REAL_B, REAL_A, REAL_B],
            [f"{(g - e).abs().max().item():.3g}" for g, e in zip(legs, expect)]))

    snap = denoise.gc._impl.snapshot()
    assert len(snap) == 1, f"expected ONE denoise graph key, got {len(snap)}"
    assert sum(e.replay_count for e in snap) > 0, "no REPLAY happened; nothing was gated"
    assert denoise.gc._impl.cache_invariant_ok()
