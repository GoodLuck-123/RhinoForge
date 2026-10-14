'Wall oss p4a mask device regression.'
import os

import pytest
import torch

import rpu_backend

from rpu_backend.model_registry import model_path

CK = str(model_path("wall-oss-0.5"))

EQUIV_THRESH = 0.9999
P_ALIGNED, H_EQ = 16, 16      # LTM-legal: the only prefix where LTM and MASK_2D compare
P_REAL, P_EXEC, H_ARB = 100, 112, 32   # arbitrary real prefix, padded to the 16 grid
STEPS = 2                              # the mask is re-uploaded per Euler step; 2 is enough
VOCAB = 151000


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0).item()


def _action_mask(P, H, *, causal_off_by_one=False):
    '[H, P+H] fp16 additive mask: action row i attends the full prefix + cols [P, P+i].'
    m = torch.full((H, P + H), float("-inf"), dtype=torch.float16)
    m[:, :P] = 0.0
    for i in range(H):
        m[i, P: P + i + (0 if causal_off_by_one else 1)] = 0.0
    return m


def _prefill(model, ids):
    'Fill the shared prefix K/V with `ids` (WallOssLLM always prefills from 0).'
    model.llm.forward(ids)
    assert model.llm.cache.position == int(ids.shape[1])


def _action_forward(model, P, H, x, dof_mask, *, mask, op_id):
    'One expert-1 action forward over the shared prefix KV (already at [0, P)).'
    cache = model.llm.cache
    cache.reset_to_position(P)
    emb = model._action_embed(0.0, x, dof_mask)
    emb_rpu = emb.to(dtype=torch.float16, device="rpu").contiguous()
    pos = (torch.arange(P, P + H, dtype=torch.int32)[:, None]
           .expand(H, 3).contiguous().to("rpu"))
    sig = rpu_backend.graph.GraphSignature(
        op_id=op_id, shapes=[H, model.hidden_size],
        dyn_dims=[model.num_layers, P], dtypes=[torch.float16])
    model._graph_cache.clear()      # every leg is a fresh BUILD: replay is a
                                    # different file's subject, not a variable here
    with model._graph_cache.capture(sig):
        raw = torch.ops.rpu.causal_decoder_forward(
            model._handle, emb_rpu, cache.k_caches, cache.v_caches,
            mask, P, mask is None, pos, [])
    return raw.float().cpu().reshape(1, H, model.hidden_size)


@pytest.fixture(scope="module")
def action():
    'Expert-1 action decoder (non-fused → the `causal_decoder_forward` MASK_2D path).'
    if not os.path.isdir(CK):
        pytest.skip(f"wall-oss-0.5 checkpoint not found at {CK}")
    from rpu_backend.adapters.wall_oss import build_wall_oss_action

    torch.manual_seed(0)
    model = build_wall_oss_action(CK, max_seq_len=512)
    yield model
    model.close()


def test_padding_columns_cannot_reach_the_answer(action):
    'An arbitrary real prefix, padded: columns [real_P, P) must contribute EXACTLY 0.'
    model = action
    torch.manual_seed(0)
    head = torch.randint(0, VOCAB, (1, P_REAL))
    tail_a = torch.randint(0, VOCAB, (1, P_EXEC - P_REAL))
    tail_b = tail_a + 1                       # same shape, different content
    ids_a = torch.cat([head, tail_a], dim=1)
    ids_b = torch.cat([head, tail_b], dim=1)
    assert not torch.equal(ids_a, ids_b)

    noise = torch.randn(1, H_ARB, model.action_dim)
    dof = torch.ones(1, H_ARB, model.action_dim)
    pos = (torch.arange(P_EXEC, P_EXEC + H_ARB, dtype=torch.int32)[:, None]
           .expand(H_ARB, 3).contiguous())

    def run(ids, real_p):
        _prefill(model, ids)
        # Fresh BUILD every leg: replay under a shared key is a different file's
        # subject (test_wall_oss_denoise_real_prefix_replay.py), not a variable here.
        model._graph_cache.clear()
        return model.denoise_with_mask(
            P_EXEC, noise, dof, pos, num_steps=STEPS,
            real_prefix_len=real_p)["x_norm"].clone()

    # Control FIRST: if the padding tokens do not move the answer when they are
    # visible, the masked comparison below is vacuous.
    open_a = run(ids_a, P_EXEC)
    open_b = run(ids_b, P_EXEC)
    assert not torch.equal(open_a, open_b), (
        f"with real_prefix_len={P_EXEC} (nothing masked) the two prefixes produced "
        f"the SAME action; the probe is blind, not the mask correct")

    masked_a = run(ids_a, P_REAL)
    masked_b = run(ids_b, P_REAL)
    assert torch.equal(masked_a, masked_b), (
        f"real_prefix_len={P_REAL} of an execution prefix P={P_EXEC}: changing only "
        f"the MASKED tokens [{P_REAL},{P_EXEC}) changed the action by "
        f"max|delta|={(masked_a - masked_b).abs().max().item():.6g} "
        f"(cos {_cos(masked_a, masked_b):.7f}) — the padding columns are not -inf")


def test_mask2d_reproduces_the_trusted_ltm_answer(action):
    'At an LTM-legal prefix the two paths must agree — that is what the mask MEANS.'
    model = action
    P, H = P_ALIGNED, H_EQ
    torch.manual_seed(1)
    _prefill(model, torch.randint(0, VOCAB, (1, P)))

    x = torch.randn(1, H, model.action_dim)
    dof = torch.ones(1, H, model.action_dim)
    out_ltm = _action_forward(model, P, H, x, dof, mask=None, op_id="p4a_ltm")
    out_2d = _action_forward(model, P, H, x, dof,
                             mask=_action_mask(P, H), op_id="p4a_mask2d")

    assert torch.isfinite(out_2d).all(), "MASK_2D produced nan/inf"
    cos = _cos(out_ltm, out_2d)
    assert cos >= EQUIV_THRESH, (
        f"MASK_2D vs LTM cos={cos:.7f} < {EQUIV_THRESH} at P={P} H={H} "
        f"(mask [{H},{P + H}])")

    out_bad = _action_forward(
        model, P, H, x, dof,
        mask=_action_mask(P, H, causal_off_by_one=True), op_id="p4a_offbyone")
    cos_bad = _cos(out_ltm, out_bad)
    assert cos_bad < EQUIV_THRESH, (
        f"an off-by-one causal band scored cos={cos_bad:.7f} against LTM — the "
        f"comparison cannot tell a correct mask from a wrong one, so the assertion "
        f"above is vacuous")


def test_ltm_still_rejects_a_prefix_that_is_not_16_aligned(action):
    'The reason MASK_2D exists. If LTM ever accepts this, the mask path is dead code.'
    model = action
    torch.manual_seed(2)
    _prefill(model, torch.randint(0, VOCAB, (1, P_REAL)))

    x = torch.randn(1, H_ARB, model.action_dim)
    dof = torch.ones(1, H_ARB, model.action_dim)
    with pytest.raises((RuntimeError, ValueError)):
        _action_forward(model, P_REAL, H_ARB, x, dof, mask=None, op_id="p4a_ltm_arb")

    out = _action_forward(model, P_REAL, H_ARB, x, dof,
                          mask=_action_mask(P_REAL, H_ARB), op_id="p4a_mask2d_arb")
    assert torch.isfinite(out).all(), (
        f"MASK_2D produced nan/inf at the arbitrary prefix P={P_REAL} "
        f"(mask [{H_ARB},{P_REAL + H_ARB}])")
