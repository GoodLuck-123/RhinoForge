'Pi05 mask replay device regression.'
import json
import os

import pytest

from rpu_backend.tests.fixtures.pi05_fuse_fixture import pi05_fuse_fixture  # noqa: F401


pytestmark = pytest.mark.skipif(
    os.environ.get("PI05_HARDWARE") != "1",
    reason="Requires hardware + Pi0.5 weights; set PI05_HARDWARE=1 with the device lock.",
)


def test_changed_mask_replay_matches_fresh_graph(pi05_fuse_fixture):
    import torch
    from rpu_backend.adapters.pi05 import _pi05_graph_runtime_policy
    from rpu_backend.graph import GraphCache
    from rpu_backend.runtime.masks import versioned_cpu_mask

    fx = pi05_fuse_fixture
    cfg = fx["config"]
    policy = _pi05_graph_runtime_policy()
    assert policy.fmb_fast_replay and policy.fmb_deep_fast_replay, (
        "This regression must exercise the normal Pi05 fast/deep replay policy")
    descriptor = fx["action_plan"].selected.stage_tuple.physical_descriptor
    assert descriptor
    shape = (1, cfg.chunk_size, cfg.max_action_dim)
    generator = torch.Generator(device="cpu").manual_seed(42)
    hidden = (torch.randn(shape, generator=generator) * 0.2).half().to("rpu")
    cond = fx["adarms_cond_all"].select(0, 0).contiguous()
    mask_a = versioned_cpu_mask(fx["mask_4d"])
    prefix_len = int(fx["prefix_len"])
    assert prefix_len > 0 and mask_a.shape[-1] > prefix_len
    assert (mask_a[..., :prefix_len] == 0).any()
    mask_b = versioned_cpu_mask(mask_a)
    mask_b[..., :prefix_len] = float("-inf")
    assert torch.isfinite(mask_b[..., prefix_len:]).any(dim=-1).all(), (
        "Each query must retain an attended action key")

    def forward(cache, mask):
        # A fresh returned buffer also exercises mutable output DMA on replay.
        output = torch.empty(shape, dtype=torch.float16, device="rpu")
        with cache.capture(fx["sig"]):
            torch.ops.rpu.pi05_denoise_step_forward(
                fx["handle"], hidden,
                fx["past_key_values"].k_caches, fx["past_key_values"].v_caches,
                cond, mask, output, prefix_len, descriptor,
            )
        result = output.cpu().clone()
        assert result.shape == shape and torch.isfinite(result).all()
        return result

    # Finish references before the uninterrupted replay sequence. A fresh
    # reference between mutations would replace the native memo's tensor owner
    # and could conceal an identity-only cache failing to see an in-place edit.
    references = {}
    for name, mask in (("A", mask_a), ("B", mask_b)):
        fresh = GraphCache(runtime_policy=policy)
        try:
            references[name] = forward(fresh, mask)
        finally:
            fresh.clear()
    delta = (references["A"].float() - references["B"].float()).abs()
    assert float(delta.max()) > 0, (
        "Masked-prefix positive control is insensitive: cannot detect a stale mask")
    assert references["A"].count_nonzero() > 0

    stable = versioned_cpu_mask(mask_a)
    other = versioned_cpu_mask(mask_b)
    assert not torch.is_inference(stable) and not torch.is_inference(other)
    with torch.inference_mode():
        unversioned = mask_b.clone()
    assert torch.is_inference(unversioned)
    replay = GraphCache(runtime_policy=policy)
    cases = []
    try:
        # BUILD + five REPLAYs: two edits of one versioned owner, a distinct
        # same-shape owner, then two calls of one unversioned inference owner.
        for index, (label, expected, mask) in enumerate((
            ("versioned_A", "A", stable),
            ("versioned_inplace_B", "B", stable),
            ("versioned_inplace_A", "A", stable),
            ("different_owner_B", "B", other),
            ("inference_B", "B", unversioned),
            ("inference_inplace_A", "A", unversioned),
        )):
            if index in (1, 2):
                old_version = stable._version
                stable.copy_(mask_b if expected == "B" else mask_a)
                assert stable._version > old_version
            if torch.is_inference(mask):
                with torch.inference_mode():
                    if index == 5:
                        unversioned.copy_(mask_a)
                    actual = forward(replay, mask)
            else:
                actual = forward(replay, mask)
            assert torch.equal(actual, references[expected]), (
                label, float((actual.float() - references[expected].float()).abs().max()))
            if index == 0:
                replay.freeze()
            cases.append({"case": label, "byte_equal_fresh_graph": True})
        assert replay.is_frozen() and replay.size() == 1 and replay.cache_invariant_ok()
        entries = replay.snapshot()
        assert len(entries) == 1
        entry = entries[0]
        assert entry.recapture_count == 0 and entry.prepared_segment_hit_total >= 5
        assert entry.kernel_count > 0, "An empty Graph is not replay evidence"
        stats = replay.lookup(fx["sig"]).debug_stats()
        report = {
            "passed": True, "cases": cases,
            "positive_control_max_abs_difference": float(delta.max()),
            "prepared_segment_hits": entry.prepared_segment_hit_total,
            "recapture_count": entry.recapture_count,
            "graph_phase": replay.phase,
            "kernel_count_diagnostic": entry.kernel_count,
            "graph_stats": dict(stats) if hasattr(stats, "keys") else str(stats),
            "physical_descriptor": list(descriptor),
            "boundary": "single step; production loop tested separately",
        }
        print(json.dumps(report, sort_keys=True), flush=True)
    finally:
        replay.clear()
