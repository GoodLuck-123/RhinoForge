'Hyvla action mask replay device regression.'
import dataclasses
import json
import os

import pytest


pytestmark = [pytest.mark.hardware, pytest.mark.skipif(
    os.environ.get("RPU_HW_AVAILABLE") != "1", reason="Requires device + device lock")]


def test_hyvla_unroll_mask_mutation_matches_original_schedule():
    import torch
    import rpu_backend  # noqa: F401
    from rpu_backend.api.cache import RPUCache
    from rpu_backend.graph import GraphCache, GraphRuntimePolicy, GraphSignature
    from rpu_backend.runtime.execution_planner import _decode_native_stage_domain
    from rpu_backend.runtime.masks import versioned_cpu_mask
    from rpu_backend.runtime.weights import (
        tp_col_swizzle_mc_weight as col, tp_row_swizzle_mc_weight as row,
    )

    torch.set_num_threads(4)
    gen = torch.Generator(device="cpu").manual_seed(42)
    h, nq, nkv, hd, inter, layers, steps = 1024, 16, 8, 128, 2048, 2, 3
    m, prefix, width = 51, 240, 32
    total, capacity = prefix + m, 304

    def random(shape, scale=0.03):
        return (torch.randn(shape, generator=gen) * scale).half()

    def weight(n, k, cores, partition):
        value = random((n, k), 0.5 / k ** 0.5)
        return (col(value, cores) if partition else row(value, cores)).to("rpu")

    weights = {key: [] for key in (
        "q", "k", "v", "o", "qn", "kn", "input", "post", "gate", "up", "down")}
    for _ in range(layers):
        for key, n, k, cores, partition in (
            ("q", nq * hd, h, 8, 1), ("k", nkv * hd, h, 8, 1),
            ("v", nkv * hd, h, 8, 1), ("o", h, nq * hd, 8, 0),
            ("gate", inter, h, 8, 1), ("up", inter, h, 8, 1),
            ("down", h, inter, 8, 0),
        ):
            weights[key].append(weight(n, k, cores, partition))
        for key, n in (("qn", hd), ("kn", hd), ("input", h), ("post", h)):
            weights[key].append(torch.ones(n, dtype=torch.float16, device="rpu"))
    final_norm = torch.ones(h, dtype=torch.float16, device="rpu")
    cos = torch.ones(capacity, hd // 2, dtype=torch.float16, device="rpu")
    sin = torch.zeros_like(cos)
    action_weights = (
        weight(h, width, 8, 1), weight(h, h, 8, 0), random((h,)).to("rpu"),
        weight(width, h, 1, 1), random((width,)).to("rpu"),
        random((steps, h), 0.2).to("rpu"),
    )
    x0 = random((1, m, width), 0.2).to("rpu")
    state = random((h,), 0.2).to("rpu")
    cache = RPUCache(layers, 1, capacity, nkv, hd, attn_tp=8)
    # Random raw swizzled storage is a valid arbitrary FP16 prefix. The suffix
    # is overwritten before attention; no logical-prefix generation is needed.
    for tensor in (*cache.k_caches, *cache.v_caches):
        tensor.copy_(random(tuple(tensor.shape), 0.2))
    mask_a = versioned_cpu_mask(torch.zeros(1, 1, m, total))
    mask_b = versioned_cpu_mask(mask_a)
    mask_b[..., :prefix] = float("-inf")
    policy = dataclasses.replace(
        GraphRuntimePolicy.from_environment(),
        fmb_fast_replay=True, fmb_deep_fast_replay=True,
        provenance=("HY action mask lifetime regression",),
    )

    def create(mask_once):
        handle = int(torch.ops.rpu.hyvla_expert_create())
        try:
            torch.ops.rpu.hyvla_expert_set_runtime_config(
                handle, True, True, mask_once, True, True, True, True)
            torch.ops.rpu.hyvla_expert_set_weights(
                handle, *(weights[key] for key in weights), cos, sin, final_norm,
                nq, nkv, hd, h, inter, 1e-5, m, 8)
            torch.ops.rpu.hyvla_expert_set_action_weights(
                handle, *action_weights, width, steps)
            domain = torch.ops.rpu.hyvla_expert_resolve_stage_domain(handle, m, prefix, total)
            candidates = _decode_native_stage_domain(
                domain, execution_len=m, logical_len=m, position=prefix,
                graph_mode="COMPOSITE_CHILD")
            assert candidates
            descriptor = tuple(candidates[0].physical_descriptor)
            site_index = descriptor.index(1728596774874433151)
            assert descriptor[site_index + 1] == (7 if mask_once else 3)
            return handle, descriptor
        except BaseException:
            torch.ops.rpu.hyvla_expert_destroy(handle)
            raise

    def forward(handle, descriptor, graph, mask):
        output = torch.empty(steps, m, width, dtype=torch.float16, device="rpu")
        signature = GraphSignature(
            op_id="hyvla_mask_lifetime_regression", shapes=[m, h],
            dyn_dims=[layers, prefix, steps, *descriptor], dtypes=[torch.float16])
        cache.reset_to_position(prefix)
        with graph.capture(signature):
            torch.ops.rpu.hyvla_expert_unroll_forward(
                handle, x0, cache.k_caches, cache.v_caches, state, cos, sin,
                mask, output, -1.0 / steps, prefix, steps, descriptor)
        # Capture end executes the synchronous graph; CPU output is the read boundary.
        actual = output.cpu().clone()
        assert torch.isfinite(actual).all() and actual.shape == (steps, m, width)
        return actual

    refs = {}
    handle, descriptor = create(False)
    try:
        for name, mask in (("A", mask_a), ("B", mask_b)):
            graph = GraphCache(runtime_policy=policy)
            try:
                refs[name] = forward(handle, descriptor, graph, mask)
            finally:
                graph.clear()
    finally:
        torch.ops.rpu.hyvla_expert_destroy(handle)
    delta = float((refs["A"].float() - refs["B"].float()).abs().max())
    assert delta > 0, "Masked-prefix positive control has no observable effect"

    handle, descriptor = create(True)
    graph = GraphCache(runtime_policy=policy)
    stable = versioned_cpu_mask(mask_a)
    other = versioned_cpu_mask(mask_b)
    with torch.inference_mode():
        unversioned = mask_b.clone()
    records = []
    try:
        # Uninterrupted BUILD + five REPLAYs; no fresh run can replace the
        # memo owner between the in-place edits. Compare every body, all rows.
        for i, (label, expected, mask) in enumerate((
            ("versioned_A", "A", stable), ("inplace_B", "B", stable),
            ("inplace_A", "A", stable), ("other_owner_B", "B", other),
            ("inference_B", "B", unversioned), ("inference_inplace_A", "A", unversioned),
        )):
            if i in (1, 2):
                stable.copy_(mask_b if expected == "B" else mask_a)
            if torch.is_inference(mask):
                with torch.inference_mode():
                    if i == 5:
                        mask.copy_(mask_a)
                    actual = forward(handle, descriptor, graph, mask)
            else:
                actual = forward(handle, descriptor, graph, mask)
            assert torch.equal(actual, refs[expected]), (
                label, float((actual.float() - refs[expected].float()).abs().max()))
            if i == 0:
                graph.freeze()
            records.append({"case": label, "all_bodies_equal_original_schedule": True})
        entry, = graph.snapshot()
        assert graph.is_frozen() and graph.size() == 1 and graph.cache_invariant_ok()
        assert entry.kernel_count > 0 and entry.recapture_count == 0
        assert entry.prepared_segment_hit_total >= 5
        print(json.dumps({
            "passed": True, "geometry": [layers, steps, m, total, h, nq, nkv, hd],
            "positive_control_max_abs": delta, "cases": records,
            "prepared_segment_hits": entry.prepared_segment_hit_total,
            "recapture_count": entry.recapture_count,
            "physical_descriptor": descriptor,
            "boundary": "synthetic native action; not full HY VLA E2E or a performance claim",
        }, sort_keys=True), flush=True)
    finally:
        graph.clear()
        torch.ops.rpu.hyvla_expert_destroy(handle)
