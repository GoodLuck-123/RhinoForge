#!/usr/bin/env python3
"""Real-checkpoint LingBot2 layer-0 dense/sparse Graph comparison.

Run this script under the board's normal idle check and exclusive device lock.
The parent process serializes fresh correctness and performance workers, keeping
one native model per process. This is a one-layer component probe with explicit
deterministic KV/AdaRMS inputs, not public-model or exact-Z2 E2E acceptance.
The supplied CPU MoE golden contributes its hidden input only: its output is not
an oracle for attention, residuals and final RMSNorm in the complete layer.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def _metrics(actual, reference):
    import torch

    assert actual.shape == reference.shape
    a, b = actual.double().reshape(-1), reference.double().reshape(-1)
    delta = a - b
    return {
        "max_abs": float(delta.abs().max()),
        "mse": float(delta.square().mean()),
        "relative_l2": float(delta.norm() / b.norm().clamp_min(1e-30)),
        "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        "actual_rms": float(a.square().mean().sqrt()),
        "reference_rms": float(b.square().mean().sqrt()),
    }


def _validate_sparse_route_stage(stage, router, *, padded_rows, top_k):
    """Check real route tasks in the existing relay; padding is not a task.

    The slab's first region is FP16 weights; IDs/counts carry raw unsigned
    16-bit integers in the same FP16 storage. Counting (token, expert) pairs
    detects missing or duplicated routes even when expert totals still match.
    This fixture requires nonzero weights for every selected router member.
    """
    import torch

    assert stage.device.type == router.device.type == "cpu"
    assert stage.dtype == router.dtype == torch.float16
    assert stage.ndim == 1 and stage.is_contiguous() and router.ndim == 2
    rows, experts = router.shape
    assert 0 < rows <= padded_rows and 0 < top_k <= experts
    align_half = lambda elements: ((elements * 2 + 255) // 256) * 128
    expert_offset = align_half(padded_rows * experts)
    token_offset = expert_offset + align_half(padded_rows * top_k)
    count_offset = token_offset + align_half(padded_rows * top_k)
    assert stage.numel() >= count_offset + align_half(experts), "truncated sparse route stage"
    prefix_elements = count_offset + align_half(experts)
    assert stage.numel() in (prefix_elements, prefix_elements + padded_rows * experts), "unknown sparse route stage format"
    assert torch.isfinite(router).all()
    weights = stage[:padded_rows * experts].view(padded_rows, experts)[:rows]
    assert torch.equal(weights, router), "sparse route weights differ from captured router"
    raw = stage.view(torch.int16).to(torch.int64).bitwise_and(0xffff)
    tasks = rows * top_k
    expert_ids = raw[expert_offset:expert_offset + tasks]
    token_ids = raw[token_offset:token_offset + tasks]
    counts = raw[count_offset:count_offset + experts]
    assert bool(expert_ids.lt(experts).all()), "sparse route expert ID out of range"
    assert bool(token_ids.lt(rows).all()), "sparse route token ID out of range"
    assert bool((expert_ids[1:] >= expert_ids[:-1]).all()), "sparse routes are not expert sorted"
    ordered_pairs = expert_ids * rows + token_ids
    assert bool((ordered_pairs[1:] > ordered_pairs[:-1]).all()), "sparse tasks must be unique and expert/token ordered"
    if stage.numel() > prefix_elements:
        inverse = raw[prefix_elements:].reshape(padded_rows, experts)
        expected_inverse = torch.zeros_like(inverse)
        expected_inverse[token_ids, expert_ids] = torch.arange(1, tasks + 1)
        assert torch.equal(inverse, expected_inverse), "inverse task indices differ from exact route order or padded zero rows"
    expected = router.ne(0).to(torch.int64)
    assert bool(expected.sum(dim=1).eq(top_k).all()), "fixture requires nonzero selected router weights"
    membership = torch.bincount(token_ids * experts + expert_ids,
                                minlength=rows * experts).reshape(rows, experts)
    assert torch.equal(membership, expected), "sparse route membership differs from router"
    assert torch.equal(counts, expected.sum(dim=0)), "sparse route expert counts differ from router"
    return {"sparse_route_membership": membership.clone(),
            "sparse_route_counts": counts.clone()}


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Complete checkpoint or exact layer-0 extraction plus final norm")
    parser.add_argument("--golden", type=Path, required=True,
                        help="moe_golden_layer0.pt (only its input is used)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp16", "w8a16"), default="fp16")
    parser.add_argument("--mode", choices=("compare", "dense", "sparse"), default="compare")
    parser.add_argument("--phase", choices=("all", "correctness", "performance"), default="all",
                        help="compare runs both phases by default; select correctness for just two workers")
    parser.add_argument("--concentrated-routing", action="store_true",
                        help="Replace the cold fixture's router bias: all tokens choose experts 0..3")
    parser.add_argument("--runtime-graph", action="store_true",
                        help="Enable the fused router cold capability when compatible")
    parser.add_argument("--routed-gemm", action="store_true",
                        help="Use RoutedCol on dense Gate/Up (single dense worker only)")
    args = parser.parse_args()
    if not args.checkpoint.is_dir() or not args.golden.is_file():
        parser.error("checkpoint directory and layer-0 golden must exist")
    if os.environ.get("RPU_LINGBOT2_MULTIVIEW_SPM_Z2") == "1":
        parser.error("this component probe does not accept the exact-Z2 profile")
    if args.mode != "compare" and args.phase == "all":
        parser.error("a single dense/sparse worker requires --phase correctness or performance")
    if args.routed_gemm and args.mode != "dense":
        parser.error("--routed-gemm requires --mode dense")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    return args


def _compare(args):
    reports = {}
    phases = ("correctness", "performance") if args.phase == "all" else (args.phase,)
    for phase in phases:
        for mode in ("dense", "sparse"):
            path = args.output.with_name(f"{args.output.stem}.{mode}.{phase}.json")
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--checkpoint", str(args.checkpoint), "--golden", str(args.golden),
                       "--output", str(path), "--dtype", args.dtype,
                       "--mode", mode, "--phase", phase]
            if args.concentrated_routing:
                command.append("--concentrated-routing")
            if args.runtime_graph:
                command.append("--runtime-graph")
            subprocess.run(command, check=True)
            reports[f"{mode}.{phase}"] = json.loads(path.read_text())
    import torch

    comparisons = {}
    if "correctness" in phases:
        saved = {}
        for mode in ("dense", "sparse"):
            path = args.output.with_name(f"{args.output.stem}.{mode}.correctness.pt")
            saved[mode] = torch.load(path, map_location="cpu", weights_only=True)
        for variant in ("A", "B"):
            dense, sparse = saved["dense"][variant], saved["sparse"][variant]
            # Both workers install identical attention/router/base precision. Only
            # the routed expert execution is changed, so selected weights must agree.
            assert torch.equal(dense["router"], sparse["router"]), "dense/sparse router drift"
            assert torch.equal(dense["router_h"], sparse["router_h"]), "pre-MoE hidden drift"
            assert torch.equal(dense["attn_resid"], sparse["attn_resid"]), "attention drift"
            assert torch.equal(dense["innorm"], sparse["innorm"]), "input AdaRMS snapshot drift"
            comparisons[variant] = {
                "whole_layer": _metrics(sparse["output"], dense["output"]),
                "before_final_norm": _metrics(sparse["layer0"], dense["layer0"]),
                "router_exact": True,
            }
    result = {"passed_runtime_checks": True, "dtype": args.dtype,
              "scope": "real layer-0 weights; deterministic synthetic KV/AdaRMS; no Z2 or model-quality claim",
              "modified_router_bias": args.concentrated_routing,
              "numeric_comparison_record_only": comparisons, "workers": reports}
    if args.concentrated_routing:
        result["scope"] += "; modified cold router bias forces experts 0..3 (28 zero-count groups)"
    if "performance" in phases:
        result["wall_speedup"] = (reports["dense.performance"]["wall_ms_median"] /
                                  reports["sparse.performance"]["wall_ms_median"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def _configure(args):
    # Clear branch/debug inputs that would change a worker's intended route.
    for name in ("RPU_L2_BUFONLY", "RPU_L2_ADDR", "RPU_L2_ROUTED_ONLY",
                 "RPU_L2_DOWN_ACC16", "RPU_L2_STAGES", "RPU_L2_CAPTURE_GATE",
                 "RPU_LINGBOT2_DEBUG_DENSE_SOFT_ROUTER", "RPU_LINGBOT2_EXPERT_W4A16"):
        os.environ.pop(name, None)
    sparse = args.mode == "sparse"
    diagnostic = args.phase == "correctness"
    os.environ.update({
        "RPU_LINGBOT2_GROUPED_EXPERTS": str(int(not sparse)),
        "RPU_LINGBOT2_SPARSE_EXPERTS": str(int(sparse)),
        "RPU_LINGBOT2_RUNTIME_GRAPH": str(int(args.runtime_graph)),
        "RPU_LINGBOT2_ROUTED_GEMM": str(int(args.routed_gemm)),
        "RPU_LINGBOT2_EXPERT_W8A16": str(int(args.dtype == "w8a16")),
        "RPU_LINGBOT2_BASE_W8A16": str(int(args.dtype == "w8a16")),
        "RPU_LINGBOT2_FP16_TOP4": "1", "RPU_ADARMS_FUSED_BCAST": "0",
        "RPU_LINGBOT2_DENOISE_UNROLL": "0", "RPU_LINGBOT2_MULTIVIEW_SPM_Z2": "0",
        "RPU_L2_CAPTURE_L0": str(int(diagnostic)),
        "RPU_L2_CAPTURE_INNORM": str(int(diagnostic)),
        "RPU_L2_CAP_RESID": str(int(diagnostic)),
        "RPU_LINGBOT2_DEBUG_DUMP_ROUTER_H": str(int(diagnostic)),
    })


def _worker(args):
    _configure(args)
    import torch
    import rpu_backend
    from types import SimpleNamespace
    from rpu_backend.adapters.lingbot_vla_v2 import convert
    from rpu_backend.api import RPUCache
    from rpu_backend.graph import GraphCache, GraphRuntimePolicy, GraphSignature
    from rpu_backend.runtime.decoder import plan_bounded_prefill_execution

    torch.set_num_threads(1)
    torch.manual_seed(42)
    q, h, prefix, rope, horizon = 51, 768, 225, 32, 512
    source = torch.load(args.golden, map_location="cpu", weights_only=True)["input"]
    assert source.shape == (1, 50, h) and torch.isfinite(source).all()
    a = torch.cat((torch.zeros_like(source[:, :1]), source), dim=1).half()
    b = (-a).contiguous()
    source_gw, readers = convert.ckpt_reader(str(args.checkpoint))

    def gw(key):
        value = source_gw(key)
        if args.concentrated_routing and key.endswith(".mlp.e_score_correction_bias"):
            assert value.shape == (32,)
            value = torch.zeros_like(value)
            # Sigmoid scores lie in [0,1]. A representable +8 selection-only
            # bias makes experts 0..3 win for every row without altering weights.
            value[:4] = 8
        return value
    geom = convert._default_geom()
    geom["EXP_LAYERS"] = 1
    # A retained final norm is required even when using a layer-0 extraction.
    assert gw(convert.EXP_P + "norm.weight").shape == (h,)
    cold = SimpleNamespace(bufonly=False, addr=False, schunk=32512,
                           rchunk_requested=0 if args.mode == "sparse" else 1632,
                           dense_soft_router=False,
                           fp16_top4=True, routed_only=False, down_acc16=False,
                           sparse_experts=args.mode == "sparse",
                           sparse_w8a16=args.dtype == "w8a16",
                           runtime_graph=args.runtime_graph,
                           routed_gemm=args.routed_gemm)
    owned, graphs = {}, []
    report = {"passed_runtime_checks": False, "mode": args.mode,
              "phase": args.phase, "dtype": args.dtype,
              "checkpoint": str(args.checkpoint.resolve()),
              "golden_input": str(args.golden.resolve()),
              "backend": str(Path(rpu_backend.__file__).resolve()),
              "scope": "real layer-0 weights; deterministic synthetic prefix KV and AdaRMS",
              "modified_router_bias": args.concentrated_routing,
              "runtime_graph": args.runtime_graph,
              "routed_gemm": args.routed_gemm,
              "rows": q, "prefix_rows": prefix, "rope_offset": rope}
    if args.concentrated_routing:
        report["scope"] += "; modified cold router bias forces experts 0..3 (28 zero-count groups)"
    try:
        with torch.inference_mode():
            def publish(handle, keep):
                owned.update(handle=handle, keep=keep)

            handle, keep = convert.build_expert_moe(
                gw, max_seq=horizon, rope_theta=5_000_000,
                runtime_config=cold, geom=geom, on_handle_created=publish)
            films = [((torch.randn(1, h) * .02) + (1 if i % 2 == 0 else 0)).half().to("rpu")
                     for i in range(4)]
            torch.ops.rpu.lingbot_v2_moe_set_adarms_step_mutable(handle, *films)
            kv = RPUCache(1, 1, horizon, 8, 128, device="rpu", attn_tp=8)
            for tensor in (*kv.k_caches, *kv.v_caches):
                tensor.copy_((torch.randn(tuple(tensor.shape)) * .1).half().to("rpu"))
            hidden = a.to("rpu")
            # One explicit additive mask, all prefix and suffix rows visible.
            mask = torch.zeros((1, 1, q, prefix + q), dtype=torch.float16)
            plans = []
            length, chunk = plan_bounded_prefill_execution(
                q, q, 0, position=prefix, alignment=1, padding_rows=0,
                exact_chunk_size=64, graph_mode="COMPOSITE_CHILD",
                queue_owner_id=handle, lease_owner_id=handle,
                resolve_stage_domain=lambda n: torch.ops.rpu.lingbot_v2_moe_resolve_action_stage_domain(
                    handle, n, prefix, prefix + n, rope), plan_result_sink=plans.append)
            assert (length, chunk) == (q, 64)
            selected = plans[0].selected
            descriptor = tuple(selected.stage_tuple.physical_descriptor)
            assert dict(selected.stage_tuple.physical_metadata)["manifest_state"] == 1
            report["physical_descriptor"] = list(descriptor)
            signature = GraphSignature(op_id="lingbot2_sparse_layer0",
                shapes=[q, h], dyn_dims=[prefix, rope, *plans[0].graph_key_words()],
                dtypes=[torch.float16])
            runtime_policy = GraphRuntimePolicy.from_environment()

            def new_graph():
                graph = GraphCache(runtime_policy=runtime_policy)
                graphs.append(graph)
                return graph

            def run(graph, value):
                hidden.copy_(value)
                kv.reset_to_position(prefix)
                with graph.capture(signature):
                    output = torch.ops.rpu.lingbot_v2_moe_forward(
                        handle, hidden, kv.k_caches, kv.v_caches, mask,
                        prefix, rope, -1, descriptor)
                result = output.cpu().clone()
                assert result.shape == (1, q, h) and torch.isfinite(result).all()
                assert bool(result.float().square().sum(dim=-1).gt(0).all()), "zero output row"
                return result

            route_checks = []

            def capture(output):
                router_stage = torch.ops.rpu.lingbot_v2_moe_debug_rw_stage(handle).cpu()
                router = router_stage[0, :, :q].T.contiguous()
                assert torch.isfinite(router).all()
                assert torch.equal(router.ne(0).sum(dim=-1), torch.full((q,), 4))
                if args.concentrated_routing:
                    assert bool(router[:, :4].gt(0).all())
                    assert torch.equal(router[:, 4:], torch.zeros_like(router[:, 4:])), "zero-count experts received tokens"
                captured = {"output": output, "router": router.clone(),
                    "innorm": torch.ops.rpu.lingbot_v2_moe_debug_innorm(handle).cpu()[:q].clone(),
                    "router_h": torch.ops.rpu.lingbot_v2_moe_debug_router_h(handle).cpu()[0, :q].clone(),
                    "attn_resid": torch.ops.rpu.lingbot_v2_moe_debug_attn_resid(handle).cpu()[0, :q].clone(),
                    "layer0": torch.ops.rpu.lingbot_v2_moe_debug_l0_out(handle).cpu()[:q].clone()}
                assert captured["innorm"].shape == (q, h)
                assert torch.isfinite(captured["innorm"]).all(), "invalid input AdaRMS snapshot"
                if args.mode == "sparse":
                    stage = torch.ops.rpu.lingbot_v2_moe_debug_sparse_route_stage(handle).cpu()[0]
                    captured.update(_validate_sparse_route_stage(
                        stage, router, padded_rows=router_stage.shape[-1], top_k=geom["TOP_K"]))
                    route_checks.append(True)
                return captured

            if args.phase == "correctness":
                controls = {}
                for name, value in (("A", a), ("B", b)):
                    graph = new_graph()
                    controls[name] = capture(run(graph, value))
                    graph.clear()
                if not args.concentrated_routing:
                    assert not torch.equal(controls["A"]["router"].ne(0), controls["B"]["router"].ne(0)), "A/B must change expert membership"
                assert not torch.equal(controls["A"]["output"], controls["B"]["output"]), "A/B must change output"
                assert not torch.equal(controls["A"]["innorm"], controls["B"]["innorm"]), "A/B must refresh input AdaRMS snapshot"
                graph = new_graph()
                first = capture(run(graph, a))
                for key in first:
                    assert torch.equal(first[key], controls["A"][key]), f"fresh A differs: {key}"
                graph.freeze()
                for name, value in (("B", b), ("A", a), ("A", a)):
                    actual = capture(run(graph, value))
                    for key in actual:
                        assert torch.equal(actual[key], controls[name][key]), f"replay {name} differs: {key}"
                entry, = graph.snapshot()
                assert entry.replay_count == 3
                torch.save(controls, args.output.with_suffix(".pt"))
                report["a_b_router_membership_changes"] = int(
                    (controls["A"]["router"].ne(0) != controls["B"]["router"].ne(0)).any(dim=-1).sum())
                report["a_b_output_delta"] = _metrics(controls["B"]["output"], controls["A"]["output"])
                report["input_adarms_snapshot_replay_exact"] = True
                report["expert_token_counts"] = {
                    name: value["router"].ne(0).sum(dim=0).tolist()
                    for name, value in controls.items()}
                if args.mode == "sparse":
                    report["sparse_route_membership_and_counts_exact"] = True
                    report["sparse_route_structure_checks"] = len(route_checks)
            else:
                graph = new_graph()
                baseline = run(graph, a)
                graph.freeze()
                for _ in range(2):
                    assert torch.equal(run(graph, a), baseline)
                durations = []
                for _ in range(5):
                    start = time.perf_counter()
                    actual = run(graph, a)
                    durations.append((time.perf_counter() - start) * 1000)
                    assert torch.equal(actual, baseline)
                entry, = graph.snapshot()
                assert entry.replay_count == 7
                report.update(warmups=2, measured=5, wall_ms=durations,
                              wall_ms_median=statistics.median(durations),
                              timing="wall time including input/output copy; all debug captures disabled")
            assert entry.recapture_count == 0 and entry.kernel_count > 0
            assert graph.size() == 1 and graph.cache_invariant_ok() and graph.is_frozen()
            report.update(passed_runtime_checks=True, replay_count=entry.replay_count,
                          recapture_count=entry.recapture_count, kernel_count=entry.kernel_count,
                          dma_count=entry.dma_count, graph_entries=graph.size(), ready=graph.is_frozen())
    finally:
        # Preserve keepalive references until every graph retires and the native
        # handle is destroyed. Do not continue uncertain teardown after failure.
        for graph in graphs:
            graph.clear()
        if owned:
            torch.ops.rpu.lingbot_v2_moe_destroy(owned["handle"])
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    args = _args()
    if args.mode == "compare":
        _compare(args)
    else:
        _worker(args)


if __name__ == "__main__":
    main()
