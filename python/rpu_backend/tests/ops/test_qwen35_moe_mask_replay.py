from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tensors', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--expect-schedule', choices=('baseline', 'retained'), required=True)
    args = parser.parse_args()
    import torch
    import rpu_backend as rb
    from rpu_backend.api.qwen3_5_moe_cache import Qwen3_5MoeCache
    from rpu_backend.adapters.qwen3_5_moe.weights import build_route_tables
    from rpu_backend.quant.fp8 import quantize_fp8_e4m3_per_output
    from rpu_backend.adapters.qwen3_5.text import _build_partial_mrope_cos_sin
    from rpu_backend.adapters.qwen3_5.weights import pad_gdn_scalar
    from rpu_backend.graph import GraphCache, GraphSignature, GraphRuntimePolicy
    from rpu_backend.runtime.execution_planner import _decode_native_stage_domain
    from rpu_backend.runtime.weights import transform_linear_weight
    from rpu_backend.tests._hw_closure_evidence import complete_descriptor_evidence
    torch.manual_seed(42)
    (h, inter, nq, nkv, hd) = (2048, 512, 16, 8, 256)
    (nvh, nkh, dk, dv, convk) = (32, 16, 128, 128, 4)
    (convdim, length) = (8192, 256)
    is_full = [int(layer % 4 == 3) for layer in range(40)]
    (layers, experts, topk) = (40, 256, 8)
    empty = torch.empty(0)
    policy = replace(GraphRuntimePolicy.from_environment(), fmb_fast_replay=True, fmb_deep_fast_replay=True)
    report = {'backend': str(Path(rb.__file__).resolve()), 'cases': [], 'geometry': {'hidden': h, 'intermediate': inter, 'heads': nq, 'kv_heads_effective': nkv, 'head_dim': hd, 'gdn_value_heads': nvh, 'gdn_key_heads': nkh, 'layer_is_full': is_full, 'length': length, 'cores': 8}, 'boundary': 'synthetic mixed E4M3 native owner; tied immutable weights, independent L40 state; no model-quality or real-model performance claim', 'weight_geometry': {'experts': experts, 'topk': topk, 'route_capacity': 512}, 'mask_dma_expected': {'legacy': 120, 'retained': 2}}
    outputs = {}
    references = torch.load(args.reference, map_location='cpu', weights_only=True) if args.reference else None

    def linear(n, k, partition, host=None):
        if host is None:
            host = (torch.randn(n, k) * (0.2 / k ** 0.5)).half()
        return transform_linear_weight(host, partition, num_cores=8).to('rpu')

    def ones(n):
        return torch.ones(n, dtype=torch.float16).to('rpu')

    def only_full(x):
        return [x if f else empty for f in is_full]

    def only_gdn(x):
        return [empty if f else x for f in is_full]

    def conv(width):
        host = (torch.randn(width, convk) * 0.15).half()
        return host.reshape(8, width // 8, convk).permute(0, 2, 1).contiguous().to('rpu')

    def ba():
        host = torch.zeros(128, h, dtype=torch.float16)
        host.view(8, 16, h)[:, :nvh // 8] = (torch.randn(8, nvh // 8, h) * (0.1 / h ** 0.5)).half()
        return linear(128, h, 1, host)

    def fp8(n, k, partition):
        host = (torch.randn(n, k) * (0.2 / k ** 0.5)).half()
        (codes, scale) = quantize_fp8_e4m3_per_output(host)
        return (transform_linear_weight(codes, partition, num_cores=8).to('rpu'), scale.to('rpu'))
    (q, qs) = map(only_full, fp8(nq * hd, h, 1))
    (k, ks) = map(only_full, fp8(nkv * hd, h, 1))
    (v, vs) = map(only_full, fp8(nkv * hd, h, 1))
    (o, os) = map(only_full, fp8(h, nq * hd, 0))
    (ag, ags) = map(only_full, fp8(nq * hd, h, 1))
    (qn, kn) = (only_full(ones(hd)), only_full(ones(hd)))
    (inrm, postrm) = ([ones(h)] * layers, [ones(h)] * layers)
    router = [linear(experts, h, 1)] * layers
    (shared_g, shared_u) = ([linear(inter, h, 1)] * layers, [linear(inter, h, 1)] * layers)
    shared_d = [linear(h, inter, 0)] * layers
    scalar = (torch.randn(1, h) * (0.1 / h ** 0.5)).half().expand(128, h).contiguous()
    shared_scalar = [linear(128, h, 1, scalar)] * layers
    routed = torch.full((experts, h * inter), 24, dtype=torch.uint8).to('rpu')
    routed_g = routed_u = routed_d = [routed] * layers
    routed_gs = routed_us = [torch.full((experts, inter), 0.0078125, dtype=torch.float16).to('rpu')] * layers
    routed_ds = [torch.full((experts, h), 0.0078125, dtype=torch.float16).to('rpu')] * layers
    (gz, gzs) = map(only_gdn, fp8(nvh * dv, h, 1))
    (gout, gouts) = map(only_gdn, fp8(h, nvh * dv, 0))
    alog = pad_gdn_scalar(torch.zeros(nvh).half(), num_heads=nvh, num_cores=8).to('rpu')
    dt = pad_gdn_scalar(torch.full((nvh,), -2.0).half(), num_heads=nvh, num_cores=8).to('rpu')
    gn = only_gdn(ones(dv))
    (gq, gqs) = map(only_gdn, fp8(nkh * dk, h, 1))
    (gk, gks) = map(only_gdn, fp8(nkh * dk, h, 1))
    (gv, gvs) = map(only_gdn, fp8(nvh * dv, h, 1))
    (gcq, gck, gcv) = (only_gdn(conv(n)) for n in (nkh * dk, nkh * dk, nvh * dv))
    (gb, ga) = (only_gdn(ba()), only_gdn(ba()))
    (cos, sin) = _build_partial_mrope_cos_sin(hd, 64, 10000000.0, 512)
    final = ones(h)
    (expert_ids, token_ids) = build_route_tables(512, experts, topk)
    scales = [scale for group in (qs, ks, vs, os, ags, routed_gs, routed_us, routed_ds, gzs, gouts, gqs, gks, gvs) for scale in group]
    handle = int(torch.ops.rpu.qwen3_5_moe_create())
    graphs = []
    try:
        torch.ops.rpu.qwen3_5_moe_set_execution_cores(handle, 8)
        torch.ops.rpu.qwen3_5_moe_set_weights(handle, q, k, v, o, qn, kn, ag, scales, inrm, postrm, router, routed_g, routed_u, routed_d, shared_g, shared_u, shared_d, shared_scalar, cos, sin, final, expert_ids, token_ids, is_full, nq, nkv, hd, h, experts, topk, inter, inter, 8, 5, 1e-06, True, [11, 11, 10], gz, gout, only_gdn(alog), only_gdn(dt), gn, nvh, dk, dv, convdim, convk, gq, gk, gv, gcq, gck, gcv, gb, ga)
        torch.ops.rpu.qwen3_5_moe_set_prefill_rope(handle, cos[:length].contiguous(), sin[:length].contiguous())
        torch.ops.rpu.qwen3_5_moe_set_valid_prefill_len(handle, length)
        torch.ops.rpu.qwen3_5_moe_set_retained_prefill_graph(handle, True)
        cache = Qwen3_5MoeCache(layers, 1, 512, nkv, hd, ['full_attention' if f else 'linear_attention' for f in is_full], nvh, dk, dv, convdim, convk, attn_tp=8)
        inputs = [(torch.randn(1, length, h) * 0.2).half().to('rpu') for _ in range(3)]
        for chunk in (128,):
            wire = torch.ops.rpu.qwen3_5_moe_resolve_prefill_stage_domain(handle, length, length, chunk)
            domain = _decode_native_stage_domain(wire, execution_len=length, logical_len=length, position=0, graph_mode='RETAINED_CACHE')
            pairs = []
            for row in domain:
                evidence = complete_descriptor_evidence(row.physical_descriptor, expected_lifecycle='RETAINED_CACHE', component_path='synthetic_qwen35_moe', stage='prefill')
                routes = evidence['routes']
                if row.compute_chunk == chunk and all((r['selector'] == 1 for r in routes if r['family'] == 1)):
                    pairs.append((row, evidence, routes))
            assert len(pairs) == 1, 'one explicit DDR-attention candidate required'
            (row, evidence, routes) = pairs[0]
            masks = [r for r in routes if r['site_id'] == 4606644302213524820]
            if args.expect_schedule == 'retained':
                assert len(masks) == 1 and masks[0]['selector'] == 2
                assert masks[0]['arguments'] == [64, 64, 2, 0, 30, chunk, length // chunk, 8, 8]
            else:
                assert not masks or masks[0]['selector'] == 1
            graph = GraphCache(runtime_policy=policy)
            graphs.append(graph)
            sig = GraphSignature(op_id='qwen35_moe_gdn_mask', shapes=[1, length, h], dyn_dims=list(row.physical_descriptor), dtypes=[torch.float16])
            measured = []
            seen = {}
            for iteration in range(7):
                index = iteration % len(inputs)
                cache.reset()
                next((t for t in reversed(cache.conv_states) if t.numel())).cpu()
                start = time.perf_counter_ns()
                with graph.capture(sig):
                    result = torch.ops.rpu.qwen3_5_moe_forward(handle, inputs[index], cache.k_caches, cache.v_caches, cache.gdn_states, cache.conv_states, None, 0, True, planned_chunk_size=0, planned_stage_descriptor=row.physical_descriptor)
                output = result.cpu().clone()
                elapsed = (time.perf_counter_ns() - start) / 1000000.0
                assert output.shape == (1, length, h) and torch.isfinite(output).all()
                assert output.count_nonzero() > 0
                if index in seen:
                    assert torch.equal(output, seen[index]), 'reset/replay output changed'
                seen[index] = output
                key = f'c{chunk}_input{index}'
                outputs[key] = output
                if references is not None:
                    assert torch.equal(output, references[key]), (key, float((output - references[key]).abs().max()))
                for (kind, tensors) in (('recurrent', cache.gdn_states), ('conv', cache.conv_states), ('key', cache.k_caches), ('value', cache.v_caches)):
                    for (layer, tensor) in enumerate(tensors):
                        if not tensor.numel():
                            continue
                        state = tensor.cpu().clone()
                        assert torch.isfinite(state).all()
                        state_key = f'{key}_{kind}_{layer}'
                        if state_key in outputs:
                            assert torch.equal(state, outputs[state_key]), state_key
                        outputs[state_key] = state
                        if references is not None:
                            assert torch.equal(state, references[state_key]), state_key
                if iteration >= 2:
                    measured.append(elapsed)
                if iteration == 0:
                    graph.freeze()
            assert not torch.equal(seen[0], seen[1]), 'changed-input positive control is insensitive'
            assert graph.size() == 1 and graph.cache_invariant_ok()
            (entry,) = graph.snapshot()
            assert entry.replay_count >= 6 and entry.recapture_count == 0
            report['cases'].append({'chunk': chunk, 'mask_schedule': masks, 'physical_descriptor': list(row.physical_descriptor), 'host_forward_ms': measured, 'median_ms': statistics.median(measured), 'byte_equal_reset_replay': True, 'reference_compared': references is not None, 'graph': {name: getattr(entry, name) for name in ('replay_count', 'recapture_count', 'kernel_count', 'dma_count')}})
        if references is not None:
            assert outputs.keys() == references.keys()
        report['passed'] = True
    finally:
        for graph in graphs:
            graph.clear()
        torch.ops.rpu.qwen3_5_moe_destroy(handle)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    args.tensors.parent.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, args.tensors)
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
