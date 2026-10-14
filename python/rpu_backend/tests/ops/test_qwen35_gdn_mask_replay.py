from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

def _layer_types(profile, full_depth):
    if full_depth:
        if profile != '2b':
            raise ValueError('--full-depth requires --profile 2b')
        return [0, 0, 0, 1] * 6
    return [0, 1, 0]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tensors', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--expect-schedule', choices=('baseline', 'retained'), required=True)
    parser.add_argument('--profile', choices=('4b', '2b'), default='4b')
    parser.add_argument('--full-depth', action='store_true')
    args = parser.parse_args()
    try:
        is_full = _layer_types(args.profile, args.full_depth)
    except ValueError as error:
        parser.error(str(error))
    nl = len(is_full)
    gdn_layers = [layer for (layer, full) in enumerate(is_full) if not full]
    import torch
    import rpu_backend as rb
    from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache
    from rpu_backend.adapters.qwen3_5.text import _build_partial_mrope_cos_sin
    from rpu_backend.adapters.qwen3_5.weights import pad_gdn_scalar
    from rpu_backend.graph import GraphCache, GraphSignature, GraphRuntimePolicy
    from rpu_backend.runtime.execution_planner import _decode_native_stage_domain
    from rpu_backend.runtime.weights import transform_linear_weight
    from rpu_backend.tests._hw_closure_evidence import complete_descriptor_evidence
    torch.manual_seed(42)
    (h, inter, nq) = (2048, 6144, 8) if args.profile == '2b' else (2560, 9216, 16)
    (nkv, hd) = (8, 256)
    (nvh, nkh, dk, dv, convk) = (16 if args.profile == '2b' else 32, 16, 128, 128, 4)
    (convdim, length) = (2 * nkh * dk + nvh * dv, 256)
    empty = torch.empty(0)
    policy = replace(GraphRuntimePolicy.from_environment(), fmb_fast_replay=True, fmb_deep_fast_replay=True)
    report = {'backend': str(Path(rb.__file__).resolve()), 'profile': args.profile, 'cases': [], 'synthetic_full_depth': args.full_depth, 'shared_immutable_weights': True, 'geometry': {'hidden': h, 'intermediate': inter, 'heads': nq, 'kv_heads_effective': nkv, 'head_dim': hd, 'gdn_value_heads': nvh, 'gdn_key_heads': nkh, 'layer_is_full': is_full, 'num_layers': nl, 'length': length, 'cores': 8}, 'boundary': 'synthetic FP16 native owner; reset before each forward; no model-quality claim'}
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
    (q, k, v) = (only_full(linear(n, h, 1)) for n in (nq * hd, nkv * hd, nkv * hd))
    o = only_full(linear(h, nq * hd, 0))
    ag = only_full(linear(nq * hd, h, 1))
    (qn, kn) = (only_full(ones(hd)), only_full(ones(hd)))
    (inrm, postrm) = ([ones(h)] * nl, [ones(h)] * nl)
    (gate, up, down) = ([linear(inter, h, 1)] * nl, [linear(inter, h, 1)] * nl, [linear(h, inter, 0)] * nl)
    (gz, gout) = (only_gdn(linear(nvh * dv, h, 1)), only_gdn(linear(h, nvh * dv, 0)))
    alog = pad_gdn_scalar(torch.zeros(nvh).half(), num_heads=nvh, num_cores=8).to('rpu')
    dt = pad_gdn_scalar(torch.full((nvh,), -2.0).half(), num_heads=nvh, num_cores=8).to('rpu')
    gn = only_gdn(ones(dv))
    (gq, gk, gv) = (only_gdn(linear(n, h, 1)) for n in (nkh * dk, nkh * dk, nvh * dv))
    (gcq, gck, gcv) = (only_gdn(conv(n)) for n in (nkh * dk, nkh * dk, nvh * dv))
    (gb, ga) = (only_gdn(ba()), only_gdn(ba()))
    (cos, sin) = _build_partial_mrope_cos_sin(hd, 64, 10000000.0, 512)
    final = ones(h)
    handle = int(torch.ops.rpu.qwen3_5_create())
    graphs = []
    try:
        torch.ops.rpu.qwen3_5_set_weights(handle, q, k, v, o, qn, kn, ag, inrm, postrm, gate, up, down, cos, sin, final, is_full, nq, nkv, hd, h, inter, 1e-06, True, [11, 11, 10], gz, gout, only_gdn(alog), only_gdn(dt), gn, nvh, dk, dv, convdim, convk, gq, gk, gv, gcq, gck, gcv, gb, ga, *[[empty] * nl] * 15)
        torch.ops.rpu.qwen3_5_set_prefill_rope(handle, cos[:length].contiguous(), sin[:length].contiguous())
        torch.ops.rpu.qwen3_5_set_chunk_envelope(handle, 512, 256)
        torch.ops.rpu.qwen3_5_set_valid_prefill_len(handle, length)
        torch.ops.rpu.qwen3_5_set_retained_prefill_graph(handle, True)
        torch.ops.rpu.qwen3_5_set_fast_replay(handle, True)
        cache = Qwen3_5Cache(nl, 1, 512, nkv, hd, ['full_attention' if f else 'linear_attention' for f in is_full], nvh, dk, dv, convdim, convk, attn_tp=8)
        inputs = [(torch.randn(1, length, h) * 0.2).half().to('rpu') for _ in range(3)]
        for chunk in (128, 256):
            wire = torch.ops.rpu.qwen3_5_resolve_prefill_stage_domain(handle, length, length, chunk)
            domain = _decode_native_stage_domain(wire, execution_len=length, logical_len=length, position=0, graph_mode='RETAINED_CACHE')
            pairs = []
            for row in domain:
                evidence = complete_descriptor_evidence(row.physical_descriptor, expected_lifecycle='RETAINED_CACHE', component_path='synthetic_qwen35', stage='prefill')
                routes = evidence['routes']
                if row.compute_chunk == chunk and all((r['selector'] == 1 for r in routes if r['family'] == 1)):
                    pairs.append((row, evidence, routes))
            assert len(pairs) == 1, 'one explicit DDR-attention candidate required'
            (row, evidence, routes) = pairs[0]
            masks = [r for r in routes if r['site_id'] == 120356759196541611]
            mask_arguments = [64, 64, 2, gdn_layers[0], len(gdn_layers), chunk, length // chunk, 8, 8]
            if args.expect_schedule == 'retained':
                assert len(masks) == 1 and masks[0]['selector'] == 2
            else:
                assert not masks or masks[0]['selector'] == 1
            if masks:
                assert len(masks) == 1 and masks[0]['arguments'] == mask_arguments
            graph = GraphCache(runtime_policy=policy)
            graphs.append(graph)
            sig = GraphSignature(op_id='qwen35_gdn_mask', shapes=[1, length, h], dyn_dims=list(row.physical_descriptor), dtypes=[torch.float16])
            measured = []
            seen = {}
            for iteration in range(7):
                index = iteration % len(inputs)
                cache.reset()
                cache.conv_states[gdn_layers[-1]].cpu()
                start = time.perf_counter_ns()
                with graph.capture(sig):
                    result = torch.ops.rpu.qwen3_5_forward(handle, inputs[index], cache.k_caches, cache.v_caches, cache.gdn_states, cache.conv_states, None, 0, True, planned_chunk_size=0, planned_stage_descriptor=row.physical_descriptor)
                elapsed = (time.perf_counter_ns() - start) / 1000000.0
                output = result.cpu().clone()
                assert output.shape == (1, length, h) and torch.isfinite(output).all()
                assert output.count_nonzero() > 0
                if index in seen:
                    assert torch.equal(output, seen[index]), 'reset/replay output changed'
                seen[index] = output
                key = f'c{chunk}_input{index}'
                outputs[key] = output
                if references is not None:
                    assert torch.equal(output, references[key]), (key, float((output - references[key]).abs().max()))
                for (kind, tensors) in (('recurrent', cache.gdn_states), ('conv', cache.conv_states)):
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
        torch.ops.rpu.qwen3_5_destroy(handle)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    args.tensors.parent.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, args.tensors)
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
