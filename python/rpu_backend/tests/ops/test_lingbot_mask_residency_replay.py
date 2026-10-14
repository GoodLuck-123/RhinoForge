from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path

def main():
    import torch
    import rpu_backend as rb
    from rpu_backend.api import RPUCache
    from rpu_backend.graph import GraphCache, GraphSignature, GraphRuntimePolicy
    from rpu_backend.runtime.decoder import plan_bounded_prefill_execution
    from rpu_backend.runtime.masks import versioned_cpu_mask
    from rpu_backend.runtime.weights import transform_linear_weight
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resident', type=int, choices=(0, 1), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control', type=Path, help='--resident 0 result .pt')
    parser.add_argument('--unroll', action='store_true', help='two-body LingbotDenoise; old SO supplies resident=0 control')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    (h, ff, nq, nkv, hd, layers, q, prefix, horizon) = (768, 2816, 16, 2, 128, 2, 51, 75, 256)
    policy = GraphRuntimePolicy.from_environment()
    report = dict(passed=False, resident=bool(args.resident), backend=str(Path(rb.__file__).resolve()), geometry=dict(h=h, ff=ff, nq=nq, nkv=nkv, hd=hd, layers=layers, q=q, prefix=prefix, horizon=horizon), effective_policy=asdict(policy))
    report['unroll'] = args.unroll
    graphs = []
    handle = None
    try:
        with torch.inference_mode():
            weights = {name: [] for name in ('q', 'k', 'v', 'o', 'g', 'u', 'd', 'in', 'post', 'qb', 'kb', 'vb')}
            for _ in range(layers):
                for (name, rows, cols, part, cores) in (('q', nq * hd, h, 1, 2), ('k', nkv * hd, h, 1, 2), ('v', nkv * hd, h, 1, 2), ('o', h, nq * hd, 0, 2), ('g', ff, h, 1, 8), ('u', ff, h, 1, 8), ('d', h, ff, 0, 8)):
                    host = (torch.randn(rows, cols) * (0.3 / cols ** 0.5)).half()
                    weights[name].append(transform_linear_weight(host, part, num_cores=cores).to('rpu'))
                for (name, width) in (('in', h), ('post', h)):
                    weights[name].append(torch.ones(width, dtype=torch.float16, device='rpu'))
                for (name, width) in (('qb', nq * hd), ('kb', nkv * hd), ('vb', nkv * hd)):
                    weights[name].append((torch.randn(width) * 0.01).half().to('rpu'))
            theta = torch.arange(horizon).double()[:, None] / 10000 ** (torch.arange(hd // 2).double() * 2 / hd)
            (cos, sin) = (theta.cos().half().to('rpu'), theta.sin().half().to('rpu'))
            final = torch.ones(h, dtype=torch.float16, device='rpu')
            if args.unroll:
                (steps, action_dim, padded_dim) = (2, 75, 80)

                def action_weight(rows, cols, zero_tail=False):
                    host = (torch.randn(rows, cols) * (0.3 / cols ** 0.5)).half()
                    if zero_tail:
                        host[action_dim:] = 0
                    return transform_linear_weight(host, 1, num_cores=1).to('rpu')
                wc = action_weight(h, padded_dim)
                mo = action_weight(h, h)
                mo_bias = (torch.randn(h) * 0.01).half().to('rpu')
                op = action_weight(padded_dim, h, zero_tail=True)
                op_bias_host = (torch.randn(padded_dim) * 0.01).half()
                op_bias_host[action_dim:] = 0
                op_bias = op_bias_host.to('rpu')
                times = (torch.randn(steps, h) * 0.1).half().to('rpu')
                in_s = (1 + torch.randn(steps, layers, h) * 0.01).half().to('rpu')
                in_shift = (torch.randn(steps, layers, h) * 0.01).half().to('rpu')
                post_s = (1 + torch.randn(steps, layers, h) * 0.01).half().to('rpu')
                post_shift = (torch.randn(steps, layers, h) * 0.01).half().to('rpu')
                handle = int(torch.ops.rpu.lingbot_denoise_create())
                torch.ops.rpu.lingbot_denoise_set_weights(handle, *(weights[name] for name in ('q', 'k', 'v', 'o', 'in', 'post', 'g', 'u', 'd')), cos, sin, final, nq, nkv, hd, h, ff, 1e-06, True, weights['qb'], weights['kb'], weights['vb'], wc, mo, mo_bias, op, op_bias, times, in_s, in_shift, post_s, post_shift, action_dim, padded_dim, steps, q)
                torch.ops.rpu.lingbot_denoise_set_chunk_size(handle, 64)
                x_host = (torch.randn(1, q, padded_dim) * 0.2).half()
                x_host[:, 0] = 0
                x_host[:, :, action_dim:] = 0
                x0 = x_host.to('rpu')
                trajectory = torch.empty(steps, q, padded_dim, dtype=torch.float16, device='rpu')
                state = (torch.randn(h) * 0.2).half().to('rpu')
                resolve = lambda length: torch.ops.rpu.lingbot_denoise_resolve_action_stage_domain(handle, length, prefix, prefix + length)
            else:
                steps = 1
                handle = int(torch.ops.rpu.causal_decoder_create(False, False))
                torch.ops.rpu.causal_decoder_set_explicit_mask_spm_residency(handle, bool(args.resident))
                torch.ops.rpu.causal_decoder_set_weights(handle, *(weights[name] for name in ('q', 'k', 'v', 'o')), [], [], weights['in'], weights['post'], *(weights[name] for name in ('g', 'u', 'd')), cos, sin, final, nq, nkv, hd, h, ff, 1e-06, True, [], [], weights['qb'], weights['kb'], weights['vb'])
                torch.ops.rpu.causal_decoder_set_chunk_envelope(handle, horizon, 64)
                torch.ops.rpu.causal_decoder_set_chunk_size_override(handle, 64)
                scales = torch.ones(layers, h, dtype=torch.float16, device='rpu')
                shifts = torch.zeros(layers, h, dtype=torch.float16, device='rpu')
                torch.ops.rpu.causal_decoder_set_adarms_step_mutable(handle, scales, shifts, scales, shifts)
                resolve = lambda length: torch.ops.rpu.causal_decoder_resolve_prefill_stage_domain(handle, length, prefix, False, prefix + length, 0, 4, q)
            cache = RPUCache(layers, 1, horizon, nkv, hd, device='rpu', attn_tp=2)
            for item in (*cache.k_caches, *cache.v_caches):
                item.copy_((torch.randn(tuple(item.shape)) * 0.7).half().to('rpu'))
            hidden = (torch.randn(1, q, h) * 0.2).half().to('rpu')
            box = []
            (execution, chunk) = plan_bounded_prefill_execution(q, q, 0, resolve_stage_domain=resolve, position=prefix, alignment=1, padding_rows=0, exact_chunk_size=64, graph_mode='COMPOSITE_CHILD', queue_owner_id=handle, lease_owner_id=handle, plan_result_sink=box.append)
            assert execution == q and chunk == 64
            selected = box[0].selected.stage_tuple
            desc = tuple(selected.physical_descriptor)
            assert desc and dict(selected.physical_metadata)['manifest_state'] == 1
            site = 301270362681343383
            if args.resident:
                assert desc.count(site) == 1
                offset = desc.index(site)
                assert desc[offset + 1:offset + 4] == (2, 0, 8)
                assert desc[0] == 3 and desc[offset + 4] == 0
                schedule = desc[offset + 5:offset + 13]
                assert schedule[:6] == (q, prefix + q, 2, 8, steps, layers)
                assert schedule[6] == schedule[7]
                report['mask_schedule'] = list(schedule)
            else:
                assert site not in desc
            report['physical_metadata'] = dict(selected.physical_metadata)
            sig = GraphSignature(op_id='lingbot_v1_mask_residency', shapes=[q, h], dyn_dims=[prefix, steps, *box[0].graph_key_words()], dtypes=[torch.float16])

            def mask(begin, end):
                host = torch.full((q, prefix + q), -50000.0, dtype=torch.float16)
                host[:, begin:end] = 0
                return versioned_cpu_mask(host)
            templates = [mask(0, 37), mask(37, 75), mask(20, 55), mask(0, 37)]
            assert all((not tensor.is_inference() for tensor in templates))

            def forward(graph, current):
                cache.reset_to_position(prefix)
                with graph.capture(sig):
                    if args.unroll:
                        torch.ops.rpu.lingbot_denoise_unroll_forward(handle, x0, cache.k_caches, cache.v_caches, state, current, trajectory, -1.0 / steps, prefix, steps, desc)
                        out = trajectory
                    else:
                        out = torch.ops.rpu.causal_decoder_forward(handle, hidden, cache.k_caches, cache.v_caches, current, prefix, False, None, [], None, None, -1, 0, False, 0, desc)
                value = out.cpu().clone()
                assert torch.isfinite(value).all()
                return value
            references = []
            for current in templates:
                fresh = GraphCache(runtime_policy=policy)
                graphs.append(fresh)
                references.append(forward(fresh, current))
                fresh.clear()
            assert float((references[0] - references[1]).abs().max()) > 0.0001
            assert float((references[1] - references[2]).abs().max()) > 0.0001
            (live_a, live_b) = (versioned_cpu_mask(templates[0]), versioned_cpu_mask(templates[1]))
            replay = GraphCache(runtime_policy=policy)
            graphs.append(replay)
            (outputs, versions) = ([], [])
            for index in range(4):
                current = live_a if index in (0, 3) else live_b
                if index == 2:
                    before = live_b._version
                    live_b.copy_(templates[2])
                    assert live_b._version > before
                actual = forward(replay, current)
                assert torch.equal(actual, references[index]), (index, float((actual - references[index]).abs().max()))
                outputs.append(actual)
                versions.append(current._version)
            (entry,) = replay.snapshot()
            assert replay.size() == 1 and replay.cache_invariant_ok()
            assert entry.replay_count == 3 and entry.recapture_count == 0 and (entry.kernel_count > 0)
            payload = {'outputs': outputs, 'k': [item.cpu().clone() for item in cache.k_caches], 'v': [item.cpu().clone() for item in cache.v_caches]}
            if args.control:
                control = torch.load(args.control, map_location='cpu', weights_only=True)
                for name in ('outputs', 'k', 'v'):
                    assert len(payload[name]) == len(control[name])
                    for (actual, expected) in zip(payload[name], control[name], strict=True):
                        assert torch.equal(actual, expected), (name, float((actual - expected).abs().max()))
                report['byte_equal_nonresident_control'] = True
            torch.save(payload, args.output.with_suffix('.pt'))
            report.update(passed=True, byte_equal_fresh_graph=True, outer_inference_mode=True, versions=versions, mask_change_max_abs=float((references[0] - references[1]).abs().max()), graph={name: getattr(entry, name) for name in ('replay_count', 'recapture_count', 'kernel_count', 'dma_count')})
    finally:
        for graph in graphs:
            graph.clear()
        if handle is not None:
            if args.unroll:
                torch.ops.rpu.lingbot_denoise_destroy(handle)
            else:
                torch.ops.rpu.causal_decoder_destroy(handle)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
