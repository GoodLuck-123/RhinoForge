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
    from rpu_backend.runtime.weights import transform_linear_weight

    def fixture_mask(value):
        with torch.inference_mode(False), torch.no_grad():
            return value.detach().to(device='cpu', dtype=torch.float16, copy=True).contiguous()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control', type=Path)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--unroll', action='store_true')
    parser.add_argument('--direct-action-input', action='store_true', help='exercise v3 96D IO without the legacy action/time MLP')
    parser.add_argument('--full-v3', action='store_true', help='use all 18 expert layers and ten steps with v3 direct IO')
    parser.add_argument('--packed-qkv', action='store_true', help='test the packed projection and per-core output copies; requires --full-v3')
    parser.add_argument('--aligned-kv', action='store_true', help='test aligned 32-row KV writes and zero tail; requires --full-v3')
    parser.add_argument('--w8a16', action='store_true', help='quantize the seven expert projections; requires --full-v3')
    parser.add_argument('--full-w8a16', action='store_true', help='also quantize action IO and cold device AdaRMS projections; requires --full-v3')
    parser.add_argument('--precompute-gate-tanh', action='store_true', help='precompute fixed AdaRMS gate TANH on device; requires --full-w8a16')
    parser.add_argument('--high-precision', action='store_true', help='use high-precision nonlinearities/Newton RMSNorm; requires --full-w8a16')
    parser.add_argument('--high-precision-fusions', action='store_true', help='fuse HIGH Newton+shift and SiLU+mul; control must be the split-HIGH run')
    parser.add_argument('--vector-k-norm', action='store_true', help='use zero-padded 32-row Newton K norm; requires HIGH and aligned KV')
    parser.add_argument('--vector-q-norm', action='store_true', help='use zero-padded 64-row Newton Q norm; requires HIGH')
    parser.add_argument('--partial-rope', action='store_true', help='use owned partial_mrope with full-head rotation; requires HIGH')
    args = parser.parse_args()
    if args.high_precision_fusions and (not args.high_precision):
        parser.error('--high-precision-fusions requires --high-precision')
    if args.vector_k_norm and (not (args.high_precision and args.aligned_kv)):
        parser.error('--vector-k-norm requires --high-precision and --aligned-kv')
    if args.vector_q_norm and (not args.high_precision):
        parser.error('--vector-q-norm requires --high-precision')
    if args.partial_rope and (not args.high_precision):
        parser.error('--partial-rope requires --high-precision')
    if args.high_precision and (not args.full_w8a16 or (args.control and (not args.high_precision_fusions))):
        parser.error('--high-precision requires full W8 and its own fresh/replay oracle')
    if args.precompute_gate_tanh and (not args.full_w8a16):
        parser.error('--precompute-gate-tanh requires --full-w8a16')
    if args.full_w8a16:
        args.w8a16 = True
    if args.packed_qkv and args.w8a16:
        parser.error('W8A16 is incompatible with --packed-qkv')
    if args.full_v3:
        args.unroll = args.direct_action_input = True
        torch.set_num_threads(8)
    if (args.packed_qkv or args.aligned_kv or args.w8a16) and (not args.full_v3):
        parser.error('--packed-qkv/--aligned-kv/--w8a16 requires --full-v3')
    if args.direct_action_input and (not args.unroll):
        parser.error('--direct-action-input requires --unroll')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    (h, ff, nq, nkv, hd, layers, q) = (1024, 3072, 16, 8, 128, 2, 31)
    (physical_prefix, logical_prefix, horizon) = (48, 45, 256)
    steps = 2 if args.unroll else 1
    if args.full_v3:
        (layers, steps) = (18, 10)
    policy = GraphRuntimePolicy.from_environment()
    report = dict(passed=False, baseline=args.baseline, unroll=args.unroll, direct_action_input=args.direct_action_input, full_v3=args.full_v3, packed_qkv=args.packed_qkv, aligned_kv=args.aligned_kv, expert_w8a16=args.w8a16, full_w8a16=args.full_w8a16, precompute_gate_tanh=args.precompute_gate_tanh, high_precision=args.high_precision, high_precision_fusions=args.high_precision_fusions, vector_k_norm=args.vector_k_norm, vector_q_norm=args.vector_q_norm, partial_rope=args.partial_rope, backend=str(Path(rb.__file__).resolve()), effective_policy=asdict(policy), geometry=dict(h=h, ff=ff, nq=nq, nkv=nkv, hd=hd, layers=layers, q=q, physical_prefix=physical_prefix, logical_rope_start=logical_prefix, bodies=steps))
    (graphs, handle) = ([], None)
    try:
        with torch.inference_mode():

            def weight(rows, cols, part, cores=8, padded_rows=None, padded_cols=None, projection=None):
                host = (torch.randn(rows, cols) * (0.3 / cols ** 0.5)).half()
                if padded_rows is not None:
                    host[padded_rows:] = 0
                if padded_cols is not None:
                    host[:, padded_cols:] = 0
                if args.w8a16 and projection in scales:
                    from rpu_backend.quant._common import quantize_linear_per_channel
                    (host, scale) = quantize_linear_per_channel(host)
                    scales[projection].append(scale.to('rpu'))
                    if args.full_w8a16 and projection in ('pw', 'final'):
                        cold_columns[projection].append(transform_linear_weight(host, 1, num_cores=8).to('rpu'))
                return transform_linear_weight(host, part, num_cores=cores).to('rpu')

            def bias(width, gate=False, pad=None):
                value = (torch.randn(width) * 0.01).half()
                if gate:
                    value[2 * h:3 * h] += 0.5
                if pad is not None:
                    value[pad:] = 0
                return value.to('rpu')
            names = ('q', 'k', 'v', 'o', 'g', 'u', 'd', 'aw', 'ab', 'mw', 'mb', 'pw', 'pb', 'qn', 'kn')
            w = {name: [] for name in names}
            scales = {name: [] for name in names[:7]}
            if args.full_w8a16:
                scales.update({name: [] for name in ('aw', 'mw', 'pw', 'action', 'state', 'state_mask', 'action_mask', 'final', 'out')})
            cold_columns = {'pw': [], 'final': []}
            for _ in range(layers):
                for (name, rows, cols, part) in (('q', nq * hd, h, 1), ('k', nkv * hd, h, 1), ('v', nkv * hd, h, 1), ('o', h, nq * hd, 0), ('g', ff, h, 1), ('u', ff, h, 1), ('d', h, ff, 0), ('aw', 3 * h, h, 0), ('mw', 3 * h, h, 0), ('pw', 6 * h, h, 0)):
                    w[name].append(weight(rows, cols, part, projection=name))
                w['ab'].append(bias(3 * h, gate=True))
                w['mb'].append(bias(3 * h, gate=True))
                w['pb'].append(bias(6 * h))
                if args.full_w8a16:
                    pair_bias = w['pb'][-1].cpu()
                    pair_bias[2 * h:3 * h] += 0.5
                    pair_bias[5 * h:6 * h] += 0.5
                    w['pb'][-1] = pair_bias.to('rpu')
                for name in ('qn', 'kn'):
                    w[name].append(torch.ones(hd, dtype=torch.float16, device='rpu'))
            theta = torch.arange(horizon).double()[:, None] / 5000000 ** (torch.arange(hd // 2).double() * 2 / hd)
            (cos, sin) = (theta.cos().half().to('rpu'), theta.sin().half().to('rpu'))
            handle = int(torch.ops.rpu.rhino_vla_create())
            if args.full_w8a16:
                torch.ops.rpu.rhino_vla_set_runtime_config(handle, False, True, False, True, False, False, True, True, True, True, False, args.aligned_kv)
            else:
                extra = (False, False, args.packed_qkv, args.aligned_kv) if args.packed_qkv or args.aligned_kv else ()
                torch.ops.rpu.rhino_vla_set_runtime_config(handle, False, False, False, False, False, False, False, False, *extra)
            weight_args = (*(w[name] for name in names), nq, nkv, hd, h, ff, 1e-06)
            if args.w8a16:
                scale_args = tuple((scales[name] for name in names[:7]))
                if args.full_w8a16:
                    scale_args += tuple((scales[name] for name in ('aw', 'mw', 'pw')))
                setter = torch.ops.rpu.rhino_vla_set_weights_full_w8a16 if args.full_w8a16 else torch.ops.rpu.rhino_vla_set_weights_w8a16
                setter(handle, *weight_args, *scale_args)
                rejected = []
                for fault in ('missing', 'shape', 'dtype', 'nan', 'zero'):
                    bad = [list(items) for items in scale_args]
                    if fault == 'missing':
                        bad[-1] = []
                    else:
                        scale = scale_args[-1][-1].cpu().clone()
                        if fault == 'shape':
                            scale = scale[:-1]
                        elif fault == 'dtype':
                            scale = scale.float()
                        elif fault == 'nan':
                            scale[-1] = float('nan')
                        else:
                            scale[-1] = 0
                        bad[-1][-1] = scale.to('rpu')
                    try:
                        setter(handle, *weight_args, *bad)
                    except RuntimeError as exc:
                        assert 'scale' in str(exc).lower()
                        rejected.append(fault)
                    else:
                        raise AssertionError('native W8 admission accepted ' + fault)
                report['rejected_invalid_scales'] = rejected
            else:
                torch.ops.rpu.rhino_vla_set_weights(handle, *weight_args)
            if args.high_precision:
                torch.ops.rpu.rhino_vla_set_high_precision(handle, True)
            if args.high_precision_fusions:
                torch.ops.rpu.rhino_vla_set_high_precision_fusions(handle, True)
            if args.vector_k_norm:
                torch.ops.rpu.rhino_vla_set_vector_k_norm(handle, True)
            if args.vector_q_norm:
                torch.ops.rpu.rhino_vla_set_vector_q_norm(handle, True)
            if args.partial_rope:
                torch.ops.rpu.rhino_vla_set_partial_rope(handle, True)
            torch.ops.rpu.rhino_vla_set_chunk_envelope(handle, horizon, 32)
            torch.ops.rpu.rhino_vla_set_chunk_size(handle, 32)
            torch.ops.rpu.rhino_vla_set_fast_replay(handle, True)
            torch.ops.rpu.rhino_vla_set_rope_position(handle, logical_prefix)
            if args.unroll:
                (dim, padded, ah) = (72, 80, 30)
                if args.direct_action_input:
                    dim = padded = 96
                loop_weights = (weight(h, padded, 1, 1, padded_cols=dim, projection='action'), bias(h), weight(h, padded, 1, 1, padded_cols=dim), bias(h), weight(h, h, 1, 1), weight(h, h, 1, 1), bias(h), weight(h, h, 1, 1), bias(h), weight(h, padded, 1, 1, padded_cols=dim, projection='state'), bias(h), weight(h, padded, 1, 1, padded_cols=dim, projection='state_mask'), bias(h), weight(h, padded, 1, 1, padded_cols=dim, projection='action_mask'), bias(h), weight(3 * h, h, 0, projection='final'), bias(3 * h, gate=True), weight(padded, h, 1, 1, padded_rows=dim, projection='out'), bias(padded, pad=dim))
                variant = ()
                if args.direct_action_input:
                    empty = torch.empty(0, dtype=torch.float16, device='rpu')
                    loop_weights = (*loop_weights[:2], *(empty,) * 7, *loop_weights[9:])
                    variant = (True,)
                loop_args = (handle, *loop_weights, cos, sin, dim, padded, dim, padded, ah, q, *variant)
                if args.full_w8a16:
                    io_scales = [scales[name][0] for name in ('action', 'state', 'state_mask', 'action_mask', 'final', 'out')]
                    io_setter = torch.ops.rpu.rhino_vla_set_denoise_loop_weights_w8a16
                    io_setter(*loop_args, io_scales)
                    rejected = []
                    for fault in ('missing', 'shape', 'dtype', 'nan', 'zero'):
                        bad = list(io_scales)
                        if fault == 'missing':
                            bad = bad[:-1]
                        else:
                            scale = bad[-1].cpu().clone()
                            if fault == 'shape':
                                scale = scale[:-1]
                            elif fault == 'dtype':
                                scale = scale.float()
                            elif fault == 'nan':
                                scale[-1] = float('nan')
                            else:
                                scale[-1] = 0
                            bad[-1] = scale.to('rpu')
                        try:
                            io_setter(*loop_args, bad)
                        except RuntimeError as exc:
                            assert 'scale' in str(exc).lower()
                            rejected.append(fault)
                        else:
                            raise AssertionError('native full W8 IO accepted ' + fault)
                    report['rejected_invalid_io_scales'] = rejected
                else:
                    torch.ops.rpu.rhino_vla_set_denoise_loop_weights(*loop_args)
                torch.ops.rpu.rhino_vla_prepare_denoise_loop_schedule(handle, steps, -1.0 / steps)
                xhost = (torch.randn(1, ah, padded) * 0.2).half()
                xhost[:, :, dim:] = 0
                x0 = xhost.to('rpu')
                xout = torch.empty_like(x0)
                statehost = (torch.randn(1, padded) * 0.2).half()
                statehost[:, dim:] = 0
                state = statehost.to('rpu')
                sm = torch.ones_like(state)
                sm[:, dim:] = 0
                am = torch.ones_like(x0)
                am[:, :, dim:] = 0
                cond_all = (torch.randn(steps, h) * 0.2).half().to('rpu')
                if args.full_w8a16:
                    from types import SimpleNamespace
                    from rpu_backend.adapters.rhinovla.convert import _prepare_denoise_adarms_tables
                    owner = SimpleNamespace(_rpu_full_w8_cond_owners={'cold_weights': cold_columns['pw'], 'cold_scales': scales['pw'], 'cold_biases': w['pb']})
                    loop_meta = {'final_norm_w_col': cold_columns['final'][0], 'io_scales': io_scales, 'final_norm_b': loop_weights[16]}
                    (pair, final, cold_w, cold_s, cold_b) = _prepare_denoise_adarms_tables(owner, cond_all.cpu(), full_w8a16=True, loop_weights=loop_meta, precompute_gate_tanh=args.precompute_gate_tanh, high_precision=args.high_precision)
                    table_setter = torch.ops.rpu.rhino_vla_set_denoise_loop_adarms_tables_w8a16
                    table_setter(handle, pair, final, cold_w, cold_s, cold_b, args.precompute_gate_tanh, args.high_precision)
                    rejected = []
                    for fault in ('missing', 'shape', 'dtype', 'nan', 'zero'):
                        bad = list(cold_s)
                        if fault == 'missing':
                            bad = bad[:-1]
                        else:
                            scale = bad[-1].cpu().clone()
                            if fault == 'shape':
                                scale = scale[:-1]
                            elif fault == 'dtype':
                                scale = scale.float()
                            elif fault == 'nan':
                                scale[-1] = float('nan')
                            else:
                                scale[-1] = 0
                            bad[-1] = scale.to('rpu')
                        try:
                            table_setter(handle, pair, final, cold_w, bad, cold_b, False, args.high_precision)
                        except RuntimeError as exc:
                            assert 'scale' in str(exc).lower() or 'owner' in str(exc).lower()
                            rejected.append(fault)
                        else:
                            raise AssertionError('native full W8 cold tables accepted ' + fault)
                    report['rejected_invalid_cold_scales'] = rejected
            cache = RPUCache(layers, 1, horizon, nkv, hd, device='rpu', attn_tp=8)

            def tail_row(item, kind):
                row = physical_prefix + q
                return item[:, row // 16, :, :, :, row % 16, :] if kind == 'k' else item[:, row // 16, :, :, :, :, row % 16]
            for (kind, items) in (('k', cache.k_caches), ('v', cache.v_caches)):
                for item in items:
                    host = (torch.randn(tuple(item.shape)) * 0.7).half()
                    if args.aligned_kv:
                        tail_row(host, kind).fill_(float('nan'))
                    item.copy_(host.to('rpu'))
            hidden = (torch.randn(1, q, h) * 0.2).half().to('rpu')
            cond = (torch.randn(h) * 0.2).half().to('rpu')
            aligned_k = (physical_prefix + q + 15) // 16 * 16
            box = []
            (execution, chunk) = plan_bounded_prefill_execution(q, q, 0, position=physical_prefix, resolve_stage_domain=lambda length: torch.ops.rpu.rhino_vla_resolve_action_stage_domain(handle, length, length, physical_prefix, aligned_k, True, False, 32), alignment=1, padding_rows=0, exact_chunk_size=32, graph_mode='COMPOSITE_CHILD', queue_owner_id=handle, lease_owner_id=handle, plan_result_sink=box.append)
            assert (execution, chunk) == (q, 32)
            selected = box[0].selected.stage_tuple
            desc = tuple(selected.physical_descriptor)
            assert desc and dict(selected.physical_metadata)['manifest_state'] == 1
            site = 2981318290883577335
            if args.baseline:
                assert site not in desc
            else:
                assert desc.count(site) == 1 and desc[0] == 3
                i = desc.index(site)
                assert desc[i + 1:i + 5] == (2, 0, 9, 0)
                schedule = desc[i + 5:i + 14]
                assert schedule[:6] == (q, aligned_k, 8, 8, steps, layers)
                assert schedule[6] == schedule[7] and schedule[8] == physical_prefix + q
                report['mask_schedule'] = list(schedule)
            report['physical_metadata'] = dict(selected.physical_metadata)
            if args.full_w8a16:
                site = 7661653982490056347
                assert desc.count(site) == 1
                offset = desc.index(site)
                assert desc[offset + 1:offset + 5] == (1, int(args.precompute_gate_tanh), 6, 0)
                assert desc[offset + 5:offset + 11] == (18, 10, 37, 5, 8, 16)
                report['full_w8_action_policy'] = list(desc[offset:offset + 11])
                for (site, dimensions) in ((8994515274536934858, (30, 1024, 96, 1, 1)), (4957523574588877102, (30, 1024, 96, 1, 1)), (1340638388713089312, (1, 1024, 96, 1, 1)), (4225731628177473256, (1, 1024, 96, 1, 1)), (1703582534478793943, (30, 96, 1024, 1, 1))):
                    offset = desc.index(site)
                    assert desc[offset + 2:offset + 5] == (1, 5, 0)
                    assert desc[offset + 5:offset + 10] == dimensions
            if args.high_precision:
                offset = desc.index(1749523321564235290)
                expected_flags = int(args.precompute_gate_tanh) | (2 if args.high_precision_fusions else 0) | (4 if args.vector_k_norm else 0) | (8 if args.vector_q_norm else 0) | (16 if args.partial_rope else 0)
                assert desc[offset + 1:offset + 5] == (1, expected_flags, 10, 0)
                assert desc[offset + 5:offset + 14] == (1, 31, 1024, 3072, 18, 10, 16, 8, 128)
                report['high_precision_action_policy'] = list(desc[offset:offset + 15])
                report['high_precision_flags'] = expected_flags
            if not args.baseline:
                rope_policies = {}
                for (name, site) in (('q', 3231791188675653111), ('k', 8158949646342380956)):
                    assert desc.count(site) == 1
                    offset = desc.index(site)
                    assert desc[offset + 1:offset + 6] == (2 if args.partial_rope else 1, 0, 1, 0, logical_prefix)
                    rope_policies[name] = list(desc[offset:offset + 6])
                report['rope_policies'] = rope_policies
            assert report['physical_metadata']['kv_insert_physical_rows'] == (32 if args.aligned_kv else 31)
            sig = GraphSignature(op_id='rhino_mask_residency', shapes=[q, h], dyn_dims=[physical_prefix, logical_prefix, steps, *box[0].graph_key_words()], dtypes=[torch.float16])

            def mask(begin, end):
                host = torch.full((1, 1, q, aligned_k), torch.finfo(torch.float16).min, dtype=torch.float16)
                host[..., begin:end] = 0
                host[..., physical_prefix:physical_prefix + q] = 0
                return fixture_mask(host)
            masks = [mask(0, 45), mask(0, 20), mask(20, 45), mask(0, 45)]
            assert all((not item.is_inference() for item in masks))

            def forward(graph, current):
                cache.reset_to_position(physical_prefix)
                with graph.capture(sig):
                    if args.unroll:
                        out = torch.ops.rpu.rhino_vla_denoise_loop_forward(handle, x0, cache.k_caches, cache.v_caches, cond_all, current, state, sm, am, xout, -1.0 / steps, physical_prefix, steps, desc)
                    else:
                        out = torch.ops.rpu.rhino_vla_forward(handle, hidden, cond, cache.k_caches, cache.v_caches, cos, sin, current, physical_prefix, False, desc)
                value = out.cpu().clone()
                assert torch.isfinite(value).all()
                return value
            references = []
            for current in masks:
                fresh = GraphCache(runtime_policy=policy)
                graphs.append(fresh)
                references.append(forward(fresh, current))
                fresh.clear()
            assert float((references[0] - references[1]).abs().max()) > 0.0001
            assert float((references[1] - references[2]).abs().max()) > 0.0001
            replay = GraphCache(runtime_policy=policy)
            graphs.append(replay)
            (live_a, live_b) = (fixture_mask(masks[0]), fixture_mask(masks[1]))
            (actuals, versions) = ([], [])
            for index in range(4):
                current = live_a if args.baseline or index in (0, 3) else live_b
                if index == 2 and (not args.baseline):
                    before = live_b._version
                    live_b.copy_(masks[2])
                    assert live_b._version > before
                actual = forward(replay, current)
                expected = references[0 if args.baseline else index]
                assert torch.equal(actual, expected), (index, float((actual - expected).abs().max()))
                actuals.append(actual)
                versions.append(current._version)
            (entry,) = replay.snapshot()
            assert replay.size() == 1 and replay.cache_invariant_ok()
            assert entry.replay_count == 3 and entry.recapture_count == 0 and (entry.kernel_count > 0)
            payload = {'outputs': references if args.baseline else actuals, 'k': [item.cpu().clone() for item in cache.k_caches], 'v': [item.cpu().clone() for item in cache.v_caches]}
            if args.aligned_kv:
                for kind in ('k', 'v'):
                    assert all((torch.count_nonzero(tail_row(item, kind)) == 0 for item in payload[kind])), 'KV padding tail was not zeroed'
                report['zeroed_padding_row'] = physical_prefix + q
            if args.control:
                control = torch.load(args.control, map_location='cpu', weights_only=True)
                for name in payload:
                    assert len(payload[name]) == len(control[name])
                    for (actual, expected) in zip(payload[name], control[name], strict=True):
                        if args.aligned_kv and name in ('k', 'v'):
                            assert torch.count_nonzero(tail_row(actual, name)) == 0, 'KV padding tail was not zeroed'
                            expected = expected.clone()
                            tail_row(expected, name).zero_()
                        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16)), (name, float((actual - expected).abs().max()))
                report['byte_equal_old_fresh_controls'] = True
                if args.aligned_kv:
                    report['control_zeroed_padding_row'] = physical_prefix + q
            torch.save(payload, args.output.with_suffix('.pt'))
            report.update(passed=True, byte_equal_fresh_graph=True, outer_inference_mode=True, replay_mask_pattern='A/A/A/A' if args.baseline else 'A/B/in-place-C/A', versions=versions, mask_change_max_abs=float((references[0] - references[1]).abs().max()), graph={name: getattr(entry, name) for name in ('replay_count', 'recapture_count', 'kernel_count', 'dma_count')})
    finally:
        for graph in graphs:
            graph.clear()
        if handle is not None:
            torch.ops.rpu.rhino_vla_destroy(handle)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
