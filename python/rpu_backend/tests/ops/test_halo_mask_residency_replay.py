from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
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
    parser.add_argument('--expect-mask-residency', choices=('legacy', 'retained'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control', type=Path, help='Prior-library legacy arm .pt')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    (h, ff, nq, nkv, hd, layers, q, prefix, horizon) = (1536, 8960, 12, 2, 128, 2, 18, 4706, 4736)
    policy = GraphRuntimePolicy.from_environment()
    report = dict(passed=False, expected_residency=args.expect_mask_residency, backend=str(Path(rb.__file__).resolve()), effective_policy=asdict(policy), native_library=os.environ.get('RPU_BACKEND_SO'), geometry=dict(h=h, ff=ff, nq=nq, nkv=nkv, hd=hd, layers=layers, q=q, prefix=prefix, horizon=horizon), full_model=False)
    (handle, graphs) = (None, [])
    try:
        names = ('create', 'set_weights', 'set_moe_weights', 'set_chunk_envelope', 'resolve_stage_domain', 'step_forward', 'destroy')
        report['native_schemas'] = {name: str(getattr(torch.ops.rpu, 'halo_action_expert_' + name).default._schema) for name in names}
        with torch.inference_mode():

            def branch():
                weights = {name: [] for name in ('q', 'k', 'v', 'o', 'qn', 'kn', 'in', 'post', 'g', 'u', 'd', 'qb', 'kb', 'vb')}
                for _ in range(layers):
                    for (name, rows, cols, part, cores) in (('q', nq * hd, h, 1, 2), ('k', nkv * hd, h, 1, 2), ('v', nkv * hd, h, 1, 2), ('o', h, nq * hd, 0, 2), ('g', ff, h, 1, 8), ('u', ff, h, 1, 8), ('d', h, ff, 0, 8)):
                        host = (torch.randn(rows, cols) * (0.3 / cols ** 0.5)).half()
                        weights[name].append(transform_linear_weight(host, part, num_cores=cores).to('rpu'))
                    for (name, width) in (('qn', hd), ('kn', hd), ('in', h), ('post', h)):
                        weights[name].append((1 + torch.randn(width) * 0.01).half().to('rpu'))
                    for (name, width) in (('qb', nq * hd), ('kb', nkv * hd), ('vb', nkv * hd)):
                        weights[name].append((torch.randn(width) * 0.01).half().to('rpu'))
                weights['final'] = (1 + torch.randn(h) * 0.01).half().to('rpu')
                return weights
            (text, action) = (branch(), branch())
            theta = 11 / 1000000.0 ** (torch.arange(hd // 2).double() * 2 / hd)
            cos = theta.cos().half().expand(horizon, -1).contiguous().to('rpu')
            sin = theta.sin().half().expand(horizon, -1).contiguous().to('rpu')
            handle = int(torch.ops.rpu.halo_action_expert_create())
            order = ('q', 'k', 'v', 'o', 'qn', 'kn', 'in', 'post', 'g', 'u', 'd')
            torch.ops.rpu.halo_action_expert_set_weights(handle, *(text[name] for name in order), cos, sin, text['final'], nq, nkv, hd, h, ff, 1e-06, True, text['qb'], text['kb'], text['vb'])
            rows = torch.zeros(q, dtype=torch.float16)
            rows[[0, 17]] = 1
            torch.ops.rpu.halo_action_expert_set_moe_weights(handle, *(action[name] for name in order), action['qb'], action['kb'], action['vb'], action['final'], rows.to('rpu'), q)
            torch.ops.rpu.halo_action_expert_set_chunk_envelope(handle, horizon, 32)
            cache = RPUCache(layers, 1, horizon, nkv, hd, device='rpu', attn_tp=2)
            for item in (*cache.k_caches, *cache.v_caches):
                item.copy_((torch.randn(tuple(item.shape)) * 0.7).half().to('rpu'))
            hidden = (torch.randn(1, q, h) * 0.2).half().to('rpu')
            box = []
            (execution, chunk) = plan_bounded_prefill_execution(q, q, 0, resolve_stage_domain=lambda length: torch.ops.rpu.halo_action_expert_resolve_stage_domain(handle, prefix, 32), position=prefix, alignment=1, padding_rows=0, exact_chunk_size=32, graph_mode='COMPOSITE_CHILD', queue_owner_id=handle, lease_owner_id=handle, plan_result_sink=box.append)
            assert execution == q and chunk == 32
            selected = box[0].selected.stage_tuple
            desc = tuple(selected.physical_descriptor)
            assert desc and dict(selected.physical_metadata)['manifest_state'] == 1
            site = 3893593376471794725
            if args.expect_mask_residency == 'retained':
                assert desc.count(site) == 1
                offset = desc.index(site)
                assert desc[offset + 1:offset + 5] == (2, 0, 6, 0)
                assert desc[offset + 5:offset + 11] == (1, q, prefix + q, 2, layers, q)
                report['mask_schedule'] = list(desc[offset + 1:offset + 11])
            else:
                assert site not in desc
            report['physical_metadata'] = dict(selected.physical_metadata)
            report['logical_mask_uploads_per_graph'] = 1 if args.expect_mask_residency == 'retained' else layers
            sig = GraphSignature(op_id='halo_action_mask_residency', shapes=[q, h], dyn_dims=[prefix, *box[0].graph_key_words()], dtypes=[torch.float16])

            def mask(begin, end, versioned=True):
                value = torch.full((1, 1, q, prefix + q), -50000.0, dtype=torch.float16)
                value[:, :, :, begin:end] = 0
                return versioned_cpu_mask(value) if versioned else value
            templates = [mask(0, 32), mask(32, 64), mask(64, 96), mask(96, 128, False), mask(0, 32)]
            assert templates[3].is_inference() and all((not templates[i].is_inference() for i in (0, 1, 2, 4)))

            def forward(graph, current):
                cache.reset_to_position(prefix)
                with graph.capture(sig):
                    out = torch.ops.rpu.halo_action_expert_step_forward(handle, hidden, cache.k_caches, cache.v_caches, cos, sin, current, prefix, desc)
                value = out.cpu().clone()
                assert value.shape == (1, q, h) and torch.isfinite(value).all()
                return value
            references = []
            for current in templates:
                fresh = GraphCache(runtime_policy=policy)
                graphs.append(fresh)
                references.append(forward(fresh, current))
                fresh.clear()
            changes = [float((references[i] - references[i + 1]).abs().max()) for i in range(3)]
            assert min(changes) > 0.0001, changes
            (live_a, live_b) = (versioned_cpu_mask(templates[0]), versioned_cpu_mask(templates[1]))
            replay = GraphCache(runtime_policy=policy)
            graphs.append(replay)
            outputs = []
            versions = []
            for index in range(5):
                if index == 2:
                    old = live_b._version
                    live_b.copy_(templates[2])
                    assert live_b._version > old
                current = live_a if index in (0, 4) else templates[3] if index == 3 else live_b
                actual = forward(replay, current)
                assert torch.equal(actual, references[index]), (index, float((actual - references[index]).abs().max()))
                outputs.append(actual)
                versions.append(None if current.is_inference() else current._version)
            (entry,) = replay.snapshot()
            assert replay.size() == 1 and replay.cache_invariant_ok()
            assert entry.replay_count == 4 and entry.recapture_count == 0 and (entry.kernel_count > 0)
            payload = {'outputs': outputs, 'k': [x.cpu().clone() for x in cache.k_caches], 'v': [x.cpu().clone() for x in cache.v_caches]}
            if args.control:
                control = torch.load(args.control, map_location='cpu', weights_only=True)
                for name in payload:
                    for (actual, expected) in zip(payload[name], control[name], strict=True):
                        assert torch.equal(actual, expected), (name, float((actual - expected).abs().max()))
                report['byte_equal_legacy_control'] = True
            torch.save(payload, args.output.with_suffix('.pt'))
            report.update(passed=True, byte_equal_fresh_graph=True, outer_inference_mode=True, versions=versions, mask_change_max_abs=changes, graph={name: getattr(entry, name) for name in ('replay_count', 'recapture_count', 'kernel_count', 'dma_count')})
    finally:
        for graph in graphs:
            graph.clear()
        if handle is not None:
            torch.ops.rpu.halo_action_expert_destroy(handle)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
