from __future__ import annotations
import argparse
import json
from pathlib import Path

def main():
    import torch
    import rpu_backend as rb
    from rpu_backend.api import RPUCache
    from rpu_backend.graph import GraphCache, GraphSignature
    from rpu_backend.runtime.decoder import plan_bounded_prefill_execution
    from rpu_backend.runtime.weights import transform_linear_weight
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expect-mask-residency', choices=('retained', 'legacy'), default='retained')
    args = parser.parse_args()
    torch.manual_seed(927)
    (h, ff, nq, hd, cross, layers, q, k) = (1536, 6144, 32, 48, 2048, 5, 41, 82)
    report = {'backend': str(Path(rb.__file__).resolve()), 'geometry': dict(hidden=h, intermediate=ff, heads=nq, head_dim=hd, cross_dim=cross, layers=layers, q=q, k=k)}
    weights = {name: [] for name in ('q', 'k', 'v', 'o', 'f1', 'f2', 'aw', 'ab', 'qb', 'kb', 'vb', 'ob', 'f1b', 'f2b')}
    for layer in range(layers):
        kv_width = cross if layer % 2 == 0 else h
        for (name, n, cols, part) in (('q', h, h, 1), ('k', h, kv_width, 1), ('v', h, kv_width, 1), ('o', h, h, 0), ('f1', ff, h, 1), ('f2', h, ff, 0), ('aw', 2 * h, h, 0)):
            host = (torch.randn(n, cols) * (0.3 / cols ** 0.5)).half()
            weights[name].append(transform_linear_weight(host, part, num_cores=8).to('rpu'))
        for (name, n) in (('ab', 2 * h), ('qb', h), ('kb', h), ('vb', h), ('ob', h), ('f1b', ff), ('f2b', h)):
            weights[name].append(torch.zeros(n, dtype=torch.float16, device='rpu'))
    handle = int(torch.ops.rpu.gr00t_dit_create())
    graphs = []
    try:
        torch.ops.rpu.gr00t_dit_set_weights(handle, *(weights[name] for name in ('q', 'k', 'v', 'o', 'f1', 'f2', 'aw', 'ab', 'qb', 'kb', 'vb', 'ob', 'f1b', 'f2b')), [1, 0, 1, 0, 1], nq, hd, h, ff, cross, 1e-05)
        kv = RPUCache(layers, 1, 96, nq, hd, device='rpu')
        hidden = (torch.randn(1, q, h) * 0.2).half().to('rpu')
        cond = (torch.randn(h) * 0.2).half().to('rpu')
        vl = (torch.randn(1, k, cross) * 0.7).half().to('rpu')
        box = []
        plan_bounded_prefill_execution(q, q, 0, resolve_stage_domain=lambda length: torch.ops.rpu.gr00t_dit_resolve_stage_domain(handle, length, k, None), alignment=1, padding_rows=0, graph_mode='COMPOSITE_CHILD', queue_owner_id=handle, plan_result_sink=box.append)
        plan = box[0]
        desc = tuple(plan.selected.stage_tuple.physical_descriptor)
        assert desc
        mask_site = 4273136855374514935
        if args.expect_mask_residency == 'retained':
            site = desc.index(mask_site)
            assert desc[site + 1] == 2, 'Test must exercise retained text/image SPM'
            typed_mask_schedule = 2
        else:
            if mask_site in desc:
                assert desc[desc.index(mask_site) + 1] == 1
            typed_mask_schedule = 'legacy'
        sig = GraphSignature(op_id='gr00t_mask_ordinal', shapes=[q, h], dyn_dims=[k, *plan.graph_key_words()], dtypes=[torch.float16])

        def mask(start, end):
            out = torch.full((q, k), float('-inf'), dtype=torch.float16)
            out[:, start:end] = 0
            return out
        pairs = [(mask(0, 41), mask(41, 82)), (mask(10, 30), mask(55, 75)), (mask(41, 82), mask(0, 41)), (mask(0, 41), mask(41, 82))]

        def forward(graph, pair):
            with graph.capture(sig):
                out = torch.ops.rpu.gr00t_dit_forward(handle, hidden, cond, kv.k_caches, kv.v_caches, None, vl, pair[0], pair[1], desc)
            result = out.cpu().clone()
            assert torch.isfinite(result).all()
            return result
        references = []
        for pair in pairs:
            fresh = GraphCache()
            graphs.append(fresh)
            references.append(forward(fresh, pair))
            fresh.clear()
        assert (references[0] - references[1]).abs().max().item() > 0.0001
        replay = GraphCache()
        graphs.append(replay)
        outputs = []
        replay_pair = tuple((value.clone() for value in pairs[0]))
        for (pair, expected) in zip(pairs, references):
            for (owner, value) in zip(replay_pair, pair):
                owner.copy_(value)
            actual = forward(replay, replay_pair)
            assert torch.equal(actual, expected), float((actual - expected).abs().max())
            outputs.append(actual)
        (entry,) = replay.snapshot()
        assert replay.size() == 1 and replay.cache_invariant_ok()
        assert entry.replay_count == 3 and entry.recapture_count == 0 and (entry.kernel_count > 0)
        report.update(passed=True, byte_equal_fresh_graph=True, typed_mask_schedule=typed_mask_schedule, expected_mask_residency=args.expect_mask_residency, same_owner_inplace_mask_changes=True, boundary='five plain DiT layers (text/self/image/self/text); denoise hooks covered by public E2E', logical_mask_uploads_per_call=2 if args.expect_mask_residency == 'retained' else 3, mask_change_max_abs=float((references[0] - references[1]).abs().max()), graph={name: getattr(entry, name) for name in ('replay_count', 'recapture_count', 'kernel_count', 'dma_count')})
        torch.save(outputs, args.output.with_suffix('.pt'))
    finally:
        for graph in graphs:
            graph.clear()
        torch.ops.rpu.gr00t_dit_destroy(handle)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
