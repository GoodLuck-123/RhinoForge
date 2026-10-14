from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

def _validate_fixture_scratch(hidden, num_q_heads, num_kv_heads, head_dims):
    attention_cores = min(8, num_kv_heads)
    assert num_kv_heads % attention_cores == 0
    assert num_q_heads % attention_cores == 0 and num_q_heads >= num_kv_heads
    required = num_kv_heads // attention_cores * max(head_dims)
    if hidden < required:
        raise ValueError(f'synthetic hidden={hidden} cannot hold K-norm scratch width={required}')

def main():
    import torch
    import rpu_backend as rb
    from rpu_backend.adapters import gemma4
    from rpu_backend.adapters.gemma4.cache import Gemma4KVCache
    from rpu_backend.adapters.gemma4.geometry import LayerGeom
    from rpu_backend.adapters.gemma4.weights import set_gemma4_weights
    from rpu_backend.runtime.weights import transform_linear_weight
    from rpu_backend.runtime.execution_planner import _decode_native_stage_domain
    from rpu_backend.graph import GraphCache, GraphSignature, GraphRuntimePolicy
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tensors', type=Path, help='Save all actual/reference CPU outputs')
    parser.add_argument('--mixed-global-first', action='store_true', help='Use global512 followed by two sliding256 layers')
    parser.add_argument('--expect-mask-residency', choices=('any', 'retained', 'legacy'), default='any')
    args = parser.parse_args()
    torch.manual_seed(42)
    (h, intermediate, hd, nq, nkv, horizon) = (512, 1024, 256, 8, 2, 576)
    globals_ = [True, False, False] if args.mixed_global_first else [False, False]
    geom = [LayerGeom(i, 'full_attention' if global_ else 'sliding_attention', global_, 512 if global_ else hd, nq, nkv, 'proportional' if global_ else 'default', None if global_ else 512, False, i, True) for (i, global_) in enumerate(globals_)]
    _validate_fixture_scratch(h, nq, nkv, [g.head_dim for g in geom])
    policy_factory = getattr(gemma4, '_gemma4_graph_runtime_policy', None)
    policy = policy_factory() if policy_factory else replace(GraphRuntimePolicy.from_environment(), fmb_fast_replay=False, fmb_deep_fast_replay=False)
    assert not policy.fmb_fast_replay and (not policy.fmb_deep_fast_replay)
    report = {'backend': str(Path(rb.__file__).resolve()), 'effective_policy': asdict(policy), 'owner_policy_available': policy_factory is not None, 'geometry': [asdict(g) for g in geom], 'mask_schedules': {}, 'cases': []}
    tensor_outputs = {}

    def weight(n, k, partition, cores):
        host = (torch.randn(n, k) * (0.2 / k ** 0.5)).half()
        return transform_linear_weight(host, partition, num_cores=cores).to('rpu')
    lists = {name: [] for name in ('q', 'k', 'v', 'o', 'q_norm', 'k_norm', 'input_norm', 'post_attn_norm', 'pre_ff_norm', 'post_ff_norm', 'gate', 'up', 'down')}
    for layer in geom:
        width = layer.head_dim
        for (name, n, k, part, cores) in (('q', nq * width, h, 1, 2), ('k', nkv * width, h, 1, 2), ('v', nkv * width, h, 1, 2), ('o', h, nq * width, 0, 2), ('gate', intermediate, h, 1, 8), ('up', intermediate, h, 1, 8), ('down', h, intermediate, 0, 8)):
            lists[name].append(weight(n, k, part, cores))
        for name in ('q_norm', 'k_norm', 'input_norm', 'post_attn_norm', 'pre_ff_norm', 'post_ff_norm'):
            lists[name].append(torch.ones(width if name in ('q_norm', 'k_norm') else h, dtype=torch.float16, device='rpu'))
    theta = torch.arange(horizon).float()[:, None] / 10000 ** (torch.arange(hd // 2).float() * 2 / hd)
    (cos, sin) = (theta.cos().half().to('rpu'), theta.sin().half().to('rpu'))
    global_hd = max((layer.head_dim for layer in geom))
    theta_global = torch.arange(horizon).float()[:, None] / 10000 ** (torch.arange(global_hd // 2).float() * 2 / global_hd)
    (cos_global, sin_global) = (theta_global.cos().half().to('rpu'), theta_global.sin().half().to('rpu'))
    final = torch.ones(h, dtype=torch.float16, device='rpu')
    handle = int(torch.ops.rpu.gemma4_create())
    caches = []
    try:
        set_gemma4_weights(handle, geom, lists, (cos, sin, cos_global, sin_global), final, num_q_heads=nq, num_kv_heads_max=nkv, head_dim_max=global_hd, hidden_size=h, intermediate_size=intermediate, eps=1e-06, sliding_window=512)
        torch.ops.rpu.gemma4_set_chunk_envelope(handle, horizon, 16)

        def new_cache():
            return Gemma4KVCache(geom, max_seq_len=horizon, device='rpu')

        def descriptor(length, position, chunk):
            canonical = 0 if length == 1 else position
            maximum = min((position + 1 + 15) // 16 * 16, horizon) if length == 1 else position + length
            wire = torch.ops.rpu.gemma4_resolve_stage_domain(handle, length, canonical, maximum, chunk)
            candidates = _decode_native_stage_domain(tuple(wire), execution_len=length, logical_len=length, position=canonical, graph_mode='RETAINED_CACHE')
            assert len(candidates) == 1
            desc = tuple(candidates[0].physical_descriptor)
            from rpu_backend.tests._hw_closure_evidence import complete_descriptor_evidence
            evidence = complete_descriptor_evidence(desc, expected_lifecycle='RETAINED_CACHE', component_path='synthetic_gemma4', stage='decode' if length == 1 else 'prefill')
            schedules = [r for r in evidence['routes'] if r['site_id'] == 858194843690279753]
            if args.expect_mask_residency == 'retained':
                assert len(schedules) == 1
                assert schedules[0]['selector'] == (2 if length <= chunk else 1)
                assert schedules[0]['arguments'][1] == (1 if args.mixed_global_first else 0)
            elif args.expect_mask_residency == 'legacy':
                assert not schedules
            report['mask_schedules'][f'{length}:{position}:{chunk}'] = schedules
            return desc

        def forward(cache, hidden, position, desc, graph, sig):
            with graph.capture(sig):
                out = torch.ops.rpu.gemma4_forward(handle, hidden, cache.k_caches, cache.v_caches, None, position, True, 0, None, desc)
            return out.cpu().clone()

        def equal(actual, expected, label):
            assert actual.shape == expected.shape and torch.isfinite(actual).all(), label
            assert torch.equal(actual, expected), (label, float((actual.float() - expected.float()).abs().max()))

        def paired_caches():
            (first, second) = (new_cache(), new_cache())
            for arrays in ((first.k_caches, second.k_caches), (first.v_caches, second.v_caches)):
                for (a, b) in zip(*arrays):
                    host = (torch.randn(tuple(a.shape)) * 0.15).half()
                    a.copy_(host.to('rpu'))
                    b.copy_(host.to('rpu'))
            return (first, second)
        (control, candidate) = paired_caches()
        replay = GraphCache(runtime_policy=policy)
        caches.append(replay)
        decode_sig = None
        decode_cases = []
        for (step, position) in enumerate((512, 513, 514, 515)):
            hidden = (torch.randn(1, 1, h) * 0.2).half().to('rpu')
            desc = descriptor(1, position, 16)
            sig = GraphSignature(op_id='gemma4_mask_decode', shapes=[1, h], dyn_dims=[33, *desc], dtypes=[torch.float16])
            if decode_sig is None:
                decode_sig = desc
            assert desc == decode_sig, 'same-bucket decode must use the same descriptor'
            fresh = GraphCache(runtime_policy=policy)
            try:
                expected = forward(control, hidden, position, desc, fresh, sig)
            finally:
                fresh.clear()
            decode_cases.append((position, hidden, desc, sig, expected))
        for (position, hidden, desc, sig, expected) in decode_cases:
            actual = forward(candidate, hidden, position, desc, replay, sig)
            equal(actual, expected, f'decode-{position}')
            tensor_outputs[f'decode_{position}_actual'] = actual
            tensor_outputs[f'decode_{position}_reference'] = expected
            report['cases'].append({'kind': 'decode', 'position': position, 'byte_equal_fresh_graph': True, 'graph_size': replay.size()})
        assert replay.size() == 1 and replay.cache_invariant_ok()
        entries = replay.snapshot()
        assert len(entries) == 1 and entries[0].replay_count >= 3 and (entries[0].recapture_count == 0)
        report['decode_graph'] = [{name: getattr(row, name) for name in ('replay_count', 'recapture_count', 'kernel_count', 'dma_count')} for row in entries]
        multi = GraphCache(runtime_policy=policy)
        caches.append(multi)
        desc = descriptor(33, 512, 16)
        sig = GraphSignature(op_id='gemma4_mask_chunks', shapes=[33, h], dyn_dims=list(desc), dtypes=[torch.float16])
        multi_cases = []
        for repeat in range(4):
            (control, candidate) = paired_caches()
            hidden = (torch.randn(1, 33, h) * 0.2).half().to('rpu')
            parts = []
            for start in (0, 16, 32):
                length = min(16, 33 - start)
                part_desc = descriptor(length, 512 + start, 16)
                part_sig = GraphSignature(op_id='gemma4_mask_single_chunk', shapes=[length, h], dyn_dims=[512 + start, *part_desc], dtypes=[torch.float16])
                fresh = GraphCache(runtime_policy=policy)
                try:
                    parts.append(forward(control, hidden[:, start:start + length].contiguous(), 512 + start, part_desc, fresh, part_sig))
                finally:
                    fresh.clear()
            multi_cases.append((candidate, hidden, torch.cat(parts, dim=1)))
        for (repeat, (candidate, hidden, expected)) in enumerate(multi_cases):
            actual = forward(candidate, hidden, 512, desc, multi, sig)
            equal(actual, expected, f'multi-{repeat}')
            tensor_outputs[f'multi_{repeat}_actual'] = actual
            tensor_outputs[f'multi_{repeat}_reference'] = expected
            report['cases'].append({'kind': 'multi_chunk', 'repeat': repeat, 'chunks': [16, 16, 1], 'byte_equal_chunk_reference': True, 'graph_size': multi.size()})
        assert multi.size() == 1 and multi.cache_invariant_ok()
        entries = multi.snapshot()
        assert len(entries) == 1 and entries[0].replay_count >= 3 and (entries[0].recapture_count == 0)
        report['multi_graph'] = [{name: getattr(row, name) for name in ('replay_count', 'recapture_count', 'kernel_count', 'dma_count')} for row in entries]
        report['passed'] = True
    finally:
        for cache in caches:
            cache.clear()
        torch.ops.rpu.gemma4_destroy(handle)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=str) + '\n')
    if args.tensors:
        args.tensors.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor_outputs, args.tensors)
    print(json.dumps(report, indent=2, default=str))
if __name__ == '__main__':
    main()
