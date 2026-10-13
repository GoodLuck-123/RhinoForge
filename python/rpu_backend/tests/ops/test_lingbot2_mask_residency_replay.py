from pathlib import Path
import argparse
import hashlib
import json
import os

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control', type=Path)
    parser.add_argument('--unroll', action='store_true')
    parser.add_argument('--grouped-w8', action='store_true')
    parser.add_argument('--baseline', action='store_true')
    args = parser.parse_args()
    mask_site = None if args.baseline else 2878184572497809152
    os.environ.update(RPU_LINGBOT2_GROUPED_EXPERTS=str(int(args.grouped_w8)), RPU_LINGBOT2_BASE_W8A16=str(int(args.grouped_w8)), RPU_LINGBOT2_EXPERT_W8A16=str(int(args.grouped_w8)), RPU_LINGBOT2_EXPERT_W4A16='0', RPU_ADARMS_FUSED_BCAST='0')
    import torch
    import rpu_backend
    from types import SimpleNamespace
    from rpu_backend.adapters.lingbot_vla_v2 import convert
    from rpu_backend.api import RPUCache
    from rpu_backend.graph import GraphCache, GraphSignature, GraphRuntimePolicy
    from rpu_backend.runtime.decoder import plan_bounded_prefill_execution
    from rpu_backend.runtime.weights import transform_linear_weight
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    geom = convert._default_geom()
    geom['EXP_LAYERS'] = 2
    (h, q, prefix, rope, layers, horizon) = (768, 51, 225, 32, 2, 512)
    bodies = 2 if args.unroll else 1
    stored = {}

    def gw(key):
        if key in stored:
            return stored[key]
        suffix = key.split('.layers.', 1)[-1].split('.', 1)[-1]
        shapes = {'self_attn.q_proj.weight': (4096, h), 'self_attn.k_proj.weight': (1024, h), 'self_attn.v_proj.weight': (1024, h), 'self_attn.o_proj.weight': (h, 4096), 'self_attn.q_proj.bias': (4096,), 'self_attn.k_proj.bias': (1024,), 'self_attn.v_proj.bias': (1024,), 'input_layernorm.weight': (h,), 'post_attention_layernorm.weight': (h,), 'mlp.gate.weight': (32, h), 'mlp.e_score_correction_bias': (32,), 'mlp.experts.gate_proj': (32, 512, h), 'mlp.experts.up_proj': (32, 512, h), 'mlp.experts.down_proj': (32, h, 512), 'mlp.shared_expert.gate_proj.weight': (704, h), 'mlp.shared_expert.up_proj.weight': (704, h), 'mlp.shared_expert.down_proj.weight': (h, 704)}
        if key == convert.EXP_P + 'norm.weight':
            value = torch.ones(h)
        else:
            shape = shapes[suffix]
            generator = torch.Generator().manual_seed(42 + int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'little'))
            value = torch.randn(shape, generator=generator) * (0.3 / shape[-1] ** 0.5 if len(shape) > 1 else 0.01)
            if 'layernorm' in suffix:
                value += 1
        stored[key] = value
        return value

    def cpu_mask(lo, hi):
        with torch.inference_mode(False), torch.no_grad():
            value = torch.full((1, 1, q, prefix + q), torch.finfo(torch.float16).min, dtype=torch.float16)
            value[..., lo:hi] = 0
            value[..., prefix:] = 0
            return value
    report = dict(passed=False, scope='synthetic native component; no Z2 composite/official-model credit', backend=str(Path(rpu_backend.__file__).resolve()), grouped_w8=args.grouped_w8, layers=layers, bodies=bodies, q=q, physical_prefix=prefix, real_prefix=215, rope_offset=rope)
    handle = None
    keep = None
    graphs = []
    owned = {}
    try:
        with torch.inference_mode():

            def publish(created, retained):
                owned.update(handle=int(created), keep=retained)
            (handle, keep) = convert.build_expert_moe(gw, max_seq=horizon, rope_theta=5000000, runtime_config=SimpleNamespace(bufonly=False, addr=False, schunk=32512, rchunk_requested=1632 if args.grouped_w8 else 0, dense_soft_router=False, fp16_top4=True, routed_only=False, down_acc16=False), geom=geom, on_handle_created=publish)
            stored.clear()
            film_cpu = [(torch.randn(layers, h) * 0.02 + (1 if i % 2 == 0 else 0)).half() for i in range(4)]
            films = [x.to('rpu') for x in film_cpu]
            if args.unroll:

                def weight(n, k):
                    return transform_linear_weight((torch.randn(n, k) * (0.2 / k ** 0.5)).half(), 1, num_cores=1).to('rpu')

                def bias(n):
                    return (torch.randn(n) * 0.01).half().to('rpu')
                head = (weight(h, 64), bias(h), weight(h, 64), weight(h, h), bias(h), weight(64, h), bias(64))
                time_all = (torch.randn(bodies, h) * 0.1).half().to('rpu')
                all_films = [x.unsqueeze(0).repeat(bodies, 1, 1).contiguous().to('rpu') for x in film_cpu]
                torch.ops.rpu.lingbot_v2_moe_set_denoise_unroll(handle, *head, time_all, *all_films, 32, 32, 64, 64, bodies)
                x0 = (torch.randn(1, q, 64) * 0.2).half()
                x0[:, :, 32:] = 0
                state = (torch.randn(1, 64) * 0.2).half()
                state[:, 32:] = 0
                x0 = x0.to('rpu')
                state = state.to('rpu')
                final = torch.empty_like(x0)
            else:
                torch.ops.rpu.lingbot_v2_moe_set_adarms_step_mutable(handle, *films)
            cache = RPUCache(layers, 1, horizon, 8, 128, device='rpu', attn_tp=8)
            for item in (*cache.k_caches, *cache.v_caches):
                item.copy_((torch.randn(tuple(item.shape)) * 0.7).half().to('rpu'))
            hidden = (torch.randn(1, q, h) * 0.2).half().to('rpu')
            box = []
            (length, chunk) = plan_bounded_prefill_execution(q, q, 0, position=prefix, alignment=1, padding_rows=0, exact_chunk_size=64, graph_mode='COMPOSITE_CHILD', queue_owner_id=handle, lease_owner_id=handle, resolve_stage_domain=lambda n: torch.ops.rpu.lingbot_v2_moe_resolve_action_stage_domain(handle, n, prefix, prefix + n, rope), plan_result_sink=box.append)
            assert (length, chunk) == (q, 64)
            selected = box[0].selected
            desc = tuple(selected.stage_tuple.physical_descriptor)
            assert dict(selected.stage_tuple.physical_metadata)['manifest_state'] == 1
            if mask_site is not None:
                assert desc.count(mask_site) == 1
                index = desc.index(mask_site)
                assert desc[index + 1:index + 5] == (2, 0, 6, 0), 'candidate must actually select residency'
                assert desc[index + 5:index + 11] == (q, prefix + q, 8, 8, bodies, layers)
                report['mask_route_words'] = list(desc[index:index + 5 + desc[index + 3]])
            else:
                assert 2878184572497809152 not in desc, 'baseline must use its unchanged native library'
            policy = GraphRuntimePolicy.from_environment()
            sig = GraphSignature(op_id='lingbot2_mask_growth_smoke', shapes=[q, h], dyn_dims=[prefix, rope, bodies, *box[0].graph_key_words()], dtypes=[torch.float16])

            def run(graph, mask):
                cache.reset_to_position(prefix)
                with graph.capture(sig):
                    if args.unroll:
                        torch.ops.rpu.lingbot_v2_moe_denoise_unroll_forward(handle, x0, state, cache.k_caches, cache.v_caches, mask, final, -1 / bodies, prefix, rope, bodies, desc)
                        out = final
                    else:
                        out = torch.ops.rpu.lingbot_v2_moe_forward(handle, hidden, cache.k_caches, cache.v_caches, mask, prefix, rope, -1, desc)
                result = out.cpu().clone()
                assert torch.isfinite(result).all()
                return result
            masks = [cpu_mask(0, 215), cpu_mask(0, 100), cpu_mask(100, 215), cpu_mask(0, 215)]
            references = []
            for mask in masks:
                graph = GraphCache(runtime_policy=policy)
                graphs.append(graph)
                references.append(run(graph, mask))
                graph.clear()
            assert (references[0] - references[1]).abs().max() > 1e-05
            graph = GraphCache(runtime_policy=policy)
            graphs.append(graph)
            (a, b) = (cpu_mask(0, 215), cpu_mask(0, 100))
            actuals = []
            for i in range(4):
                current = a if mask_site is None or i in (0, 3) else b
                if i == 2 and mask_site is not None:
                    version = b._version
                    b.copy_(masks[2])
                    assert b._version > version
                value = run(graph, current)
                assert torch.equal(value, references[0 if mask_site is None else i])
                actuals.append(value)
            (entry,) = graph.snapshot()
            assert entry.replay_count == 3 and entry.recapture_count == 0 and (entry.kernel_count > 0)
            assert graph.size() == 1 and graph.cache_invariant_ok()
            payload = dict(outputs=references if mask_site is None else actuals, k=[x.cpu().clone() for x in cache.k_caches], v=[x.cpu().clone() for x in cache.v_caches])
            if args.control:
                control = torch.load(args.control, map_location='cpu', weights_only=True)
                for name in payload:
                    for (left, right) in zip(payload[name], control[name], strict=True):
                        assert torch.equal(left, right), name
            torch.save(payload, args.output.with_suffix('.pt'))
            report.update(passed=True, replay_count=entry.replay_count, recapture_count=entry.recapture_count, dma_count=entry.dma_count, mask_changes=mask_site is not None, control_compared=bool(args.control))
    finally:
        for graph in graphs:
            graph.clear()
        if owned:
            torch.ops.rpu.lingbot_v2_moe_destroy(owned['handle'])
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
if __name__ == '__main__':
    main()
