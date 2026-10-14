from __future__ import annotations
import sys
import torch
import torch.nn.functional as F
import rpu_backend
SEED = 20260614
EPS = 1e-06

def _rmsnorm_cpu(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return w.float() * (xf * torch.rsqrt(var + eps))

def _run_case(name: str, M: int, C: int, scale: float) -> bool:
    torch.manual_seed(SEED)
    x = torch.randn(M, C, dtype=torch.float32) * scale
    w = torch.randn(C, dtype=torch.float32) * 0.1 + 1.0
    ref = _rmsnorm_cpu(x, w, EPS)
    x_rpu = x.half().to('rpu')
    w_rpu = w.half().to('rpu')
    out = torch.ops.rpu.rms_norm(x_rpu, [C], w_rpu, EPS)
    out_cpu = out.float().cpu()
    assert out_cpu.shape == ref.shape, f'{name}: shape {tuple(out_cpu.shape)} != {tuple(ref.shape)}'
    cos = F.cosine_similarity(out_cpu.reshape(-1), ref.reshape(-1), dim=0).item()
    mse = (out_cpu - ref).pow(2).mean().item()
    max_abs = (out_cpu - ref).abs().max().item()
    row_cos = F.cosine_similarity(out_cpu, ref, dim=1).mean().item()
    ok = cos >= 0.999
    print(f"  [{name:14s}] M={M:4d} C={C:4d} scale={scale:>4g}  cos={cos:.6f} row_cos={row_cos:.6f} mse={mse:.3e} max_abs={max_abs:.3e} -> {('PASS' if ok else 'FAIL')}")
    if not ok:
        r = (out_cpu[0, :6] / ref[0, :6]).tolist()
        print(f'       row0 ratio out/ref[:6] = {[round(v, 4) for v in r]}')
        print(f'       row0 out[:6]  = {[round(v, 4) for v in out_cpu[0, :6].tolist()]}')
        print(f'       row0 ref[:6]  = {[round(v, 4) for v in ref[0, :6].tolist()]}')
    return ok

def main() -> int:
    ok = True
    ok &= _run_case('C2048_small', 128, 2048, 1.0)
    ok &= _run_case('C1280_small', 196, 1280, 1.0)
    ok &= _run_case('C1280_x10', 196, 1280, 10.0)
    ok &= _run_case('C1280_x50', 196, 1280, 50.0)
    print(f"\n{('ALL PASS' if ok else 'FAILURES')}")
    return 0 if ok else 1
if __name__ == '__main__':
    sys.exit(main())

def test_rmsnorm_merger_boundary():
    assert main() == 0

def _require_spm_numeric(actual, expected, label):
    from rpu_backend.tests.conformance._numeric import format_metrics, require_same_semantic
    metrics = require_same_semantic(actual.cpu(), expected, actual_dtype=torch.float16, cosine_min=0.999, relative_l2_max=0.02, row_relative_l2_max=0.02, label=label)
    print(f'{label}: {format_metrics(metrics)}')

def test_hyvla_merger_immediate_boundaries(record_property):
    gen = torch.Generator().manual_seed(SEED)
    try:
        for (groups, scale) in ((16, 0.25), (49, 4.0)):
            x = (torch.randn(4, groups, 256, generator=gen) * scale).half()
            torch.ops.rpu.spm_alloc_reset_temporary()
            pooled = torch.ops.rpu.hyvla_merger_pool(x.to('rpu'))
            _require_spm_numeric(pooled, x.float().sum(0).expand_as(x).contiguous(), f'HY pool G={groups} scale={scale}')
            for score_scale in (0.25, 8.0):
                scores = (torch.randn(x.shape, generator=gen) * score_scale).half()
                torch.ops.rpu.spm_alloc_reset_temporary()
                combined = torch.ops.rpu.hyvla_merger_combine(x.to('rpu'), scores.to('rpu'))
                expected = (x.float() * torch.softmax(scores.float(), dim=0)).sum(0)
                _require_spm_numeric(combined, expected, f'HY combine G={groups} score_scale={score_scale}')
    finally:
        torch.ops.rpu.spm_alloc_reset_temporary()

def _hyvla_fused_cpu(x, wa, b0, wb, w2, b2, wp2, bp2, wp1=None, bp1=None):
    x = x.float()
    if wp1 is not None:
        x = F.linear(x, wp1.float(), bp1.float())
    pooled = x.sum(0)
    scores = F.linear(x, wa.float(), b0.float()) + F.linear(pooled, wb.float())
    scores = F.linear(F.gelu(scores, approximate='tanh'), w2.float(), b2.float())
    combined = (x * torch.softmax(scores, dim=0)).sum(0)
    return F.linear(F.gelu(combined, approximate='tanh'), wp2.float(), bp2.float())

def test_hyvla_merger_fused_mutable_graph(record_property):
    from rpu_backend.runtime.weights import transform_linear_weight
    from rpu_backend.tests._hw_closure_evidence import stable_replay
    gen = torch.Generator().manual_seed(SEED)
    for groups in (16, 49):
        for proj1 in (False, True):
            (width, input_width) = (256, 128 if proj1 else 256)
            logical = {}
            for name in ('wa', 'wb', 'w2', 'wp2', *(('wp1',) if proj1 else ())):
                k = input_width if name == 'wp1' else width
                logical[name] = (torch.randn(width, k, generator=gen) * (0.7 / k ** 0.5)).half()
            logical['wb'] *= 0.25
            for name in ('b0', 'b2', 'bp2', *(('bp1',) if proj1 else ())):
                logical[name] = (torch.randn(width, generator=gen) * 0.05).half()
            packed = {name: (transform_linear_weight(value.clone(), partition=1, num_cores=8) if name.startswith('w') else value).to('rpu') for (name, value) in logical.items()}
            a = (torch.randn(4, groups, input_width, generator=gen) * 0.7).half()
            b = (torch.randn(a.shape, generator=gen) * 2 + 0.3).half()
            refs = [_hyvla_fused_cpu(value, **logical) for value in (a, b, a)]
            assert not torch.allclose(refs[0], refs[1], rtol=0.02, atol=0.02)
            cache = rpu_backend.graph.GraphCache()
            cache.begin_warmup()
            sig = rpu_backend.graph.GraphSignature(op_id='test_hyvla_merger_fused', shapes=[groups, width, input_width, int(proj1)], dtypes=[torch.float16])
            (inputs, outputs, snapshots) = ([], [], [])
            try:
                for (iteration, (value, ref)) in enumerate(zip((a, b, a), refs, strict=True)):
                    current = value.to('rpu')
                    inputs.append(current)
                    assert len({item.data_ptr() for item in inputs}) == len(inputs)
                    torch.ops.rpu.spm_alloc_reset_temporary()
                    before = cache.snapshot()
                    with cache.capture(sig):
                        out = torch.ops.rpu.hyvla_merger_fused(current, **packed)
                    outputs.append(out)
                    assert len({item.data_ptr() for item in outputs}) == len(outputs)
                    after = cache.snapshot()
                    assert len(after) == cache.size() == 1 and cache.cache_invariant_ok()
                    entry = after[0]
                    assert entry.recapture_count == 0 and entry.kernel_count > 0
                    assert entry.data_node_count >= 5 and (not entry.non_replayable_reason)
                    if iteration == 0:
                        assert not before and entry.replay_count == 0
                        assert entry.graph_lifetime_id > 0 and entry.build_generation > 0
                        assert entry.execution_ordinal == 1
                        cache.freeze()
                    else:
                        assert cache.is_frozen() and stable_replay(cache, before, after, 1)
                        assert entry.execution_ordinal == before[0].execution_ordinal + 1
                    _require_spm_numeric(out, ref, f'HY fused G={groups} proj1={proj1} iteration={iteration}')
                    snapshots.append(out.cpu().clone())
                    assert all((torch.equal(old.cpu(), saved) for (old, saved) in zip(outputs, snapshots, strict=True)))
                assert torch.equal(snapshots[0], snapshots[2]), 'A/B/A replay is not byte-stable'
            finally:
                cache.clear()
                assert cache.size() == 0 and cache.cache_invariant_ok()
                torch.ops.rpu.spm_alloc_reset_temporary()
