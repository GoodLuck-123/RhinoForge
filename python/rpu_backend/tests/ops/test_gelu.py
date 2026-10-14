import sys
import torch
import torch.nn.functional as F
import rpu_backend

def _cpu_gelu(x: torch.Tensor, approximate: str) -> torch.Tensor:
    return F.gelu(x, approximate=approximate)

def _require_equal(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    actual_cpu = actual.cpu()
    if not torch.equal(actual_cpu, expected):
        diff = (actual_cpu.float() - expected.float()).abs().max().item()
        raise AssertionError(f'{label}: CPU-fallback output drift, max_abs={diff}')

def main() -> int:
    torch.manual_seed(17)
    a = torch.randn(4, 17, dtype=torch.float16)
    b = (torch.randn(4, 17) * 2.5 - 0.75).to(torch.float16)
    for approximate in ('none', 'tanh'):
        out = F.gelu(a.to('rpu'), approximate=approximate)
        if out.device.type != 'rpu':
            raise AssertionError(f'eager GELU returned {out.device}, expected the original device device')
        _require_equal(out, _cpu_gelu(a, approximate), f'eager/{approximate}')
    cache = rpu_backend.graph.GraphCache()
    sig = rpu_backend.graph.GraphSignature(op_id='eager_gelu_cpu_fallback', shapes=list(a.shape), dyn_dims=[], dtypes=[torch.float16])
    outputs = []
    in_scope_states = []
    for (label, cpu_input) in (('A0', a), ('B', b), ('A1', a)):
        with cache.capture(sig) as graph:
            in_scope_states.append(graph.state())
            out = F.gelu(cpu_input.to('rpu'), approximate='none')
        out_cpu = out.cpu().clone()
        expected = _cpu_gelu(cpu_input, 'none')
        if not torch.equal(out_cpu, expected):
            diff = (out_cpu.float() - expected.float()).abs().max().item()
            raise AssertionError(f'capture/{label}: max_abs={diff}')
        outputs.append(out_cpu)
    if in_scope_states != [1, 3, 3]:
        raise AssertionError(f'expected RECORDING/REPLAYING/REPLAYING, got {in_scope_states}')
    if not torch.equal(outputs[0], outputs[2]) or torch.equal(outputs[0], outputs[1]):
        raise AssertionError('A/B/A stale-output gate failed')
    if cache.size() != 1 or not cache.cache_invariant_ok():
        raise AssertionError(f'GELU fallback GraphCache invariant failed: size={cache.size()}')
    snapshot = list(cache.snapshot())
    if len(snapshot) != 1:
        raise AssertionError(f'expected one cache snapshot, got {len(snapshot)}')
    entry = snapshot[0]
    if entry.host_callback_tier2_count != 1:
        raise AssertionError(f'aten::gelu did not route through exactly one Tier2 CPU host callback: count={entry.host_callback_tier2_count}')
    if entry.kernel_count != 0:
        raise AssertionError(f'eager GELU unexpectedly launched {entry.kernel_count} device kernels')
    if entry.replay_count < 2 or entry.recapture_count != 0:
        raise AssertionError(f'GELU fallback did not reach stable REPLAY: replay={entry.replay_count}, recapture={entry.recapture_count}')
    chain_a = torch.randn(4, 32, dtype=torch.float16)
    chain_b = (torch.randn(4, 32) * 1.75 + 0.25).to(torch.float16)
    chain_cache = rpu_backend.graph.GraphCache()
    chain_sig = rpu_backend.graph.GraphSignature(op_id='kernel_gelu_kernel_cpu_fallback', shapes=list(chain_a.shape), dyn_dims=[], dtypes=[torch.float16])
    chain_outputs = []
    chain_states = []
    for (label, cpu_input) in (('A0', chain_a), ('B', chain_b), ('A1', chain_a)):
        rpu_input = cpu_input.to('rpu')
        with chain_cache.capture(chain_sig) as graph:
            chain_states.append(graph.state())
            pre = torch.add(rpu_input, rpu_input)
            mid = F.gelu(pre, approximate='tanh')
            chained = torch.add(mid, mid)
        actual = chained.cpu().clone()
        cpu_pre = torch.add(cpu_input, cpu_input)
        cpu_mid = F.gelu(cpu_pre, approximate='tanh')
        expected = torch.add(cpu_mid, cpu_mid)
        if not torch.equal(actual, expected):
            diff = (actual.float() - expected.float()).abs().max().item()
            raise AssertionError(f'kernel/HCB/kernel/{label}: max_abs={diff}')
        chain_outputs.append(actual)
    if chain_states != [1, 3, 3]:
        raise AssertionError(f'kernel/HCB/kernel expected RECORDING/REPLAYING/REPLAYING, got {chain_states}')
    if not torch.equal(chain_outputs[0], chain_outputs[2]) or torch.equal(chain_outputs[0], chain_outputs[1]):
        raise AssertionError('kernel/HCB/kernel A/B/A stale-output gate failed')
    chain_snapshot = list(chain_cache.snapshot())
    if len(chain_snapshot) != 1 or chain_cache.size() != 1 or (not chain_cache.cache_invariant_ok()):
        raise AssertionError('kernel/HCB/kernel GraphCache invariant failed')
    chain_entry = chain_snapshot[0]
    if chain_entry.host_callback_tier2_count != 1:
        raise AssertionError(f'kernel/HCB/kernel expected one Tier2 callback, got {chain_entry.host_callback_tier2_count}')
    if chain_entry.kernel_count < 2:
        raise AssertionError(f'kernel/HCB/kernel expected at least two device kernels, got {chain_entry.kernel_count}')
    if chain_entry.replay_count < 2 or chain_entry.recapture_count != 0:
        raise AssertionError(f'kernel/HCB/kernel did not reach stable REPLAY: replay={chain_entry.replay_count}, recapture={chain_entry.recapture_count}')
    print(f'PASS: eager exact/tanh GELU use CPU fallback; capture has one Tier2 host callback, BUILD -> {entry.replay_count} REPLAY, and both pure-HCB and kernel/HCB/kernel A/B/A outputs are fresh')
    return 0
if __name__ == '__main__':
    sys.exit(main())
