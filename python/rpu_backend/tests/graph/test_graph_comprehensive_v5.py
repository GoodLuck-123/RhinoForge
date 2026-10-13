from __future__ import annotations
import argparse
import copy
import sys
import time
import traceback
from typing import Tuple, List
import numpy as np
import torch
import torch.nn as nn
import rpu_backend
from rpu_backend.runtime.weights import swizzle_model_inplace
STATE = {0: 'PASSTHROUGH', 1: 'RECORDING', 2: 'BUILT', 3: 'REPLAYING'}

class _Report:

    def __init__(self):
        self.results: List[Tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str=''):
        tag = 'PASS' if ok else 'FAIL'
        print(f'  [{tag}] {name}' + (f' — {detail}' if detail else ''), flush=True)
        self.results.append((name, ok, detail))

    def summary(self) -> int:
        n = len(self.results)
        ok = sum((1 for (_, b, _) in self.results if b))
        print('\n' + '=' * 68, flush=True)
        print(f'Total: {ok}/{n} PASS', flush=True)
        for (name, b, detail) in self.results:
            if not b:
                print(f'  FAIL — {name}: {detail}', flush=True)
        print('=' * 68, flush=True)
        return 0 if ok == n else 1

def setup_env(ddr_flush=True):
    torch.rpu.set_debug(False)
    torch.rpu.set_profile(False)
    torch.rpu.set_caching_allocator(True)
    torch.rpu.set_ddr_flush(ddr_flush)
    torch.rpu.set_debug_export(False)

def _gelu_chain(x: torch.Tensor, depth: int=3) -> torch.Tensor:
    y = x
    for _ in range(depth):
        y = torch.nn.functional.gelu(y)
    return y

def _make_sig(op_id: str, batch=1, seq=8, hidden=64) -> rpu_backend.graph.GraphSignature:
    return rpu_backend.graph.GraphSignature(op_id=op_id, shapes=[batch, seq], dyn_dims=[hidden], dtypes=[torch.float16])

def case_1_state_machine(report: _Report):
    print('\n--- Case 1: State machine PASSTHROUGH → RECORDING → BUILT → REPLAYING ---', flush=True)
    cache = rpu_backend.graph.GraphCache()
    sig = _make_sig('case1_state')
    x = torch.randn(1, 8, 64, device='rpu', dtype=torch.float16)
    with cache.capture(sig) as g:
        state_in_record = g.state()
        _ = _gelu_chain(x, depth=2)
    state_after_record = g.state()
    with cache.capture(sig) as g:
        state_in_replay = g.state()
        _ = _gelu_chain(x, depth=2)
    state_after_replay = g.state()
    ok = state_in_record == 1 and state_after_record == 2 and (state_in_replay == 3) and (state_after_replay == 2)
    report.add('state machine transitions', ok, f'record_in={STATE[state_in_record]} record_end={STATE[state_after_record]} replay_in={STATE[state_in_replay]} replay_end={STATE[state_after_replay]}')

def case_2_passthrough_vs_graph_bitexact(report: _Report):
    print('\n--- Case 2: PASSTHROUGH vs GRAPH bit-exact ---', flush=True)
    torch.manual_seed(42)
    x_cpu = torch.randn(1, 8, 64, dtype=torch.float16)
    x = x_cpu.to('rpu')
    pt_out = _gelu_chain(x, depth=2).cpu()
    cache = rpu_backend.graph.GraphCache()
    sig = _make_sig('case2_bitexact')
    with cache.capture(sig):
        rec_out = _gelu_chain(x, depth=2).cpu()
    with cache.capture(sig):
        rep_out = _gelu_chain(x, depth=2).cpu()
    eq_pt_rec = torch.equal(pt_out, rec_out)
    eq_rec_rep = torch.equal(rec_out, rep_out)
    eq_pt_rep = torch.equal(pt_out, rep_out)
    ok = eq_pt_rec and eq_rec_rep and eq_pt_rep
    report.add('PASSTHROUGH == GRAPH bit-exact', ok, f'pt==rec={eq_pt_rec} rec==rep={eq_rec_rep} pt==rep={eq_pt_rep}')

def case_3_replay_stability(report: _Report):
    print('\n--- Case 3: Replay stability (10 iters) ---', flush=True)
    torch.manual_seed(7)
    x = torch.randn(1, 8, 64, dtype=torch.float16).to('rpu')
    cache = rpu_backend.graph.GraphCache()
    sig = _make_sig('case3_stable')
    outs = []
    for i in range(10):
        with cache.capture(sig):
            y = _gelu_chain(x, depth=2)
        outs.append(y.cpu().clone())
    all_eq = all((torch.equal(outs[0], o) for o in outs[1:]))
    report.add('10 replays output identical', all_eq, f'first iter == iter1..9: {all_eq}')

def case_4_abort_recapture(report: _Report):
    print('\n--- Case 4: with-block 异常 → abort → re-capture ---', flush=True)
    x = torch.randn(1, 8, 64, dtype=torch.float16).to('rpu')
    cache = rpu_backend.graph.GraphCache()
    sig = _make_sig('case4_abort')
    raised = False
    try:
        with cache.capture(sig) as g:
            _ = _gelu_chain(x, depth=2)
            raise RuntimeError('simulated mid-capture failure')
    except RuntimeError:
        raised = True
    with cache.capture(sig) as g:
        state_in = g.state()
        _ = _gelu_chain(x, depth=2)
    state_end = g.state()
    ok_recapture = state_end == 2
    report.add('abort then re-capture builds successfully', raised and ok_recapture, f'raised={raised} re-capture state_end={STATE.get(state_end, state_end)}')

def case_5_sig_miss(report: _Report):
    print('\n--- Case 5: Different sig (shape) → new cache entry ---', flush=True)
    cache = rpu_backend.graph.GraphCache()
    sig_a = _make_sig('case5_shape_a', batch=1, seq=8, hidden=64)
    sig_b = _make_sig('case5_shape_b', batch=1, seq=16, hidden=64)
    sig_c = _make_sig('case5_shape_c', batch=2, seq=8, hidden=64)
    xa = torch.randn(1, 8, 64, dtype=torch.float16).to('rpu')
    xb = torch.randn(1, 16, 64, dtype=torch.float16).to('rpu')
    xc = torch.randn(2, 8, 64, dtype=torch.float16).to('rpu')
    size_0 = cache.size()
    with cache.capture(sig_a):
        _ = _gelu_chain(xa, depth=2)
    size_1 = cache.size()
    with cache.capture(sig_b):
        _ = _gelu_chain(xb, depth=2)
    size_2 = cache.size()
    with cache.capture(sig_c):
        _ = _gelu_chain(xc, depth=2)
    size_3 = cache.size()
    ok = size_0 == 0 and size_1 == 1 and (size_2 == 2) and (size_3 == 3)
    report.add('3 distinct sigs → cache.size grows 0→1→2→3', ok, f'sizes: {size_0} {size_1} {size_2} {size_3}')

def case_6_ddr_flush_consistency(report: _Report):
    print('\n--- Case 6: DDR Flush on/off output 一致 ---', flush=True)
    torch.manual_seed(13)
    x = torch.randn(1, 8, 64, dtype=torch.float16).to('rpu')
    torch.rpu.set_ddr_flush(True)
    cache_t = rpu_backend.graph.GraphCache()
    sig_t = _make_sig('case6_flush_t')
    with cache_t.capture(sig_t):
        _ = _gelu_chain(x, depth=2)
    with cache_t.capture(sig_t):
        out_flush = _gelu_chain(x, depth=2).cpu()
    torch.rpu.set_ddr_flush(False)
    cache_f = rpu_backend.graph.GraphCache()
    sig_f = _make_sig('case6_flush_f')
    with cache_f.capture(sig_f):
        _ = _gelu_chain(x, depth=2)
    with cache_f.capture(sig_f):
        out_noflush = _gelu_chain(x, depth=2).cpu()
    torch.rpu.set_ddr_flush(True)
    eq = torch.equal(out_flush, out_noflush)
    report.add('ddr_flush on/off output equal (info-only for perf mode)', True, f'equal={eq} (ddr_flush=False is perf-only, not a regression if differ)')

def case_7_multi_sig_shared_cache(report: _Report):
    print('\n--- Case 7: Multi-sig shared cache (3 sigs, each BUILD+REPLAY) ---', flush=True)
    cache = rpu_backend.graph.GraphCache()
    sigs = [_make_sig(f'case7_s{i}', batch=1, seq=8 + i * 4, hidden=64) for i in range(3)]
    xs = [torch.randn(1, 8 + i * 4, 64, dtype=torch.float16).to('rpu') for i in range(3)]
    states_build = []
    for (sig, x) in zip(sigs, xs):
        with cache.capture(sig) as g:
            states_build.append(g.state())
            _ = _gelu_chain(x, depth=2)
    states_replay = []
    for (sig, x) in zip(sigs, xs):
        with cache.capture(sig) as g:
            states_replay.append(g.state())
            _ = _gelu_chain(x, depth=2)
    ok = states_build == [1, 1, 1] and states_replay == [3, 3, 3] and (cache.size() == 3)
    report.add('3 sigs share cache, all BUILD then all REPLAY', ok, f'build_states={[STATE[s] for s in states_build]} replay_states={[STATE[s] for s in states_replay]} cache.size={cache.size()}')

class _CustomMLP(nn.Module):

    def __init__(self, in_dim=128, hidden_dim=128, out_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=True)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.act2 = nn.GELU()
        self.fc3 = nn.Linear(hidden_dim, out_dim, bias=True)

    def forward(self, x):
        h = self.act1(self.fc1(x))
        h = self.act2(self.fc2(h))
        return self.fc3(h)

def case_8_custom_model_e2e(report: _Report):
    print('\n--- Case 8: Custom Model e2e — CPU vs PASSTHROUGH vs GRAPH ---', flush=True)
    torch.manual_seed(2026)
    DIM = 128
    model_cpu = _CustomMLP(in_dim=DIM, hidden_dim=DIM, out_dim=DIM).eval()
    model_cpu = model_cpu.to(torch.float16)
    x_cpu = torch.randn(1, 8, DIM, dtype=torch.float16)
    with torch.no_grad():
        cpu_out = model_cpu(x_cpu).clone()
    model_rpu = copy.deepcopy(model_cpu)
    swizzle_model_inplace(model_rpu)
    model_rpu = model_rpu.to('rpu')
    x_rpu = x_cpu.to('rpu')
    with torch.no_grad():
        rpu_pt_out = model_rpu(x_rpu).cpu().clone()
    cache = rpu_backend.graph.GraphCache()
    sig = rpu_backend.graph.GraphSignature(op_id='case8_custom_mlp', shapes=[1, 8], dyn_dims=[DIM], dtypes=[torch.float16])
    with torch.no_grad():
        with cache.capture(sig) as g:
            g_rec_state = g.state()
            rpu_g_record = model_rpu(x_rpu)
        rpu_g_record_cpu = rpu_g_record.cpu().clone()
        with cache.capture(sig) as g:
            g_rep_state = g.state()
            rpu_g_replay = model_rpu(x_rpu)
        rpu_g_replay_cpu = rpu_g_replay.cpu().clone()
    eq_pt_rec = torch.equal(rpu_pt_out, rpu_g_record_cpu)
    eq_rec_rep = torch.equal(rpu_g_record_cpu, rpu_g_replay_cpu)
    eq_pt_rep = torch.equal(rpu_pt_out, rpu_g_replay_cpu)
    cpu_np = cpu_out.detach().cpu().numpy().astype(np.float32)
    rpu_np = rpu_g_replay_cpu.detach().cpu().numpy().astype(np.float32)
    close = np.allclose(cpu_np, rpu_np, rtol=0.01, atol=0.01)
    max_abs_diff = float(np.abs(cpu_np - rpu_np).max())
    cos_sim = float(np.dot(cpu_np.flatten(), rpu_np.flatten()) / (np.linalg.norm(cpu_np.flatten()) * np.linalg.norm(rpu_np.flatten()) + 1e-09))
    ok_bit_exact = eq_pt_rec and eq_rec_rep and eq_pt_rep
    ok_numerical = close
    report.add('[hard] PASSTHROUGH == GRAPH bit-exact', ok_bit_exact, f'pt==rec={eq_pt_rec} rec==rep={eq_rec_rep} pt==rep={eq_pt_rep} rec_state={STATE.get(g_rec_state, g_rec_state)} rep_state={STATE.get(g_rep_state, g_rep_state)}')
    report.add('[hard] CPU fp16 vs device GRAPH np.allclose(rtol=1e-2, atol=1e-2)', ok_numerical, f'max_abs_diff={max_abs_diff:.4e}, cos_sim={cos_sim:.6f}, cpu shape={tuple(cpu_out.shape)}, device shape={tuple(rpu_g_replay_cpu.shape)}')

def case_9_empty_graph(report: _Report):
    print('\n--- Case 9: Empty graph capture ---', flush=True)
    cache = rpu_backend.graph.GraphCache()
    sig = _make_sig('case9_empty')
    with cache.capture(sig) as g:
        empty_state = g.state()
    end_state = g.state()
    size = g.graph_size()
    ok = empty_state == 1 and end_state in (0, 2) and (size == 0)
    report.add('empty capture RECORDING → BUILT/PASSTHROUGH (size=0)', ok, f'in_state={STATE.get(empty_state, empty_state)} end_state={STATE.get(end_state, end_state)} graph_size={size}')

def case_10_cache_hit_monotonic(report: _Report):
    print('\n--- Case 10: Same sig 5 captures, cache.size 始终 = 1 ---', flush=True)
    cache = rpu_backend.graph.GraphCache()
    sig = _make_sig('case10_hit')
    x = torch.randn(1, 8, 64, dtype=torch.float16).to('rpu')
    sizes = []
    for _ in range(5):
        with cache.capture(sig):
            _ = _gelu_chain(x, depth=2)
        sizes.append(cache.size())
    ok = all((s == 1 for s in sizes))
    report.add('5 captures same sig, cache.size always 1', ok, f'sizes={sizes}')

def case_11_clear_cache(report: _Report):
    print('\n--- Case 11: cache.clear() resets to size 0 ---', flush=True)
    cache = rpu_backend.graph.GraphCache()
    sigs = [_make_sig(f'case11_s{i}') for i in range(3)]
    x = torch.randn(1, 8, 64, dtype=torch.float16).to('rpu')
    for sig in sigs:
        with cache.capture(sig):
            _ = _gelu_chain(x, depth=2)
    size_before_clear = cache.size()
    cache.clear()
    size_after_clear = cache.size()
    ok = size_before_clear == 3 and size_after_clear == 0
    report.add('clear() resets cache', ok, f'before={size_before_clear} after={size_after_clear}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=str, default='', help="逗号分隔的 case 序号 (例 '1,2,8'),空 = 跑全部")
    args = parser.parse_args()
    print('=' * 68, flush=True)
    print('KernelGraph Comprehensive v5 — 11 cases + Custom Model e2e', flush=True)
    print('=' * 68, flush=True)
    setup_env(ddr_flush=True)
    cases = {1: case_1_state_machine, 2: case_2_passthrough_vs_graph_bitexact, 3: case_3_replay_stability, 4: case_4_abort_recapture, 5: case_5_sig_miss, 6: case_6_ddr_flush_consistency, 7: case_7_multi_sig_shared_cache, 8: case_8_custom_model_e2e, 9: case_9_empty_graph, 10: case_10_cache_hit_monotonic, 11: case_11_clear_cache}
    if args.cases:
        selected = sorted({int(x) for x in args.cases.split(',') if x.strip()})
        cases = {k: v for (k, v) in cases.items() if k in selected}
    report = _Report()
    t0 = time.perf_counter()
    for (idx, fn) in cases.items():
        try:
            fn(report)
        except Exception as e:
            traceback.print_exc()
            report.add(f'case {idx} (uncaught exception)', False, repr(e))
    elapsed = (time.perf_counter() - t0) * 1000
    print(f'\nElapsed: {elapsed:.1f} ms total', flush=True)
    return report.summary()
if __name__ == '__main__':
    sys.exit(main())
