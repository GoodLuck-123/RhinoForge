from __future__ import annotations
import argparse
import sys
import time
import traceback
import torch
import rpu_backend
STATE = {0: 'PASSTHROUGH', 1: 'RECORDING', 2: 'BUILT', 3: 'REPLAYING'}

def setup_env() -> None:
    torch.rpu.set_caching_allocator(True)
    torch.rpu.set_ddr_flush(True)
    torch.rpu.set_debug(False)
    torch.rpu.set_profile(False)

def make_sig(op_id: str, shape: tuple[int, ...]) -> rpu_backend.graph.GraphSignature:
    return rpu_backend.graph.GraphSignature(op_id=op_id, shapes=list(shape[:2]), dyn_dims=[shape[-1]], dtypes=[torch.float16])

def graph1_rpu(x: torch.Tensor, depth: int) -> torch.Tensor:
    y = x
    for _ in range(depth):
        y = torch.nn.functional.gelu(y)
    return y

def cpu_barrier_op(x_cpu: torch.Tensor) -> torch.Tensor:
    return (x_cpu.to(torch.float32) * 0.75 + 0.125).to(torch.float16).contiguous()

def graph2_rpu(x: torch.Tensor, depth: int) -> torch.Tensor:
    y = x
    for _ in range(depth):
        y = torch.nn.functional.gelu(y)
    return y

def passthrough_once(x: torch.Tensor, d1: int, d2: int) -> torch.Tensor:
    y_rpu = graph1_rpu(x, d1)
    y_cpu = y_rpu.clone().cpu()
    z_cpu = cpu_barrier_op(y_cpu)
    z_rpu = z_cpu.to('rpu')
    out_rpu = graph2_rpu(z_rpu, d2)
    return out_rpu.clone().cpu()

def split_graph_once(x: torch.Tensor, d1: int, d2: int, cache1: rpu_backend.graph.GraphCache, cache2: rpu_backend.graph.GraphCache, sig1: rpu_backend.graph.GraphSignature, sig2: rpu_backend.graph.GraphSignature) -> tuple[torch.Tensor, dict]:
    info: dict = {}
    with cache1.capture(sig1) as g1:
        info['g1_in'] = g1.state()
        y_rpu = graph1_rpu(x, d1)
    info['g1_end'] = g1.state()
    info['g1_replayable'] = g1.replayable()
    info['g1_size'] = g1.graph_size()
    y_cpu = y_rpu.clone().cpu()
    z_cpu = cpu_barrier_op(y_cpu)
    z_rpu = z_cpu.to('rpu')
    with cache2.capture(sig2) as g2:
        info['g2_in'] = g2.state()
        out_rpu = graph2_rpu(z_rpu, d2)
    info['g2_end'] = g2.state()
    info['g2_replayable'] = g2.replayable()
    info['g2_size'] = g2.graph_size()
    return (out_rpu.clone().cpu(), info)

def _sample(t: torch.Tensor) -> list[float]:
    return t[0, 0, :4].float().tolist()

def _snapshot(cache: rpu_backend.graph.GraphCache) -> dict:
    snap = cache.snapshot()
    return {'size': cache.size(), 'replay': sum((e.replay_count for e in snap)), 'recapture': sum((e.recapture_count for e in snap)), 'non_replayable': [e.non_replayable_reason for e in snap if e.non_replayable_reason], 'entries': snap}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape', default='1,8,64', help='comma-separated device input shape')
    ap.add_argument('--depth1', type=int, default=2)
    ap.add_argument('--depth2', type=int, default=2)
    args = ap.parse_args()
    shape = tuple((int(x) for x in args.shape.split(',')))
    if len(shape) < 2:
        raise ValueError('--shape must have at least 2 dims')
    print('=' * 72, flush=True)
    print('Manual Graph1 -> CPU materialization -> Graph2 barrier', flush=True)
    print(f'  shape={shape} depth1={args.depth1} depth2={args.depth2}', flush=True)
    print('=' * 72, flush=True)
    setup_env()
    torch.manual_seed(12345)
    x_cpu = torch.linspace(-2.0, 2.0, steps=int(torch.tensor(shape).prod()), dtype=torch.float16).reshape(shape)
    x = x_cpu.to('rpu')
    print('\n--- PASSTHROUGH baseline ---', flush=True)
    t0 = time.time()
    out_pt1 = passthrough_once(x, args.depth1, args.depth2)
    out_pt2 = passthrough_once(x, args.depth1, args.depth2)
    print(f'  elapsed={time.time() - t0:.3f}s sample={_sample(out_pt1)}', flush=True)
    pt_stable = torch.equal(out_pt1, out_pt2)
    print(f'  baseline byte-exact={pt_stable}', flush=True)
    cache1 = rpu_backend.graph.GraphCache()
    cache2 = rpu_backend.graph.GraphCache()
    sig1 = make_sig('manual_barrier_graph1', shape)
    sig2 = make_sig('manual_barrier_graph2', shape)
    print('\n--- Split graph run 1 (record) ---', flush=True)
    t1 = time.time()
    (out_g1, info1) = split_graph_once(x, args.depth1, args.depth2, cache1, cache2, sig1, sig2)
    print(f'  elapsed={time.time() - t1:.3f}s sample={_sample(out_g1)}', flush=True)
    print(f"  g1 {STATE.get(info1['g1_in'])}->{STATE.get(info1['g1_end'])} size={info1['g1_size']} replayable={info1['g1_replayable']}", flush=True)
    print(f"  g2 {STATE.get(info1['g2_in'])}->{STATE.get(info1['g2_end'])} size={info1['g2_size']} replayable={info1['g2_replayable']}", flush=True)
    print('\n--- Split graph run 2 (replay) ---', flush=True)
    t2 = time.time()
    (out_g2, info2) = split_graph_once(x, args.depth1, args.depth2, cache1, cache2, sig1, sig2)
    print(f'  elapsed={time.time() - t2:.3f}s sample={_sample(out_g2)}', flush=True)
    print(f"  g1 {STATE.get(info2['g1_in'])}->{STATE.get(info2['g1_end'])} size={info2['g1_size']} replayable={info2['g1_replayable']}", flush=True)
    print(f"  g2 {STATE.get(info2['g2_in'])}->{STATE.get(info2['g2_end'])} size={info2['g2_size']} replayable={info2['g2_replayable']}", flush=True)
    snap1 = _snapshot(cache1)
    snap2 = _snapshot(cache2)
    print('\n--- snapshots ---', flush=True)
    print(f"  graph1: size={snap1['size']} replay={snap1['replay']} recapture={snap1['recapture']} non_replayable={snap1['non_replayable']}", flush=True)
    print(f"  graph2: size={snap2['size']} replay={snap2['replay']} recapture={snap2['recapture']} non_replayable={snap2['non_replayable']}", flush=True)
    chk = lambda b, label: print(f"  [{('PASS' if b else 'FAIL')}] {label}", flush=True)
    print('\n' + '=' * 72, flush=True)
    chk(pt_stable, 'PASSTHROUGH baseline byte-exact')
    chk(torch.equal(out_pt1, out_g1), 'PASSTHROUGH == split graph run 1 byte-exact')
    chk(torch.equal(out_g1, out_g2), 'split graph run 1 == run 2 byte-exact')
    chk(info1['g1_in'] == 1 and info1['g2_in'] == 1, 'run 1 records both graphs')
    chk(info2['g1_in'] == 3 and info2['g2_in'] == 3, 'run 2 replays both graphs')
    chk(info1['g1_replayable'] and info1['g2_replayable'], 'both graph entries replayable')
    chk(not snap1['non_replayable'] and (not snap2['non_replayable']), 'no non_replayable entries')
    ok = pt_stable and torch.equal(out_pt1, out_g1) and torch.equal(out_g1, out_g2) and (info1['g1_in'] == 1) and (info1['g2_in'] == 1) and (info2['g1_in'] == 3) and (info2['g2_in'] == 3) and info1['g1_replayable'] and info1['g2_replayable'] and (not snap1['non_replayable']) and (not snap2['non_replayable'])
    print('=' * 72, flush=True)
    print('PASS' if ok else 'FAIL', flush=True)
    return 0 if ok else 1
if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
