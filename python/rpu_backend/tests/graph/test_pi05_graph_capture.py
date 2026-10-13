from __future__ import annotations
import argparse
import os
import sys
import time
import traceback
import types
import torch
import rpu_backend
from rpu_backend.model_registry import model_path

def _stub_lerobot_transformers_check():
    import transformers.models.siglip as _sl
    stub = types.ModuleType('transformers.models.siglip.check')
    stub.check_whether_transformers_replace_is_installed_correctly = lambda : True
    sys.modules['transformers.models.siglip.check'] = stub
    _sl.check = stub
_stub_lerobot_transformers_check()

def setup_env():
    torch.rpu.set_caching_allocator(True)
    torch.rpu.set_ddr_flush(True)
    torch.rpu.set_debug(False)
    torch.rpu.set_profile(False)
SEED = 12345
VLM_CHUNK_SIZE_CORRECTNESS = 160

def _apply_pi05_runtime_switches(policy):
    policy._lerobot_policy.model._rpu_noise_seed = SEED
    vlm = policy._lerobot_policy.model.paligemma_with_expert.paligemma.model.language_model
    vlm._rpu_chunk_size = VLM_CHUNK_SIZE_CORRECTNESS
    vlm._rpu_spm_mode = False
    print(f'  [setup] _rpu_noise_seed={SEED}, vlm._rpu_chunk_size={vlm._rpu_chunk_size}, vlm._rpu_spm_mode={vlm._rpu_spm_mode}', flush=True)

def _build_dummy_batch():
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    return {'observation.images.image': torch.zeros(1, 3, 224, 224, dtype=torch.float32), 'observation.images.wrist_image': torch.zeros(1, 3, 224, 224, dtype=torch.float32), 'observation.state': torch.zeros(1, 8, dtype=torch.float32), OBS_LANGUAGE_TOKENS: torch.zeros(1, 32, dtype=torch.int64), OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 32, dtype=torch.bool), 'task': ['pick up the object']}

def load_policy():
    from rpu_backend.api import Pi05Policy
    for name in ('pi05-libero-finetuned', 'pi05-base'):
        p = model_path(name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'model.safetensors')):
            print(f'  loading {name} from {p} ...', flush=True)
            t0 = time.time()
            policy = Pi05Policy.from_pretrained(str(p), dtype=torch.float16)
            print(f'  loaded in {time.time() - t0:.1f}s', flush=True)
            print("  .to('rpu') ...", flush=True)
            t1 = time.time()
            policy.to('rpu')
            print(f"  to('rpu') in {time.time() - t1:.1f}s", flush=True)
            return policy
    raise FileNotFoundError('Neither pi05-libero-finetuned nor pi05-base has model.safetensors.')

def _make_sig(num_steps: int) -> 'rpu_backend.graph.GraphSignature':
    return rpu_backend.graph.GraphSignature(op_id='pi05_predict_action_chunk', shapes=[1, 32], dyn_dims=[num_steps], dtypes=[torch.float16])

def _forward_chunk(policy, batch, num_steps):
    with torch.no_grad():
        return policy._forward_rpu(batch, return_chunk=True, num_steps=num_steps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-steps', type=int, default=2, help='denoise steps (low-power default: 2; full default: 10)')
    args = ap.parse_args()
    print('=' * 60, flush=True)
    print('Pi0.5 PASSTHROUGH vs GRAPH (byte-exact, low-power)', flush=True)
    print(f'  num_steps={args.num_steps}', flush=True)
    print('=' * 60, flush=True)
    setup_env()
    torch.manual_seed(SEED)
    policy = load_policy()
    _apply_pi05_runtime_switches(policy)
    batch = _build_dummy_batch()
    sig = _make_sig(args.num_steps)
    print('\n--- Warmup (1 forward + spm reset) ---', flush=True)
    t0 = time.time()
    _ = _forward_chunk(policy, batch, args.num_steps)
    del _
    torch.ops.rpu.spm_alloc_reset_temporary()
    print(f'  warmup: {time.time() - t0:.2f}s', flush=True)

    def _measured_forward(label: str):
        torch.ops.rpu.spm_alloc_reset_temporary()
        t = time.time()
        dev = _forward_chunk(policy, batch, args.num_steps)
        dt = time.time() - t
        cpu = dev.clone().cpu()
        print(f'  [{label}] forward: {dt:.2f}s sample[0,0,:4]={cpu[0, 0, :4].float().tolist()}', flush=True)
        return cpu
    print('\n--- Run 1: PASSTHROUGH (no graph) ---', flush=True)
    out_pt = _measured_forward('PASSTHROUGH-1a')
    print(f'  shape={list(out_pt.shape)} dtype={out_pt.dtype}', flush=True)
    print('\n--- Run 1b: PASSTHROUGH again (baseline determinism check) ---', flush=True)
    out_pt_b = _measured_forward('PASSTHROUGH-1b')
    pt_byte_exact = torch.equal(out_pt, out_pt_b)
    print(f'  Run 1 == Run 1b (byte-exact): {pt_byte_exact}', flush=True)
    print('\n--- Run 2: GRAPH RECORDING ---', flush=True)
    cache = rpu_backend.graph.GraphCache()
    torch.ops.rpu.spm_alloc_reset_temporary()
    t0 = time.time()
    with cache.capture(sig) as g:
        in2 = g.state()
        out_g1_dev = _forward_chunk(policy, batch, args.num_steps)
    end2 = g.state()
    rep2 = g.replayable()
    size2 = g.graph_size()
    out_g1 = out_g1_dev.clone().cpu()
    print(f'  forward: {time.time() - t0:.2f}s', flush=True)
    print(f'  in_state={in2} after_end={end2} graph_size={size2} replayable={rep2}', flush=True)
    print(f'  sample[0,0,:4]={out_g1[0, 0, :4].float().tolist()}', flush=True)
    print('\n--- Run 3: GRAPH REPLAYING (same sig) ---', flush=True)
    torch.ops.rpu.spm_alloc_reset_temporary()
    t0 = time.time()
    with cache.capture(sig) as g:
        in3 = g.state()
        out_g2_dev = _forward_chunk(policy, batch, args.num_steps)
    end3 = g.state()
    rep3 = g.replayable()
    size3 = g.graph_size()
    out_g2 = out_g2_dev.clone().cpu()
    print(f'  forward: {time.time() - t0:.2f}s', flush=True)
    print(f'  in_state={in3} after_end={end3} graph_size={size3} replayable={rep3}', flush=True)
    print(f'  sample[0,0,:4]={out_g2[0, 0, :4].float().tolist()}', flush=True)
    snap = cache.snapshot()
    print('\n--- cache snapshot ---', flush=True)
    for (i, e) in enumerate(snap):
        print(f'  entry #{i}: kernel={e.kernel_count} seg={e.segment_count} replay={e.replay_count} recapture={e.recapture_count}', flush=True)
        if e.non_replayable_reason:
            print(f'    !!! non_replayable: {e.non_replayable_reason!r}', flush=True)
    STATE = {0: 'PASSTHROUGH', 1: 'RECORDING', 2: 'BUILT', 3: 'REPLAYING'}
    chk = lambda b, label: print(f"  [{('PASS' if b else 'FAIL')}] {label}", flush=True)
    print('\n' + '=' * 60, flush=True)
    chk(pt_byte_exact, 'Run 1 == Run 1b (PASSTHROUGH baseline determinism)')
    chk(in2 == 1, f'Run 2 in_scope=RECORDING (got {STATE.get(in2)})')
    chk(not rep2, f'Run 2 replayable=False (支线 S: got {rep2})')
    chk(in3 == 1, f'Run 3 in_scope=RECORDING (支线 S: got {STATE.get(in3)})')
    chk(size3 == size2, f'Run 3 graph_size {size3} == Run 2 {size2} (Q2: sync_point 在每个子系统边界 flush,残留视 num_steps 而定)')
    chk(torch.equal(out_pt, out_g1), 'Run 1 == Run 2 (PASSTHROUGH == RECORDING+sync) byte-exact')
    chk(torch.equal(out_g1, out_g2), 'Run 2 == Run 3 (RECORDING+sync determinism) byte-exact')
    chk(torch.equal(out_pt, out_g2), 'Run 1 == Run 3 (PASSTHROUGH == RECORDING+sync) byte-exact')
    all_ok = pt_byte_exact and in2 == 1 and (not rep2) and (in3 == 1) and (size3 == size2) and torch.equal(out_pt, out_g1) and torch.equal(out_g1, out_g2) and torch.equal(out_pt, out_g2)
    print('=' * 60, flush=True)
    print('PASS' if all_ok else 'FAIL', flush=True)
    return 0 if all_ok else 1
if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
