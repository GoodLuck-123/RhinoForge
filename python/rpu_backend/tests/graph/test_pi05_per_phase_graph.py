from __future__ import annotations
import argparse
import os
import sys
import time
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
VLM_CHUNK_SIZE = 160

def _apply_pi05_runtime_switches(policy):
    policy._lerobot_policy.model._rpu_noise_seed = SEED
    vlm = policy._lerobot_policy.model.paligemma_with_expert.paligemma.model.language_model
    vlm._rpu_chunk_size = VLM_CHUNK_SIZE
    vlm._rpu_spm_mode = False
    print(f'  [setup] _rpu_noise_seed={SEED} vlm._rpu_chunk_size={vlm._rpu_chunk_size} vlm._rpu_spm_mode={vlm._rpu_spm_mode}', flush=True)

def _build_dummy_batch():
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    return {'observation.images.image': torch.zeros(1, 3, 224, 224, dtype=torch.float32), 'observation.images.wrist_image': torch.zeros(1, 3, 224, 224, dtype=torch.float32), 'observation.state': torch.zeros(1, 8, dtype=torch.float32), OBS_LANGUAGE_TOKENS: torch.zeros(1, 32, dtype=torch.int64), OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 32, dtype=torch.bool), 'task': ['pick up the object']}

def load_policy():
    from rpu_backend.api import Pi05Policy
    for name in ('pi05-libero-finetuned', 'pi05-base'):
        p = model_path(name)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'model.safetensors')):
            t0 = time.time()
            policy = Pi05Policy.from_pretrained(str(p), dtype=torch.float16)
            print(f'  loaded in {time.time() - t0:.1f}s', flush=True)
            t1 = time.time()
            policy.to('rpu')
            print(f"  to('rpu') in {time.time() - t1:.1f}s", flush=True)
            return policy
    raise FileNotFoundError('no pi0.5 model available')

def _install_per_phase_wraps(policy):
    vt = policy._lerobot_policy.model.paligemma_with_expert.paligemma.model.vision_tower
    pwe = policy._lerobot_policy.model.paligemma_with_expert
    cache_siglip = rpu_backend.graph.GraphCache()
    cache_pwe = rpu_backend.graph.GraphCache()
    counters = {'vt': 0, 'pwe': 0}
    state_log = []
    orig_vt_forward = vt.forward
    orig_pwe_forward = pwe.forward

    def vt_wrapped(*args, **kwargs):
        counters['vt'] += 1
        sig = rpu_backend.graph.GraphSignature(op_id='pi05_siglip_encode', shapes=[1, 3, 224, 224], dyn_dims=[counters['vt']], dtypes=[torch.float16])
        with cache_siglip.capture(sig) as g:
            in_state = g.state()
            out = orig_vt_forward(*args, **kwargs)
        state_log.append(('siglip', counters['vt'], in_state, g.replayable(), g.graph_size()))
        return out

    def pwe_wrapped(*args, **kwargs):
        counters['pwe'] += 1
        if counters['pwe'] == 1:
            op_id = 'pi05_vlm_prefill'
            iter_id = 1
        else:
            op_id = 'pi05_denoise'
            iter_id = counters['pwe'] - 1
        sig = rpu_backend.graph.GraphSignature(op_id=op_id, shapes=[1, 1], dyn_dims=[iter_id], dtypes=[torch.float16])
        with cache_pwe.capture(sig) as g:
            in_state = g.state()
            out = orig_pwe_forward(*args, **kwargs)
        state_log.append((op_id, counters['pwe'], in_state, g.replayable(), g.graph_size()))
        return out
    vt.forward = vt_wrapped
    pwe.forward = pwe_wrapped

    def restore():
        vt.forward = orig_vt_forward
        pwe.forward = orig_pwe_forward
    return (cache_siglip, cache_pwe, restore, counters, state_log)

def _forward_chunk(policy, batch, num_steps):
    with torch.no_grad():
        return policy._forward_rpu(batch, return_chunk=True, num_steps=num_steps)

def _measured(label, policy, batch, num_steps):
    torch.ops.rpu.spm_alloc_reset_temporary()
    t = time.time()
    dev = _forward_chunk(policy, batch, num_steps)
    dt = time.time() - t
    cpu = dev.clone().cpu()
    print(f'  [{label}] forward: {dt:.2f}s sample[0,0,:4]={cpu[0, 0, :4].float().tolist()}', flush=True)
    return cpu

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--num-steps', type=int, default=2)
    args = ap.parse_args()
    print('=' * 60, flush=True)
    print('Pi0.5 per-phase graph wrap test', flush=True)
    print(f'  num_steps={args.num_steps}', flush=True)
    print('=' * 60, flush=True)
    setup_env()
    torch.manual_seed(SEED)
    policy = load_policy()
    _apply_pi05_runtime_switches(policy)
    batch = _build_dummy_batch()
    print('\n--- Warmup ---', flush=True)
    _ = _forward_chunk(policy, batch, args.num_steps)
    del _
    torch.ops.rpu.spm_alloc_reset_temporary()
    print('\n--- Run 1a/1b: PASSTHROUGH baseline ---', flush=True)
    out_pt_a = _measured('PT-1a', policy, batch, args.num_steps)
    out_pt_b = _measured('PT-1b', policy, batch, args.num_steps)
    pt_stable = torch.equal(out_pt_a, out_pt_b)
    print(f'  Run 1a == Run 1b: {pt_stable}', flush=True)
    print('\n--- Installing per-phase graph wraps ---', flush=True)
    (cs, cp, restore, counters, state_log) = _install_per_phase_wraps(policy)
    try:
        print('\n--- Run 2: per-phase graph (capture cold) ---', flush=True)
        out_g1 = _measured('PHASE-2', policy, batch, args.num_steps)
        print(f'  counters: {counters}', flush=True)
        print('\n--- Run 3: per-phase graph (potential replay) ---', flush=True)
        counters['vt'] = 0
        counters['pwe'] = 0
        out_g2 = _measured('PHASE-3', policy, batch, args.num_steps)
        print(f'  counters: {counters}', flush=True)
    finally:
        restore()
    print('\n--- Per-phase graph state log ---', flush=True)
    STATE = {0: 'PASSTHROUGH', 1: 'RECORDING', 2: 'BUILT', 3: 'REPLAYING'}
    for (tag, idx, in_state, rep, size) in state_log:
        print(f'  {tag} call#{idx}: in_state={STATE.get(in_state)} replayable={rep} graph_size={size}', flush=True)
    snap_s = cs.snapshot()
    snap_p = cp.snapshot()
    print(f'\n  cache_siglip entries: {len(snap_s)}', flush=True)
    for e in snap_s:
        print(f'    kernel={e.kernel_count} seg={e.segment_count} reason={e.non_replayable_reason!r}', flush=True)
    print(f'  cache_pwe entries: {len(snap_p)}', flush=True)
    for e in snap_p:
        print(f'    kernel={e.kernel_count} seg={e.segment_count} reason={e.non_replayable_reason!r}', flush=True)
    chk = lambda b, label: print(f"  [{('PASS' if b else 'FAIL')}] {label}", flush=True)
    print('\n' + '=' * 60, flush=True)
    chk(pt_stable, 'PASSTHROUGH baseline byte-exact')
    chk(torch.equal(out_pt_a, out_g1), 'Run 1 == Run 2 (PASSTHROUGH == per-phase graph) byte-exact')
    chk(torch.equal(out_g1, out_g2), 'Run 2 == Run 3 (per-phase graph self-consistent) byte-exact')
    chk(torch.equal(out_pt_a, out_g2), 'Run 1 == Run 3 (PASSTHROUGH == per-phase graph) byte-exact')
    all_ok = pt_stable and torch.equal(out_pt_a, out_g1) and torch.equal(out_g1, out_g2) and torch.equal(out_pt_a, out_g2)
    print('=' * 60, flush=True)
    print('PASS' if all_ok else 'FAIL', flush=True)
    return 0 if all_ok else 1
if __name__ == '__main__':
    sys.exit(main())
