from __future__ import annotations
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
SEED = 12345
VLM_CHUNK_SIZE = 160
NUM_STEPS = 2

def setup_env():
    torch.rpu.set_caching_allocator(True)
    torch.rpu.set_ddr_flush(True)
    torch.rpu.set_debug(False)
    torch.rpu.set_profile(False)

def _apply_pi05_runtime_switches(policy):
    policy._lerobot_policy.model._rpu_noise_seed = SEED
    vlm = policy._lerobot_policy.model.paligemma_with_expert.paligemma.model.language_model
    vlm._rpu_chunk_size = VLM_CHUNK_SIZE
    vlm._rpu_spm_mode = False

def _build_dummy_batch():
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    return {'observation.images.image': torch.zeros(1, 3, 224, 224, dtype=torch.float32), 'observation.images.wrist_image': torch.zeros(1, 3, 224, 224, dtype=torch.float32), 'observation.state': torch.zeros(1, 8, dtype=torch.float32), OBS_LANGUAGE_TOKENS: torch.zeros(1, 32, dtype=torch.int64), OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 32, dtype=torch.bool), 'task': ['pick up the object']}

def load_policy():
    from rpu_backend.api import Pi05Policy
    p = model_path('pi05-libero-finetuned')
    t0 = time.time()
    policy = Pi05Policy.from_pretrained(str(p), dtype=torch.float16).to('rpu')
    print(f'  loaded + to(rpu) in {time.time() - t0:.1f}s', flush=True)
    return policy

def _forward(policy, batch):
    with torch.no_grad():
        return policy._forward_rpu(batch, return_chunk=True, num_steps=NUM_STEPS)

def _measured(label, policy, batch):
    torch.ops.rpu.spm_alloc_reset_temporary()
    t = time.time()
    dev = _forward(policy, batch)
    dt = time.time() - t
    cpu = dev.clone().cpu()
    print(f'  [{label}] {dt:.2f}s sample[0,0,:4]={cpu[0, 0, :4].float().tolist()}', flush=True)
    return cpu

def _wrap_vt_only(policy):
    vt = policy._lerobot_policy.model.paligemma_with_expert.paligemma.model.vision_tower
    cache = rpu_backend.graph.GraphCache()
    counter = {'n': 0}
    orig = vt.forward

    def wrapped(*a, **kw):
        counter['n'] += 1
        sig = rpu_backend.graph.GraphSignature(op_id='vt_only_siglip', shapes=[1, 3, 224, 224], dyn_dims=[counter['n']], dtypes=[torch.float16])
        with cache.capture(sig):
            return orig(*a, **kw)
    vt.forward = wrapped

    def restore():
        vt.forward = orig
    return (restore, counter)

def _wrap_pwe_only(policy):
    pwe = policy._lerobot_policy.model.paligemma_with_expert
    cache = rpu_backend.graph.GraphCache()
    counter = {'n': 0}
    orig = pwe.forward

    def wrapped(*a, **kw):
        counter['n'] += 1
        op = 'pwe_only_prefill' if counter['n'] == 1 else 'pwe_only_denoise'
        sig = rpu_backend.graph.GraphSignature(op_id=op, shapes=[1, 1], dyn_dims=[counter['n']], dtypes=[torch.float16])
        with cache.capture(sig):
            return orig(*a, **kw)
    pwe.forward = wrapped

    def restore():
        pwe.forward = orig
    return (restore, counter)

def main():
    setup_env()
    torch.manual_seed(SEED)
    print('=' * 60, flush=True)
    print(f'Pi0.5 二分定位 (SEED={SEED}, num_steps={NUM_STEPS})', flush=True)
    print('=' * 60, flush=True)
    policy = load_policy()
    _apply_pi05_runtime_switches(policy)
    batch = _build_dummy_batch()
    print('\n--- Warmup ---', flush=True)
    _ = _forward(policy, batch)
    del _
    torch.ops.rpu.spm_alloc_reset_temporary()
    print('\n--- C: full eager baseline ---', flush=True)
    out_c1 = _measured('C-eager-1', policy, batch)
    out_c2 = _measured('C-eager-2', policy, batch)
    c_stable = torch.equal(out_c1, out_c2)
    print(f'  eager stable: {c_stable}', flush=True)
    print('\n--- A: wrap vt only (pwe eager) ---', flush=True)
    (restore_a, ca) = _wrap_vt_only(policy)
    try:
        out_a = _measured('A-vt-only', policy, batch)
        print(f"  vt calls: {ca['n']}", flush=True)
    finally:
        restore_a()
    print('\n--- B: wrap pwe only (vt eager) ---', flush=True)
    (restore_b, cb) = _wrap_pwe_only(policy)
    try:
        out_b = _measured('B-pwe-only', policy, batch)
        print(f"  pwe calls: {cb['n']}", flush=True)
    finally:
        restore_b()
    chk = lambda b, label: print(f"  [{('PASS' if b else 'FAIL')}] {label}", flush=True)
    print('\n' + '=' * 60, flush=True)
    chk(c_stable, 'C: eager baseline byte-exact (C1 == C2)')
    chk(torch.equal(out_c1, out_a), 'A == C: vt graph wrap matches eager → SigLIP graph 干净')
    chk(torch.equal(out_c1, out_b), 'B == C: pwe graph wrap matches eager → pwe (Gemma/AdaRMS/denoise) graph 干净')
    print('=' * 60, flush=True)
    if torch.equal(out_c1, out_a) and (not torch.equal(out_c1, out_b)):
        print('DIAGNOSIS: 问题在 pwe (Gemma prefill / AdaRMS / denoise loop)', flush=True)
    elif not torch.equal(out_c1, out_a) and torch.equal(out_c1, out_b):
        print('DIAGNOSIS: 问题在 vt (SigLIP 多次调用)', flush=True)
    elif not torch.equal(out_c1, out_a) and (not torch.equal(out_c1, out_b)):
        print('DIAGNOSIS: 两边都有问题 (或更深层 root cause)', flush=True)
    else:
        print('DIAGNOSIS: 两边都干净 — 问题只在 wrap 全部时出现 (跨子图 interaction)', flush=True)
    return 0
if __name__ == '__main__':
    sys.exit(main())
