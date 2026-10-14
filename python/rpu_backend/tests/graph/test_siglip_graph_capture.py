from __future__ import annotations
import sys
import traceback
import torch
import rpu_backend
from rpu_backend.model_registry import model_path
from transformers import PaliGemmaForConditionalGeneration, AutoConfig

def setup_env():
    torch.rpu.set_caching_allocator(True)
    torch.rpu.set_ddr_flush(True)
    torch.rpu.set_debug(False)
    torch.rpu.set_profile(False)

def load_siglip_vision():
    repo = str(model_path('paligemma2-3b-pt-224'))
    cfg = AutoConfig.from_pretrained(repo)
    cfg.text_config.use_bidirectional_attention = True
    hf = PaliGemmaForConditionalGeneration.from_pretrained(repo, config=cfg, torch_dtype=torch.float16, device_map='cpu', low_cpu_mem_usage=True)
    hf = hf.half()
    hf.eval()
    hf._paligemma_source_name_or_path = repo
    from rpu_backend.tests.fixtures.paligemma_siglip import patch_siglip_from_paligemma_checkpoint
    return patch_siglip_from_paligemma_checkpoint(hf)

def main():
    setup_env()
    vision = load_siglip_vision()
    torch.manual_seed(0)
    pixel_values = torch.randn(1, 3, 224, 224, dtype=torch.float16, device='rpu')
    print('=' * 60, flush=True)
    print('SigLIP PASSTHROUGH vs GRAPH (byte-exact)', flush=True)
    print('=' * 60, flush=True)
    print('\n--- Run 1: PASSTHROUGH (no graph) ---', flush=True)
    with torch.no_grad():
        out_pt_dev = vision(pixel_values=pixel_values).last_hidden_state
    out_pt = out_pt_dev.clone().cpu()
    print(f'  shape={out_pt.shape} sample[0,0,:4]={out_pt[0, 0, :4].tolist()}', flush=True)
    print('\n--- Run 2: GRAPH RECORDING ---', flush=True)
    cache = rpu_backend.graph.GraphCache()
    sig = rpu_backend.graph.GraphSignature(op_id='siglip_e2e', shapes=[1, 3, 224, 224], dyn_dims=[1152], dtypes=[torch.float16])
    with cache.capture(sig) as g:
        in2 = g.state()
        with torch.no_grad():
            out_g1_dev = vision(pixel_values=pixel_values).last_hidden_state
    end2 = g.state()
    rep2 = g.replayable()
    size2 = g.graph_size()
    out_g1 = out_g1_dev.clone().cpu()
    print(f'  in_state={in2} after_end={end2} graph_size={size2} replayable={rep2}', flush=True)
    print(f'  sample[0,0,:4]={out_g1[0, 0, :4].tolist()}', flush=True)
    print('\n--- Run 3: GRAPH REPLAYING (same sig) ---', flush=True)
    with cache.capture(sig) as g:
        in3 = g.state()
        with torch.no_grad():
            out_g2_dev = vision(pixel_values=pixel_values).last_hidden_state
    end3 = g.state()
    rep3 = g.replayable()
    size3 = g.graph_size()
    out_g2 = out_g2_dev.clone().cpu()
    print(f'  in_state={in3} after_end={end3} graph_size={size3} replayable={rep3}', flush=True)
    print(f'  sample[0,0,:4]={out_g2[0, 0, :4].tolist()}', flush=True)
    snap = cache.snapshot()
    print('\n--- cache snapshot ---', flush=True)
    for (i, e) in enumerate(snap):
        print(f'  entry #{i}: kernel={e.kernel_count} seg={e.segment_count} replay={e.replay_count} recapture={e.recapture_count}', flush=True)
        if e.non_replayable_reason:
            print(f'    !!! non_replayable: {e.non_replayable_reason!r}', flush=True)
    STATE = {0: 'PASSTHROUGH', 1: 'RECORDING', 2: 'BUILT', 3: 'REPLAYING'}
    chk = lambda b, label: print(f"  [{('PASS' if b else 'FAIL')}] {label}", flush=True)
    print('\n' + '=' * 60, flush=True)
    chk(in2 == 1, f'Run 2 in_scope=RECORDING (got {STATE.get(in2)})')
    chk(end2 == 2, f'Run 2 after_end=BUILT (got {STATE.get(end2)})')
    chk(rep2, f'Run 2 replayable=True (got {rep2})')
    chk(size2 > 0, f'Run 2 graph_size > 0 (got {size2})')
    chk(in3 == 3, f'Run 3 in_scope=REPLAYING (got {STATE.get(in3)})')
    chk(end3 == 2, f'Run 3 after_end=BUILT (got {STATE.get(end3)})')
    chk(rep3, f'Run 3 replayable=True (got {rep3})')
    chk(size3 == size2, f'Run 3 graph_size {size3} == Run 2 {size2}')
    chk(torch.equal(out_pt, out_g1), 'Run 1 == Run 2 (PASSTHROUGH == RECORDING) bit-exact')
    chk(torch.equal(out_g1, out_g2), 'Run 2 == Run 3 (RECORDING == REPLAYING) bit-exact')
    chk(torch.equal(out_pt, out_g2), 'Run 1 == Run 3 (PASSTHROUGH == REPLAYING) bit-exact')
    all_ok = in2 == 1 and end2 == 2 and rep2 and (size2 > 0) and (in3 == 3) and (end3 == 2) and rep3 and (size3 == size2) and torch.equal(out_pt, out_g1) and torch.equal(out_g1, out_g2) and torch.equal(out_pt, out_g2)
    print('=' * 60, flush=True)
    print('PASS' if all_ok else 'FAIL', flush=True)
    return 0 if all_ok else 1
if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
