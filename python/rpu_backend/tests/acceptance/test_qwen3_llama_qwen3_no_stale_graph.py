from __future__ import annotations
import gc
import os
import pytest
import torch
_HF_HUB_ID = {'qwen3-0.6b': 'Qwen/Qwen3-0.6B', 'llama-3.2-1b': 'unsloth/Llama-3.2-1B'}

def _model_dir(slug: str) -> str:
    try:
        from rpu_backend.model_registry import model_path
        local = model_path(slug)
        if os.path.exists(local):
            return str(local)
    except Exception:
        pass
    from rpu_backend.model_registry import cache_root
    hub_id = _HF_HUB_ID.get(slug)
    if hub_id is not None:
        snap_dir = f"{cache_root()}/.hf/hub/models--{hub_id.replace('/', '--')}/snapshots"
        if os.path.isdir(snap_dir):
            snaps = sorted(os.listdir(snap_dir))
            for s in snaps:
                p = os.path.join(snap_dir, s)
                if os.path.isfile(os.path.join(p, 'config.json')):
                    return p
    return _HF_HUB_ID[slug]

def _measure(model_id_or_path: str, label: str) -> tuple[torch.Tensor, torch.Tensor]:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from rpu_backend import RPUModelForCausalLM, RPUCache
    tok = AutoTokenizer.from_pretrained(model_id_or_path)
    prompt = 'The quick brown fox jumps over the lazy dog.'
    inputs = tok(prompt, return_tensors='pt')
    cpu = AutoModelForCausalLM.from_pretrained(model_id_or_path, torch_dtype=torch.float16).eval()
    with torch.no_grad():
        cpu_logits = cpu(inputs.input_ids).logits.detach()
    del cpu
    gc.collect()
    rpu = RPUModelForCausalLM.from_pretrained(model_id_or_path, dtype=torch.float16).to('rpu')
    cache = RPUCache.from_model(rpu, input_ids=inputs.input_ids, max_new_tokens=0)
    with torch.no_grad():
        rpu_logits = rpu(inputs.input_ids.to('rpu'), past_key_values=cache).logits.cpu()
    del rpu, cache
    gc.collect()
    cos = torch.nn.functional.cosine_similarity(rpu_logits.float().flatten().unsqueeze(0), cpu_logits.float().flatten().unsqueeze(0)).item()
    print(f'[U-1] {label} cos_sim={cos:.6f}')
    return (cpu_logits, rpu_logits)

@pytest.mark.hardware
@pytest.mark.skipif(os.environ.get('RPU_HW_AVAILABLE') != '1', reason='HW-gated: same-process Qwen3/Llama graph replay requires device device')
def test_qwen3_llama_qwen3_no_stale_graph():
    qwen3_path = _model_dir('qwen3-0.6b')
    llama_path = _model_dir('llama-3.2-1b')
    (cpu_q1, rpu_q1) = _measure(qwen3_path, 'Qwen3 first')
    cos_q1 = torch.nn.functional.cosine_similarity(rpu_q1.float().flatten().unsqueeze(0), cpu_q1.float().flatten().unsqueeze(0)).item()
    assert rpu_q1.shape == cpu_q1.shape and rpu_q1.numel() > 0
    assert torch.isfinite(rpu_q1).all() and torch.isfinite(cpu_q1).all()
    print(f'Qwen3 CPU cosine={cos_q1} (diagnostic); task_quality=NOT_EVALUATED formal_certification=false')
    (cpu_l, rpu_l) = _measure(llama_path, 'Llama')
    cos_l = torch.nn.functional.cosine_similarity(rpu_l.float().flatten().unsqueeze(0), cpu_l.float().flatten().unsqueeze(0)).item()
    assert cos_l >= 0.999, f'Llama run cos_sim={cos_l} < 0.999'
    (cpu_q2, rpu_q2) = _measure(qwen3_path, 'Qwen3 second')
    assert rpu_q2.shape == cpu_q2.shape and rpu_q2.numel() > 0
    assert torch.isfinite(rpu_q2).all() and torch.isfinite(cpu_q2).all()
    mse = torch.mean((rpu_q1.float() - rpu_q2.float()) ** 2).item()
    cos = torch.nn.functional.cosine_similarity(rpu_q1.float().flatten().unsqueeze(0), rpu_q2.float().flatten().unsqueeze(0)).item()
    print(f'[U-1] Qwen3 first vs second MSE={mse:.10f} cosine={cos:.10f}')
    assert mse == 0.0 or cos >= 0.999999, f'U-1 stale-graph FAIL: Qwen3 second run differs from Qwen3 first run after Llama interleaved (MSE={mse} cosine={cos}). instance_id + model_state_gen guard insufficient — investigate graph cache reuse.'
    print('[U-1] OK: no stale graph cache reuse across Qwen3 <-> Llama (instance_id + model_state_gen verified)')
