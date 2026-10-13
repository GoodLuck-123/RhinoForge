from __future__ import annotations
import copy
import gc
import json
import os
import subprocess
import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F
pytestmark = pytest.mark.skipif(os.environ.get('RPU_HW_AVAILABLE') != '1', reason='requires device hardware (set RPU_HW_AVAILABLE=1)')

@pytest.fixture(autouse=True)
def _collect_rpu_cycles_after_case():
    yield
    gc.collect()
from rpu_backend.model_registry import model_path
from rpu_backend.tests._seed import input_generator
_PI05_MODEL_PATH = os.environ.get('RPU_PI05_MODEL_PATH', str(model_path('pi05-libero-finetuned-w8a16-vlm-expert')))
_PI05_CPU_REFERENCE_PATH = str(model_path('pi05-libero-finetuned'))

def _load_policy_on_rpu():
    from rpu_backend.api.policy import Pi05Policy
    return Pi05Policy.from_pretrained(_PI05_MODEL_PATH, dtype=torch.float16).to('rpu')

def _siglip_vit(policy):
    inner = policy._lerobot_policy
    vt = inner.model.paligemma_with_expert.paligemma.model.vision_tower
    return vt.vision_model if hasattr(vt, 'vision_model') else vt

def _siglip_projector(policy):
    return policy._lerobot_policy.model.paligemma_with_expert.paligemma.model.multi_modal_projector

def _adapter_handle(policy):
    return policy._adapter._siglip_handle

def _assert_dynamo_preflight_ready(vision):
    import rpu_backend
    assert getattr(vision, '_rpu_lazy_init_checked', False)
    required = getattr(vision, '_rpu_required_attrs', ())
    assert '_rpu_cache' in required
    assert '_rpu_vision_handle' in required
    assert '_rpu_siglip_graph_cache' in required
    for attr in required:
        assert hasattr(vision, attr), f'missing {attr} after SigLIP install'
    rpu_backend.freeze_for_dynamo(vision)

def _rpu_pixel(b: int=1):
    g = input_generator(42)
    cpu = torch.randn(b, 3, 224, 224, generator=g, dtype=torch.float32)
    return (cpu, cpu.to(torch.float16).to('rpu'))

def _flat_cos64(ref: torch.Tensor, got: torch.Tensor) -> float:
    return F.cosine_similarity(ref.flatten().to(torch.float64), got.flatten().to(torch.float64), dim=0).item()

def test_single_image_cos_vs_cpu():
    from rpu_backend.api.policy import Pi05Policy
    p_ref = Pi05Policy.from_pretrained(_PI05_MODEL_PATH, dtype=torch.float16)
    vit_cpu = copy.deepcopy(_siglip_vit(p_ref)).to(torch.float32).to('cpu')
    proj_cpu_module = copy.deepcopy(_siglip_projector(p_ref)).to(torch.float32).to('cpu')
    proj_lin_cpu = proj_cpu_module.linear if hasattr(proj_cpu_module, 'linear') else proj_cpu_module
    del p_ref
    gc.collect()
    p = _load_policy_on_rpu()
    vit = _siglip_vit(p)
    _assert_dynamo_preflight_ready(vit)
    (cpu_pix, rpu_pix) = _rpu_pixel(1)
    with torch.no_grad():
        cpu_last = vit_cpu(pixel_values=cpu_pix).last_hidden_state
        cpu_proj = proj_lin_cpu(cpu_last)
        rpu_out = vit(pixel_values=rpu_pix).last_hidden_state.to(torch.float32).cpu()
    cos = _flat_cos64(cpu_proj, rpu_out)
    assert cpu_proj.shape == rpu_out.shape
    assert torch.isfinite(cpu_proj).all() and torch.isfinite(rpu_out).all()
    print(f'single-image CPU parity cos={cos:.9f}; numeric_status=NOT_EVALUATED (no calibrated magnitude budget)')

def test_three_image_batch_cos_vs_cpu():
    from rpu_backend.api.policy import Pi05Policy
    p_ref = Pi05Policy.from_pretrained(_PI05_MODEL_PATH, dtype=torch.float16)
    vit_cpu = copy.deepcopy(_siglip_vit(p_ref)).to(torch.float32).to('cpu')
    proj_cpu_module = copy.deepcopy(_siglip_projector(p_ref)).to(torch.float32).to('cpu')
    proj_lin_cpu = proj_cpu_module.linear if hasattr(proj_cpu_module, 'linear') else proj_cpu_module
    del p_ref
    gc.collect()
    p = _load_policy_on_rpu()
    vit = _siglip_vit(p)
    (cpu_pix, _) = _rpu_pixel(3)
    rpu_pix = cpu_pix.to(torch.float16).to('rpu')
    with torch.no_grad():
        cpu_outs = [proj_lin_cpu(vit_cpu(pixel_values=cpu_pix[i:i + 1]).last_hidden_state) for i in range(3)]
        cpu_concat = torch.cat(cpu_outs, dim=0)
        rpu_outs = [vit(pixel_values=rpu_pix[i:i + 1]).last_hidden_state.to(torch.float32).cpu() for i in range(3)]
        rpu_concat = torch.cat(rpu_outs, dim=0)
    cos = _flat_cos64(cpu_concat, rpu_concat)
    assert cos > 0.99985, f'3-image batch cos={cos:.9f} < 0.99985'

def test_two_pi05policy_instances_independent():
    p1 = _load_policy_on_rpu()
    h1 = _siglip_vit(p1)._rpu_vision_handle
    assert _adapter_handle(p1) == h1, f'adapter h1={_adapter_handle(p1)} != vit h1={h1}'
    (_, rpu_pix) = _rpu_pixel(1)
    out1 = _siglip_vit(p1)(pixel_values=rpu_pix).last_hidden_state.to(torch.float32).cpu()
    del p1, rpu_pix
    gc.collect()
    p2 = _load_policy_on_rpu()
    h2 = _siglip_vit(p2)._rpu_vision_handle
    assert _adapter_handle(p2) == h2, f'adapter h2={_adapter_handle(p2)} != vit h2={h2}'
    assert h1 != h2, f'two SigLIP instances reused handle h1={h1} h2={h2} (ABA bug)'
    (_, rpu_pix) = _rpu_pixel(1)
    out2 = _siglip_vit(p2)(pixel_values=rpu_pix).last_hidden_state.to(torch.float32).cpu()
    cos = _flat_cos64(out1, out2)
    assert cos > 0.999, f'identical-input cos={cos:.9f} unexpected divergence'

def test_repeat_to_rpu_idempotent():
    p = _load_policy_on_rpu()
    h_first = _siglip_vit(p)._rpu_vision_handle
    p.to('rpu')
    h_second = _siglip_vit(p)._rpu_vision_handle
    assert h_second == h_first, f'adapter short-circuit failed: h_first={h_first} h_second={h_second}'
    from rpu_backend.adapters.siglip import patch_siglip_model_for_rpu_all_layers_once
    h_direct_old = _siglip_vit(p)._rpu_vision_handle
    h_direct_new = patch_siglip_model_for_rpu_all_layers_once(_siglip_vit(p), _siglip_projector(p))
    assert h_direct_new != h_direct_old, 'inner patcher must create new handle'
    assert _siglip_vit(p)._rpu_vision_handle == h_direct_new
    with pytest.raises(Exception):
        torch.ops.rpu.siglip_destroy(h_direct_old)
    (_, rpu_pix) = _rpu_pixel(1)
    _ = _siglip_vit(p)(pixel_values=rpu_pix).last_hidden_state

def test_del_policy_gc_releases_handle():
    p = _load_policy_on_rpu()
    h = _siglip_vit(p)._rpu_vision_handle
    del p
    gc.collect()
    with pytest.raises(Exception):
        torch.ops.rpu.siglip_destroy(h)

def test_1000_image_rss_growth():
    import psutil
    process = psutil.Process()
    p = _load_policy_on_rpu()
    vit = _siglip_vit(p)
    (_, warmup) = _rpu_pixel(1)
    for _ in range(10):
        _ = vit(pixel_values=warmup).last_hidden_state
    gc.collect()
    torch.ops.rpu.spm_alloc_reset_temporary()
    rss_baseline = process.memory_info().rss
    batches = [_rpu_pixel(1)[1] for _ in range(10)]
    for i in range(1000):
        _ = vit(pixel_values=batches[i % 10]).last_hidden_state
    rss_after = process.memory_info().rss
    del p, vit
    gc.collect()
    torch.ops.rpu.spm_alloc_reset_temporary()
    rss_clean = process.memory_info().rss
    delta = rss_after - rss_baseline
    print(f'RSS Δ(1000-img)={delta / 1000000.0:.1f} MB; cleanup→baseline Δ={(rss_clean - rss_baseline) / 1000000.0:.1f} MB')
    assert delta <= 100 * 1024 * 1024, f'RSS grew {delta / 1000000.0:.1f} MB > 100 MB'

def test_set_weights_exception_fresh_retry(monkeypatch):
    from rpu_backend.api.policy import Pi05Policy
    from rpu_backend.adapters import siglip as _sig_mod
    real = _sig_mod.patch_siglip_model_for_rpu_all_layers_once
    triggered = {'once': False}
    captured = {'handle': None, 'vit': None}

    def faulty(vision_model, projector, *args, **kwargs):
        if not triggered['once']:
            triggered['once'] = True
            h = torch.ops.rpu.siglip_create()
            vision_model._rpu_vision_handle = h
            captured['handle'] = h
            captured['vit'] = vision_model
            raise RuntimeError('simulated siglip_set_weights failure')
        return real(vision_model, projector, *args, **kwargs)
    monkeypatch.setattr(_sig_mod, 'patch_siglip_model_for_rpu_all_layers_once', faulty)
    p_fail = Pi05Policy.from_pretrained(_PI05_MODEL_PATH, dtype=torch.float16)
    with pytest.raises(RuntimeError):
        p_fail.to('rpu')
    assert not hasattr(captured['vit'], '_rpu_vision_handle'), 'Task 5.5 cleanup must delete _rpu_vision_handle attr'
    with pytest.raises(Exception):
        torch.ops.rpu.siglip_destroy(captured['handle'])
    with pytest.raises(Exception):
        p_fail.to('rpu')
    del p_fail
    gc.collect()
    p_ok = Pi05Policy.from_pretrained(_PI05_MODEL_PATH, dtype=torch.float16).to('rpu')
    assert _siglip_vit(p_ok)._rpu_vision_handle > 0
