import pytest
import torch
from rpu_backend.api.errors import RPUUnsupportedDtypeError, UnsupportedModelError

def test_loader_rejects_non_fp16_before_hf_load():
    from rpu_backend.adapters.paligemma.loader import _validate_dtype_and_kwargs
    with pytest.raises(RPUUnsupportedDtypeError, match='float16'):
        _validate_dtype_and_kwargs(torch.bfloat16, {})

@pytest.mark.parametrize('reserved', ['torch_dtype', 'device_map', 'device', 'tp_plan', 'low_cpu_mem_usage', 'load_in_4bit', 'load_in_8bit', 'quantization_config', 'config'])
def test_loader_rejects_reserved_hf_kwargs(reserved):
    from rpu_backend.adapters.paligemma.loader import _validate_dtype_and_kwargs
    with pytest.raises(ValueError, match=reserved):
        _validate_dtype_and_kwargs(torch.float16, {reserved: object()})

def test_loader_rejects_config_mutating_hf_kwargs():
    from rpu_backend.adapters.paligemma.loader import _validate_dtype_and_kwargs
    with pytest.raises(ValueError, match='text_config'):
        _validate_dtype_and_kwargs(torch.float16, {'text_config': {'use_bidirectional_attention': False}})

def test_resolve_logical_name_prefers_model_registry(monkeypatch, tmp_path):
    from rpu_backend.adapters.paligemma import loader
    target = tmp_path / 'paligemma2-3b-pt-224'
    target.mkdir()
    monkeypatch.setattr(loader, 'model_path', lambda name: target)
    assert loader._resolve_name_or_path('paligemma2-3b-pt-224') == str(target)

@pytest.mark.parametrize(('logical_name', 'repo_id'), [('paligemma2-3b-pt-224', 'google/paligemma2-3b-pt-224'), ('paligemma2-3b-mix-224', 'google/paligemma2-3b-mix-224')])
def test_resolve_logical_name_uses_hf_repo_when_cache_missing(monkeypatch, tmp_path, logical_name, repo_id):
    from rpu_backend.adapters.paligemma import loader
    missing = tmp_path / logical_name
    monkeypatch.setattr(loader, 'model_path', lambda name: missing)
    assert loader._resolve_name_or_path(logical_name) == repo_id

def test_resolve_non_logical_name_passthrough():
    from rpu_backend.adapters.paligemma.loader import _resolve_name_or_path
    assert _resolve_name_or_path('google/paligemma2-3b-pt-224') == 'google/paligemma2-3b-pt-224'

def test_loader_downgrade_fails_before_hf_access(monkeypatch):
    from rpu_backend.adapters.paligemma import loader
    calls = []

    class BombLoader:

        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append((path, kwargs))
            raise AssertionError('HF loader must not be reached')
    monkeypatch.setattr(loader, 'AutoConfig', BombLoader)
    monkeypatch.setattr(loader, 'PaliGemmaForConditionalGeneration', BombLoader)
    with pytest.raises(UnsupportedModelError, match='fixed and recertified'):
        loader.load_and_construct_paligemma_policy('paligemma2-3b-pt-224', dtype=torch.float16)
    assert calls == []
