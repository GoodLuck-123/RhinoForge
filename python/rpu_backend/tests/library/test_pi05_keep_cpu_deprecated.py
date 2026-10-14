from __future__ import annotations
import os
from types import SimpleNamespace
import pytest
from rpu_backend.adapters.pi05 import Pi05Adapter
from rpu_backend.api.errors import RPUBackendError

def _make_dummy_adapter() -> Pi05Adapter:
    inst = Pi05Adapter.__new__(Pi05Adapter)
    inst._rpu_is_ready = False
    inst._closed = False
    inst._rpu_execution = {}
    inst._legacy_vlm_chunk_size = 0
    config = SimpleNamespace(chunk_size=50, image_features={"image_a": {}, "image_b": {}})
    inst._lerobot_policy = SimpleNamespace(config=config, model=SimpleNamespace(config=config))
    inst._lerobot_policy._rpu_swizzled = False
    inst._lerobot_policy._rpu_swizzle_started = False
    return inst

@pytest.mark.parametrize('value', ['siglip', 'vlm', 'expert', 'siglip,vlm'])
def test_keep_cpu_env_raises(monkeypatch, value):
    monkeypatch.setenv('RPU_PI05_KEEP_CPU', value)
    adapter = _make_dummy_adapter()
    with pytest.raises(RuntimeError) as excinfo:
        adapter.to_rpu()
    assert 'v3.0 bisection scaffold' in str(excinfo.value), f"Expected the deprecation message to mention 'v3.0 bisection scaffold'; got: {excinfo.value!r}"
    assert getattr(adapter._lerobot_policy, '_rpu_swizzle_started', False) is False, 'RPU_PI05_KEEP_CPU guard fired AFTER state mutation — instance is poisoned (R2-FIX-2 regression: guard placement wrong).'

def test_keep_cpu_unset_works(monkeypatch):
    monkeypatch.delenv('RPU_PI05_KEEP_CPU', raising=False)
    adapter = _make_dummy_adapter()
    try:
        adapter.to_rpu()
    except RuntimeError as exc:
        assert 'v3.0 bisection scaffold' not in str(exc), f'Guard fired with empty/unset RPU_PI05_KEEP_CPU: {exc!r}'
    except (AttributeError, RPUBackendError):
        pass

def test_keep_cpu_recovery_after_clear(monkeypatch):
    monkeypatch.setenv('RPU_PI05_KEEP_CPU', 'siglip')
    adapter = _make_dummy_adapter()
    with pytest.raises(RuntimeError) as excinfo:
        adapter.to_rpu()
    assert 'v3.0 bisection scaffold' in str(excinfo.value)
    monkeypatch.delenv('RPU_PI05_KEEP_CPU')
    assert getattr(adapter._lerobot_policy, '_rpu_swizzle_started', False) is False, 'Instance retained _rpu_swizzle_started=True after fail-loud raise — guard placement is AFTER _rpu_swizzle_started=True (R2-FIX-2 regression).'
