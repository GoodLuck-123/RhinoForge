from __future__ import annotations
from pathlib import Path
import pytest
import torch
from rpu_backend.adapters.pi05.runtime import _load_pi05_probe_tensor

def test_pi05_probe_tensor_uses_safe_tensor_payload(tmp_path):
    path = tmp_path / 'probe.pt'
    expected = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    torch.save(expected, path)
    actual = _load_pi05_probe_tensor(str(path), name='probe', expected_shape=(1, 2, 3))
    assert torch.equal(actual, expected)

def test_pi05_probe_tensor_rejects_non_tensor(tmp_path):
    path = tmp_path / 'mapping.pt'
    torch.save({'value': torch.zeros(1)}, path)
    with pytest.raises(TypeError, match='must contain a tensor'):
        _load_pi05_probe_tensor(str(path), name='probe', expected_shape=(1,))

def test_pi05_probe_tensor_rejects_shape_drift(tmp_path):
    path = tmp_path / 'shape.pt'
    torch.save(torch.zeros(2, 3), path)
    with pytest.raises(ValueError, match='shape \\(1, 2, 3\\)'):
        _load_pi05_probe_tensor(str(path), name='probe', expected_shape=(1, 2, 3))

def test_pi05_probe_tensor_rejects_non_finite_values(tmp_path):
    path = tmp_path / 'nan.pt'
    torch.save(torch.tensor([float('nan')]), path)
    with pytest.raises(ValueError, match='only finite'):
        _load_pi05_probe_tensor(str(path), name='probe', expected_shape=(1,))
