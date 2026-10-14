from __future__ import annotations
import ast
from pathlib import Path
import pytest
import torch
from rpu_backend.runtime import decoder

def test_fused_decoder_keeps_the_live_input_without_republishing_device_writes():
    source = (Path(__file__).resolve().parents[4] / 'src/core/fused_model_base.cpp').read_text(encoding='utf-8')
    assert 'RpuGetDevAddr(hidden_states.data_ptr())' in source
    assert 'RpuKernelGraph::active().keep_alive(hidden_states)' in source
    assert 'rpu_ddr_flush_force_sized(\n        hidden_states.data_ptr' not in source

def test_decoder_output_validation_is_not_optimized_away():
    tree = ast.parse(Path(decoder.__file__).read_text(encoding='utf-8'))
    validator = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_validate_decoder_base_output')
    assert not any(isinstance(node, ast.Assert) for node in ast.walk(validator))
    assert any(isinstance(node, ast.Raise) for node in ast.walk(validator))

def test_decoder_base_output_accepts_hidden_states():
    decoder._validate_decoder_base_output(torch.zeros(1, 4, 128), 128)

@pytest.mark.parametrize('raw', [torch.zeros(1, 4, 256), torch.zeros(4, 128), torch.zeros(1, 4, 128, 1)])
def test_decoder_base_output_rejects_logits_or_wrong_rank(raw):
    with pytest.raises(RuntimeError, match='use _apply_fused_lm_head_for_rpu'):
        decoder._validate_decoder_base_output(raw, 128)

def test_decoder_base_output_rejects_non_tensor():
    with pytest.raises(TypeError, match='expected a tensor'):
        decoder._validate_decoder_base_output([], 128)
