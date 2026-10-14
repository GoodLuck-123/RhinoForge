"""Lazy Vision cold tensors remain mutable when later requests use no_grad."""
from types import SimpleNamespace as NS

import pytest
import torch

from rpu_backend.adapters.qwen3_5 import vision


@pytest.mark.parametrize("lazy,g05,math_precision", [
    (False, False, 'base'), (True, False, 'base'), (False, True, 'base'), (False, True, 'high')])
def test_vision_cold_install_does_not_create_inference_buffers(monkeypatch, lazy, g05, math_precision):
    config = NS(depth=24, num_heads=16, hidden_size=1024, intermediate_size=4096) if g05 else NS()
    tower = NS(config=config, blocks=[], merger=object())
    model = NS(model=NS(visual=tower))
    cold_modes = []
    monkeypatch.setattr(torch.ops.rpu, 'qwen3_5_vision_set_high_precision', lambda *_: None, raising=False)

    def allocate_cold_buffers(*args, **kwargs):
        # CPU doubles for the native position-index backing and the dummy KV
        # buffers allocated by the cold implementation. Forward updates these
        # owners in place; their context must not inherit the first request.
        cold_modes.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
        assert kwargs.get("_g05_w8a16", False) is g05
        assert kwargs.get("_g05_acc16_experiment", False) is g05
        assert kwargs.get('_g05_math_precision', 'base') == math_precision
        tower._rpu_vision_position_idx_keepalive = torch.zeros(8, 2, dtype=torch.int16)
        tower.kv = torch.zeros(8, dtype=torch.float16)
        return 7

    monkeypatch.setattr(vision, "_install_qwen3_5_vision_for_rpu_impl", allocate_cold_buffers)
    with torch.inference_mode():
        if lazy:
            handle = vision._install_lazy_qwen3_5_vision(model, NS(execution_generation=3))
        else:
            handle = vision.install_qwen3_5_vision_for_rpu(
                tower, _allow_certified_g05_k1=g05,
                execution_config={"vision": {"linear_acc32": False}} if g05 else None,
                _g05_w8a16=g05, _g05_acc16_experiment=g05, _g05_math_precision=math_precision)
        assert torch.is_inference_mode_enabled()
    assert handle == 7
    # Match the next request's position refresh and cache reset under no_grad.
    with torch.no_grad():
        tower._rpu_vision_position_idx_keepalive.narrow(0, 0, 4).copy_(
            torch.ones(4, 2, dtype=torch.int16))
        tower.kv.zero_()
    assert cold_modes == [(False, False)]
    assert not tower._rpu_vision_position_idx_keepalive.is_inference()
    assert tower._rpu_vision_position_idx_keepalive._version == 1
    assert tower.kv._version == 1


def test_failed_vision_cold_install_restores_request_context(monkeypatch):
    tower = NS(config=NS(), blocks=[], merger=object())
    modes = []

    def fail(*args, **kwargs):
        modes.append((torch.is_inference_mode_enabled(), torch.is_grad_enabled()))
        raise ValueError("cold failure")

    monkeypatch.setattr(vision, "_install_qwen3_5_vision_for_rpu_impl", fail)
    with torch.inference_mode():
        with pytest.raises(ValueError, match="cold failure"):
            vision.install_qwen3_5_vision_for_rpu(tower)
        assert torch.is_inference_mode_enabled()
    assert modes == [(False, False)]


@pytest.mark.parametrize("text_already_ran", [False, True])
@pytest.mark.parametrize("w8", [False, True])
def test_validated_tree_allows_first_vision_conversion(monkeypatch, text_already_ran, w8):
    from rpu_backend.runtime import RPUConfigError
    from rpu_backend.runtime import hw_attrs

    root = torch.nn.Module()
    block = root.visual = torch.nn.Module()
    hidden, intermediate, heads = (1024, 4096, 16) if w8 else (64, 128, 8)
    block.attn = torch.nn.Module()
    block.attn.qkv = torch.nn.Linear(hidden, 3 * hidden, dtype=torch.float16)
    block.attn.proj = torch.nn.Linear(hidden, hidden, dtype=torch.float16)
    block.mlp = torch.nn.Module()
    block.mlp.linear_fc1 = torch.nn.Linear(hidden, intermediate, dtype=torch.float16)
    block.mlp.linear_fc2 = torch.nn.Linear(intermediate, hidden, dtype=torch.float16)
    swizzles = []

    def swizzle(weight, **kwargs):
        swizzles.append(tuple(weight.shape))
        return weight.clone()

    monkeypatch.setattr(vision, "tp_col_swizzle_mc_weight", swizzle)
    monkeypatch.setattr(vision, "tp_row_swizzle_mc_weight", swizzle)
    hw_attrs.install_hw_attr_validator(root)
    if text_already_ran:
        hw_attrs.mark_first_forward_done(root)
    vision._convert_vision_block_weights_for_rpu(block, heads, hidden, _g05_w8a16=w8)
    assert block._rpu_vision_w8a16 is w8
    assert block._rpu_q_w.dtype == (torch.int8 if w8 else torch.float16)
    assert block._rpu_qwen3_5_vision_weights_converted is True
    assert "_rpu_qwen3_5_vision_conversion_started" not in vars(block)
    count = len(swizzles)
    vision._convert_vision_block_weights_for_rpu(block, heads, hidden, _g05_w8a16=w8)
    assert len(swizzles) == count == 6
    with pytest.raises(RuntimeError, match="precision is immutable"):
        vision._convert_vision_block_weights_for_rpu(block, heads, hidden, _g05_w8a16=not w8)
    with pytest.raises(RPUConfigError, match="Unknown hardware attribute"):
        block._rpu_vision_w8a16_typo = True
    hw_attrs.validate_postinstall(root)
