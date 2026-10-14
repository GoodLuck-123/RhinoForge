"""Small host storage migration test for AWQ and persistent direct-zero owners."""
import pytest
import torch

from test_qwen35_legacy_finalized_upload import (
    _fixture, _StopAtNativeWeights, host_device,
)
from rpu_backend.adapters.qwen3_5 import text
from rpu_backend.adapters.qwen3_5.quant_scope import _quanttrio_policy
from rpu_backend.quant.awq_gemm import swizzle_awq_negative_zero


@pytest.mark.parametrize("free_sources", [False, True])
def test_awq_upload_finalizes_cpu_sources_and_retains_zero_owners(
        monkeypatch, host_device, free_sources):
    monkeypatch.setenv("RPU_QWEN3_5_FREE_HF_WEIGHTS", str(int(free_sources)))
    inner, config = _fixture()
    config.intermediate_size = 1024  # Eight row partitions each own a whole G128 group.
    for layer in inner.layers:
        layer.mlp.gate_proj = torch.nn.Linear(128, 1024, bias=False, dtype=torch.float16)
        layer.mlp.up_proj = torch.nn.Linear(128, 1024, bias=False, dtype=torch.float16)
        layer.mlp.down_proj = torch.nn.Linear(1024, 128, bias=False, dtype=torch.float16)
    # Reuse the small mixed-attention storage fixture after geometry admission.
    # The real scope, packing, group scale, zero swizzle and release code runs.
    config.quant_config = _quanttrio_policy([False, True])
    monkeypatch.setattr(text, "resolve_quant_config", lambda _: config.quant_config)
    monkeypatch.setattr(text, "validate_legacy_text_profile", lambda _: False)
    for linear, name in text._iter_text_projections(inner, [False, True]):
        if hasattr(linear, "weight_scale"):
            del linear.weight_scale
        if name.startswith("layers.1.mlp."):
            n, k = linear.weight.shape
            linear.weight = torch.nn.Parameter(torch.randint(0, 16, (n, k), dtype=torch.uint8),
                                                requires_grad=False)
            linear.register_buffer("weight_scale", torch.rand(k // 128, n).half() + 0.125)
            linear.register_buffer("weight_zero", torch.randint(0, 16, (k // 128, n), dtype=torch.uint8))
        else:
            linear.weight = torch.nn.Parameter(linear.weight.half(), requires_grad=False)
    original_zeros = {role: getattr(inner.layers[1].mlp, role + "_proj").weight_zero.clone()
                      for role in ("gate", "up", "down")}
    raw_weights = [module.weight.detach() for module in inner.modules()
                   if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d))]
    raw_pointers = {weight.data_ptr() for weight in raw_weights}
    original_module_to, original_tensor_to = torch.nn.Module.to, torch.Tensor.to

    def module_to(module, *args, **kwargs):
        assert module is not inner and module not in inner.layers
        assert not isinstance(module, (torch.nn.Linear, torch.nn.Conv1d))
        return original_module_to(module, *args, **kwargs)

    def tensor_to(tensor, *args, **kwargs):
        device = kwargs.get("device", args[0] if args else None)
        if isinstance(device, (str, torch.device)) and torch.device(device).type == "rpu":
            assert tensor.data_ptr() not in raw_pointers
        return original_tensor_to(tensor, *args, **kwargs)

    captured = {}
    def capture(*args, **kwargs):
        captured.update(kwargs)
        # Inspect actual published resource ownership before emulating a native
        # rejection. Finalized owners must survive removal of raw HF leaves.
        captured["retirement"] = inner._test_pending.retirement
        raise _StopAtNativeWeights

    monkeypatch.setattr(torch.nn.Module, "to", module_to)
    monkeypatch.setattr(torch.Tensor, "to", tensor_to)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_weights", capture, raising=False)
    pending = inner._test_pending = text._Qwen35NativeState()
    with pytest.raises(_StopAtNativeWeights):
        text._install_qwen3_5_text_for_rpu_impl(
            inner, config, max_seq_len=16, chunk_size_cap=0, exact_chunk_size=0,
            padding_budget=64, padding_rows="auto", execution_config={}, control_snapshot=(),
            graph_cache=None, prefill_graph=None, cpu_stage_weights=False, pending_state=pending)
    for role, partition in (("gate", 1), ("up", 1), ("down", 0)):
        zeros = captured[role + "_negative_zero"]
        assert len(zeros) == 2 and zeros[0].numel() == 0
        torch.testing.assert_close(zeros[1], swizzle_awq_negative_zero(original_zeros[role], partition, 8))
        assert any(owner is zeros for owner in captured["retirement"].keepalive)
        raw = getattr(inner.layers[1].mlp, role + "_proj")
        assert (raw.weight.numel() == 0) == free_sources
        assert (raw.weight_zero.numel() == 0) == free_sources
