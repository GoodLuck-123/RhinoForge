"""Action IO quantization and actual W8 cold AdaRMS table dispatch."""
from types import SimpleNamespace as NS

import pytest
import torch

from rpu_backend.adapters.rhinovla import convert
from rpu_backend.quant._common import quantize_linear_per_channel


@pytest.fixture
def cpu_transfers(monkeypatch):
    original_to = torch.Tensor.to
    original_empty = torch.empty
    rpu_transfers = []

    def to(tensor, *args, **kwargs):
        if args and args[0] == "rpu":
            rpu_transfers.append((tuple(tensor.shape), tensor.dtype))
            args = ("cpu", *args[1:])
        if kwargs.get("device") == "rpu":
            rpu_transfers.append((tuple(tensor.shape), kwargs.get("dtype", tensor.dtype)))
            kwargs["device"] = "cpu"
        return original_to(tensor, *args, **kwargs)

    def empty(*args, **kwargs):
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch, "empty", empty)
    return rpu_transfers


def _unswizzle(weight, *, col, cores):
    n, k = weight.shape
    if col:
        return weight.view(n // (16 * cores), k // 32, cores, 16, 32).permute(
            2, 0, 3, 1, 4).reshape(n, k)
    return weight.view(n // 16, k // (32 * cores), cores, 16, 32).permute(
        0, 3, 2, 1, 4).reshape(n, k)


def test_io_quantizes_all_six_weights_with_matching_layouts_and_scales(cpu_transfers):
    io = NS(config=NS(width=1024, action_dim=96, state_dim=96, action_horizon=30),
            action_in_proj=torch.nn.Linear(96, 1024),
            state_proj=torch.nn.Linear(96, 1024),
            action_out_proj=torch.nn.Linear(1024, 96))
    masks = {name: torch.nn.Linear(96, 1024)
             for name in ("state_mask_proj", "action_mask_proj")}
    norm = NS(norm=NS(weight=torch.randn(1024) * 0.1 + 1),
              cond=torch.nn.Linear(1024, 3072))
    expert = NS(norm=norm)
    folded, folded_bias = convert._fold_final_norm_dense(norm)
    originals = [io.action_in_proj.weight, io.state_proj.weight,
                 masks["state_mask_proj"].weight, masks["action_mask_proj"].weight,
                 folded, io.action_out_proj.weight]
    expected = [quantize_linear_per_channel(w) for w in originals]
    result = convert._prepare_denoise_loop_weights(expert, io, masks, full_w8a16=True)
    names = ("action_in_w", "state_w", "state_mask_w", "action_mask_w",
             "final_norm_w", "action_out_w")
    assert result["direct_action_input"] and result["suffix_len"] == 31
    assert len(result["io_scales"]) == 6
    for index, (name, (quantized, scale)) in enumerate(zip(names, expected)):
        weight = result[name]
        assert weight.dtype == torch.int8 and weight.is_contiguous()
        assert torch.equal(_unswizzle(weight, col=index != 4, cores=8 if index == 4 else 1), quantized)
        assert torch.equal(result["io_scales"][index], scale)
    assert torch.equal(_unswizzle(result["final_norm_w_col"], col=True, cores=8), expected[4][0])
    assert torch.equal(result["final_norm_b"], folded_bias.half())
    assert result["action_time_in_w"].numel() == 0
    assert all(w.dtype == torch.float32 for w in originals)
    fp16 = convert._prepare_denoise_loop_weights(expert, io, masks)
    assert "io_scales" not in fp16 and "final_norm_w_col" not in fp16
    assert all(fp16[name].dtype == torch.float16 for name in names)


def _cold_owners():
    # Numerical projection is the native op's job. Small distinct owners make
    # this test isolate caller ordering, retention and the two scale offsets.
    pairs = [torch.tensor([[index]], dtype=torch.int8) for index in range(18)]
    scales = [torch.tensor([index + 1.0], dtype=torch.float16) for index in range(18)]
    biases = [torch.tensor([index / 8], dtype=torch.float16) for index in range(18)]
    expert = NS(_rpu_full_w8_cond_owners={
        "cold_weights": pairs, "cold_scales": scales, "cold_biases": biases})
    loop = {"final_norm_w_col": torch.tensor([[18]], dtype=torch.int8),
            "io_scales": [None] * 4 + [torch.ones(1, dtype=torch.float16)],
            "final_norm_b": torch.zeros(1, dtype=torch.float16)}
    return expert, loop


@pytest.mark.parametrize("high_precision", (False, True))
@pytest.mark.parametrize("precompute_gate_tanh", (False, True))
def test_cold_tables_dispatch_real_w8_api_and_retain_owners(
        cpu_transfers, monkeypatch, precompute_gate_tanh, high_precision):
    expert, loop = _cold_owners()
    calls = []
    tanh_calls = []
    high_tanh_calls = []
    original_tanh = torch.tanh

    def linear(cond, weight, scale, bias):
        assert cond.dtype == torch.float16 and tuple(cond.shape) == (10, 1024)
        assert weight.dtype == torch.int8
        assert scale.dtype == bias.dtype == torch.float16
        index = len(calls)
        calls.append((weight, scale, bias))
        return torch.full((10, 3072 if index == 18 else 6144), index / 8,
                          dtype=torch.float16)

    def tanh(gates):
        assert gates.dtype == torch.float16 and gates.is_contiguous()
        assert gates.shape == (10, 2048)
        tanh_calls.append(gates.clone())
        return original_tanh(gates)

    def high_tanh(gates):
        high_tanh_calls.append(gates.clone())
        return tanh(gates)

    monkeypatch.setattr(torch.ops.rpu, "rhino_vla_tanh_high_precision", high_tanh, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "linear_w8a16", linear, raising=False)
    monkeypatch.setattr(torch, "tanh", tanh)
    monkeypatch.setattr(torch.nn.functional, "linear",
                        lambda *a, **kw: pytest.fail("cold full W8 used CPU Linear"))
    pair, final, weights, scales, biases = convert._prepare_denoise_adarms_tables(
        expert, torch.randn(10, 1024), full_w8a16=True, loop_weights=loop,
        precompute_gate_tanh=precompute_gate_tanh, high_precision=high_precision)
    assert pair.shape == (10, 18, 6144) and final.shape == (10, 3072)
    assert pair.dtype == final.dtype == torch.float16
    assert len(calls) == len(weights) == len(scales) == len(biases) == 19
    assert loop["adarms_cold_owners"] == (weights, scales, biases)
    assert len(tanh_calls) == (18 if precompute_gate_tanh else 0)
    assert len(high_tanh_calls) == (18 if precompute_gate_tanh and high_precision else 0)
    assert cpu_transfers.count(((10, 2048), torch.float16)) == len(tanh_calls)
    for index, call in enumerate(calls):
        assert all(a is b for a, b in zip(call, (weights[index], scales[index], biases[index])))
        actual = final if index == 18 else pair[:, index]
        expected = torch.full_like(actual, index / 8)
        expected[:, :1024] += 1
        if index != 18:
            expected[:, 3072:4096] += 1
            if precompute_gate_tanh:
                expected[:, 2048:3072] = original_tanh(expected[:, 2048:3072])
                expected[:, 5120:6144] = original_tanh(expected[:, 5120:6144])
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("damage", ("steps", "nonfinite", "owners"))
def test_bad_cold_inputs_fail_before_projection(cpu_transfers, monkeypatch, damage):
    expert, loop = _cold_owners()
    cond = torch.zeros(10, 1024)
    if damage == "steps":
        cond = cond[:9]
    elif damage == "nonfinite":
        cond[0, 0] = float("inf")
    else:
        del loop["final_norm_w_col"]
    monkeypatch.setattr(torch.ops.rpu, "linear_w8a16",
                        lambda *a: pytest.fail("invalid owner reached native Linear"), raising=False)
    with pytest.raises(ValueError, match="RhinoVLA full W8 AdaRMS requires"):
        convert._prepare_denoise_adarms_tables(expert, cond, full_w8a16=True, loop_weights=loop)
    assert "adarms_cold_owners" not in loop


def test_gate_tanh_opt_in_requires_full_w8_before_any_work(monkeypatch):
    monkeypatch.setattr(torch, "tanh", lambda *a: pytest.fail("invalid cold mode dispatched TANH"))
    with pytest.raises(ValueError, match="gate TANH precompute requires full W8"):
        convert._prepare_denoise_adarms_tables(object(), object(), precompute_gate_tanh=True)
