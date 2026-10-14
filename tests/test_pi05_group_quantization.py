"""Real K128 scales and the explicit NVFP4 ABI must survive the public loader."""
from types import SimpleNamespace
import pytest
import torch
from rpu_backend.api.errors import RPUBackendError
from rpu_backend.adapters.pi05.loader import _pop_pi05_w8a16_tensors, _install_pi05_w8a16_tensors
from rpu_backend.adapters.pi05.w4pack import pack_int4_projections_inplace
from rpu_backend.quant.convert_pi05 import _convert_tensor, PI05_EXPERT_ANCHOR
from rpu_backend.quant.int4_pgrp_pack import quantize_int4_group_wise
from rpu_backend.quant.nvfp4_pack import swizzle_pack_nvfp4_v2


@pytest.mark.parametrize("projection,partition", [("q_proj", 1), ("o_proj", 0)])
def test_k128_distinct_groups_survive_loader_and_physical_packing(projection, partition):
    n, k = 256, 1024
    # Every group needs a different scale. A per-channel repeat loses small groups.
    amplitudes = torch.tensor([.007, .014, .028, .056, .112, .224, .448, .896])
    weight = (torch.linspace(-1, 1, 128).repeat(n, 8).reshape(n, 8, 128)
              * amplitudes[None, :, None]).reshape(n, k).half()
    name = "model" + PI05_EXPERT_ANCHOR + "0.self_attn." + projection + ".weight"
    tensors, _, _, _ = _convert_tensor(name, weight, bits=4, w4_format="int4_g128")
    scale = tensors[name.replace(".weight", ".weight_scale")].clone()
    assert scale.shape == (8, n) and torch.unique(scale[:, 0]).numel() == 8
    q = tensors[name].clone()
    deq = (q.view(n, 8, 128).float() * scale.t()[:, :, None]).reshape(n, k)
    per_channel_step = weight.abs().amax(1) / 7
    legacy = (weight / per_channel_step[:, None]).round() * per_channel_step[:, None]
    assert (deq - weight).square().mean() < (legacy - weight).square().mean() * .4
    module = torch.nn.Linear(k, n, bias=False)
    policy = SimpleNamespace(named_modules=lambda: [(name.removesuffix(".weight"), module)])
    cfg = {"method": "w4a16", "mode": "group_symmetric", "int4_group_size": 128,
           "int4_components": ["expert"], "mixed_int8_suffixes": ["k_proj.weight", "v_proj.weight"]}
    consumed = _pop_pi05_w8a16_tensors(policy, tensors, cfg)
    assert not tensors
    _install_pi05_w8a16_tensors(policy, consumed)
    attn = torch.nn.Module()
    setattr(attn, projection, module)
    decoder = SimpleNamespace(layers=[SimpleNamespace(self_attn=attn)])
    pack_int4_projections_inplace(decoder, {projection}, attn_num_cores=8, group_size=128)
    # Generator's individual controller address, independent of packer's reshape.
    physical = module.weight_scale.flatten()
    local_g, local_n = (8, n // 8) if partition else (1, n)
    nblocks = (local_n + 63) // 64
    for core in (0, 3, 7):
        for g in range(local_g):
            for col in (0, local_n - 1):
                offset = ((((g // 4) * nblocks + col // 64) * 8 + core) * 4 + g % 4) * 64 + col % 64
                gg, nn = (g, core * local_n + col) if partition else (core * local_g + g, col)
                assert physical[offset] == scale[gg, nn]
    assert pack_int4_projections_inplace(decoder, {projection}, attn_num_cores=8, group_size=128) == 0


def test_loader_rejects_group_scale_without_group_metadata():
    name = "model" + PI05_EXPERT_ANCHOR + "0.self_attn.q_proj.weight"
    tensors, _, _, _ = _convert_tensor(name, torch.randn(128, 1024), bits=4, w4_format="int4_g128")
    policy = SimpleNamespace(named_modules=lambda: [(name.removesuffix(".weight"), torch.nn.Linear(1024, 128))])
    with pytest.raises(RPUBackendError, match="must be"):
        _pop_pi05_w8a16_tensors(policy, tensors, {"method": "w8a16"})


@pytest.mark.parametrize("partition", [0, 1])
def test_nvfp4_v2_kernel_lane_and_scale_addresses(partition):
    n, k, cores = 128, 512, 8
    # Nonperiodic codes expose K16 nibble swaps; a repeating 0..15 pattern
    # would make the two halves of each new nibble pair indistinguishable.
    codes = torch.randint(0, 16, (n, k), dtype=torch.uint8,
                          generator=torch.Generator().manual_seed(128))
    scales = torch.arange(k // 16 * n).reshape(k // 16, n).remainder(251).byte()
    packed, physical = swizzle_pack_nvfp4_v2(codes, scales, partition, cores)
    # Check the generator's 16N x 64K microtile and K16-separated nibble pairs.
    local_n, local_k = (n // cores, k) if partition else (n, k // cores)
    for core in (0, 7):
        for nn in (0, local_n - 1):
            for kk in sorted({0, 15, 16, 31, 32, 47, 48, 63, local_k - 1}):
                gn, gk = (core * local_n + nn, kk) if partition else (nn, core * local_k + kk)
                block = (((nn // 16) * (local_k // 64) + kk // 64) * cores + core) * 16 + nn % 16
                byte_offset = block * 32 + (kk % 64 // 32) * 16 + kk % 16
                shift = 4 * (kk % 32 // 16)
                assert ((packed.flatten()[byte_offset] >> shift) & 15) == codes[gn, gk]
                sg = kk // 16
                scale_offset = ((((sg // 2) * cores + core) * (local_n // 16) + nn // 16) * 2 + sg % 2) * 16 + nn % 16
                assert physical.flatten()[scale_offset] == scales[gk // 16, gn]


def test_nvfp4_loader_keeps_fp32_tensor_scale_and_rejects_precision_mismatch():
    from rpu_backend.quant.nvfp4_pack import dequantize_nvfp4
    name = "model" + PI05_EXPERT_ANCHOR + "0.self_attn.q_proj.weight"
    weight = torch.randn(128, 1024, generator=torch.Generator().manual_seed(12)).half()
    values, _, _, _ = _convert_tensor(name, weight, bits=4, w4_format="nvfp4")
    ts_name = name.removesuffix(".weight") + ".tensor_scale"
    cfg = {"method": "nvfp4a16", "int4_components": ["expert"],
           "mixed_int8_suffixes": ["k_proj.weight", "v_proj.weight"]}
    module = torch.nn.Linear(1024, 128, bias=False)
    policy = SimpleNamespace(named_modules=lambda: [(name.removesuffix(".weight"), module)])
    invalid = dict(values)
    invalid[ts_name] = invalid[ts_name].half()
    with pytest.raises(RPUBackendError, match="FP32 tensor scale"):
        _pop_pi05_w8a16_tensors(policy, invalid, cfg)
    with pytest.raises(RPUBackendError, match="must be"):
        _pop_pi05_w8a16_tensors(policy, dict(values), {"method": "w4a16"})
    consumed = _pop_pi05_w8a16_tensors(policy, values, cfg)
    _install_pi05_w8a16_tensors(policy, consumed)
    assert not values and module.tensor_scale.dtype == torch.float32
    reconstructed = dequantize_nvfp4(module.weight, module.weight_scale, module.tensor_scale)
    assert (reconstructed - weight).norm() / weight.norm() < .15


def test_native_optional_tensor_list_schemas_parse_without_loading_rpu():
    import re
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "src/core/rpu_dispatch_registrations.inc").read_text()
    lib = torch.library.Library("pi05_quant_schema_regression", "DEF")
    for op in ("adarms_set_weights", "pi05_denoise_step_set_weights"):
        start = source.index('m.def("' + op + '(')
        end = source.index(');', start)
        schema = ''.join(re.findall(r'"([^"\n]*)"', source[start:end]))
        lib.define(schema)
