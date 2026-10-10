"""Board-free exact Expert NVFP4 selection, physical packing and table ownership."""
import dataclasses

import pytest
import torch

from rpu_backend.adapters.hy_vla import nvfp4, weights
from rpu_backend.quant.nvfp4_pack import dequantize_nvfp4, quantize_nvfp4


NV = "RPU_HY_VLA_EXPERT_NVFP4"
RES = "RPU_HY_VLA_EXPERT_GATEUP_RESIDENT"
LAYERS = "RPU_HY_VLA_EXPERT_GATEUP_RESIDENT_LAYERS"
ENV = {NV: "1", "RPU_HY_VLA_W8A16": "vit,vlm", "RPU_HY_VLA_ATTN_TP8": "1"}
PROJECTIONS = ("q", "k", "v", "o", "gate", "up", "down")


@pytest.mark.parametrize("count", [12, 24])
def test_nvfp4_cold_selection_is_separate_frozen_and_explicit(count):
    environment = dict(ENV)
    plan = weights._HyVlaWeightColdPlan.from_environment(environment)
    environment[NV] = "0"
    assert plan.expert_nvfp4 and plan.expert_bits == 16
    assert (plan.vit_bits, plan.vlm_text_bits, plan.vlm_vision_bits) == (8, 8, 8)
    assert not plan.gateup_resident
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.expert_nvfp4 = False
    resident_environment = ENV | {RES: "1", LAYERS: str(count)}
    resident = weights._HyVlaWeightColdPlan.from_environment(resident_environment)
    resident_environment[LAYERS] = "6"
    assert resident.gateup_resident_layers == count and resident.expert_nvfp4
    with pytest.raises(dataclasses.FrozenInstanceError):
        resident.gateup_resident_layers = 6
    off = weights._HyVlaWeightColdPlan.from_environment(environment)
    assert not off.expert_nvfp4 and off.expert_bits == 16
    metadata = nvfp4.HyVlaExpertNvfp4Metadata(resident_layers=count)
    assert metadata.accumulation == "acc16" and metadata.source_rounding == "fp16_before_quantize"
    with pytest.raises(dataclasses.FrozenInstanceError):
        metadata.resident_layers = 6


@pytest.mark.parametrize("override", [
    {NV: "expert"}, {NV: "yes"}, {NV: "2"},
    {"RPU_HY_VLA_W8A16": "all"}, {"RPU_HY_VLA_W8A16": "expert,vit,vlm"},
    {"RPU_HY_VLA_W4A16": "expert"}, {"RPU_HY_VLA_W4A16": "vlm_text"},
    {"RPU_HY_VLA_W8A16": "vit,vlm_text"}, {"RPU_HY_VLA_W8A16": "0"},
    {"RPU_HY_VLA_ATTN_TP8": "0"}, {"RPU_HY_VLA_ACTION_MLP_MC": "0"},
    {RES: "1"}, {RES: "1", LAYERS: "6"},
    {RES: "1", LAYERS: "12", "RPU_HY_VLA_EXPERT_O_RESIDENT_LAYERS": "12"},
])
def test_conflicting_or_incomplete_nvfp4_plan_rejects(override):
    with pytest.raises(ValueError, match="NVFP4|O_RESIDENT"):
        weights._HyVlaWeightColdPlan.from_environment(ENV | override)


@pytest.mark.parametrize("value", ["6", "13", "23", "25", "32", "024", "24.0"])
def test_nvfp4_resident_count_rejects_unsupported_or_nonliteral_values(value):
    with pytest.raises(ValueError, match=LAYERS):
        weights._HyVlaWeightColdPlan.from_environment(ENV | {RES: "1", LAYERS: value})


@pytest.mark.parametrize("override", [
    {RES: "0"},
    {NV: "0"},
    {NV: "0", "RPU_HY_VLA_W8A16": "all"},
])
def test_twenty_four_layers_requires_enabled_nvfp4_residency(override):
    with pytest.raises(ValueError, match=LAYERS):
        weights._HyVlaWeightColdPlan.from_environment(ENV | {RES: "1", LAYERS: "24"} | override)


@pytest.mark.parametrize("partition,shape", [(1, (1024, 1024)), (0, (1024, 2048))])
def test_canonical_cpu_conversion_rounding_and_independent_unpack(partition, shape):
    n, k = shape
    raw = ((torch.arange(n*k, dtype=torch.float32) % 83 - 41) / 37).reshape(n, k)
    raw[:, :16] = 0
    packed, striped, ts = nvfp4.quantize_expert_nvfp4_cpu(raw, partition, 8)
    codes, scales, expected_ts = quantize_nvfp4(raw.half().contiguous())
    assert packed.dtype == striped.dtype == torch.uint8 and ts.dtype == torch.float32
    assert packed.shape == (n, k//2) and striped.shape == (k//16, n) and ts.ndim == 0
    assert torch.equal(ts, expected_ts)
    # Independently invert K32-pair nibble grouping and TP controller interleave.
    pairs = packed.reshape(-1, 2, 16)
    physical = torch.stack((pairs & 15, pairs >> 4), dim=2)
    local_n, local_k = (n//8, k) if partition else (n, k//8)
    physical = physical.reshape(local_n//16, local_k//64, 8, 16, 64)
    restored = physical.permute(2, 0, 3, 1, 4) if partition else physical.permute(0, 3, 2, 1, 4)
    assert torch.equal(restored.reshape(n, k), codes)
    scale_physical = striped.reshape(local_k//32, 8, local_n//16, 2, 16)
    logical_scale = (scale_physical.permute(1, 2, 4, 0, 3) if partition
                     else scale_physical.permute(2, 4, 1, 0, 3)).reshape(n, k//16).t()
    assert torch.equal(logical_scale, scales)
    decoded = dequantize_nvfp4(codes, logical_scale, ts)
    assert torch.isfinite(decoded).all() and torch.equal(decoded[:, :16], torch.zeros_like(decoded[:, :16]))
    assert bool((decoded < 0).any()) and bool((decoded > 0).any())


def test_resident_pack_moves_only_core_dimension_and_preserves_all_byte_patterns():
    installed = (torch.arange(2048*512, dtype=torch.int32) % 256).to(torch.uint8).reshape(2048, 512)
    packed = nvfp4.pack_expert_gateup_nvfp4_installed(installed)
    assert packed.shape == (8, 131072) and packed.dtype == torch.uint8
    physical = installed.reshape(16, 16, 8, 512)
    for core in range(8):
        assert torch.equal(packed[core], physical[:, :, core].reshape(-1))
    assert torch.equal(packed.view(torch.float16).clone().view(torch.uint8), packed)
    for invalid in (installed.half(), installed.view(-1), installed[:, :-1]):
        with pytest.raises(ValueError, match="installed uint8"):
            nvfp4.pack_expert_gateup_nvfp4_installed(invalid)


@pytest.mark.parametrize("count", [0, 12, 24])
def test_expert_loader_owns_seven_tables_without_requantizing_resident_weights(monkeypatch, count):
    calls, quant_calls, uploads = [], [], []
    class Source:
        def __getitem__(self, key):
            calls.append(key)
            layer = int(key.split('.')[1])
            return torch.tensor([float(layer+1), float('gate_proj' in key), float('up_proj' in key)])
    template = (torch.arange(2048*512, dtype=torch.int32) % 256).to(torch.uint8).reshape(2048, 512)
    def quant(raw, partition, cores):
        index = len(quant_calls)
        quant_calls.append((raw.clone(), partition, cores))
        projection = PROJECTIONS[index % 7]
        packed = template + index if projection in ('gate', 'up') else torch.tensor([index % 256], dtype=torch.uint8)
        return packed, torch.tensor([index % 256], dtype=torch.uint8), torch.tensor(index+0.25, dtype=torch.float32)
    def upload(tensor):
        uploads.append((tensor.dtype, tuple(tensor.shape)))
        return tensor.clone()
    def forbidden(*args, **kwargs):
        pytest.fail('NVFP4 entered FP16/W8/W4 quantization or resident packing')
    monkeypatch.setattr(weights, 'quantize_expert_nvfp4_cpu', quant)
    monkeypatch.setattr(weights, '_rpu_nvfp4', upload)
    monkeypatch.setattr(weights, '_rpu16', lambda value: value.half().clone())
    monkeypatch.setattr(weights, '_dup_kv', lambda value, *a, **kw: value + torch.tensor([0., 10., 0.]))
    monkeypatch.setattr(weights, '_quant', forbidden)
    monkeypatch.setattr(weights, '_pack_expert_gateup_weight', forbidden)
    monkeypatch.setattr(weights, '_pack_expert_gateup_w8_installed', forbidden)
    cold = weights._HyVlaWeightColdPlan.from_environment(ENV | ({RES: '1', LAYERS: str(count)} if count else {}))
    result = weights._build_expert(Source(), Source(), weights.HyVlaConfig(), cold)
    assert len(quant_calls) == 32*7 and all(p[2] == 8 for p in quant_calls)
    for layer in range(32):
        for j, name in enumerate(PROJECTIONS):
            assert quant_calls[layer*7+j][1] == (0 if name in ('o', 'down') else 1)
            if name in ('k', 'v'):
                assert quant_calls[layer*7+j][0][1] == 10
    tables = []
    for j, name in enumerate(PROJECTIONS):
        assert len(result[name+'_w']) == len(result[name+'_ws']) == 32
        table, = result[name+'_ts']
        assert table.dtype == torch.float32 and table.shape == (32,)
        assert torch.equal(table, torch.arange(32, dtype=torch.float32)*7+j+0.25)
        tables.append(table)
    assert len({t.data_ptr() for t in tables}) == 7
    assert sum(dtype == torch.float32 and shape == (32,) for dtype, shape in uploads) == 7
    assert all(isinstance(value, list) and all(isinstance(t, torch.Tensor) for t in value) for value in result.values())
    assert all('_v.' in name or 'layernorm' in name for name in calls)
    for layer in range(32):
        for name in ('gate', 'up'):
            assert calls.count(f'layers.{layer}.mlp_v.{name}_proj.weight') == 1
    if count:
        carrier, = result['gateup_resident']
        assert carrier.shape == (8, count * 131072) and carrier.dtype == torch.float16
        assert carrier.numel() * carrier.element_size() == 8 * count * 262144
        payload = carrier.view(torch.uint8).reshape(8, count, 2, 131072)
        for layer in range(count):
            for j, name in enumerate(('gate', 'up')):
                installed = result[name + '_w'][layer]
                assert torch.equal(installed, template + layer * 7 + 4 + j)
                physical = installed.reshape(16, 16, 8, 512)
                for core in range(8):
                    assert torch.equal(payload[core, layer, j], physical[:, :, core].reshape(-1))
    else:
        assert result['gateup_resident'] == []


@pytest.mark.parametrize('bad', [torch.tensor(float('nan')), torch.tensor(0.), torch.tensor(-1.), torch.ones(2)])
def test_nonfinite_nonpositive_or_misshaped_tensor_scale_rejects_before_table_upload(monkeypatch, bad):
    class Source:
        def __getitem__(self, key): return torch.zeros(1)
    monkeypatch.setattr(weights, 'quantize_expert_nvfp4_cpu', lambda *args: (
        torch.zeros(1, dtype=torch.uint8), torch.zeros(1, dtype=torch.uint8), bad))
    monkeypatch.setattr(weights, '_rpu_nvfp4', lambda t: t.clone())
    monkeypatch.setattr(weights, '_rpu16', lambda t: t.half())
    monkeypatch.setattr(weights, '_dup_kv', lambda t, *args, **kw: t)
    cold = weights._HyVlaWeightColdPlan.from_environment(ENV)
    with pytest.raises(ValueError, match='positive finite FP32'):
        weights._build_expert(Source(), Source(), weights.HyVlaConfig(), cold)
