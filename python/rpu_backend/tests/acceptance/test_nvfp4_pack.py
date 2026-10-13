import sys
from pathlib import Path
import torch
import pytest
_QUANT = Path(__file__).resolve().parents[2] / 'quant'
sys.path.insert(0, str(_QUANT))
from nvfp4_pack import dequantize_nvfp4, fp4_e2m1_decode, fp4_e2m1_encode, quantize_nvfp4, swizzle_pack_nvfp4, swizzle_nvfp4_scales

def test_fp4_all_codes_roundtrip():
    codes = torch.arange(16, dtype=torch.uint8)
    assert torch.equal(fp4_e2m1_encode(fp4_e2m1_decode(codes)), codes)

def test_fp4_ties_round_to_even_mantissa():
    x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
    assert fp4_e2m1_encode(x).tolist() == [0, 2, 2, 4, 4, 6, 6]

def test_nvfp4_quant_shapes_and_pack_size():
    weight = torch.randn(256, 1024, generator=torch.Generator().manual_seed(7))
    (codes, scales, tensor_scale) = quantize_nvfp4(weight)
    dequant = dequantize_nvfp4(codes, scales, tensor_scale)
    packed = swizzle_pack_nvfp4(codes, partition=1, num_cores=8)
    assert codes.shape == weight.shape
    assert scales.shape == (64, 256)
    assert scales.dtype == torch.uint8
    assert tensor_scale.shape == ()
    assert tensor_scale.dtype == torch.float32
    assert dequant.shape == weight.shape
    assert packed.dtype == torch.uint8
    assert packed.numel() == weight.numel() // 2

def test_nvfp4_dequant_reconstructs_the_weight():
    weight = torch.randn(256, 1024, generator=torch.Generator().manual_seed(7))
    (codes, scales, tensor_scale) = quantize_nvfp4(weight)
    dequant = dequantize_nvfp4(codes, scales, tensor_scale)
    cos = torch.nn.functional.cosine_similarity(weight.flatten(), dequant.flatten(), dim=0).item()
    rel_l2 = ((dequant - weight).norm() / weight.norm()).item()
    assert cos > 0.99, f'NVFP4 dequant cos={cos:.6f} vs the fp32 weight'
    assert rel_l2 < 0.15, f'NVFP4 dequant relative L2 error={rel_l2:.4f}'

def test_nvfp4_pack_preserves_every_code_and_actually_swizzles():
    weight = torch.randn(256, 1024, generator=torch.Generator().manual_seed(7))
    (codes, _, _) = quantize_nvfp4(weight)
    packed = swizzle_pack_nvfp4(codes, partition=1, num_cores=8)
    nibbles = torch.stack([packed & 15, packed >> 4 & 15], dim=1).reshape(-1)
    flat = codes.reshape(-1) & 15
    assert torch.equal(torch.bincount(flat.long(), minlength=16), torch.bincount(nibbles.long(), minlength=16)), 'swizzle_pack_nvfp4 changed the multiset of fp4 codes'
    assert not torch.equal(nibbles, flat), 'packed nibbles are in row-major order — the TP swizzle was a no-op'

@pytest.mark.parametrize('cores', [1, 2, 4, 6, 8])
@pytest.mark.parametrize('partition', [0, 1])
def test_tiled_nvfp4_payload_decodes_by_kernel_coordinates(partition, cores):
    n = 48 * cores if partition else 48
    k = 192 if partition else 192 * cores
    (ni, ki) = torch.meshgrid(torch.arange(n), torch.arange(k), indexing='ij')
    codes = ((ni * 7 + ki // 16 * 3 + ki % 16) % 16).byte()
    packed = swizzle_pack_nvfp4(codes, partition, cores)
    (local_n, local_k) = (n // cores, k) if partition else (n, k // cores)
    core = ni // local_n if partition else ki // local_k
    local_ni = ni % local_n if partition else ni
    local_ki = ki if partition else ki % local_k
    weight_byte = (((local_ni // 16 * (local_k // 64) + local_ki // 64) * cores + core) * 16 + local_ni % 16) * 32
    weight_byte += local_ki % 64 // 32 * 16 + local_ki % 16
    recovered = packed[weight_byte] >> 4 * (local_ki % 32 // 16) & 15
    assert torch.equal(recovered, codes)
    scales = (torch.arange(k // 16 * n).reshape(k // 16, n) % 251).byte()
    striped = swizzle_nvfp4_scales(scales, partition, cores)
    scale_byte = (((local_ki // 32 * cores + core) * (local_n // 16) + local_ni // 16) * 2 + local_ki % 32 // 16) * 16 + local_ni % 16
    recovered_scale = striped.reshape(-1)[scale_byte]
    assert torch.equal(recovered_scale, scales[ki // 16, ni])

@pytest.mark.parametrize('shape,partition,cores', [((4, 48), 1, 2), ((6, 32), 0, 2), ((4, 32), 2, 2), ((4, 32), 1, 0)])
def test_nvfp4_scale_packing_rejects_incompatible_geometry(shape, partition, cores):
    with pytest.raises(ValueError):
        swizzle_nvfp4_scales(torch.zeros(shape, dtype=torch.uint8), partition, cores)
