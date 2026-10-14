import os
import sys
from pathlib import Path
import numpy as np
import pytest
import torch
_QUANT = Path(__file__).resolve().parents[2] / 'quant'
sys.path.insert(0, str(_QUANT))
import _common as _qc
import int4_pgrp_pack as _pg

def _ref_pack_int4_signed(flat_int8: np.ndarray) -> np.ndarray:
    a = flat_int8.astype(np.int8)
    if a.size % 2 != 0:
        a = np.append(a, np.int8(0))
    a = np.where(a < 0, a + 16, a).astype(np.uint8) & 15
    return (a[1::2] << 4 | a[0::2]).astype(np.uint8)

def _ref_swizzle(vals: torch.Tensor, partition: int, num_cores: int) -> torch.Tensor:
    (N, K) = vals.shape
    e = 64
    if partition == 1:
        v = vals.view(num_cores, N // 16 // num_cores, 16, K // e, e).permute(1, 3, 0, 2, 4)
    else:
        v = vals.view(N // 16, 16, num_cores, K // e // num_cores, e).permute(0, 3, 2, 1, 4)
    return v.contiguous().reshape(-1)

def _ref_pgrp_swizzle(vals: torch.Tensor, partition: int, num_cores: int) -> torch.Tensor:
    swizzled = _ref_swizzle(vals, partition, num_cores).view(-1, 64)
    perm = []
    for block in range(2):
        for i in range(16):
            perm.extend((block * 32 + i, block * 32 + 16 + i))
    return swizzled[:, perm].contiguous().reshape(-1)

@pytest.mark.parametrize('partition,n,k', [(1, 256, 128), (0, 256, 512)])
def test_pgrp_swizzle_pack_matches_reference(partition, n, k):
    torch.manual_seed(0)
    (N, K, cores) = (n, k, 8)
    vals = torch.randint(-8, 8, (N, K), dtype=torch.int8)
    packed = _pg.swizzle_pack_int4_pgrp(vals, partition, cores)
    golden = _ref_pack_int4_signed(_ref_pgrp_swizzle(vals.to(torch.int64), partition, cores).numpy())
    assert packed.dtype == torch.uint8
    assert packed.numel() == N * K // 2
    np.testing.assert_array_equal(packed.numpy(), golden)

def test_quantize_int4_per_channel_ranges():
    torch.manual_seed(0)
    W = torch.randn(64, 128).to(torch.float16)
    (vals, scale) = _qc.quantize_linear_per_channel(W, bits=4)
    assert vals.dtype == torch.int8 and scale.dtype == torch.float16
    assert int(vals.max()) <= 7 and int(vals.min()) >= -8
    assert scale.numel() == 64

def test_per_channel_scale_is_repeated_into_pgrp_payload():
    torch.manual_seed(4)
    vals = torch.randint(-8, 8, (256, 512), dtype=torch.int8)
    scale = torch.arange(1, 257, dtype=torch.float16)
    (packed, physical) = _pg.pack_int4_per_channel_as_pgrp(vals, scale, partition=0, num_cores=8)
    logical = scale.reshape(1, -1).expand(16, -1).contiguous()
    expected_scale = _pg.swizzle_int4_pgrp_scale(logical, group_size=32, partition=0, num_cores=8)
    assert torch.equal(packed, _pg.swizzle_pack_int4_pgrp(vals, partition=0, num_cores=8))
    assert torch.equal(physical, expected_scale)
    assert physical.size(0) == 32

@pytest.mark.parametrize('partition,group_size,g,n', [(1, 32, 5, 130), (0, 64, 10, 65), (1, 128, 5, 130)])
def test_pgrp_scale_swizzle_physical_order_and_metadata(partition, group_size, g, n):
    cores = 2
    logical = torch.arange(1, g * n + 1, dtype=torch.float16).reshape(g, n)
    actual = _pg.swizzle_int4_pgrp_scale(logical, group_size, partition, cores)
    local_g = g if partition == 1 else g // cores
    local_n = n // cores if partition == 1 else n
    expected = torch.zeros(((local_g + 3) // 4, (local_n + 63) // 64, cores, 4, 64), dtype=torch.float16)
    for global_g in range(g):
        for global_n in range(n):
            core = global_n // local_n if partition == 1 else global_g // local_g
            lg = global_g if partition == 1 else global_g % local_g
            ln = global_n % local_n if partition == 1 else global_n
            expected[lg // 4, ln // 64, core, lg % 4, ln % 64] = logical[global_g, global_n]
    assert actual.shape == (group_size, expected.numel() // group_size)
    assert actual.is_contiguous()
    assert torch.equal(actual.flatten(), expected.flatten())
    assert torch.count_nonzero(actual) == logical.numel()

def test_pgrp_scale_swizzle_rejects_invalid_abi():
    scale = torch.ones((5, 130), dtype=torch.float16)
    with pytest.raises(ValueError, match='group_size'):
        _pg.swizzle_int4_pgrp_scale(scale, 16, 1, 2)
    with pytest.raises(ValueError, match='N%cores'):
        _pg.swizzle_int4_pgrp_scale(scale, 32, 1, 3)
    with pytest.raises(ValueError, match='G%cores'):
        _pg.swizzle_int4_pgrp_scale(scale, 32, 0, 2)
    with pytest.raises(ValueError, match='fp16'):
        _pg.swizzle_int4_pgrp_scale(scale.float(), 32, 1, 2)
