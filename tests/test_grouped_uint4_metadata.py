"""Independent element-address checks for the current grouped UINT4 metadata ABI."""
import numpy as np
import pytest
import torch

from rpu_backend.quant.grouped_uint4 import pack_grouped_uint4_metadata


def _metadata_addresses(*, experts, global_q, global_n, partition, cores):
    """Index the documented DDR stripes, without undoing the packer's reshape."""
    local_q = global_q if partition == "col" else global_q // cores
    local_n = global_n // cores if partition == "col" else global_n
    q_blocks, n_blocks = (local_q + 3) // 4, (local_n + 63) // 64
    e, q, n = np.indices((experts, global_q, global_n), dtype=np.int64)
    core = n // local_n if partition == "col" else q // local_q
    q, n = q % local_q, n % local_n
    # Byte offsets are 512 per core, 512*cores per N block, and
    # 512*cores*n_blocks per group block. Inner FP16 offsets are /2.
    expert_step = q_blocks * n_blocks * cores * 512
    offsets = (e * expert_step + (q // 4) * n_blocks * cores * 512
               + (n // 64) * cores * 512 + core * 512
               + (q % 4) * 128 + (n % 64) * 2)
    return offsets // 2


def inverse_metadata(physical, *, global_q, global_n, partition, cores=8):
    """Return canonical [E,Q,N] from independently calculated DDR addresses."""
    addresses = _metadata_addresses(experts=physical.shape[0], global_q=global_q,
                                    global_n=global_n, partition=partition, cores=cores)
    words = physical.numpy().view(np.uint16).reshape(-1)
    return torch.from_numpy(words[addresses].copy().view(np.float16))


@pytest.mark.parametrize("partition,cores,local_q,local_n", [
    ("col", 8, 16, 64),  # real Gate/Up
    ("row", 8, 1, 2048),  # real TP8 Down G64; three zero Q slots per stripe
    ("col", 8, 1, 16), ("col", 8, 3, 48), ("col", 8, 5, 80),
    ("row", 8, 2, 16), ("row", 8, 3, 48), ("row", 8, 5, 80),
    ("col", 1, 4, 128), ("row", 1, 4, 128),
    ("col", 4, 5, 80), ("row", 4, 5, 80),
])
def test_metadata_independent_addresses_preserve_bits_and_zero_tail(partition, cores, local_q, local_n):
    e = 2
    q = local_q if partition == "col" else local_q * cores
    n = local_n * cores if partition == "col" else local_n
    # Exercise FP16 bits (including signed zero and subnormals), not a numerical
    # conversion whose rounding could hide permutations of large integers.
    words = ((np.arange(e*q*n, dtype=np.uint32) * 73) % 0x7c00).astype(np.uint16).reshape(e, q, n)
    words[0, 0, 0] = 0x8000
    source = torch.from_numpy(words.view(np.float16))
    physical = pack_grouped_uint4_metadata(source, partition=partition, num_cores=cores)
    assert physical.is_contiguous() and physical.dtype == torch.float16
    assert physical.shape == (e, (local_q+3)//4, (local_n+63)//64, cores, 4, 64)
    restored = inverse_metadata(physical, global_q=q, global_n=n, partition=partition, cores=cores)
    assert torch.equal(restored.view(torch.int16), source.view(torch.int16))
    addresses = _metadata_addresses(experts=e, global_q=q, global_n=n, partition=partition, cores=cores)
    touched = np.zeros(physical.numel(), dtype=bool)
    touched[addresses.reshape(-1)] = True
    assert int(touched.sum()) == source.numel()
    assert np.all(physical.numpy().view(np.uint16).reshape(-1)[~touched] == 0)


def test_metadata_noncontiguous_source_and_negative_zero_use_identical_stripes():
    source = torch.arange(2*128*8, dtype=torch.float16).view(2, 128, 8).transpose(1, 2)
    assert not source.is_contiguous()
    packed = pack_grouped_uint4_metadata(source, partition="row")
    assert torch.equal(inverse_metadata(packed, global_q=8, global_n=128, partition="row"), source)
    negative_zero = pack_grouped_uint4_metadata(torch.full_like(source, -8), partition="row")
    assert torch.all(negative_zero[:, :, :, :, 0, :] == -8)
    assert torch.all(negative_zero[:, :, :, :, 1:, :].view(torch.int16) == 0)


@pytest.mark.parametrize("shape,dtype,partition,cores,message", [
    ((2, 4, 128), torch.float32, "col", 8, "CPU FP16"),
    ((4, 128), torch.float16, "col", 8, "CPU FP16"),
    ((0, 4, 128), torch.float16, "col", 8, "nonempty"),
    ((2, 4, 127), torch.float16, "col", 8, "N%"),
    ((2, 7, 128), torch.float16, "row", 8, "Q%"),
    ((2, 8, 127), torch.float16, "row", 8, "N%"),
    ((2, 8, 128), torch.float16, "other", 8, "partition"),
    ((2, 8, 128), torch.float16, "col", True, "integer num_cores"),
    ((2, 8, 128), torch.float16, "col", 0, "integer num_cores"),
    ((2, 8, 128), torch.float16, "col", 9, "integer num_cores"),
])
def test_metadata_invalid_domain_rejected(shape, dtype, partition, cores, message):
    with pytest.raises(ValueError, match=message):
        pack_grouped_uint4_metadata(torch.zeros(shape, dtype=dtype), partition=partition, num_cores=cores)
