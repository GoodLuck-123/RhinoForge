"""Cold, lossless packing for the exact HyVLA W8 resident-O experiment."""
from collections.abc import Sequence

import torch


O_WEIGHT_BYTES_PER_CORE = 262144
O_RESIDENT_LAYERS = 12


def pack_expert_o_w8_installed(weight: torch.Tensor) -> torch.Tensor:
    """Installed row-TP8 INT8 [1024,2048] DDR bytes -> [core,262144]."""
    if not isinstance(weight, torch.Tensor) or weight.dtype != torch.int8 or tuple(weight.shape) != (1024, 2048):
        raise ValueError("HyVLA resident O requires installed INT8 [1024,2048]")
    # Installed row layout is [N/16, localK/32, core, 16*32]. Remove only
    # controller interleave; preserve the original quantized bytes and scales.
    return (weight.cpu().contiguous().reshape(64, 8, 8, 512)
            .permute(2, 0, 1, 3).contiguous().reshape(8, O_WEIGHT_BYTES_PER_CORE))


def pack_expert_o_w8_layers(weights: Sequence[torch.Tensor]) -> torch.Tensor:
    """Twelve installed O weights -> [8,1572864] opaque half DMA carrier."""
    if len(weights) != O_RESIDENT_LAYERS:
        raise ValueError("HyVLA resident O requires exactly twelve installed weights")
    # view changes storage interpretation only: no numerical half conversion.
    packed = torch.stack([pack_expert_o_w8_installed(w) for w in weights], dim=1)
    return packed.reshape(8, -1).view(torch.float16)
