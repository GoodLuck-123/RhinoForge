"""Byte-preserving GU4/GU8 FP8 scale packing; no quantization or device operations.

Input tensors already use swizzle_nvfp4_scales' physical controller stripes.
Carrier Half values are opaque DMA bytes, not FP16 numerical values.
"""
from collections.abc import Sequence
import torch

SUPPORTED_LAYER_COUNTS = (4, 8)
CORES = 8
PROJECTION_BYTES_PER_CORE = 16384
LAYER_BYTES_PER_CORE = 32768


def _installed_scale(value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or \
            value.dtype != torch.uint8 or tuple(value.shape) != (64, 2048) or not value.is_contiguous():
        raise ValueError('HyVLA resident scales require installed contiguous CPU Byte [64,2048]')


def pack_hyvla_nvfp4_resident_scales(
    gate_scales: Sequence[torch.Tensor], up_scales: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Pack four or eight layers to opaque contiguous CPU Half [8,layers*16384].

    Byte layout is [core,layer,Gate/Up,K32,N16,K16-pair,output-lane]. A layer
    occupies32768 bytes/core; Up starts16384 bytes after Gate. The installed
    source layout is [K32,core,N16,K16-pair,output-lane].
    """
    layers = len(gate_scales)
    if layers not in SUPPORTED_LAYER_COUNTS or len(up_scales) != layers:
        raise ValueError('HyVLA resident scale bundle requires exactly four or eight matching Gate and Up layers')
    result = torch.empty((CORES,layers,2,PROJECTION_BYTES_PER_CORE),dtype=torch.uint8)
    for layer in range(layers):
        for projection, values in enumerate((gate_scales,up_scales)):
            scale = values[layer]
            _installed_scale(scale)
            result[:,layer,projection].copy_(scale.reshape(32,8,16,2,16)
                .permute(1,0,2,3,4).reshape(8,PROJECTION_BYTES_PER_CORE))
    return result.reshape(CORES,layers*LAYER_BYTES_PER_CORE).view(torch.float16)


def unpack_hyvla_nvfp4_resident_scales(
    carrier: torch.Tensor
) -> tuple[list[torch.Tensor],list[torch.Tensor]]:
    """Invert an exact four/eight-layer bundle to installed Gate and Up tensors."""
    if not isinstance(carrier,torch.Tensor) or carrier.device.type != 'cpu' or \
            carrier.dtype != torch.float16 or tuple(carrier.shape) not in ((8,65536),(8,131072)) or not carrier.is_contiguous():
        raise ValueError('HyVLA resident scale carrier requires contiguous CPU Half [8,65536] or [8,131072]')
    layers = carrier.shape[1]*carrier.element_size()//LAYER_BYTES_PER_CORE
    data = carrier.view(torch.uint8).reshape(CORES,layers,2,PROJECTION_BYTES_PER_CORE)
    gate, up = [], []
    for layer in range(layers):
        for projection,target in enumerate((gate,up)):
            target.append(data[:,layer,projection].reshape(8,32,16,2,16)
                .permute(1,0,2,3,4).contiguous().reshape(64,2048))
    return gate,up
