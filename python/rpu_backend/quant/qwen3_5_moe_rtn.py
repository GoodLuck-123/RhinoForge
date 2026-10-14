"""RTN recipe and source-bound decoder for the routed Qwen3.5 G128 profile.

The checkpoint uses GPTQ-v1 containers with zero-minus-one=7. The algorithm
is RTN; the official GPTQ direct-zero-8 source uses a separate decoder.
"""
from __future__ import annotations

import torch

from .grouped_uint4 import (
    _unpack_group128_matrix,
    pack_grouped_uint4,
    pack_uint4_qweight,
    runtime_group_scales,
)


def pack_rtn_qweight(codes: torch.Tensor) -> torch.Tensor:
    """Compatibility entry point for the RTN producer's UINT4 container."""
    return pack_uint4_qweight(codes)


def quantize_rtn_group128(weight: torch.Tensor) -> dict[str, torch.Tensor]:
    """Match the GPU reference: FP32 absmax*2/15 codes, then FP16 scales.

    This small-matrix helper also documents the source recipe. Deployment
    consumes the existing RTN checkpoint and never requantizes its codes.
    """
    if (weight.device.type != "cpu" or weight.dim() != 2
            or not weight.is_floating_point()):
        raise ValueError("RTN requires a CPU floating [N,K] matrix")
    n, k = weight.shape
    if n <= 0 or k <= 0 or k % 128 or n % 8:
        raise ValueError("RTN requires positive N%8==0 and K%128==0")
    source = weight.detach().float().reshape(n, k // 128, 128)
    if not bool(torch.isfinite(source).all()):
        raise ValueError("RTN source contains nonfinite values")
    amax = source.abs().amax(-1, keepdim=True)
    scale = torch.where(amax == 0, torch.full_like(amax, 2 / 15), amax * (2 / 15))
    codes = (torch.round(source / scale) + 8).clamp_(0, 15).to(torch.uint8)
    stored_scale = scale.squeeze(-1).t().to(torch.float16).contiguous()
    if not bool(torch.isfinite(stored_scale).all()):
        raise ValueError("RTN scale overflows FP16")
    return {
        "qweight": pack_rtn_qweight(codes.reshape(n, k)),
        "scales": stored_scale,
        "qzeros": torch.full((k // 128, n // 8), 0x77777777, dtype=torch.int32),
        "g_idx": torch.arange(k, dtype=torch.int32) // 128,
    }


def unpack_rtn_matrix(
    qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor,
    g_idx: torch.Tensor, *, n: int, k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the fixed RTN zero-minus-one/G128 serialization."""
    return _unpack_group128_matrix(
        qweight, scales, qzeros, g_idx, n=n, k=k,
        zero_word=0x77777777, source_name="RTN")
