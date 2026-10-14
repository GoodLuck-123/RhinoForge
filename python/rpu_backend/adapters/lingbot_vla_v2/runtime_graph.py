"""Cold weight transforms for the LingBot v2 runtime graph profile."""

from __future__ import annotations

import torch


@torch.no_grad()
def fold_lingbot_language_rmsnorm_(text_model) -> None:
    """Fold trunk RMSNorm gains into their consumers before half/quantization.

    Q/K head normalization remains explicit. Trunk gain tensors become units;
    the cold native profile omits them and binds the MAIN no-gamma payload.
    Validate the entire model first so an invalid late layer cannot leave an
    earlier layer folded. This transform is deliberately not idempotent.
    """
    if getattr(text_model, "_lingbot_runtime_norm_folded", False):
        raise ValueError("LingBot language RMSNorm gains were already folded")
    if getattr(text_model, "_rpu_decoder_handle", None) is not None:
        raise ValueError("LingBot language RMSNorm fold must precede native installation")
    layers = getattr(text_model, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("LingBot language RMSNorm fold requires decoder layers")

    work = []
    owners = set()
    for index, layer in enumerate(layers):
        for norm_name, consumers in (
            ("input_layernorm", ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj")),
            ("post_attention_layernorm", ("mlp.gate_proj", "mlp.up_proj")),
        ):
            norm = layer.get_submodule(norm_name)
            gamma = getattr(norm, "weight", None)
            label = f"layer {index} {norm_name}"
            if (not isinstance(gamma, torch.Tensor) or gamma.device.type != "cpu"
                    or gamma.dtype != torch.float32 or gamma.ndim != 1
                    or not gamma.is_contiguous() or not torch.isfinite(gamma).all()):
                raise ValueError(f"{label}: fold requires finite contiguous CPU FP32 gain")
            owner = gamma.untyped_storage().data_ptr()
            if owner in owners:
                raise ValueError("LingBot language fold rejects shared gain or projection storage")
            owners.add(owner)
            projections = []
            for name in consumers:
                projection = layer.get_submodule(name)
                weight = getattr(projection, "weight", None)
                if (not isinstance(weight, torch.Tensor) or weight.device.type != "cpu"
                        or weight.dtype != torch.float32 or weight.ndim != 2
                        or weight.shape[1] != gamma.numel() or not weight.is_contiguous()
                        or getattr(projection, "weight_scale", None) is not None
                        or getattr(projection, "_rpu_weight_swizzled", False)):
                    raise ValueError(f"{label} -> {name}: fold requires unpacked CPU FP32 weight")
                owner = weight.untyped_storage().data_ptr()
                if owner in owners:
                    raise ValueError("LingBot language fold rejects shared projection storage")
                owners.add(owner)
                projections.append(weight)
            work.append((gamma, projections))

    for gamma, projections in work:
        for weight in projections:
            weight.mul_(gamma.unsqueeze(0))
        gamma.fill_(1)
    text_model._lingbot_runtime_norm_folded = True
