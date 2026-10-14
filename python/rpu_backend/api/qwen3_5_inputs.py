"""Shared CPU embedding admission for dense and MoE Qwen3.5 adapters."""
from __future__ import annotations

import torch

# Trust the installed standard PyTorch delegates at adapter import. Runtime
# producer replacements must retain Module semantics through the public route.
_DECODE_EMBEDDING_FORWARD = torch.nn.Embedding.forward
_DECODE_MODULE_CALL = torch.nn.Module.__call__
_DECODE_MODULE_CALL_IMPL = torch.nn.Module._call_impl
_DECODE_MODULE_GETATTRIBUTE = torch.nn.Module.__getattribute__
_DECODE_MODULE_GETATTR = torch.nn.Module.__getattr__
_DECODE_FUNCTIONAL_EMBEDDING = torch.nn.functional.embedding

def _ordinary_decode_embedding_cpu(fusion, embedding, input_ids, position):
    """Fresh CPU gather for plain decode or initial text prefill."""
    if (type(input_ids) is not torch.Tensor or input_ids.device.type != "cpu"
            or input_ids.dtype != torch.int64 or input_ids.ndim != 2
            or input_ids.shape[0] != 1 or not input_ids.is_contiguous()):
        return None
    decode = input_ids.shape[1] == 1 and int(position) > 0
    prefill = input_ids.shape[1] > 1 and int(position) == 0
    if not (decode or prefill):
        return None
    vision = getattr(fusion, "visual", None)
    if (any(getattr(vision, name, None) is not None for name in
            ("_rpu_vision_handle", "_rpu_vision_installing_handle"))
            or getattr(fusion, "rope_deltas", None) is not None):
        return None
    inner = getattr(fusion, "language_model", fusion)
    state = getattr(inner, "_rpu_qwen3_5", None)
    moe_prefill = prefill and state is None
    if state is None:
        state = getattr(inner, "_rpu_qwen3_5_moe", None)
    if (getattr(state, "adaptive_mode", "unbound") is not None
            or getattr(state, "action_mode", False)):
        return None
    # Bypassing Module.__call__ must not bypass custom embedding producers/hooks.
    # A9 stamps a known subclass over the original class; other subclasses stay
    # on the public producer path, including dynamic __getattribute__ overrides.
    from rpu_backend.runtime.hw_attrs import _VALIDATED_CLS_CACHE
    module = torch.nn.modules.module
    if (type(embedding) not in
            (torch.nn.Embedding, _VALIDATED_CLS_CACHE.get(torch.nn.Embedding))
            or type(embedding).forward is not _DECODE_EMBEDDING_FORWARD
            or type(embedding).__call__ is not _DECODE_MODULE_CALL
            or type(embedding)._call_impl is not _DECODE_MODULE_CALL_IMPL
            or type(embedding).__getattribute__ is not _DECODE_MODULE_GETATTRIBUTE
            or type(embedding).__getattr__ is not _DECODE_MODULE_GETATTR
            or torch.nn.functional.embedding is not _DECODE_FUNCTIONAL_EMBEDDING
            or any(name in vars(embedding) for name in
                   ("forward", "_call_impl", "_slow_forward"))
            or embedding.max_norm is not None
            or getattr(embedding, "_compiled_call_impl", None) is not None
            or torch._C._get_tracing_state() is not None
            or any(getattr(embedding, name, None) for name in
                   ("_forward_pre_hooks", "_forward_hooks",
                    "_backward_pre_hooks", "_backward_hooks"))
            or any(getattr(module, name, None) for name in
                   ("_global_forward_pre_hooks", "_global_forward_hooks",
                    "_global_backward_pre_hooks", "_global_backward_hooks"))):
        return None
    weight = embedding.weight
    if (type(weight) not in (torch.Tensor, torch.nn.Parameter)
            or weight.device.type != "rpu" or weight.dtype != torch.float16
            or not weight.is_contiguous() or not torch.rpu.get_ddr_flush_force()):
        return None
    if prefill and (input_ids.is_neg() or input_ids.is_conj()
                    or weight.is_neg() or weight.is_conj()
                    or (moe_prefill and (input_ids.storage_offset() != 0
                                        or weight.storage_offset() != 0))):
        return None
    return torch.ops.rpu.qwen3_5_decode_embedding_cpu(
        weight, input_ids, -1 if embedding.padding_idx is None else embedding.padding_idx,
        embedding.scale_grad_by_freq, embedding.sparse)
