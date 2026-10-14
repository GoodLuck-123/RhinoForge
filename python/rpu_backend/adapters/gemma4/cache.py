"""Gemma4 mixed-width KV cache (T1c).

``Gemma4KVCache`` subclasses ``RPUCache`` to inherit the HuggingFace ``Cache``
interface (get_seq_length / update / reset / get_mask_sizes / position tracking)
that ``model.generate`` needs, but REPLACES the uniform single-geometry slot
allocation with a per-layer mixed-width plan (sliding head_dim=256 / global 512)
and a forced source-map: shared layers (idx >= first shared) allocate NO slot of
their own and instead ALIAS their source layer's slot tensor.

This matches the C++ ``Gemma4Model::build_layer_subgraph`` contract: it inserts
K/V only for non-shared layers (into ``k_caches[layer_idx]``) and SDPA reads
``k_caches[geom.kv_source_layer]`` — so the 42-entry list passed to
``gemma4_forward`` must have valid tensors at every source index. Aliasing keeps
all 42 entries valid (a shared entry IS its source tensor object).

Slot shapes come from the pure ``cache_plan`` module (offline-testable); they
mirror RPUCache's 7-D swizzle layout. Importing this module needs the rpu_backend
package (sudo on this host); the pure plan lives in cache_plan.py.
"""
from __future__ import annotations

from typing import Any

import torch

from rpu_backend.api.cache import RPUCache
from rpu_backend.runtime.weights import NUM_CORES

from .cache_plan import cache_plan
from .geometry import layer_geometry, max_head_dim, max_num_kv_heads


def _reset_generation_prefix(cache, model, token_count: int) -> bool:
    """Clear every physical KV lane a bounded generation can read."""
    if (type(cache) is not Gemma4KVCache or cache.batch_size != 1
            or type(cache.max_seq_len) is not int
            or not 0 < token_count <= cache.max_seq_len
            or cache.geom != getattr(model, "_rpu_gemma4_geom", None)):
        return False
    try:
        if cache.attn_tp != min(NUM_CORES, max_num_kv_heads(cache.geom)):
            return False
        plan = cache_plan(cache.geom, cache.max_seq_len, cache.attn_tp)
    except (AttributeError, TypeError, ValueError, ZeroDivisionError):
        return False
    if (cache.num_layers != plan["num_layers"]
            or cache._owner_layers != plan["owners"]
            or cache.kv_source_layer_for_layer != plan["source_map"]
            or len(cache.k_caches) != plan["num_layers"]
            or len(cache.v_caches) != plan["num_layers"]):
        return False

    owners, ranges = [], []
    for layer in plan["layers"]:
        for tensors, shape in ((cache.k_caches, layer["k_shape"]),
                               (cache.v_caches, layer["v_shape"])):
            tensor = tensors[layer["idx"]]
            if tensor is not tensors[layer["source"]]:
                return False
            if not layer["owns"]:
                continue
            if (not isinstance(tensor, torch.Tensor)
                    or tensor.device.type != "rpu" or tensor.dtype != torch.float16
                    or tuple(tensor.shape) != shape or not tensor.is_contiguous()
                    or tensor.storage_offset() != 0 or tensor.requires_grad
                    or (tensor.is_inference() and not torch.is_inference_mode_enabled())):
                return False
            storage = tensor.untyped_storage()
            if (tensor.nbytes != storage.nbytes() or tensor.data_ptr() % 256
                    or tensor.nbytes % 256):
                return False
            owners.append(tensor)
            ranges.append((tensor.data_ptr(), tensor.data_ptr() + tensor.nbytes))
    ranges.sort()
    if not owners or any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
        return False

    # Sequence blocks are outermost after B=1. Include all lanes and the last
    # 16-row SDPA read block, without relying on prefill overwrites or masking.
    blocks = (token_count + 15) // 16
    views = [tensor[:, :blocks].view(1, 1, -1) for tensor in owners]
    torch.ops.rpu.zero_contiguous_tensor_blocks_sized(views, 0)
    RPUCache.reset_to_position(cache, 0)
    return True


class Gemma4KVCache(RPUCache):
    """Per-layer mixed-width KV cache with a forced cross-layer source-map."""

    # Sliding-window visibility is enforced in the fused C++ mask (T4), not via
    # the HF cache type; keep False like RPUCache so HF masking stays bypassed.
    is_sliding_window = False

    def __init__(
        self,
        geom: list[Any],
        max_seq_len: int,
        batch_size: int = 1,
        device: str = "rpu",
        dtype: torch.dtype = torch.float16,
        attn_tp: int | None = None,
    ):
        if batch_size != 1:
            raise NotImplementedError(
                f"Gemma4KVCache batch_size={batch_size} unsupported (only 1).")

        # attn_tp matches the C++ Gemma4Model: min(NUM_CORES, num_kv_heads_max).
        if attn_tp is None:
            attn_tp = min(NUM_CORES, max_num_kv_heads(geom))

        # Intentionally bypass RPUCache.__init__ (uniform allocation): set the
        # state RPUCache's inherited methods read, then do per-layer allocation.
        self.geom = geom
        self.num_layers = len(geom)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.attn_tp = attn_tp
        # Incidental single-geometry attrs (max width) for any inherited method
        # that still references them; the real per-layer geometry is self.geom.
        self.head_dim = max_head_dim(geom)
        self.num_kv_heads = max_num_kv_heads(geom)

        plan = cache_plan(geom, max_seq_len, attn_tp, batch_size, NUM_CORES)
        # Forced source-map (T1c contract): always produced.
        self.kv_source_layer_for_layer: list[int] = plan["source_map"]
        self._owner_layers: list[int] = plan["owners"]

        # Allocate a distinct slot for each OWNER (non-shared) layer; SHARED
        # layers alias their source slot (same tensor object). C++ reads
        # k_caches[kv_source_layer] for SDPA and inserts only for owners.
        self.k_caches: list[torch.Tensor] = [None] * self.num_layers  # type: ignore[list-item]
        self.v_caches: list[torch.Tensor] = [None] * self.num_layers  # type: ignore[list-item]
        for L in plan["layers"]:
            if L["owns"]:
                self.k_caches[L["idx"]] = torch.zeros(
                    L["k_shape"], dtype=dtype, device=device)
                self.v_caches[L["idx"]] = torch.zeros(
                    L["v_shape"], dtype=dtype, device=device)
        for L in plan["layers"]:
            if not L["owns"]:
                src = L["source"]
                self.k_caches[L["idx"]] = self.k_caches[src]  # alias source slot
                self.v_caches[L["idx"]] = self.v_caches[src]

        self.position = 0
        self._seen_tokens = 0
        self.layers = []  # HF Cache compat (data lives in k_caches/v_caches)

    def reset(self):
        """Zero all DISTINCT slots (shared aliases share their source tensor)."""
        self.position = 0
        self._seen_tokens = 0
        for i in self._owner_layers:
            self.k_caches[i].zero_()
            self.v_caches[i].zero_()

    def to_dynamic_cache(self, device="cpu"):
        raise NotImplementedError(
            "Gemma4KVCache mixed-width unswizzle is not implemented: per-layer "
            "head_dim differs (256/512) and KV-share aliases slots, so a single "
            "uniform unswizzle is wrong. Not needed for E4B greedy text gen.")

    @classmethod
    def from_config(cls, text_config: Any, max_seq_len: int, **kwargs) -> "Gemma4KVCache":
        """Build from a Gemma4 ``text_config`` (HF config or the JSON dict)."""
        return cls(layer_geometry(text_config), max_seq_len, **kwargs)

    def __repr__(self) -> str:
        return (f"Gemma4KVCache(num_layers={self.num_layers}, "
                f"owners={len(self._owner_layers)}, "
                f"max_seq_len={self.max_seq_len}, position={self.position})")
