"""Model execution and tensor layout contracts."""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

import rpu_backend
from rpu_backend.adapters.halo.runtime import _halo_execution_serialized as execution_serialized
from rpu_backend.api.cache import RPUCache
from rpu_backend.adapters.halo.runtime import (
    HALO_TEXT_CACHE_COMPONENT,
    HaloExecutionController,
    _HaloResources,
    _cleanup_halo_build,
    _close_halo_runner,
    _S_CHUNK,
    _coerce_halo_execution_controller,
    _plan_halo_fmb_execution,
    _record_halo_plan,
    _swizzle_prefix_7d,
)
from rpu_backend.adapters.halo.weights import (
    validate_halo_core_layout,
    HaloBranchWeights,
    HaloConfig,
    HaloHostWeights,
    load_halo_branches,
)


def _destroy_causal_decoder(handle: int) -> None:
    """释放通用 CausalDecoderModel handle（weakref.finalize 在 GC 时调）。"""
    torch.ops.rpu.causal_decoder_destroy(handle)


def _to_rpu_half(t: torch.Tensor) -> torch.Tensor:
    return t.to(dtype=torch.float16, device="rpu").contiguous()


def _dynamic_layer_kv(dc: object, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Model execution and tensor layout contracts."""
    layers = getattr(dc, "layers", None)
    if layers is not None:
        return layers[layer].keys, layers[layer].values
    return dc.key_cache[layer], dc.value_cache[layer]  # type: ignore[attr-defined]


def build_seq_rope_tables(cfg: HaloConfig, max_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """**顺序** position 的 1D RoPE 表 ``[max_seq_len, head_dim/2]`` fp16 device（kernel 格式）。

    行 p = position p 的 rotary 基（``inv_freq`` theta=1e6，与 transformers Qwen2
    默认 rope、attention_scaling=1.0 一致）。fp64 构造再降 fp16 匹配 golden。
    **与 weights.build_rope_tables 的 act 常数表不同**：文本 prefill position 递增，
    kernel 用 ``cos_sin_start=position`` 索引此表（见模块 docstring 的 1D RoPE 说明）。
    """
    if max_seq_len <= 0 or max_seq_len % _S_CHUNK != 0:
        raise ValueError(
            f"max_seq_len {max_seq_len} must be a positive multiple of {_S_CHUNK}"
        )
    if cfg.head_dim <= 0 or cfg.head_dim % 2 != 0:
        raise ValueError(
            f"head_dim {cfg.head_dim} must be a positive even integer"
        )
    half = cfg.head_dim // 2
    inv_freq = 1.0 / (
        cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float64) / cfg.head_dim)
    )                                                            # [half]
    positions = torch.arange(max_seq_len, dtype=torch.float64)   # [max_seq]
    freqs = positions[:, None] * inv_freq[None, :]               # [max_seq, half]
    cos = freqs.cos().to(torch.float16)
    sin = freqs.sin().to(torch.float16)
    expected_shape = (max_seq_len, half)
    if tuple(cos.shape) != expected_shape or tuple(sin.shape) != expected_shape:
        raise RuntimeError(
            "build_seq_rope_tables produced invalid table shapes: "
            f"cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, "
            f"expected={expected_shape}"
        )
    return _to_rpu_half(cos), _to_rpu_half(sin)


def build_causal_prefix_mask(prefix_len: int, seq: int) -> torch.Tensor:
    """``[seq, prefix_len+seq]`` fp16 additive causal-prefix mask（CPU）。

    query i（绝对 position prefix_len+i）看列 ``[0, prefix_len+i]``（全前缀 + 对新
    seq-block 因果），其余 ``-inf``。prefix_len=0 时退化为纯因果 ``[seq,seq]``。
    与 Wall-OSS ``action.py:_build_action_mask`` 同式；op 的 ``sdpa_prepare_mask``
    把它拷到稳定 device DDR 槽（无需手动 flush）。常量（prefix_len/seq 固定）→ BUILD 期烘焙。
    """
    if prefix_len < 0 or seq <= 0:
        raise ValueError(f"prefix_len {prefix_len} >= 0 and seq {seq} > 0 required")
    m = torch.full((seq, prefix_len + seq), float("-inf"), dtype=torch.float16)
    for i in range(seq):
        m[i, : prefix_len + i + 1] = 0.0
    return m


def _inject_text_prefix(owner, prefix_k, prefix_v, prefix_len):
    """Update both text runners' seven-dimensional banks without changing DMA bases."""
    cfg, cache = owner.cfg, owner.cache
    if len(prefix_k) != cfg.num_layers or len(prefix_v) != cfg.num_layers:
        raise ValueError(f"prefix layers {len(prefix_k)}/{len(prefix_v)} != {cfg.num_layers}")
    if not 0 <= prefix_len <= owner.max_seq_len:
        raise ValueError(f"prefix_len {prefix_len} outside [0,{owner.max_seq_len}]")
    expect = (prefix_len, cfg.num_kv_heads, cfg.head_dim)
    # Validate all layers before changing any live bank.
    for i, (k, v) in enumerate(zip(prefix_k, prefix_v)):
        if tuple(k.shape) != expect or tuple(v.shape) != expect:
            raise ValueError(f"layer {i} prefix {tuple(k.shape)}/{tuple(v.shape)} != {expect}")

    def identity():
        return tuple((bank.data_ptr(), tuple(bank.shape), bank.device, bank.dtype)
                     for banks in (cache.k_caches, cache.v_caches) for bank in banks)

    invalidated = False

    def invalidate():
        nonlocal invalidated
        if invalidated:
            return
        invalidated = True
        owner._prefix_epoch += 1
        owner._prefix_cache_identity = None
        try:
            # clear() preserves READY: this is not a configuration unfreeze.
            owner._graph_cache.clear()
            if not owner._graph_cache.cache_invariant_ok():
                raise RuntimeError("HALO text prefix GraphCache invariant failed")
        except BaseException:
            session = getattr(owner, "_execution_session", None)
            if session is not None:
                session.poison()
            raise

    previous = getattr(owner, "_prefix_cache_identity", None)
    if previous is not None and previous != identity():
        invalidate()  # Also catches same-shape external replacement / set_().
    try:
        for i, (k, v) in enumerate(zip(prefix_k, prefix_v)):
            swz_k = _swizzle_prefix_7d(k, owner.max_seq_len, value_layout=False)
            swz_v = _swizzle_prefix_7d(v, owner.max_seq_len, value_layout=True)
            kc, vc = cache.k_caches[i], cache.v_caches[i]
            if all(bank.device.type == "rpu" and bank.dtype == swz.dtype
                   and tuple(bank.shape) == tuple(swz.shape) and bank.is_contiguous()
                   for bank, swz in ((kc, swz_k), (vc, swz_v))) and kc.device == vc.device:
                kc.copy_(swz_k)
                vc.copy_(swz_v)
            else:
                # Prepare both replacements before publishing either pointer.
                new_k, new_v = swz_k.to("rpu"), swz_v.to("rpu")
                invalidate()
                cache.k_caches[i], cache.v_caches[i] = new_k, new_v
        cache.reset_to_position(prefix_len)
        owner._prefix_cache_identity = identity()
    except BaseException as error:
        # A failed copy may have written a partial prefix. No old graph may
        # replay it; a successful next injection rewrites every bank in full.
        try:
            invalidate()
        except BaseException as cleanup:
            error.add_note(f"HALO text prefix Graph retirement failed: {cleanup!r}")
        raise


@dataclasses.dataclass
class HaloTextCacheUpdate:
    """HALO und 文本-prompt KV-prefill runner（经 :func:`build_halo_text_cache` 构造）。

    复用通用 ``causal_decoder_*`` op + HALO und/base 权重。``_cos``/``_sin`` 是顺序
    RoPE 表（与 handle 等寿命，set_weights 已绑同一份）。所有权：device 权重/表/cache 在
    runner 存活期有效；``_handle`` 由 finalize 释放。
    """

    cfg: HaloConfig
    text: HaloBranchWeights
    host: HaloHostWeights
    cache: RPUCache
    max_seq_len: int
    _handle: int
    _graph_cache: "rpu_backend.graph.GraphCache"
    _cos: torch.Tensor
    _sin: torch.Tensor
    _handle_finalizer: object | None = dataclasses.field(
        init=False, default=None, repr=False
    )
    _closed: bool = dataclasses.field(init=False, default=False, repr=False)
    # 前缀代数（编进 GraphSignature）：优化 A 后**仅在 _inject_prefix 回落 reassign（换新
    # device 张量 → 新 DMA 地址）时 bump** 强制重 BUILD 防陈旧指针；in-place copy_（稳址）不 bump
    # → 同 regime 命中 replay；action runner 的 first-insert guard 独立保留。
    _prefix_epoch: int = dataclasses.field(init=False, default=0)
    _prefix_cache_identity: tuple | None = dataclasses.field(init=False, default=None, repr=False)

    def close(self) -> None:
        """Clear graph ownership, then retire the decoder handle once."""
        _close_halo_runner(self, graphs=(self._graph_cache,),
                           handles={"_handle": (self._handle, _destroy_causal_decoder)})

    def _embed(self, packed_text_ids: torch.Tensor) -> torch.Tensor:
        """文本 token → ``[1, seq, H]`` fp16 device。

        跟 HALO runtime 路径（``runtime._assemble_x_emb``）：``F.embedding`` →
        ``.float()`` → ``.to(bf16).to(fp16)``，匹配 golden 的 bf16-autocast embed。
        """
        ids = packed_text_ids.to(torch.long).cpu()
        emb = torch.nn.functional.embedding(ids, self.host.embed_tokens).float()  # [seq,H]
        emb = emb.to(torch.bfloat16).to(torch.float16)
        return emb.unsqueeze(0).to(device="rpu").contiguous()                     # [1,seq,H]

    def _inject_prefix(
        self, prefix_k: List[torch.Tensor], prefix_v: List[torch.Tensor], prefix_len: int
    ) -> None:
        """Model execution and tensor layout contracts."""
        _inject_text_prefix(self, prefix_k, prefix_v, prefix_len)

    def _rope_base(self, packed_text_position_ids: torch.Tensor, seq: int) -> int:
        """Model execution and tensor layout contracts."""
        pos = packed_text_position_ids.to(torch.long).cpu()
        if tuple(pos.shape) != (seq,):
            raise ValueError(f"packed_text_position_ids shape {tuple(pos.shape)} != ({seq},)")
        base = int(pos[0].item())
        expected = torch.arange(base, base + seq, dtype=torch.long)
        if not torch.equal(pos, expected):
            raise ValueError(
                f"packed_text_position_ids {pos.tolist()} 非连续递增；解耦方案只支持连续"
                f" RoPE 段（须 per-row gather 才能处理非连续）")
        if base + seq > self.max_seq_len:
            raise ValueError(
                f"rope_base+seq {base + seq} > max_seq_len {self.max_seq_len}（RoPE 表行数不足）")
        return base

    def _execution_plan(self, length: int, position: int):
        """Read the installed text domain without embedding, KV writes or Graph capture."""
        if (type(length) is not int or length <= 0
                or type(position) is not int or position < 0
                or position + length > self.max_seq_len):
            raise ValueError("HALO text planning requires positive length and an in-capacity position")
        return _plan_halo_fmb_execution(
            self,
            component=HALO_TEXT_CACHE_COMPONENT,
            stage="prefill",
            logical_len=length,
            position=position,
            handle=self._handle,
            native_prefix="causal_decoder",
            resolve_stage_domain=lambda execution_len: (
                torch.ops.rpu.causal_decoder_resolve_prefill_stage_domain(
                    self._handle, int(execution_len), position,
                    False, position + length, 0, 4, length,
                )
            ),
            kv_route="DDR_REQUIRED",
            graph_cache=self._graph_cache,
            plan_signature=(False, int(position + length), 0, 4),
        )

    @torch.no_grad()
    @execution_serialized
    def prefill(
        self,
        packed_text_ids: torch.Tensor,
        packed_text_position_ids: torch.Tensor,
        prefix_kv: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Model execution and tensor layout contracts."""
        cfg = self.cfg
        seq = int(packed_text_ids.shape[0])
        if prefix_kv is not None:
            prefix_k, prefix_v = prefix_kv
            prefix_len = int(prefix_k[0].shape[0])
        else:
            prefix_len = 0
        a6_plan, planned_stage_descriptor = self._execution_plan(seq, prefix_len)
        rope_base = self._rope_base(packed_text_position_ids, seq)
        if prefix_kv is not None:
            self._inject_prefix(prefix_k, prefix_v, prefix_len)
        else:
            self.cache.reset()                       # 清空（img_uncond 空前缀）
        ie = self._embed(packed_text_ids)            # [1,seq,H] fp16 device
        mask = build_causal_prefix_mask(prefix_len, seq)   # [seq, P+seq] fp16 CPU
        planned_chunk_size = a6_plan.selected.stage_tuple.compute_chunk
        # sig 含 prefix_len（KV 插入偏移）+ seq + rope_base（RoPE 起点，BUILD 期烘焙进
        # cos_sin_start kernel 常量，必须进 key 防陈旧 replay）+ prefix_epoch（DMA 指针
        # 代数）→ 两 regime 独立 BUILD、新前缀强制重 BUILD。
        sig = rpu_backend.graph.GraphSignature(
            op_id="halo_text_cache_prefill",
            shapes=[seq, cfg.hidden_size],
            dyn_dims=[cfg.num_layers, prefix_len, seq, rope_base,
                      self._prefix_epoch, planned_chunk_size,
                      *a6_plan.graph_key_words()],
            dtypes=[torch.float16],
        )
        with self._graph_cache.capture(sig):
            torch.ops.rpu.causal_decoder_forward(
                self._handle, ie, self.cache.k_caches, self.cache.v_caches,
                mask,                # 显式 2D additive causal-prefix mask（CPU fp16）
                prefix_len,          # position：新 K/V 插入于 [P, P+seq)，mask/seq_k 用此
                False,               # is_causal=False → MASK_2D 路径（mask 编码因果）
                None,                # position_ids（M-RoPE 专用，1D 路径忽略）
                [],                  # deepstack_dense_visual_embeds（无）
                cos_sin_offset=rope_base,  # 1D RoPE 表起点（与插入偏移解耦）。
                planned_chunk_size=0,
                planned_stage_descriptor=planned_stage_descriptor,
                # op 第 9/10 位是 GR00T 的 rope_cos_il/rope_sin_il,cos_sin_offset 移到第 11 位 → 必须关键字传。
            )
        self.cache.update_position(seq)
        # 优化 C（2026-06-24）：只去-swizzle 尾段 [P:P+seq]，免全量 P+seq 去-swizzle + 全量 H2D。
        result = self._read_tail_kv(prefix_len, seq)
        _record_halo_plan(
            self, HALO_TEXT_CACHE_COMPONENT, "prefill", a6_plan,
            authority="NATIVE_FMB_DESCRIPTOR", kv_route="DDR_REQUIRED",
        )
        return result

    def _read_tail_kv(
        self, prefix_len: int, seq: int
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """读出尾段 ``[prefix_len, prefix_len+seq)`` 的逐层 KV（优化 C：局部去-swizzle）。

        镜像 ``RPUCache.to_dynamic_cache`` 的去-swizzle 数学，但只切覆盖尾 ``seq`` 行的 seq-block
        （7-D 的 dim 1，块宽 ``sKeyChunk=16``）→ 搬运 + 计算 ∝ ``nb/sKeyVx``（部署 2/296），
        免对全 ``P+seq`` 行做去-swizzle + H2D。返回 ``{layer: (k,v)}`` 各 ``[seq,nkv,hd]`` fp16 CPU
        （与原 ``to_dynamic_cache``-切片路逐元素一致）。
        """
        cache = self.cache
        cfg = self.cfg
        chunk = cache.sKeyChunk
        blk0 = prefix_len // chunk
        blk1 = (prefix_len + seq - 1) // chunk + 1
        lo = prefix_len - blk0 * chunk                      # 尾段在去-swizzle 块内的起始行
        nb = blk1 - blk0
        eff = cache.nKVHeadChunk * cache.nKVHeadVx
        out: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for layer in range(cfg.num_layers):
            # 切 seq-block 维后才 .to("cpu") → 只搬 nb 块（非全 sKeyVx）。
            k7 = cache.k_caches[layer][:, blk0:blk1].to("cpu")
            v7 = cache.v_caches[layer][:, blk0:blk1].to("cpu")
            b = k7.shape[0]
            k = k7.permute(0, 4, 2, 1, 5, 3, 6).contiguous().reshape(
                b, eff, nb * chunk, cfg.head_dim)
            v = v7.permute(0, 4, 2, 1, 6, 3, 5).contiguous().reshape(
                b, eff, nb * cache.sValChunk, cfg.head_dim)
            k_layer = k[0, : cfg.num_kv_heads, lo:lo + seq, :]                     # [nkv,seq,hd]
            v_layer = v[0, : cfg.num_kv_heads, lo:lo + seq, :]
            out[layer] = (k_layer.permute(1, 0, 2).contiguous(),                   # [seq,nkv,hd]
                          v_layer.permute(1, 0, 2).contiguous())
        return out


def build_halo_text_cache(
    ckpt_path: "str | Path",
    *,
    cfg: HaloConfig = HaloConfig(),
    max_seq_len: int = 4736,
    rpu_execution=None,
    execution_controller: HaloExecutionController | None = None,
) -> HaloTextCacheUpdate:
    """装载 HALO und/base 权重 → 通用 causal_decoder handle → set_weights（含 qk-norm + qkv bias）。

    Args:
        ckpt_path: HALO ema.safetensors（bf16）。
        max_seq_len: KV cache 容量 + 顺序 RoPE 表行数，须 16 倍数且 ≥ 部署最长 prefill
            （cond 4698+8=4706 → 默认 4736）。
    """
    validate_halo_core_layout(cfg)
    execution_controller = _coerce_halo_execution_controller(
        rpu_execution,
        execution_controller,
        entry_point="build_halo_text_cache",
        components=(HALO_TEXT_CACHE_COMPONENT,),
    )
    if max_seq_len % _S_CHUNK != 0:
        raise ValueError(f"max_seq_len {max_seq_len} must be a multiple of {_S_CHUNK}")
    text, _act, host = load_halo_branches(Path(ckpt_path), cfg)
    cos, sin = build_seq_rope_tables(cfg, max_seq_len)
    handle = int(torch.ops.rpu.causal_decoder_create())
    resources = _HaloResources(execution_controller, handles={"_handle": (handle, _destroy_causal_decoder)})
    try:
        # 通用 set_weights arg 序（rpu_qwen3_model.cpp:54-89）：HALO und 同时填
        # q_norm/k_norm（Qwen3 式 qk-norm）与 q/k/v_bias（Qwen2 式 qkv bias），
        # mrope_section=[]（1D RoPE）、deepstack_lang_layers=[]。
        torch.ops.rpu.causal_decoder_set_weights(
            handle,
            text.q_w, text.k_w, text.v_w, text.o_w,
            text.q_norm, text.k_norm,
            text.input_norm, text.post_norm,
            text.gate_w, text.up_w, text.down_w,
            cos, sin, text.final_norm_w,
            cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim,
            cfg.hidden_size, cfg.intermediate_size,
            cfg.rms_norm_eps, True,       # use_silu (SwiGLU)
            [], [],                       # mrope_section, deepstack_lang_layers
            text.q_bias, text.k_bias, text.v_bias)
        # Use the same max_seq_len bound as the KV cache and RoPE table.
        # Register it before SPM planning to prevent oversized chunk selection.
        # A zero chunk size preserves automatic planning.
        torch.ops.rpu.causal_decoder_set_chunk_envelope(handle, int(max_seq_len), 0)
        cache = RPUCache(
            num_layers=cfg.num_layers, batch_size=1, max_seq_len=max_seq_len,
            num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim, attn_tp=cfg.attn_tp)
        graph = rpu_backend.graph.GraphCache()
        resources.graphs.append(graph)
        runner = HaloTextCacheUpdate(
            cfg=cfg, text=text, host=host, cache=cache, max_seq_len=max_seq_len,
            _handle=handle, _graph_cache=graph, _cos=cos, _sin=sin)
        resources.bind(runner)
        execution_controller.attach(
            runner, {HALO_TEXT_CACHE_COMPONENT: (runner._graph_cache,)}
        )
    except BaseException as build_error:
        # 构造任一步失败 → 孤儿 handle 立即回收（同 build_halo_action_expert）。
        _cleanup_halo_build(resources, build_error)
        raise
    return runner
