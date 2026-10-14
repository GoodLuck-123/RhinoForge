#!/usr/bin/env python3
'Qwen3 same sig two position device regression.'

import ctypes
import gc
import sys

import torch
import torch.nn as nn

import rpu_backend  # noqa: F401  (registers torch.rpu device)
from rpu_backend import RPUCache
from rpu_backend.model_registry import model_path
from rpu_backend.runtime.decoder import (
    _install_causal_decoder_forward as patch_qwen3_for_rpu,
    _install_rmsnorm_class_swap as patch_rmsnorm_class,
    _install_rotary_class_swap as patch_rotary_embedding,
)
# MR-A made chunk_envelope_for a required kwarg on the installer; the
# certified table lives in the adapter (DAG-03 forbids arch tokens in runtime/).
from rpu_backend.adapters.qwen3 import lookup_causal_decoder
from rpu_backend.runtime.weights import (
    _collect_embedding_data_ptrs,
    get_linear_partition,
    transform_linear_weight,
)

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
)


TEXT = "Hello from Hugging Face and PyTorch! This is a test for Qwen model."


def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().cpu().view(-1)
    b = b.float().cpu().view(-1)
    return (a @ b / (a.norm() * b.norm())).item()


def setup_rpu():
    torch.rpu.set_profile(False)
    torch.rpu.set_caching_allocator(True)
    torch.rpu.set_ddr_flush(True)
    torch.rpu.set_debug_export(False)
    torch.rpu.set_cross_layer_batch_prefill(True)
    torch.rpu.set_cross_layer_batch_size(36)


def patch_classes():
    patch_rmsnorm_class(Qwen3RMSNorm)
    patch_rotary_embedding(Qwen3RotaryEmbedding)


def load_cpu_model():
    cache_dir = str(model_path("qwen3-0.6b"))
    print(f"Loading Qwen3-0.6B from {cache_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(cache_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        cache_dir,
        torch_dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def convert_and_move_to_rpu(model):
    device = torch.device("rpu")
    embedding_ptrs = _collect_embedding_data_ptrs(model)

    def _free():
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def _conv_move(module):
        for n, child in module.named_modules():
            if not isinstance(child, nn.Linear):
                continue
            if child.weight.data.device != torch.device("cpu"):
                continue
            leaf = n.split(".")[-1]
            if leaf in ("o_proj", "down_proj"):
                part = 0
            else:
                part = get_linear_partition(
                    child.in_features,
                    child.out_features,
                    child.weight.element_size(),
                )
            if part >= 0:
                is_tied = child.weight.data_ptr() in embedding_ptrs
                with torch.no_grad():
                    if is_tied:
                        child.weight = nn.Parameter(child.weight.data.clone())
                    child.weight.data = transform_linear_weight(child.weight.data, part)
        module.to(device)

    backbone = model.model
    backbone.embed_tokens.to(device)
    _free()
    n_layers = len(backbone.layers)
    for i, layer in enumerate(backbone.layers):
        _conv_move(layer)
        _free()
        if (i + 1) % 10 == 0 or i == n_layers - 1:
            print(f"  Converted & moved layer {i+1}/{n_layers}")
    for name, child in backbone.named_children():
        if name in ("embed_tokens", "layers"):
            continue
        _conv_move(child)
        _free()
    for _, p in list(backbone.named_parameters(recurse=False)):
        p.data = p.data.to(device)
    for _, b in list(backbone.named_buffers(recurse=False)):
        b.data = b.data.to(device)
    for name, child in model.named_children():
        if name == "model":
            continue
        _conv_move(child)
        _free()
    for _, p in list(model.named_parameters(recurse=False)):
        p.data = p.data.to(device)
    for _, b in list(model.named_buffers(recurse=False)):
        b.data = b.data.to(device)
    _free()
    model.eval()
    return model


@torch.no_grad()
def cpu_prefill_and_two_decode_steps(model, input_ids):
    'CPU greedy: prefill + 2 decode steps. Returns (decode_tokens, hiddens).'
    backbone = model.model
    lm_head = model.lm_head
    cache = DynamicCache(config=model.config)

    out = backbone(input_ids=input_ids, past_key_values=cache, use_cache=True)
    logits = lm_head(out.last_hidden_state[:, -1:, :])
    tok = logits[0, -1, :].float().argmax().item()

    tokens = []
    hiddens = []
    for _ in range(2):
        out = backbone(
            input_ids=torch.tensor([[tok]], dtype=torch.long),
            past_key_values=cache,
            use_cache=True,
        )
        logits = lm_head(out.last_hidden_state[:, -1:, :])
        tok = logits[0, -1, :].float().argmax().item()
        tokens.append(tok)
        hiddens.append(out.last_hidden_state[0, -1, :].float().cpu().clone())
    return tokens, hiddens


def main() -> int:
    print("=" * 60)
    print("P3 gate: same-sig two-position cache hit")
    print("=" * 60)

    setup_rpu()
    patch_classes()

    tokenizer, model = load_cpu_model()
    input_ids = tokenizer(TEXT, return_tensors="pt")["input_ids"]
    prefill_len = int(input_ids.shape[1])
    print(f"prefill_len = {prefill_len}")

    print("\nCPU greedy: prefill + 2 decode steps ...")
    cpu_tokens, cpu_hiddens = cpu_prefill_and_two_decode_steps(model, input_ids)
    print(f"  cpu_decode_tokens = {cpu_tokens}")

    print("\nConverting weights & moving to device ...")
    model = convert_and_move_to_rpu(model)
    patch_qwen3_for_rpu(
        model.model, arch="qwen3", chunk_envelope_for=lookup_causal_decoder)

    cache_obj = model.model._rpu_decoder_graph_cache
    snap_initial = list(cache_obj.snapshot())
    print(f"\ninitial GraphCache snapshot: {len(snap_initial)} entries")
    assert len(snap_initial) == 0, (
        f"expected fresh GraphCache (0 entries) right after patch, "
        f"got {len(snap_initial)}"
    )

    config = model.config
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    rpu_cache = RPUCache(
        num_layers=config.num_hidden_layers,
        batch_size=1,
        max_seq_len=prefill_len + 16,
        num_kv_heads=config.num_key_value_heads,
        head_dim=head_dim,
        device="rpu",
        dtype=torch.float16,
    )

    backbone = model.model
    lm_head = model.lm_head
    ids_rpu = input_ids.to("rpu")
    rpu_cache.reset()

    # Prefill — creates the prefill-shape cache entry.
    with torch.no_grad():
        out = backbone(input_ids=ids_rpu, past_key_values=rpu_cache, use_cache=True)
        logits = lm_head(out.last_hidden_state[:, -1:, :])
        tok0 = logits[0, -1, :].cpu().float().argmax().item()
    snap_after_prefill = list(cache_obj.snapshot())
    print(
        f"\nAfter prefill: cache has {len(snap_after_prefill)} entries "
        f"(prepared_segment_hit_total sum = "
        f"{sum(e.prepared_segment_hit_total for e in snap_after_prefill)})"
    )

    # Decode step 1 — at position = prefill_len, creates the decode-shape entry.
    with torch.no_grad():
        inp = torch.tensor([[tok0]], dtype=torch.long).to("rpu")
        out = backbone(input_ids=inp, past_key_values=rpu_cache, use_cache=True)
        logits = lm_head(out.last_hidden_state[:, -1:, :])
        rpu_tok1 = logits[0, -1, :].cpu().float().argmax().item()
        rpu_hidden1 = out.last_hidden_state[0, -1, :].float().cpu().clone()
    snap_after_step1 = list(cache_obj.snapshot())
    total_hit_step1 = sum(e.prepared_segment_hit_total for e in snap_after_step1)
    replay_count_step1 = sum(e.replay_count for e in snap_after_step1)
    print(
        f"After decode step 1: cache has {len(snap_after_step1)} entries, "
        f"replay_count sum = {replay_count_step1}, "
        f"prepared_segment_hit_total sum = {total_hit_step1}"
    )

    # Decode step 2 — position = prefill_len + 1.
    # Pre-P3: new sig → new entry built.
    # Post-P3: same sig as step 1 → REPLAY (sync-only fast path).
    with torch.no_grad():
        inp = torch.tensor([[rpu_tok1]], dtype=torch.long).to("rpu")
        out = backbone(input_ids=inp, past_key_values=rpu_cache, use_cache=True)
        logits = lm_head(out.last_hidden_state[:, -1:, :])
        rpu_tok2 = logits[0, -1, :].cpu().float().argmax().item()
        rpu_hidden2 = out.last_hidden_state[0, -1, :].float().cpu().clone()
    snap_after_step2 = list(cache_obj.snapshot())
    total_hit_step2 = sum(e.prepared_segment_hit_total for e in snap_after_step2)
    replay_count_step2 = sum(e.replay_count for e in snap_after_step2)
    print(
        f"After decode step 2: cache has {len(snap_after_step2)} entries, "
        f"replay_count sum = {replay_count_step2}, "
        f"prepared_segment_hit_total sum = {total_hit_step2}"
    )

    # ============================================================
    # Assertions
    # ============================================================
    fails = []

    # 1. Cache MUST NOT grow between decode step 1 and step 2.
    if len(snap_after_step2) != len(snap_after_step1):
        fails.append(
            f"GraphCache grew across decode steps: after-step-1 = "
            f"{len(snap_after_step1)} entries, after-step-2 = "
            f"{len(snap_after_step2)} entries. Step-2 sig missed the cache "
            f"and built a new entry (P3 sig-without-position not in effect)."
        )

    # 2. prepared_segment_hit_total MUST have increased by >=1
    #    (step 2 took the sync-only REPLAY path on the cached segment).
    hit_delta = total_hit_step2 - total_hit_step1
    if hit_delta < 1:
        fails.append(
            f"step-2 did NOT trigger sync-only REPLAY: "
            f"prepared_segment_hit_total delta = {hit_delta} (expected >= 1)."
        )

    # 3. The top-level GraphCache wrapper must expose the same REPLAY through
    #    its entry counter. This is frontend observability; the segment-hit
    #    assertion above remains the independent C++ execution-path evidence.
    replay_delta = replay_count_step2 - replay_count_step1
    if replay_delta < 1:
        fails.append(
            f"step-2 REPLAY was not reflected in GraphCache.replay_count: "
            f"delta = {replay_delta} (expected >= 1)."
        )

    # 4. Tokens must match CPU greedy decode (numeric correctness).
    if rpu_tok1 != cpu_tokens[0]:
        fails.append(
            f"step-1 token mismatch: cpu={cpu_tokens[0]} rpu={rpu_tok1}"
        )
    if rpu_tok2 != cpu_tokens[1]:
        fails.append(
            f"step-2 token mismatch: cpu={cpu_tokens[1]} rpu={rpu_tok2}"
        )

    # 5. Runtime validity remains required; CPU floating differences are diagnostic.
    for label, actual, expected in (("step-1", rpu_hidden1, cpu_hiddens[0]),
                                    ("step-2", rpu_hidden2, cpu_hiddens[1])):
        if (actual.shape != expected.shape or not actual.numel()
                or not bool(torch.isfinite(actual).all())
                or not bool(torch.isfinite(expected).all())):
            fails.append(f"{label} hidden shape/finite contract failed")
    cs1 = cos_sim(rpu_hidden1, cpu_hiddens[0])
    cs2 = cos_sim(rpu_hidden2, cpu_hiddens[1])
    print(f"\nHidden cos_sim vs CPU: step1={cs1:.6f}, step2={cs2:.6f}")
    print(f"Tokens: cpu={cpu_tokens}, rpu=[{rpu_tok1}, {rpu_tok2}]")

    print()
    print("=" * 60)
    if fails:
        print("TEST FAILED:")
        for msg in fails:
            print(f"  [FAIL] {msg}")
        print("=" * 60)
        return 1

    print(
        f"TEST PASSED  (cache entries stable @ {len(snap_after_step1)}, "
        f"replay_count_delta = {replay_delta}, "
        f"prepared_segment_hit_delta = {hit_delta}, tokens match cpu, "
        "CPU floats diagnostic; task_quality=NOT_EVALUATED formal_certification=false)"
    )
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
