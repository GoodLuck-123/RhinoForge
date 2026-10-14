#!/usr/bin/env python3
'Qwen3 batch decode device regression.'

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F

import rpu_backend
from rpu_backend import RPUCache, RPUModelForCausalLM
from rpu_backend.model_registry import model_path
from rpu_backend.runtime.decoder import _causal_lm_runtime_complete

from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BATCHES = (2, 3, 8, 16)
PROMPT_TOKENS = 16
COS_GATE = 0.99995

PROMPTS = [
    "Astronomy note: Mercury circles the Sun quickly, while distant planets "
    "follow longer paths measured carefully by observers over many years.",
    "Biology note: plant cells convert light into stored chemical energy and "
    "release oxygen through a sequence of coordinated reactions.",
    "Mathematics note: a proof begins with precise assumptions and proceeds "
    "through justified steps until the proposed conclusion follows.",
    "History note: merchants connected coastal cities by sea and exchanged "
    "tools, foods, languages, and navigation knowledge across regions.",
    "Computing note: a compiler analyzes source text, constructs an internal "
    "representation, optimizes it, and emits instructions for a processor.",
    "Geology note: moving tectonic plates reshape continents, build mountain "
    "ranges, and produce earthquakes along active boundaries.",
    "Chemistry note: atoms form molecules by sharing or transferring electrons, "
    "and the resulting bonds determine many material properties.",
    "Literature note: a narrator can reveal events selectively, allowing tone, "
    "memory, and perspective to shape how readers interpret a story.",
    "Physics note: conservation laws constrain motion and energy exchange, even "
    "when a complicated system contains many interacting objects.",
    "Ecology note: forests contain linked communities whose plants, animals, "
    "fungi, water, and soil continuously exchange matter and energy.",
    "Economics note: prices coordinate decisions among buyers and sellers, but "
    "institutions and incomplete information influence every market.",
    "Music note: rhythm organizes sound in time, while melody and harmony create "
    "patterns that listeners recognize across a performance.",
    "Medicine note: clinicians combine symptoms, examination findings, tests, "
    "and patient history before choosing an appropriate treatment.",
    "Engineering note: a bridge design balances load paths, material strength, "
    "weather exposure, maintenance access, and construction cost.",
    "Linguistics note: languages change as speakers reuse sounds and words in "
    "new contexts while communities interact across generations.",
    "Oceanography note: currents transport heat, nutrients, salt, and organisms "
    "between basins and strongly affect regional climate.",
]


@dataclass(frozen=True)
class GraphState:
    size: int
    replay: int
    recapture: int
    invariant_ok: bool


def setup_rpu() -> None:
    torch.rpu.set_profile(False)
    torch.rpu.set_caching_allocator(True)
    torch.rpu.set_debug_export(False)
    torch.rpu.set_spm_debug(False)
    torch.rpu.set_cross_layer_batch_prefill(True)
    torch.rpu.set_cross_layer_batch_size(36)


def load_models():
    'Load independent CPU-fp32 reference and device-fp16 candidate models.'
    path = str(model_path("qwen3-0.6b"))
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    cpu_model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    rpu_model = RPUModelForCausalLM.from_pretrained(
        path,
        dtype=torch.float16,
        device="rpu",
        local_files_only=True,
    ).eval()
    if not _causal_lm_runtime_complete(rpu_model):
        raise AssertionError("canonical Qwen3 adapter runtime is incomplete")
    if not bool(rpu_model.model._rpu_batch_decode_enabled):
        raise AssertionError("canonical Qwen3 adapter lacks batch capability")
    return tokenizer, cpu_model, rpu_model


def build_real_token_batch(
    tokenizer, batch: int, decode_steps: int, prompt_tokens: int = PROMPT_TOKENS
):
    'Return distinct equal-length prompt rows and per-row forced tokens.'
    if batch > len(PROMPTS):
        raise ValueError(f"need {batch} prompts, only {len(PROMPTS)} defined")
    need = max(prompt_tokens, decode_steps)
    encoded = [
        tokenizer.encode(text, add_special_tokens=False)
        for text in PROMPTS[:batch]
    ]
    too_short = [i for i, row in enumerate(encoded) if len(row) < need]
    if too_short:
        raise RuntimeError(
            f"prompt rows {too_short} have fewer than {need} real tokens"
        )
    prompt = torch.tensor(
        [row[:prompt_tokens] for row in encoded], dtype=torch.long
    )
    forced = torch.tensor(
        [row[-decode_steps:] for row in encoded], dtype=torch.long
    )
    if any(torch.equal(prompt[0], prompt[b]) for b in range(1, batch)):
        raise RuntimeError("batch prompts must contain different real tokens")
    return prompt, forced


def graph_state(cache) -> GraphState:
    entries = list(cache.snapshot())
    return GraphState(
        size=len(entries),
        replay=sum(int(entry.replay_count) for entry in entries),
        recapture=sum(int(entry.recapture_count) for entry in entries),
        invariant_ok=bool(cache.cache_invariant_ok()),
    )


@torch.inference_mode()
def cpu_reference(model, input_ids, forced):
    'CPU-fp32 logits for every prefill and teacher-forced decode position.'
    out = model(input_ids=input_ids, use_cache=True)
    logits = [out.logits.float().cpu()]
    past = out.past_key_values
    for t in range(forced.shape[1]):
        out = model(
            input_ids=forced[:, t:t + 1],
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        logits.append(out.logits.float().cpu())
    return torch.cat(logits, dim=1)


def _assert_kv_slots_distinct(cache, batch: int) -> None:
    'A wrong slot-0 alias must fail before logits can hide it.'
    first = cache.k_caches[0][0].cpu()
    aliases = [
        b for b in range(1, batch)
        if torch.equal(first, cache.k_caches[0][b].cpu())
    ]
    if aliases:
        raise AssertionError(f"KV cache rows alias slot 0: {aliases}")


@torch.inference_mode()
def rpu_candidate(
    model,
    input_ids,
    forced,
    *,
    fuse_lm_head,
    explicit_all_ones_mask=False,
):
    'Teacher-forced device logits plus GraphCache lifecycle evidence.'
    cfg = model.config
    head_dim = getattr(
        cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
    )
    batch, seq_len = map(int, input_ids.shape)
    cache = RPUCache(
        num_layers=cfg.num_hidden_layers,
        batch_size=batch,
        max_seq_len=((seq_len + 15) // 16) * 16 + forced.shape[1] + 16,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim,
        device="rpu",
        dtype=torch.float16,
    )
    graph_cache = model.model._rpu_decoder_graph_cache
    graph_cache.clear()
    prefill_mask = (
        torch.ones_like(input_ids) if explicit_all_ones_mask else None
    )

    if fuse_lm_head:
        def prefill(ids):
            return model(
                input_ids=ids,
                attention_mask=prefill_mask,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=0,
            ).logits

        def decode(ids):
            return model(
                input_ids=ids,
                attention_mask=(
                    torch.ones((batch, cache.position + 1), dtype=torch.long)
                    if explicit_all_ones_mask else None
                ),
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            ).logits
    else:
        def prefill(ids):
            hidden = model.model(
                input_ids=ids,
                attention_mask=prefill_mask,
                past_key_values=cache,
                use_cache=True,
            ).last_hidden_state
            return model.lm_head(hidden)

        def decode(ids):
            hidden = model.model(
                input_ids=ids,
                attention_mask=(
                    torch.ones((batch, cache.position + 1), dtype=torch.long)
                    if explicit_all_ones_mask else None
                ),
                past_key_values=cache,
                use_cache=True,
            ).last_hidden_state
            return model.lm_head(hidden)

    before = graph_state(graph_cache)
    logits = [prefill(input_ids.to("rpu")).float().cpu()]
    if cache.position != seq_len:
        raise AssertionError(
            f"prefill cache position {cache.position} != logical length {seq_len}"
        )
    after_prefill = graph_state(graph_cache)
    _assert_kv_slots_distinct(cache, batch)

    after_first_decode = after_second_decode = None
    for t in range(forced.shape[1]):
        ids = forced[:, t:t + 1].contiguous().to("rpu")
        logits.append(decode(ids).float().cpu())
        if t == 0:
            after_first_decode = graph_state(graph_cache)
        elif t == 1:
            after_second_decode = graph_state(graph_cache)

    after_all = graph_state(graph_cache)
    expected_position = seq_len + int(forced.shape[1])
    if cache.position != expected_position:
        raise AssertionError(
            f"final cache position {cache.position} != {expected_position}"
        )
    if forced.shape[1] < 2:
        raise AssertionError("decode_steps must include at least two decode calls")

    lifecycle_ok = (
        before.invariant_ok
        and after_prefill.invariant_ok
        and after_first_decode.invariant_ok
        and after_second_decode.invariant_ok
        and after_all.invariant_ok
        and after_prefill.size <= before.size + 1
        and after_prefill.replay - before.replay >= batch - 1
        and after_prefill.recapture == before.recapture
        and after_first_decode.size == after_prefill.size + 1
        and after_first_decode.recapture == after_prefill.recapture
        and after_second_decode.size == after_first_decode.size
        and after_second_decode.replay > after_first_decode.replay
        and after_second_decode.recapture == after_first_decode.recapture
        and after_all.size == after_second_decode.size
        and after_all.recapture == after_second_decode.recapture
    )
    if not lifecycle_ok:
        raise AssertionError(
            "GraphCache lifecycle failed: "
            f"before={before}, prefill={after_prefill}, "
            f"decode1={after_first_decode}, decode2={after_second_decode}, "
            f"final={after_all}"
        )
    return torch.cat(logits, dim=1), after_all


def validate_logits(reference, candidate, *, cos_gate=COS_GATE):
    'Return the batch-vs-B1 result, accepting only measured fp16 near-ties.'
    cosine = F.cosine_similarity(reference, candidate, dim=-1)
    flips = torch.nonzero(
        reference.argmax(dim=-1) != candidate.argmax(dim=-1),
        as_tuple=False,
    )
    near_ties = []
    wide_flips = []
    for row, position in flips.tolist():
        ref = reference[row, position]
        top2 = torch.topk(ref, 2).values
        gap = float(top2[0] - top2[1])
        noise = float((ref - candidate[row, position]).abs().max())
        item = (row, position, gap, noise, float(cosine[row, position]))
        (near_ties if gap <= noise else wide_flips).append(item)
    min_cos = float(cosine.min())
    return (
        min_cos >= cos_gate and not wide_flips,
        min_cos,
        flips,
        near_ties,
        wide_flips,
        cosine,
    )


def rpu_b1_reference(
    model,
    input_ids,
    forced,
    *,
    fuse_lm_head,
    explicit_all_ones_mask=False,
):
    'Run each distinct row with its own cache and freshly built B=1 Graph.'
    rows = []
    for row in range(input_ids.shape[0]):
        logits, _ = rpu_candidate(
            model,
            input_ids[row:row + 1],
            forced[row:row + 1],
            fuse_lm_head=fuse_lm_head,
            explicit_all_ones_mask=explicit_all_ones_mask,
        )
        rows.append(logits)
    return torch.cat(rows, dim=0)


def run_case(cpu_model, rpu_model, tokenizer, batch, decode_steps, *,
             fuse_lm_head, prompt_tokens=PROMPT_TOKENS,
             explicit_all_ones_mask=False, cos_gate=COS_GATE) -> bool:
    input_ids, forced = build_real_token_batch(
        tokenizer, batch, decode_steps, prompt_tokens=prompt_tokens
    )
    cpu_fp32 = cpu_reference(cpu_model, input_ids, forced)
    reference = rpu_b1_reference(
        rpu_model,
        input_ids,
        forced,
        fuse_lm_head=fuse_lm_head,
        explicit_all_ones_mask=explicit_all_ones_mask,
    )
    candidate, final_graph = rpu_candidate(
        rpu_model,
        input_ids,
        forced,
        fuse_lm_head=fuse_lm_head,
        explicit_all_ones_mask=explicit_all_ones_mask,
    )

    ok, min_cos, flips, near_ties, wide_flips, _ = validate_logits(
        reference, candidate, cos_gate=cos_gate
    )
    ref_argmax = reference.argmax(dim=-1)
    rpu_argmax = candidate.argmax(dim=-1)
    cpu_cosine = F.cosine_similarity(cpu_fp32, reference, dim=-1)
    cpu_flips = int(
        (cpu_fp32.argmax(dim=-1) != reference.argmax(dim=-1)).sum()
    )
    print(
        f"  B={batch:<2} S={input_ids.shape[1]:<2} steps={decode_steps:<2} "
        f"-> {'PASS' if ok else 'FAIL'}  min_cos={min_cos:.6f}  "
        f"greedy_exact={str(not len(flips)).lower()}  "
        f"near_tie_flips={len(near_ties)}  wide_flips={len(wide_flips)}  "
        f"graph_entries={final_graph.size}  "
        f"graph_replays={final_graph.replay}  "
        f"mask={'ones' if explicit_all_ones_mask else 'none'}"
    )
    print(
        "    cpu_fp32_vs_rpu_b1 (diagnostic, non-gating): "
        f"min_cos={float(cpu_cosine.min()):.6f} argmax_flips={cpu_flips}"
    )
    for row, position, gap, noise, cos in near_ties[:8]:
        print(
            f"    accepted near-tie row={row} position={position}: "
            f"b1={int(ref_argmax[row, position])} "
            f"batch={int(rpu_argmax[row, position])} "
            f"gap={gap:.6f} noise={noise:.6f} cos={cos:.6f}"
        )
    for row, position, gap, noise, cos in wide_flips[:8]:
        print(
            f"    wide argmax mismatch row={row} position={position}: "
            f"b1={int(ref_argmax[row, position])} "
            f"batch={int(rpu_argmax[row, position])} "
            f"gap={gap:.6f} noise={noise:.6f} cos={cos:.6f}"
        )
    if min_cos < cos_gate:
        print(f"    batch-vs-B1 logit cosine below required gate {cos_gate}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batches",
        default=",".join(str(batch) for batch in DEFAULT_BATCHES),
        help="comma-separated batch sizes",
    )
    parser.add_argument(
        "--decode-steps",
        type=int,
        default=8,
        help="number of teacher-forced decode positions after full prefill",
    )
    parser.add_argument(
        "--fuse-lm-head",
        action="store_true",
        help="exercise the fused decode lm_head path",
    )
    parser.add_argument(
        "--cos-gate",
        type=float,
        default=COS_GATE,
        help="batch-vs-B1 logit cosine gate (default: formal 0.99995)",
    )
    args = parser.parse_args()
    batches = [int(value) for value in args.batches.split(",") if value.strip()]
    if args.decode_steps < 2:
        parser.error("--decode-steps must be at least 2 to prove stable REPLAY")
    if any(batch < 2 or batch > len(PROMPTS) for batch in batches):
        parser.error(f"batch sizes must be in [2, {len(PROMPTS)}]")
    if not 0.0 < args.cos_gate <= 1.0:
        parser.error("--cos-gate must be in (0, 1]")

    setup_rpu()
    tokenizer, cpu_model, rpu_model = load_models()
    if args.fuse_lm_head:
        from rpu_backend.adapters.qwen3 import _apply_fused_lm_head_for_rpu
        _apply_fused_lm_head_for_rpu(rpu_model)

    print("=" * 76)
    print("Qwen3-0.6B device B=1 vs batched decode")
    print(
        f"  batches={batches}  fuse_lm_head={args.fuse_lm_head}  "
        f"cos_gate={args.cos_gate}  "
        f"support_level={'formal' if args.cos_gate >= COS_GATE else 'controlled'}  "
        "argmax=exact-or-measured-near-tie  "
        "cpu_fp32=diagnostic-only"
    )
    print("=" * 76)

    results = [
        (f"B={batch},S={PROMPT_TOKENS}", run_case(
            cpu_model,
            rpu_model,
            tokenizer,
            batch,
            args.decode_steps,
            fuse_lm_head=args.fuse_lm_head,
            cos_gate=args.cos_gate,
        ))
        for batch in batches
    ]
    results.append((
        "B=2,S=17,mask=ones",
        run_case(
            cpu_model,
            rpu_model,
            tokenizer,
            2,
            args.decode_steps,
            fuse_lm_head=args.fuse_lm_head,
            prompt_tokens=17,
            explicit_all_ones_mask=True,
            cos_gate=args.cos_gate,
        ),
    ))
    failed = [label for label, passed in results if not passed]
    print("=" * 76)
    if failed:
        print(f"FAILED cases: {failed}")
        return 1
    print(
        f"All {len(results)} batch cases passed device B=1-vs-batch cosine >= "
        f"{args.cos_gate}, exact-or-measured-near-tie argmax, KV isolation, "
        "and GraphCache lifecycle."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
