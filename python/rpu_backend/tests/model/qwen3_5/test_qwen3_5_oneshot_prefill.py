'Qwen3 5 oneshot prefill device regression.'
import os
import sys

import torch

import rpu_backend as _rb  # noqa: F401
from rpu_backend import model_registry
from rpu_backend.adapters.qwen3_5 import Qwen3_5Adapter
from rpu_backend.api import RPUModelForConditionalGeneration
from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache

MODEL_DIR = os.environ.get(
    "QWEN35_MODEL_DIR", str(model_registry.model_path("qwen3_5-4b")))
MAX_SEQ = 256
NUM_CORES = int(os.environ.get("QWEN35_NUM_CORES", "8"))

# graph_pybind.cpp: 0=PASSTHROUGH, 1=RECORDING, 2=BUILT, 3=REPLAYING.
STATE_PASSTHROUGH = 0
STATE_BUILT = 2


def main() -> int:
    if not os.path.isdir(MODEL_DIR):
        raise FileNotFoundError(
            f"Qwen3.5 checkpoint not found at {MODEL_DIR}; set RPU_MODEL_CACHE or QWEN35_MODEL_DIR")

    print(f"[load] {MODEL_DIR} cores={NUM_CORES}", flush=True)
    hf = RPUModelForConditionalGeneration.from_pretrained(
        MODEL_DIR, dtype=torch.float16, local_files_only=True,
        rpu_execution={"model": {"num_cores": NUM_CORES},
                       "prefill": {"chunk_size": 64}})
    adapter = Qwen3_5Adapter(hf)
    model = adapter.to_rpu(max_seq_len=MAX_SEQ)

    cache_obj = adapter._graph_cache
    fusion = getattr(model, "model", model)
    inner = getattr(fusion, "language_model", fusion)
    state = inner._rpu_qwen3_5
    print(f"[cache] max_entries={cache_obj.max_entries()} size={cache_obj.size()}", flush=True)

    size_before = cache_obj.size()
    prefill_cache = getattr(state, "prefill_graph_cache", None)
    kv = Qwen3_5Cache.from_model(model, max_seq_len=MAX_SEQ)
    ids_a = torch.arange(65).view(1, 65).to("rpu")
    ids_b = (torch.arange(65) + 17).view(1, 65).to("rpu")

    def repeated_prefill(ids, cache, *, debug=False):
        outputs = []
        active_cache_name = "prefill_debug_graph_cache" if debug else "prefill_graph_cache"
        for repeat in range(2):
            cache.reset()
            active_cache = getattr(state, active_cache_name, None)
            before = list(active_cache.snapshot()) if active_cache is not None else []
            with torch.no_grad():
                output = model(input_ids=ids, past_key_values=cache, logits_to_keep=1)
            outputs.append(output.logits.detach().cpu().clone())
            assert cache.position == ids.shape[1]
            if prefill_cache is not None:
                active_cache = getattr(state, active_cache_name)
                after = list(active_cache.snapshot())
                assert active_cache.size() == len(after) == 1
                assert active_cache.cache_invariant_ok() and after[0].kernel_count > 1
                assert not after[0].recapture_count and not after[0].non_replayable_reason
                if repeat:
                    assert before[0].signature == after[0].signature
                    assert before[0].graph_lifetime_id == after[0].graph_lifetime_id
                    assert before[0].build_generation == after[0].build_generation
                    assert after[0].replay_count == before[0].replay_count + 1
            else:
                assert state.prefill_graph.state() == STATE_PASSTHROUGH
                assert state.prefill_graph.graph_size() > 1
        assert torch.isfinite(outputs[0]).all() and torch.equal(*outputs)
        print(f"[prefill] len={ids.shape[1]} debug={debug} repeated output stable; mode="
              f"{'RETAINED_CACHE' if prefill_cache is not None else 'BOUNDED_ONESHOT'}", flush=True)
        return outputs[0], after[0] if prefill_cache is not None else None

    # Same signature must upload changed input contents, and a different KV
    # object must bind fresh backing without keeping a captured stale pointer.
    output_a, entry_a = repeated_prefill(ids_a, kv)
    output_b, entry_b = repeated_prefill(ids_b, kv)
    assert not torch.equal(output_a, output_b), "changed tokens produced stale logits"
    fresh_kv = Qwen3_5Cache.from_model(model, max_seq_len=MAX_SEQ)
    restored_a, entry_restored = repeated_prefill(ids_a, fresh_kv)
    assert torch.equal(output_a, restored_a), "new KV object changed repeated A output"
    if prefill_cache is not None:
        assert entry_a.signature == entry_b.signature == entry_restored.signature
        assert entry_a.graph_lifetime_id == entry_b.graph_lifetime_id == entry_restored.graph_lifetime_id
        assert entry_a.build_generation == entry_b.build_generation == entry_restored.build_generation
        assert entry_b.replay_count == entry_a.replay_count + 2
        assert entry_restored.replay_count == entry_b.replay_count + 2

    # Logical lengths 65 and 66 share physical padding (128), but real_len is
    # graph identity. Return to A after each switch to check state reset too.
    _, entry_tail = repeated_prefill(torch.arange(66).view(1, 66).to("rpu"), kv)
    restored_a, entry_before_debug = repeated_prefill(ids_a, kv)
    assert torch.equal(output_a, restored_a)
    if prefill_cache is not None:
        assert entry_tail.signature != entry_a.signature
        assert entry_tail.graph_lifetime_id != entry_a.graph_lifetime_id
        assert entry_before_debug.graph_lifetime_id != entry_tail.graph_lifetime_id

        # Normal A -> debug B -> normal A changes shared RoPE physical shape.
        # Both caches must retire stale captured pointers before reallocation.
        old_debug = bool(torch.rpu.get_debug_export())
        assert not old_debug, "run this regression with debug export initially disabled"
        try:
            torch.rpu.set_debug_export(True)
            repeated_prefill(torch.arange(192).view(1, 192).to("rpu"), kv, debug=True)
            assert prefill_cache.size() == 0, "normal cache retained old RoPE backing"
        finally:
            torch.rpu.set_debug_export(old_debug)
        restored_a, entry_after_debug = repeated_prefill(ids_a, kv)
        assert torch.equal(output_a, restored_a), "debug shape switch changed repeated A output"
        assert entry_after_debug.graph_lifetime_id != entry_before_debug.graph_lifetime_id
        assert state.prefill_debug_graph_cache.size() == 0, "debug cache retained old RoPE backing"
    assert cache_obj.size() == size_before, "prefill occupied the decode cache"

    # Decode keeps one stable signature and should occupy exactly one cache slot.
    snapshot_before_decode = list(cache_obj.snapshot())
    replay_before = sum(entry.replay_count for entry in snapshot_before_decode)
    recapture_before = sum(entry.recapture_count for entry in snapshot_before_decode)
    kv = Qwen3_5Cache.from_model(model, max_seq_len=MAX_SEQ)
    with torch.no_grad():
        model(input_ids=torch.arange(8).view(1, 8).to("rpu"),
              past_key_values=kv, logits_to_keep=1)
    size_after_decode_prefill = cache_obj.size()
    for step in range(3):
        with torch.no_grad():
            model(input_ids=torch.tensor([[step]]).to("rpu"), past_key_values=kv)
        print(f"[decode] step={step} cache_size={cache_obj.size()}", flush=True)

    grew = cache_obj.size() - size_after_decode_prefill
    assert grew == 1, f"decode should occupy exactly one cache entry, got +{grew}"
    snapshot_after_decode = list(cache_obj.snapshot())
    replay_after = sum(entry.replay_count for entry in snapshot_after_decode)
    recapture_after = sum(entry.recapture_count for entry in snapshot_after_decode)
    assert cache_obj.cache_invariant_ok(), "decode GraphCache invariant failed"
    assert replay_after - replay_before >= 2, (
        "decode graph did not stably replay after its first BUILD: "
        f"replay delta={replay_after - replay_before}")
    assert recapture_after == recapture_before, (
        "decode graph unexpectedly recaptured: "
        f"recapture {recapture_before} -> {recapture_after}")

    print(
        "PASS: bounded prefill lifecycle and repeated outputs; decode has one BUILD, stable "
        f"REPLAY (delta={replay_after - replay_before}), and no recapture"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
