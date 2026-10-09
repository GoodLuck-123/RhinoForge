# Knowledge workflows

## Query

1. Start at [INDEX.md](INDEX.md).
2. Read the smallest relevant concept or synthesis page.
3. Follow its public source links before changing code or making a support
   claim.
4. Use `rg` only after the indexed sources do not answer the question.

See [RETRIEVAL.md](RETRIEVAL.md) for the Markdown-first retrieval contract and
the boundary for optional semantic indexing.

For model assessment or implementation, use the
[`rhinoforge-port` skill](../.agents/skills/rhinoforge-port/SKILL.md) with
[model porting](../docs/model_porting.md).

## Add or update knowledge

1. Confirm the statement in public source or documentation.
2. Update the existing page that owns the fact; create a page only for a new,
   reusable concept.
3. Follow [SCHEMA.md](SCHEMA.md), add direct source links, and update
   [INDEX.md](INDEX.md).
4. Check relative links and scan the changed files for private infrastructure,
   credentials, restricted implementation detail, opaque asset internals, and
   diagnostic payloads.

## Save a reusable answer

1. First update an existing concept or synthesis page when it owns the result.
2. Add a short page under [queries/](queries/README.md) only when the original
   question and decision context remain useful.
3. Keep direct public source links and mark unresolved points; do not store raw
   chat transcripts or diagnostic artifacts.

## Add an external source

Prefer a stable public URL. When an offline snapshot materially improves
provenance or availability, follow [raw/README.md](raw/README.md), confirm its
license or redistribution basis, and cite it from a maintained page.

## Resolve conflicts

For runnable examples, consult the [configuration index](../examples/configs/README.md)
and validate the selected file with `examples/run_model.py --check-config`.
Keep one canonical template per model/version/component/precision/workload;
remove superseded copies and update script defaults, documentation and tests
when reorganizing paths. Model TOML files belong in their family directories.
The example catalog binds configuration identity and input settings; it does
not expand the model's runtime admission or certify performance or quality.
For performance comparisons, use the actual example command and bind the
checkpoint, precision, input, execution settings, and runtime assets. For
all example targets, keep persistent construction outside inference mode and
apply the selected mode only to inference; see
[host execution settings](../docs/runtime_config.md#host-execution-settings).
Count retained Graph queues together with temporary capture and first-call
validation queues when checking a fixed Graph arena pool. A successful replay
alone does not cover the peak during cold construction.
For steady-state timing, finish numerical checks and release their temporary
exports before warming the real REPLAY path. The shared runner defaults to two
warmup calls, reports them separately, and releases raw returned outputs after
snapshotting them outside the timer. With `warmup >= 2`, omit
`warmup_decode_steps` or match `input.decode_steps`; config validation rejects
shorter warmup before model loading. Fixed-length tests use `stop_on_eos=false`.
Two short decode calls do not warm later positions or signatures. See [performance measurement](../docs/performance.md).
For cache-maintenance changes, trace the actual CPU reads and writes through
their producing copy operations. Device-only handoffs retain storage and DMA
ownership without republishing data; host-backed device copies still need both
boundaries. Check in-place request updates and mask changes before reusing a
cached upload. See [component handoffs](concepts/component-handoff.md).
For Pi0.5 fixed-input comparisons, use the public `input.noise` field for both
Graph preparation and inference; retain default-input failures separately.
See the [policy API](../docs/api_reference.md#policy-apis) and
[model family map](synthesis/model-family-map.md#pi05).
When adding quantized examples, trace the checkpoint loader and installed
projection storage, not just the `dtype` label. Document the component scope
and retain its admission controls. A controlled planning bound is not a
certification: bind the exact recipe, input, numerical and lifetime evidence
before widening its admitted scope. For caller factories, bind precision inputs,
reject conflicting factory settings and verify the installed model before
inference. See [quantization](../docs/quantization.md).

Use public API documentation for user-visible behavior and source for actual
implementation. Treat model-support status as profile-specific. Use the entry
points and runtime boundaries linked from [Examples](../docs/model_support.md)
and follow [model validation policy](../docs/validation_policy.md). Update stale
knowledge in the same change; do not preserve conflicting summaries.


For the bounded Qwen3 speech APIs, start with the
[speech profile](../docs/qwen3_audio_r1.zh.md). Keep the frozen checkpoint guards
and TTS reviewed-source hashes before loading; do not install the upstream
package over the pinned Transformers runtime. Normal CPU construction plus
strict safetensors loading preserves TTS weights and nonpersistent RoPE buffers.
Speech is a dedicated API, not a generic CausalLM/VLM registry alias. ASR prompt
length changes clear the old Graph pair before eager audio Linear execution,
leaving launch arena capacity; same-length requests replay the pair. TTS owns
two decoder handles under one process owner. Retire both Graphs before handles
and weights. Test retained outputs and close/reload across component handoffs.
Speech loaders select the common caching allocator before device allocation.
Reuse private KV storage with `reset_to_position(0)` at each new sequence;
the causal path overwrites live rows before reading them. TTS speaker reuse is
bounded to the latest decoded audio content, never just a file path or mtime.
Its x-vector-only profile does not consume reference codec tokens. ASR CPU
convolutions accumulate FP16-rounded weights in FP32 and round outputs back to
FP16; preserve this boundary in reference comparisons. See the
[speech optimization evidence](../docs/qwen3_audio_optimization_r1.zh.md).
