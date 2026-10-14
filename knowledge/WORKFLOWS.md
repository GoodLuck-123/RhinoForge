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
and validate the selected file with `python -m examples.run --config <TOML> --check-config`.
Keep one template per model/version/component/precision/workload. Put TOML files
in their model-family directories and runner implementations in `examples/runners/`.
Update the template, documentation and configuration checks together. Use the
current paths from the index in commands and file-copy instructions.
Keep persistent model construction outside inference mode; apply the selected mode to inference only. Warm a new Graph with BUILD and REPLAY. With `warmup >= 2`, omit `warmup_decode_steps` or match `input.decode_steps`; fixed-length runs use `stop_on_eos=false`.
The [LingBot v2 runner](../examples/configs/lingbot/v2/README.md) permits `warmup=0` after its A/B/A and READY preparation; retain an independent first output and compare subsequent replays against it.
The [exact-Z2 entry](../examples/configs/lingbot/legacy/exact_z2/README.md) uses its checkpoint processor and its own input template; keep it separate from the v2 replay runner.
The [HALO example](../examples/configs/halo/README.md) resolves `halo_action_expert`
directly to `HALO/action_expert.safetensors` under the model cache. Check the
paired single-step fixture before weight loading; do not append checkpoint directories.
Count retained, temporary-capture and validation queues against the same cold arena pool. Snapshot policy is per-instance and frozen; unsupported SDK/asset combinations use ordinary replay. See [runtime configuration](../docs/runtime_config.md).
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
and retain its admission controls. Bind the recipe, input and lifetime requirements before extending a profile. For caller factories, bind precision inputs,
reject conflicting factory settings and verify the installed model before
inference. See [quantization](../docs/quantization.md).

Use public API documentation for user-visible behavior and source for actual
implementation. Treat model-support status as profile-specific. Use the entry
points and runtime boundaries linked from [Examples](../docs/model_support.md)
and follow [model validation policy](../docs/validation_policy.md). Update stale
knowledge in the same change; do not preserve conflicting summaries.

## Speech changes

Start with the [Qwen3 speech guide](../docs/qwen3_audio.md) and retain these contracts:

- Validate frozen checkpoints and prepared TTS source hashes before loading.
  Construct TTS on CPU and load safetensors strictly, preserving nonpersistent
  FP32 RoPE buffers. Apply compatibility edits only to the prepared source copy.
- ASR uses explicit SigLIP audio mode: fourteen heads padded to sixteen,
  896-wide residuals, ERF GELU, original FP16 clamp and head scale. The current
  audio Graph includes both output projections and their GELU; its profile hash
  and physical plan include that tail. Keep model-owned staging stable and refresh
  the final output's mutable DMA address per call. Eager-only wrappers cannot be
  inserted directly into capture.
- ASR clears old text Graphs before audio execution on a prompt-length change.
  Preserve CPU FP32 convolution of FP16-rounded weights and FP16 output boundaries.
  Prefill input composition stays on device; single-token embedding lookup stays on CPU.
  Enable the fused output head only for causal-LM callers and retire the decoder
  before releasing that head.
- TTS Talker, Predictor and optional codec share one owner policy and five arenas.
  Enable `causal_decoder_set_tts_codec_mode` only on the installed codec handle;
  its default-off state preserves other decoders' RoPE parameters and layout hash.
  Keep original CPU FP32 codec weights; fold LayerScale into new frontend copies
  before FP16 conversion and preserve FP32 rotary frequencies. Use the device
  frontend for 1–63 codec frames and the original CPU FP32 sliding attention for
  64–255 frames. Missing setter selects CPU; report the route and propagate errors.
- Reset logical KV position per sequence. Talker keeps EOS and repetition penalties;
  Predictor uses a fixed 15-step sampling loop and resets KV per frame. Both use
  top-k 50, temperature 0.9 and top-p 1.0. Require EOS before waveform decoding;
  each segment allows up to 256 steps. Compare deterministic outputs after
  resetting the same seed when changing sampling.
- Predictor checked decode replay retains the ordinary input/output route and
  sampling. Bind its exact five-layer FP16/32-row/1D-RoPE grammar, versioned
  immutable weights and tables, KV owners, Graph lifetime and physical layout
  before skipping host emission. Refresh position and attention fields on every
  replay, including resets to zero; use ordinary emission when the proof cannot
  be established, and fail explicitly on inconsistencies after skipping starts.
- Accept up to 1024 characters. Keep requests within 128 characters and 32 tokens
  on their single-segment path. Split longer text at punctuation with a 12-token
  target and 128-character bound; retain tiny sentence endings up to 16 tokens.
  Use sentence stops and the request-initial CPU RNG state for each segment.
  Later segments restore the caller RNG state left by the first; return owned
  concatenated audio only after every segment succeeds.
- Bind output-head reuse to the exact current hidden object and consume it once.
  Predictor borrows the final device row for its immediate head; requests for
  hidden states retain owned CPU outputs. Never retain a borrowed row across
  another replay. Cache only the latest decoded reference speaker embedding.
- Talker reuses the 28-layer/1024 prefill envelope within its 512-position cache.
  Exclude physical padding from logical KV position; replace only the old prefill
  Graph when prompt length changes. Predictor keeps one decode Graph; codec keeps
  one current-frame Graph. Retire Graphs before handles and weights.
- CPU codec matrix multiplication and adjacent-block overlap-add preserve output
  lengths; other convolutions retain their original path. Keep thread settings
  caller-controlled and check retained outputs, input changes and close/reload
  when changing component handoffs.

## Version and release changes

Update [VERSION](../VERSION) and the matching Launch dependency lock together.
Run `python scripts/check_release_version.py` and the affected package checks.
Cut `release/X.Y` before advancing `main` to the next `.dev0` version. Preserve
published tags and assets; record each source commit and payload checksum.
See the [release policy](../docs/release_policy.md).

## Tensor diagnostics and cached masks

Keep diagnostic graph modes separate from normal retained graphs and bind cold
choices before installing a native owner. Test mode drift, frozen-cache misses
and retirement. Tensor getters share storage; callers keep a replay-independent
value with `.cpu().clone()`. Cached attention masks must compare actual input
contents, including inference tensors, and own versioned CPU storage.
