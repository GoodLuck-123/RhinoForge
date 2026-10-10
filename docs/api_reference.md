# API reference

[简体中文](api_reference.zh.md) | English

The distribution is named `rhinoforge`; its Python package remains
`rpu_backend`. Importing the package registers PyTorch's `PrivateUse1` device as
`rpu` and creates the `torch.rpu` namespace.

Choose the checkpoint, precision and input profile from [examples](model_support.md).

## Top-level package

`rpu_backend.__all__` contains exactly four names:

| Name | Contract |
|---|---|
| `__version__` | Installed RhinoForge distribution version |
| `RPUCache` | FP16 KV cache for fused causal attention |
| `RPUModelForCausalLM` | Loader for registered decoder-only CausalLM profiles |
| `reset_graph_cache()` | Clear the process graph cache; a no-op without the native backend |

The API compatibility level is available as `rpu_backend.api.API_VERSION` and
currently reports `5.0.0`; it is independent of the distribution version.

## Causal language models

```python
import torch

from rpu_backend import RPUCache, RPUModelForCausalLM

model = RPUModelForCausalLM.from_pretrained(
    checkpoint,
    dtype=torch.float16,
    device="rpu",              # omit to load on CPU, then call model.to("rpu")
    rpu_execution={
        "prefill": {
            "chunk_size": "auto",
            "padding_rows": "auto",
            "padding_budget": 64,
        }
    },
)
cache = RPUCache.from_model(
    model,
    input_ids=input_ids,
    max_new_tokens=32,
)
```

### `RPUModelForCausalLM.from_pretrained`

```python
RPUModelForCausalLM.from_pretrained(
    hf_repo_or_path,
    *,
    dtype=torch.float16,
    device=None,
    rpu_execution=None,
    quantization=None,  # CausalLM: explicit dense Qwen3 W4/G32 installation recipe
    **hf_kwargs,
)
```

The loader reads the Hugging Face configuration before the full weight load,
selects an adapter by `config.architectures[0]`, and rejects unsupported
profiles early. It currently admits the built-in Qwen3 and Llama CausalLM
architectures. Only `torch.float16` is accepted by this entry point; a
quantized checkpoint carries its own checked metadata.

`rpu_execution` is a read-only per-handle mapping. The CausalLM entry accepts a
`prefill` stage with:

- `chunk_size`: `"auto"` or a positive multiple of 16;
- `padding_rows`: `"auto"` or a non-negative exact row count; and
- `padding_budget`: a non-negative integer, mutually exclusive with an exact
  `padding_rows` value.

For Qwen3, `prefill.linear_acc32` selects ACC16 (`false`, default) or ACC32
(`true`). The cold root `model.num_cores` budget is `4`, `6` or `8`; plain FP16
batch-one profiles admit reduced budgets, while quantized profiles retain eight
cores. Other CausalLM adapters declare their own accepted fields. The adapter
may narrow these values to the exact profile envelope. Create a new model to
change them after device installation. Qwen3-VL, Qwen3.5 and Pi0.5 expose their
component precision and topology controls through the same mapping; see the
[target-specific configuration catalog](runtime_config.md#configuration-and-execution).

The generic CausalLM and image-text loaders, and `Pi05Policy.from_pretrained`,
support only built-in model code. `trust_remote_code` must be exactly `False`;
custom Hugging Face model code is rejected before configuration or weight
loading.

### `RPUCache`

Prefer `RPUCache.from_model` over the low-level constructor:

```python
RPUCache.from_model(
    model,
    *,
    input_ids=None,
    max_new_tokens=...,
    max_seq_len=None,
    batch_size=1,
    device="rpu",
)
```

Pass exactly one sizing form:

- `input_ids` plus `max_new_tokens`; or
- an explicit positive `max_seq_len`.

The cache exposes `reset()`, `reset_to_position(pos)`, `get_seq_length()`,
`get_max_length()`, `get_cache(layer_idx)`, and `to_dynamic_cache(device="cpu")`.
Batch sizes above one are limited to the explicitly admitted Qwen3 path and
require equal real-token lengths across rows.

`rpu_backend.api.Qwen3_5Cache.from_config(text_config, max_seq_len, ...)` adds
the recurrent and convolution state used by Qwen3.5 text. It may reset fully to
position zero but cannot reconstruct a non-zero recurrent state by rewinding.

## Image-text models

`rpu_backend.api.RPUModelForConditionalGeneration` is the Hugging Face-style
loader for the built-in Qwen3-VL conditional-generation path:

```python
from rpu_backend.api import RPUModelForConditionalGeneration

model = RPUModelForConditionalGeneration.from_pretrained(
    checkpoint,
    dtype=torch.float16,
    device="rpu",
    rpu_execution={
        "prefill": {"chunk_size": "auto"},
        "vision": {"chunk_size": "auto"},
    },
)
```

It has the same CPU-first loading and fail-fast profile behavior as the
CausalLM loader. The `vision` stage accepts `chunk_size`; the `prefill` stage
accepts the fields described above. Video and profiles outside
[Examples](model_support.md) are not implied by this class.

Qwen3.5 text/vision and Gemma4 use their model-specific adapters.

### Qwen3.5 27B

Official 27B uses uniform W8 Text/VL at 4/6/8 cores or W4 group32 at 8 cores. Root6 assigns MLP to 6 cores and attention/GDN/head/Vision to 4. It requires the existing quantization opt-in and, for image inputs, the Vision opt-in. It uses caching allocation, four cold Graph arenas, finalized uploads from CPU weights, physical KV24, bounded one-shot prefill and retained decode. FP16 runtime is not provided. [Configuration and weights](../examples/configs/qwen3_5/README.md).

## Policy APIs

The following classes are exported from `rpu_backend.api`:

| Class | Primary construction and inference methods |
|---|---|
| `Pi05Policy` | `from_pretrained(...)` or `from_lerobot_policy(...)`; `to("rpu")`; `prepare_graphs(...)`; `predict_action_chunk(...)`; `select_action(...)` |
| `RhinoVLAPolicy` | `from_runtime(...)`, `from_factory(...)`, or `from_pretrained(..., runtime_factory=...)`; `prepare_graphs(...)`; `predict(...)`; `predict_action_chunk(...)` |
| `WallOssPolicy` | `from_checkpoint(...)` or `from_pretrained(...)`; `to("rpu")`; `prepare_graphs(...)`; `infer(...)`; `predict_action_chunk(...)` |
| `Lingbot2Policy` | `from_checkpoint(...)`; `to("rpu")`; `prepare_graphs(...)`; `infer(...)`; `predict_action_chunk(...)`; `close()` |
| `HyEmbodiedPolicy` | `from_checkpoint(...)`; `to("rpu")`; `infer(...)`; `predict_action_chunk(...)`; `close()` |

Public LIBERO optimized profiles use
`Pi05Policy.from_pretrained(..., optimized_profile=...)` with `precision`
(`fp16`, `w8a16`, `w8_action_nvfp4`, or `w8_prefill_a8_action_nvfp4`),
`num_cameras` (2 or 3), and `text_tokens` (32, 64, 96, or 128).
Precision must match checkpoint metadata and the required operator capabilities.
`prepare_graphs(...)` precomputes AdaRMS by default for these profiles. See
[`examples/configs/pi05/`](../examples/configs/pi05/).

`Pi05Policy.prepare_graphs(batch, noise=...)` and
`Pi05Policy.predict_action_chunk(batch, noise=...)` accept an optional initial
noise tensor. It must be finite CPU `float32` with shape
`[batch_size, chunk_size, max_action_dim]` from the loaded policy configuration.
Use the same tensor for graph preparation and inference when comparing fixed
inputs. Omitting it preserves the policy's normal seeded noise generation.
Pi0.5 example TOML files expose this as `input.noise`, a path to a tensor saved
with `torch.save`; the optimized examples require shape `[1, 50, 32]`.

Pi0.5 `prepare_graphs()` reports Graph readiness and replay diagnostics.
It checks shape, dtype, finite values and Graph invariants.
`replay_parity.numerical_status` is `DIAGNOSTIC`; `task_quality` is `NOT_EVALUATED`.

For the release-bound Pi0.5 component profiles, pass `profile="pi0.5-base"`
or `profile="pi0.5-libero-v044"` together with the first preprocessed
`admission_batch`. The loader verifies the exact public payload and request
envelope before loading weights; subsequent Graph and inference requests are
checked against the same profile.

`WallOssActionOutput`, `Lingbot2ActionOutput`, and
`HyEmbodiedActionOutput` are the corresponding structured return types. A
physical-unit action field applies the configured normalization.

`WallOssPolicy.from_checkpoint(...)` binds the checkpoint, camera roster,
normalization and precision; `.to("rpu")` installs the runtime. Supply images
in the same order as `camera_names` when calling `infer(...)`. The policy
validates actual request geometry and state against the installed model.
`prepare_graphs(...)` optionally binds an explicit finite prefix range for
representative inputs; reuse a range only for inputs admitted by that policy.

`RhinoVLAPolicy` deliberately delegates checkpoint composition and preprocessing
to an explicit model-repository runtime factory. The runtime must expose the
execution capabilities and effective `rpu_execution` configuration it
consumed; the facade rejects a silent mismatch.

A string `runtime_factory="module:callable"` imports and executes Python code
from that module. Treat it as a trusted extension point: use only an installed,
reviewed model integration and do not copy the value from an untrusted TOML or
request. Passing an already imported callable has the same trust requirement.

`HyEmbodiedPolicy.from_checkpoint` does not read `norm_stats.pkl` by default.
Physical-action decoding requires `norm_stats_path`, the explicit
`trust_norm_stats_pickle=True` opt-in, and `norm_stats_sha256` for the exact
file bytes. The hash is checked before those same bytes are deserialized.
The fixed three-camera facade accepts `prefix_len` only in
`{192, 208, 224, 240}`. Request images, noise, state embeddings/state, and end
effector pose must be finite; invalid values are rejected before policy
execution.

Use each profile's documented constructor, input schema and precision.

### LingBot v2 runtime Graph

Use `rpu_backend.adapters.lingbot_vla_v2.runtime.build_lingbot_vla_v2(..., graph_runtime_policy=...)` and `get_action` for the current FP16/W8A16 entry. The [input guide](../examples/configs/lingbot/v2/README.md) specifies two/three cameras, training YAML, language storage and normalized `[1,50,55]` outputs. The older `Lingbot2Policy.from_checkpoint` facade accepts `rpu_execution` and retains the separate exact-Z2 profile.

## Error hierarchy

All user-facing errors are available from `rpu_backend.api`:

```text
RPUBackendError
├── RPUConfigError
├── UnsupportedModelError
├── RPUUnsupportedDtypeError
├── RPUSingleHandleError
├── SPMExhaustionError
└── WeightShapeMismatchError
```

Catch `RPUBackendError` for backend-level failures, or a subclass when the
caller has a specific recovery action. Standard `TypeError` and `ValueError`
remain in use for malformed Python arguments.

## Graph API

`GraphRuntimePolicy` accepts cold `batch_snapshot_mode="auto"` or `"off"`. `GraphCache` and the Pi0.5, Wall-OSS and LingBot v2 builders accept the policy before weight installation. AUTO uses supported SDK/asset capabilities and otherwise falls back to ordinary replay; OFF preserves Graph execution. See [snapshot policy](runtime_config.md#graph-snapshot-policy).

The public graph namespace is `rpu_backend.graph`:

```python
from rpu_backend.graph import GraphCache, GraphSignature

cache = GraphCache(max_entries=4)
sig = GraphSignature(
    op_id="my_model_forward",
    shapes=(seq_len, hidden_size),
    dyn_dims=(num_layers,),
    dtypes=(torch.float16,),
)

with cache.capture(sig):
    output = torch.ops.rpu.my_model_forward(...)
```

`GraphCache` provides `capture`, `begin_warmup`, `freeze`, `is_frozen`,
`lookup`, `evict`, `clear`, `size`, `max_entries`, `snapshot`, and
`cache_invariant_ok`. A frozen cache is lookup-only and rejects an online
BUILD. Most applications should let the model adapter own its graph cache;
manual graph construction is primarily a porting interface.

`Graph.runtime_policy` and `GraphCache.runtime_policy` are read-only views of
the policy bound at construction. Extracted child graphs inherit that policy.
After a failed capture, the scope releases its borrowed graph reference so
`evict()` or `clear()` can run while the exception traceback is retained.
References explicitly retained by callers still prevent graph retirement.

User-level diagnostics remain public. On a `GraphCache` instance,
`debug_bucket_counts()`, `debug_branch_counts()`, `dump_signature_tree()`, and
`explain_miss(signature)` report cache organization, signature state, and miss
reasons. `Graph.debug_stats()` returns an aggregate mapping of execution
counters for the most recent graph submission. These methods are useful for
lifecycle diagnosis, but their output is not numerical or support evidence.
`Graph.dump_replay_plan()` returns a pointer-free linear node/segment summary;
`Graph.dump_tree()` presents the same execution structure grouped by segment.
Neither includes launch argument words, device instructions, or raw addresses.

`Graph` is the lower-level capture object and `get_default_graph_cache()`
returns a thread-local cache. New adapters should normally use a dedicated
per-model `GraphCache` so ownership and teardown are explicit.

## Adapter registry and plugins

The registry lives in `rpu_backend.runtime.registry`:

```python
from rpu_backend.runtime.registry import (
    get_adapter,
    list_adapters,
    register_adapter,
)
```

Built-in architecture bindings are declared in
`rpu_backend.adapters._manifest.BUILTIN_ADAPTERS`. The current manifest binds:

- `Qwen3ForCausalLM`;
- `LlamaForCausalLM`;
- `DINOv3ViTModel`;
- `Gemma4ForConditionalGeneration`;
- `Qwen3_5ForConditionalGeneration`; and
- `Qwen3VLForConditionalGeneration`.

An external distribution may declare an entry point in the
`rpu_backend.plugins` group. The loaded callable receives a small namespace
whose `register_adapter(architecture, adapter_class)` method adds bindings:

```toml
[project.entry-points."rpu_backend.plugins"]
my_adapter = "my_package.rpu_plugin:register"
```

```python
def register(api):
    api.register_adapter("MyModelForCausalLM", MyModelAdapter)
```

Plugins are loaded lazily after an unknown-architecture lookup, or explicitly
with `discover_plugins()`. They cannot override a built-in architecture or a
previously registered plugin binding. Loading an entry point executes code from
the installed plugin distribution; install and enable only plugins you trust.

The registry is process-wide and has no public unregister operation. Once a
built-in or plugin architecture is bound, keep that binding for the process
lifetime; start a fresh process after changing the installed plugin set.

## Model-path registry

`rpu_backend.model_registry` provides local path resolution:

```python
from rpu_backend.model_registry import MODELS, cache_root, model_path

path = model_path("qwen3-0.6b")
```

`cache_root()` uses `RPU_MODEL_CACHE` when set and otherwise resolves to
`~/.cache/rhinoforge/models`. `model_path(name)` appends the checked-in relative
path and raises `KeyError` for an unknown alias. An alias is only a path mapping;
it is not a support or asset-availability claim.

## Weight-layout API for adapter authors

The canonical helpers are in `rpu_backend.runtime.weights`:

- `get_linear_partition(in_features, out_features, dwidth=2, num_cores=8)`;
- `swizzle_linear(weight, partition, num_cores=8)`; and
- `swizzle_model_inplace(model, ..., skip_names=...)`.

Swizzling changes parameter storage in place and must run exactly once. A
specialized linear owned by a fused subsystem belongs in `skip_names` and must
be transformed by that subsystem's one authoritative conversion path.

The existing dense FP16 Qwen3-8B loader validates all original weights before
installation, reserves its large vocabulary head and embedding first, then
converts and transfers one Linear at a time. It releases CPU weight storage
between transfers to reduce simultaneous CPU/device allocation. The public loading
API is unchanged. A failed partial installation still requires a fresh model.

## `torch.rpu` device controls

The commonly useful device-level functions include:

- `is_available()`, `device_count()`, and `current_device()`;
- `set_caching_allocator(bool)`, `empty_cache()`, and `memory_stats()`;
- `set_ddr_flush(bool)` and `set_ddr_flush_force(bool)`;
- `set_debug_level(0..5)` and `get_debug_level()`;
- `set_cross_layer_batch_prefill(bool)` and
  `set_cross_layer_batch_size(int)`; and
- `shutdown()`.

Developer diagnostics are also available through `torch.rpu`:

- `set_debug`, `get_debug`, `set_profile`, `get_profile`, and
  `reset_profile_accumulators`;
- `set_spm_debug`, `get_spm_debug`, and `spm_alloc_dump`.

These diagnostics expose logs and aggregate counters, not model inputs,
intermediate activations, raw addresses, or Graph register/resource payloads.

`set_caching_allocator(bool)` is a cold, process-wide tensor-storage choice.
The first explicit call, or the first non-empty device tensor allocation when no
choice was made, freezes the policy for the process. Repeating the same choice
is idempotent; requesting the opposite choice raises an error and requires a
fresh process. Call it only before device tensor materialization. `empty_cache()`
releases unused cached blocks and cannot release live tensor or Graph-owned
storage. `memory_stats()` reports the selected mode in
`caching_allocator_enabled` and whether it is frozen in
`caching_allocator_policy_frozen`.

CPU/device boundary flushing is enabled by default and should not be disabled in
normal inference. Chunk size is a per-handle `rpu_execution` setting, not a
process-global `torch.rpu` setter. Internal `torch.ops.rpu.*` schemas are host
wrapper interfaces used by adapters; unless a function is documented here or
in a model example, treat it as an implementation detail.

## Bounded speech APIs

```python
from rpu_backend.api import Qwen3ASRModel, Qwen3TTSModel
```

| API | Input and result |
|---|---|
| `Qwen3ASRModel.from_pretrained(checkpoint, dtype=torch.float16)` | Local pinned Qwen3-ASR-0.6B checkpoint |
| `model.transcribe(audio, sample_rate=16000, max_new_tokens=64)` | One finite CPU floating-point waveform, mono 16 kHz, 0.5–8 seconds; token limit 1–64. Returns `ASRTranscription(text, raw_text, token_ids)` |
| `Qwen3TTSModel.from_pretrained(checkpoint, source=prepared_package, dtype=torch.float16)` | Local pinned Qwen3-TTS-12Hz-0.6B-Base checkpoint and prepared `qwen_tts/` source directory |
| `model.synthesize(text, reference_audio=local_file, language="Chinese", max_new_tokens=64)` | Text of 1–128 characters and at most 32 tokenizer tokens; `Chinese` or `English`; finite mono 24 kHz reference audio, 0.5–20 seconds; token limit 2–64. Returns an independent float32 mono waveform and sample rate 24000 |

ASR uses greedy generation without context prompts, timestamps or streaming.
TTS uses non-streaming greedy speaker-embedding-only voice cloning. Its Talker
prompt plus generation limit must fit 128 positions; the token cap can truncate
speech. Downloads, preparation, CPU/device precision boundaries and commands are
in the [speech guide](qwen3_audio.md).

Both APIs support `with`, idempotent `close()` and `execution_stats()`. Serialize
requests, statistics and close, with one live device model/policy per process.
Loaders validate assets before transformation and select the caching allocator
before device allocation. TTS also declares five owner-specific arenas. Restart
if the process has an incompatible frozen allocator or arena plan.

ASR requires `siglip_set_audio_mode`, `siglip_set_audio_projection` and
`assemble_inputs_embeds` in the rebuilt native extension. TTS codec uses the
explicit `causal_decoder_set_tts_codec_mode`; without it, the frontend stays on
CPU. Other execution errors propagate. Codec input is limited to 1–63 frames.

`execution_stats()` exposes call and cache state:

- ASR keeps one audio Graph for the current audio-token length and at most the
  current prompt length's prefill/decode pair. `audio_graph_size`,
  `audio_graph_replays` and `audio_graph_invariant_ok` describe the audio cache;
  `graph_size`, `graph_replays` and `graph_invariant_ok` describe the text cache.
  Replay totals reset when cached signatures are replaced. `audio_linear_calls`
  counts logical projections.
- TTS reports `codec_frontend.device` and `fallback_reason`, output-head calls,
  and each component's calls, tokens and Graph state. Talker replaces its prefill
  Graph on a prompt-length change while retaining decode; Predictor retains one
  decode Graph; codec replaces its Graph on a frame-count change.

Each sequence resets logical KV position while retaining private storage. TTS
caches only the latest reference speaker embedding by decoded audio content;
every request still reads and validates the file. Output storage is independent
of replay. Retire Graphs before native handles and weights. CPU thread count is
caller-controlled; both CLIs expose `--threads`, defaulting to 4.
