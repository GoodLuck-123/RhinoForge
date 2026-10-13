# Attention layout boundaries

RhinoForge's public unified attention boundary uses
`[batch, sequence, heads, head_dim]`. The common PyTorch attention convention
uses `[batch, heads, sequence, head_dim]`. Both describe the same logical
tensor, but silently treating one as the other changes which values belong to
each token and head.

## Boundary rules

- Reshape a contiguous projection from `[batch, sequence, hidden]` to
  `[batch, sequence, heads, head_dim]` when `hidden == heads * head_dim`.
- Crossing from the unified boundary to the common PyTorch convention requires
  swapping the sequence and head dimensions. Materialize contiguous storage
  when the next consumer requires it; do not rely on a transposed stride being
  accepted implicitly.
- Keep query, key, value, and returned attention output in one declared layout
  throughout a wrapper. Convert once at the boundary instead of adding
  compensating transposes at individual call sites. [Public attention schema](../../src/core/rpu_dispatch_registrations.inc)
- Treat `RPUCache` as the owner of cached K/V storage. Callers size it and
  update its logical position through the public cache API; they do not
  reinterpret its storage as a regular PyTorch attention tensor. [RPUCache](../../docs/api_reference.md#rpucache)
- Mask shape, causal mode, cache position, and head geometry are semantic
  inputs. They belong in profile validation and, where applicable, the Graph
  signature. [Model porting](../../docs/model_porting.md#7-graph-and-cache-invariants)
- Batch support follows the envelope named for the exact release profile. [Examples](../../docs/model_support.md)

If attention output is numerically wrong while projection norms look
reasonable, verify the layout at every public boundary before changing model
math or adding a model-specific workaround.

## Mask cache and replay

Use `rpu_backend.runtime.masks.versioned_cpu_mask(mask)` when creating an
adapter-owned mask cache entry. It makes an independent, contiguous CPU FP16
copy with mutation tracking, including inside `torch.inference_mode()`. Reuse
that tensor on cache hits; copying it on every forward defeats identity/version
reuse. Keep it on CPU until native preparation rather than uploading it first.
Caller-owned inference tensors may lack a version counter and require a
conservative content refresh.

Cache keys include query/key geometry, real and padded lengths, visibility
rules and window position. The native preparation step checks tensor identity
and mutation version, then publishes changed contents into a stable device slot
before Graph replay. In-place mask edits must therefore reach preparation even
when the Graph signature and storage addresses are unchanged.

Device-slot reuse and temporary SPM reuse are separate lifetimes. A mask may
remain in SPM across layers only when its allocation covers every intervening
writer. A recorded upload runs on each Graph replay; no request assumes a
previous phase left the mask intact. If a profile's shared buffer plan cannot
reserve the longer lifetime, admission retains its per-layer copy schedule.
Once a physical descriptor selects a schedule, execution must follow it.

The no-growth residency helper evaluates the complete buffer declarations with
the actual allocator and accepts a longer lifetime only if temporary peak does
not increase. Selected aliases, writers through another alias, per-layer
scratch and outside-layer scratch need separate lifetime handling. Owners
that permit additional SPM instead perform cold admission against the complete
held, persistent and temporary budget, checking actual rows and padded capacity.
Their selected `PER_LAYER` or `FORWARD` schedule becomes part of the physical
descriptor and allocation identity. Independent pipeline leases require their
own lifetime proof.

Model-specific boundaries follow these rules:

- Pi0.5 and Wall action paths can share a prepared mask across layers or denoise
  steps only within their declared allocation lifetime.
- HALO reuses the base decoder's prepared mask instead of preparing it twice.
- Gemma4 prepares each sliding-window chunk separately. Equal chunk sizes do not
  imply equal contents, and decode position changes require refresh. Ordinary
  Graph replay remains available; component fast-skip stays disabled where
  RoPE/KV updates still require the normal host walk.
- Qwen3-VL's ordinary all-valid causal path validates caller visibility for
  position handling and lets the native decoder construct its causal mask.
- GR00T text/image masks use separate slots. RhinoVLA refreshes changed live
  masks before any host-skip or unrolled action replay.

Implementation: [owned mask helper](../../python/rpu_backend/runtime/masks.py),
[shared decoder](../../src/fused/rpu_qwen3_model.h),
[Gemma4 mask handling](../../python/rpu_backend/adapters/gemma4/_mask.py).

## Sources

- [Public attention registration](../../src/core/rpu_dispatch_registrations.inc)
- [Public tensor-operation boundary](../../src/core/rpu_tensor_ops.inc)
- [Runtime decoder](../../python/rpu_backend/runtime/decoder.py)
- [Cache implementation](../../python/rpu_backend/api/cache.py)
- [Architecture](../../docs/architecture.md)
