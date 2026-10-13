# GraphCache capture

`GraphSignature` is the key for one admitted execution shape and semantic mode.
`GraphCache.capture(signature)` builds the graph on the first call and reuses it
on later calls with the same signature. [Graph API](../../docs/api_reference.md#graph-api)

## Correct reuse

- A production adapter normally owns a dedicated per-model `GraphCache` and
  wraps its top-level native forward in `capture`. [Architecture](../../docs/architecture.md#graph-and-batch-execution)
- Every value that can change the result must be encoded in the signature or
  updated at a stable contract point through a mutable transfer. [Model porting](../../docs/model_porting.md#7-graph-and-cache-invariants)
- A repeated signature is healthy when it builds once, later replay counters
  increase, cache size stays fixed, no second build occurs, and
  `cache_invariant_ok()` remains true. [Architecture](../../docs/architecture.md#graph-and-batch-execution)
- `freeze()` changes a warmed cache to lookup-only mode; a missing or invalid
  entry is rejected instead of being built online. [GraphCache implementation](../../python/rpu_backend/graph/_runtime.py)
- Cache ownership includes teardown. Prefer an explicit per-model cache over a
  shared default when writing a new adapter. [Graph API](../../docs/api_reference.md#graph-api)

`Graph` and `GraphCache` retain a read-only construction policy; extracted
children inherit it. Capture failure preserves the primary exception, attaches
cleanup failures as notes, and releases the scope’s borrowed graph reference.
The caller can evict or clear while keeping the traceback, provided it has not
retained another graph reference. [Capture implementation](../../python/rpu_backend/graph/_runtime.py)

A bounded one-shot graph is an explicit model-contract exception for a
variable-shape path. It must execute one non-trivial graph, return to
passthrough state, and leave retained-cache size unchanged. [Model porting](../../docs/model_porting.md#7-graph-and-cache-invariants)

`GraphRuntimePolicy.batch_snapshot_mode` is cold per-instance configuration. AUTO requires matching Launch capabilities and operator identities; otherwise replay follows the ordinary prepared path. OFF disables snapshots while retaining Graph execution. Pass one frozen policy to related model/component builders. [Snapshot policy](../../docs/runtime_config.md#graph-snapshot-policy).

## Composite and variable-shape owners

A shared execution configuration can control components with different Graph
lifetimes. Qwen3.5 uses bounded one-shot text prefill, retained text decode and
a bounded one-entry vision cache that rebuilds on a shape change. A native
composite can instead own one outer Graph and its shared SPM lease. These modes
share admission and generation rules, not necessarily one cache instance.

A committed execution update retires the affected plans. The next forward
builds with the new generation, followed by stable replay. `clear()` preserves
a cache's frozen state; the owner calls `begin_warmup()` before a required
rebuild. See [runtime profile updates](runtime-profiles.md#safe-profile-changes).

## Sources

- [Architecture: graph and batch execution](../../docs/architecture.md#graph-and-batch-execution)
- [Graph API](../../docs/api_reference.md#graph-api)
- [Canonical Python runtime](../../python/rpu_backend/graph/_runtime.py)
- [Model porting: graph invariants](../../docs/model_porting.md#7-graph-and-cache-invariants)
