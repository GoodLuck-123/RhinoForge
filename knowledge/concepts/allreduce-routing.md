# Automatic residual reduction

Residual all-reduce has one shared production route. Model call sites provide
the validated tensor geometry, residual ownership and installed core topology;
the common FP16 wrapper uses the consumer core count, with eight as the default. There is no public environment or
per-model selector for choosing another route.

## Contract

- Inputs use the declared FP16 or model-approved mixed-precision boundary.
- The output geometry, source shards, residual storage, and participating
  producer cores are validated before launch.
- The ordinary FP16 route requires `1 <= producer cores <= consumer cores <= 8`;
  each model profile determines the admitted pair. A partial producer clears
  every consumer shard before writing its active prefix.
- The wrapper selects the generated scheduling variant from the per-core data
  size. Callers do not duplicate that choice.
- Immediate multi-core launch still requires broadcast mode; Graph and batch
  paths apply it centrally.
- BUILD and REPLAY must use the same admitted geometry and live residual
  ownership.

Local-SPM, mixed-precision and fixed-layout collective wrappers retain their
eight-core bounds. They do not add a global route switch or admit another
model profile.

## Porting consequence

Do not copy retired route, chunk, or two-stage selectors into a TOML or adapter.
A new producer should prepare its shards through the shared helper, call the
common residual wrapper, and verify same-semantic parity plus stable Graph
lifecycle at the maximum admitted shape.

## Sources

- [Architecture](../../docs/architecture.md#graph-and-batch-execution)
- [Runtime configuration](../../docs/runtime_config.md)
- [Runtime wrapper](../../src/core/rpu_runtime_extras.cpp)
- [Shared kernel cache declarations](../../src/core/rpu_kernel_cache.h)
- [Multi-core broadcast](multicore-broadcast.md)
- [GraphCache capture](graphcache-capture.md)
