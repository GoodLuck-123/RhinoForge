# RhinoForge agent guide

RhinoForge is an inference-only PyTorch `PrivateUse1` backend exposed as
`torch.rpu`. Its distribution name is `rhinoforge`; its Python package remains
`rpu_backend`.

## Read before changing code

1. Read [model support](docs/model_support.md) for the exact public profile.
2. Read [architecture](docs/architecture.md) for runtime ownership and execution
   contracts.
3. Read [API reference](docs/api_reference.md) before changing a public surface.
4. Use the relevant checks in [model validation](docs/validation_policy.md).
   Full profile certification applies only to a requested acceptance/porting run;
   ordinary fixes follow the validation scope below.
5. For model work, start with the
   [capability review](docs/model_porting_capability_review.md), then follow
   [model porting](docs/model_porting.md), the decoder
   [step-by-step checklist](docs/new_model_step_by_step.md), and the
   [`rhinoforge-port` skill](.agents/skills/rhinoforge-port/SKILL.md) as
   applicable. Use [porting pitfalls](docs/pitfalls.md) before adding a
   workaround.
6. Query [the knowledge index](knowledge/INDEX.md) before searching source for a
   documented concept.

Use the [unified installation and development guide](docs/getting_started.md)
for Miniconda3, the single `RHINOFORGE_ENV`, source/wheel builds and complete
environment bundles; use [restricted runtime assets](docs/runtime_assets.md) for
external asset identities and SDK setup,
[model testing](docs/model_testing.md) for runnable entry points and profiling,
and [runtime configuration](docs/runtime_config.md) for supported controls.

## Validation scope and budget

- Use the smallest meaningful validation set for the requested change. Run
  affected host tests once; do not expand to the full suite when targeted checks
  already cover the changed path.
- For a device-specific fix, default to one representative hardware smoke test per
  affected execution path. Use 3–10 replay iterations, or 2 warmups plus 5–10
  measured iterations for performance, unless the user requests another protocol.
- Stop when the requested behavior and its direct regression checks pass. Reuse
  valid evidence for the same build/profile instead of recertifying it.
- Full model/precision/length matrices, more than 30 iterations, repeated clean-room
  benchmarks, cryptographic evidence closures, sealed generations, custom result
  publication harnesses, and layered validation gates require an explicit user
  request.
- Keep diagnostic follow-ups bounded to the failed hypothesis. Once the execution
  path is confirmed, record ordinary accumulated numerical differences instead of
  escalating them into additional gates.

Follow the [acceptance contracts](docs/validation_policy.md#numerical-evaluation): preserve runtime, artifact and deterministic-replay checks; match reference semantics when diagnosing differences. Historical universal floating-point cutoffs are not model acceptance gates. Keep independently specified task requirements.

## Required engineering contracts

- Bind support to the checkpoint, precision, input envelope, execution, Launch and operator assets.
- Reject unsupported profiles before full weight loading or irreversible
  transformation.
- Transform a model's weight layout exactly once. Reload a clean model after a
  partial transformation or failed installation.
- Respect process ownership: the main fused CausalLM path permits one live
  device-resident model or policy per process.
- Keep `FusedModelBase::declare_buffers` deterministic and free of launch side
  effects. Declare state that must survive Graph replay as persistent.
- For a repeated Graph signature, prove one BUILD followed by stable REPLAY:
  replay count grows, cache size stays fixed, and `cache_invariant_ok()` remains
  true. Treat a bounded one-shot path as an explicit model-contract exception.
- Include every semantic input in the Graph signature or update it through the
  documented mutable DMA path.
- Use fixed DMA only for addresses stable across replay. Give Python-visible
  outputs independent storage when callers may retain them.
- Route multi-core work through the framework batch/Graph path or explicitly
  enable host-parameter broadcast for an immediate launch.
- Keep device-program implementation details and opaque operator-asset formats
  outside this repository. Public host wrappers may describe tensor metadata,
  memory locations, synchronization, core selection, and parameter packing.
- Treat exported activations and low-level hardware trace artifacts as
  sensitive application data.

## Change discipline

- Make the smallest change that satisfies a verified profile; reuse public
  helpers before adding abstractions or dependencies.
- Keep profile guards, cache ownership, teardown, and error paths explicit.
- Run board-free checks first. New ports must satisfy the profile, warmup, Graph, input and end-to-end requirements in [model porting](docs/model_porting.md); existing-model fixes follow the validation scope above.
- Update public documentation and [the knowledge workflow](knowledge/WORKFLOWS.md)
  when a public contract changes.
