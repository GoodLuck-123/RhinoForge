# Model validation

[简体中文](validation_policy.zh.md) | English

Use checks that cover the changed behavior:

- Run affected host tests once.
- For a device execution change, use one representative smoke test per path, normally 3–10 replays.
- Check failure boundaries before weight or KV mutation, input freshness, output ownership and BUILD/REPLAY lifecycle.
- Investigate invalid shapes, non-finite values, wrong routes and cache corruption. Stop when the behavior and direct regression checks pass.

Use an isolated environment with matching runtime assets. Confirm the board is idle and acquire its device lock. Full model/precision/length matrices, more than 30 iterations, repeated clean-room runs and custom release-validation workflows require an explicit request.

<a id="numerical-evaluation"></a>
## Acceptance contracts

Runtime checks cover shape, dtype, finite values, input freshness, route ownership and KV/cache/SPM/DMA/Graph invariants. Preserve artifact identity, integer or structural equality, and explicitly deterministic replay.

When diagnosing differing outputs, match weights, quantization, inputs, preprocessing and reference semantics. Floating-point rounding and accumulation differences do not fail a model solely because of a historical universal scalar cutoff. Preserve independently specified task requirements and operator-test tolerances; they apply to their own contracts.

Record the tested build, profile and input scope. Keep model evaluation data and reports outside this source repository.
