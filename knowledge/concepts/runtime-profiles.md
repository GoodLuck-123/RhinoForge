# Runtime profiles

A RhinoForge runtime profile is the complete execution contract for one model
configuration. Bind its settings before installation.

## Profile identity

Treat a profile as a tuple of:

- model/checkpoint revision and precision or quantization format;
- admitted input, sequence, image/view, batch, and action envelope;
- per-handle execution configuration and its current generation;
- RhinoForge, Rhino Launch, and main/expansion operator-asset runtime set; and
- effective public runtime configuration.

Changing any member creates a new profile that needs its own admission,
correctness and lifecycle checks.

## Configuration layers

The public TOML runner applies allowed `example.opt_in` settings before importing
the backend. The deployment environment supplies runtime asset locations and
other deployment-owned values. Model facades may then provide profile-owned
defaults without overwriting explicit values. Native and model state can bind
settings at import, first native use, model construction, or Graph BUILD.

Therefore:

- start a fresh process when changing an import-, native-, model-, or
  build-bound value;
- prefer facade or loader configuration over exporting model-specific switches
  globally;
- record effective values, not only the TOML text; and
- never infer that changing `os.environ` reconfigured an existing model or
  Graph.

The [Pi0.5 runner](../../examples/runners/pi05/optimized.py) selects cold
settings from precision, camera count and language length, then passes
`rpu_execution` to the [policy API](../../python/rpu_backend/api/policy.py).
Apply these settings before weight installation and Graph preparation.
Host thread and allocator settings remain application-owned and are applied by
the common example runner before Torch is imported.

## Safe profile changes

Classify a setting before changing it:

| Class | Typical effect | Required response |
|---|---|---|
| Deployment binding | Selects verified runtime/model assets | Re-verify compatibility and provenance |
| Execution planning | Changes chunks, padding, memory, or Graph signatures | Use the owning session's update API between requests; retire affected Graphs |
| Numerical profile | Changes precision, accumulation, quantization, or model inputs | Treat as a new correctness profile |
| Performance topology | Changes batching, replay, fusion, or scheduling | Prove semantic equivalence and lifecycle before timing |
| Diagnostic | Adds logs, exports, traces, or synthetic overrides | Keep out of production and review generated artifacts |

Do not compose a production profile by copying individual fast-path switches
from another model. Use the owning facade's complete admitted profile and
override only settings documented for that exact path.

`reconfigure_rpu_execution` replaces each supplied stage's hot fields and keeps
omitted stages. An empty stage clears planning overrides while preserving bound
cold fields such as `linear_acc32`. The same rule applies to component stages.
An unchanged mapping leaves the generation and Graphs unchanged.

The session serializes forward and reconfiguration. A changed mapping is
validated before live state changes; native controls, Python configuration and
generation commit together. A recoverable failure restores the old settings.
Failed rollback or Graph retirement marks execution unsafe and requires a fresh
process. Graph retirement precedes native-handle destruction. Reading
configuration or statistics does not execute the model.

After a committed change, the next forward builds the replacement Graph and
later matching calls replay it. A frozen cache must enter `begin_warmup()`
before rebuilding: `clear()` alone preserves its frozen state. Owners perform
these steps through the update API. Changes during a forward or after failure
or close are rejected. Query `rpu_execution_stats` for the active generation.
Precision and core topology remain cold fields.

Machine-readable schema names, hash domain separators and capability identifiers
are protocol data. Keep their exact spelling across planner, native consumer and
tools; changing a display name must not change these values.

## Ownership and teardown

Import- and first-native-use settings belong to the process. Model settings
belong to one constructed handle, and BUILD settings belong to the resulting
Graph entries. A policy `close()` releases the resources it owns; it does not
undo process-cold registration, native selection, or irreversible weight
conversion. Use one live policy owner per process and start a fresh process to
change any process-, native-, model-, or BUILD-bound profile.

Residual collective routing and Linear tiling are not profile switches. The
shared wrappers select their generated schedules from validated tensor
geometry. A profile still owns its precision, admitted shapes, mixed-precision
boundaries, Graph plan, and task evidence; do not add retired route or tile
selectors to reproduce an older run.

## Sources

- [Runtime configuration](../../docs/runtime_config.md)
- [Examples](../../docs/model_support.md)
- [Run and debug models](../../docs/model_testing.md)
- [Restricted runtime assets](../../docs/runtime_assets.md)
- [Automatic residual reduction](allreduce-routing.md)
- [Generated Linear auto-tiling](linear-autotiling.md)
