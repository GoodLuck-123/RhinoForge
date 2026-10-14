# Release assets

This directory contains the operator-manifest generator and provenance manifests
for public model sources and weights. Runtime binaries, opaque operator payloads,
and model checkpoints are distributed separately under their own terms.

RhinoForge `1.3.0.dev0` is a source-only development version. Matching runtime
binaries are not published. Build the Launch revision and use the two operator
assets pinned in [runtime-dependencies.json](runtime-dependencies.json).
Rhino Launch uses release version `1.3.0.dev0`, CMake version `1.3.0` and ABI
version `1`. Keep each operator payload with its adjacent `.ref.kernels` file.
For packaged runtime assets, select a published release and its matching source.
See the [release policy](../docs/release_policy.md).

Keep model evaluation data and reports outside source releases, package artifacts and release notes.

For version changes and installation steps, see the [changelog](../CHANGELOG.md)
and [upgrade guide](../docs/upgrading_1_2.md). Request separately distributed
assets through the [runtime asset guide](../docs/runtime_assets.md#request-runtime-assets).

For current examples, see [Examples](../examples/README.md); for
installation, see [runtime assets](../docs/runtime_assets.md). Use the
[proportionate validation policy](../docs/validation_policy.md) for code changes.

`generate_kernel_manifest.py` is a distribution utility. Its output lists public
host-side kernel references intersected with a provider-supplied kernel inventory;
it neither modifies nor verifies the implementation inside an opaque asset.
Changing a sidecar cannot add missing kernels to the payload.
