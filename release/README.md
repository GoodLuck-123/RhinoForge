# Release assets

This directory contains the operator-manifest generator and provenance manifests
for public model sources and weights. Runtime binaries, opaque operator payloads,
and model checkpoints are distributed separately under their own terms.

RhinoForge v1.2.0 uses the provider-delivered
`RhinoForge-runtime-v1.2.0.tar.gz`, extracted under `RhinoForge-runtime-v1.2.0/`,
with `rhinoOpLib_current.ref` and `rhinoExpansionOpLib_current.ref`, each with
its adjacent `.ref.kernels` manifest. Their identities are pinned in
[runtime-dependencies.json](runtime-dependencies.json).
Rhino Launch uses the same `1.2.0` release version, with CMake version `1.2.0`
and ABI version `1`. `RELEASE.txt` selects the matching Release package. Verify the outer checksum and inner `SHA256SUMS` before using
the [provider paths](../docs/runtime_assets.md#configure-paths).

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
