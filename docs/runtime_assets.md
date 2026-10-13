# External runtime assets

[简体中文](runtime_assets.zh.md) | English

RhinoForge source and wheels do not contain Rhino Launch binaries, the external
main and expansion operator assets, the board SDK, or model weights. Obtain a runtime package
compatible with the source revision you are installing from the runtime provider.
Follow that package's distribution terms and integrity instructions.

For RhinoForge v1.2.0rc1, request
`RhinoForge-runtime-v1.2.0rc1.tar.gz` and its `.tar.gz.sha256` file from the provider.
The archive extracts to `RhinoForge-runtime-v1.2.0rc1/` and contains
`rhinoOpLib_current.ref` and `rhinoExpansionOpLib_current.ref`, each with an adjacent `.ref.kernels` manifest.
RhinoForge, Runtime and Rhino Launch share the `1.2.0rc1` release version.
Launch's numeric CMake version is `1.2.0`; its ABI version remains `1`. Use the
exact nested Launch package named by `RELEASE.txt`. See the [release policy](release_policy.md).

## Request runtime assets

Submit an [asset request](https://github.com/HUIXI-AI/RhinoForge/issues/new)
with the source version or commit, board platform, and required Launch, main and
DDR-expansion components. Keep restricted files, credentials and machine details
out of the public issue. Asset access follows the provider's distribution terms.

## Required components

- **Rhino Launch 1.2.0rc1**, built as a sanitized Release package, with headers,
  shared libraries and `lib/cmake/rhino_launch/rhino_launchConfig.cmake`.
  Current CMake checks `graph-arena-pool-v2`, `register-state-token`, Chrome
  hwperf support and the absence of the development Program API.
- **Two operator assets**, built from the main and DDR-expansion branches respectively, matching this source, each with an adjacent
  `<asset>.kernels` manifest. It must contain the kernels required by the public
  models and precision profiles being used.
- **The matching board SDK/runtime**, configured according to the platform's
  installation instructions.

Keep build-time and run-time Launch paths consistent.
Do not mix an old operator payload with a new sidecar or merge payloads locally.

## Configure paths

Verify the provider's checksums when receiving the package, then extract it to a
stable directory outside the source checkout and Python environment. Set paths
to the installed Launch prefix and the unchanged operator asset:

```bash
export RHINO_LAUNCH_ROOT="/absolute/path/to/matching/launch-install"
export RPU_KERNEL_LIB_PATH="/absolute/path/to/RhinoForge-runtime-v1.2.0rc1/rhinoOpLib_current.ref"
export RPU_KERNEL_LIB_PATH_EXPANSION="/absolute/path/to/RhinoForge-runtime-v1.2.0rc1/rhinoExpansionOpLib_current.ref"
export CMAKE_PREFIX_PATH="$RHINO_LAUNCH_ROOT${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
export LD_LIBRARY_PATH="$RHINO_LAUNCH_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -r "$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch/rhino_launchConfig.cmake"
test -r "$RPU_KERNEL_LIB_PATH"
test -r "$RPU_KERNEL_LIB_PATH.kernels"
test -r "$RPU_KERNEL_LIB_PATH_EXPANSION"
test -r "$RPU_KERNEL_LIB_PATH_EXPANSION.kernels"
```

Replace the sample paths with the provider's actual filenames. Remove conflicting
Launch paths from the inherited environment while preserving required SDK paths.
Credentials and machine paths belong in the shell environment, outside model
TOMLs. Continue with [Getting started](getting_started.md) or the
[source build guide](development_build.md).

## Operator roles and replacement

The main and DDR-expansion roles are fixed. [runtime-dependencies.json](../release/runtime-dependencies.json) records their source commits and SHA256 identities. Source and installed-package entries use the same roles across models.

`RPU_KERNEL_LIB_PATH` and `RPU_KERNEL_LIB_PATH_EXPANSION` may point to identical installed copies. Entries reject hash mismatches, swapped roles and conflicting inherited paths. GR00T configurations reject `example.expansion_ref`. Start a new process after replacing either asset.

Update the complete main and expansion assets together with their adjacent
manifests and the source/size/SHA256 records. A partial build or a locally merged
payload is not a replacement for either role. New operator implementations
belong to the operator repository; this repository owns model routing and host
wrappers. To reproduce an older source, use that release's archived runtime set
and original checksums together. Do not point the current source at an older
payload while retaining current dependency metadata.

## Operator manifest

Keep the `.kernels` file adjacent to its asset. Its first line is
`rhinoforge-kernels-v1`, its second is `asset-size=<bytes>`, and the remaining lines
are unique ASCII kernel names in sorted order. The runtime validates this format
and the asset size before resolving kernels. This check catches missing metadata
and size mismatches; it does not verify the operator implementation or replace
the provider's integrity checks.

If a required kernel is missing, obtain a compatible public-model asset from the
provider. Adding a name to the sidecar does not add its implementation to the
payload. Operator source and payload contents remain outside this repository.

## Common setup errors

| Symptom | Action |
|---|---|
| CMake cannot find `rhino_launch` | Set the Launch prefix or pass `-Ccmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch"` |
| CMake rejects a required capability | Obtain a matching sanitized Release Launch package |
| Launch cannot be loaded | Check its `lib` path and the board SDK library paths |
| Asset or sidecar is missing, or a kernel is absent | Obtain the matching asset and unchanged sidecar together |
| device is unavailable | Check device permissions and the platform runtime setup |
