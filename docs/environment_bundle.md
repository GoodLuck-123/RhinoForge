# RhinoForge dependency environment bundle design

[简体中文](environment_bundle.zh.md) | English

**Status: the design is agreed; the new dependency bundle and its tools have
not been built, validated or published.** No Release URL or SHA256 exists for
this bundle. Install today using the [Miniconda3 path](getting_started.md#source-build).
This page specifies delivery for maintainers; it does not introduce a separate
user or developer environment.

## Four separate deliveries

| Component | Contents | Delivery |
|---|---|---|
| Board `platform` | OS, kernel, driver, complete SDK including user-space libraries, permissions | Platform provider |
| Dependency environment | Python, third-party dependencies, build/development and media tools | GitHub Releases after preparation and validation |
| RhinoForge | Git history, source, examples, documentation; native extension built locally | GitHub tag/commit checkout and local build |
| Runtime assets | Rhino Launch development package, main/DDR-expansion operators and sidecars | Separate delivery under the provider's terms |

Model weights, inputs and model-specific external source remain separate.
The public dependency archive excludes RF source, wheels, installed `rhinoforge` /
`rpu_backend`, `rpu_backend.so`, all board SDK libraries, Launch, operator assets,
models and private caches. `libHxMem.so`, `libHxil.so` and `librpu_api.so` belong to
the `platform` SDK and are supplied by the `platform`.

## Recommended `platform`

The recommended, previously validated board SDK is
**`EA_R1SDK_2026_08F-2156007a`**. Existing validation used R1, Linux aarch64,
Ubuntu 22.04, glibc 2.35, GCC 11.4 and Python 3.12.15. This is evidence from
previous deployments, **not validation of the new dependency bundle**.
The provider confirms the matching OS/kernel/driver combination, installation
and access policy. Other SDK versions require confirmation and validation;
the recommendation is not a universal minimum.

**Rhino Launch 1.2.0 is a different component version**, currently ABI 1 at
`3dd1b410f5f46373997fef3dfbf9b67eecec6a84`. Launch and both operator assets remain
exactly pinned by [runtime-dependencies.json](../release/runtime-dependencies.json).
Environment, RF, board SDK and Launch versions are independent.

## Archive contents

The first public archive has one `full` `profile` covering default dependencies,
`vision`, `vla`, `audio` and `dev`, shared by deployment and development. Read requirements
from the chosen source's [pyproject.toml](../pyproject.toml), then lock every direct
and transitive dependency.

| Category | Planned contents |
|---|---|
| Python | Python 3.12, development headers, pip, compatible packaging/setuptools/wheel, `conda-unpack` |
| Inference | Torch, Transformers, NumPy, Accelerate, Hugging Face Hub and all transitive dependencies |
| Extensions | torchvision, torchaudio, LeRobot, Diffusers and image/audio libraries from the selected extras |
| Build and development | scikit-build-core, CMake, Ninja, build, pytest, conda-pack |
| Media tools | SoX CLI, FFmpeg, libsndfile and required runtime libraries |
| Reproduction and redistribution | Manifest, exact locks, origins/checksums, licenses and required third-party sources |

Keep conda-pack's prefix relocation records. Build with Miniconda3, but ship only
the application environment, without Miniconda base or the conda manager.
The host still supplies Bash, Git, download/extraction utilities and system C/C++
build tools: GCC/G++, binutils, glibc development headers and pkg-config. The
planned installer uses host Python 3.10 or newer; RF runs/builds with the bundled
Python 3.12. Use the host preparation commands in the unified guide.

Dependencies change less often than application code, but security fixes,
Python/Torch ABI changes and model requirements still require new versions.
Current pins are a recipe starting point, not evidence that a new public
archive has passed compatibility or security review.

## Deployment and development

Both paths join the same source build, model preparation, run and development steps:

1. Path A: install Miniconda3, create `RHINOFORGE_ENV`, install third-party dependencies.
2. Path B: download the bundle, verify SHA256, extract/relocate, activate `RHINOFORGE_ENV`.
3. Checkout GitHub source, configure `platform` SDK and separate assets, check requirements, build RF.
4. Prepare models/inputs and run under the `platform`'s device policy; develop in the same environment.

The one-command goal covers dependency environment deployment. It avoids
individual dependency downloads, but **RF still needs a local build**. Offline
use also requires the source, system tools, SDK, runtime assets and models to
have been transferred beforehand.

Start at a fixed tag/commit and create a development branch in the Git clone.
Keep source outside the environment. Use the same `RHINOFORGE_ENV` for regular and
editable installation. Reinstall/rebuild after Python/C++ changes; do not retain
an incompatible native extension after updating Python files.
Source changes preserving dependencies and ABI can reuse the environment.
Check changed requirements before selecting an environment; do not silently
upgrade Torch or other dependencies.

Planned deployment layout, not the contents of an existing archive:

    rhinoforge-work/
      dependencies/
        activate.sh
        environment-manifest.json
        locks/, licenses/, sources/    # Third-party recipes, licenses, required sources
        env/                          # RHINOFORGE_ENV; initially contains no RF
      source/                         # RHINOFORGE_SOURCE; user's Git clone
      build/                          # RHINOFORGE_BUILD
      runtime/                        # Separately supplied Launch and operators
      models/, results/               # Separately prepared/generated by the user

The SDK may live at another `platform`-selected location. Activation sets Python
and environment identity without selecting nonexistent bundled source/assets
or overwriting SDK loader paths. Activate first, then configure Launch,
operators and SDK with the [runtime guide](runtime_assets.md).
Install at the final path; to relocate again, deploy from the original archive.
Architecture/glibc checks are necessary conditions, not support for arbitrary
Linux/SDK combinations.

## Binding environments to GitHub source

Use independent identifiers, for example `deps-2026.10.1` and
`rhinoforge-deps-2026.10.1-full-linux-aarch64-py312.tar.gz`.
These are naming examples, **not published versions**. Keep published bytes and
checksums immutable; do not rely on a mutable latest URL for reproduction.
Multiple source commits can reference one environment.

A planned repository file, `release/environment.json`, will identify the
recommended environment for that checkout. Neither the file nor its reader is
implemented yet. Populate real download fields at publication.

| Field | Purpose |
|---|---|
| `schema`, `environment_id`, `profile` | Dependency format, independent version and extras |
| `platform`, `python` | Architecture, validated OS/glibc baseline and Python ABI |
| `requirements_sha256` | Normalized default/extras/build requirements and security constraints, excluding RF's version |
| `archive_url`, `archive_sha256` | Actual published archive and its digest |
| `manifest_sha256`, `lock_sha256` | Binding to the manifest and exact locks |

The archive's `environment-manifest.json` uses a new `rhinoforge-dependency-env/v1`
`schema` and records environment identity, builder versions, exact Conda/pip
versions and build strings, origins, download digests and `recipe_source_commit`.
That commit identifies the recipe; it neither embeds RF nor restricts the
environment to one RF commit. SDK recommendations and Launch/operator identities
remain in source documentation and the runtime lock, outside the package inventory.

Before building, planned tooling compares identity/digests and validates the
checkout's Python, runtime and build requirements, including extras and markers.
pip check cannot validate an uninstalled RF distribution's requirements.
Only after these checks should builds use `--no-build-isolation` `--no-deps` to
avoid downloads or dependency changes. Changed requirements need revalidation
and an updated source binding, even if the same archive satisfies them; a changed
digest is not silently accepted.

## Preparation and publication

1. Create a fresh dependency environment with Miniconda3. Install Conda packages
   before locked pip packages; preserve Conda-managed files. Do not install RF
   or copy a deployed environment.
2. Check the inventory and files for RF installations, editable paths, SDK
   libraries, Launch/operators, models, credentials and private configuration.
   Collect licenses, notices and required corresponding source for actual
   third-party builds. SoX materials do not cover FFmpeg, libsndfile or others.
3. Pack the new format, deploy at a different absolute prefix and verify
   relocation, dependencies, imports and media tools. The new installer retains
   SHA256/platform/path checks and failure rollback, and reports that dependencies
   are installed but RF has not been built.
4. Build the chosen GitHub commit in that deployed environment. With the
   recommended SDK and separate assets, run Qwen3-0.6B FP16 and check affected
   vision/vla/audio paths under the [validation policy](validation_policy.md).
   Imports alone do not validate models. Do not write RF build/run results
   back into the dependency archive being published.
5. Publish the archive, SHA256, manifest, exact locks, licenses and required
   sources on GitHub Releases; update the source binding and unified guide.
   Environment Releases use independent `deps-*` tags and need dedicated checks.
   Existing `check_release_version.py` only accepts RF `v*` tags matching `VERSION`;
   do not reuse or remove that source-release check.

Do not present placeholder URLs or digests as available downloads. Releases
is the planned common entry point; the `platform` SDK and controlled assets
retain their separate delivery channels.

## Existing tools and historical archives

`scripts/pack_conda_bundle.py` and `scripts/install_conda_bundle.py` currently serve
the legacy `rhinoforge-conda-bundle/v1` `full` archive: installed RF, source,
Launch/operators and optional SDK libraries. They do not implement this design
and cannot produce or install the new public dependency archive.
New tools must distinguish formats and reject legacy archives as dependency-only
packages. Keep the historical installer for old deliveries.

Do not rename an internal `full` archive or delete a few directories and publish
it as the new package. This design changes no SDK, Launch or operator asset;
subsequent work produces a new third-party dependency artifact.
