# RhinoForge dependency environment bundle

[简体中文](environment_bundle.zh.md) | English

The current environment is [`deps-2026.10.1` (preview)](https://github.com/HUIXI-AI/RhinoForge/releases/tag/deps-2026.10.1)
for Linux aarch64 and Python 3.12. Deploy through the
[unified guide](getting_started.md#bundle-deploy). This page describes its contents,
source binding and maintenance; it does not introduce a separate developer environment.

## Four separate deliveries

| Component | Contents | Delivery |
|---|---|---|
| Board platform | OS, kernel, driver, complete SDK including user-space libraries, permissions | Platform provider |
| Dependency environment | Python, third-party dependencies, build/development and media tools | GitHub Releases |
| RhinoForge | Git history, source, examples, documentation; native extension built locally | GitHub tag/commit checkout and local build |
| Runtime assets | Rhino Launch development package, main/DDR-expansion operators and sidecars | Separate delivery under the provider's terms |

Model weights, inputs and model-specific external source remain separate.
The public dependency archive excludes RF source, wheels, installed `rhinoforge` /
`rpu_backend`, `rpu_backend.so`, all board SDK libraries, Launch, operator assets,
models and private caches. `libHxMem.so`, `libHxil.so` and `librpu_api.so` belong to
the platform SDK and are supplied by the platform.

## Recommended platform

The recommended, previously validated board SDK is
**`EA_R1SDK_2026_08F-2156007a`**. Existing validation used R1, Linux aarch64,
Ubuntu 22.04, glibc 2.35, GCC 11.4 and Python 3.12.15. See the
[installation guide](getting_started.md#platform-prerequisites) for the recommended
baseline and platform preparation.
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

| Category | Contents |
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
standalone installer uses host Python 3.10 or newer; RF runs/builds with the bundled
Python 3.12. Use the host preparation commands in the unified guide.

Dependencies change less often than application code, but security fixes,
Python/Torch ABI changes and model requirements still require new versions.
Current pins are a recipe starting point, not evidence that a new public
archive has passed compatibility or security review.

## Deployment and development

Both paths join the same source build, model preparation, run and development steps:

1. Path A: install Miniconda3, create `RHINOFORGE_ENV`, install third-party dependencies.
2. Path B: download the bundle, verify SHA256, extract/relocate, activate `RHINOFORGE_ENV`.
3. Checkout GitHub source, configure platform SDK and separate assets, check requirements, build RF.
4. Prepare models/inputs and run under the platform's device policy; develop in the same environment.

The one-command goal covers dependency environment deployment. It avoids
individual dependency downloads, but **RF still needs a local build**. Offline
use also requires the source, system tools, SDK, runtime assets and models to
have been transferred beforehand.

Start at a fixed tag/commit and create a development branch in the Git clone.
Keep source outside the environment. Use the same `RHINOFORGE_ENV` for regular and
editable installation. Reinstall/rebuild after Python/C++ changes; do not retain
an incompatible native extension after updating Python files.
Source changes preserving the normalized dependency recipe and ABI can reuse
the environment. A changed recipe currently requires a new environment archive
and version, even when installed versions still satisfy the new constraints.
Do not silently upgrade Torch or other dependencies.

Deployment layout used by the tools:

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

The SDK may live at another platform-selected location. Activation sets Python
and environment identity without selecting nonexistent bundled source/assets
or overwriting SDK loader paths. Activate first, then configure Launch,
operators and SDK with the [runtime guide](runtime_assets.md).
Install at the final path; to relocate again, deploy from the original archive.
Architecture/glibc checks are necessary conditions, not support for arbitrary
Linux/SDK combinations.

## Binding environments to GitHub source

The independent environment ID is `deps-2026.10.1`; its archive is
`rhinoforge-deps-2026.10.1-full-linux-aarch64-py312.tar.gz`.
Keep published bytes and checksums immutable; do not rely on a mutable latest URL
for reproduction. Multiple source commits can reference one environment.

[`release/environment.json`](../release/environment.json) identifies the recommended
environment for a checkout and records its actual download information.

| Field | Purpose |
|---|---|
| `schema`, `environment_id`, `profile` | Dependency format, independent version and extras |
| platform, `python` | Architecture, validated OS/glibc baseline and Python ABI |
| `requirements_sha256` | Normalized default/extras/build requirements and security constraints, excluding RF's version |
| `archive_url`, `archive_sha256` | Actual published archive and its digest |
| `manifest_sha256`, `lock_sha256` | Binding to the manifest and exact locks |

The archive's `environment-manifest.json` uses a new `rhinoforge-dependency-env/v1`
`schema` and records environment identity, builder versions, exact Conda/pip
versions and build strings, origins, download digests and `recipe_source_commit`.
That commit identifies the recipe; it neither embeds RF nor restricts the
environment to one RF commit. SDK recommendations and Launch/operator identities
remain in source documentation and the runtime lock, outside the package inventory.

Before building, `scripts/dependency_environment.py` compares identity/digests and validates the
checkout's Python, runtime and build requirements, including extras and markers.
`pip check` cannot validate an uninstalled RF distribution's requirements.
Only after these checks should builds use `--no-build-isolation` `--no-deps` to
avoid downloads or dependency changes. The recipe digest is part of the immutable
archive manifest: changed requirements need a new archive/version, validation and
source binding, even if the old installed versions satisfy the new constraints.

## Preparation and publication

1. Create a fresh dependency environment with Miniconda3. Install Conda packages
   before locked pip packages; preserve Conda-managed files. Do not install RF
   or copy a deployed environment.
2. Check the inventory and files for RF installations, editable paths, SDK
   libraries, Launch/operators, models, credentials and private configuration.
   Collect licenses, notices and required corresponding source for actual
   third-party builds. SoX materials do not cover FFmpeg, libsndfile or others.
3. Pack the new format, deploy at a different absolute prefix and verify
   relocation, dependencies, imports and media tools. `install_dependency_env.py` retains
   SHA256/platform/path checks and failure rollback, and reports that dependencies
   are installed but RF has not been built.
4. Build the chosen GitHub commit in that deployed environment. With the
   recommended SDK and separate assets, run Qwen3-0.6B FP16 and check affected
   vision/vla/audio paths under the [validation policy](validation_policy.md).
   Imports alone do not validate models. Do not write RF build/run results
   back into the dependency archive being published.
5. Publish the archive, SHA256, manifest, exact locks, licenses and required
   sources on GitHub Releases; update the source binding and unified guide.
   Environment Releases use independent `deps-*` tags. `check_release_version.py`
   dispatches those tags to `check_dependency_release.py` to check environment
   identity, the source recipe and published digests. RF `v*` tag/`VERSION`
   validation remains unchanged.

GitHub Releases is the common download entry point for dependency environments.
The platform SDK and controlled assets retain separate delivery channels. An
environment preview does not replace the stable RF source release or its latest entry.

## Existing tools and historical archives

`scripts/pack_conda_bundle.py` and `scripts/install_conda_bundle.py` currently serve
the legacy `rhinoforge-conda-bundle/v1` `full` archive: installed RF, source,
Launch/operators and optional SDK libraries. They continue to serve historical deliveries. The new
`pack_dependency_env.py` / `install_dependency_env.py` use an independent schema
and reject legacy full archives as dependency-only packages. Use the new tools
for the public dependency workflow.

Do not rename an internal `full` archive or delete a few directories and publish
it as the new package. The dependency archive changes no SDK, Launch or operator
asset; it is a separately built third-party environment artifact.

## Tool usage and release checks

Maintainers prepare third-party dependencies in a clean prefix with no RF
installation, plus reviewed `licenses/`, `sources/` and `locks/` outside that
prefix. Record manually built SoX in `locks/native-packages.json`: version, public
source URL, source SHA256 and installed-file SHA256 values. For a loader cache
generated by Conda post-link scripts, first verify package ownership of the
generator and referenced libraries and compare regenerated output, then register
only the reviewed file digest. Do not admit arbitrary unknown files through this
inventory. The packer rejects unmanaged files. Ordinary deployment users skip
this preparation.

```bash
python scripts/dependency_environment.py --source "$RHINOFORGE_SOURCE" --print-requirements
python scripts/pack_dependency_env.py \
  --env "$RHINOFORGE_CLEAN_DEPENDENCIES" --source "$RHINOFORGE_SOURCE" \
  --redistribution "$RHINOFORGE_REDISTRIBUTION" \
  --environment-id deps-2026.10.1 \
  --output "$RHINOFORGE_DIST/rhinoforge-deps-2026.10.1-full-linux-aarch64-py312.tar.gz"
```

These variables identify the clean dependency prefix, source checkout, reviewed
third-party materials and existing output directory. Keep the original Conda
package cache until packing finishes; it and the Miniconda manager are excluded
from publication. Check the source binding against real archive/manifest bytes:

```bash
python scripts/check_dependency_release.py --tag deps-2026.10.1 \
  --archive "$RHINOFORGE_DIST/rhinoforge-deps-2026.10.1-full-linux-aarch64-py312.tar.gz" \
  --manifest "$RHINOFORGE_DIST/environment-manifest.json"
```

Checksums, origins and license checks do not establish that every third-party
package is free of vulnerabilities. Pinned model dependencies can retain known
upstream issues; package versions, triggers and limitations are documented in
the archive's `KNOWN_ISSUES.md`. Use reviewed models and inputs. This environment
is not a sandbox for arbitrary external model code or files.
