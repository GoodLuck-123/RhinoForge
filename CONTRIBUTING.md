# Contributing to RhinoForge

Contributions should be small, reviewable, and tied to a concrete inference or
model-porting need. Open an issue before beginning a large model port or a new
fused subsystem so the intended profile and public operator coverage can be
agreed first.

## Development setup

Follow the [source build guide](docs/development_build.md) to prepare Python 3.12
with development headers, an explicit C++17 compiler, a fresh venv and matching
external assets. Current `1.3.0.dev0` source uses the archived sanitized Release
Rhino Launch **1.2.0** pinned by
[runtime-dependencies.json](release/runtime-dependencies.json), not a same-version
development Launch package. After the guide's build-dependency installation,
install the contributor extra from the repository root:

```bash
python -m pip install -e '.[dev]' -c constraints/security.txt --no-build-isolation \
  -Cbuild-dir="$RHINOFORGE_DEV_BUILD" \
  -Ccmake.define.rhino_launch_DIR="$RHINO_LAUNCH_ROOT/lib/cmake/rhino_launch" \
  -Cbuild.tool-args=-j2
```

Set `RPU_KERNEL_LIB_PATH` and `RPU_KERNEL_LIB_PATH_EXPANSION` to the corresponding
release-matched assets, each with its unchanged sidecar. Follow
[runtime setup](docs/runtime_assets.md) for nested extraction, platform SDK and
device access, and [getting started](docs/getting_started.md) for device locking.
Restricted packages and assets must stay outside the repository.

## Change requirements

- Keep the Python import name `rpu_backend` and the public interfaces documented
  in [`docs/api_reference.md`](docs/api_reference.md).
- Treat FP16, SPM capacity, multi-core dispatch, graph lifetime, and irreversible
  weight swizzling as explicit contracts rather than inferred behavior.
- Reject unsupported model profiles before downloading or mutating weights.
- Reuse public operators and the adapter template before adding a fused C++
  subsystem. Follow [`docs/model_porting.md`](docs/model_porting.md).
- Add the smallest board-free test that protects new validation or planning
  logic. Model execution changes also require CPU-reference versus device evidence
  on the affected profile.
- Update the model, asset, API, runtime-configuration, or quantization document
  whenever a public contract changes.

Do not commit restricted device implementation details, internal program formats,
operator-asset format parsers, disassemblers, generator sources, restricted
binaries or assets, credentials, private paths, private infrastructure details,
or customer-confidential material. Host-side operator wrappers, SPM planning,
multi-core scheduling, graph/batch execution, DMA behavior, and launch ABI
integration are in scope.

## Validation

Run the checks applicable to the change:

```bash
python -m compileall -q python/rpu_backend examples
python -m pytest
python -m build --sdist --wheel
```

Hardware changes additionally need the profile-specific numerical,
warmup/replay, graph-lifecycle, and multi-input checks described in
[`docs/model_porting.md`](docs/model_porting.md). Record the exact source,
Rhino Launch, operator-asset, model-checkpoint, and configuration versions used.

Before submitting, inspect the complete diff for generated files, private
information, restricted artifacts, and unrelated cleanup. Contributions are
accepted under the repository's [Apache-2.0 license](LICENSE).

## Sign your work

RhinoForge uses the [Developer Certificate of Origin 1.1](DCO.md). Add a
`Signed-off-by` line to every commit with Git's standard sign-off option:

```bash
git commit -s
```

The sign-off certifies that you have the right to submit the contribution under
the project license. Unless explicitly stated otherwise, intentionally
submitted contributions are licensed under Apache-2.0 as described by section
5 of that license. Public pull requests pass through the project's
maintainer-operated release and hardware validation gates before they are
eligible to merge.
