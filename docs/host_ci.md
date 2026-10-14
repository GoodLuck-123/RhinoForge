# Host CI Runner

The `host-runtime` job uses the project-locked Runner tagged `rhinoforge-host`
on a dedicated CPU server. Native builds and board checks use a separate
Runner and tag; this job does not access a device.

## Dependency image

Build the Docker image from
[`scripts/ci/host-image`](../scripts/ci/host-image). It provides Python 3.12,
CPU PyTorch 2.10.0, CMake 4.1.3, Ninja 1.13.0 and the pinned host-test dependencies. The Pi0.5
host tests require LeRobot 0.5.1 and its actual import dependencies; robot and
training extras and model assets are not installed. Keep these pins aligned
with the `vla` optional dependencies in [pyproject.toml](../pyproject.toml).

Build the image on the Runner host before enabling a new image version:

```sh
docker build -t rhinoforge-host-ci:py312 scripts/ci/host-image
```

The image supplies `RPU_HOST_PYTHON=/usr/local/bin/python` and
`RPU_HOST_SITE_PACKAGES=/usr/local/lib/python3.12/site-packages`.
[`run_host_tests.sh`](../scripts/ci/run_host_tests.sh) creates a new venv for each
job and adds only the current checkout and image dependency directory to its
Python path. Jobs do not install the backend or modify developer environments.
Dependency images remain separate from the source checkout and contain no
editable backend install or model weights.

## Runner isolation

Use a project-locked Runner for tagged RhinoForge host jobs. Keep its deployment
configuration in the administrator's private operations directory. Job containers
must not mount board devices, developer home directories, Conda environments or
the host Docker socket. Use a dedicated cache and bounded CPU/memory resources.

When dependencies change, drain the Runner, build a new image tag and update its
allowed/default image. One normal host CI run checks the deployment. Hardware
verification uses a separate board worker with an idle-device check and the
existing device lock; avoid host CI load during device checks.
