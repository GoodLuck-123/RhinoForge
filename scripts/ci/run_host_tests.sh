#!/usr/bin/env bash
# Project Runner supplies read-only dependencies; each job owns a fresh venv.
set -euo pipefail
: "${CI_PROJECT_DIR:?Run through the project CI job}"
: "${CI_JOB_ID:?Run through the project CI job}"
: "${RPU_HOST_PYTHON:?Runner must provide its dedicated Python 3.12}"
: "${RPU_HOST_SITE_PACKAGES:?Runner must provide its dedicated dependency directory}"

cd "$CI_PROJECT_DIR"
rf_job_venv="$CI_PROJECT_DIR/.ci-venv-$CI_JOB_ID"
test ! -e "$rf_job_venv"
"$RPU_HOST_PYTHON" -I -B -m venv "$rf_job_venv"

# Do not inherit developer site-packages or an editable rpu_backend install.
export PYTHONPATH="$CI_PROJECT_DIR:$CI_PROJECT_DIR/python:$RPU_HOST_SITE_PACKAGES"
export PYTHONNOUSERSITE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1
export PIP_REQUIRE_VIRTUALENV=true PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
# Explicit nodes keep local regression checks proportional to the change.
if (( $# == 0 )); then
    set -- tests python/rpu_backend/tests/acceptance
fi
exec "$rf_job_venv/bin/python" -m pytest -q -ra "$@"
