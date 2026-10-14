FROM docker.m.daocloud.io/library/python:3.12-slim

# Host regressions compile small C++ probes; no device runtime is installed.
ARG DEBIAN_MIRROR=https://mirrors.ustc.edu.cn
RUN sed -i "s|http://deb.debian.org|${DEBIAN_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 update \
    && apt-get install -y --no-install-recommends bash git g++ ca-certificates \
    && apt-get clean

ENV RPU_HOST_PYTHON=/usr/local/bin/python \
    RPU_HOST_SITE_PACKAGES=/usr/local/lib/python3.12/site-packages \
    CMAKE_GENERATOR=Ninja \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Keep CPU wheels in their own cached layer when host dependencies change.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps --index-url https://download.pytorch.org/whl/cpu \
      'torch==2.10.0+cpu' 'torchvision==0.25.0+cpu'

COPY requirements-host.txt /opt/rhinoforge-ci/requirements-host.txt
ARG PYPI_INDEX_URL=https://pypi.org/simple
# Keep the reviewed host import set; do not resolve LeRobot robot/training extras.
RUN --mount=type=cache,target=/root/.cache/pip \
    sed '/^torch==/d; /^torchvision==/d' /opt/rhinoforge-ci/requirements-host.txt \
      > /opt/rhinoforge-ci/requirements-rest.txt \
    && python -m pip install --no-deps --index-url "${PYPI_INDEX_URL}" \
      -r /opt/rhinoforge-ci/requirements-rest.txt
