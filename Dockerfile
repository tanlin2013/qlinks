# syntax=docker/dockerfile:1.7

# The official Python image is multi-architecture. Docker selects the image
# matching --platform (or the Docker host by default), so the same Dockerfile
# works for linux/amd64 and linux/arm64.
ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

LABEL maintainer="TaoLin tanlin2013@gmail.com"

ARG UV_VERSION=0.12.0
ARG QLINKS_EXTRAS=
ARG QLINKS_NOTEBOOK_PACKAGES="jupyterlab>=4,<5 ipykernel>=6,<7"
ARG TARGETPLATFORM
ARG TARGETARCH

# Keep the default Docker image lightweight and independent of optional extras.
# Tensor-network extras are supported on Python < 3.14 because quimb/autograd/numba
# are currently constrained that way in pyproject.toml. Build a TN image explicitly with
# --build-arg PYTHON_VERSION=3.13 --build-arg QLINKS_EXTRAS=tn.
# The Docker-only ``notebook`` feature installs JupyterLab and ipykernel without
# making notebook tools part of qlinks' normal runtime dependency graph.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    PATH="/workspace/qlinks/.venv/bin:${PATH}"

WORKDIR /workspace/qlinks

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libblas-dev \
        liblapack-dev \
        libcairo2-dev \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

# uv.lock is cross-platform; uv evaluates markers and chooses distributions for
# the active Python/platform when synchronizing the image environment.
COPY pyproject.toml uv.lock README.md ./
RUN echo "Building qlinks for ${TARGETPLATFORM:-default} (${TARGETARCH:-default})" && \
    echo "Optional Docker features/extras: ${QLINKS_EXTRAS:-none}" && \
    uv lock --check && \
    set -- && \
    for extra in ${QLINKS_EXTRAS}; do \
        if [ "${extra}" != "notebook" ]; then set -- "$@" --extra "${extra}"; fi; \
    done && \
    uv sync --locked --no-default-groups --no-install-project "$@" && \
    if printf ' %s ' "${QLINKS_EXTRAS}" | grep -q ' notebook '; then \
        uv pip install --python .venv/bin/python ${QLINKS_NOTEBOOK_PACKAGES}; \
    fi

COPY . .
RUN set -- && \
    for extra in ${QLINKS_EXTRAS}; do \
        if [ "${extra}" != "notebook" ]; then set -- "$@" --extra "${extra}"; fi; \
    done && \
    uv sync --locked --no-default-groups "$@" && \
    uv run --no-default-groups python scripts/verify_optional_environment.py --extras "${QLINKS_EXTRAS}"

# PyCharm can use the project virtual environment's Python directly.
CMD ["python"]
