# syntax=docker/dockerfile:1.7

# The official Python image is multi-architecture. Docker selects the image
# matching --platform (or the Docker host by default), so the same Dockerfile
# works for linux/amd64 and linux/arm64.
ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

LABEL maintainer="TaoLin tanlin2013@gmail.com"

ARG POETRY_VERSION=2.3.4
ARG QLINKS_EXTRAS=
ARG TARGETPLATFORM
ARG TARGETARCH

# Keep the default Docker image lightweight and independent of optional extras.
# Tensor-network extras are supported on Python < 3.14 because quimb/autograd/numba
# are currently constrained that way in pyproject.toml. Build a TN image explicitly with
# --build-arg PYTHON_VERSION=3.13 --build-arg QLINKS_EXTRAS=tn.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_INSTALLER_ONLY_BINARY="numpy,llvmlite,numba"

WORKDIR /workspace/qlinks

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libblas-dev \
        liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip wheel setuptools && \
    python -m pip install "poetry==${POETRY_VERSION}"

# poetry.lock contains hashes for distributions on all supported platforms; it
# does not copy a macOS wheel into Linux. Poetry evaluates the locked markers
# and chooses the wheel matching TARGETPLATFORM/TARGETARCH at install time.
COPY pyproject.toml poetry.lock README.md ./
RUN echo "Building qlinks for ${TARGETPLATFORM:-default} (${TARGETARCH:-default})" && \
    echo "Optional extras: ${QLINKS_EXTRAS:-none}" && \
    poetry check --lock && \
    if [ -n "${QLINKS_EXTRAS}" ]; then \
        poetry install --only main --extras "${QLINKS_EXTRAS}" --no-root --no-ansi; \
    else \
        poetry install --only main --no-root --no-ansi; \
    fi

COPY . .
RUN if [ -n "${QLINKS_EXTRAS}" ]; then \
        poetry install --only main --extras "${QLINKS_EXTRAS}" --no-ansi; \
    else \
        poetry install --only main --no-ansi; \
    fi && \
    python scripts/verify_optional_environment.py --extras "${QLINKS_EXTRAS}"

# PyCharm can use /usr/local/bin/python directly as the Docker interpreter.
CMD ["python"]
