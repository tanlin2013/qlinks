# syntax=docker/dockerfile:1.7

# The official Python image is multi-architecture. Docker selects the image
# matching --platform (or the Docker host by default), so the same Dockerfile
# works for linux/amd64 and linux/arm64.
ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

LABEL maintainer="TaoLin tanlin2013@gmail.com"

ARG POETRY_VERSION=2.3.4
ARG QLINKS_EXTRAS=tn
ARG TARGETPLATFORM
ARG TARGETARCH

# Never compile LLVM inside the image. Both llvmlite 0.48 and numba 0.66
# publish CPython 3.14 wheels for Linux x86_64 and aarch64.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_INSTALLER_ONLY_BINARY="llvmlite,numba"

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
    poetry check --lock && \
    poetry install \
        --only main \
        --extras "${QLINKS_EXTRAS}" \
        --no-root \
        --no-ansi

COPY . .
RUN poetry install \
        --only main \
        --extras "${QLINKS_EXTRAS}" \
        --no-ansi && \
    python scripts/verify_tn_environment.py

# PyCharm can use /usr/local/bin/python directly as the Docker interpreter.
CMD ["python"]
