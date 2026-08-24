# syntax=docker/dockerfile:1.26

# The official Python image is multi-architecture. Docker selects the image
# matching --platform (or the Docker host by default), so the same Dockerfile
# works for linux/amd64 and linux/arm64.
ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

LABEL maintainer="TaoLin tanlin2013@gmail.com"

ARG UV_VERSION=0.12.0
ARG QLINKS_EXTRAS=
ARG PRIMME_VERSION=3.2.3
ARG TARGETPLATFORM
ARG TARGETARCH

# Keep the default Docker image lightweight and independent of optional extras.
# Tensor-network extras are supported on Python < 3.14 because quimb/autograd/numba
# are currently constrained that way in pyproject.toml. Build a TN image explicitly with
# --build-arg PYTHON_VERSION=3.13 --build-arg QLINKS_EXTRAS=tn.
#
# ``primme`` is a Docker-only scientific runtime feature rather than a qlinks
# package extra. It is compiled into the isolated evidence image after uv has
# synchronized the locked qlinks environment. Keep it on Python 3.13 until the
# source build is separately certified on Python 3.14.
#
# ``notebook`` is a Docker feature backed by qlinks' non-default uv dependency group;
# other feature names map to project extras. The current locked nbconvert stack imports
# lxml.html.clean at runtime. That dependency is already locked through the docs group,
# so notebook images include docs as well until the dependency can be split into a
# dedicated notebook-runtime group during a future lock refresh.
#
# The project source is bind-mounted over /workspace/qlinks by the evidence-job
# launcher. Keep the uv environment outside that mount so the image-installed
# dependencies remain visible when local source replaces the image source tree.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/qlinks-venv \
    PATH="/opt/qlinks-venv/bin:${PATH}"

WORKDIR /workspace/qlinks

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libblas-dev \
        liblapack-dev \
        libcairo2-dev \
        libxml2-dev \
        libxslt1-dev \
        pkg-config \
        zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir "uv==${UV_VERSION}"

# uv.lock is cross-platform; uv evaluates markers and chooses distributions for
# the active Python/platform when synchronizing the image environment.
COPY pyproject.toml uv.lock README.md ./
RUN echo "Building qlinks for ${TARGETPLATFORM:-default} (${TARGETARCH:-default})" && \
    echo "Optional Docker features/extras: ${QLINKS_EXTRAS:-none}" && \
    uv lock --check && \
    set -- --no-default-groups && \
    for feature in ${QLINKS_EXTRAS}; do \
        if [ "${feature}" = "notebook" ]; then \
            set -- "$@" --group notebook --group docs; \
        elif [ "${feature}" = "primme" ]; then \
            :; \
        else \
            set -- "$@" --extra "${feature}"; \
        fi; \
    done && \
    uv sync --locked --no-install-project "$@"

COPY . .
RUN set -- --no-default-groups && \
    for feature in ${QLINKS_EXTRAS}; do \
        if [ "${feature}" = "notebook" ]; then \
            set -- "$@" --group notebook --group docs; \
        elif [ "${feature}" = "primme" ]; then \
            :; \
        else \
            set -- "$@" --extra "${feature}"; \
        fi; \
    done && \
    uv sync --locked "$@" && \
    if printf ' %s ' "${QLINKS_EXTRAS}" | grep -q ' primme '; then \
        python -c 'import sys; raise SystemExit(0 if sys.version_info < (3, 14) else "primme evidence images currently require Python <3.14; build with --build-arg PYTHON_VERSION=3.13")'; \
        uv pip install --python /opt/qlinks-venv/bin/python "primme==${PRIMME_VERSION}"; \
    fi && \
    python scripts/docker/verify_optional_environment.py --extras "${QLINKS_EXTRAS}"

# PyCharm can use /opt/qlinks-venv/bin/python directly.
CMD ["python"]
