ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION} AS python

LABEL maintainer="TaoLin tanlin2013@gmail.com"

ARG WORKDIR=/home
ARG POETRY_VERSION=2.3.4

ENV PYTHONPATH="${WORKDIR}" \
    PYTHONUNBUFFERED=true \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR ${WORKDIR}

FROM python AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        libblas-dev \
        liblapack-dev

RUN python -m pip install --upgrade pip wheel setuptools && \
    python -m pip install "poetry==${POETRY_VERSION}"

COPY pyproject.toml poetry.lock ./
RUN poetry install -vvv --without dev --all-extras --no-root

COPY . ${WORKDIR}
RUN poetry install -vvv --without dev --all-extras

RUN apt-get -y clean && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/bin/bash"]
