ARG PYTHON_VERSION=3.14
FROM python:${PYTHON_VERSION} AS python

LABEL maintainer="TaoLin tanlin2013@gmail.com"

ARG WORKDIR=/home
ARG POETRY_VERSION=2.3.4

ENV PYTHONPATH="${PYTHONPATH}:$WORKDIR" \
    PATH="/root/.local/bin:$PATH" \
    PYTHONUNBUFFERED=true

WORKDIR $WORKDIR

FROM python AS runtime

COPY . $WORKDIR

RUN apt update && \
    apt-get install -y --no-install-recommends \
        gfortran \
        libblas-dev \
        liblapack-dev \
        pipx

RUN pipx install poetry==${POETRY_VERSION} && \
    poetry config virtualenvs.create false --local && \
    poetry run pip install --upgrade pip wheel setuptools && \
    poetry install -vvv --without dev --all-extras

RUN apt-get -y clean && \
    rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/bin/bash"]
