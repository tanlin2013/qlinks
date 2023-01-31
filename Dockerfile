FROM python:3.11
MAINTAINER "TaoLin" <tanlin2013@gmail.com>

ARG WORKDIR=/home/qlinks
ENV PYTHONPATH="${PYTHONPATH}:$WORKDIR" \
    PATH="/root/.local/bin:$PATH"
WORKDIR $WORKDIR
COPY . $WORKDIR

# Install fortran, blas, lapack
RUN apt update && \
    apt-get install -y --no-install-recommends \
      gfortran libblas-dev liblapack-dev
RUN apt-get -y clean && \
    rm -rf /var/lib/apt/lists/*

# Install required python packages and the module
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    poetry config virtualenvs.create false --local && \
    poetry install --no-dev

ENTRYPOINT /bin/bash
