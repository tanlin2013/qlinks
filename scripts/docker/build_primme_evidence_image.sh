#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

IMAGE_NAME="${QLINKS_PRIMME_DOCKER_IMAGE:-tanlin2013/qlinks:notebook-primme}"
PYTHON_VERSION="${QLINKS_PRIMME_PYTHON_VERSION:-3.13}"
PRIMME_VERSION="${QLINKS_PRIMME_VERSION:-3.2.3}"
PLATFORM="${QLINKS_DOCKER_PLATFORM:-}"

args=(
    docker build
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}"
    --build-arg "QLINKS_EXTRAS=notebook primme"
    --build-arg "PRIMME_VERSION=${PRIMME_VERSION}"
    --tag "${IMAGE_NAME}"
)
if [[ -n "${PLATFORM}" ]]; then
    args+=(--platform "${PLATFORM}")
fi
args+=("${REPO_ROOT}")

printf 'Building PRIMME evidence image:\n  '
printf '%q ' "${args[@]}"
printf '\n'
"${args[@]}"

printf '\nBuilt %s with Python %s and PRIMME %s.\n' \
    "${IMAGE_NAME}" "${PYTHON_VERSION}" "${PRIMME_VERSION}"
printf 'Use it with:\n  QLINKS_DOCKER_IMAGE=%q scripts/docker/docker_run_evidence_job.sh qdm ...\n' \
    "${IMAGE_NAME}"
