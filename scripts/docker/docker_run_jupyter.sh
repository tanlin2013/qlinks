#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

IMAGE_NAME="${QLINKS_DOCKER_IMAGE:-tanlin2013/qlinks:notebook}"
CONTAINER_NAME="${QLINKS_CONTAINER_NAME:-qlinks-jupyter}"
HOST_PORT="${QLINKS_HOST_PORT:-8888}"
NOTEBOOK_DIR="${QLINKS_NOTEBOOK_DIR:-${REPO_ROOT}/experimental/notebooks}"
DATA_DIR="${QLINKS_DATA_DIR:-${REPO_ROOT}/experimental/data}"
OUTPUT_DIR="${QLINKS_OUTPUT_DIR:-${REPO_ROOT}/output}"
JUPYTER_TOKEN_VALUE="${JUPYTER_TOKEN:-}"
CONTAINER_REPO_DIR="/workspace/qlinks"
CONTAINER_NOTEBOOK_DIR="${CONTAINER_REPO_DIR}/experimental/notebooks"
CONTAINER_DATA_DIR="${CONTAINER_REPO_DIR}/experimental/data"
CONTAINER_OUTPUT_DIR="/workspace/output"

mkdir -p "${NOTEBOOK_DIR}" "${DATA_DIR}" "${OUTPUT_DIR}"

if [[ -z "${JUPYTER_TOKEN_VALUE}" ]]; then
    JUPYTER_TOKEN_VALUE="$(openssl rand -hex 32)"
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart unless-stopped \
    --publish "127.0.0.1:${HOST_PORT}:8888" \
    --env JUPYTER_TOKEN="${JUPYTER_TOKEN_VALUE}" \
    --env PYTHONPATH="${CONTAINER_REPO_DIR}:${CONTAINER_NOTEBOOK_DIR}" \
    --volume "${REPO_ROOT}:${CONTAINER_REPO_DIR}" \
    --volume "${NOTEBOOK_DIR}:${CONTAINER_NOTEBOOK_DIR}" \
    --volume "${DATA_DIR}:${CONTAINER_DATA_DIR}" \
    --volume "${OUTPUT_DIR}:${CONTAINER_OUTPUT_DIR}" \
    --workdir "${CONTAINER_NOTEBOOK_DIR}" \
    "${IMAGE_NAME}" \
    sh -lc 'python -m jupyterlab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --ServerApp.root_dir="/workspace/qlinks" \
        --IdentityProvider.token="${JUPYTER_TOKEN}"'

cat <<EOF
Started ${IMAGE_NAME} as container ${CONTAINER_NAME}.
Jupyter token: ${JUPYTER_TOKEN_VALUE}

SSH tunnel from local:
  ssh -N -L ${HOST_PORT}:127.0.0.1:${HOST_PORT} user@remote-host

Open locally:
  http://127.0.0.1:${HOST_PORT}/lab/tree/experimental/notebooks?token=${JUPYTER_TOKEN_VALUE}

Mounted paths:
  repo:      ${REPO_ROOT} -> ${CONTAINER_REPO_DIR}
  notebooks: ${NOTEBOOK_DIR} -> ${CONTAINER_NOTEBOOK_DIR}
  data:      ${DATA_DIR} -> ${CONTAINER_DATA_DIR}
  output:    ${OUTPUT_DIR} -> ${CONTAINER_OUTPUT_DIR}

Notebook import paths:
  PYTHONPATH=${CONTAINER_REPO_DIR}:${CONTAINER_NOTEBOOK_DIR}
EOF
