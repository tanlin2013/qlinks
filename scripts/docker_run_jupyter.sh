#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${QLINKS_DOCKER_IMAGE:-tanlin2013/qlinks:notebook}"
CONTAINER_NAME="${QLINKS_CONTAINER_NAME:-qlinks-jupyter}"
HOST_PORT="${QLINKS_HOST_PORT:-8888}"
NOTEBOOK_DIR="${QLINKS_NOTEBOOK_DIR:-$(pwd)/notebooks}"
OUTPUT_DIR="${QLINKS_OUTPUT_DIR:-$(pwd)/output}"
JUPYTER_TOKEN_VALUE="${JUPYTER_TOKEN:-}"

mkdir -p "${NOTEBOOK_DIR}" "${OUTPUT_DIR}"

if [[ -z "${JUPYTER_TOKEN_VALUE}" ]]; then
    JUPYTER_TOKEN_VALUE="$(openssl rand -hex 32)"
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart unless-stopped \
    --publish "127.0.0.1:${HOST_PORT}:8888" \
    --env JUPYTER_TOKEN="${JUPYTER_TOKEN_VALUE}" \
    --volume "$(pwd):/workspace/qlinks" \
    --volume "${NOTEBOOK_DIR}:/workspace/notebooks" \
    --volume "${OUTPUT_DIR}:/workspace/output" \
    --workdir /workspace/qlinks \
    "${IMAGE_NAME}" \
    sh -lc 'python -m jupyterlab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --ServerApp.root_dir=/workspace \
        --IdentityProvider.token="${JUPYTER_TOKEN}"'

cat <<EOF
Started ${IMAGE_NAME} as container ${CONTAINER_NAME}.
Jupyter token: ${JUPYTER_TOKEN_VALUE}

SSH tunnel from local:
  ssh -N -L ${HOST_PORT}:127.0.0.1:${HOST_PORT} user@remote-host

Open locally:
  http://127.0.0.1:${HOST_PORT}/lab?token=${JUPYTER_TOKEN_VALUE}

Mounted paths:
  repo:      /workspace/qlinks
  notebooks: /workspace/notebooks
  output:    /workspace/output
EOF
