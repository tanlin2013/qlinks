#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${QLINKS_DOCKER_IMAGE:-tanlin2013/qlinks:notebook}"
HOST_PORT="${QLINKS_JUPYTER_PORT:-8888}"
NOTEBOOK_DIR="${QLINKS_NOTEBOOK_DIR:-${PWD}/notebooks}"
JUPYTER_TOKEN_VALUE="${JUPYTER_TOKEN:-}"

mkdir -p "${NOTEBOOK_DIR}"
if [[ -z "${JUPYTER_TOKEN_VALUE}" ]]; then
    if command -v openssl >/dev/null 2>&1; then
        JUPYTER_TOKEN_VALUE="$(openssl rand -hex 32)"
    else
        JUPYTER_TOKEN_VALUE="$(python -c 'import secrets; print(secrets.token_hex(32))')"
    fi
fi

cat <<EOF
Starting ${IMAGE_NAME} on remote-local port ${HOST_PORT}.
Jupyter token: ${JUPYTER_TOKEN_VALUE}

Bind this port through SSH from your local machine, for example:
  ssh -N -L ${HOST_PORT}:127.0.0.1:${HOST_PORT} user@remote-host

Then open:
  http://127.0.0.1:${HOST_PORT}/lab?token=${JUPYTER_TOKEN_VALUE}
EOF

exec docker run \
    --rm \
    --interactive \
    --tty \
    --publish "127.0.0.1:${HOST_PORT}:8888" \
    --env "JUPYTER_TOKEN=${JUPYTER_TOKEN_VALUE}" \
    --volume "$(pwd):/workspace/qlinks" \
    --volume "${NOTEBOOK_DIR}:/workspace/notebooks" \
    --workdir /workspace/qlinks \
    "${IMAGE_NAME}" \
    sh -lc 'python -m jupyterlab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --ServerApp.root_dir=/workspace \
        --ServerApp.token="${JUPYTER_TOKEN}"'
