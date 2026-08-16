#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${QLINKS_DOCKER_IMAGE:-qlinks:tn}"
if [[ $# -eq 0 ]]; then
    set -- bash
fi

exec docker run \
    --rm \
    --interactive \
    --tty \
    --volume "$(pwd):/workspace/qlinks" \
    --workdir /workspace/qlinks \
    "${IMAGE_NAME}" \
    "$@"
