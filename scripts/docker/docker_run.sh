#!/usr/bin/env bash
set -euo pipefail

# Run helpers default to published images and always refresh the selected tag
# before creating a container. Use a direct `docker run --pull never ...` when
# intentionally testing a locally built image such as qlinks:tn.
IMAGE_NAME="${QLINKS_DOCKER_IMAGE:-tanlin2013/qlinks:tn-notebook}"
if [[ $# -eq 0 ]]; then
    set -- bash
fi

exec docker run \
    --pull always \
    --rm \
    --interactive \
    --tty \
    --volume "$(pwd):/workspace/qlinks" \
    --workdir /workspace/qlinks \
    "${IMAGE_NAME}" \
    "$@"
