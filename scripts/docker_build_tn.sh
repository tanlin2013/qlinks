#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${QLINKS_DOCKER_IMAGE:-qlinks:tn}"
PYTHON_VERSION="${QLINKS_DOCKER_PYTHON:-3.14}"
PLATFORM="${QLINKS_DOCKER_PLATFORM:-}"

build_args=(
    buildx build
    --load
    --build-arg "PYTHON_VERSION=${PYTHON_VERSION}"
    --build-arg "QLINKS_EXTRAS=tn"
    --tag "${IMAGE_NAME}"
)
if [[ -n "${PLATFORM}" ]]; then
    build_args+=(--platform "${PLATFORM}")
fi
build_args+=(.)

# --load is important when the active buildx builder does not write images to
# Docker's local image store. PyCharm can only select an image that is loaded.
docker "${build_args[@]}"

echo "Built ${IMAGE_NAME}${PLATFORM:+ for ${PLATFORM}}."
echo "Python interpreter inside image: /usr/local/bin/python"
