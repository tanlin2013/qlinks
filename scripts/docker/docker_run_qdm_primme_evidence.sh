#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

IMAGE_NAME="${QLINKS_DOCKER_IMAGE:-tanlin2013/qlinks:notebook-primme}"
PULL_POLICY="${QLINKS_DOCKER_PULL_POLICY:-never}"
RUN_LABEL="${QLINKS_EVIDENCE_RUN_ID:-qdm_checkerboard_primme}"
RUN_TIMESTAMP="${QLINKS_EVIDENCE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ "${RUN_LABEL}" =~ [0-9]{8}T[0-9]{6}Z$ ]]; then
    RUN_ID="${RUN_LABEL}"
else
    RUN_ID="${RUN_LABEL}_${RUN_TIMESTAMP}"
fi
CONTAINER_NAME="${QLINKS_CONTAINER_NAME:-qlinks-${RUN_ID//_/-}}"

CONTAINER_REPO_DIR="/workspace/qlinks"
CONTAINER_DATA_DIR="${CONTAINER_REPO_DIR}/experimental/data"
CONTAINER_OUTPUT_DIR="/workspace/output"
HOST_DATA_DIR="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve(strict=False))' "${QLINKS_DATA_DIR:-${REPO_ROOT}/experimental/data}")"
HOST_OUTPUT_DIR="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve(strict=False))' "${QLINKS_OUTPUT_DIR:-${REPO_ROOT}/output}")"
THREADS="${QLINKS_NUM_THREADS:-16}"
MEMORY_LIMIT="${QLINKS_DOCKER_MEMORY_LIMIT:-400g}"
CPUS_LIMIT="${QLINKS_DOCKER_CPUS:-}"
SHM_SIZE="${QLINKS_DOCKER_SHM_SIZE:-16g}"
DRY_RUN="${QLINKS_DOCKER_DRY_RUN:-0}"

mkdir -p "${HOST_DATA_DIR}" "${HOST_OUTPUT_DIR}" "${HOST_DATA_DIR}/evidence_cache"

DOCKER_LIMIT_ARGS=(--memory "${MEMORY_LIMIT}" --shm-size "${SHM_SIZE}")
[[ -z "${CPUS_LIMIT}" ]] || DOCKER_LIMIT_ARGS+=(--cpus "${CPUS_LIMIT}")

DOCKER_COMMAND=(
    docker run -d
    --pull "${PULL_POLICY}"
    --init
    --name "${CONTAINER_NAME}"
    --label "qlinks.evidence.run_id=${RUN_ID}"
    --label "qlinks.evidence.timestamp=${RUN_TIMESTAMP}"
    --label "qlinks.evidence.job=square_qdm_draft_evidence"
    --label "qlinks.evidence.solver=primme"
    --restart no
    "${DOCKER_LIMIT_ARGS[@]}"
    --env PYTHONUNBUFFERED=1
    --env QLINKS_EVIDENCE_RUN_ID="${RUN_ID}"
    --env QLINKS_EVIDENCE_RUN_TIMESTAMP="${RUN_TIMESTAMP}"
    --env QLINKS_DOCKER_MEMORY_LIMIT="${MEMORY_LIMIT}"
    --env MPLBACKEND=Agg
    --env OPENBLAS_NUM_THREADS="${THREADS}"
    --env OMP_NUM_THREADS="${THREADS}"
    --env MKL_NUM_THREADS="${THREADS}"
    --env NUMEXPR_NUM_THREADS="${THREADS}"
    --env VECLIB_MAXIMUM_THREADS="${THREADS}"
    --env PYTHONPATH="${CONTAINER_REPO_DIR}:${CONTAINER_REPO_DIR}/experimental/notebooks:${CONTAINER_REPO_DIR}/experimental/jobs"
    --volume "${REPO_ROOT}:${CONTAINER_REPO_DIR}"
    --volume "${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}"
    --volume "${HOST_OUTPUT_DIR}:${CONTAINER_OUTPUT_DIR}"
    --workdir "${CONTAINER_REPO_DIR}"
    "${IMAGE_NAME}"
    python experimental/jobs/run_square_qdm_draft_evidence.py
    --run-id "${RUN_ID}"
    --evidence-cache-root "${CONTAINER_DATA_DIR}/evidence_cache"
    --large-strip-folded-backend primme
    "$@"
)

printf 'Resolved PRIMME QDM Docker command:\n  '
printf '%q ' "${DOCKER_COMMAND[@]}"
printf '\n\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run requested; no container was started."
else
    "${DOCKER_COMMAND[@]}"
fi

cat <<EOF
Job: square_qdm_draft_evidence (PRIMME folded backend)
Run id: ${RUN_ID}
Container: ${CONTAINER_NAME}
Image: ${IMAGE_NAME}
Pull policy: ${PULL_POLICY}
Data mount: ${HOST_DATA_DIR} -> ${CONTAINER_DATA_DIR}
Stable cache: ${HOST_DATA_DIR}/evidence_cache
Thread limit: ${THREADS}
Memory limit: ${MEMORY_LIMIT}
Shared memory: ${SHM_SIZE}

The default pull policy is 'never' so a locally built PRIMME image is not
replaced or rejected by an unconditional registry pull. Set
QLINKS_DOCKER_PULL_POLICY=always only when using a published image tag.
EOF

if [[ "${DRY_RUN}" != "1" ]]; then
    cat <<EOF

Follow logs:
  docker logs -f ${CONTAINER_NAME}

Check status:
  docker ps -a --filter name=${CONTAINER_NAME}
EOF
fi
