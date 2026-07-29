#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

IMAGE_NAME="${QLINKS_DOCKER_IMAGE:-tanlin2013/qlinks:notebook}"
JOB_NAME="${1:-}"
if [[ -z "${JOB_NAME}" ]]; then
    cat >&2 <<'EOF'
Usage: scripts/docker_run_evidence_job.sh spin1|qdm [job-script-args...]

Examples:
  scripts/docker_run_evidence_job.sh spin1 --profile known
  scripts/docker_run_evidence_job.sh qdm --profile production --figure-formats pdf,svg
  QLINKS_DOCKER_IMAGE=tanlin2013/qlinks:tn-notebook scripts/docker_run_evidence_job.sh spin1 --profile production

The job runs in a detached container. Use the printed docker logs command to follow it.
EOF
    exit 2
fi
shift || true

case "${JOB_NAME}" in
    spin1|spin1_xy)
        JOB_SLUG="spin1_xy_draft_evidence"
        JOB_SCRIPT="experimental/jobs/run_spin1_xy_draft_evidence.py"
        ;;
    qdm|square_qdm)
        JOB_SLUG="square_qdm_draft_evidence"
        JOB_SCRIPT="experimental/jobs/run_square_qdm_draft_evidence.py"
        ;;
    *)
        echo "Unknown evidence job: ${JOB_NAME}" >&2
        echo "Expected one of: spin1, qdm" >&2
        exit 2
        ;;
esac

RUN_ID="${QLINKS_EVIDENCE_RUN_ID:-${JOB_SLUG}_$(date -u +%Y%m%dT%H%M%SZ)}"
CONTAINER_NAME="${QLINKS_CONTAINER_NAME:-qlinks-${RUN_ID//_/-}}"
CONTAINER_REPO_DIR="/workspace/qlinks"
CONTAINER_NOTEBOOK_DIR="${CONTAINER_REPO_DIR}/experimental/notebooks"
CONTAINER_DATA_DIR="${CONTAINER_REPO_DIR}/experimental/data"
HOST_DATA_DIR="${QLINKS_DATA_DIR:-${REPO_ROOT}/experimental/data}"
HOST_OUTPUT_DIR="${QLINKS_OUTPUT_DIR:-${REPO_ROOT}/output}"
CONTAINER_OUTPUT_DIR="/workspace/output"

mkdir -p "${HOST_DATA_DIR}" "${HOST_OUTPUT_DIR}"

docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart no \
    --env PYTHONUNBUFFERED=1 \
    --env MPLBACKEND=Agg \
    --env PYTHONPATH="${CONTAINER_REPO_DIR}:${CONTAINER_NOTEBOOK_DIR}" \
    --volume "${REPO_ROOT}:${CONTAINER_REPO_DIR}" \
    --volume "${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}" \
    --volume "${HOST_OUTPUT_DIR}:${CONTAINER_OUTPUT_DIR}" \
    --workdir "${CONTAINER_REPO_DIR}" \
    "${IMAGE_NAME}" \
    python "${JOB_SCRIPT}" --run-id "${RUN_ID}" "$@"

cat <<EOF
Started ${JOB_SLUG} as detached container ${CONTAINER_NAME}.
Image: ${IMAGE_NAME}
Repo mount: ${REPO_ROOT} -> ${CONTAINER_REPO_DIR}
Data mount: ${HOST_DATA_DIR} -> ${CONTAINER_DATA_DIR}
Output mount: ${HOST_OUTPUT_DIR} -> ${CONTAINER_OUTPUT_DIR}

Follow logs:
  docker logs -f ${CONTAINER_NAME}

Check status:
  docker ps -a --filter name=${CONTAINER_NAME}

Expected data directory on the host:
  ${HOST_DATA_DIR}/evidence_jobs/${RUN_ID}

Remove the stopped container after inspection:
  docker rm ${CONTAINER_NAME}
EOF
