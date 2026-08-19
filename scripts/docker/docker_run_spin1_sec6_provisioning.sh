#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

IMAGE_NAME="${QLINKS_DOCKER_IMAGE:-tanlin2013/qlinks:notebook}"
RUN_LABEL="${QLINKS_EVIDENCE_RUN_ID:-spin1_sec6_provisioning}"
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
THREADS="${QLINKS_NUM_THREADS:-1}"
MEMORY_LIMIT="${QLINKS_DOCKER_MEMORY_LIMIT:-}"
CPUS_LIMIT="${QLINKS_DOCKER_CPUS:-}"
SHM_SIZE="${QLINKS_DOCKER_SHM_SIZE:-16g}"
DRY_RUN="${QLINKS_DOCKER_DRY_RUN:-0}"

mkdir -p "${HOST_DATA_DIR}" "${HOST_OUTPUT_DIR}"

# This wrapper intentionally pins the two authoritative draft handoff roots.
# Override them through the job CLI only when testing another archived evidence set.
DEFAULT_BASELINE="${CONTAINER_DATA_DIR}/evidence_jobs/spin1_production_20260806T074051Z"
DEFAULT_SPARSE_ADDENDUM="${CONTAINER_DATA_DIR}/evidence_jobs/spin1_production_20260810T082123Z"
DEFAULT_CHECKPOINT_SOURCE="${DEFAULT_SPARSE_ADDENDUM}/checkpoints"

STAGE="compute"
HAS_STAGE=0
HAS_BASELINE=0
HAS_SPARSE_ADDENDUM=0
HAS_CHECKPOINT_SOURCE=0
FORWARDED_ARGS=()
while (($#)); do
    case "$1" in
        --stage)
            [[ $# -ge 2 ]] || { echo "--stage requires a value" >&2; exit 2; }
            STAGE="$2"
            HAS_STAGE=1
            FORWARDED_ARGS+=("$1" "$2")
            shift 2
            ;;
        --stage=*)
            STAGE="${1#*=}"
            HAS_STAGE=1
            FORWARDED_ARGS+=("$1")
            shift
            ;;
        --baseline-data-dir|--baseline-data-dir=*)
            HAS_BASELINE=1
            if [[ "$1" == *=* ]]; then
                FORWARDED_ARGS+=("$1")
                shift
            else
                [[ $# -ge 2 ]] || { echo "--baseline-data-dir requires a value" >&2; exit 2; }
                FORWARDED_ARGS+=("$1" "$2")
                shift 2
            fi
            ;;
        --sparse-convergence-data-dir|--sparse-convergence-data-dir=*)
            HAS_SPARSE_ADDENDUM=1
            if [[ "$1" == *=* ]]; then
                FORWARDED_ARGS+=("$1")
                shift
            else
                [[ $# -ge 2 ]] || { echo "--sparse-convergence-data-dir requires a value" >&2; exit 2; }
                FORWARDED_ARGS+=("$1" "$2")
                shift 2
            fi
            ;;
        --checkpoint-source-dir|--checkpoint-source-dir=*)
            HAS_CHECKPOINT_SOURCE=1
            if [[ "$1" == *=* ]]; then
                FORWARDED_ARGS+=("$1")
                shift
            else
                [[ $# -ge 2 ]] || { echo "--checkpoint-source-dir requires a value" >&2; exit 2; }
                FORWARDED_ARGS+=("$1" "$2")
                shift 2
            fi
            ;;
        *)
            FORWARDED_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "${HAS_STAGE}" == "0" ]]; then
    FORWARDED_ARGS+=("--stage" "${STAGE}")
fi
if [[ "${STAGE}" != "render" ]]; then
    [[ "${HAS_BASELINE}" == "1" ]] || FORWARDED_ARGS+=("--baseline-data-dir" "${DEFAULT_BASELINE}")
    [[ "${HAS_SPARSE_ADDENDUM}" == "1" ]] || FORWARDED_ARGS+=(
        "--sparse-convergence-data-dir" "${DEFAULT_SPARSE_ADDENDUM}"
    )
    if [[ "${HAS_CHECKPOINT_SOURCE}" == "0" ]]; then
        FORWARDED_ARGS+=("--checkpoint-source-dir" "${DEFAULT_CHECKPOINT_SOURCE}")
    fi
fi

DOCKER_LIMIT_ARGS=(--shm-size "${SHM_SIZE}")
[[ -z "${MEMORY_LIMIT}" ]] || DOCKER_LIMIT_ARGS+=(--memory "${MEMORY_LIMIT}")
[[ -z "${CPUS_LIMIT}" ]] || DOCKER_LIMIT_ARGS+=(--cpus "${CPUS_LIMIT}")

DOCKER_COMMAND=(
    docker run -d
    --init
    --name "${CONTAINER_NAME}"
    --label "qlinks.evidence.run_id=${RUN_ID}"
    --label "qlinks.evidence.timestamp=${RUN_TIMESTAMP}"
    --label "qlinks.evidence.job=spin1_xy_sec6_provisioning"
    --restart no
    "${DOCKER_LIMIT_ARGS[@]}"
    --env PYTHONUNBUFFERED=1
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
    python experimental/jobs/run_spin1_xy_sec6_provisioning.py
    --profile production
    --run-id "${RUN_ID}"
    "${FORWARDED_ARGS[@]}"
)

printf 'Resolved Docker command:\n  '
printf '%q ' "${DOCKER_COMMAND[@]}"
printf '\n\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run requested; no container was started."
else
    "${DOCKER_COMMAND[@]}"
fi

cat <<EOF
Job: spin1_xy_sec6_provisioning
Run id: ${RUN_ID}
Stage: ${STAGE}
Container: ${CONTAINER_NAME}
Image: ${IMAGE_NAME}
Data mount: ${HOST_DATA_DIR} -> ${CONTAINER_DATA_DIR}
Thread limit: ${THREADS}
Memory limit: ${MEMORY_LIMIT:-unlimited}
CPU limit: ${CPUS_LIMIT:-unlimited}
Shared memory: ${SHM_SIZE}

P0.0--P0.2 production command (representative L=14 + bridges; no 10000-eigenpair rerun):
  QLINKS_NUM_THREADS=16 QLINKS_DOCKER_MEMORY_LIMIT=400g \
    scripts/docker/docker_run_spin1_sec6_provisioning.sh --stage compute --timeout -1

P0.3 follow-up after representative concentration is secure:
  QLINKS_NUM_THREADS=16 QLINKS_DOCKER_MEMORY_LIMIT=400g \
    scripts/docker/docker_run_spin1_sec6_provisioning.sh --stage compute \
      --data-dir experimental/data/evidence_jobs/${RUN_ID} \
      --run-family-l14 --timeout -1
EOF

if [[ "${DRY_RUN}" != "1" ]]; then
    cat <<EOF

Follow logs:
  docker logs -f ${CONTAINER_NAME}

Check status:
  docker ps -a --filter name=${CONTAINER_NAME}
EOF
fi
