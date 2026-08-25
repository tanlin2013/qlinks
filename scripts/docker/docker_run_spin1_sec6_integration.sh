#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

IMAGE_NAME="${QLINKS_DOCKER_IMAGE:-tanlin2013/qlinks:notebook}"
RUN_TIMESTAMP="${QLINKS_EVIDENCE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
SOURCE_RUN_ID="${QLINKS_SEC6_SOURCE_RUN_ID:-spin1_sec6_provisioning_20260820T052954Z}"
RUN_LABEL="${QLINKS_EVIDENCE_RUN_ID:-spin1_sec6_integration}"
if [[ "${RUN_LABEL}" =~ [0-9]{8}T[0-9]{6}Z$ ]]; then
    RUN_ID="${RUN_LABEL}"
else
    RUN_ID="${RUN_LABEL}_${RUN_TIMESTAMP}"
fi

CONTAINER_REPO_DIR="/workspace/qlinks"
CONTAINER_DATA_DIR="${CONTAINER_REPO_DIR}/experimental/data"
SOURCE_DATA_DIR="${CONTAINER_DATA_DIR}/evidence_jobs/${SOURCE_RUN_ID}"
OUTPUT_DATA_DIR="${CONTAINER_DATA_DIR}/evidence_jobs/${RUN_ID}"
DEFAULT_CACHE_ROOT="${CONTAINER_DATA_DIR}/evidence_cache/spin1"

HOST_DATA_DIR="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve(strict=False))' "${QLINKS_DATA_DIR:-${REPO_ROOT}/experimental/data}")"
THREADS="${QLINKS_NUM_THREADS:-1}"
MEMORY_LIMIT="${QLINKS_DOCKER_MEMORY_LIMIT:-}"
CPUS_LIMIT="${QLINKS_DOCKER_CPUS:-}"
SHM_SIZE="${QLINKS_DOCKER_SHM_SIZE:-16g}"
DRY_RUN="${QLINKS_DOCKER_DRY_RUN:-0}"
USE_TEX="${QLINKS_SEC6_USE_TEX:-1}"
STAGE="audit"
SOLVE_POLICY="disabled; no eigensolver entry point is invoked"

while (($#)); do
    case "$1" in
        --stage)
            [[ $# -ge 2 ]] || { echo "--stage requires a value" >&2; exit 2; }
            STAGE="$2"
            shift 2
            ;;
        --stage=*)
            STAGE="${1#*=}"
            shift
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

CONTAINER_NAME="${QLINKS_CONTAINER_NAME:-qlinks-${RUN_ID//_/-}-${STAGE}-${RUN_TIMESTAMP}-$$}"

case "${STAGE}" in
    audit)
        JOB_COMMAND=(
            python experimental/jobs/spin1_sec6_integration.py
            --source-data-dir "${SOURCE_DATA_DIR}"
            --output-dir "${OUTPUT_DATA_DIR}"
        )
        ;;
    seed-dense-cache)
        JOB_COMMAND=(
            python experimental/jobs/spin1_sec6_seed_dense_cache.py
            --cache-root "${DEFAULT_CACHE_ROOT}"
            --output-dir "${OUTPUT_DATA_DIR}"
        )
        SOLVE_POLICY="dense-only L=8,10,12 at kappa/J=0.1; sparse/L=14 solves are forbidden"
        ;;
    common-windows)
        JOB_COMMAND=(
            python experimental/jobs/spin1_sec6_common_windows_certified.py
            --source-data-dir "${SOURCE_DATA_DIR}"
            --checkpoint-root "${DEFAULT_CACHE_ROOT}"
            --checkpoint-root "${SOURCE_DATA_DIR}"
            --existing-data-dir "${OUTPUT_DATA_DIR}"
            --output-dir "${OUTPUT_DATA_DIR}"
        )
        ;;
    render-preview|render-final)
        JOB_COMMAND=(
            python experimental/jobs/render_spin1_xy_sec6_integration_figures.py
            --data-dir "${OUTPUT_DATA_DIR}"
        )
        [[ "${USE_TEX}" == "0" ]] || JOB_COMMAND+=(--use-tex)
        [[ "${STAGE}" != "render-preview" ]] || JOB_COMMAND+=(--allow-incomplete)
        ;;
    *)
        echo "unknown stage: ${STAGE}" >&2
        echo "expected one of: audit, seed-dense-cache, common-windows, render-preview, render-final" >&2
        exit 2
        ;;
esac

mkdir -p "${HOST_DATA_DIR}/evidence_jobs/${RUN_ID}"

DOCKER_LIMIT_ARGS=(--shm-size "${SHM_SIZE}")
[[ -z "${MEMORY_LIMIT}" ]] || DOCKER_LIMIT_ARGS+=(--memory "${MEMORY_LIMIT}")
[[ -z "${CPUS_LIMIT}" ]] || DOCKER_LIMIT_ARGS+=(--cpus "${CPUS_LIMIT}")

DOCKER_COMMAND=(
    docker run -d
    --pull always
    --init
    --name "${CONTAINER_NAME}"
    --label "qlinks.evidence.run_id=${RUN_ID}"
    --label "qlinks.evidence.timestamp=${RUN_TIMESTAMP}"
    --label "qlinks.evidence.job=spin1_xy_sec6_integration"
    --label "qlinks.evidence.stage=${STAGE}"
    --restart no
    "${DOCKER_LIMIT_ARGS[@]}"
    --env PYTHONUNBUFFERED=1
    --env MPLBACKEND=Agg
    --env OPENBLAS_NUM_THREADS="${THREADS}"
    --env OMP_NUM_THREADS="${THREADS}"
    --env MKL_NUM_THREADS="${THREADS}"
    --env NUMEXPR_NUM_THREADS="${THREADS}"
    --env VECLIB_MAXIMUM_THREADS="${THREADS}"
    --env PYTHONPATH="${CONTAINER_REPO_DIR}:${CONTAINER_REPO_DIR}/experimental/notebooks:${CONTAINER_REPO_DIR}/experimental/jobs"
    --volume "${REPO_ROOT}:${CONTAINER_REPO_DIR}:ro"
    --volume "${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}"
    --workdir "${CONTAINER_REPO_DIR}"
    "${IMAGE_NAME}"
    "${JOB_COMMAND[@]}"
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
Job: spin1_sec6_integration
Run id: ${RUN_ID}
Stage: ${STAGE}
Source evidence: ${SOURCE_RUN_ID}
Container: ${CONTAINER_NAME}
Image: ${IMAGE_NAME}
Pull policy: always
Data mount: ${HOST_DATA_DIR} -> ${CONTAINER_DATA_DIR}
Output evidence: ${HOST_DATA_DIR}/evidence_jobs/${RUN_ID}
Stable cache: ${HOST_DATA_DIR}/evidence_cache/spin1
Thread limit: ${THREADS}
Memory limit: ${MEMORY_LIMIT:-unlimited}
CPU limit: ${CPUS_LIMIT:-unlimited}
Shared memory: ${SHM_SIZE}
Solve policy: ${SOLVE_POLICY}

Recommended sequence, reusing the same run id:
  QLINKS_EVIDENCE_RUN_ID=${RUN_ID} \
    scripts/docker/docker_run_spin1_sec6_integration.sh --stage audit

If the audit/common-window reducer reports missing L=8,10,12 spectra, seed only those dense caches:
  QLINKS_EVIDENCE_RUN_ID=${RUN_ID} \
    scripts/docker/docker_run_spin1_sec6_integration.sh --stage seed-dense-cache

Then rerun the cache-only reducer:
  QLINKS_EVIDENCE_RUN_ID=${RUN_ID} \
    scripts/docker/docker_run_spin1_sec6_integration.sh --stage common-windows

  QLINKS_EVIDENCE_RUN_ID=${RUN_ID} \
    scripts/docker/docker_run_spin1_sec6_integration.sh --stage render-preview

Use --stage render-final only after the integration audit confirms all strict Fig. 6 inputs are present.
EOF

if [[ "${DRY_RUN}" != "1" ]]; then
    cat <<EOF

Follow logs:
  docker logs -f ${CONTAINER_NAME}

Check status:
  docker ps -a --filter name=${CONTAINER_NAME}
EOF
fi
