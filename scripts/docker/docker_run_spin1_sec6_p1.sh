#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

IMAGE_NAME="${QLINKS_DOCKER_IMAGE:-tanlin2013/qlinks:notebook}"
RUN_TIMESTAMP="${QLINKS_EVIDENCE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
P0_RUN_ID="${QLINKS_SEC6_P0_RUN_ID:-spin1_sec6_integration_20260825T073925Z}"
RUN_LABEL="${QLINKS_EVIDENCE_RUN_ID:-spin1_sec6_p1_claim_upgrades}"
if [[ "${RUN_LABEL}" =~ [0-9]{8}T[0-9]{6}Z$ ]]; then
    RUN_ID="${RUN_LABEL}"
else
    RUN_ID="${RUN_LABEL}_${RUN_TIMESTAMP}"
fi

CONTAINER_REPO_DIR="/workspace/qlinks"
CONTAINER_DATA_DIR="${CONTAINER_REPO_DIR}/experimental/data"
P0_DATA_DIR="${CONTAINER_DATA_DIR}/evidence_jobs/${P0_RUN_ID}"
OUTPUT_DATA_DIR="${CONTAINER_DATA_DIR}/evidence_jobs/${RUN_ID}"
DEFAULT_CACHE_ROOT="${CONTAINER_DATA_DIR}/evidence_cache/spin1"

HOST_DATA_DIR="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve(strict=False))' "${QLINKS_DATA_DIR:-${REPO_ROOT}/experimental/data}")"
THREADS="${QLINKS_NUM_THREADS:-1}"
MEMORY_LIMIT="${QLINKS_DOCKER_MEMORY_LIMIT:-}"
CPUS_LIMIT="${QLINKS_DOCKER_CPUS:-}"
SHM_SIZE="${QLINKS_DOCKER_SHM_SIZE:-16g}"
DRY_RUN="${QLINKS_DOCKER_DRY_RUN:-0}"
STAGE="status"
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
    jacobian-l8)
        JOB_COMMAND=(
            python experimental/jobs/spin1_sec6_p1_jacobian_l8.py
            --output-dir "${OUTPUT_DATA_DIR}"
        )
        ;;
    kappa-refinement-status)
        JOB_COMMAND=(
            python experimental/jobs/spin1_sec6_p1_kappa_refinement.py
            --p0-data-dir "${P0_DATA_DIR}"
            --cache-root "${DEFAULT_CACHE_ROOT}"
            --output-dir "${OUTPUT_DATA_DIR}"
        )
        ;;
    kappa-refinement)
        JOB_COMMAND=(
            python experimental/jobs/spin1_sec6_p1_kappa_refinement.py
            --p0-data-dir "${P0_DATA_DIR}"
            --cache-root "${DEFAULT_CACHE_ROOT}"
            --output-dir "${OUTPUT_DATA_DIR}"
            --compute-missing
        )
        SOLVE_POLICY="explicit full-dense L=8,10,12 only at kappa/J=0.075,0.125,0.175; sparse/L14 forbidden"
        ;;
    three-site-status)
        JOB_COMMAND=(
            python experimental/jobs/spin1_sec6_p1_three_site_concentration.py
            --checkpoint-root "${DEFAULT_CACHE_ROOT}"
            --checkpoint-root "${P0_DATA_DIR}"
            --cache-root "${DEFAULT_CACHE_ROOT}"
            --output-dir "${OUTPUT_DATA_DIR}"
        )
        ;;
    three-site)
        JOB_COMMAND=(
            python experimental/jobs/spin1_sec6_p1_three_site_concentration.py
            --checkpoint-root "${DEFAULT_CACHE_ROOT}"
            --checkpoint-root "${P0_DATA_DIR}"
            --cache-root "${DEFAULT_CACHE_ROOT}"
            --output-dir "${OUTPUT_DATA_DIR}"
            --compute-missing
        )
        SOLVE_POLICY="cache-only spectra: build 141 three-site operators and covariance at L=8,10,12; no eigensolver"
        ;;
    status)
        JOB_COMMAND=(
            bash -lc
            "set -e; python experimental/jobs/spin1_sec6_p1_kappa_refinement.py --p0-data-dir '${P0_DATA_DIR}' --cache-root '${DEFAULT_CACHE_ROOT}' --output-dir '${OUTPUT_DATA_DIR}'; python experimental/jobs/spin1_sec6_p1_three_site_concentration.py --checkpoint-root '${DEFAULT_CACHE_ROOT}' --checkpoint-root '${P0_DATA_DIR}' --cache-root '${DEFAULT_CACHE_ROOT}' --output-dir '${OUTPUT_DATA_DIR}'"
        )
        ;;
    *)
        echo "unknown stage: ${STAGE}" >&2
        echo "expected one of: status, jacobian-l8, kappa-refinement-status, kappa-refinement, three-site-status, three-site" >&2
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
    --label "qlinks.evidence.job=spin1_xy_sec6_p1_claim_upgrades"
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
Job: spin1_sec6_p1_claim_upgrades
Run id: ${RUN_ID}
Stage: ${STAGE}
Frozen P0 evidence: ${P0_RUN_ID}
Container: ${CONTAINER_NAME}
Image: ${IMAGE_NAME}
Output evidence: ${HOST_DATA_DIR}/evidence_jobs/${RUN_ID}
Stable cache: ${HOST_DATA_DIR}/evidence_cache/spin1
Thread limit: ${THREADS}
Memory limit: ${MEMORY_LIMIT:-unlimited}
CPU limit: ${CPUS_LIMIT:-unlimited}
Shared memory: ${SHM_SIZE}
Solve policy: ${SOLVE_POLICY}

Recommended P1 sequence using one explicit run id:
  export QLINKS_EVIDENCE_RUN_ID=${RUN_ID}
  scripts/docker/docker_run_spin1_sec6_p1.sh --stage jacobian-l8
  scripts/docker/docker_run_spin1_sec6_p1.sh --stage kappa-refinement-status
  scripts/docker/docker_run_spin1_sec6_p1.sh --stage kappa-refinement
  scripts/docker/docker_run_spin1_sec6_p1.sh --stage three-site-status
  scripts/docker/docker_run_spin1_sec6_p1.sh --stage three-site

The frozen P0 CSVs are inputs only. P1 outputs are written to a separate run and
new cache namespaces. No P1 stage may launch L=14 or sparse shift-invert work.
EOF

if [[ "${DRY_RUN}" != "1" ]]; then
    cat <<EOF

Follow logs:
  docker logs -f ${CONTAINER_NAME}

Check status:
  docker ps -a --filter name=${CONTAINER_NAME}
EOF
fi
