#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

IMAGE_NAME="${QLINKS_DOCKER_IMAGE:-tanlin2013/qlinks:notebook-primme}"
PULL_POLICY="${QLINKS_DOCKER_PULL_POLICY:-never}"
RUN_TIMESTAMP="${QLINKS_EVIDENCE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_LABEL="${QLINKS_EVIDENCE_RUN_ID:-qdm_sec7_fixed_o1_p0}"
if [[ "${RUN_LABEL}" =~ [0-9]{8}T[0-9]{6}Z$ ]]; then
    RUN_ID="${RUN_LABEL}"
else
    RUN_ID="${RUN_LABEL}_${RUN_TIMESTAMP}"
fi

BASE_RUN_ID="${QLINKS_QDM_BASE_RUN_ID:-qdm_checkerboard_fullsym_finite_beta_20260810T164206Z}"
PRIMME_RUN_ID="${QLINKS_QDM_PRIMME_RUN_ID:-qdm_checkerboard_primme_staged_20260825T164226Z}"
CONTAINER_REPO_DIR="/workspace/qlinks"
CONTAINER_DATA_DIR="${CONTAINER_REPO_DIR}/experimental/data"
HOST_DATA_DIR="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve(strict=False))' "${QLINKS_DATA_DIR:-${REPO_ROOT}/experimental/data}")"
BASE_DATA_DIR="${CONTAINER_DATA_DIR}/evidence_jobs/${BASE_RUN_ID}"
PRIMME_DATA_DIR="${CONTAINER_DATA_DIR}/evidence_jobs/${PRIMME_RUN_ID}"
OUTPUT_DATA_DIR="${CONTAINER_DATA_DIR}/evidence_jobs/${RUN_ID}"
CACHE_ROOT="${CONTAINER_DATA_DIR}/evidence_cache"

THREADS="${QLINKS_NUM_THREADS:-16}"
MEMORY_LIMIT="${QLINKS_DOCKER_MEMORY_LIMIT:-400g}"
CPUS_LIMIT="${QLINKS_DOCKER_CPUS:-}"
SHM_SIZE="${QLINKS_DOCKER_SHM_SIZE:-16g}"
DRY_RUN="${QLINKS_DOCKER_DRY_RUN:-0}"
TARGET_BUDGETS="${QLINKS_QDM_TARGET_BLOCK_BUDGETS:-640,768}"
TARGET_TOLERANCES="${QLINKS_QDM_TARGET_BLOCK_TOLERANCES:-1e-9,1e-10}"
FIXED_WIDTHS="${QLINKS_QDM_FIXED_O1_WIDTHS:-0.10,0.20,0.25,0.50}"
WARM_START_VECTORS="${QLINKS_QDM_PRIMME_WARM_START_VECTORS:-512}"
STAGE="status"
SOLVE_POLICY="cache/read-only status"

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
    target-block-status)
        JOB_COMMAND=(
            python experimental/jobs/qdm_sec7_target_block.py
            --mode status
            --cache-root "${CACHE_ROOT}"
            --output-dir "${OUTPUT_DATA_DIR}"
        )
        SOLVE_POLICY="no eigensolver; validate Lx=12 sector/cage and the persisted 512-vector checkpoint"
        ;;
    target-block-refine)
        JOB_COMMAND=(
            python experimental/jobs/qdm_sec7_target_block.py
            --mode refine
            --cache-root "${CACHE_ROOT}"
            --output-dir "${OUTPUT_DATA_DIR}"
            --budgets "${TARGET_BUDGETS}"
            --tolerances "${TARGET_TOLERANCES}"
        )
        SOLVE_POLICY="target E=12 only; warm-start from validated staged PRIMME cache; no broad-window acceptance test"
        ;;
    fixed-O1-pilot)
        JOB_COMMAND=(
            python experimental/jobs/qdm_sec7_fixed_o1_pilot.py
            --base-data-dir "${BASE_DATA_DIR}"
            --primme-data-dir "${PRIMME_DATA_DIR}"
            --output-dir "${OUTPUT_DATA_DIR}"
            --widths "${FIXED_WIDTHS}"
        )
        SOLVE_POLICY="exact dense ED at Lx=4,8 only; PRIMME and Lx=12 eigensolvers forbidden"
        ;;
    status)
        JOB_COMMAND=(
            bash -lc
            "python experimental/jobs/qdm_sec7_target_block.py --mode status --cache-root '${CACHE_ROOT}' --output-dir '${OUTPUT_DATA_DIR}'"
        )
        SOLVE_POLICY="no eigensolver; target-block cache inventory only"
        ;;
    *)
        echo "unknown stage: ${STAGE}" >&2
        echo "expected one of: status, target-block-status, target-block-refine, fixed-O1-pilot" >&2
        exit 2
        ;;
esac

mkdir -p "${HOST_DATA_DIR}/evidence_jobs/${RUN_ID}" "${HOST_DATA_DIR}/evidence_cache"

DOCKER_LIMIT_ARGS=(--memory "${MEMORY_LIMIT}" --shm-size "${SHM_SIZE}")
[[ -z "${CPUS_LIMIT}" ]] || DOCKER_LIMIT_ARGS+=(--cpus "${CPUS_LIMIT}")

DOCKER_COMMAND=(
    docker run -d
    --pull "${PULL_POLICY}"
    --init
    --name "${CONTAINER_NAME}"
    --label "qlinks.evidence.run_id=${RUN_ID}"
    --label "qlinks.evidence.timestamp=${RUN_TIMESTAMP}"
    --label "qlinks.evidence.job=qdm_sec7_fixed_o1_p0"
    --label "qlinks.evidence.stage=${STAGE}"
    --restart no
    "${DOCKER_LIMIT_ARGS[@]}"
    --env PYTHONUNBUFFERED=1
    --env QLINKS_EVIDENCE_RUN_ID="${RUN_ID}"
    --env QLINKS_EVIDENCE_RUN_TIMESTAMP="${RUN_TIMESTAMP}"
    --env QLINKS_EVIDENCE_CACHE_ROOT="${CACHE_ROOT}"
    --env QLINKS_EVIDENCE_CACHE_RESUME=1
    --env QLINKS_EVIDENCE_CACHE_WRITE=1
    --env QLINKS_EVIDENCE_CACHE_FORCE_RECOMPUTE=0
    --env QLINKS_QDM_RESUMABLE_SPECTRUM=1
    --env QLINKS_QDM_FOLDED_BACKEND=primme
    --env QLINKS_QDM_PRIMME_WARM_START_VECTORS="${WARM_START_VECTORS}"
    --env MPLBACKEND=Agg
    --env OPENBLAS_NUM_THREADS="${THREADS}"
    --env OMP_NUM_THREADS="${THREADS}"
    --env MKL_NUM_THREADS="${THREADS}"
    --env NUMEXPR_NUM_THREADS="${THREADS}"
    --env VECLIB_MAXIMUM_THREADS="${THREADS}"
    --env PYTHONPATH="${CONTAINER_REPO_DIR}/experimental/jobs/qdm_resume_site:${CONTAINER_REPO_DIR}:${CONTAINER_REPO_DIR}/experimental/notebooks:${CONTAINER_REPO_DIR}/experimental/jobs"
    --volume "${REPO_ROOT}:${CONTAINER_REPO_DIR}:ro"
    --volume "${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}"
    --workdir "${CONTAINER_REPO_DIR}"
    "${IMAGE_NAME}"
    "${JOB_COMMAND[@]}"
)

printf 'Resolved Sec. VII P0 Docker command:\n  '
printf '%q ' "${DOCKER_COMMAND[@]}"
printf '\n\n'

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run requested; no container was started."
else
    "${DOCKER_COMMAND[@]}"
fi

cat <<EOF
Job: qdm_sec7_fixed_o1_p0
Run id: ${RUN_ID}
Stage: ${STAGE}
Container: ${CONTAINER_NAME}
Image: ${IMAGE_NAME}
Pull policy: ${PULL_POLICY}
Authoritative Lx=4,8 base: ${BASE_RUN_ID}
Staged PRIMME source: ${PRIMME_RUN_ID}
Output evidence: ${HOST_DATA_DIR}/evidence_jobs/${RUN_ID}
Stable cache: ${HOST_DATA_DIR}/evidence_cache
Target refinement budgets: ${TARGET_BUDGETS}
Target refinement tolerances: ${TARGET_TOLERANCES}
Fixed-O(1) pilot widths: ${FIXED_WIDTHS}
PRIMME warm-start vector cap: ${WARM_START_VECTORS}
Thread limit: ${THREADS}
Memory limit: ${MEMORY_LIMIT}
Shared memory: ${SHM_SIZE}
Solve policy: ${SOLVE_POLICY}

Recommended immediate sequence using one explicit run id:
  export QLINKS_EVIDENCE_RUN_ID=${RUN_ID}
  scripts/docker/docker_run_qdm_sec7_p0.sh --stage target-block-status
  scripts/docker/docker_run_qdm_sec7_p0.sh --stage target-block-refine
  scripts/docker/docker_run_qdm_sec7_p0.sh --stage fixed-O1-pilot

Do not launch an Lx=12 fixed-window production solve until the fixed-O1 pilot
has selected and recorded a primary total-energy half-width.
EOF

if [[ "${DRY_RUN}" != "1" ]]; then
    cat <<EOF

Follow logs:
  docker logs -f ${CONTAINER_NAME}

Check status:
  docker ps -a --filter name=${CONTAINER_NAME}
EOF
fi
