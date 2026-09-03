#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

IMAGE_NAME="${QLINKS_DOCKER_IMAGE:-tanlin2013/qlinks:notebook}"
RUN_TIMESTAMP="${QLINKS_EVIDENCE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_LABEL="${QLINKS_EVIDENCE_RUN_ID:-spin1_convention_migration}"
if [[ "${RUN_LABEL}" =~ [0-9]{8}T[0-9]{6}Z$ ]]; then
    RUN_ID="${RUN_LABEL}"
else
    RUN_ID="${RUN_LABEL}_${RUN_TIMESTAMP}"
fi

P0_SOURCE_RUN_ID="${QLINKS_SEC6_P0_RUN_ID:-spin1_sec6_integration_20260825T073925Z}"
P1_SOURCE_RUN_ID="${QLINKS_SEC6_P1_RUN_ID:-spin1_sec6_p1_claim_upgrades_20260827T055013Z}"
CONTAINER_REPO_DIR="/workspace/qlinks"
CONTAINER_DATA_DIR="${CONTAINER_REPO_DIR}/experimental/data"
P0_SOURCE_DIR="${CONTAINER_DATA_DIR}/evidence_jobs/${P0_SOURCE_RUN_ID}"
P1_SOURCE_DIR="${CONTAINER_DATA_DIR}/evidence_jobs/${P1_SOURCE_RUN_ID}"
OUTPUT_DATA_DIR="${CONTAINER_DATA_DIR}/evidence_jobs/${RUN_ID}"
P0_DERIVED_DIR="${OUTPUT_DATA_DIR}/p0"
P1_DERIVED_DIR="${OUTPUT_DATA_DIR}/p1"
VALIDATION_DIR="${OUTPUT_DATA_DIR}/validation"

HOST_DATA_DIR="$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve(strict=False))' "${QLINKS_DATA_DIR:-${REPO_ROOT}/experimental/data}")"
THREADS="${QLINKS_NUM_THREADS:-1}"
DENSE_SIZES="${QLINKS_SPIN1_CONVENTION_DENSE_SIZES:-8}"
MEMORY_LIMIT="${QLINKS_DOCKER_MEMORY_LIMIT:-}"
CPUS_LIMIT="${QLINKS_DOCKER_CPUS:-}"
SHM_SIZE="${QLINKS_DOCKER_SHM_SIZE:-16g}"
DRY_RUN="${QLINKS_DOCKER_DRY_RUN:-0}"
USE_TEX="${QLINKS_SEC6_USE_TEX:-1}"
STAGE="status"
SOLVE_POLICY="disabled; deterministic conversion/validation only"

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
    status)
        JOB_COMMAND=(
            python -c
            'import json, pathlib; root=pathlib.Path("'"${OUTPUT_DATA_DIR}"'"); print(json.dumps({"p0_mapped": (root/"p0"/"spin1_exchange_convention_migration_manifest.json").is_file(), "p1_mapped": (root/"p1"/"spin1_exchange_convention_migration_manifest.json").is_file(), "validation": (root/"validation"/"spin1_xy_exchange_convention_validation.json").is_file(), "jacobian_l8": (root/"validation"/"spin1_xy_sec6_p1_L8_cage_jacobian_conditioning.csv").is_file()}, indent=2))'
        )
        ;;
    migrate-p0)
        JOB_COMMAND=(
            python experimental/jobs/spin1_exchange_convention_migrate_evidence.py
            --source-dir "${P0_SOURCE_DIR}"
            --output-dir "${P0_DERIVED_DIR}"
            --source-run-id "${P0_SOURCE_RUN_ID}"
        )
        ;;
    migrate-p1)
        JOB_COMMAND=(
            python experimental/jobs/spin1_exchange_convention_migrate_evidence.py
            --source-dir "${P1_SOURCE_DIR}"
            --output-dir "${P1_DERIVED_DIR}"
            --source-run-id "${P1_SOURCE_RUN_ID}"
        )
        ;;
    validate)
        JOB_COMMAND=(
            python experimental/jobs/spin1_exchange_convention_validate.py
            --output-dir "${VALIDATION_DIR}"
            --dense-sizes "${DENSE_SIZES}"
        )
        SOLVE_POLICY="cheap validation only: sparse matrix checks plus dense spot checks at requested small sizes; no L14"
        ;;
    jacobian-l8)
        JOB_COMMAND=(
            python experimental/jobs/spin1_sec6_p1_jacobian_l8.py
            --output-dir "${VALIDATION_DIR}"
        )
        SOLVE_POLICY="solver-free L=8 cage/Jacobian calibration"
        ;;
    render-p0)
        JOB_COMMAND=(
            python experimental/jobs/render_spin1_xy_sec6_integration_figures.py
            --data-dir "${P0_DERIVED_DIR}"
        )
        [[ "${USE_TEX}" == "0" ]] || JOB_COMMAND+=(--use-tex)
        ;;
    *)
        echo "unknown stage: ${STAGE}" >&2
        echo "expected one of: status, migrate-p0, migrate-p1, validate, jacobian-l8, render-p0" >&2
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
    --label "qlinks.evidence.job=spin1_exchange_convention_migration"
    --label "qlinks.evidence.stage=${STAGE}"
    --restart no
    "${DOCKER_LIMIT_ARGS[@]}"
    --env PYTHONUNBUFFERED=1
    --env MPLBACKEND=Agg
    --env OPENBLAS_NUM_THREADS="${THREADS}"
    --env OMP_NUM_THREADS="${THREADS}"
    --env MKL_NUM_THREADS="${THREADS}"
    --env NUMEXPR_NUM_THREADS="${THREADS}"
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
Job: spin1_exchange_convention_migration
Run id: ${RUN_ID}
Stage: ${STAGE}
P0 source (immutable): ${P0_SOURCE_RUN_ID}
P1 source (immutable): ${P1_SOURCE_RUN_ID}
Output: ${HOST_DATA_DIR}/evidence_jobs/${RUN_ID}
Dense validation sizes: ${DENSE_SIZES}
Solve policy: ${SOLVE_POLICY}

Recommended sequence with one explicit timestamped run id:
  QLINKS_EVIDENCE_RUN_ID=${RUN_ID} scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage migrate-p0
  QLINKS_EVIDENCE_RUN_ID=${RUN_ID} scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage migrate-p1
  QLINKS_EVIDENCE_RUN_ID=${RUN_ID} scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage validate
  QLINKS_EVIDENCE_RUN_ID=${RUN_ID} scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage jacobian-l8
  QLINKS_EVIDENCE_RUN_ID=${RUN_ID} scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage render-p0

For the optional L=10 dense spot check, rerun only the validation stage with:
  QLINKS_SPIN1_CONVENTION_DENSE_SIZES=8,10 QLINKS_EVIDENCE_RUN_ID=${RUN_ID} \
    scripts/docker/docker_run_spin1_exchange_convention_migration.sh --stage validate

No L=14 eigensolve is available from this runner.
EOF

if [[ "${DRY_RUN}" != "1" ]]; then
    cat <<EOF

Follow logs:
  docker logs -f ${CONTAINER_NAME}
EOF
fi
