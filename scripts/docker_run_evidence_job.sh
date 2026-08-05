#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

IMAGE_NAME="${QLINKS_DOCKER_IMAGE:-tanlin2013/qlinks:notebook}"
JOB_NAME="${1:-}"
if [[ -z "${JOB_NAME}" ]]; then
    cat >&2 <<'USAGE'
Usage: scripts/docker_run_evidence_job.sh spin1|qdm [job-script-args...]

Examples:
  scripts/docker_run_evidence_job.sh spin1 --profile known
  scripts/docker_run_evidence_job.sh qdm --profile production --stage compute
  QLINKS_NUM_THREADS=16 QLINKS_DOCKER_MEMORY_LIMIT=400g \
    scripts/docker_run_evidence_job.sh spin1 --profile production --stage compute \
      --large-size-sizes 14 --large-size-eigenpairs 8192 --timeout -1
  scripts/docker_run_evidence_job.sh spin1 \
      --stage render \
      --source-data-dir experimental/data/evidence_jobs/spin1_production \
      --use-tex \
      --figure-formats pdf,svg \
      --export-dir output/spin1_production

Path options may be given as:
  * paths relative to the repository root,
  * absolute host paths under the repository, QLINKS_DATA_DIR, or
    QLINKS_OUTPUT_DIR, or
  * their container paths under /workspace/qlinks or /workspace/output.

A UTC timestamp is appended to the run id, evidence folder, and container name.
The job runs in a detached container. Use the printed docker logs command to
follow it. Set QLINKS_DOCKER_DRY_RUN=1 to print the resolved Docker command
without starting a container.
USAGE
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

RUN_LABEL="${QLINKS_EVIDENCE_RUN_ID:-${JOB_SLUG}}"
RUN_TIMESTAMP="${QLINKS_EVIDENCE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ "${RUN_LABEL}" =~ [0-9]{8}T[0-9]{6}Z$ ]]; then
    RUN_ID="${RUN_LABEL}"
else
    RUN_ID="${RUN_LABEL}_${RUN_TIMESTAMP}"
fi
CONTAINER_NAME="${QLINKS_CONTAINER_NAME:-qlinks-${RUN_ID//_/-}}"
CONTAINER_REPO_DIR="/workspace/qlinks"
CONTAINER_NOTEBOOK_DIR="${CONTAINER_REPO_DIR}/experimental/notebooks"
CONTAINER_DATA_DIR="${CONTAINER_REPO_DIR}/experimental/data"
CONTAINER_OUTPUT_DIR="/workspace/output"

# Resolve host mount roots before translating path-bearing job flags.
canonicalize_path() {
    python3 - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}

HOST_DATA_DIR="$(canonicalize_path "${QLINKS_DATA_DIR:-${REPO_ROOT}/experimental/data}")"
HOST_OUTPUT_DIR="$(canonicalize_path "${QLINKS_OUTPUT_DIR:-${REPO_ROOT}/output}")"
THREADS="${QLINKS_NUM_THREADS:-1}"
MEMORY_LIMIT="${QLINKS_DOCKER_MEMORY_LIMIT:-}"
CPUS_LIMIT="${QLINKS_DOCKER_CPUS:-}"
SHM_SIZE="${QLINKS_DOCKER_SHM_SIZE:-16g}"
DRY_RUN="${QLINKS_DOCKER_DRY_RUN:-0}"

path_is_within() {
    local child="$1"
    local parent="$2"
    [[ "${child}" == "${parent}" || "${child}" == "${parent}/"* ]]
}

relative_suffix() {
    local child="$1"
    local parent="$2"
    if [[ "${child}" == "${parent}" ]]; then
        printf '%s' ""
    else
        printf '%s' "${child#${parent}/}"
    fi
}

# Set RESOLVED_HOST_PATH and RESOLVED_CONTAINER_PATH for one CLI path.
# Existing container paths are mapped back to their host-mounted equivalents so
# the wrapper can validate and accurately report them.
resolve_mounted_path() {
    local raw_path="$1"
    local purpose="$2"
    local host_path container_path suffix

    case "${raw_path}" in
        "${CONTAINER_DATA_DIR}"|"${CONTAINER_DATA_DIR}/"*)
            suffix="$(relative_suffix "${raw_path}" "${CONTAINER_DATA_DIR}")"
            host_path="${HOST_DATA_DIR}${suffix:+/${suffix}}"
            container_path="${raw_path}"
            ;;
        "${CONTAINER_OUTPUT_DIR}"|"${CONTAINER_OUTPUT_DIR}/"*)
            suffix="$(relative_suffix "${raw_path}" "${CONTAINER_OUTPUT_DIR}")"
            host_path="${HOST_OUTPUT_DIR}${suffix:+/${suffix}}"
            container_path="${raw_path}"
            ;;
        "${CONTAINER_REPO_DIR}"|"${CONTAINER_REPO_DIR}/"*)
            suffix="$(relative_suffix "${raw_path}" "${CONTAINER_REPO_DIR}")"
            host_path="${REPO_ROOT}${suffix:+/${suffix}}"
            container_path="${raw_path}"
            ;;
        *)
            if [[ "${raw_path}" = /* ]]; then
                host_path="$(canonicalize_path "${raw_path}")"
            else
                host_path="$(canonicalize_path "${REPO_ROOT}/${raw_path}")"
            fi

            # Prefer the dedicated data/output mounts over the repository mount
            # when mount roots overlap.
            if path_is_within "${host_path}" "${HOST_DATA_DIR}"; then
                suffix="$(relative_suffix "${host_path}" "${HOST_DATA_DIR}")"
                container_path="${CONTAINER_DATA_DIR}${suffix:+/${suffix}}"
            elif path_is_within "${host_path}" "${HOST_OUTPUT_DIR}"; then
                suffix="$(relative_suffix "${host_path}" "${HOST_OUTPUT_DIR}")"
                container_path="${CONTAINER_OUTPUT_DIR}${suffix:+/${suffix}}"
            elif path_is_within "${host_path}" "${REPO_ROOT}"; then
                suffix="$(relative_suffix "${host_path}" "${REPO_ROOT}")"
                container_path="${CONTAINER_REPO_DIR}${suffix:+/${suffix}}"
            else
                cat >&2 <<EOF_PATH
Cannot pass ${purpose} path to the container because it is outside all mounted roots:
  ${host_path}

Place it under one of:
  repository:       ${REPO_ROOT}
  data mount:       ${HOST_DATA_DIR}
  output mount:     ${HOST_OUTPUT_DIR}

Alternatively set QLINKS_DATA_DIR or QLINKS_OUTPUT_DIR before starting the job.
EOF_PATH
                exit 2
            fi
            ;;
    esac

    RESOLVED_HOST_PATH="$(canonicalize_path "${host_path}")"
    RESOLVED_CONTAINER_PATH="${container_path}"
}

# Parse path-bearing flags so host paths are rewritten to their corresponding
# container-visible paths. Every unrelated/new job-script flag is forwarded
# unchanged.
STAGE="${QLINKS_EVIDENCE_STAGE:-all}"
STAGE_FLAG_SEEN=0
SOURCE_DATA_HOST=""
SOURCE_DATA_CONTAINER=""
DATA_HOST=""
DATA_CONTAINER=""
EXPORT_HOST=""
EXPORT_CONTAINER=""
FORWARDED_ARGS=()
QDM_MICRO_REPEATS=""
ALLOW_LARGE_DENSE_ED=0

while (($#)); do
    case "$1" in
        --stage)
            [[ $# -ge 2 ]] || { echo "--stage requires a value" >&2; exit 2; }
            STAGE="$2"
            STAGE_FLAG_SEEN=1
            FORWARDED_ARGS+=("--stage" "$2")
            shift 2
            ;;
        --stage=*)
            STAGE="${1#*=}"
            STAGE_FLAG_SEEN=1
            FORWARDED_ARGS+=("$1")
            shift
            ;;
        --source-data-dir)
            [[ $# -ge 2 ]] || { echo "--source-data-dir requires a value" >&2; exit 2; }
            resolve_mounted_path "$2" "source-data"
            SOURCE_DATA_HOST="${RESOLVED_HOST_PATH}"
            SOURCE_DATA_CONTAINER="${RESOLVED_CONTAINER_PATH}"
            FORWARDED_ARGS+=("--source-data-dir" "${SOURCE_DATA_CONTAINER}")
            shift 2
            ;;
        --source-data-dir=*)
            resolve_mounted_path "${1#*=}" "source-data"
            SOURCE_DATA_HOST="${RESOLVED_HOST_PATH}"
            SOURCE_DATA_CONTAINER="${RESOLVED_CONTAINER_PATH}"
            FORWARDED_ARGS+=("--source-data-dir=${SOURCE_DATA_CONTAINER}")
            shift
            ;;
        --data-dir)
            [[ $# -ge 2 ]] || { echo "--data-dir requires a value" >&2; exit 2; }
            resolve_mounted_path "$2" "data"
            DATA_HOST="${RESOLVED_HOST_PATH}"
            DATA_CONTAINER="${RESOLVED_CONTAINER_PATH}"
            FORWARDED_ARGS+=("--data-dir" "${DATA_CONTAINER}")
            shift 2
            ;;
        --data-dir=*)
            resolve_mounted_path "${1#*=}" "data"
            DATA_HOST="${RESOLVED_HOST_PATH}"
            DATA_CONTAINER="${RESOLVED_CONTAINER_PATH}"
            FORWARDED_ARGS+=("--data-dir=${DATA_CONTAINER}")
            shift
            ;;
        --export-dir)
            [[ $# -ge 2 ]] || { echo "--export-dir requires a value" >&2; exit 2; }
            resolve_mounted_path "$2" "export"
            EXPORT_HOST="${RESOLVED_HOST_PATH}"
            EXPORT_CONTAINER="${RESOLVED_CONTAINER_PATH}"
            FORWARDED_ARGS+=("--export-dir" "${EXPORT_CONTAINER}")
            shift 2
            ;;
        --export-dir=*)
            resolve_mounted_path "${1#*=}" "export"
            EXPORT_HOST="${RESOLVED_HOST_PATH}"
            EXPORT_CONTAINER="${RESOLVED_CONTAINER_PATH}"
            FORWARDED_ARGS+=("--export-dir=${EXPORT_CONTAINER}")
            shift
            ;;
        --microcanonical-repeats|--ed-repeats)
            [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
            QDM_MICRO_REPEATS="$2"
            FORWARDED_ARGS+=("$1" "$2")
            shift 2
            ;;
        --microcanonical-repeats=*|--ed-repeats=*)
            QDM_MICRO_REPEATS="${1#*=}"
            FORWARDED_ARGS+=("$1")
            shift
            ;;
        --allow-large-dense-ed)
            ALLOW_LARGE_DENSE_ED=1
            FORWARDED_ARGS+=("$1")
            shift
            ;;
        *)
            FORWARDED_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ "${STAGE_FLAG_SEEN}" == "0" ]]; then
    # The job scripts read QLINKS_EVIDENCE_STAGE only inside the container.
    # Forward the resolved host-side default explicitly so both sides agree.
    FORWARDED_ARGS+=("--stage" "${STAGE}")
fi

case "${STAGE}" in
    compute|render|all) ;;
    *)
        echo "Invalid --stage value: ${STAGE}; expected compute, render, or all" >&2
        exit 2
        ;;
esac

if [[ "${JOB_NAME}" == "qdm" || "${JOB_NAME}" == "square_qdm" ]]; then
    if [[ -n "${QDM_MICRO_REPEATS}" && "${ALLOW_LARGE_DENSE_ED}" != "1" ]]; then
        IFS=',' read -r -a _qdm_repeats <<< "${QDM_MICRO_REPEATS}"
        for _repeat in "${_qdm_repeats[@]}"; do
            _repeat="${_repeat//[[:space:]]/}"
            if [[ "${_repeat}" =~ ^[0-9]+$ ]] && (( _repeat >= 3 )); then
                cat >&2 <<'EOF_DENSE_GUARD'
Dense QDM microcanonical repeat >=3 is disabled by default. The complete
full-spectrum eigendecomposition scales as O(d^2) in memory and has already
killed a 400 GiB container. Use --microcanonical-repeats 1,2 for production.
Only add --allow-large-dense-ed after switching to a memory-scalable method or
confirming an effectively exclusive host with a verified memory estimate.
EOF_DENSE_GUARD
                exit 2
            fi
        done
    fi
fi

mkdir -p "${HOST_DATA_DIR}" "${HOST_OUTPUT_DIR}"

if [[ -n "${DATA_HOST}" ]]; then
    mkdir -p "${DATA_HOST}"
fi
if [[ -n "${EXPORT_HOST}" ]]; then
    mkdir -p "${EXPORT_HOST}"
fi
if [[ "${STAGE}" == "render" ]]; then
    if [[ -z "${SOURCE_DATA_HOST}" ]]; then
        echo "--stage render requires --source-data-dir" >&2
        exit 2
    fi
    if [[ ! -d "${SOURCE_DATA_HOST}" ]]; then
        echo "Render source data directory does not exist on the host:" >&2
        echo "  ${SOURCE_DATA_HOST}" >&2
        exit 2
    fi
fi

DOCKER_LIMIT_ARGS=()
if [[ -n "${MEMORY_LIMIT}" ]]; then
    DOCKER_LIMIT_ARGS+=(--memory "${MEMORY_LIMIT}")
fi
if [[ -n "${CPUS_LIMIT}" ]]; then
    DOCKER_LIMIT_ARGS+=(--cpus "${CPUS_LIMIT}")
fi
if [[ -n "${SHM_SIZE}" ]]; then
    DOCKER_LIMIT_ARGS+=(--shm-size "${SHM_SIZE}")
fi

PASSTHROUGH_ENV_ARGS=()
for env_name in \
    QLINKS_EVIDENCE_PROFILE \
    QLINKS_EVIDENCE_FIGURE_FORMATS \
    QLINKS_EVIDENCE_TIMEOUT; do
    if [[ -n "${!env_name:-}" ]]; then
        PASSTHROUGH_ENV_ARGS+=(--env "${env_name}=${!env_name}")
    fi
done

DOCKER_COMMAND=(
    docker run -d
    --init
    --name "${CONTAINER_NAME}"
    --label "qlinks.evidence.run_id=${RUN_ID}"
    --label "qlinks.evidence.timestamp=${RUN_TIMESTAMP}"
    --label "qlinks.evidence.job=${JOB_SLUG}"
    --restart no
    "${DOCKER_LIMIT_ARGS[@]}"
    "${PASSTHROUGH_ENV_ARGS[@]}"
    --env PYTHONUNBUFFERED=1
    --env QLINKS_EVIDENCE_RUN_TIMESTAMP="${RUN_TIMESTAMP}"
    --env QLINKS_DOCKER_MEMORY_LIMIT="${MEMORY_LIMIT}"
    --env MPLBACKEND=Agg
    --env OPENBLAS_NUM_THREADS="${THREADS}"
    --env OMP_NUM_THREADS="${THREADS}"
    --env MKL_NUM_THREADS="${THREADS}"
    --env NUMEXPR_NUM_THREADS="${THREADS}"
    --env VECLIB_MAXIMUM_THREADS="${THREADS}"
    --env PYTHONPATH="${CONTAINER_REPO_DIR}:${CONTAINER_NOTEBOOK_DIR}"
    --volume "${REPO_ROOT}:${CONTAINER_REPO_DIR}"
    --volume "${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}"
    --volume "${HOST_OUTPUT_DIR}:${CONTAINER_OUTPUT_DIR}"
    --workdir "${CONTAINER_REPO_DIR}"
    "${IMAGE_NAME}"
    python "${JOB_SCRIPT}" --run-id "${RUN_ID}"
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

cat <<EOF_STATUS
Job: ${JOB_SLUG}
Run label: ${RUN_LABEL}
Timestamp: ${RUN_TIMESTAMP}
Run id: ${RUN_ID}
Stage: ${STAGE}
Container: ${CONTAINER_NAME}
Image: ${IMAGE_NAME}
Repo mount: ${REPO_ROOT} -> ${CONTAINER_REPO_DIR}
Data mount: ${HOST_DATA_DIR} -> ${CONTAINER_DATA_DIR}
Output mount: ${HOST_OUTPUT_DIR} -> ${CONTAINER_OUTPUT_DIR}
Thread limit: ${THREADS}
Memory limit: ${MEMORY_LIMIT:-unlimited}
CPU limit: ${CPUS_LIMIT:-unlimited}
Shared memory: ${SHM_SIZE:-docker default}
EOF_STATUS

if [[ "${DRY_RUN}" != "1" ]]; then
    cat <<EOF_MONITOR

Started as a detached container.

Follow logs:
  docker logs -f ${CONTAINER_NAME}

Check status:
  docker ps -a --filter name=${CONTAINER_NAME}

Remove the stopped container after inspection:
  docker rm ${CONTAINER_NAME}
EOF_MONITOR
fi

if [[ "${STAGE}" == "render" ]]; then
    cat <<EOF_RENDER

Render source data directory:
  host:      ${SOURCE_DATA_HOST}
  container: ${SOURCE_DATA_CONTAINER}

Final figures and render manifests are written into that source directory.
EOF_RENDER
    if [[ -n "${EXPORT_HOST}" ]]; then
        cat <<EOF_EXPORT
They are also copied to:
  host:      ${EXPORT_HOST}
  container: ${EXPORT_CONTAINER}
EOF_EXPORT
    fi
else
    if [[ -n "${DATA_HOST}" ]]; then
        EXPECTED_DATA_HOST="${DATA_HOST}"
        EXPECTED_DATA_CONTAINER="${DATA_CONTAINER}"
    else
        EXPECTED_DATA_HOST="${HOST_DATA_DIR}/evidence_jobs/${RUN_ID}"
        EXPECTED_DATA_CONTAINER="${CONTAINER_DATA_DIR}/evidence_jobs/${RUN_ID}"
    fi
    cat <<EOF_COMPUTE

Expected data directory:
  host:      ${EXPECTED_DATA_HOST}
  container: ${EXPECTED_DATA_CONTAINER}
EOF_COMPUTE
    if [[ -n "${EXPORT_HOST}" ]]; then
        cat <<EOF_EXPORT2
Requested export directory:
  host:      ${EXPORT_HOST}
  container: ${EXPORT_CONTAINER}
EOF_EXPORT2
    fi
fi
