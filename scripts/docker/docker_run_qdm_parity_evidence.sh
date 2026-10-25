#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
cd "${REPO_ROOT}"
STAGE="${1:-exact}"
if (($#)); then shift; fi
if [[ "${STAGE}" != "exact" && -z "${QLINKS_EVIDENCE_RUN_ID:-}" ]]; then
    echo "${STAGE} requires QLINKS_EVIDENCE_RUN_ID from the completed exact stage." >&2
    echo "Set it again in this SSH session: export QLINKS_EVIDENCE_RUN_ID=YOUR_EXISTING_RUN_ID" >&2
    echo "No container was started and no new evidence folder was created." >&2
    exit 2
fi
RUN_ID="${QLINKS_EVIDENCE_RUN_ID:-qdm_parity_evidence_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ "${RUN_ID}" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "invalid run id" >&2; exit 2; }
DETACH="${QLINKS_DOCKER_DETACH:-0}"
[[ "${DETACH}" == "0" || "${DETACH}" == "1" ]] || {
    echo "QLINKS_DOCKER_DETACH must be 0 or 1" >&2; exit 2;
}
WALLTIME="${QLINKS_QDM_PRIMME_WALLTIME_SECONDS:-}"
if [[ -n "${WALLTIME}" ]]; then
    [[ "${WALLTIME}" =~ ^[1-9][0-9]*$ ]] || {
        echo "QLINKS_QDM_PRIMME_WALLTIME_SECONDS must be a positive integer" >&2; exit 2;
    }
    case "${STAGE}" in
        L12-spectrum|L12-refine) ;;
        *) echo "PRIMME walltime applies only to L12-spectrum or L12-refine" >&2; exit 2 ;;
    esac
fi
DATA_DIR="${QLINKS_DATA_DIR:-${REPO_ROOT}/experimental/data}"
case "${STAGE}" in
    L12-spectrum|L12-reconstruct|L12-refine|L12-observables|L12-figure9|sequence)
        PROTOCOL_FILE="${DATA_DIR}/evidence_jobs/${RUN_ID}/qdm_checkerboard_fixed_O1_window_recommendation.json"
        if [[ ! -f "${PROTOCOL_FILE}" ]]; then
            echo "Missing parity window protocol: ${PROTOCOL_FILE}" >&2
            echo "Reuse the completed exact-stage run ID, or run the exact stage with this same ID:" >&2
            printf '  QLINKS_EVIDENCE_RUN_ID=%q scripts/docker/docker_run_qdm_parity_evidence.sh exact\n' "${RUN_ID}" >&2
            echo "No container was started." >&2
            exit 2
        fi
        ;;
esac
mkdir -p "${DATA_DIR}/evidence_jobs/${RUN_ID}" "${DATA_DIR}/evidence_cache/qdm_target_parity_v1"
DATA_DIR="$(cd "${DATA_DIR}" && pwd -P)"
OUTPUT="/workspace/qlinks/experimental/data/evidence_jobs/${RUN_ID}"
CACHE="/workspace/qlinks/experimental/data/evidence_cache/qdm_target_parity_v1"
THREADS="${QLINKS_NUM_THREADS:-4}"
MEMORY="${QLINKS_DOCKER_MEMORY_LIMIT:-32g}"
IMAGE="${QLINKS_DOCKER_IMAGE:-tanlin2013/qlinks:notebook}"
COMMAND=(python experimental/jobs/qdm_parity_evidence.py --stage "${STAGE}" --output-dir "${OUTPUT}")
case "${STAGE}" in
    exact|target-parity|L12-preflight|gate) ;;
    L12-canonical|L12-matrix-preflight)
        MEMORY="${QLINKS_DOCKER_MEMORY_LIMIT:-128g}"
        ;;
    L12-spectrum|L12-reconstruct|L12-refine)
        MEMORY="${QLINKS_DOCKER_MEMORY_LIMIT:-128g}"
        IMAGE="${QLINKS_DOCKER_IMAGE:-tanlin2013/qlinks:notebook-primme}"
        COMMAND=(python experimental/jobs/qdm_sec7_fixed_o1_l12_spectrum.py
          --output-dir "${OUTPUT}" --cache-root "${CACHE}" --sy-character target
          --budgets "${QLINKS_QDM_PARITY_BUDGETS:-768,1024,1536,2048,3072,4096}"
          --tolerance 1e-8 --residual-tolerance 1e-6 --max-budget 8192)
        if [[ "${STAGE}" == "L12-reconstruct" ]]; then
            COMMAND+=(--reconstruct-cached-only --budgets "${QLINKS_QDM_PARITY_BUDGETS:-4096}")
        elif [[ "${STAGE}" == "L12-refine" ]]; then
            COMMAND+=(--refine-cached-budget --budgets "${QLINKS_QDM_PARITY_BUDGETS:-4096}"
              --tolerance 1e-11)
        fi
        ;;
    L12-observables)
        MEMORY="${QLINKS_DOCKER_MEMORY_LIMIT:-128g}"
        COMMAND=(python experimental/jobs/qdm_sec7_fixed_o1_l12_observables.py
          --output-dir "${OUTPUT}" --cache-root "${CACHE}" --sy-character target
          --primme-data-dir "${OUTPUT}" --residual-tolerance 1e-6
          --observable-budget-tolerance 5e-5)
        ;;
    L12-figure9)
        IMAGE="${QLINKS_DOCKER_IMAGE:-tanlin2013/qlinks:notebook-primme}"
        COMMAND=(python experimental/jobs/finalize_qdm_parity_figure9.py --output-dir "${OUTPUT}")
        ;;
    sequence)
        COMMAND=(python experimental/jobs/qdm_sec7_fixed_o1_sequence.py
          --output-dir "${OUTPUT}" --sy-character target)
        ;;
    *) echo "unknown stage: ${STAGE}" >&2; exit 2 ;;
esac
if [[ -n "${WALLTIME}" ]]; then
    # Run the limit inside Docker: an SSH disconnect or a client-side timeout
    # must not leave a supposedly bounded native solver running indefinitely.
    COMMAND=(timeout --signal=TERM --kill-after=30s "${WALLTIME}s" "${COMMAND[@]}")
fi
COMMIT="$(git rev-parse HEAD)"
LIFECYCLE=(--rm)
if [[ "${DETACH}" == "1" ]]; then
    CONTAINER_NAME="qlinks-parity-${RUN_ID}-${STAGE}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    # Retain an exited container so its logs, exit code and OOM flag survive.
    LIFECYCLE=(--detach --name "${CONTAINER_NAME}")
fi
# Preserve the solver's defaults unless a tuning override was explicitly set.
# Host environment variables do not otherwise reach the Docker process.
PRIMME_OVERRIDES=()
for VARIABLE in QLINKS_QDM_PRIMME_METHOD QLINKS_QDM_PRIMME_MAX_BASIS_SIZE \
    QLINKS_QDM_PRIMME_MAX_BLOCK_SIZE \
    QLINKS_QDM_PRIMME_MIN_RESTART_SIZE QLINKS_QDM_PRIMME_MAX_PREV_RETAIN; do
    if [[ -n "${!VARIABLE:-}" ]]; then
        PRIMME_OVERRIDES+=(--env "${VARIABLE}=${!VARIABLE}")
    fi
done
DOCKER_COMMAND=(docker run "${LIFECYCLE[@]}" --init
  --cpus "${QLINKS_DOCKER_CPUS:-${THREADS}}" --memory "${MEMORY}"
  --env MPLBACKEND=Agg --env PYTHONUNBUFFERED=1
  --env PYTHONPATH=/workspace/qlinks:/workspace/qlinks/experimental/jobs:/workspace/qlinks/experimental/notebooks
  --env QLINKS_EVIDENCE_RUN_ID="${RUN_ID}" --env QLINKS_EVIDENCE_SOURCE_COMMIT="${COMMIT}"
  --env QLINKS_EVIDENCE_CACHE_ROOT="${CACHE}" --env QLINKS_QDM_FOLDED_BACKEND=primme
  --env QLINKS_QDM_PRIMME_WARM_START_VECTORS="${QLINKS_QDM_PRIMME_WARM_START_VECTORS:-512}"
  --env QLINKS_QDM_PRIMME_PRINT_LEVEL="${QLINKS_QDM_PRIMME_PRINT_LEVEL:-2}"
  --env QLINKS_QDM_PRIMME_PROGRESS_SECONDS="${QLINKS_QDM_PRIMME_PROGRESS_SECONDS:-300}"
  "${PRIMME_OVERRIDES[@]}"
  --env OMP_NUM_THREADS="${THREADS}" --env OPENBLAS_NUM_THREADS="${THREADS}"
  --env MKL_NUM_THREADS="${THREADS}"
  --volume "${REPO_ROOT}:/workspace/qlinks:ro"
  --volume "${DATA_DIR}:/workspace/qlinks/experimental/data"
  --workdir /workspace/qlinks "${IMAGE}" "${COMMAND[@]}" "$@")
printf 'Run id: %s\nStage: %s\nOutput: %s\n' "${RUN_ID}" "${STAGE}" "${DATA_DIR}/evidence_jobs/${RUN_ID}"
printf 'Reuse in every SSH session: export QLINKS_EVIDENCE_RUN_ID=%q\n' "${RUN_ID}"
if [[ "${QLINKS_DOCKER_DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${DOCKER_COMMAND[@]}"
    printf '\n'
elif [[ "${DETACH}" == "1" ]]; then
    CONTAINER_ID="$("${DOCKER_COMMAND[@]}")"
    ID_FILE="${DATA_DIR}/evidence_jobs/${RUN_ID}/${STAGE}.container-id"
    printf '%s\n' "${CONTAINER_ID}" > "${ID_FILE}.$$.tmp"
    mv -- "${ID_FILE}.$$.tmp" "${ID_FILE}"
    printf 'Container: %s\nContainer id file: %s\n' "${CONTAINER_NAME}" "${ID_FILE}"
    echo "Launch succeeded; numerical completion and acceptance are not yet established."
    printf 'Follow logs: docker logs -f %q\n' "${CONTAINER_ID}"
    printf "Inspect status: docker inspect --format '{{json .State}}' %q\n" "${CONTAINER_ID}"
    printf 'Save logs: docker logs %q > %q 2>&1\n' "${CONTAINER_ID}" \
      "${DATA_DIR}/evidence_jobs/${RUN_ID}/${STAGE}.log"
    echo "The container runs independently of this SSH session and is retained after exit."
else
    # Report native/signal exits even when Python cannot write a traceback.
    set +e
    "${DOCKER_COMMAND[@]}" 2>&1 | tee -a "${DATA_DIR}/evidence_jobs/${RUN_ID}/${STAGE}.log"
    STATUS=("${PIPESTATUS[@]}")
    set -e
    printf 'Container exit code: %s; log writer exit code: %s\n' "${STATUS[0]}" "${STATUS[1]}" \
      | tee -a "${DATA_DIR}/evidence_jobs/${RUN_ID}/${STAGE}.log"
    if [[ "${STATUS[0]}" != "0" ]]; then exit "${STATUS[0]}"; fi
    exit "${STATUS[1]}"
fi
