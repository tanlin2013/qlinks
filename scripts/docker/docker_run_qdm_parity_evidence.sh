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
DATA_DIR="${QLINKS_DATA_DIR:-${REPO_ROOT}/experimental/data}"
case "${STAGE}" in
    L12-spectrum|L12-reconstruct|L12-observables|sequence)
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
    L12-spectrum|L12-reconstruct)
        MEMORY="${QLINKS_DOCKER_MEMORY_LIMIT:-128g}"
        IMAGE="${QLINKS_DOCKER_IMAGE:-tanlin2013/qlinks:notebook-primme}"
        COMMAND=(python experimental/jobs/qdm_sec7_fixed_o1_l12_spectrum.py
          --output-dir "${OUTPUT}" --cache-root "${CACHE}" --sy-character target
          --budgets "${QLINKS_QDM_PARITY_BUDGETS:-768,1024,1536,2048,3072,4096}"
          --tolerance 1e-8 --residual-tolerance 1e-6 --max-budget 8192)
        if [[ "${STAGE}" == "L12-reconstruct" ]]; then
            COMMAND+=(--reconstruct-cached-only --budgets "${QLINKS_QDM_PARITY_BUDGETS:-4096}")
        fi
        ;;
    L12-observables)
        MEMORY="${QLINKS_DOCKER_MEMORY_LIMIT:-128g}"
        COMMAND=(python experimental/jobs/qdm_sec7_fixed_o1_l12_observables.py
          --output-dir "${OUTPUT}" --cache-root "${CACHE}" --sy-character target
          --primme-data-dir "${OUTPUT}" --residual-tolerance 1e-6
          --observable-budget-tolerance 5e-5)
        ;;
    sequence)
        COMMAND=(python experimental/jobs/qdm_sec7_fixed_o1_sequence.py
          --output-dir "${OUTPUT}" --sy-character target)
        ;;
    *) echo "unknown stage: ${STAGE}" >&2; exit 2 ;;
esac
COMMIT="$(git rev-parse HEAD)"
DOCKER_COMMAND=(docker run --rm --init
  --cpus "${QLINKS_DOCKER_CPUS:-${THREADS}}" --memory "${MEMORY}"
  --env MPLBACKEND=Agg --env PYTHONUNBUFFERED=1
  --env PYTHONPATH=/workspace/qlinks:/workspace/qlinks/experimental/jobs:/workspace/qlinks/experimental/notebooks
  --env QLINKS_EVIDENCE_RUN_ID="${RUN_ID}" --env QLINKS_EVIDENCE_SOURCE_COMMIT="${COMMIT}"
  --env QLINKS_EVIDENCE_CACHE_ROOT="${CACHE}" --env QLINKS_QDM_FOLDED_BACKEND=primme
  --env QLINKS_QDM_PRIMME_WARM_START_VECTORS=512
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
else
    "${DOCKER_COMMAND[@]}" 2>&1 | tee -a "${DATA_DIR}/evidence_jobs/${RUN_ID}/${STAGE}.log"
fi
