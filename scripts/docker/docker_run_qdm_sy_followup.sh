#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${REPO_ROOT}"

IMAGE="${QLINKS_DOCKER_IMAGE:-tanlin2013/qlinks:notebook}"
CPUS="${QLINKS_DOCKER_CPUS:-4}"
MEMORY="${QLINKS_DOCKER_MEMORY_LIMIT:-32g}"
THREADS="${QLINKS_NUM_THREADS:-4}"
RUN_TIMESTAMP="${QLINKS_EVIDENCE_RUN_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ID="${QLINKS_EVIDENCE_RUN_ID:-qdm_sy_followup_${RUN_TIMESTAMP}}"
OUTPUT="/workspace/qlinks/experimental/data/evidence_jobs/${RUN_ID}"

mkdir -p "experimental/data/evidence_jobs/${RUN_ID}" experimental/data/evidence_cache

echo "[qdm-sy] image=${IMAGE}"
echo "[qdm-sy] run_id=${RUN_ID}"
echo "[qdm-sy] cpus=${CPUS} memory=${MEMORY} threads=${THREADS}"

# Preserve a host log and the original exit status even when Docker removes the container.
set -o pipefail
docker run --rm --init \
  --cpus "${CPUS}" \
  --memory "${MEMORY}" \
  -e MPLBACKEND=Agg \
  -e PYTHONPATH=/workspace/qlinks:/workspace/qlinks/experimental/notebooks:/workspace/qlinks/experimental/jobs \
  -e QLINKS_EVIDENCE_RUN_ID="${RUN_ID}" \
  -e QLINKS_EVIDENCE_RUN_TIMESTAMP="${RUN_TIMESTAMP}" \
  -e QLINKS_EVIDENCE_CACHE_ROOT=/workspace/qlinks/experimental/data/evidence_cache \
  -e OMP_NUM_THREADS="${THREADS}" \
  -e OPENBLAS_NUM_THREADS="${THREADS}" \
  -e MKL_NUM_THREADS="${THREADS}" \
  -v "${REPO_ROOT}:/workspace/qlinks:ro" \
  -v "${REPO_ROOT}/experimental/data:/workspace/qlinks/experimental/data" \
  -w /workspace/qlinks \
  "${IMAGE}" \
  python experimental/jobs/qdm_sy_followup.py \
    --output-dir "${OUTPUT}" \
    "$@" 2>&1 | tee "experimental/data/evidence_jobs/${RUN_ID}/run.log"

echo "[qdm-sy] completed: experimental/data/evidence_jobs/${RUN_ID}"
