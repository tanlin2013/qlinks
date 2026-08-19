#!/bin/bash
set -euo pipefail

base_ref="${1:?usage: precommit_ci.sh BASE_REF [HEAD_REF] [PHASE]}"
head_ref="${2:-HEAD}"
phase="${3:-all}"

run_file_policy() {
  uv run pre-commit validate-config

  # CI ownership boundary: Ruff belongs to Lint and repository-health belongs
  # to the static-check lane. This phase owns file hygiene and notebook
  # normalization only.
  SKIP=ruff-check,ruff-format,repository-health \
    uv run pre-commit run \
      --from-ref "${base_ref}" \
      --to-ref "${head_ref}" \
      --hook-stage pre-commit \
      --show-diff-on-failure
}

run_lock_health() {
  uv lock --check
}

run_commit_policy() {
  # commit-msg hooks cannot run retroactively on GitHub API commits. Commitizen's
  # range check is the CI-equivalent for every commit unique to this change.
  uv run cz check --rev-range "${base_ref}..${head_ref}"
}

case "${phase}" in
  files)
    run_file_policy
    ;;
  lock)
    run_lock_health
    ;;
  commits)
    run_commit_policy
    ;;
  all)
    run_file_policy
    run_lock_health
    run_commit_policy
    ;;
  *)
    echo "unknown policy phase: ${phase}" >&2
    echo "expected one of: files, lock, commits, all" >&2
    exit 2
    ;;
esac
