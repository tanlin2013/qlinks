#!/bin/bash
set -euo pipefail

base_ref="${1:?usage: precommit_ci.sh BASE_REF [HEAD_REF] [PHASE]}"
head_ref="${2:-HEAD}"
phase="${3:-all}"

run_file_policy() {
  uv run pre-commit validate-config

  # CI ownership boundary: Ruff is verified by Lint / blocking and repository
  # health has its own Policy step. The file-policy phase owns only repository
  # hygiene and notebook normalization.
  SKIP=ruff-check,ruff-format,repository-health \
    uv run pre-commit run \
      --from-ref "${base_ref}" \
      --to-ref "${head_ref}" \
      --hook-stage pre-commit \
      --show-diff-on-failure
}

run_repository_health() {
  uv run pre-commit run repository-health \
    --all-files \
    --hook-stage pre-commit \
    --show-diff-on-failure
}

run_lock_health() {
  uv run pre-commit run uv-lock-check \
    --all-files \
    --hook-stage pre-push \
    --show-diff-on-failure
}

run_test_health() {
  uv run pre-commit run test-health \
    --all-files \
    --hook-stage pre-push \
    --show-diff-on-failure
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
  repository-health)
    run_repository_health
    ;;
  lock)
    run_lock_health
    ;;
  test-health)
    run_test_health
    ;;
  commits)
    run_commit_policy
    ;;
  all)
    run_file_policy
    run_repository_health
    run_lock_health
    run_test_health
    run_commit_policy
    ;;
  *)
    echo "unknown policy phase: ${phase}" >&2
    echo "expected one of: files, repository-health, lock, test-health, commits, all" >&2
    exit 2
    ;;
esac
