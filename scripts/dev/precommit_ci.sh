#!/bin/bash
set -euo pipefail

base_ref="${1:?usage: precommit_ci.sh BASE_REF [HEAD_REF]}"
head_ref="${2:-HEAD}"

# Keep API- or web-created commits subject to the same repository policy as
# ordinary local git commits. File hooks are evaluated on the branch diff;
# repository-wide pre-push checks are replayed explicitly below.
uv run pre-commit validate-config
uv run pre-commit run \
  --from-ref "${base_ref}" \
  --to-ref "${head_ref}" \
  --hook-stage pre-commit \
  --show-diff-on-failure

uv run pre-commit run uv-lock-check \
  --all-files \
  --hook-stage pre-push \
  --show-diff-on-failure
uv run pre-commit run test-health \
  --all-files \
  --hook-stage pre-push \
  --show-diff-on-failure

# commit-msg hooks cannot run retroactively on GitHub API commits. Commitizen's
# branch check is the CI-equivalent: validate every commit unique to this range.
uv run cz check --rev-range "${base_ref}..${head_ref}"
