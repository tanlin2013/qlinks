#!/bin/bash
set -euxo pipefail

# CI ownership boundary: this lane owns Ruff only. Repository/test/lock health
# and notebook normalization belong to the Policy workflow.
uv run ruff check qlinks/ tests/
uv run ruff format --check qlinks/ tests/

# Keep maintained Python/Jupyter outside qlinks/tests covered without forcing a
# repository-wide cleanup of historical/archive material in the same PR.
if [[ -n "${GITHUB_BASE_REF:-}" ]]; then
    git fetch --no-tags origin "${GITHUB_BASE_REF}"
    base="$(git merge-base "origin/${GITHUB_BASE_REF}" HEAD)"
    mapfile -t changed_python < <(
        git diff --name-only --diff-filter=ACMR "${base}" HEAD -- '*.py' '*.pyi' '*.ipynb'
    )
    if ((${#changed_python[@]})); then
        uv run ruff check "${changed_python[@]}"
        uv run ruff format --check "${changed_python[@]}"
    fi
fi
