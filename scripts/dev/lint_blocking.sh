#!/bin/bash
set -euxo pipefail

uv run ruff check qlinks/ tests/
uv run ruff format --check qlinks/ tests/

# The pre-commit policy applies Ruff to every changed Python/Jupyter file, not
# only package/tests. Mirror that contract in the ordinary PR lint lane so
# experimental jobs and notebooks cannot make the PR look green while the
# required push-only policy check is failing.
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

uv run python tools/repository_health.py --check --quiet
