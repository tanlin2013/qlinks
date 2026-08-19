#!/bin/bash
set -euxo pipefail

phase="${1:-all}"

run_ruff() {
    uv run ruff check qlinks/ tests/
    uv run ruff format --check qlinks/ tests/

    # Keep maintained Python/Jupyter outside qlinks/tests covered without
    # forcing a repository-wide cleanup of historical/archive material.
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
}

run_repository_health() {
    uv run python tools/repository_health.py --check
}

case "${phase}" in
    ruff)
        run_ruff
        ;;
    repository-health)
        run_repository_health
        ;;
    all)
        run_ruff
        run_repository_health
        ;;
    *)
        echo "unknown blocking-lint phase: ${phase}" >&2
        echo "expected one of: ruff, repository-health, all" >&2
        exit 2
        ;;
esac
