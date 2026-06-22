#!/bin/bash
set -uxo pipefail

_advisory() {
    local label="$1"
    shift

    if "$@"; then
        echo "${label}: passed"
    else
        local status=$?
        echo "::warning title=${label}::advisory check exited with status ${status}"
    fi
}

_advisory "cruft" poetry run cruft check
# Safety CLI v3 prompts for login in non-interactive CI.
# Keep dependency vulnerability checks out of advisory lint until a
# non-interactive scanner is pinned in the lockfile.
_advisory "bandit" poetry run bandit -c pyproject.toml -r qlinks/
_advisory "mypy" poetry run mypy qlinks/
#  https://mypy.readthedocs.io/en/stable/running_mypy.html#library-stubs-not-installed
