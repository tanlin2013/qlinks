#!/bin/bash
set -euxo pipefail

poetry run cruft check
# Safety CLI v3 prompts for login in non-interactive CI.
# Keep dependency vulnerability checks out of advisory lint until a
# non-interactive scanner is pinned in the lockfile.
poetry run bandit -c pyproject.toml -r qlinks/
poetry run mypy qlinks/
#  https://mypy.readthedocs.io/en/stable/running_mypy.html#library-stubs-not-installed
