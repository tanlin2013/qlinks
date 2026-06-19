#!/bin/bash
set -euxo pipefail

poetry run cruft check
poetry run safety scan
poetry run bandit -c pyproject.toml -r qlinks/
poetry run mypy qlinks/
#  https://mypy.readthedocs.io/en/stable/running_mypy.html#library-stubs-not-installed
