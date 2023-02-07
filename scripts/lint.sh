#!/bin/bash
set -euxo pipefail

poetry run cruft check
poetry run safety check -i 39462 -i 40291
poetry run bandit -c pyproject.toml -r qlinks/
poetry run isort --check --diff qlinks/ tests/
poetry run black --check qlinks/ tests/
poetry run flake8 qlinks/ tests/
poetry run mypy \
           --install-types \
           --non-interactive \
           qlinks/
#  https://mypy.readthedocs.io/en/stable/running_mypy.html#library-stubs-not-installed
