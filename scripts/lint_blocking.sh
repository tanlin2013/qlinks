#!/bin/bash
set -euxo pipefail

poetry run isort --check --diff qlinks/ tests/
poetry run black --check qlinks/ tests/
poetry run flake8 qlinks/ tests/
