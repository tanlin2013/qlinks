#!/bin/bash
set -euxo pipefail

poetry run isort qlinks/ tests/
poetry run black qlinks/ tests/
