#!/bin/bash
set -euxo pipefail

uv run ruff check qlinks/ tests/
uv run ruff format --check qlinks/ tests/
uv run python tools/repository_health.py --check --quiet
