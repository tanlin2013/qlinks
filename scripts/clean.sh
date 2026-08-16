#!/bin/bash
set -euxo pipefail

uv run ruff check --fix qlinks/ tests/
uv run ruff format qlinks/ tests/
