#!/bin/bash
set -euxo pipefail

./scripts/lint_blocking.sh
./scripts/lint_advisory.sh
