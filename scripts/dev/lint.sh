#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

"${SCRIPT_DIR}/lint_blocking.sh"
"${SCRIPT_DIR}/lint_advisory.sh"
