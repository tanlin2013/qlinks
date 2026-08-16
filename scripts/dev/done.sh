#!/bin/bash
set -euxo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

"${SCRIPT_DIR}/clean.sh"
"${SCRIPT_DIR}/test.sh"
