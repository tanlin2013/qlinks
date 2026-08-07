#!/usr/bin/env bash
set -euo pipefail

lane="${1:-fast}"
case "${lane}" in
  fast|unit)
    marker_expression="not integration and not scientific and not manual and not gpu"
    default_coverage=1
    ;;
  integration)
    marker_expression="integration and not scientific and not manual and not gpu"
    default_coverage=0
    ;;
  scientific)
    marker_expression="scientific and not manual and not gpu"
    default_coverage=0
    ;;
  all)
    marker_expression="not manual and not gpu"
    default_coverage=1
    ;;
  manual)
    marker_expression="manual and not gpu"
    default_coverage=0
    ;;
  gpu)
    marker_expression="gpu and not manual"
    default_coverage=0
    ;;
  *)
    cat >&2 <<USAGE
Usage: scripts/test.sh [fast|integration|scientific|all|manual|gpu] [pytest arguments...]

The default lane is 'fast'. Set QLINKS_TEST_COVERAGE=0 or 1 to override the
lane's coverage default.
USAGE
    exit 2
    ;;
esac

if [[ $# -gt 0 ]]; then
  shift
fi

coverage_enabled="${QLINKS_TEST_COVERAGE:-${default_coverage}}"
pytest_args=(-m "${marker_expression}" --durations=20)

if [[ $# -eq 0 ]]; then
  pytest_args=(tests/ "${pytest_args[@]}")
fi

if [[ "${coverage_enabled}" == "1" ]]; then
  pytest_args+=(--cov=qlinks --cov-report=term-missing)
fi

poetry run pytest "${pytest_args[@]}" "$@"

if [[ "${coverage_enabled}" == "1" && "${QLINKS_COVERAGE_XML:-0}" == "1" ]]; then
  poetry run coverage xml
fi
