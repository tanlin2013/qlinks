#!/usr/bin/env bash
# Inspect a retained evidence container without changing its solver or cache.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || ( $# -eq 2 && "$2" != "--native-stack" ) ]]; then
    echo "Usage: $0 CONTAINER [--native-stack]" >&2
    exit 2
fi
CONTAINER="$1"
command -v docker >/dev/null
command -v python3 >/dev/null
LOG_FILE="$(mktemp)"
trap 'rm -f -- "${LOG_FILE}"' EXIT

printf 'Captured: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
docker inspect --format \
  'Container={{.Name}} Image={{.Config.Image}} ImageID={{.Image}} State={{json .State}}' \
  "${CONTAINER}"
docker stats --no-stream "${CONTAINER}" || true
docker top "${CONTAINER}" -eo pid,etime,time,pcpu,pmem,args || true
if ! docker logs --timestamps --tail 5000 "${CONTAINER}" > "${LOG_FILE}" 2>&1; then
    cat "${LOG_FILE}" >&2
    exit 1
fi

python3 - "${LOG_FILE}" <<'PY'
import ast
import sys
from pathlib import Path

lines = Path(sys.argv[1]).read_text(errors="replace").splitlines()
samples = []
other = []
for line in lines:
    if "'primme': 'matrix_callback_progress'" not in line:
        other.append(line)
        continue
    try:
        record = ast.literal_eval(line[line.index("{"):])
        if isinstance(record, dict) and isinstance(record.get("elapsed_seconds"), (int, float)):
            samples.append(record)
    except (SyntaxError, ValueError):
        pass

print("\nProgress summary (from at most the last 5000 log lines):")
if samples:
    last = samples[-1]
    print(f"Latest elapsed hours: {last['elapsed_seconds'] / 3600:.3f}")
    for key, value in last.items():
        if key.startswith("stats_"):
            print(f"{key}: {value}")
    recent = [s for s in samples if s["elapsed_seconds"] >= last["elapsed_seconds"] - 6 * 3600]
    if len(recent) < 2:
        recent = samples[-2:]
    if len(recent) >= 2:
        first = recent[0]
        elapsed = last["elapsed_seconds"] - first["elapsed_seconds"]
        print(f"Sample interval hours: {elapsed / 3600:.3f}")
        for field in ("stats_numOuterIterations", "stats_numMatvecs"):
            start, end = first.get(field), last.get(field)
            if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                delta = end - start
                print(f"{field} increase: {delta}")
                if field == "stats_numOuterIterations" and delta > 0:
                    print(f"Seconds per outer iteration: {elapsed / delta:.3f}")
    print("Outer iterations and matrix products are not converged-eigenpair counts.")
    print("PRIMME 3.2.3 timeDense does not include projected eigensolves; these timers are incomplete.")
else:
    print("No parseable matrix-callback progress samples in the captured log tail.")

print("\nLast 80 log lines excluding matrix-callback progress:")
print("\n".join(other[-80:]))
PY

if [[ "${2:-}" == "--native-stack" ]]; then
    command -v gdb >/dev/null || { echo "Host gdb is required for --native-stack." >&2; exit 2; }
    PYTHON_PID="$(docker top "${CONTAINER}" -eo pid,comm |
      awk '$2 ~ /^python/ {print $1; exit}')"
    [[ "${PYTHON_PID}" =~ ^[0-9]+$ ]] || {
        echo "No running Python process found; no debugger was attached." >&2
        exit 2
    }
    DEBUGGER=(gdb)
    if (( EUID != 0 )); then DEBUGGER=(sudo gdb); fi
    echo "Collecting a native stack; GDB briefly stops Python and detaches to resume it."
    "${DEBUGGER[@]}" -q -nx -batch \
      -ex 'set pagination off' \
      -ex "set sysroot /proc/${PYTHON_PID}/root" \
      -ex "attach ${PYTHON_PID}" \
      -ex 'thread apply all bt 20' \
      -ex detach
fi
