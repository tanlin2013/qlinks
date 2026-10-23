"""Inspect slow-container progress without launching or stopping a solver."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DIAGNOSE = ROOT / "scripts/docker/diagnose_qdm_primme.sh"


@pytest.mark.parametrize("native_stack", [False, True])
def test_diagnosis_preserves_native_events_and_never_changes_container_state(
    tmp_path, native_stack
):
    log = tmp_path / "solver.log"
    records = ["2026-10-23T01:00:00Z {'primme': 'starting'}", "CONVERGED 17 residual 1e-12"]
    for elapsed, outer, matvec in [(100, 10, 1000), (700, 12, 1002)]:
        record = {
            "primme": "matrix_callback_progress",
            "elapsed_seconds": elapsed,
            "stats_numOuterIterations": outer,
            "stats_numMatvecs": matvec,
            "stats_timeDense": 0.5,
        }
        records.append(f"2026-10-23T01:00:00Z {record!r}")
    log.write_text("\n".join(records) + "\n")
    docker = tmp_path / "docker"
    docker.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$1" >> "$DOCKER_CALLS"\n'
        'case "$1" in\n'
        '  inspect|stats) printf "%s\\n" "$1 snapshot" ;;\n'
        '  top) printf "PID COMMAND\\n12345 python\\n" ;;\n'
        '  logs) cat "$SOLVER_LOG" ;;\n'
        '  exec) cat >/dev/null; printf "/proc/12345/root/container/canonical-libraries\\n" ;;\n'
        "  *) exit 99 ;;\n"
        "esac\n"
    )
    debugger = tmp_path / "gdb"
    debugger.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$GDB_ARGS"\n')
    sudo = tmp_path / "sudo"
    sudo.write_text('#!/bin/bash\nexec "$@"\n')
    for executable in (docker, debugger, sudo):
        executable.chmod(0o755)
    calls = tmp_path / "docker-calls"
    gdb_args = tmp_path / "gdb-args"
    command = ["bash", str(DIAGNOSE), "current-container"]
    if native_stack:
        command.append("--native-stack")
    result = subprocess.run(
        command,
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "DOCKER_CALLS": str(calls),
            "SOLVER_LOG": str(log),
            "GDB_ARGS": str(gdb_args),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Seconds per outer iteration: 300.000" in result.stdout
    assert "stats_numMatvecs increase: 2" in result.stdout
    assert "CONVERGED 17 residual 1e-12" in result.stdout
    assert "not converged-eigenpair counts" in result.stdout
    assert "does not include projected eigensolves" in result.stdout
    expected_calls = {"inspect", "stats", "top", "logs"}
    if native_stack:
        expected_calls.add("exec")
    assert set(calls.read_text().splitlines()) == expected_calls
    assert gdb_args.exists() == native_stack
    if native_stack:
        args = gdb_args.read_text().splitlines()
        assert any(
            arg.startswith("set sysroot ") and arg.endswith(".no-host-libraries") for arg in args
        )
        assert "set solib-search-path /proc/12345/root/container/canonical-libraries" in args
        assert "file /proc/12345/exe" in args
        assert "attach 12345" in args
        assert args[-1] == "detach"
