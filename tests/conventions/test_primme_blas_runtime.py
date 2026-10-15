"""Reject reference BLAS despite unrelated optimized libraries being present."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VERIFY = ROOT / "scripts/docker/verify_primme_blas_runtime.py"


@pytest.mark.parametrize("provider", ["openblas", "reference", "mixed", "missing"])
def test_primme_dependency_audit_resolves_actual_providers(tmp_path, provider):
    (tmp_path / "primme.py").write_text(
        "# Dependency resolution is supplied by the ldd stand-in.\n"
    )
    libraries = tmp_path / ("openblas-pthread" if provider == "openblas" else "reference")
    libraries.mkdir()
    for name in ("libblas.so.3", "liblapack.so.3"):
        (libraries / name).touch()
    lines = [
        f"libblas.so.3 => {libraries / 'libblas.so.3'} (0x1)",
        f"liblapack.so.3 => {libraries / 'liblapack.so.3'} (0x2)",
    ]
    if provider == "missing":
        lines[1] = "liblapack.so.3 => not found"
    if provider == "mixed":
        optimized = tmp_path / "openblas-pthread"
        optimized.mkdir()
        (optimized / "libopenblas.so.0").touch()
        lines.append(f"libopenblas.so.0 => {optimized / 'libopenblas.so.0'} (0x3)")
    ldd = tmp_path / "ldd"
    ldd.write_text("#!/bin/sh\ncat <<'EOF'\n" + "\n".join(lines) + "\nEOF\n")
    ldd.chmod(0o755)
    env = dict(os.environ, PYTHONPATH=str(tmp_path), PATH=f"{tmp_path}:{os.environ['PATH']}")
    result = subprocess.run(
        [sys.executable, str(VERIFY), "--require-openblas"],
        env=env,
        capture_output=True,
        text=True,
    )
    if provider == "openblas":
        assert result.returncode == 0
        report = json.loads(result.stdout)
        assert report["all_providers_openblas"]
        assert len(report["blas_lapack_providers"]) == 2
    else:
        assert result.returncode != 0
        assert "OpenBLAS" in result.stderr or "unresolved" in result.stderr
