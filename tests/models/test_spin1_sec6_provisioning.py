from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experimental" / "jobs" / "run_spin1_xy_sec6_provisioning.py"


@pytest.mark.integration
def test_sec6_provisioning_runner_patches_smoke_notebook(tmp_path: Path) -> None:
    """Protect Sec. VI job/parameter wiring without starting a scientific solve."""

    result = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--profile",
            "smoke",
            "--stage",
            "compute",
            "--no-execute",
            "--data-dir",
            str(tmp_path),
            "--dense-sizes",
            "8",
            "--safe-fixed-widths",
            "1.0",
            "--skip-large-representative",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    patched = tmp_path / "run_artifacts" / "spin1_xy_sec6_provisioning_input.ipynb"
    metadata = tmp_path / "run_artifacts" / "run_metadata.json"
    assert patched.is_file()
    assert metadata.is_file()
    assert "Wrote patched input notebook" in result.stdout
    source = patched.read_text(encoding="utf-8")
    assert "DENSE_SIZES = (8,)" in source
    assert "SAFE_FIXED_HALF_WIDTHS = (1.0,)" in source
    assert "RUN_LARGE_REPRESENTATIVE = False" in source

    notebook = json.loads(source)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"{patched}:{cell['id']}", "exec")
