"""Repository conventions for Docker launcher freshness."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN_LAUNCHERS = (
    "scripts/docker/docker_run.sh",
    "scripts/docker/docker_run_jupyter.sh",
    "scripts/docker/docker_run_evidence_job.sh",
    "scripts/docker/docker_run_spin1_sec6_provisioning.sh",
    "scripts/docker/docker_run_spin1_sec6_integration.sh",
)


@pytest.mark.parametrize("relative_path", RUN_LAUNCHERS)
def test_docker_run_launchers_always_pull(relative_path: str) -> None:
    """Published-image launchers must refresh their selected tag before running."""
    script = (ROOT / relative_path).read_text(encoding="utf-8")
    assert "--pull always" in script


def test_spin1_sec6_integration_launcher_has_no_heavy_solver_entry_point() -> None:
    """The integration launcher must never dispatch the heavy provisioning job."""
    script = (ROOT / "scripts/docker/docker_run_spin1_sec6_integration.sh").read_text(
        encoding="utf-8"
    )
    assert "spin1_sec6_integration.py" in script
    assert "spin1_sec6_seed_dense_cache.py" in script
    assert "spin1_sec6_common_windows.py" in script
    assert "render_spin1_xy_sec6_integration_figures.py" in script
    assert "run_spin1_xy_sec6_provisioning.py" not in script
