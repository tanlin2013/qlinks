"""Repository conventions for Docker launcher freshness."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUN_LAUNCHERS = (
    "scripts/docker/docker_run.sh",
    "scripts/docker/docker_run_jupyter.sh",
    "scripts/docker/docker_run_evidence_job.sh",
    "scripts/docker/docker_run_spin1_sec6_provisioning.sh",
)


@pytest.mark.parametrize("relative_path", RUN_LAUNCHERS)
def test_docker_run_launchers_always_pull(relative_path: str) -> None:
    """Published-image launchers must refresh their selected tag before running."""
    script = (ROOT / relative_path).read_text(encoding="utf-8")
    assert "--pull always" in script
