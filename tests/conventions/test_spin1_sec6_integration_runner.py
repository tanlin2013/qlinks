"""Static contracts for the cache-only Spin-1 Sec. VI Docker handoff."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/docker/docker_run_spin1_sec6_integration.sh"


def test_runner_uses_stage_specific_containers_and_shared_output() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    assert 'CONTAINER_NAME="${QLINKS_CONTAINER_NAME:-qlinks-${RUN_ID//_/-}-${STAGE}}"' in script
    assert 'OUTPUT_DATA_DIR="${CONTAINER_DATA_DIR}/evidence_jobs/${RUN_ID}"' in script


def test_runner_mounts_repository_read_only_but_data_read_write() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    assert '--volume "${REPO_ROOT}:${CONTAINER_REPO_DIR}:ro"' in script
    assert '--volume "${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}"' in script


def test_runner_exposes_only_cache_postprocessing_stages() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    for stage in ("audit", "common-windows", "render-preview", "render-final"):
        assert stage in script
    assert "run_spin1_xy_sec6_provisioning.py" not in script
