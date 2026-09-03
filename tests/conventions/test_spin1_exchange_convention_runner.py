"""Static contracts for the one-time Spin-1 convention migration runner."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/docker/docker_run_spin1_exchange_convention_migration.sh"
VALIDATOR = ROOT / "experimental/jobs/spin1_exchange_convention_validate.py"
RENDER_P0 = ROOT / "experimental/jobs/spin1_exchange_convention_render_p0.py"


def test_runner_uses_immutable_authoritative_p0_p1_sources() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    assert "spin1_sec6_integration_20260825T073925Z" in script
    assert "spin1_sec6_p1_claim_upgrades_20260827T055013Z" in script
    assert "spin1_exchange_convention_migrate_evidence.py" in script
    assert "--replace-derived" not in script


def test_runner_exposes_only_bounded_migration_validation_stages() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    for stage in ("status", "migrate-p0", "migrate-p1", "validate", "jacobian-l8", "render-p0"):
        assert stage in script
    assert "eigsh" not in script
    assert "L14 eigensolve is available" not in script
    assert "No L=14 eigensolve is available" in script


def test_render_stage_prepares_standardized_figure_data_first() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    helper = RENDER_P0.read_text(encoding="utf-8")
    assert "spin1_exchange_convention_render_p0.py" in script
    integration_index = helper.index("integration.run_integration(data, data)")
    renderer_index = helper.index("renderer.render(")
    assert integration_index < renderer_index


def test_validation_job_has_no_large_size_solver_route() -> None:
    script = VALIDATOR.read_text(encoding="utf-8")
    assert "scipy.linalg" in script
    assert "eigsh" not in script
    assert 'default="8"' in script
    assert "--dense-sizes" in script
    assert "L=10 on the server" in script


def test_runner_mounts_repository_read_only_and_data_read_write() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    assert '--volume "${REPO_ROOT}:${CONTAINER_REPO_DIR}:ro"' in script
    assert '--volume "${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}"' in script
