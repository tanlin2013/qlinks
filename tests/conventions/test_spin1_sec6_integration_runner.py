"""Static contracts for the Spin-1 Sec. VI Docker handoff."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/docker/docker_run_spin1_sec6_integration.sh"
SEEDER = ROOT / "experimental/jobs/spin1_sec6_seed_dense_cache.py"
DEFORMATION_GRID = ROOT / "experimental/jobs/spin1_sec6_deformation_grid.py"
DEFORMATION_GRID_LEGACY = ROOT / "experimental/jobs/spin1_sec6_deformation_grid_legacy.py"


def test_runner_uses_unique_stage_attempt_containers_and_shared_output() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    assert (
        'CONTAINER_NAME="${QLINKS_CONTAINER_NAME:-qlinks-${RUN_ID//_/-}-${STAGE}-${RUN_TIMESTAMP}-$$}"'
        in script
    )
    assert 'OUTPUT_DATA_DIR="${CONTAINER_DATA_DIR}/evidence_jobs/${RUN_ID}"' in script


def test_runner_mounts_repository_read_only_but_data_read_write() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    assert '--volume "${REPO_ROOT}:${CONTAINER_REPO_DIR}:ro"' in script
    assert '--volume "${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}"' in script


def test_runner_exposes_only_reviewed_integration_stages() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    for stage in (
        "audit",
        "seed-dense-cache",
        "common-windows",
        "deformation-grid-status",
        "deformation-grid",
        "render-preview",
        "render-final",
    ):
        assert stage in script
    assert "run_spin1_xy_sec6_provisioning.py" not in script


def test_common_windows_uses_established_sparse_certification() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    assert "spin1_sec6_common_windows_certified.py" in script
    assert '--source-data-dir "${SOURCE_DATA_DIR}"' in script


def test_audit_refreshes_the_successful_integration_directory() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    assert "spin1_sec6_refresh_integration_audit.py" in script
    assert '--integration-data-dir "${OUTPUT_DATA_DIR}"' in script


def test_deformation_grid_has_no_sparse_or_large_size_solver_route() -> None:
    adapter = DEFORMATION_GRID.read_text(encoding="utf-8")
    kernel = DEFORMATION_GRID_LEGACY.read_text(encoding="utf-8")
    combined = adapter + "\n" + kernel
    assert "TARGET_LENGTHS = (8, 10, 12)" in kernel
    assert "KAPPA_GRID = (0.05, 0.10, 0.15, 0.20)" in kernel
    assert "REPRESENTATIVE_KAPPA_OVER_J = 0.10" in kernel
    assert "la.eigh" in kernel
    assert "eigsh" not in combined
    assert "_partial_spectrum" not in combined
    target_line = kernel.split("TARGET_LENGTHS =", maxsplit=1)[1].splitlines()[0]
    assert "14" not in target_line


def test_deformation_status_stage_does_not_enable_compute_missing() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    status_block = script.split("deformation-grid-status)", maxsplit=1)[1].split(
        "deformation-grid)", maxsplit=1
    )[0]
    compute_block = script.split("deformation-grid)", maxsplit=1)[1].split(
        "render-preview|render-final)", maxsplit=1
    )[0]
    assert "--compute-missing" not in status_block
    assert "--compute-missing" in compute_block
    assert "sparse/L14 forbidden" in compute_block


def test_dense_cache_seed_is_strictly_small_size_only() -> None:
    script = SEEDER.read_text(encoding="utf-8")
    assert "TARGET_LENGTHS = (8, 10, 12)" in script
    assert "KAPPA_OVER_J = 0.10" in script
    assert "la.eigh" in script
    assert "eigsh" not in script
    assert "_partial_spectrum" not in script
