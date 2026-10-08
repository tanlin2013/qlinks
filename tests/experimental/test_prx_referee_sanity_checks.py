from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "experimental" / "jobs" / "prx_referee_sanity_checks.py"
SPIN1 = ROOT / "experimental" / "jobs" / "prx_referee_sanity_spin1.py"
QDM = ROOT / "experimental" / "jobs" / "prx_referee_sanity_qdm.py"
RUNNER = ROOT / "scripts" / "docker" / "docker_run_prx_referee_sanity_checks.sh"


def test_referee_sanity_jobs_are_syntactically_valid_and_bounded() -> None:
    job = JOB.read_text(encoding="utf-8")
    spin1 = SPIN1.read_text(encoding="utf-8")
    qdm = QDM.read_text(encoding="utf-8")
    ast.parse(job)
    ast.parse(spin1)
    ast.parse(qdm)
    assert "DEFAULT_LENGTHS = (8, 10, 12)" in job
    assert "QDM_LX = 8" in qdm
    assert "QDM_REPEATS = 2" in qdm
    assert "eigsh(" not in job + spin1 + qdm
    assert "primme.eigsh" not in (job + spin1 + qdm).lower()
    assert "spin1_sec6_seed_dense_cache.py" in job


def test_required_deliverables_and_verdict_fields_are_frozen() -> None:
    source = JOB.read_text(encoding="utf-8") + SPIN1.read_text(encoding="utf-8")
    source += QDM.read_text(encoding="utf-8")
    for name in (
        "spin1_level_statistics.csv",
        "spin1_symmetry_class_audit.md",
        "spin1_level_statistics.pdf",
        "qdm_fig9a_branch_audit.csv",
        "qdm_fig9a_symmetry_audit.md",
        "qdm_fig9a_branch_diagnostic.pdf",
        "verdict.json",
    ):
        assert name in source
    for field in (
        "spin1_nonintegrability_check",
        "spin1_rmt_class",
        "qdm_hidden_symmetry_found",
        "qdm_fig9_recompute_required",
        "heavy_new_run_required",
    ):
        assert field in source


def test_spin1_statistics_require_complete_spectra() -> None:
    source = SPIN1.read_text(encoding="utf-8")
    assert "vectors.shape == (dimension, dimension)" in source
    assert "energies.size == dimension" in source
    assert "validate_cached_spectrum" in source
    assert "CACHE_MISSING_OR_INCOMPLETE" in source


def test_qdm_dense_fallback_is_restricted_to_8x4_small_sector() -> None:
    source = QDM.read_text(encoding="utf-8")
    assert "EXPECTED_SECTOR_DIMENSIONS[QDM_LX]" in source
    assert "la.eigh" in source
    assert "small_dense_fallback" in source
    assert "QDM_LX = 8" in source
    assert "QDM_LY = 4" in source


def test_docker_runner_mounts_only_data_writable() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "tanlin2013/qlinks:notebook" in source
    assert '${REPO_ROOT}:/workspace/qlinks:ro' in source
    data_mount = '${REPO_ROOT}/experimental/data:/workspace/qlinks/experimental/data'
    assert data_mount in source
    assert "MPLBACKEND=Agg" in source
    assert "prx_referee_sanity_checks.py" in source
