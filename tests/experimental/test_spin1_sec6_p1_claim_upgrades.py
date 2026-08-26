"""Contracts for optional Spin-1 Sec. VI P1 claim-upgrade jobs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
RUNNER = ROOT / "scripts/docker/docker_run_spin1_sec6_p1.sh"
JACOBIAN = JOBS / "spin1_sec6_p1_jacobian_l8.py"
REFINEMENT = JOBS / "spin1_sec6_p1_kappa_refinement.py"
THREE_SITE = JOBS / "spin1_sec6_p1_three_site_concentration.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_three_site_charge_algebra_has_locked_dimension_and_is_orthonormal() -> None:
    module = _load(THREE_SITE, "spin1_sec6_p1_three_site_test")
    patterns, names, basis = module.charge_conserving_three_site_hermitian_basis()
    assert patterns.shape == (27, 3)
    assert len(names) == 141
    assert len(basis) == 141
    gram = np.asarray([[np.trace(left.conj().T @ right) for right in basis] for left in basis])
    np.testing.assert_allclose(gram, np.eye(141), atol=1.0e-10)


def test_p1_solver_boundaries_are_mechanical() -> None:
    refinement = REFINEMENT.read_text(encoding="utf-8")
    three_site = THREE_SITE.read_text(encoding="utf-8")
    jacobian = JACOBIAN.read_text(encoding="utf-8")

    assert "TARGET_LENGTHS = (8, 10, 12)" in refinement
    assert "MIDPOINT_KAPPA = (0.075, 0.125, 0.175)" in refinement
    assert "_compute_dense_point" in refinement
    assert "eigsh" not in refinement
    assert "_partial_spectrum" not in refinement

    assert "TARGET_LENGTHS = (8, 10, 12)" in three_site
    assert "LOCAL_ALGEBRA_DIMENSION = 141" in three_site
    assert "discover_checkpoint_directories" in three_site
    assert "eigsh" not in three_site
    assert "eigh(" not in three_site
    assert "_partial_spectrum" not in three_site

    assert "LENGTH = 8" in jacobian
    assert "eigsh" not in jacobian
    assert "eigh(" not in jacobian


def test_p1_runner_keeps_status_and_three_site_no_solve() -> None:
    script = RUNNER.read_text(encoding="utf-8")
    assert 'P0_RUN_ID="${QLINKS_SEC6_P0_RUN_ID:-spin1_sec6_integration_20260825T073925Z}"' in script
    assert '--volume "${REPO_ROOT}:${CONTAINER_REPO_DIR}:ro"' in script
    assert '--volume "${HOST_DATA_DIR}:${CONTAINER_DATA_DIR}"' in script
    assert "sparse/L14 forbidden" in script

    refinement_status = script.split("kappa-refinement-status)", maxsplit=1)[1].split(
        "kappa-refinement)", maxsplit=1
    )[0]
    refinement_compute = script.split("kappa-refinement)", maxsplit=1)[1].split(
        "three-site-status)", maxsplit=1
    )[0]
    three_site_status = script.split("three-site-status)", maxsplit=1)[1].split(
        "three-site)", maxsplit=1
    )[0]
    three_site_compute = script.split("three-site)", maxsplit=1)[1].split("status)", maxsplit=1)[0]

    assert "--compute-missing" not in refinement_status
    assert "--compute-missing" in refinement_compute
    assert "--compute-missing" not in three_site_status
    assert "--compute-missing" in three_site_compute
    assert "no eigensolver" in three_site_compute


def test_jacobian_job_explicitly_audits_the_l6_half_ring_collision() -> None:
    script = JACOBIAN.read_text(encoding="utf-8")
    assert "length=6" in script
    assert "distance=3" in script
    assert "L6_half_ring_collision" in script
