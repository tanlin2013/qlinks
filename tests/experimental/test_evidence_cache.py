from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
JOBS = ROOT / "experimental" / "jobs"
if str(JOBS) not in sys.path:
    sys.path.insert(0, str(JOBS))

from evidence_cache import (  # noqa: E402
    CacheValidationStatus,
    load_spectral_checkpoint,
    save_spectral_checkpoint,
    sparse_matrix_fingerprint,
    spectral_checkpoint_directory,
)
from qdm_checkerboard_large_strip import PartialSpectrum  # noqa: E402
from qdm_resumable_spectrum import make_resumable_folded_solver  # noqa: E402


def _diagonal_problem(matrix: sp.csr_array, target: float) -> dict[str, object]:
    return {
        "problem_family": "test",
        "sector_dimension": int(matrix.shape[0]),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "hamiltonian_fingerprint": sparse_matrix_fingerprint(matrix),
        "target_energy": float(target),
    }


def test_spectral_checkpoint_round_trip_and_problem_validation(tmp_path: Path) -> None:
    matrix = sp.csr_array(np.diag(np.arange(6.0)))
    problem = _diagonal_problem(matrix, 2.5)
    energies = np.array([2.0, 3.0])
    vectors = np.eye(6, dtype=np.complex128)[:, [2, 3]]
    residuals = np.zeros(2)
    directory = spectral_checkpoint_directory(
        namespace="test/spectrum",
        problem=problem,
        budget=2,
        backend="unit",
        cache_root=tmp_path,
    )
    save_spectral_checkpoint(
        directory,
        energies=energies,
        eigenvectors=vectors,
        residuals=residuals,
        transformed_residuals=np.zeros(2),
        problem=problem,
        backend="unit",
        requested_budget=2,
        solver_tolerance=1.0e-9,
    )

    loaded = load_spectral_checkpoint(
        directory,
        expected_problem=problem,
        hamiltonian=matrix,
        requested_solver_tolerance=1.0e-8,
    )
    assert loaded is not None
    assert loaded.status is CacheValidationStatus.VALID_FINAL
    assert loaded.validation["sample_maximum_residual"] == pytest.approx(0.0)

    incompatible_problem = {**problem, "target_energy": 2.75}
    incompatible = load_spectral_checkpoint(
        directory,
        expected_problem=incompatible_problem,
        hamiltonian=matrix,
    )
    assert incompatible is not None
    assert incompatible.status is CacheValidationStatus.INCOMPATIBLE


def test_spectral_checkpoint_can_be_warm_start_only(tmp_path: Path) -> None:
    matrix = sp.csr_array(np.diag(np.arange(4.0)))
    problem = _diagonal_problem(matrix, 1.5)
    directory = spectral_checkpoint_directory(
        namespace="test/spectrum",
        problem=problem,
        budget=2,
        backend="unit",
        cache_root=tmp_path,
    )
    vectors = np.eye(4, dtype=np.complex128)[:, [1, 2]]
    save_spectral_checkpoint(
        directory,
        energies=np.array([1.0, 2.0]),
        eigenvectors=vectors,
        residuals=np.array([1.0e-3, 1.0e-3]),
        transformed_residuals=None,
        problem=problem,
        backend="unit",
        requested_budget=2,
        solver_tolerance=1.0e-4,
    )

    loaded = load_spectral_checkpoint(
        directory,
        expected_problem=problem,
        requested_solver_tolerance=1.0e-8,
    )
    assert loaded is not None
    assert loaded.status is CacheValidationStatus.VALID_WARM_START


def test_qdm_wrapper_reuses_completed_arpack_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QLINKS_EVIDENCE_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("QLINKS_EVIDENCE_CACHE_RESUME", "1")
    monkeypatch.setenv("QLINKS_EVIDENCE_CACHE_WRITE", "1")
    monkeypatch.setenv("QLINKS_QDM_FOLDED_BACKEND", "arpack")
    calls = 0

    def fake_original(
        hamiltonian,
        *,
        target_energy,
        subspace_size,
        tolerance=1.0e-8,
        maxiter=None,
        ncv_factor=2.05,
        random_seed=20260811,
        cluster_tolerance=None,
    ):
        nonlocal calls
        calls += 1
        del maxiter, ncv_factor, random_seed, cluster_tolerance
        dense = np.asarray(sp.csr_array(hamiltonian).toarray())
        values, vectors = np.linalg.eigh(dense)
        order = np.argsort(np.abs(values - float(target_energy)))[: int(subspace_size)]
        values = values[order]
        vectors = vectors[:, order]
        residuals = np.linalg.norm(
            dense @ vectors - vectors * values[None, :], axis=0
        )
        return PartialSpectrum(
            energies=values,
            eigenvectors=vectors,
            residuals=residuals,
            sigma=float(target_energy),
            target_energy=float(target_energy),
            method="fake_arpack",
            requested_subspace_size=int(subspace_size),
            transformed_residuals=np.zeros(values.size),
            peak_rss_gib=0.0,
        )

    solver = make_resumable_folded_solver(fake_original)
    matrix = sp.csr_array(np.diag(np.arange(8.0, dtype=float)))
    first = solver(matrix, target_energy=3.5, subspace_size=4, tolerance=1.0e-8)
    second = solver(matrix, target_energy=3.5, subspace_size=4, tolerance=1.0e-8)

    assert calls == 1
    np.testing.assert_allclose(second.energies, first.energies)
    assert second.method == "folded_spectrum_cache_arpack"


@pytest.mark.integration
def test_primme_backend_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("primme")
    monkeypatch.setenv("QLINKS_EVIDENCE_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("QLINKS_EVIDENCE_CACHE_RESUME", "1")
    monkeypatch.setenv("QLINKS_EVIDENCE_CACHE_WRITE", "1")
    monkeypatch.setenv("QLINKS_QDM_FOLDED_BACKEND", "primme")
    monkeypatch.setenv("QLINKS_QDM_PRIMME_WARM_START_VECTORS", "2")

    def should_not_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("ARPACK fallback should not be used by explicit PRIMME backend")

    solver = make_resumable_folded_solver(should_not_run)
    matrix = sp.csr_array(np.diag(np.arange(12.0, dtype=float)))
    result = solver(matrix, target_energy=5.25, subspace_size=4, tolerance=1.0e-8)
    assert result.energies.size == 4
    assert result.method == "folded_spectrum_primme"
    np.testing.assert_allclose(
        np.sort(np.abs(result.energies - 5.25)),
        np.sort(np.array([0.25, 0.75, 1.25, 1.75])),
        atol=1.0e-6,
    )
