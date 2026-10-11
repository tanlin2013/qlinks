"""Tiny invariants for folded-subspace reconstruction and immutable recovery."""

from __future__ import annotations

import importlib.machinery
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
for directory in (ROOT / "experimental/jobs", ROOT / "experimental/notebooks"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from evidence_cache import save_spectral_checkpoint, spectral_checkpoint_directory  # noqa: E402
from qdm_resumable_spectrum import (  # noqa: E402
    CACHE_NAMESPACE,
    folded_problem_description,
    make_resumable_folded_solver,
    reconstruct_cached_spectrum,
)
from qdm_ritz_reconstruction import (  # noqa: E402
    REFINED_BACKEND,
    reconstruct_hamiltonian_ritz_pairs,
)


def _mixed_pair():
    h = sp.diags_array([11.8, 12.200004, 20.0, 21.0, 22.0, 23.0], format="csr")
    vectors = np.zeros((6, 2), dtype=np.complex128)
    vectors[:2] = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    return h, vectors


@pytest.mark.parametrize("nonorthogonal", [False, True])
def test_near_folded_pair_is_resolved_across_the_complete_span(nonorthogonal):
    h, vectors = _mixed_pair()
    if nonorthogonal:
        vectors = vectors @ np.array([[1.0, 0.2j], [0.0, 1.5]])
    partial, details = reconstruct_hamiltonian_ritz_pairs(
        h,
        vectors,
        target_energy=12,
        requested_subspace_size=2,
        block_size=1,
    )
    np.testing.assert_allclose(partial.energies, [11.8, 12.200004], atol=1e-12)
    np.testing.assert_allclose(
        partial.eigenvectors.conj().T @ partial.eigenvectors, np.eye(2), atol=1e-12
    )
    assert partial.maximum_residual < 1e-12
    assert np.max(partial.transformed_residuals) < 1e-12
    assert details["full_orthogonality_frobenius_residual"] < 1e-12


def test_reconstruction_does_not_hide_a_missing_partner():
    h, vectors = _mixed_pair()
    partial, _ = reconstruct_hamiltonian_ritz_pairs(
        h,
        vectors[:, :1],
        target_energy=12,
        requested_subspace_size=1,
    )
    assert partial.maximum_residual > 0.19


@pytest.mark.parametrize("bad_input", ["rank_deficient", "nonfinite"])
def test_unusable_subspace_fails_explicitly(bad_input):
    h, vectors = _mixed_pair()
    if bad_input == "rank_deficient":
        vectors[:, 1] = 0
        error, message = RuntimeError, "rank deficient"
    else:
        vectors[0, 0] = np.nan
        error, message = ValueError, "nonfinite"
    with pytest.raises(error, match=message):
        reconstruct_hamiltonian_ritz_pairs(h, vectors, target_energy=12, requested_subspace_size=2)


def _save_original(root, h, vectors, *, budget=2):
    problem = folded_problem_description(h, target_energy=12)
    source = spectral_checkpoint_directory(
        namespace=CACHE_NAMESPACE,
        problem=problem,
        budget=budget,
        backend="primme",
        cache_root=root,
    )
    energies = np.full(vectors.shape[1], 12.000002)
    residuals = np.linalg.norm(h @ vectors - vectors * energies, axis=0)
    save_spectral_checkpoint(
        source,
        energies=energies,
        eigenvectors=vectors,
        residuals=residuals,
        transformed_residuals=np.full(vectors.shape[1], 1e-5),
        problem=problem,
        backend="primme",
        requested_budget=budget,
        solver_tolerance=1e-8,
    )
    return source


def test_cached_recovery_preserves_source_bytes_and_is_repeatable(tmp_path):
    h, vectors = _mixed_pair()
    source = _save_original(tmp_path, h, vectors)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    partial, details = reconstruct_cached_spectrum(
        h, target_energy=12, budget=2, cache_root=tmp_path
    )
    assert partial.maximum_residual < 1e-12
    assert details["source_maximum_H_residual"] > 0.19
    assert details["source_checkpoint_metadata"] == json.loads(before["metadata.json"])
    assert {path.name: path.read_bytes() for path in source.iterdir()} == before
    second, second_details = reconstruct_cached_spectrum(
        h, target_energy=12, budget=2, cache_root=tmp_path
    )
    np.testing.assert_allclose(second.energies, partial.energies)
    assert second_details == details
    assert REFINED_BACKEND in second.method


def test_cached_recovery_rejects_a_different_hamiltonian(tmp_path):
    h, vectors = _mixed_pair()
    _save_original(tmp_path, h, vectors)
    changed = h.copy()
    changed.data[0] += 0.1
    with pytest.raises(RuntimeError, match="no compatible original"):
        reconstruct_cached_spectrum(changed, target_energy=12, budget=2, cache_root=tmp_path)


@pytest.mark.integration
def test_observable_loader_accepts_the_reconstructed_backend(tmp_path):
    from qdm_sec7_fixed_o1_l12_observables import _load_covered_checkpoints

    h = sp.diags_array([11.4, 11.8, 12.200004, 12.6, 20.0, 21.0], format="csr")
    vectors = np.eye(6, dtype=np.complex128)[:, :4]
    vectors[:, 1:3] = vectors[:, 1:3] @ np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    _save_original(tmp_path, h, vectors, budget=4)
    reconstruct_cached_spectrum(h, target_energy=12, budget=4, cache_root=tmp_path)
    context = SimpleNamespace(h_sector=h, tower_energy=12)
    covered = _load_covered_checkpoints(
        context=context,
        cache_root=tmp_path,
        half_width=0.5,
        residual_tolerance=1e-6,
    )
    assert len(covered) == 1
    assert covered[0][1].metadata["backend"] == REFINED_BACKEND
    assert covered[0][2]["window_state_count"] == 2


def test_primme_wrapper_reconstructs_then_reuses_the_versioned_cache(tmp_path, monkeypatch):
    h, vectors = _mixed_pair()
    calls = []

    def fake_eigsh(operator, **kwargs):
        calls.append(kwargs)
        return np.array([0.04, 0.200004**2]), vectors, {}

    fake = SimpleNamespace(
        eigsh=fake_eigsh, __spec__=importlib.machinery.ModuleSpec("primme", loader=None)
    )
    monkeypatch.setitem(sys.modules, "primme", fake)
    monkeypatch.setenv("QLINKS_QDM_FOLDED_BACKEND", "primme")
    monkeypatch.setenv("QLINKS_EVIDENCE_CACHE_ROOT", str(tmp_path))
    monkeypatch.setenv("QLINKS_EVIDENCE_CACHE_FORCE_RECOMPUTE", "0")

    def forbidden_original(*args, **kwargs):
        raise AssertionError("iterative fallback must not be called")

    solver = make_resumable_folded_solver(forbidden_original)
    first = solver(h, target_energy=12, subspace_size=2)
    second = solver(h, target_energy=12, subspace_size=2)
    assert len(calls) == 1
    assert first.maximum_residual < 1e-12
    assert REFINED_BACKEND in second.method


@pytest.mark.integration
@pytest.mark.parametrize("missing_partner", [False, True])
def test_recovery_stage_reports_quality_without_closing_one_budget(
    tmp_path, monkeypatch, missing_partner
):
    import qdm_sec7_fixed_o1_l12_spectrum as stage
    from qdm_parity_protocol import PROTOCOL

    h, vectors = _mixed_pair()
    if missing_partner:
        vectors = vectors[:, :1]
    cache = tmp_path / "cache"
    source = _save_original(cache, h, vectors, budget=768)
    output = tmp_path / "output"
    output.mkdir()
    (output / stage.RECOMMENDATION_NAME).write_text(
        json.dumps(
            {
                "status": "recommended",
                "recommended_half_width": 0.25,
                "neighbor_controls": [0.5],
                "symmetry_protocol": PROTOCOL,
                "parity_policy": "measured_target",
            }
        )
    )
    context = SimpleNamespace(
        h_sector=h,
        tower_energy=12,
        lx=12,
        phase=0.05,
        sector=SimpleNamespace(sector_dimension=6, labels={"Sy_character": -1}),
    )
    monkeypatch.setattr(stage, "recover_reference_geometry", lambda: None)
    monkeypatch.setattr(stage, "build_context", lambda **kwargs: context)

    def forbidden_solver(*args, **kwargs):
        raise AssertionError("recovery must not start an iterative solver")

    monkeypatch.setattr(stage, "make_resumable_folded_solver", lambda original: forbidden_solver)
    # Preserve the failed cluster reconstruction row when replacing its diagnostics.
    import pandas as pd

    pd.DataFrame(
        [
            {
                "requested_subspace_size": 768,
                "window_half_width": 0.5,
                "window_state_count": vectors.shape[1],
                "window_coverage_complete": False,
                "window_maximum_residual": 0.2,
                "symmetry_protocol": PROTOCOL,
            }
        ]
    ).to_csv(output / stage.CONVERGENCE_NAME, index=False)
    (output / stage.OBSERVABLES_ACCEPTANCE_NAME).write_text(json.dumps({"closed": True}))
    kwargs = dict(
        output_dir=output,
        cache_root=cache,
        budgets_raw="768",
        tolerance=1e-8,
        residual_tolerance=1e-6,
        max_budget=8192,
        sy_character="target",
        reconstruct_cached_only=True,
    )
    if missing_partner:
        with pytest.raises(RuntimeError, match="still fails H-residual"):
            stage.run(**kwargs)
    else:
        stage.run(**kwargs)
    report = json.loads((output / stage.RECONSTRUCTION_ACCEPTANCE_NAME).read_text())
    assert report["closed"] is not missing_partner
    assert report["spectral_acceptance_closed"] is False
    assert report["iterative_solver_called"] is False
    stale = json.loads((output / stage.OBSERVABLES_ACCEPTANCE_NAME).read_text())
    assert stale["closed"] is False
    original_report = output / "reconstruction_source_reports" / stage.OBSERVABLES_ACCEPTANCE_NAME
    assert json.loads(original_report.read_text())["closed"] is True
    assert source.exists()
    archived = json.loads((output / "reconstruction_source_rows/budget_00000768.json").read_text())
    assert archived["window_maximum_residual"] == 0.2
