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
    is_primme_checkpoint_backend,
    reconstruct_hamiltonian_ritz_pairs,
    refinement_backend,
)


def _mixed_pair():
    h = sp.diags_array([11.8, 12.200004, 20.0, 21.0, 22.0, 23.0], format="csr")
    vectors = np.zeros((6, 2), dtype=np.complex128)
    vectors[:2] = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    return h, vectors


@pytest.mark.parametrize("interval", ["0", "1e-12"])
@pytest.mark.parametrize("tuned", [False, True])
def test_solver_reports_progress_without_changing_the_spectral_problem(
    tmp_path, monkeypatch, capsys, interval, tuned
):
    h, _ = _mixed_pair()

    def fake_eigsh(operator, **kwargs):
        assert kwargs["printLevel"] == 2
        assert kwargs["tol"] == 1e-11
        assert "monitor" not in kwargs and "return_history" not in kwargs
        if tuned:
            assert kwargs["method"] == "PRIMME_GD_plusK"
            assert kwargs["maxBlockSize"] == 2
            assert kwargs["minRestartSize"] == 3
            assert kwargs["maxPrevRetain"] == 1
        vectors = np.eye(6, dtype=complex)[:, :2]
        action = operator.matmat(vectors)
        np.testing.assert_allclose(action, vectors * np.array([0.2**2, 0.200004**2]))
        return np.array([0.2**2, 0.200004**2]), vectors, {}

    monkeypatch.setitem(
        sys.modules,
        "primme",
        SimpleNamespace(
            eigsh=fake_eigsh,
            get_eigsh_param=lambda field: 12 if field == "stats_numMatvecs" else 3,
            __spec__=importlib.machinery.ModuleSpec("primme", loader=None),
        ),
    )
    for name, value in {
        "QLINKS_QDM_FOLDED_BACKEND": "primme",
        "QLINKS_EVIDENCE_CACHE_ROOT": str(tmp_path),
        "QLINKS_EVIDENCE_CACHE_RESUME": "0",
        "QLINKS_EVIDENCE_CACHE_WRITE": "0",
        "QLINKS_QDM_PRIMME_PROGRESS_SECONDS": interval,
        "QLINKS_QDM_PRIMME_PRINT_LEVEL": "2",
    }.items():
        monkeypatch.setenv(name, value)
    for name, value in {
        "QLINKS_QDM_PRIMME_METHOD": "PRIMME_GD_plusK",
        "QLINKS_QDM_PRIMME_MAX_BLOCK_SIZE": "2",
        "QLINKS_QDM_PRIMME_MIN_RESTART_SIZE": "3",
        "QLINKS_QDM_PRIMME_MAX_PREV_RETAIN": "1",
    }.items():
        if tuned:
            monkeypatch.setenv(name, value)
        else:
            monkeypatch.delenv(name, raising=False)
    result = make_resumable_folded_solver(lambda *a, **kw: pytest.fail("cold fallback"))(
        h, target_energy=12, subspace_size=2, tolerance=1e-11
    )
    np.testing.assert_allclose(result.energies, [11.8, 12.200004])
    assert result.maximum_residual < 1e-12
    output = capsys.readouterr().out
    assert "'primme': 'starting'" in output and "'primme': 'returned'" in output
    assert f"'max_block_size': {2 if tuned else None}" in output
    assert ("matrix_callback_progress" in output) == (interval != "0")
    if interval != "0":
        assert "'stats_numMatvecs': 12" in output
        assert "stats_timeDense" in output


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


@pytest.mark.integration
def test_physical_comparison_marks_the_tiny_parity_irrep_as_a_dense_symmetry_check(
    tmp_path,
    monkeypatch,
):
    import qdm_parity_backend_worker as worker
    from qdm_folded_backend_comparison import _load_convergence, compare_frames

    context = SimpleNamespace(
        h_sector=sp.csr_array([[4.0]]),
        tower_energy=4,
        lx=4,
        sector=SimpleNamespace(sector_dimension=1, labels={"Sy_character": -1}),
    )

    def build(**kwargs):
        assert kwargs["sy_character"] == "target"
        return context

    monkeypatch.setattr(worker, "recover_reference_geometry", lambda: None)
    monkeypatch.setattr(worker, "build_context", build)
    # The production worker is process-isolated; restore its cache environment
    # when exercising it in the same interpreter as the other cache tests.
    for name, value in {
        "QLINKS_QDM_FOLDED_BACKEND": "arpack",
        "QLINKS_EVIDENCE_CACHE_ROOT": str(tmp_path / "cache"),
        "QLINKS_EVIDENCE_CACHE_RESUME": "1",
        "QLINKS_EVIDENCE_CACHE_FORCE_RECOMPUTE": "0",
        "QLINKS_EVIDENCE_CACHE_WRITE": "1",
    }.items():
        monkeypatch.setenv(name, value)
    output = tmp_path / "arpack"
    frame = worker.compare_backend(
        "arpack",
        output=output,
        cache_root=tmp_path / "cache",
        repeats=[1],
        budgets=[16, 32],
        tolerance=1e-8,
    )
    assert frame.comparison_role.eq("dense_symmetry_check").all()
    assert frame.returned_eigenpairs.eq(1).all()
    assert frame.Sy.eq(-1).all()
    loaded = _load_convergence(output, "arpack")
    assert len(compare_frames(loaded, loaded.copy())) == 2
    invalid = frame.assign(sector_dimension=875)
    invalid.to_csv(output / "qdm_checkerboard_L12_spectral_convergence.csv", index=False)
    with pytest.raises(RuntimeError, match="dense fallback"):
        _load_convergence(output, "arpack")


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


@pytest.mark.parametrize("basis_override", [None, 3])
def test_same_budget_refinement_uses_all_vectors_and_preserves_both_sources(
    tmp_path, monkeypatch, basis_override
):
    h, vectors = _mixed_pair()
    h = h.astype(np.complex128)
    vectors[2, 0] = 1e-4  # A sane Gram matrix with genuine off-subspace error.
    raw = _save_original(tmp_path, h, vectors)
    reconstructed, _ = reconstruct_cached_spectrum(
        h, target_energy=12, budget=2, cache_root=tmp_path
    )
    assert reconstructed.maximum_residual > 1e-6
    old = raw.parent.parent / REFINED_BACKEND / raw.name
    preserved = {path: path.read_bytes() for root in (raw, old) for path in root.iterdir()}
    calls = []

    def fake_eigsh(operator, **kwargs):
        calls.append(kwargs)
        assert kwargs["v0"].shape == (6, 2)
        assert kwargs["tol"] == 1e-11
        if basis_override:
            assert kwargs["ncv"] == basis_override
        else:
            assert kwargs["ncv"] >= 4
        assert "lock" not in kwargs
        return np.array([0.04, 0.200004**2]), np.eye(6, dtype=complex)[:, :2], {}

    monkeypatch.setitem(
        sys.modules,
        "primme",
        SimpleNamespace(
            eigsh=fake_eigsh,
            __spec__=importlib.machinery.ModuleSpec("primme", loader=None),
        ),
    )
    for name, value in {
        "QLINKS_QDM_FOLDED_BACKEND": "primme",
        "QLINKS_EVIDENCE_CACHE_ROOT": str(tmp_path),
        "QLINKS_EVIDENCE_CACHE_RESUME": "1",
        "QLINKS_EVIDENCE_CACHE_FORCE_RECOMPUTE": "0",
        "QLINKS_EVIDENCE_CACHE_WRITE": "1",
        "QLINKS_QDM_PRIMME_WARM_START_VECTORS": "1",
    }.items():
        monkeypatch.setenv(name, value)
    if basis_override:
        monkeypatch.setenv("QLINKS_QDM_PRIMME_MAX_BASIS_SIZE", str(basis_override))
    else:
        monkeypatch.delenv("QLINKS_QDM_PRIMME_MAX_BASIS_SIZE", raising=False)
    solver = make_resumable_folded_solver(
        lambda *args, **kwargs: pytest.fail("cold ARPACK fallback")
    )
    refined = solver(
        h, target_energy=12, subspace_size=2, tolerance=1e-11, refine_existing_budget=True
    )
    assert refined.maximum_residual < 1e-12
    assert refined.full_orthogonality_residual < 1e-12
    assert all(path.read_bytes() == content for path, content in preserved.items())
    destination = raw.parent.parent / refinement_backend(1e-11) / raw.name
    metadata = json.loads((destination / "metadata.json").read_text())
    assert metadata["solver"]["warm_start_vectors"] == 2  # Overrides the ordinary cap of one.
    assert metadata["solver"]["same_budget_warm_start"]
    if basis_override:
        assert metadata["solver"]["max_basis_size"] == basis_override
    cached = solver(
        h, target_energy=12, subspace_size=2, tolerance=1e-11, refine_existing_budget=True
    )
    assert len(calls) == 1
    assert refinement_backend(1e-11) in cached.method
    assert cached.full_orthogonality_residual < 1e-12
    assert is_primme_checkpoint_backend(metadata["backend"])
    assert refinement_backend(1e-11) != refinement_backend(1.5e-11)


@pytest.mark.parametrize("fault", ["missing", "bad_gram", "incomplete", "wrong_problem"])
def test_refinement_without_a_suitable_full_checkpoint_never_starts_a_cold_solve(
    tmp_path,
    monkeypatch,
    fault,
):
    h, vectors = _mixed_pair()
    if fault != "missing":
        if fault == "incomplete":
            vectors = vectors[:, :1]
        raw = _save_original(tmp_path, h, vectors)
        reconstruct_cached_spectrum(h, target_energy=12, budget=2, cache_root=tmp_path)
        if fault == "bad_gram":
            path = raw.parent.parent / REFINED_BACKEND / raw.name / "metadata.json"
            metadata = json.loads(path.read_text())
            metadata["solver"]["full_orthogonality_frobenius_residual"] = 1.0
            path.write_text(json.dumps(metadata))
        if fault == "wrong_problem":
            h.data[0] += 0.1

    def forbidden(*args, **kwargs):
        pytest.fail("no iterative solve may start without the full compatible checkpoint")

    monkeypatch.setitem(
        sys.modules,
        "primme",
        SimpleNamespace(
            eigsh=forbidden,
            __spec__=importlib.machinery.ModuleSpec("primme", loader=None),
        ),
    )
    for name, value in {
        "QLINKS_QDM_FOLDED_BACKEND": "primme",
        "QLINKS_EVIDENCE_CACHE_ROOT": str(tmp_path),
        "QLINKS_EVIDENCE_CACHE_RESUME": "1",
        "QLINKS_EVIDENCE_CACHE_FORCE_RECOMPUTE": "0",
    }.items():
        monkeypatch.setenv(name, value)
    solver = make_resumable_folded_solver(forbidden)
    with pytest.raises(RuntimeError, match="no cold iterative solve"):
        solver(h, target_energy=12, subspace_size=2, tolerance=1e-11, refine_existing_budget=True)


@pytest.mark.integration
@pytest.mark.parametrize("refinement", [False, True])
@pytest.mark.parametrize("missing_partner", [False, True])
def test_recovery_stage_reports_quality_without_closing_one_budget(
    tmp_path, monkeypatch, missing_partner, refinement
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

    calls = []

    def refined_solver(*args, **kwargs):
        # This test isolates workflow state; numerical refinement is tested
        # separately on the six-dimensional leaking subspace above.
        assert kwargs["refine_existing_budget"]
        calls.append(kwargs)
        count = kwargs["subspace_size"]
        return SimpleNamespace(
            energies=np.linspace(11.4, 12.6, count),
            residuals=np.full(count, 0.2 if missing_partner else 1e-12),
            method="folded_spectrum_primme_full_ritz",
            full_orthogonality_residual=1e-13,
            transformed_residuals=np.full(count, 1e-13),
            peak_rss_gib=0.1,
            maximum_residual=0.2 if missing_partner else 1e-12,
        )

    monkeypatch.setattr(
        stage,
        "make_resumable_folded_solver",
        lambda original: refined_solver if refinement else forbidden_solver,
    )
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
        tolerance=1e-11 if refinement else 1e-8,
        residual_tolerance=1e-6,
        max_budget=8192,
        sy_character="target",
        reconstruct_cached_only=not refinement,
        refine_cached_budget=refinement,
    )
    if missing_partner:
        with pytest.raises(RuntimeError, match="still fails H-residual"):
            stage.run(**kwargs)
    else:
        stage.run(**kwargs)
    report_name = (
        stage.REFINEMENT_ACCEPTANCE_NAME if refinement else stage.RECONSTRUCTION_ACCEPTANCE_NAME
    )
    report = json.loads((output / report_name).read_text())
    assert report["closed"] is not missing_partner
    assert report["spectral_acceptance_closed"] is False
    assert report["iterative_solver_called"] is refinement
    assert len(calls) == (1 if refinement else 0)  # Existing CSV row never blocks refinement.
    stale = json.loads((output / stage.OBSERVABLES_ACCEPTANCE_NAME).read_text())
    assert stale["closed"] is False
    kind = "refinement" if refinement else "reconstruction"
    original_report = output / f"{kind}_source_reports" / stage.OBSERVABLES_ACCEPTANCE_NAME
    assert json.loads(original_report.read_text())["closed"] is True
    assert source.exists()
    archived_path = next((output / f"{kind}_source_rows").glob("budget_00000768*.json"))
    archived = json.loads(archived_path.read_text())
    assert archived["window_maximum_residual"] == 0.2
