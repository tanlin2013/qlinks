from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from qlinks.caging import (
    adjacent_gap_ratio_report,
    basis_permutation_from_variable_permutation,
    cyclic_symmetry_sector_basis,
    diagnose_eigenpair,
    project_operator_to_sector,
    refine_sector_by_involution,
    select_microcanonical_window_by_count,
)


def test_diagnose_eigenpair_and_microcanonical_selection() -> None:
    hamiltonian = np.diag([0.0, 1.0, 2.0, 3.0])
    report = diagnose_eigenpair(hamiltonian, np.array([0.0, 1.0, 0.0, 0.0]))
    assert np.isclose(report.energy, 1.0)
    assert report.residual_norm == 0.0
    assert report.variance == 0.0

    window = select_microcanonical_window_by_count(
        np.array([0.0, 0.9, 1.2, 3.0]),
        target_energy=1.0,
        target_count=2,
    )
    assert window.indices == (1, 2)
    assert window.n_states == 2
    assert np.isclose(window.half_width, 0.2)


def test_cyclic_and_reflection_sector_basis() -> None:
    basis = np.asarray(
        [
            [1, -1, -1, 1],
            [1, 1, -1, -1],
            [-1, 1, 1, -1],
            [-1, -1, 1, 1],
        ],
        dtype=np.int64,
    )
    translation = basis_permutation_from_variable_permutation(
        basis,
        np.roll(np.arange(4), 1),
    )
    sector = cyclic_symmetry_sector_basis(
        translation,
        order=4,
        momentum_index=0,
    )
    assert sector.sector_dimension == 1
    np.testing.assert_allclose(
        (sector.basis.conj().T @ sector.basis).toarray(),
        np.eye(1),
        atol=1.0e-12,
    )

    reflection = basis_permutation_from_variable_permutation(
        basis,
        np.asarray([0, 3, 2, 1]),
    )
    refined = refine_sector_by_involution(sector, reflection, eigenvalue=1)
    assert refined.sector_dimension == 1

    operator = sp.eye(4, format="csr")
    projected = project_operator_to_sector(operator, refined)
    np.testing.assert_allclose(projected, np.eye(1), atol=1.0e-12)


def test_adjacent_gap_ratio_report_filters_degeneracies() -> None:
    levels = np.asarray([0.0, 1.0, 1.0, 2.4, 4.0, 6.0])
    report = adjacent_gap_ratio_report(
        levels,
        trim_fraction=0.0,
        degeneracy_tolerance=1.0e-12,
    )
    assert len(report.ratios) == 2
    assert 0.0 < report.mean_ratio <= 1.0
