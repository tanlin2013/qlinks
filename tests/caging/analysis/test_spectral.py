from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from qlinks.caging.analysis.spectral import (
    adjacent_gap_ratio_report,
    basis_permutation_from_variable_permutation,
    commuting_cyclic_symmetry_sector_basis,
    cyclic_symmetry_sector_basis,
    diagnose_eigenpair,
    gaussian_spectral_filter,
    microcanonical_ensemble_from_spectrum,
    product_basis_diagonal_phase_factors,
    project_operator_to_sector,
    refine_sector_by_involution,
    select_microcanonical_window_by_count,
    select_microcanonical_window_by_width,
    spectral_observable_moments,
    thermodynamic_energy_window_plan,
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


def test_thermodynamic_width_filter_and_projected_second_moment() -> None:
    plan = thermodynamic_energy_window_plan(
        volume=16,
        energy_density=0.25,
        width_prefactor=0.5,
        local_energy_scale=2.0,
    )
    assert np.isclose(plan.target_energy, 4.0)
    assert np.isclose(plan.half_width, 4.0)
    assert np.isclose(plan.energy_density_half_width, 0.25)

    energies = np.asarray([0.0, 2.0, 4.0, 6.0, 8.0])
    window = select_microcanonical_window_by_width(
        energies,
        target_energy=4.0,
        half_width=2.0,
    )
    assert window.indices == (1, 2, 3)

    smooth = gaussian_spectral_filter(
        energies,
        target_energy=4.0,
        sigma=1.5,
    )
    assert np.isclose(sum(smooth.weights), 1.0)
    assert smooth.effective_state_count > 1.0

    # P O P vanishes in this one-dimensional sector, while P O^2 P=1.
    projected_o = np.zeros((1, 1), dtype=np.complex128)
    projected_o2 = np.ones((1, 1), dtype=np.complex128)
    vectors = np.ones((1, 1), dtype=np.complex128)
    moments = spectral_observable_moments(
        projected_o,
        vectors,
        squared_operator=projected_o2,
    )
    assert moments.mean == 0.0
    assert moments.second_moment == 1.0
    assert moments.variance == 1.0


def test_commuting_cyclic_sector_and_product_basis_phases() -> None:
    # Regular action of Z2 x Z2 on four basis states.
    tx = np.asarray([1, 0, 3, 2], dtype=np.int64)
    ty = np.asarray([2, 3, 0, 1], dtype=np.int64)
    sector = commuting_cyclic_symmetry_sector_basis(
        (tx, ty),
        orders=(2, 2),
        momentum_indices=(0, 0),
    )
    assert sector.sector_dimension == 1
    np.testing.assert_allclose(
        (sector.basis.conj().T @ sector.basis).toarray(),
        np.eye(1),
        atol=1.0e-12,
    )

    configs = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int64)
    phases = product_basis_diagonal_phase_factors(configs, [0.2, -0.4])
    np.testing.assert_allclose(
        phases,
        np.exp(1.0j * np.asarray([0.0, 0.2, -0.4, -0.2])),
    )


def test_low_rank_microcanonical_ensemble() -> None:
    energies = np.asarray([-2.0, -0.5, 0.5, 2.0])
    vectors = np.eye(4, dtype=np.complex128)
    ensemble = microcanonical_ensemble_from_spectrum(
        energies,
        vectors,
        target_energy=0.0,
        half_width=0.6,
        volume=4,
    )
    assert ensemble.selection.indices == (1, 2)
    assert ensemble.n_states == 2
    assert ensemble.hilbert_dimension == 4
    assert np.isclose(ensemble.energy_density_half_width, 0.125)

    rho = ensemble.density_matrix()
    np.testing.assert_allclose(rho, np.diag([0.0, 0.5, 0.5, 0.0]))
    assert np.isclose(np.trace(rho), 1.0)

    observable = np.diag([1.0, 2.0, 4.0, 8.0])
    assert np.isclose(ensemble.expectation(observable), 3.0)
    moments = ensemble.observable_moments(observable)
    assert np.isclose(moments.mean, 3.0)
    assert np.isclose(moments.second_moment, 10.0)
    assert np.isclose(moments.variance, 1.0)
