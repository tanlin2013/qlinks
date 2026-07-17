import numpy as np
import scipy.sparse as sp

from qlinks.caging import (
    combine_perturbations_from_coefficients,
    diagnose_cage_stability,
    linearized_cage_obstruction_from_hamiltonians,
    partition_cage_hamiltonian,
    random_cage_stability_ensemble,
    scan_cage_stability_branch,
    subspace_principal_overlaps,
    subspace_projector_distance,
)


def _assemble_hamiltonian(
    boundary: np.ndarray,
    *,
    internal: np.ndarray | None = None,
    external: np.ndarray | None = None,
) -> np.ndarray:
    if internal is None:
        internal = np.zeros((2, 2), dtype=np.complex128)
    if external is None:
        external = np.diag([2.0, 3.0]).astype(np.complex128)
    return np.block(
        [
            [internal, boundary.conj().T],
            [boundary, external],
        ]
    )


def _toy_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_boundary = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    base = _assemble_hamiltonian(base_boundary)

    strong_boundary = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    strong = _assemble_hamiltonian(
        strong_boundary,
        internal=np.eye(2, dtype=np.complex128),
        external=np.zeros((2, 2), dtype=np.complex128),
    )

    structural_boundary = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    structural = _assemble_hamiltonian(
        structural_boundary,
        external=np.zeros((2, 2), dtype=np.complex128),
    )

    incompatible_boundary = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)
    incompatible = _assemble_hamiltonian(
        incompatible_boundary,
        external=np.zeros((2, 2), dtype=np.complex128),
    )

    cage_state = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)
    return base, strong, structural, incompatible, cage_state


def test_partition_cage_hamiltonian_handles_sparse_matrix() -> None:
    base, _strong, _structural, _incompatible, _state = _toy_problem()
    blocks = partition_cage_hamiltonian(sp.csr_matrix(base), support=(0, 1))

    assert blocks.support.tolist() == [0, 1]
    assert blocks.complement.tolist() == [2, 3]
    np.testing.assert_allclose(blocks.internal.toarray(), np.zeros((2, 2)))
    np.testing.assert_allclose(
        blocks.boundary.toarray(),
        np.array([[1.0, 1.0], [0.0, 0.0]]),
    )


def test_diagnose_cage_stability_separates_boundary_and_invariant_kernels() -> None:
    base, _strong, _structural, _incompatible, cage_state = _toy_problem()
    diagnostic = diagnose_cage_stability(
        base,
        support=(0, 1),
        state=cage_state,
        tolerance=1.0e-12,
    )

    assert diagnostic.boundary_rank == 1
    assert diagnostic.boundary_nullity == 1
    assert diagnostic.invariant_cage_dimension == 1
    assert np.isclose(diagnostic.interference_gap, np.sqrt(2.0))
    assert diagnostic.state_boundary_residual < 1.0e-12
    assert diagnostic.state_internal_eigen_residual < 1.0e-12
    assert diagnostic.state_full_residual < 1.0e-12
    assert np.isclose(diagnostic.state_invariant_weight, 1.0)


def test_branch_scan_distinguishes_deformed_cage_from_fixed_initial_state() -> None:
    base, _strong, structural, _incompatible, cage_state = _toy_problem()
    report = scan_cage_stability_branch(
        base,
        structural,
        support=(0, 1),
        parameters=(0.0, 0.5, 1.0),
        reference_state=cage_state,
        tolerance=1.0e-12,
    )

    assert report.invariant_dimensions.tolist() == [1, 1, 1]
    assert np.nanmax(report.continued_full_residuals) < 1.0e-12
    assert report.fixed_state_full_residuals[-1] > 0.5
    assert report.points[-1].continued_overlap_with_reference < 1.0
    assert report.points[-1].projector_distance_from_reference > 0.0


def test_branch_scan_detects_incompatible_rank_increase() -> None:
    base, _strong, _structural, incompatible, cage_state = _toy_problem()
    report = scan_cage_stability_branch(
        base,
        incompatible,
        support=(0, 1),
        parameters=(0.0, 0.25),
        reference_state=cage_state,
        tolerance=1.0e-12,
    )

    assert report.invariant_dimensions.tolist() == [1, 0]
    assert report.points[-1].continued_state is None
    assert report.points[-1].minimum_principal_overlap == 0.0
    assert np.isclose(report.points[-1].projector_distance_from_reference, 1.0)


def test_random_ensemble_separates_compatible_and_incompatible_directions() -> None:
    base, strong, structural, incompatible, cage_state = _toy_problem()
    compatible_report = random_cage_stability_ensemble(
        base,
        (strong, structural),
        support=(0, 1),
        strengths=(0.1, 0.5),
        n_samples=16,
        reference_state=cage_state,
        minimum_subspace_overlap=0.0,
        random_seed=7,
        tolerance=1.0e-12,
    )
    incompatible_report = random_cage_stability_ensemble(
        base,
        (incompatible,),
        support=(0, 1),
        strengths=(0.1,),
        n_samples=8,
        reference_state=cage_state,
        minimum_subspace_overlap=0.0,
        random_seed=7,
        tolerance=1.0e-12,
    )

    assert [item.survival_fraction for item in compatible_report.aggregates] == [1.0, 1.0]
    assert incompatible_report.aggregates[0].survival_fraction == 0.0


def test_linearized_obstruction_finds_structure_compatible_combination() -> None:
    base, _strong, structural, incompatible, cage_state = _toy_problem()
    report = linearized_cage_obstruction_from_hamiltonians(
        base,
        (structural, incompatible),
        support=(0, 1),
        cage_state=cage_state,
        coefficient_field="real",
        tolerance=1.0e-12,
    )

    assert report.rank == 1
    assert report.compatible_dimension == 1
    assert report.perturbation_diagnostics[0].first_order_eigenstate_compatible
    assert not report.perturbation_diagnostics[0].preserves_state
    assert not report.perturbation_diagnostics[1].first_order_eigenstate_compatible

    compatible_perturbation = combine_perturbations_from_coefficients(
        (structural, incompatible),
        report.compatible_coefficient_basis,
    )[0]
    compatible_blocks = partition_cage_hamiltonian(compatible_perturbation, (0, 1))
    assert np.linalg.norm(compatible_blocks.boundary[1]) < 1.0e-12


def test_subspace_comparison_handles_rotation_and_dimension_change() -> None:
    basis_a = np.array([[1.0], [0.0]], dtype=np.complex128)
    basis_b = np.array([[1.0], [1.0]], dtype=np.complex128) / np.sqrt(2.0)
    basis_full = np.eye(2, dtype=np.complex128)

    overlaps = subspace_principal_overlaps(basis_a, basis_b)
    assert np.allclose(overlaps, [1.0 / np.sqrt(2.0)])
    assert np.isclose(subspace_projector_distance(basis_a, basis_b), 1.0 / np.sqrt(2.0))
    assert np.isclose(subspace_projector_distance(basis_a, basis_full), 1.0)
