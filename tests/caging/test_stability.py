import numpy as np
import scipy.sparse as sp

from qlinks.caging import (
    cage_compatibility_hierarchy_from_hamiltonians,
    combine_perturbations_from_coefficients,
    diagnose_cage_stability,
    estimate_power_law_exponent,
    linearized_cage_obstruction_from_hamiltonians,
    partition_cage_hamiltonian,
    random_cage_stability_ensemble,
    scan_cage_stability_branch,
    scan_support_eigenstate_branch,
    subspace_complement_basis,
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


def _nonintegrable_tangent_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cage_state = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)
    orthogonal_state = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    basis_change = np.column_stack([cage_state, orthogonal_state])

    base_internal_local = np.diag([0.0, 1.0]).astype(np.complex128)
    perturbation_internal_local = np.array(
        [[0.0, -1.0], [-1.0, 0.0]],
        dtype=np.complex128,
    )
    base_boundary_local = np.array(
        [[0.0, 1.0], [0.0, 0.0]],
        dtype=np.complex128,
    )
    perturbation_boundary_local = np.array(
        [[-1.0, 1.0], [0.0, 0.0]],
        dtype=np.complex128,
    )

    base_internal = basis_change @ base_internal_local @ basis_change.conj().T
    perturbation_internal = basis_change @ perturbation_internal_local @ basis_change.conj().T
    base_boundary = base_boundary_local @ basis_change.conj().T
    perturbation_boundary = perturbation_boundary_local @ basis_change.conj().T

    base = _assemble_hamiltonian(base_boundary, internal=base_internal)
    perturbation = _assemble_hamiltonian(
        perturbation_boundary,
        internal=perturbation_internal,
        external=np.zeros((2, 2), dtype=np.complex128),
    )
    return base, perturbation, cage_state


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


def test_compatibility_hierarchy_detects_tangent_only_direction() -> None:
    base, perturbation, cage_state = _nonintegrable_tangent_problem()
    hierarchy = cage_compatibility_hierarchy_from_hamiltonians(
        base,
        (perturbation,),
        support=(0, 1),
        cage_state=cage_state,
        coefficient_field="real",
        tolerance=1.0e-12,
    )

    assert hierarchy.first_order.compatible_dimension == 1
    assert hierarchy.fixed_state.compatible_dimension == 0
    assert hierarchy.tangent_only_dimension == 1
    assert hierarchy.fixed_subspace_inclusion_residual < 1.0e-12


def test_support_eigenstate_branch_exposes_quadratic_leakage() -> None:
    base, perturbation, cage_state = _nonintegrable_tangent_problem()
    parameters = np.array([0.0, 1.0e-5, 2.0e-5, 5.0e-5, 1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3])
    branch = scan_support_eigenstate_branch(
        base,
        perturbation,
        support=(0, 1),
        parameters=parameters,
        reference_state=cage_state,
        tolerance=1.0e-14,
    )
    exponent = estimate_power_law_exponent(
        branch.parameters,
        branch.boundary_residuals,
        minimum_residual=1.0e-16,
    )

    assert branch.exact_cage_flags[0]
    assert not np.any(branch.exact_cage_flags[1:])
    assert exponent is not None
    assert np.isclose(exponent, 2.0, atol=1.0e-2)


def test_subspace_complement_basis_returns_parent_orthogonal_remainder() -> None:
    parent = np.eye(3, dtype=np.complex128)[:, :2]
    child = np.array([[1.0], [0.0], [0.0]], dtype=np.complex128)
    complement = subspace_complement_basis(parent, child, tolerance=1.0e-12)

    assert complement.shape == (3, 1)
    assert np.linalg.norm(child.conj().T @ complement) < 1.0e-12
    assert np.isclose(abs(complement[1, 0]), 1.0)


def test_summarize_cage_record_stability_compares_preferred_representatives():
    from types import SimpleNamespace

    from qlinks.caging import summarize_cage_record_stability

    base, strong, structural, incompatible, cage_state = _toy_problem()
    records = (
        SimpleNamespace(
            local_state=cage_state,
            support=np.array([0, 1], dtype=np.int64),
            signature=(0, 4),
        ),
        SimpleNamespace(
            local_state=cage_state,
            support=np.array([0, 1], dtype=np.int64),
            signature=(0, 4),
        ),
    )
    classifications = (
        SimpleNamespace(label="regional_candidate", n_collective_cancellation_source_probes=0),
        SimpleNamespace(label="extended_candidate", n_collective_cancellation_source_probes=3),
    )

    summaries = summarize_cage_record_stability(
        base,
        (strong, structural, incompatible),
        records,
        classification_reports=classifications,
        tolerance=1.0e-12,
    )

    assert len(summaries) == 2
    assert summaries[0].requires_collective_cancellation is False
    assert summaries[1].requires_collective_cancellation is True
    assert np.isclose(summaries[0].inverse_participation_ratio, 0.5)
    assert summaries[0].formal_compatible_dimension == summaries[1].formal_compatible_dimension


def test_fixed_manifold_compatibility_allows_internal_rotation() -> None:
    from qlinks.caging import fixed_cage_manifold_compatibility

    boundary = np.zeros((1, 3), dtype=np.complex128)
    manifold = np.eye(3, dtype=np.complex128)[:, :2]
    rotate_inside = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=np.complex128,
    )
    couple_outside = np.array(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=np.complex128,
    )
    report = fixed_cage_manifold_compatibility(
        boundary,
        manifold,
        (np.zeros_like(boundary), np.zeros_like(boundary)),
        internal_perturbations=(rotate_inside, couple_outside),
        tolerance=1.0e-12,
    )
    assert report.manifold_dimension == 2
    assert report.compatible_dimension == 1
    np.testing.assert_allclose(np.abs(report.compatible_coefficient_basis[:, 0]), [1.0, 0.0])


def test_chiral_index_separates_index_and_paired_zero_modes() -> None:
    from qlinks.caging import diagnose_chiral_index

    block = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.complex128)
    report = diagnose_chiral_index(block, trim_isolated_rows=False, tolerance=1.0e-12)
    assert report.kernel_plus_dimension == 2
    assert report.kernel_minus_dimension == 1
    assert report.index == 1
    assert report.index_protected_plus_zero_modes == 1
    assert report.paired_zero_mode_count == 1
    assert np.isclose(report.singular_gap, 1.0)


def test_locality_restricted_chiral_profile_detects_regional_zero_mode() -> None:
    from qlinks.caging import diagnose_locality_restricted_chiral_profile

    hamiltonian = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    state = np.array([1.0, -1.0, 0.0, 0.0], dtype=np.complex128) / np.sqrt(2.0)
    report = diagnose_locality_restricted_chiral_profile(
        hamiltonian,
        ((0, 1),),
        target_state=state,
        tolerance=1.0e-12,
    )

    assert report.n_regional_target_zero_modes == 1
    assert report.entries[0].chiral_index.kernel_plus_dimension == 1
    assert report.entries[0].chiral_index.index_protected_plus_zero_modes == 0
    assert report.entries[0].chiral_index.paired_zero_mode_count == 1
    assert report.entries[0].target_boundary_residual < 1.0e-12


def test_regional_chiral_kernel_span_finds_uncaptured_collective_mode() -> None:
    from qlinks.caging import regional_chiral_kernel_span

    hamiltonian = np.zeros((6, 6), dtype=np.complex128)
    hamiltonian[4, 0] = hamiltonian[0, 4] = 1.0
    hamiltonian[4, 1] = hamiltonian[1, 4] = 1.0
    hamiltonian[5, 2] = hamiltonian[2, 5] = 1.0
    hamiltonian[5, 3] = hamiltonian[3, 5] = 1.0
    local_a = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]) / np.sqrt(2.0)
    local_b = np.array([0.0, 0.0, 1.0, -1.0, 0.0, 0.0]) / np.sqrt(2.0)
    collective = np.array([1.0, 1.0, -1.0, -1.0, 0.0, 0.0]) / 2.0
    target = np.column_stack([local_a, local_b, collective])

    report = regional_chiral_kernel_span(
        hamiltonian,
        ((0, 1), (2, 3)),
        target,
        tolerance=1.0e-12,
    )

    assert report.regional_span_dimension == 2
    assert report.target_dimension == 3
    assert report.captured_target_dimension == 2
    assert report.uncaptured_target_dimension == 1


def test_regional_cage_quotient_isolates_collective_direction() -> None:
    from qlinks.caging import regional_cage_quotient

    hamiltonian = np.zeros((6, 6), dtype=np.complex128)
    hamiltonian[4, 0] = hamiltonian[0, 4] = 1.0
    hamiltonian[4, 1] = hamiltonian[1, 4] = 1.0
    hamiltonian[5, 2] = hamiltonian[2, 5] = 1.0
    hamiltonian[5, 3] = hamiltonian[3, 5] = 1.0
    local_a = np.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]) / np.sqrt(2.0)
    local_b = np.array([0.0, 0.0, 1.0, -1.0, 0.0, 0.0]) / np.sqrt(2.0)
    collective = np.array([1.0, 1.0, -1.0, -1.0, 0.0, 0.0]) / 2.0
    report = regional_cage_quotient(
        hamiltonian,
        ((0, 1), (2, 3)),
        np.column_stack([local_a, local_b, collective]),
        tolerance=1.0e-12,
    )
    assert report.intersection_dimension == 2
    assert report.quotient_dimension == 1
    assert report.inclusion_residual < 1.0e-12
    assert abs(np.vdot(report.quotient_basis[:, 0], collective)) > 1.0 - 1.0e-12
