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


def test_signed_boundary_holonomy_detects_z2_cycle_sign() -> None:
    from qlinks.caging.stability import diagnose_signed_boundary_holonomy

    positive = np.asarray([[1.0, 1.0], [1.0, 1.0]])
    negative = np.asarray([[1.0, 1.0], [1.0, -1.0]])

    positive_report = diagnose_signed_boundary_holonomy(positive)
    negative_report = diagnose_signed_boundary_holonomy(negative)

    assert positive_report.cycle_rank == 1
    assert positive_report.sign_signature == (1,)
    assert negative_report.cycle_rank == 1
    assert negative_report.sign_signature == (-1,)
    assert negative_report.negative_cycle_count == 1


def test_relative_mod2_cycle_quotients_regional_cycles() -> None:
    from qlinks.caging.stability import diagnose_relative_mod2_cycles

    boundary = np.asarray([[1.0, 1.0], [1.0, 1.0]])

    separated = diagnose_relative_mod2_cycles(boundary, regions=((0,), (1,)))
    covered = diagnose_relative_mod2_cycles(boundary, regions=((0, 1),))

    assert separated.full_cycle_dimension == 1
    assert separated.regional_cycle_span_dimension == 0
    assert separated.relative_cycle_dimension == 1
    assert separated.relative_cycle_basis.shape == (1, 4)

    assert covered.full_cycle_dimension == 1
    assert covered.regional_cycle_span_dimension == 1
    assert covered.relative_cycle_dimension == 0


def test_boundary_cancellation_matroid_isolates_weighted_collective_class() -> None:
    from qlinks.caging import diagnose_boundary_cancellation_matroid

    boundary = np.asarray([[1.0, 1.0, 1.0, 1.0]], dtype=np.complex128)
    report = diagnose_boundary_cancellation_matroid(
        boundary,
        regions=((0, 1), (2, 3)),
        tolerance=1.0e-12,
    )
    collective = np.asarray([1.0, 1.0, -1.0, -1.0], dtype=np.complex128) / 2.0

    assert report.rank == 1
    assert report.dependency_dimension == 3
    assert report.regional_dependency_span_dimension == 2
    assert report.relative_dependency_dimension == 1
    assert report.regional_circuit_count == 2
    assert report.inclusion_residual < 1.0e-12
    assert report.edge_flow_conservation_residual < 1.0e-12
    assert abs(np.vdot(report.relative_dependency_basis[:, 0], collective)) > 1.0 - 1.0e-12


def test_boundary_cancellation_matroid_scan_detects_relative_rank_jump() -> None:
    from qlinks.caging import scan_boundary_cancellation_matroid

    base = np.asarray(
        [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.complex128,
    )
    perturbation = np.asarray(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, -1.0, -1.0]],
        dtype=np.complex128,
    )
    branch = scan_boundary_cancellation_matroid(
        base,
        perturbation,
        regions=((0, 1), (2, 3)),
        parameters=(0.0, 1.0e-3, 1.0),
        tolerance=1.0e-12,
    )

    np.testing.assert_array_equal(branch.dependency_dimensions, [3, 2, 2])
    np.testing.assert_array_equal(branch.regional_dimensions, [2, 2, 2])
    np.testing.assert_array_equal(branch.relative_dimensions, [1, 0, 0])


def test_periodic_boundary_cancellation_scaling_separates_flat_and_lifted_bands() -> None:
    from qlinks.caging import scan_periodic_boundary_cancellation_scaling

    base = np.asarray(
        [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.complex128,
    )
    collective_mass = np.asarray(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, -1.0, -1.0]],
        dtype=np.complex128,
    )
    regions = ((0, 1), (2, 3))

    flat = scan_periodic_boundary_cancellation_scaling(
        base,
        regions,
        (2, 4, 8),
        tolerance=1.0e-12,
    )
    lifted = scan_periodic_boundary_cancellation_scaling(
        base,
        regions,
        (2, 4, 8),
        coupling_terms=((0, collective_mass),),
        tolerance=1.0e-12,
    )

    assert flat.scaling_label == "extensive_zero_band"
    np.testing.assert_array_equal(flat.relative_dependency_dimensions, [2, 4, 8])
    np.testing.assert_allclose(flat.relative_dependency_densities, 1.0)
    assert np.isclose(flat.relative_dimension_growth_exponent, 1.0)

    assert lifted.scaling_label == "fully_lifted"
    np.testing.assert_array_equal(lifted.relative_dependency_dimensions, [0, 0, 0])
    np.testing.assert_allclose(lifted.minimum_positive_relative_gaps, 2.0)
    assert abs(lifted.positive_relative_gap_exponent) < 1.0e-12


def test_periodic_boundary_cancellation_scaling_detects_isolated_gapless_mode() -> None:
    from qlinks.caging import scan_periodic_boundary_cancellation_scaling

    base = np.asarray(
        [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.complex128,
    )
    collective_response = np.asarray(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, -1.0, -1.0]],
        dtype=np.complex128,
    )
    report = scan_periodic_boundary_cancellation_scaling(
        base,
        ((0, 1), (2, 3)),
        (16, 32, 64, 128),
        coupling_terms=((0, collective_response), (1, -collective_response)),
        tolerance=1.0e-12,
    )

    assert report.scaling_label == "isolated_zero_momenta"
    np.testing.assert_array_equal(report.relative_dependency_dimensions, [1, 1, 1, 1])
    np.testing.assert_allclose(
        report.relative_dependency_densities,
        np.asarray([1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0, 1.0 / 128.0]),
    )
    assert all(point.relative_zero_momentum_indices == (0,) for point in report.points)
    assert report.positive_relative_gap_exponent is not None
    assert np.isclose(report.positive_relative_gap_exponent, -1.0, atol=0.02)


def test_periodic_boundary_fourier_sum_matches_explicit_block_circulant_nullity() -> None:
    from qlinks.caging import scan_periodic_boundary_cancellation_scaling
    from qlinks.caging.nullspace import nullspace_svd

    base = np.asarray(
        [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.complex128,
    )
    coupling = np.asarray(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, -1.0, -1.0]],
        dtype=np.complex128,
    )
    n_repeats = 5
    n_rows, n_columns = base.shape
    explicit = np.zeros(
        (n_repeats * n_rows, n_repeats * n_columns),
        dtype=np.complex128,
    )
    for cell in range(n_repeats):
        rows = slice(cell * n_rows, (cell + 1) * n_rows)
        local_columns = slice(cell * n_columns, (cell + 1) * n_columns)
        next_cell = (cell + 1) % n_repeats
        next_columns = slice(next_cell * n_columns, (next_cell + 1) * n_columns)
        explicit[rows, local_columns] += base + coupling
        explicit[rows, next_columns] -= coupling

    report = scan_periodic_boundary_cancellation_scaling(
        base,
        ((0, 1), (2, 3)),
        (n_repeats,),
        coupling_terms=((0, coupling), (1, -coupling)),
        tolerance=1.0e-12,
    )
    explicit_nullity = nullspace_svd(explicit, tolerance=1.0e-12).shape[1]

    assert report.points[0].dependency_dimension == explicit_nullity


def test_periodic_boundary_scaling_rejects_coupling_that_breaks_regional_circuits() -> None:
    import pytest

    from qlinks.caging import scan_periodic_boundary_cancellation_scaling

    base = np.asarray(
        [[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.complex128,
    )
    bad_coupling = np.asarray(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=np.complex128,
    )

    with pytest.raises(ValueError, match="does not preserve the base regional"):
        scan_periodic_boundary_cancellation_scaling(
            base,
            ((0, 1), (2, 3)),
            (4,),
            coupling_terms=((0, bad_coupling),),
            tolerance=1.0e-12,
        )


def _physical_square_qdm_periodic_cage_unit_cell():
    from qlinks.caging import (
        LocalQDMCageSearchConfig,
        RobustQDMLocalCageSearchConfig,
        SquareQDMPeriodicProductUnitCell,
        robust_qdm_local_cage_search,
    )
    from qlinks.models import SquareQDMModel

    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
        coup_kin=1.0,
        coup_pot=1.0,
    )
    config = RobustQDMLocalCageSearchConfig(
        local_config=LocalQDMCageSearchConfig(
            halo_layers=0,
            boundary_mode="relaxed",
            prune_inactive_local_basis_states=True,
            tolerance=1.0e-10,
            degenerate_basis_strategy="ipr",
            ipr_random_seed=1234,
        ),
        region_strategies=("stripe",),
        stripe_widths=(1,),
        stripe_directions=(0, 1),
        max_regions_per_strategy=None,
        block_signatures=((0, 2),),
        max_records_per_region=2,
        min_blocks=2,
        max_blocks=None,
        max_product_support_size=2048,
        max_paddings_per_stage=100,
        max_paddings_per_packing=10,
        include_sectors=True,
        padding_stages=("static",),
        tolerance=1.0e-9,
        store_full_states=False,
    )
    certified, context = robust_qdm_local_cage_search(
        model,
        config=config,
        return_context=True,
    )
    return SquareQDMPeriodicProductUnitCell.from_padding(
        model,
        context.blocks,
        certified.reports[4].padding,
        repeat_axis="x",
    )


def test_physical_periodic_product_cancellation_scaling_uses_actual_qdm_flips() -> None:
    from qlinks.caging import scan_square_qdm_periodic_product_cancellation_scaling

    report = scan_square_qdm_periodic_product_cancellation_scaling(
        _physical_square_qdm_periodic_cage_unit_cell(),
        (1, 2, 3),
        max_support_size=64,
        tolerance=1.0e-9,
    )

    assert report.has_unique_product_kernel
    np.testing.assert_array_equal(report.boundary_nullities, [1, 1, 1])
    np.testing.assert_allclose(report.interference_gaps, 2.0)
    np.testing.assert_array_equal(report.kinetic_constraint_ranks, [2, 4, 6])
    np.testing.assert_allclose(report.kinetic_compatible_fractions, 7.0 / 8.0)
    assert np.isclose(report.interference_gap_exponent, 0.0, atol=1.0e-12)
    assert np.isclose(report.kinetic_constraint_rank_exponent, 1.0, atol=1.0e-12)
    for point in report.points:
        assert point.product_state_boundary_residual < 1.0e-12
        assert point.product_state_kernel_weight > 1.0 - 1.0e-12
        assert point.potential_compatibility.rank == 0
        assert point.kinetic_compatibility.rank == 2 * point.repeats
        assert len(point.kinetic_compatibility.equal_coupling_pairs) == 2 * point.repeats


def test_periodic_product_support_materialization_respects_size_cap() -> None:
    import pytest

    from qlinks.caging import materialize_square_qdm_periodic_product_support

    instance = _physical_square_qdm_periodic_cage_unit_cell().instantiate(3)
    with pytest.raises(ValueError, match="exceeds max_support_size"):
        materialize_square_qdm_periodic_product_support(
            instance,
            max_support_size=63,
        )


def test_real_local_sign_obstruction_is_global_phase_invariant() -> None:
    from qlinks.caging import diagnose_real_local_sign_obstruction

    a = (0, 0, 0)
    b = (1, 1, 1)
    words = (
        (a, a),
        (b, b),
        (a, b),
    )
    amplitudes = np.array([1.0, -1.0, 1.0], dtype=np.complex128)

    report = diagnose_real_local_sign_obstruction(
        words,
        amplitudes,
        window_size=1,
        tolerance=1.0e-12,
    )
    flipped = diagnose_real_local_sign_obstruction(
        words,
        -amplitudes,
        window_size=1,
        tolerance=1.0e-12,
    )

    assert report.is_obstructed
    assert report.obstruction_dimension == 1
    assert flipped.obstruction_dimension == report.obstruction_dimension
    assert report.obstruction_witness is not None
    assert int(np.sum(report.obstruction_witness)) > 0


def test_collective_square_qdm_local_grammar_has_only_product_kernel_at_8x4() -> None:
    from qlinks.caging import (
        CageSearchConfig,
        CageSearcher,
        scan_square_qdm_collective_locality_extension,
    )
    from qlinks.models import SquareQDMModel

    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
        coup_kin=1.0,
        coup_pot=1.0,
    )
    build = model.build(
        basis_solver="dfs",
        builder="sparse",
        backend="scipy",
        sort_basis=True,
    )
    search = CageSearcher.from_model_build_result(
        build,
        config=CageSearchConfig(
            search_type="type1",
            tolerance=1.0e-10,
            degenerate_basis_strategy="ipr",
            ipr_n_restarts=32,
            ipr_candidate_count=32,
            ipr_random_seed=1234,
        ),
    ).run()
    collective = search[(0, 4), 8]
    support_configs = np.asarray(
        [build.basis.state(int(index)) for index in collective.support],
        dtype=np.int64,
    )

    report = scan_square_qdm_collective_locality_extension(
        model,
        support_configs,
        _physical_square_qdm_periodic_cage_unit_cell(),
        ((3, 8),),
        max_words=1_000,
        max_product_support_size=32,
        dense_column_limit=512,
        maximum_nullity=8,
        ipr_restarts=32,
        tolerance=1.0e-9,
    )
    point = report.points[0]

    assert point.support_size == 192
    assert point.boundary_nullity == 4
    assert point.nullity_is_resolved
    assert point.product_translation_span_dimension == 4
    assert point.kernel_product_intersection_dimension == 4
    assert point.collective_quotient_dimension == 0
    assert point.kernel_is_exhausted_by_product_translations
    assert point.product_containment_residual < 1.0e-8
    np.testing.assert_allclose(point.principal_overlaps, 1.0, atol=1.0e-8)
    assert point.localized_support_sizes == (16, 16, 16, 16)


def test_cyclic_amplitude_bond_profile_detects_exact_schmidt_rank() -> None:
    from qlinks.caging import diagnose_cyclic_amplitude_bond_profile

    zero = (0, 0, 0)
    one = (1, 1, 1)
    words = (
        (zero, zero, zero, zero),
        (one, one, one, one),
    )
    report = diagnose_cyclic_amplitude_bond_profile(
        words,
        np.asarray([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0),
        tolerance=1.0e-12,
    )

    assert report.cut_ranks == (2, 2, 2)
    assert report.exact_open_bond_dimension == 2
    assert report.periodic_bond_dimension_lower_bound == 2
    assert report.translation_support_closed
    assert np.isclose(report.translation_eigenvalue, 1.0)
    assert report.translation_residual is not None
    assert report.translation_residual < 1.0e-12


def test_square_qdm_finite_bond_transfer_invariant_resolves_trivial_sector() -> None:
    from qlinks.caging import diagnose_square_qdm_finite_bond_transfer_invariant
    from qlinks.models import SquareQDMModel

    model = SquareQDMModel(
        lx=4,
        ly=4,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
        coup_kin=1.0,
        coup_pot=1.0,
    )
    build = model.build(
        basis_solver="dfs",
        builder="sparse",
        backend="scipy",
        sort_basis=True,
    )
    support_configs = np.asarray(
        [build.basis.state(index) for index in range(build.hamiltonian.shape[0])],
        dtype=np.int64,
    )
    uniform = np.ones((support_configs.shape[0], 1), dtype=np.complex128)
    uniform /= np.linalg.norm(uniform)

    report = diagnose_square_qdm_finite_bond_transfer_invariant(
        model,
        support_configs,
        uniform,
        tolerance=1.0e-10,
    )

    assert report.kernel_dimension == 1
    assert report.reference_dimension == 0
    assert report.relative_dimension == 1
    assert report.relative_trivial_sector_dimension == 1
    assert report.has_one_dimensional_trivial_spatial_quotient
    assert report.relative_sector_signature == ((0, 0, 1),)
    assert report.kernel_symmetry_residual < 1.0e-10
    assert report.group_relation_residual < 1.0e-10
