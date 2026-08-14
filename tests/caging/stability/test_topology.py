import numpy as np


def test_fixed_manifold_compatibility_allows_internal_rotation() -> None:
    from qlinks.caging.stability import fixed_cage_manifold_compatibility

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
    from qlinks.caging.stability import diagnose_chiral_index

    block = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.complex128)
    report = diagnose_chiral_index(block, trim_isolated_rows=False, tolerance=1.0e-12)
    assert report.kernel_plus_dimension == 2
    assert report.kernel_minus_dimension == 1
    assert report.index == 1
    assert report.index_protected_plus_zero_modes == 1
    assert report.paired_zero_mode_count == 1
    assert np.isclose(report.singular_gap, 1.0)


def test_locality_restricted_chiral_profile_detects_regional_zero_mode() -> None:
    from qlinks.caging.stability import diagnose_locality_restricted_chiral_profile

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
    from qlinks.caging.stability import regional_chiral_kernel_span

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
    from qlinks.caging.stability import regional_cage_quotient

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
    from qlinks.caging.stability.topology import diagnose_signed_boundary_holonomy

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
    from qlinks.caging.stability.topology import diagnose_relative_mod2_cycles

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
