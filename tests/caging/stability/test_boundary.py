import numpy as np
import pytest  # noqa: F401


def test_boundary_cancellation_matroid_isolates_weighted_collective_class() -> None:
    from qlinks.caging.stability import diagnose_boundary_cancellation_matroid

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
    from qlinks.caging.stability import scan_boundary_cancellation_matroid

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
    from qlinks.caging.stability import scan_periodic_boundary_cancellation_scaling

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
    from qlinks.caging.stability import scan_periodic_boundary_cancellation_scaling

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
    from qlinks.caging.nullspace import nullspace_svd
    from qlinks.caging.stability import scan_periodic_boundary_cancellation_scaling

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
    from qlinks.caging.stability import scan_periodic_boundary_cancellation_scaling

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
