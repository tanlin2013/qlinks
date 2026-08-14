import numpy as np
import pytest

from tests.caging.stability._helpers import _physical_square_qdm_periodic_cage_unit_cell


@pytest.mark.integration
@pytest.mark.scientific
def test_physical_periodic_product_cancellation_scaling_uses_actual_qdm_flips() -> None:
    from qlinks.caging.stability import scan_square_qdm_periodic_product_cancellation_scaling

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

    from qlinks.caging.stability import materialize_square_qdm_periodic_product_support

    instance = _physical_square_qdm_periodic_cage_unit_cell().instantiate(3)
    with pytest.raises(ValueError, match="exceeds max_support_size"):
        materialize_square_qdm_periodic_product_support(
            instance,
            max_support_size=63,
        )


def test_real_local_sign_obstruction_is_global_phase_invariant() -> None:
    from qlinks.caging.stability import diagnose_real_local_sign_obstruction

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


@pytest.mark.integration
@pytest.mark.scientific
def test_collective_square_qdm_local_grammar_has_only_product_kernel_at_8x4() -> None:
    from qlinks.caging import (
        CageSearchConfig,
        CageSearcher,
    )
    from qlinks.caging.stability import scan_square_qdm_collective_locality_extension
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
    from qlinks.caging.stability import diagnose_cyclic_amplitude_bond_profile

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
    from qlinks.caging.stability import diagnose_square_qdm_finite_bond_transfer_invariant
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


def test_reduced_constraint_fredholm_candidate_distinguishes_square_and_tall_maps() -> None:
    from qlinks.caging.stability import diagnose_reduced_constraint_fredholm_candidate

    kernel = np.asarray([[1.0], [1.0]], dtype=np.complex128) / np.sqrt(2.0)
    square = diagnose_reduced_constraint_fredholm_candidate(
        np.asarray([[1.0, -1.0]], dtype=np.complex128),
        kernel_basis=kernel,
        tolerance=1.0e-12,
    )
    tall = diagnose_reduced_constraint_fredholm_candidate(
        np.asarray(
            [[1.0, -1.0], [2.0, -2.0], [0.0, 0.0]],
            dtype=np.complex128,
        ),
        kernel_basis=kernel,
        tolerance=1.0e-12,
    )

    assert square.admits_intrinsic_scalar_winding
    assert square.classification == "square_fredholm_symbol_candidate"
    assert square.codomain_excess == 0
    assert np.isclose(square.reduced_gap, np.sqrt(2.0))
    assert tall.is_reduced_injective
    assert not tall.admits_intrinsic_scalar_winding
    assert tall.classification == "rectangular_stiefel_no_intrinsic_winding"
    assert tall.codomain_excess == 2
    assert np.isclose(tall.reduced_gap, np.sqrt(10.0))


@pytest.mark.integration
@pytest.mark.scientific
def test_compact_qdm_reduced_winding_is_constant_and_trivial_at_fixed_width() -> None:
    from qlinks.caging.stability import diagnose_square_qdm_compact_cage_reduced_winding

    report = diagnose_square_qdm_compact_cage_reduced_winding(
        _physical_square_qdm_periodic_cage_unit_cell(),
        (1, 2, 3),
        max_support_size=64,
        tolerance=1.0e-9,
    )

    assert report.classification == "local_constant_symbol_trivial_winding"
    assert report.local_pair_offsets == ((0, 2), (9, 11))
    assert report.reduced_coupling_winding == 0
    assert np.isclose(report.reduced_coupling_gap, np.sqrt(2.0))
    np.testing.assert_allclose(
        report.reduced_coupling_symbol,
        np.sqrt(2.0) * np.eye(2),
        atol=1.0e-12,
    )
    assert not report.state_space_has_intrinsic_scalar_winding_candidate
    assert report.has_uniform_fixed_width_gap
    assert [point.state_complement.codomain_excess for point in report.points] == [
        5,
        49,
        321,
    ]
    np.testing.assert_allclose(
        [point.state_complement.reduced_gap for point in report.points],
        2.0,
    )
    assert [point.kinetic_quotient_dimension for point in report.points] == [2, 4, 6]
    np.testing.assert_allclose(
        [point.kinetic_quotient_gap for point in report.points],
        np.sqrt(2.0),
    )
    np.testing.assert_allclose(
        [point.intercell_gram_norm for point in report.points],
        0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        [point.unit_cell_gram_residual for point in report.points],
        0.0,
        atol=1.0e-12,
    )
