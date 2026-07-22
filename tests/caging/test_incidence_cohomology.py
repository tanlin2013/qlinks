import numpy as np

from qlinks.basis.configs import basis_configs_from_build_result
from qlinks.caging import (
    diagnose_boundary_incidence_cohomology,
    diagnose_hard_core_laurent_lift,
    partition_cage_hamiltonian,
)
from qlinks.models import SpinOneXYChainModel, spin_one_xy_scar_tower_states


def test_flat_two_channel_boundary_is_zeroth_cohomology() -> None:
    boundary = np.asarray(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    state = np.asarray([1.0, -1.0, 1.0, -1.0], dtype=np.complex128)

    report = diagnose_boundary_incidence_cohomology(
        boundary,
        state,
        tolerance=1.0e-12,
    )

    assert report.is_flat_incidence_problem
    assert report.kernel_is_exact_h0
    assert report.betti_0 == 1
    assert report.betti_1 == 1
    assert report.kernel_dimension == 1
    assert report.h0_intersection_dimension == 1
    assert report.state_h0_weight is not None
    np.testing.assert_allclose(report.state_h0_weight, 1.0, atol=1.0e-12)


def test_odd_unsigned_cycle_has_nonflat_transport_and_no_kernel() -> None:
    boundary = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )

    report = diagnose_boundary_incidence_cohomology(boundary, tolerance=1.0e-12)

    assert report.is_two_channel
    assert not report.is_flat_incidence_problem
    assert not report.kernel_is_exact_h0
    assert report.kernel_dimension == 0
    assert report.gauge_flatness_residual is not None
    assert report.gauge_flatness_residual > 1.0e-3


def test_three_channel_boundary_is_not_graph_incidence() -> None:
    boundary = np.asarray([[1.0, 1.0, 1.0]], dtype=np.complex128)

    report = diagnose_boundary_incidence_cohomology(boundary, tolerance=1.0e-12)

    assert not report.is_two_channel
    assert not report.is_flat_incidence_problem
    assert report.betti_0 is None
    assert report.betti_1 is None


def test_spin_one_xy_tower_is_order_two_hard_core_laurent_lift() -> None:
    length = 6
    model = SpinOneXYChainModel(
        length=length,
        boundary_condition="periodic",
        j_xy=1.0,
        total_sz=0,
    )
    build = model.build(
        builder="optimized",
        basis_solver="dfs",
        sort_basis=True,
        on_missing="raise",
    )
    configs = basis_configs_from_build_result(build)
    states, _labels = spin_one_xy_scar_tower_states(
        basis_configs=configs,
        length=length,
    )
    tower = states[:, 0]
    support = np.flatnonzero(np.abs(tower) > 1.0e-10)
    boundary = partition_cage_hamiltonian(build.hamiltonian, support).boundary

    report = diagnose_hard_core_laurent_lift(
        configs[support],
        tower[support],
        boundary,
        tolerance=1.0e-10,
    )

    assert report.is_cyclotomic_hard_core_lift
    assert report.primitive_root_order == 2
    assert report.uniform_transport_factor is not None
    np.testing.assert_allclose(report.uniform_transport_factor, -1.0, atol=1.0e-12)
    assert report.incidence_cohomology.betti_0 == 1
    assert report.incidence_cohomology.kernel_is_exact_h0
    assert report.has_unit_circle_symbol_zero
    assert not report.toeplitz_fredholm_index_is_defined
    assert report.one_site_translation_character is not None
    np.testing.assert_allclose(report.one_site_translation_character, -1.0, atol=1.0e-12)
