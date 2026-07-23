import numpy as np

from qlinks.caging import (
    diagnose_incidence_constraint_interface,
    diagnose_scalar_laurent_bulk_phase,
    diagnose_scalar_laurent_domain_wall,
)


def test_scalar_laurent_bulk_phase_detects_fredholm_transition() -> None:
    inside = diagnose_scalar_laurent_bulk_phase(0.5, tolerance=1.0e-12)
    outside = diagnose_scalar_laurent_bulk_phase(2.0, tolerance=1.0e-12)
    critical = diagnose_scalar_laurent_bulk_phase(-1.0, tolerance=1.0e-12)

    assert inside.is_fredholm
    assert inside.winding_number == 1
    assert inside.toeplitz_index == -1
    assert outside.is_fredholm
    assert outside.winding_number == 0
    assert outside.toeplitz_index == 0
    assert not critical.is_fredholm
    assert critical.winding_number is None
    assert critical.toeplitz_index is None


def test_scalar_laurent_index_jump_binds_interface_mode() -> None:
    report = diagnose_scalar_laurent_domain_wall(
        2.0,
        0.5,
        left_length=20,
        right_length=20,
        tolerance=1.0e-12,
    )

    assert report.classification == "fredholm_interface_mode"
    assert report.predicted_right_interface_modes == 1
    assert report.predicted_left_interface_modes == 0
    assert report.kernel_dimension == 1
    assert report.is_exponentially_interface_localized
    assert report.interface_site_weight is not None
    assert report.interface_site_weight > 0.5
    assert report.interface_window_weight is not None
    assert report.interface_window_weight > 0.8
    assert report.residual < 1.0e-10


def test_unit_modulus_transport_wall_is_critical_and_delocalized() -> None:
    report = diagnose_scalar_laurent_domain_wall(
        -1.0,
        1.0,
        left_length=20,
        right_length=20,
        tolerance=1.0e-12,
    )

    assert report.classification == "critical_transport_no_fredholm_index"
    assert report.predicted_right_interface_modes is None
    assert report.predicted_left_interface_modes is None
    assert not report.is_exponentially_interface_localized
    assert report.inverse_participation_ratio is not None
    np.testing.assert_allclose(
        report.inverse_participation_ratio,
        1.0 / report.site_count,
        atol=1.0e-12,
    )
    assert report.interface_site_weight is not None
    np.testing.assert_allclose(
        report.interface_site_weight,
        1.0 / report.site_count,
        atol=1.0e-12,
    )


def _square_cycle_boundary() -> tuple[np.ndarray, np.ndarray]:
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
    return boundary, state


def test_flat_incidence_interface_merges_but_does_not_create_h0_mode() -> None:
    boundary, state = _square_cycle_boundary()
    interface = np.zeros((2, 8), dtype=np.complex128)
    interface[0, 0] = 1.0
    interface[0, 4] = -state[0] / state[0]
    interface[1, 1] = 1.0
    interface[1, 5] = -state[1] / state[1]

    report = diagnose_incidence_constraint_interface(
        boundary,
        boundary,
        interface,
        tolerance=1.0e-12,
    )

    assert report.classification == "flat_gluing_merges_local_h0_sectors"
    assert report.decoupled_kernel_dimension == 2
    assert report.combined_kernel_dimension == 1
    assert report.interface_created_dimension == 0
    assert report.interface_removed_dimension == 1
    assert report.kernel_equals_h0
    assert report.connected_component_count == 1


def test_frustrated_incidence_interface_lifts_local_h0_mode() -> None:
    boundary, state = _square_cycle_boundary()
    interface = np.zeros((2, 8), dtype=np.complex128)
    interface[0, 0] = 1.0
    interface[0, 4] = -state[0] / state[0]
    interface[1, 1] = 1.0
    interface[1, 5] = state[1] / state[1]

    report = diagnose_incidence_constraint_interface(
        boundary,
        boundary,
        interface,
        tolerance=1.0e-12,
    )

    assert report.classification == "frustrated_interface_lifts_local_h0_sector"
    assert report.combined_kernel_dimension == 0
    assert report.interface_created_dimension == 0
    assert report.interface_removed_dimension == 2
    assert report.gauge_flatness_residual is not None
    assert report.gauge_flatness_residual > 1.0e-3


def test_spin_one_xy_bond_sign_wall_does_not_change_cyclotomic_root() -> None:
    from qlinks.basis.configs import basis_configs_from_build_result
    from qlinks.caging import diagnose_hard_core_laurent_lift, partition_cage_hamiltonian
    from qlinks.models import SpinOneXYChainModel, spin_one_xy_scar_tower_states

    length = 4
    couplings = tuple(
        (site, (site + 1) % length, 1.0 if site < length // 2 else -1.0) for site in range(length)
    )
    model = SpinOneXYChainModel(
        length=length,
        boundary_condition="periodic",
        j_xy=0.0,
        total_sz=0,
        extra_xy_couplings=couplings,
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

    np.testing.assert_allclose(build.hamiltonian @ tower, 0.0, atol=1.0e-12)
    assert report.is_cyclotomic_hard_core_lift
    np.testing.assert_allclose(report.uniform_transport_factor, -1.0, atol=1.0e-12)
    assert report.primitive_root_order == 2
    assert not report.toeplitz_fredholm_index_is_defined
