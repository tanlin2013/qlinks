from __future__ import annotations

import numpy as np

from qlinks.basis.configs import basis_configs_from_build_result
from qlinks.models import (
    spin_one_xy_fixed_magnetization_dimension,
    spin_one_xy_hxy_h3_imaginary_j2_model,
    spin_one_xy_hxy_h3_model,
    spin_one_xy_periodic_range_couplings,
    spin_one_xy_phase_compatibility,
    spin_one_xy_scar_tower_states,
    spin_one_xy_tower_thermal_activities,
)


def test_fixed_magnetization_dimension() -> None:
    assert spin_one_xy_fixed_magnetization_dimension(2, 0) == 3
    assert spin_one_xy_fixed_magnetization_dimension(3, 3) == 1
    assert spin_one_xy_fixed_magnetization_dimension(3, 4) == 0


def test_exact_tower_thermal_activities_match_direct_counting() -> None:
    report = spin_one_xy_tower_thermal_activities(
        length=4,
        total_sz=0,
        xy_matrix_element=2.0,
    )
    assert report.sector_dimension == 19
    assert report.one_zero_count == 7
    assert report.two_site_remainder_count == 3
    assert np.isclose(report.y2_activity, 7.0 / 19.0)
    assert np.isclose(report.directed_q_activity, 24.0 / 19.0)
    assert np.isclose(report.z2_activity, 48.0 / 19.0)
    assert np.isclose(report.directed_q_limit, 0.5 * report.z2_limit)
    assert report.p0_limit > 0.0


def test_periodic_odd_range_is_phase_compatible() -> None:
    couplings = spin_one_xy_periodic_range_couplings(
        length=8,
        distance=3,
        coefficient=0.4,
    )
    phases = (-1.0) ** np.arange(8)
    report = spin_one_xy_phase_compatibility(couplings, phases=phases)
    assert report.is_compatible
    assert report.max_residual < 1.0e-12

    broken = tuple((*couplings[:-1], (couplings[-1][0], couplings[-1][1], 0.4j)))
    broken_report = spin_one_xy_phase_compatibility(broken, phases=phases)
    assert not broken_report.is_compatible


def test_hxy_h3_model_uses_manuscript_coupling_convention() -> None:
    model = spin_one_xy_hxy_h3_model(
        length=8,
        j=1.0,
        j3=0.1,
        total_sz=-2,
    )
    assert model.boundary_condition.value == "periodic"
    assert np.isclose(model.j_xy, 2.0)
    assert len(model.extra_xy_couplings) == 8
    assert all(np.isclose(coupling[2], 0.2) for coupling in model.extra_xy_couplings)

    phases = (-1.0) ** np.arange(8)
    report = spin_one_xy_phase_compatibility(
        model.extra_xy_couplings,
        phases=phases,
    )
    assert report.is_compatible


def test_hxy_h3_imaginary_j2_family_preserves_pi_tower() -> None:
    length = 8
    model = spin_one_xy_hxy_h3_imaginary_j2_model(
        length=length,
        j=1.0,
        j3=0.1,
        kappa=0.07,
        total_sz=-2,
    )
    assert np.isclose(model.j_xy, 2.0)
    assert len(model.extra_xy_couplings) == 2 * length

    phases = (-1.0) ** np.arange(length)
    report = spin_one_xy_phase_compatibility(
        model.extra_xy_couplings,
        phases=phases,
    )
    assert report.is_compatible
    assert report.max_residual < 1.0e-12

    j2 = model.extra_xy_couplings[length:]
    assert all(np.isclose(coupling[2], 0.14j) for coupling in j2)

    build = model.build(builder="optimized", basis_solver="dfs", sort_basis=True)
    configs = basis_configs_from_build_result(build)
    tower, labels = spin_one_xy_scar_tower_states(
        basis_configs=configs, length=length, normalize=True
    )
    assert labels
    assert tower.shape[1] == 1
    assert np.linalg.norm(build.hamiltonian @ tower[:, 0]) < 1.0e-11

    reference = spin_one_xy_hxy_h3_imaginary_j2_model(
        length=length, j=1.0, j3=0.1, kappa=0.0, total_sz=-2
    )
    assert len(reference.extra_xy_couplings) == length
