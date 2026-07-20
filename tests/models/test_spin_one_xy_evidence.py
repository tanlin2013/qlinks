from __future__ import annotations

import numpy as np

from qlinks.models import (
    spin_one_xy_fixed_magnetization_dimension,
    spin_one_xy_periodic_range_couplings,
    spin_one_xy_phase_compatibility,
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
    assert np.isclose(report.z2_activity, 48.0 / 19.0)
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
