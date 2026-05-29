from fractions import Fraction

import numpy as np
import pytest

from qlinks.constraints import (
    HoneycombElectricWindingSector,
    SquareQDMElectricWindingSector,
    SquareWindingSector,
)
from qlinks.constraints.winding import raw_targets_from_user_targets
from qlinks.lattice import HoneycombLattice, SquareLattice
from qlinks.models import SquareQDMModel
from qlinks.variables import LocalSpace, VariableLayout


def _sector_covector(lattice, sector) -> np.ndarray:
    covector = np.zeros(lattice.num_links, dtype=np.int64)
    covector[sector.link_ids] = sector.signs
    return covector


@pytest.mark.parametrize("direction", ["x", "y"])
def test_square_winding_sector_annihilates_plaquette_boundaries_2_by_2(
    direction: str,
) -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )
    sector = SquareWindingSector(
        layout=layout,
        lattice=lattice,
        direction=direction,
        target=0,
    )

    covector = _sector_covector(lattice, sector)
    plaquette_incidence = lattice.plaquette_incidence_matrix().toarray()

    np.testing.assert_array_equal(
        covector @ plaquette_incidence,
        np.zeros(lattice.num_plaquettes, dtype=np.int64),
    )


def test_square_winding_sector_target_satisfaction() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )
    sector_probe = SquareWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target=0,
        flux_normalization="integer_flux",
    )

    config = np.ones(lattice.num_links, dtype=np.int64)
    actual_value = sector_probe.value(config)

    matching_sector = SquareWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target=actual_value,
        flux_normalization="integer_flux",
    )
    nonmatching_sector = SquareWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target=actual_value + 2,
        flux_normalization="integer_flux",
    )

    assert matching_sector.is_satisfied(config)
    assert not nonmatching_sector.is_satisfied(config)


def test_square_winding_requires_periodic_lattice() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    with pytest.raises(ValueError, match="requires a periodic"):
        SquareWindingSector(
            layout=layout,
            lattice=lattice,
            direction="x",
            target=0,
        )


def test_square_winding_spin_half_allows_fractional_targets() -> None:
    lattice = SquareLattice(3, 3, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    allowed = SquareWindingSector.allowed_targets(
        layout=layout,
        lattice=lattice,
        direction="x",
        flux_normalization="spin_half",
    )

    assert Fraction(1, 2) in allowed
    assert Fraction(3, 2) in allowed


def test_square_winding_integer_flux_uses_raw_targets() -> None:
    lattice = SquareLattice(3, 3, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    allowed = SquareWindingSector.allowed_targets(
        layout=layout,
        lattice=lattice,
        direction="x",
        flux_normalization="integer_flux",
    )

    assert Fraction(1, 1) in allowed
    assert Fraction(3, 1) in allowed


def test_honeycomb_electric_winding_binary_value() -> None:
    lattice = HoneycombLattice(
        3,
        3,
        boundary_condition="periodic",
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    sector = HoneycombElectricWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target=1,
        value_convention="binary",
    )

    config = layout.default_config()

    # Binary default config is usually all zeros.
    # For binary QDM convention, E = 2n - 1, so each selected cut link
    # contributes -1.
    expected = -sector.affected_variables().size

    assert sector.value(config) == expected
    assert sector.is_satisfied(config) == (expected == 0)


def test_honeycomb_electric_winding_binary_all_occupied() -> None:
    lattice = HoneycombLattice(
        3,
        3,
        boundary_condition="periodic",
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    sector = HoneycombElectricWindingSector(
        layout=layout,
        lattice=lattice,
        direction="y",
        target=1,
        value_convention="binary",
    )

    config = np.ones(layout.n_variables, dtype=np.int64)

    # For binary convention, E = 2n - 1. If n = 1, every selected cut link
    # contributes +1.
    expected = sector.affected_variables().size

    assert sector.value(config) == expected
    assert sector.is_satisfied(config) == (expected == 0)


def test_honeycomb_electric_winding_flux_pm_value() -> None:
    lattice = HoneycombLattice(
        3,
        3,
        boundary_condition="periodic",
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    sector = HoneycombElectricWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target="3/2",
        value_convention="flux_pm",
        flux_normalization="spin_half",
    )

    config = np.full(layout.n_variables, -1, dtype=np.int64)

    expected = -sector.affected_variables().size

    assert sector.value(config) == expected


def test_honeycomb_electric_winding_flux_pm_all_positive() -> None:
    lattice = HoneycombLattice(
        3,
        3,
        boundary_condition="periodic",
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    sector = HoneycombElectricWindingSector(
        layout=layout,
        lattice=lattice,
        direction="y",
        target="3/2",
        value_convention="flux_pm",
        flux_normalization="spin_half",
    )

    config = np.ones(layout.n_variables, dtype=np.int64)

    expected = sector.affected_variables().size

    assert sector.value(config) == expected


def test_honeycomb_flux_pm_spin_half_allows_fractional_targets() -> None:
    lattice = HoneycombLattice(3, 3, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    allowed = HoneycombElectricWindingSector.allowed_targets(
        layout=layout,
        lattice=lattice,
        direction="y",
        value_convention="flux_pm",
        flux_normalization="spin_half",
    )

    assert Fraction(3, 2) in allowed
    assert Fraction(1, 2) in allowed
    assert 1 not in allowed


def test_honeycomb_flux_pm_integer_flux_uses_raw_targets() -> None:
    lattice = HoneycombLattice(3, 3, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    allowed = HoneycombElectricWindingSector.allowed_targets(
        layout=layout,
        lattice=lattice,
        direction="y",
        value_convention="flux_pm",
        flux_normalization="integer_flux",
    )

    assert allowed == (Fraction(-3, 1), Fraction(-1, 1), Fraction(1, 1), Fraction(3, 1))


def test_honeycomb_electric_winding_affected_variables_are_nonempty() -> None:
    lattice = HoneycombLattice(
        3,
        3,
        boundary_condition="periodic",
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    sector_x = HoneycombElectricWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target=1,
        value_convention="binary",
    )

    sector_y = HoneycombElectricWindingSector(
        layout=layout,
        lattice=lattice,
        direction="y",
        target=1,
        value_convention="binary",
    )

    assert sector_x.affected_variables().size > 0
    assert sector_y.affected_variables().size > 0

    assert np.all(sector_x.affected_variables() >= 0)
    assert np.all(sector_y.affected_variables() >= 0)

    assert np.all(sector_x.affected_variables() < layout.n_variables)
    assert np.all(sector_y.affected_variables() < layout.n_variables)


def test_honeycomb_electric_winding_target_satisfaction() -> None:
    lattice = HoneycombLattice(
        3,
        3,
        boundary_condition="periodic",
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    config = layout.default_config()

    probe_sector = HoneycombElectricWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target=1,
        value_convention="binary",
    )

    actual_value = probe_sector.value(config)

    matching_sector = HoneycombElectricWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target=actual_value,
        value_convention="binary",
    )

    nonmatching_sector = HoneycombElectricWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target=actual_value + 2,
        value_convention="binary",
    )

    assert matching_sector.is_satisfied(config)
    assert not nonmatching_sector.is_satisfied(config)


def test_honeycomb_electric_winding_partial_check_fully_assigned() -> None:
    lattice = HoneycombLattice(
        3,
        3,
        boundary_condition="periodic",
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    config = layout.default_config()

    probe_sector = HoneycombElectricWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target=1,
        value_convention="binary",
    )

    actual_value = probe_sector.value(config)

    sector = HoneycombElectricWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target=actual_value,
        value_convention="binary",
    )

    assigned_mask = np.ones(layout.n_variables, dtype=bool)

    assert sector.partial_check(config, assigned_mask)


def test_honeycomb_electric_winding_partial_check_fully_assigned_rejects() -> None:
    lattice = HoneycombLattice(
        3,
        3,
        boundary_condition="periodic",
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    config = layout.default_config()

    probe_sector = HoneycombElectricWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target=1,
        value_convention="binary",
    )

    actual_value = probe_sector.value(config)

    sector = HoneycombElectricWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target=actual_value + 2,
        value_convention="binary",
    )

    assigned_mask = np.ones(layout.n_variables, dtype=bool)

    assert not sector.partial_check(config, assigned_mask)


def test_honeycomb_electric_winding_partial_check_not_fully_assigned_can_still_pass() -> None:
    lattice = HoneycombLattice(
        3,
        3,
        boundary_condition="periodic",
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    sector = HoneycombElectricWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target=1,
        value_convention="binary",
    )

    config = layout.default_config()
    assigned_mask = np.zeros(layout.n_variables, dtype=bool)

    # With no cut variables assigned, target 0 should still be reachable
    # as long as the cut has even length. For a 3x3 honeycomb, this mainly
    # checks that the partial_check method does not over-prune.
    assert sector.partial_check(config, assigned_mask)


def test_honeycomb_electric_winding_requires_periodic_boundary() -> None:
    lattice = HoneycombLattice(
        3,
        3,
        boundary_condition="open",
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    with pytest.raises(ValueError, match="PBC"):
        HoneycombElectricWindingSector(
            layout=layout,
            lattice=lattice,
            direction="x",
            target=1,
            value_convention="binary",
        )


def test_honeycomb_electric_winding_rejects_bad_direction() -> None:
    lattice = HoneycombLattice(
        3,
        3,
        boundary_condition="periodic",
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    with pytest.raises(ValueError, match="direction"):
        HoneycombElectricWindingSector(
            layout=layout,
            lattice=lattice,
            direction="z",  # type: ignore[arg-type]
            target=1,
            value_convention="binary",
        )


def test_honeycomb_electric_winding_rejects_bad_value_convention() -> None:
    lattice = HoneycombLattice(
        3,
        3,
        boundary_condition="periodic",
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    with pytest.raises(ValueError, match="value_convention"):
        HoneycombElectricWindingSector(
            layout=layout,
            lattice=lattice,
            direction="x",
            target=1,
            value_convention="bad",  # type: ignore[arg-type]
        )


def test_square_winding_sector_spin_half_target() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )
    probe_sector = SquareWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target=0,
        flux_normalization="spin_half",
    )

    config = np.ones(layout.n_variables, dtype=np.int64)
    config[probe_sector.variable_indices] = probe_sector.signs

    raw_value = probe_sector.value(config)
    assert raw_value % 2 == 0

    matching_sector = SquareWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target=raw_value // 2,
        flux_normalization="spin_half",
    )

    assert matching_sector.is_satisfied(config)


def _winding_covector(
    lattice: SquareLattice,
    sector: SquareWindingSector,
) -> np.ndarray:
    covector = np.zeros(lattice.num_links, dtype=np.int64)
    covector[sector.link_ids] = 1
    return covector


@pytest.mark.parametrize(
    ("lattice_size_x", "lattice_size_y"),
    [
        (2, 2),
        (4, 4),
    ],
)
def test_square_winding_sector_annihilates_plaquette_boundaries_pbc(
    lattice_size_x: int,
    lattice_size_y: int,
) -> None:
    lattice = SquareLattice(lattice_size_x, lattice_size_y, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    plaquette_incidence = lattice.plaquette_incidence_matrix().toarray()

    for direction in ("x", "y"):
        sector = SquareWindingSector(
            layout=layout,
            lattice=lattice,
            direction=direction,
            target=0,
        )

        covector = np.zeros(lattice.num_links, dtype=np.int64)
        covector[sector.link_ids] = sector.signs

        np.testing.assert_array_equal(
            covector @ plaquette_incidence,
            np.zeros(lattice.num_plaquettes, dtype=np.int64),
        )


@pytest.mark.parametrize("builder_name", ["sparse", "bitmask"])
@pytest.mark.parametrize(
    ("lattice_size_x", "lattice_size_y"),
    [
        (2, 2),
        (4, 4),
    ],
)
def test_square_qdm_electric_winding_sector_preserves_plaquette_flips(
    builder_name: str,
    lattice_size_x: int,
    lattice_size_y: int,
) -> None:
    """QDM plaquette flips should preserve the selected electric winding sector."""
    model = SquareQDMModel(
        lx=lattice_size_x,
        ly=lattice_size_y,
        boundary_condition="periodic",
        winding_x=0,
        winding_y=0,
        winding_convention="electric",
        coup_kin=1.0,
        coup_pot=0.0,
    )

    build_result = model.build(
        basis_solver="dfs",
        builder=builder_name,
        backend="scipy",
        sort_basis=True,
        on_missing="raise",
    )

    assert build_result.kinetic is not None
    assert build_result.kinetic.nnz > 0


def test_square_qdm_electric_winding_has_signed_covector_2x2_pbc() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    for direction in ("x", "y"):
        sector = SquareQDMElectricWindingSector(
            layout=layout,
            lattice=lattice,
            direction=direction,
            target=0,
        )

        assert sector.link_ids.size > 0
        assert sector.signs.size == sector.link_ids.size
        assert set(sector.signs.tolist()) <= {-1, 1}


def test_square_winding_allowed_targets_spin_half_are_user_facing() -> None:
    lattice = SquareLattice(4, 4, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    allowed = SquareWindingSector.allowed_targets(
        layout=layout,
        lattice=lattice,
        direction="x",
        flux_normalization="spin_half",
    )

    assert 0 in allowed

    raw_allowed = SquareWindingSector.allowed_internal_targets(
        layout=layout,
        lattice=lattice,
        direction="x",
    )

    assert set(
        raw_targets_from_user_targets(
            allowed,
            flux_normalization="spin_half",
        )
    ) <= set(raw_allowed)


def test_square_winding_rejects_illegal_target() -> None:
    lattice = SquareLattice(4, 4, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    allowed = SquareWindingSector.allowed_targets(
        layout=layout,
        lattice=lattice,
        direction="x",
        flux_normalization="spin_half",
    )

    illegal = max(allowed) + 1

    with pytest.raises(ValueError, match="Illegal"):
        SquareWindingSector(
            layout=layout,
            lattice=lattice,
            direction="x",
            target=illegal,
            flux_normalization="spin_half",
        )


def test_square_qdm_electric_winding_allowed_targets() -> None:
    lattice = SquareLattice(4, 4, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    allowed = SquareQDMElectricWindingSector.allowed_targets(
        layout=layout,
        lattice=lattice,
        direction="x",
    )

    assert 0 in allowed


def test_honeycomb_electric_winding_allowed_targets() -> None:
    lattice = HoneycombLattice(3, 3, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    allowed = HoneycombElectricWindingSector.allowed_targets(
        layout=layout,
        lattice=lattice,
        direction="x",
        value_convention="flux_pm",
        flux_normalization="spin_half",
    )

    assert isinstance(allowed, tuple)
