import numpy as np

from qlinks.constraints import (
    GaussLawConstraint,
    HoneycombElectricWindingSector,
    SquareQDMElectricWindingSector,
    SquareWindingSector,
)
from qlinks.lattice import HoneycombLattice, SquareLattice
from qlinks.models import SquareQLMModel
from qlinks.variables import LocalSpace, VariableLayout
from tests.helpers.constraints import (
    assert_partial_check_matches_full_check_on_complete_configs,
    first_allowed_target,
)


def test_gauss_law_partial_check_matches_full_check_on_complete_configs() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    constraints = GaussLawConstraint.all_sites(
        layout=layout,
        lattice=lattice,
        charges=0,
        charge_normalization="spin_half",
    )

    configs = np.array(
        [
            np.ones(layout.n_variables, dtype=np.int64),
            -np.ones(layout.n_variables, dtype=np.int64),
            np.array([1, -1, 1, -1, -1, 1, -1, 1], dtype=np.int64),
        ],
        dtype=np.int64,
    )

    for constraint in constraints:
        assert_partial_check_matches_full_check_on_complete_configs(
            constraint,
            configs,
        )


def test_square_winding_partial_check_matches_full_check_on_complete_configs() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    sectors = (
        SquareWindingSector(
            layout=layout,
            lattice=lattice,
            direction="x",
            target=0,
            flux_normalization="spin_half",
        ),
        SquareWindingSector(
            layout=layout,
            lattice=lattice,
            direction="y",
            target=0,
            flux_normalization="spin_half",
        ),
    )

    configs = np.array(
        [
            np.ones(layout.n_variables, dtype=np.int64),
            -np.ones(layout.n_variables, dtype=np.int64),
            np.array([1, -1, 1, -1, -1, 1, -1, 1], dtype=np.int64),
        ],
        dtype=np.int64,
    )

    for sector in sectors:
        assert_partial_check_matches_full_check_on_complete_configs(
            sector,
            configs,
        )


def test_square_qdm_electric_winding_partial_check_matches_full_check_on_complete_configs() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    sectors = (
        SquareQDMElectricWindingSector(
            layout=layout,
            lattice=lattice,
            direction="x",
            target=0,
        ),
        SquareQDMElectricWindingSector(
            layout=layout,
            lattice=lattice,
            direction="y",
            target=0,
        ),
    )

    configs = np.array(
        [
            np.zeros(layout.n_variables, dtype=np.int64),
            np.ones(layout.n_variables, dtype=np.int64),
            np.array([1, 0, 1, 0, 0, 1, 0, 1], dtype=np.int64),
        ],
        dtype=np.int64,
    )

    for sector in sectors:
        assert_partial_check_matches_full_check_on_complete_configs(
            sector,
            configs,
        )


def test_honeycomb_electric_winding_partial_check_matches_full_check_on_complete_configs() -> None:
    lattice = HoneycombLattice(
        3,
        3,
        boundary_condition="periodic",
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    target_x = first_allowed_target(
        HoneycombElectricWindingSector,
        layout=layout,
        lattice=lattice,
        direction="x",
        value_convention="flux_pm",
        flux_normalization="spin_half",
    )
    target_y = first_allowed_target(
        HoneycombElectricWindingSector,
        layout=layout,
        lattice=lattice,
        direction="y",
        value_convention="flux_pm",
        flux_normalization="spin_half",
    )

    sectors = (
        HoneycombElectricWindingSector(
            layout=layout,
            lattice=lattice,
            direction="x",
            target=target_x,
            value_convention="flux_pm",
            flux_normalization="spin_half",
        ),
        HoneycombElectricWindingSector(
            layout=layout,
            lattice=lattice,
            direction="y",
            target=target_y,
            value_convention="flux_pm",
            flux_normalization="spin_half",
        ),
    )

    configs = np.array(
        [
            np.ones(layout.n_variables, dtype=np.int64),
            -np.ones(layout.n_variables, dtype=np.int64),
        ],
        dtype=np.int64,
    )

    for sector in sectors:
        assert_partial_check_matches_full_check_on_complete_configs(
            sector,
            configs,
        )


def test_square_qlm_dfs_matches_bruteforce_with_orientation_aware_winding() -> None:
    model = SquareQLMModel(
        lx=2,
        ly=2,
        boundary_condition="periodic",
        charges=0,
        winding_x=0,
        winding_y=0,
    )

    basis_dfs = model.build_basis(
        solver="dfs",
        sort=True,
    )

    basis_brute = model.build_basis(
        solver="brute_force",
        sort=True,
    )

    np.testing.assert_array_equal(
        basis_dfs.states,
        basis_brute.states,
    )
