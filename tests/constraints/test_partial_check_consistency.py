import numpy as np
import numpy.typing as npt

from qlinks.constraints import (
    GaussLawConstraint,
    SquareWindingSector,
    SquareQDMElectricWindingSector,
    HoneycombElectricWindingSector,
)
from qlinks.lattice import SquareLattice, HoneycombLattice
from qlinks.variables import LocalSpace, VariableLayout
from qlinks.models import SquareQLMModel


def assert_partial_check_matches_full_check(
    condition,
    configs: npt.ArrayLike,
) -> None:
    states = np.asarray(configs, dtype=np.int64)

    if states.ndim == 1:
        states = states.reshape(1, -1)

    for config in states:
        assigned_mask = np.ones(config.shape, dtype=bool)

        partial = condition.partial_check(config, assigned_mask)
        full = condition.is_satisfied(config)

        assert partial == full, (
            f"{condition.name}: partial_check disagrees with is_satisfied "
            f"on complete config {config}. "
            f"partial={partial}, full={full}"
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
        assert_partial_check_matches_full_check(
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
        assert_partial_check_matches_full_check(
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
        assert_partial_check_matches_full_check(
            sector,
            configs,
        )


def test_honeycomb_electric_winding_partial_check_matches_full_check_on_complete_configs() -> None:
    lattice = HoneycombLattice(3, 3, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    sectors = (
        HoneycombElectricWindingSector(
            layout=layout,
            lattice=lattice,
            direction="x",
            target=0,
            flux_normalization="spin_half",
        ),
        HoneycombElectricWindingSector(
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
        ],
        dtype=np.int64,
    )

    for sector in sectors:
        assert_partial_check_matches_full_check(
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
