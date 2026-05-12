import numpy as np
import pytest

from qlinks.lattice import SquareLattice
from qlinks.operators import (
    ToricCodePlaquetteFluxOperator,
    ToricCodeStarFlipOperator,
)
from qlinks.operators.base import OperatorAction
from qlinks.variables import LocalSpace, VariableLayout


def test_toric_code_star_flip_affected_variables_are_incident_links() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    op = ToricCodeStarFlipOperator(
        layout=layout,
        lattice=lattice,
        site_id=0,
        coefficient=-1.0,
    )

    expected_link_ids = np.asarray(
        lattice.incident_links(0),
        dtype=np.int64,
    )
    expected_variables = np.asarray(
        [layout.link_variable_index(int(link_id)) for link_id in expected_link_ids],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(
        np.sort(op.affected_variables()),
        np.sort(expected_variables),
    )

    # On square PBC, each vertex has four incident links.
    assert op.affected_variables().shape == (4,)


def test_toric_code_star_flip_apply_flips_incident_links_only() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    op = ToricCodeStarFlipOperator(
        layout=layout,
        lattice=lattice,
        site_id=0,
        coefficient=-2.5,
    )

    config = np.ones(layout.n_variables, dtype=np.int64)

    actions = op.apply(config)

    assert len(actions) == 1
    assert isinstance(actions[0], OperatorAction)
    assert actions[0].coefficient == complex(-2.5)

    expected = config.copy()
    expected[op.affected_variables()] *= -1

    np.testing.assert_array_equal(actions[0].config, expected)

    # Input should not be mutated.
    np.testing.assert_array_equal(
        config,
        np.ones(layout.n_variables, dtype=np.int64),
    )


def test_toric_code_star_flip_is_involution_on_config() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    op = ToricCodeStarFlipOperator(
        layout=layout,
        lattice=lattice,
        site_id=0,
        coefficient=-1.0,
    )

    config = np.array(
        [1, -1, 1, -1, -1, 1, -1, 1],
        dtype=np.int64,
    )

    first = op.apply(config)[0].config
    second = op.apply(first)[0].config

    np.testing.assert_array_equal(second, config)


@pytest.mark.parametrize(
    "config",
    [
        np.ones(8, dtype=np.int64),
        np.array([-1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64),
        np.array([1, -1, -1, 1, -1, 1, 1, -1], dtype=np.int64),
    ],
)
def test_toric_code_plaquette_flux_apply_is_diagonal(config: np.ndarray) -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    op = ToricCodePlaquetteFluxOperator(
        layout=layout,
        lattice=lattice,
        plaquette_id=0,
        coefficient=-3.0,
    )

    expected_flux = int(np.prod(config[op.affected_variables()]))

    actions = op.apply(config)

    assert len(actions) == 1
    assert isinstance(actions[0], OperatorAction)

    np.testing.assert_array_equal(actions[0].config, config)
    assert actions[0].coefficient == complex(-3.0 * expected_flux)


def test_toric_code_plaquette_flux_affected_variables_are_plaquette_links() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    plaquette_id = 0
    op = ToricCodePlaquetteFluxOperator(
        layout=layout,
        lattice=lattice,
        plaquette_id=plaquette_id,
        coefficient=-1.0,
    )

    plaquette = lattice.plaquettes[plaquette_id]
    expected_variables = np.asarray(
        [layout.link_variable_index(int(link_id)) for link_id in plaquette.links],
        dtype=np.int64,
    )

    np.testing.assert_array_equal(
        np.sort(op.affected_variables()),
        np.sort(expected_variables),
    )

    assert op.affected_variables().shape == (4,)


def test_toric_code_star_flip_rejects_non_flux_local_space() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    with pytest.raises(ValueError, match=r"\{-1, \+1\}"):
        ToricCodeStarFlipOperator(
            layout=layout,
            lattice=lattice,
            site_id=0,
            coefficient=-1.0,
        )


def test_toric_code_plaquette_flux_rejects_non_flux_local_space() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    with pytest.raises(ValueError, match=r"\{-1, \+1\}"):
        ToricCodePlaquetteFluxOperator(
            layout=layout,
            lattice=lattice,
            plaquette_id=0,
            coefficient=-1.0,
        )


def test_toric_code_star_flip_validates_config_values() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    op = ToricCodeStarFlipOperator(
        layout=layout,
        lattice=lattice,
        site_id=0,
        coefficient=-1.0,
    )

    bad_config = np.zeros(layout.n_variables, dtype=np.int64)

    with pytest.raises(ValueError):
        op.apply(bad_config)


def test_toric_code_plaquette_flux_validates_config_values() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.spin_half_flux(),
    )

    op = ToricCodePlaquetteFluxOperator(
        layout=layout,
        lattice=lattice,
        plaquette_id=0,
        coefficient=-1.0,
    )

    bad_config = np.zeros(layout.n_variables, dtype=np.int64)

    with pytest.raises(ValueError):
        op.apply(bad_config)
