import numpy as np
import pytest

from qlinks.lattice import SquareLattice
from qlinks.operators import (
    PlaquettePatternOperator,
    PlaquettePatternTransition,
    qdm_flippability_projectors,
)
from qlinks.variables import LocalSpace, VariableLayout


def test_qdm_plaquette_flip_2_by_2_open() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    op = PlaquettePatternOperator.qdm_flip(
        layout=layout,
        lattice=lattice,
        plaquette_id=0,
        coefficient=-1.0,
    )

    # For 2x2 open square, plaquette link order is [0, 3, 2, 1].
    # To make local plaquette values 1010, set:
    #   link 0 = 1
    #   link 3 = 0
    #   link 2 = 1
    #   link 1 = 0
    config = np.array([1, 0, 1, 0])

    actions = op.apply(config)

    assert len(actions) == 1
    assert actions[0].coefficient == -1.0 + 0j

    # Local 1010 -> 0101 in plaquette order [0, 3, 2, 1].
    # Therefore:
    #   link 0 -> 0
    #   link 3 -> 1
    #   link 2 -> 0
    #   link 1 -> 1
    np.testing.assert_array_equal(actions[0].config, np.array([0, 1, 0, 1]))


def test_qdm_plaquette_flip_reverse() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    op = PlaquettePatternOperator.qdm_flip(
        layout=layout,
        lattice=lattice,
        plaquette_id=0,
    )

    config = np.array([0, 1, 0, 1])
    actions = op.apply(config)

    assert len(actions) == 1
    np.testing.assert_array_equal(actions[0].config, np.array([1, 0, 1, 0]))


def test_qdm_plaquette_flip_no_match() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    op = PlaquettePatternOperator.qdm_flip(
        layout=layout,
        lattice=lattice,
        plaquette_id=0,
    )

    actions = op.apply(np.array([1, 1, 0, 0]))

    assert actions == ()


def test_custom_plaquette_pattern_transition() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    transition = PlaquettePatternTransition(
        initial=np.array([0, 0, 0, 0]),
        final=np.array([1, 1, 1, 1]),
        coefficient=2.0,
    )

    op = PlaquettePatternOperator(
        layout=layout,
        lattice=lattice,
        plaquette_id=0,
        transitions=(transition,),
    )

    actions = op.apply(np.array([0, 0, 0, 0]))

    assert len(actions) == 1
    assert actions[0].coefficient == 2.0 + 0j
    np.testing.assert_array_equal(actions[0].config, np.array([1, 1, 1, 1]))


def test_plaquette_pattern_rejects_wrong_pattern_length() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    transition = PlaquettePatternTransition(
        initial=np.array([0, 0]),
        final=np.array([1, 1]),
    )

    with pytest.raises(ValueError, match="wrong length"):
        PlaquettePatternOperator(
            layout=layout,
            lattice=lattice,
            plaquette_id=0,
            transitions=(transition,),
        )


def test_qdm_flippability_projectors() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    projectors = qdm_flippability_projectors(
        layout=layout,
        lattice=lattice,
        plaquette_id=0,
        coefficient=3.0,
    )

    assert len(projectors) == 2

    config_1010 = np.array([1, 0, 1, 0])
    config_0101 = np.array([0, 1, 0, 1])
    config_other = np.array([1, 1, 0, 0])

    assert len(projectors[0].apply(config_1010)) == 1
    assert projectors[0].apply(config_1010)[0].coefficient == 3.0 + 0j

    assert len(projectors[1].apply(config_0101)) == 1
    assert projectors[1].apply(config_0101)[0].coefficient == 3.0 + 0j

    assert projectors[0].apply(config_other) == ()
    assert projectors[1].apply(config_other) == ()


def test_qdm_flip_uses_local_visual_square_plaquette_on_thin_torus() -> None:
    lattice = SquareLattice(4, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    plaquette_id = lattice.plaquette_id_from_cell(1, 1)

    operator = PlaquettePatternOperator.qdm_flip(
        layout=layout,
        lattice=lattice,
        plaquette_id=plaquette_id,
    )

    config = np.zeros(layout.n_variables, dtype=np.int64)

    # Pattern 1010 on links (6, 11, 4, 7).
    for link_id, value in zip((6, 11, 4, 7), (1, 0, 1, 0), strict=True):
        config[layout.link_variable_index(link_id)] = value

    actions = operator.apply(config)

    assert len(actions) == 1

    new_config = actions[0].config

    for link_id, expected in zip((6, 11, 4, 7), (0, 1, 0, 1), strict=True):
        assert new_config[layout.link_variable_index(link_id)] == expected

    # The old wrong edge should not have been touched.
    assert new_config[layout.link_variable_index(5)] == config[layout.link_variable_index(5)]
