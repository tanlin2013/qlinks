import numpy as np

from qlinks.constraints import NearestNeighborBlockadeConstraint
from qlinks.lattice import ChainLattice, SquareLattice
from qlinks.variables import LocalSpace, VariableLayout


def test_single_blockade_constraint() -> None:
    lattice = ChainLattice(2, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())

    constraint = NearestNeighborBlockadeConstraint(
        layout=layout,
        site_i=0,
        site_j=1,
        occupied_value=1,
    )

    assert constraint.is_satisfied(np.array([0, 0]))
    assert constraint.is_satisfied(np.array([1, 0]))
    assert constraint.is_satisfied(np.array([0, 1]))
    assert not constraint.is_satisfied(np.array([1, 1]))

    np.testing.assert_array_equal(constraint.affected_variables(), np.array([0, 1]))


def test_blockade_constraints_from_chain() -> None:
    lattice = ChainLattice(4, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())

    constraints = NearestNeighborBlockadeConstraint.from_lattice(lattice, layout)

    assert len(constraints) == 3

    valid = np.array([1, 0, 1, 0])
    invalid = np.array([1, 1, 0, 0])

    assert all(c.is_satisfied(valid) for c in constraints)
    assert not all(c.is_satisfied(invalid) for c in constraints)


def test_blockade_constraints_from_square() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())

    constraints = NearestNeighborBlockadeConstraint.from_lattice(lattice, layout)

    assert len(constraints) == lattice.num_links

    valid = np.array([1, 0, 0, 1])
    invalid = np.array([1, 1, 0, 0])

    assert all(c.is_satisfied(valid) for c in constraints)
    assert not all(c.is_satisfied(invalid) for c in constraints)
