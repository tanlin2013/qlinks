import numpy as np
import pytest

from qlinks.constraints import DimerCoveringConstraint
from qlinks.lattice import ChainLattice, SquareLattice
from qlinks.variables import LocalSpace, VariableLayout


def test_dimer_constraint_open_chain_middle_site() -> None:
    lattice = ChainLattice(3, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    constraint = DimerCoveringConstraint.from_lattice_site(
        lattice=lattice,
        layout=layout,
        site_id=1,
        required_count=1,
    )

    np.testing.assert_array_equal(constraint.link_ids, np.array([0, 1]))

    assert constraint.is_satisfied(np.array([1, 0]))
    assert constraint.is_satisfied(np.array([0, 1]))
    assert not constraint.is_satisfied(np.array([1, 1]))
    assert not constraint.is_satisfied(np.array([0, 0]))


def test_dimer_constraint_all_sites_chain_perfect_matching() -> None:
    lattice = ChainLattice(4, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    constraints = DimerCoveringConstraint.all_sites(
        lattice=lattice,
        layout=layout,
        required_counts=1,
    )

    # Perfect matching on links 0 and 2.
    config = np.array([1, 0, 1])

    assert len(constraints) == 4
    assert all(c.is_satisfied(config) for c in constraints)


def test_dimer_constraint_square_site() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    constraint = DimerCoveringConstraint.from_lattice_site(
        lattice=lattice,
        layout=layout,
        site_id=0,
        required_count=1,
    )

    assert constraint.is_satisfied(np.array([1, 0, 0, 0]))
    assert constraint.is_satisfied(np.array([0, 1, 0, 0]))
    assert not constraint.is_satisfied(np.array([1, 1, 0, 0]))


def test_dimer_all_sites_rejects_bad_shape() -> None:
    lattice = ChainLattice(3, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    with pytest.raises(ValueError, match="required_counts must have shape"):
        DimerCoveringConstraint.all_sites(
            lattice=lattice,
            layout=layout,
            required_counts=np.array([1, 1]),
        )
