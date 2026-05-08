import numpy as np
import pytest

from qlinks.constraints import GaussLawConstraint
from qlinks.lattice import ChainLattice, SquareLattice
from qlinks.variables import LocalSpace, VariableLayout


def test_gauss_law_open_chain_site_1() -> None:
    lattice = ChainLattice(3, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    # Links:
    #   0: 0 -> 1
    #   1: 1 -> 2
    #
    # At site 1:
    #   incoming link 0 has sign +1
    #   outgoing link 1 has sign -1
    constraint = GaussLawConstraint.from_lattice_site(
        lattice=lattice,
        layout=layout,
        site_id=1,
        charge=0,
    )

    np.testing.assert_array_equal(constraint.link_ids, np.array([0, 1]))
    np.testing.assert_array_equal(constraint.signs, np.array([1, -1]))

    assert constraint.value(np.array([1, 1])) == 0
    assert constraint.is_satisfied(np.array([1, 1]))

    assert constraint.value(np.array([1, -1])) == 2
    assert not constraint.is_satisfied(np.array([1, -1]))


def test_gauss_law_boundary_site() -> None:
    lattice = ChainLattice(3, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    constraint = GaussLawConstraint.from_lattice_site(
        lattice=lattice,
        layout=layout,
        site_id=0,
        charge=-1,
    )

    assert constraint.value(np.array([1, 1])) == -1
    assert constraint.is_satisfied(np.array([1, 1]))


def test_gauss_law_all_sites() -> None:
    lattice = ChainLattice(3, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    constraints = GaussLawConstraint.all_sites(
        lattice=lattice,
        layout=layout,
        charges=np.array([-1, 0, 1]),
    )

    config = np.array([1, 1])

    assert len(constraints) == 3
    assert all(c.is_satisfied(config) for c in constraints)


def test_gauss_law_all_sites_rejects_bad_charge_shape() -> None:
    lattice = ChainLattice(3, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    with pytest.raises(ValueError, match="charges must have shape"):
        GaussLawConstraint.all_sites(lattice, layout, charges=np.array([0, 0]))


def test_gauss_law_square_lattice_site() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    # Site 0 has outgoing x-link 0 and outgoing y-link 1.
    constraint = GaussLawConstraint.from_lattice_site(
        lattice=lattice,
        layout=layout,
        site_id=0,
        charge=-2,
    )

    np.testing.assert_array_equal(constraint.link_ids, np.array([0, 1]))
    np.testing.assert_array_equal(constraint.signs, np.array([-1, -1]))

    assert constraint.is_satisfied(np.array([1, 1, -1, -1]))


def test_gauss_law_direct_constructor_rejects_bad_sign() -> None:
    lattice = ChainLattice(2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    with pytest.raises(ValueError, match="signs must be"):
        GaussLawConstraint(
            layout=layout,
            site_id=0,
            link_ids=np.array([0]),
            signs=np.array([0]),
            charge=0,
        )
        