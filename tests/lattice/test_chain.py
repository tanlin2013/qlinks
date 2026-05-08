import numpy as np
import pytest

from qlinks.lattice import BoundaryCondition, ChainLattice


def test_open_chain_counts() -> None:
    lattice = ChainLattice(4, boundary_condition=BoundaryCondition.OPEN)

    assert lattice.ndim == 1
    assert lattice.num_sites == 4
    assert lattice.num_links == 3
    assert lattice.num_plaquettes == 0
    assert lattice.boundary_condition == BoundaryCondition.OPEN


def test_open_chain_links() -> None:
    lattice = ChainLattice(4, boundary_condition="open")

    np.testing.assert_array_equal(
        lattice.link_endpoints,
        np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
            ]
        ),
    )

    assert not any(link.wrap for link in lattice.links)


def test_open_chain_incidence() -> None:
    lattice = ChainLattice(4, boundary_condition="open")

    expected = np.array(
        [
            [-1, 0, 0],
            [+1, -1, 0],
            [0, +1, -1],
            [0, 0, +1],
        ],
        dtype=np.int8,
    )

    np.testing.assert_array_equal(lattice.incidence_matrix().toarray(), expected)


def test_open_chain_neighbors() -> None:
    lattice = ChainLattice(4, boundary_condition="open")

    np.testing.assert_array_equal(lattice.neighbors(0), np.array([1]))
    np.testing.assert_array_equal(lattice.neighbors(1), np.array([0, 2]))
    np.testing.assert_array_equal(lattice.neighbors(2), np.array([1, 3]))
    np.testing.assert_array_equal(lattice.neighbors(3), np.array([2]))


def test_open_chain_translation() -> None:
    lattice = ChainLattice(4, boundary_condition="open")

    assert lattice.translate_site(0, (1,)) == 1
    assert lattice.translate_site(1, (1,)) == 2
    assert lattice.translate_site(3, (1,)) is None

    assert lattice.translate_site(0, (-1,)) is None
    assert lattice.translate_site(3, (-1,)) == 2


def test_periodic_chain_counts() -> None:
    lattice = ChainLattice(4, boundary_condition=BoundaryCondition.PERIODIC)

    assert lattice.num_sites == 4
    assert lattice.num_links == 4
    assert lattice.boundary_condition == BoundaryCondition.PERIODIC


def test_periodic_chain_links() -> None:
    lattice = ChainLattice(4, boundary_condition="periodic")

    np.testing.assert_array_equal(
        lattice.link_endpoints,
        np.array(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [3, 0],
            ]
        ),
    )

    assert lattice.links[-1].wrap


def test_periodic_chain_neighbors() -> None:
    lattice = ChainLattice(4, boundary_condition="periodic")

    np.testing.assert_array_equal(lattice.neighbors(0), np.array([1, 3]))
    np.testing.assert_array_equal(lattice.neighbors(3), np.array([0, 2]))


def test_periodic_chain_translation() -> None:
    lattice = ChainLattice(4, boundary_condition="periodic")

    assert lattice.translate_site(0, (1,)) == 1
    assert lattice.translate_site(3, (1,)) == 0

    assert lattice.translate_site(0, (-1,)) == 3
    assert lattice.translate_site(2, (-1,)) == 1


def test_chain_length_one_open() -> None:
    lattice = ChainLattice(1, boundary_condition="open")

    assert lattice.num_sites == 1
    assert lattice.num_links == 0
    assert lattice.translate_site(0, (1,)) is None


def test_chain_length_one_periodic_has_no_self_link() -> None:
    lattice = ChainLattice(1, boundary_condition="periodic")

    assert lattice.num_sites == 1
    assert lattice.num_links == 0
    assert lattice.translate_site(0, (1,)) == 0
    assert lattice.translate_site(0, (-1,)) == 0


def test_reject_bad_length() -> None:
    with pytest.raises(ValueError, match="positive"):
        ChainLattice(0)
