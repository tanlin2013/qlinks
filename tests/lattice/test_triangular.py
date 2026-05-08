import numpy as np

from qlinks.lattice import TriangularLattice


def test_triangular_open_counts_2_by_2() -> None:
    lattice = TriangularLattice(2, 2, boundary_condition="open")

    assert lattice.num_sites == 4
    assert lattice.num_links > 0
    assert lattice.num_plaquettes > 0
    assert len(lattice.qdm_plaquette_ids()) > 0


def test_triangular_periodic_counts() -> None:
    lattice = TriangularLattice(3, 3, boundary_condition="periodic")

    assert lattice.num_sites == 9
    assert lattice.num_links == 27

    for site_id in lattice.site_ids:
        assert lattice.neighbors(int(site_id)).size == 6


def test_triangular_rhombi_are_length_four() -> None:
    lattice = TriangularLattice(3, 3, boundary_condition="open")

    for pid in lattice.qdm_plaquette_ids():
        assert lattice.plaquette_links(pid).size == 4
