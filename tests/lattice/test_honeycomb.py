import numpy as np

from qlinks.lattice import HoneycombLattice


def test_honeycomb_open_counts() -> None:
    lattice = HoneycombLattice(2, 2, boundary_condition="open")

    assert lattice.num_sites == 8
    assert lattice.num_links > 0
    assert lattice.num_plaquettes == 1
    assert lattice.plaquette_links(0).size == 6


def test_honeycomb_hexagons_are_length_six() -> None:
    lattice = HoneycombLattice(2, 2, boundary_condition="open")

    assert lattice.plaquette_links(0).size == 6


def test_honeycomb_periodic_site_degrees() -> None:
    lattice = HoneycombLattice(3, 3, boundary_condition="periodic")

    assert lattice.num_sites == 18
    assert lattice.num_links == 27

    for site_id in lattice.site_ids:
        assert lattice.neighbors(int(site_id)).size == 3


def test_honeycomb_embedding_has_equal_neighbor_lengths() -> None:
    lattice = HoneycombLattice(4, 4, boundary_condition="open")

    lengths = []

    for link in lattice.links:
        p0 = np.asarray(lattice.site_embedded_position(int(link.source)), dtype=float)
        p1 = np.asarray(lattice.site_embedded_position(int(link.target)), dtype=float)
        lengths.append(np.linalg.norm(p1 - p0))

    lengths = np.asarray(lengths)

    np.testing.assert_allclose(lengths, np.ones_like(lengths), atol=1e-12)


def test_honeycomb_embedding_primitive_vectors_are_nondegenerate() -> None:
    lattice = HoneycombLattice(4, 4, boundary_condition="periodic")

    vectors = np.asarray(lattice.primitive_vectors, dtype=float)

    assert vectors.shape == (2, 2)
    assert abs(np.linalg.det(vectors)) > 1e-12
