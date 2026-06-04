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


def test_honeycomb_small_torus_keeps_cell_anchored_links() -> None:
    lattice = HoneycombLattice(
        2,
        2,
        boundary_condition="periodic",
    )

    # Each unit cell has one A site and three outgoing A-B links.
    assert lattice.num_links == 3 * lattice.lx * lattice.ly

    seen = {
        (
            tuple(int(v) for v in lattice.sites[int(link.source)].cell),
            str(link.kind),
        )
        for link in lattice.links
    }

    assert len(seen) == lattice.num_links


def test_honeycomb_small_torus_hexagons_have_six_distinct_links() -> None:
    lattice = HoneycombLattice(
        2,
        2,
        boundary_condition="periodic",
    )

    assert lattice.num_plaquettes == lattice.lx * lattice.ly

    for plaquette in lattice.plaquettes:
        assert plaquette.kind == "hexagon"
        assert len(plaquette.links) == 6
        assert len(set(int(link_id) for link_id in plaquette.links)) == 6


def test_honeycomb_small_torus_hexagon_links_are_unique_across_all_plaquettes() -> None:
    lattice = HoneycombLattice(
        2,
        2,
        boundary_condition="periodic",
    )

    for plaquette in lattice.plaquettes:
        assert plaquette.kind == "hexagon"
        assert len(plaquette.links) == 6
        assert len(set(int(link_id) for link_id in plaquette.links)) == 6
        assert len(plaquette.orientations) == 6
        assert set(int(o) for o in plaquette.orientations) <= {-1, 1}
