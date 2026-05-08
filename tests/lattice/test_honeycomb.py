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
