from __future__ import annotations

from qlinks.lattice import BoundaryCondition, KagomeLattice


def test_kagome_periodic_geometry_counts() -> None:
    lattice = KagomeLattice(2, 3, boundary_condition=BoundaryCondition.PERIODIC)

    assert lattice.num_sites == 18
    assert lattice.num_links == 36
    assert len(lattice.qdm_plaquette_ids()) == 6
    assert len(lattice.qlm_plaquette_ids()) == 6

    degrees = [len(lattice.incident_links(site_id)) for site_id in range(lattice.num_sites)]
    assert set(degrees) == {4}

    hexagons = [lattice.plaquettes[pid] for pid in lattice.qdm_plaquette_ids()]
    assert all(plaquette.kind == "hexagon" for plaquette in hexagons)
    assert all(len(plaquette.links) == 6 for plaquette in hexagons)


def test_kagome_open_geometry_has_bulk_hexagon() -> None:
    lattice = KagomeLattice(2, 2, boundary_condition="open")

    assert lattice.num_sites == 12
    assert lattice.num_links == 17
    assert len(lattice.qdm_plaquette_ids()) == 1
    assert lattice.plaquettes[lattice.qdm_plaquette_ids()[0]].kind == "hexagon"


def test_kagome_site_id_periodic_wraps() -> None:
    lattice = KagomeLattice(2, 3, boundary_condition="periodic")

    assert lattice.site_id(2, 0, 0) == lattice.site_id(0, 0, 0)
    assert lattice.site_id(0, 3, 2) == lattice.site_id(0, 0, 2)
