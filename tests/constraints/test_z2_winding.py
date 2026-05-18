import numpy as np

from qlinks.constraints import TriangularZ2WindingSector
from qlinks.lattice import TriangularLattice
from qlinks.variables import LocalSpace, VariableLayout


def test_triangular_z2_winding_binary_value() -> None:
    lattice = TriangularLattice(3, 3, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    sector = TriangularZ2WindingSector(
        layout=layout,
        lattice=lattice,
        direction="a",
        target=0,
        value_convention="binary",
    )

    config = layout.default_config()

    assert sector.value(config) in (0, 1)
    assert sector.is_satisfied(config) == (sector.value(config) == 0)


def test_triangular_z2_winding_flux_value() -> None:
    lattice = TriangularLattice(3, 3, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    sector = TriangularZ2WindingSector(
        layout=layout,
        lattice=lattice,
        direction="b",
        target=0,
        value_convention="flux_pm",
    )

    config = np.full(layout.n_variables, -1, dtype=np.int64)

    assert sector.value(config) == 0


def test_triangular_z2_winding_affected_variables_nonempty() -> None:
    lattice = TriangularLattice(3, 3, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.binary())

    sector = TriangularZ2WindingSector(
        layout=layout,
        lattice=lattice,
        direction="a",
        target=0,
    )

    assert sector.affected_variables().size > 0


def test_triangular_z2_winding_cuts_annihilate_qdm_rhombi() -> None:
    lattice = TriangularLattice(
        3,
        3,
        boundary_condition="periodic",
        include_triangles=True,
        include_rhombi=True,
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    for direction in ("a", "b"):
        sector = TriangularZ2WindingSector(
            layout=layout,
            lattice=lattice,
            direction=direction,
            target=0,
        )

        cut = np.zeros(lattice.num_links, dtype=np.int64)
        cut[sector.link_ids] = 1

        for plaquette_id in lattice.qdm_plaquette_ids():
            plaquette = lattice.plaquettes[int(plaquette_id)]

            assert int(np.sum(cut[list(plaquette.links)]) % 2) == 0


def test_triangular_z2_winding_cut_can_include_c_links() -> None:
    lattice = TriangularLattice(
        3,
        3,
        boundary_condition="periodic",
        include_triangles=True,
        include_rhombi=True,
    )
    layout = VariableLayout.from_lattice_links(
        lattice,
        LocalSpace.binary(),
    )

    sector_a = TriangularZ2WindingSector(
        layout=layout,
        lattice=lattice,
        direction="a",
        target=0,
    )
    sector_b = TriangularZ2WindingSector(
        layout=layout,
        lattice=lattice,
        direction="b",
        target=0,
    )

    kinds_a = {lattice.links[int(link_id)].kind for link_id in sector_a.link_ids}
    kinds_b = {lattice.links[int(link_id)].kind for link_id in sector_b.link_ids}

    assert "c" in kinds_a or "c" in kinds_b
