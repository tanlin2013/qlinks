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
