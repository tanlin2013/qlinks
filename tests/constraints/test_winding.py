import numpy as np
import pytest

from qlinks.constraints import SquareWindingSector
from qlinks.lattice import SquareLattice
from qlinks.variables import LocalSpace, VariableLayout


def test_square_winding_sector_x_2_by_2() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    sector = SquareWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target=2,
    )

    # For 2x2 periodic square, there are two x-wrapping links.
    assert sector.link_ids.size == 2

    config = np.ones(lattice.num_links, dtype=np.int64)

    assert sector.value(config) == 2
    assert sector.is_satisfied(config)


def test_square_winding_sector_y_2_by_2() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    sector = SquareWindingSector(
        layout=layout,
        lattice=lattice,
        direction="y",
        target=-2,
    )

    assert sector.link_ids.size == 2

    config = -np.ones(lattice.num_links, dtype=np.int64)

    assert sector.value(config) == -2
    assert sector.is_satisfied(config)


def test_square_winding_sector_unsatisfied() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="periodic")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    sector = SquareWindingSector(
        layout=layout,
        lattice=lattice,
        direction="x",
        target=0,
    )

    config = np.ones(lattice.num_links, dtype=np.int64)

    assert not sector.is_satisfied(config)


def test_square_winding_requires_periodic_lattice() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = VariableLayout.from_lattice_links(lattice, LocalSpace.spin_half_flux())

    with pytest.raises(ValueError, match="requires a periodic"):
        SquareWindingSector(
            layout=layout,
            lattice=lattice,
            direction="x",
            target=0,
        )
