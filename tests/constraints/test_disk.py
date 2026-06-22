from __future__ import annotations

import numpy as np
import pytest

from qlinks.constraints import SquareDiskDiagonalLineSumSector
from qlinks.constraints.disk import square_disk_line_label_for_cell, square_disk_line_labels
from qlinks.lattice import SquareLattice
from qlinks.variables import LocalSpace, VariableLayout


def _square_layout(lattice: SquareLattice) -> VariableLayout:
    return VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())


def test_square_disk_line_labels_open_lattice() -> None:
    lattice = SquareLattice(3, 2, boundary_condition="open")

    assert square_disk_line_labels(lattice, family="x_plus_y") == (0, 1, 2, 3)
    assert square_disk_line_labels(lattice, family="x_minus_y") == (-1, 0, 1, 2)


def test_square_disk_line_labels_periodic_lattice_use_gcd_aliases() -> None:
    lattice = SquareLattice(4, 2, boundary_condition="periodic")

    assert square_disk_line_labels(lattice, family="x_plus_y") == (0, 1)
    assert square_disk_line_labels(lattice, family="x_minus_y") == (0, 1)
    assert square_disk_line_label_for_cell((3, 1), lattice, family="x_plus_y") == 0
    assert square_disk_line_label_for_cell((3, 1), lattice, family="x_minus_y") == 0


def test_square_disk_diagonal_line_sum_sector_value_and_check() -> None:
    lattice = SquareLattice(3, 2, boundary_condition="open")
    layout = _square_layout(lattice)
    sector = SquareDiskDiagonalLineSumSector(
        layout=layout,
        lattice=lattice,
        family="x_plus_y",
        target=(1, 1, 1, 0),
    )

    # Site ordering is x-major: (0,0), (0,1), (1,0), (1,1), (2,0), (2,1).
    config = np.asarray([1, 1, 0, 0, 1, 0], dtype=np.int64)

    assert sector.labels == (0, 1, 2, 3)
    assert sector.value(config) == (1, 1, 1, 0)
    assert sector.is_satisfied(config)

    failed = sector.check(np.asarray([1, 1, 0, 0, 0, 0], dtype=np.int64))
    assert not failed.satisfied
    assert failed.residual == (1, 1, 0, 0)


def test_square_disk_diagonal_line_sum_sector_partial_check_prunes_bounds() -> None:
    lattice = SquareLattice(3, 2, boundary_condition="open")
    layout = _square_layout(lattice)
    sector = SquareDiskDiagonalLineSumSector(
        layout=layout,
        lattice=lattice,
        family="x_plus_y",
        target=(0, 1, 0, 0),
    )

    possible = np.asarray([0, 1, 0, 0, 0, 0], dtype=np.int64)
    possible_assigned = np.asarray([True, True, False, False, False, False])
    assert sector.partial_check(possible, possible_assigned)

    too_many = np.asarray([1, 0, 0, 0, 0, 0], dtype=np.int64)
    too_many_assigned = np.asarray([True, False, False, False, False, False])
    assert not sector.partial_check(too_many, too_many_assigned)

    impossible_to_reach = np.asarray([0, 0, 0, 0, 0, 0], dtype=np.int64)
    all_assigned = np.ones(6, dtype=bool)
    assert not sector.partial_check(impossible_to_reach, all_assigned)


def test_square_disk_diagonal_line_sum_sector_rejects_bad_target_length() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = _square_layout(lattice)

    with pytest.raises(ValueError, match="must have length"):
        SquareDiskDiagonalLineSumSector(
            layout=layout,
            lattice=lattice,
            family="x_plus_y",
            target=(0, 1),
        )


def test_square_disk_line_label_rejects_unknown_family() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")

    with pytest.raises(ValueError, match="family must"):
        square_disk_line_label_for_cell((0, 0), lattice, family="bad")  # type: ignore[arg-type]
