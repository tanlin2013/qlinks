from __future__ import annotations

import numpy as np

from qlinks.lattice import SquareLattice
from qlinks.operators import DiskDiagonalHopOperator, DiskDiagonalHopProjector
from qlinks.variables import LocalSpace, VariableLayout


def _square_layout(lattice: SquareLattice) -> VariableLayout:
    return VariableLayout.from_lattice_sites(lattice, LocalSpace.binary())


def test_disk_diagonal_hop_pairs_for_open_square_lattice() -> None:
    lattice = SquareLattice(3, 3, boundary_condition="open")

    assert DiskDiagonalHopOperator.pairs_for_family(lattice, family="x_plus_y") == (
        (1, 3),
        (2, 4),
        (4, 6),
        (5, 7),
    )
    assert DiskDiagonalHopOperator.pairs_for_family(lattice, family="x_minus_y") == (
        (0, 4),
        (1, 5),
        (3, 7),
        (4, 8),
    )


def test_disk_diagonal_hop_apply_moves_one_disk() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = _square_layout(lattice)
    op = DiskDiagonalHopOperator(
        layout=layout,
        lattice=lattice,
        source_site=1,
        target_site=2,
        coefficient=2.0,
    )

    actions = op.apply(np.asarray([0, 1, 0, 0], dtype=np.int64))

    assert len(actions) == 1
    assert actions[0].coefficient == 2.0 + 0j
    np.testing.assert_array_equal(actions[0].config, np.asarray([0, 0, 1, 0], dtype=np.int64))
    np.testing.assert_array_equal(op.affected_variables(), np.asarray([1, 2], dtype=np.int64))


def test_disk_diagonal_hop_requires_source_occupied_and_target_empty() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = _square_layout(lattice)
    op = DiskDiagonalHopOperator(layout=layout, lattice=lattice, source_site=1, target_site=2)

    assert op.apply(np.asarray([0, 0, 0, 0], dtype=np.int64)) == ()
    assert op.apply(np.asarray([0, 1, 1, 0], dtype=np.int64)) == ()


def test_disk_diagonal_hop_respects_blockade_after_move() -> None:
    lattice = SquareLattice(3, 3, boundary_condition="open")
    layout = _square_layout(lattice)
    op = DiskDiagonalHopOperator(
        layout=layout,
        lattice=lattice,
        source_site=2,  # (0, 2)
        target_site=4,  # (1, 1)
        enforce_nearest_neighbor_blockade=True,
    )

    blocked = np.zeros(9, dtype=np.int64)
    blocked[2] = 1
    blocked[3] = 1  # (1, 0), nearest neighbor of target site 4.
    assert op.apply(blocked) == ()

    op_without_blockade = DiskDiagonalHopOperator(
        layout=layout,
        lattice=lattice,
        source_site=2,
        target_site=4,
        enforce_nearest_neighbor_blockade=False,
    )
    actions = op_without_blockade.apply(blocked)
    assert len(actions) == 1
    np.testing.assert_array_equal(
        actions[0].config,
        np.asarray([0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64),
    )


def test_disk_diagonal_hop_projector_diagonal_value() -> None:
    lattice = SquareLattice(2, 2, boundary_condition="open")
    layout = _square_layout(lattice)
    projector = DiskDiagonalHopProjector(
        layout=layout,
        source_site=1,
        target_site=2,
        coefficient=3.0,
    )

    assert projector.diagonal_value(np.asarray([0, 1, 0, 0], dtype=np.int64)) == 3.0 + 0j
    assert projector.diagonal_value(np.asarray([0, 0, 0, 0], dtype=np.int64)) is None
    assert projector.diagonal_value(np.asarray([0, 1, 1, 0], dtype=np.int64)) is None
    np.testing.assert_array_equal(
        projector.affected_variables(), np.asarray([1, 2], dtype=np.int64)
    )
