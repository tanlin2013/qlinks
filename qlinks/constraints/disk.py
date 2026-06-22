from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.constraints.base import BaseSectorCondition, ConstraintResult
from qlinks.lattice import BoundaryCondition, SquareLattice
from qlinks.variables import VariableLayout

DiskDiagonalFamily = Literal["x_plus_y", "x_minus_y"]


def square_disk_line_labels(
    lattice: SquareLattice,
    *,
    family: DiskDiagonalFamily,
) -> tuple[int, ...]:
    """Return stable labels for square-lattice disk diagonal lines.

    ``family='x_plus_y'`` labels anti-diagonals with constant ``x + y``;
    ``family='x_minus_y'`` labels diagonals with constant ``x - y``.  Periodic
    lattices use labels modulo the corresponding linear size, so these are the
    line sums conserved by diagonal hopping on a torus.
    """

    if family not in ("x_plus_y", "x_minus_y"):
        raise ValueError("family must be 'x_plus_y' or 'x_minus_y'.")

    if lattice.boundary_condition == BoundaryCondition.PERIODIC:
        # A +1,+1 hop preserves x-y modulo gcd(lx, ly).  A +1,-1 hop preserves
        # x+y modulo gcd(lx, ly).  For square tori this is simply L.  The gcd
        # form also handles rectangular periodic tests without inventing labels
        # for disconnected aliases.
        n_labels = int(np.gcd(int(lattice.lx), int(lattice.ly)))
        return tuple(range(n_labels))

    if family == "x_plus_y":
        return tuple(range(0, int(lattice.lx) + int(lattice.ly) - 1))

    return tuple(range(-(int(lattice.ly) - 1), int(lattice.lx)))


def square_disk_line_label_for_cell(
    cell: Sequence[int],
    lattice: SquareLattice,
    *,
    family: DiskDiagonalFamily,
) -> int:
    """Return the diagonal-line label for one square-lattice cell/site."""

    if len(cell) != 2:
        raise ValueError(f"Expected a 2D square-lattice cell, got {tuple(cell)!r}.")

    x = int(cell[0])
    y = int(cell[1])

    if family == "x_plus_y":
        value = x + y
    elif family == "x_minus_y":
        value = x - y
    else:
        raise ValueError("family must be 'x_plus_y' or 'x_minus_y'.")

    if lattice.boundary_condition == BoundaryCondition.PERIODIC:
        modulus = int(np.gcd(int(lattice.lx), int(lattice.ly)))
        return int(value % modulus)

    return int(value)


@dataclass(frozen=True, slots=True)
class SquareDiskDiagonalLineSumSector(BaseSectorCondition):
    """Fix all disk-number sums along one square-lattice diagonal family.

    This is a diagonal symmetry-sector filter for the quantum disk model.  It is
    intentionally independent of the Hamiltonian implementation: a model can
    use it whenever its local moves preserve one of the two diagonal line-sum
    families.
    """

    layout: VariableLayout
    lattice: SquareLattice
    family: DiskDiagonalFamily
    target: tuple[int, ...]
    occupied_value: int = 1
    name: str = "square_disk_diagonal_line_sum_sector"

    def __post_init__(self) -> None:
        labels = square_disk_line_labels(self.lattice, family=self.family)
        target = tuple(int(v) for v in self.target)
        if len(target) != len(labels):
            raise ValueError(
                f"target for {self.family!r} must have length {len(labels)}, " f"got {len(target)}."
            )

        label_to_offset = {label: offset for offset, label in enumerate(labels)}
        variable_indices_by_offset: list[list[int]] = [[] for _ in labels]

        for site in self.lattice.sites:
            variable_index = int(self.layout.site_variable_index(int(site.id)))
            self.layout.local_space(variable_index).validate_value(self.occupied_value)
            label = square_disk_line_label_for_cell(site.cell, self.lattice, family=self.family)
            variable_indices_by_offset[label_to_offset[int(label)]].append(variable_index)

        object.__setattr__(self, "target", target)
        object.__setattr__(self, "_labels", tuple(int(v) for v in labels))
        object.__setattr__(
            self,
            "_variable_indices_by_offset",
            tuple(np.asarray(v, dtype=np.int64) for v in variable_indices_by_offset),
        )
        object.__setattr__(self, "_affected", np.arange(self.layout.n_variables, dtype=np.int64))
        object.__setattr__(self, "_occupied_value", int(self.occupied_value))

    @property
    def labels(self) -> tuple[int, ...]:
        return self._labels

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._affected.copy()

    def value(self, config: npt.ArrayLike) -> tuple[int, ...]:
        arr = self._as_config(config)
        occupied = self._occupied_value
        return tuple(
            int(np.count_nonzero(arr[indices] == occupied))
            for indices in self._variable_indices_by_offset
        )

    def check(self, config: npt.ArrayLike) -> ConstraintResult:
        actual = self.value(config)
        return ConstraintResult(
            satisfied=actual == self.target,
            name=self.name,
            residual=actual,
            message=f"{self.name}({self.family}): value={actual}, target={self.target}",
        )

    def partial_check(
        self,
        config: npt.ArrayLike,
        assigned_mask: npt.ArrayLike,
    ) -> bool:
        arr = np.asarray(config, dtype=np.int64)
        assigned = np.asarray(assigned_mask, dtype=bool)
        occupied = self._occupied_value

        for target_value, indices in zip(
            self.target,
            self._variable_indices_by_offset,
            strict=True,
        ):
            assigned_local = assigned[indices]
            current = int(np.count_nonzero(arr[indices[assigned_local]] == occupied))
            remaining = int(np.count_nonzero(~assigned_local))
            if current > int(target_value):
                return False
            if current + remaining < int(target_value):
                return False

        return True
