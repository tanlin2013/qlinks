from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from qlinks.lattice import SquareLattice
from qlinks.operators.base import BaseLocalOperator, OperatorAction
from qlinks.variables import VariableLayout

DiskDiagonalFamily = Literal["x_plus_y", "x_minus_y"]


def _neighbor_sets(lattice: SquareLattice) -> tuple[frozenset[int], ...]:
    neighbors: list[set[int]] = [set() for _ in range(lattice.num_sites)]
    for link in lattice.links:
        neighbors[int(link.source)].add(int(link.target))
        neighbors[int(link.target)].add(int(link.source))
    return tuple(frozenset(values) for values in neighbors)


def _square_diagonal_hop_pairs(
    lattice: SquareLattice,
    *,
    family: DiskDiagonalFamily,
) -> tuple[tuple[int, int], ...]:
    """Return undirected diagonal bonds for one conserved diagonal family.

    ``family='x_plus_y'`` uses displacement ``(+1, -1)`` and therefore keeps
    ``x + y`` fixed.  ``family='x_minus_y'`` uses displacement ``(+1, +1)`` and
    therefore keeps ``x - y`` fixed.
    """

    if family == "x_plus_y":
        displacement = (1, -1)
    elif family == "x_minus_y":
        displacement = (1, 1)
    else:
        raise ValueError("family must be 'x_plus_y' or 'x_minus_y'.")

    pairs: set[tuple[int, int]] = set()
    for site in lattice.sites:
        x, y = (int(site.cell[0]), int(site.cell[1]))
        tx = x + displacement[0]
        ty = y + displacement[1]
        try:
            target = int(lattice.site_id(tx, ty))
        except IndexError:
            continue

        source = int(site.id)
        if source == target:
            continue
        pairs.add(tuple(sorted((source, target))))

    return tuple(sorted(pairs))


@dataclass(frozen=True, slots=True)
class DiskDiagonalHopOperator(BaseLocalOperator):
    """Move one hard-core disk along a diagonal bond."""

    layout: VariableLayout
    lattice: SquareLattice
    source_site: int
    target_site: int
    coefficient: complex = 1.0
    occupied_value: int = 1
    empty_value: int = 0
    enforce_nearest_neighbor_blockade: bool = True
    name: str = "disk_diagonal_hop"

    def __post_init__(self) -> None:
        source_var = int(self.layout.site_variable_index(int(self.source_site)))
        target_var = int(self.layout.site_variable_index(int(self.target_site)))
        self.layout.local_space(source_var).validate_value(int(self.occupied_value))
        self.layout.local_space(source_var).validate_value(int(self.empty_value))
        self.layout.local_space(target_var).validate_value(int(self.occupied_value))
        self.layout.local_space(target_var).validate_value(int(self.empty_value))

        object.__setattr__(self, "_source_var", source_var)
        object.__setattr__(self, "_target_var", target_var)
        object.__setattr__(
            self,
            "_affected",
            np.asarray([source_var, target_var], dtype=np.int64),
        )
        object.__setattr__(self, "_neighbor_sets", _neighbor_sets(self.lattice))

    @classmethod
    def pairs_for_family(
        cls,
        lattice: SquareLattice,
        *,
        family: DiskDiagonalFamily,
    ) -> tuple[tuple[int, int], ...]:
        return _square_diagonal_hop_pairs(lattice, family=family)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._affected.copy()

    def _satisfies_blockade_after_move(self, arr: npt.NDArray[np.int64]) -> bool:
        if not self.enforce_nearest_neighbor_blockade:
            return True

        occupied = int(self.occupied_value)
        moved_site = int(self.target_site)
        for neighbor in self._neighbor_sets[moved_site]:
            if neighbor == int(self.source_site):
                continue
            neighbor_var = int(self.layout.site_variable_index(int(neighbor)))
            if int(arr[neighbor_var]) == occupied:
                return False

        return True

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]:
        arr = self._as_config(config)
        source_var = self._source_var
        target_var = self._target_var

        if int(arr[source_var]) != int(self.occupied_value):
            return ()
        if int(arr[target_var]) != int(self.empty_value):
            return ()

        out = arr.copy()
        out[source_var] = int(self.empty_value)
        out[target_var] = int(self.occupied_value)

        if not self._satisfies_blockade_after_move(out):
            return ()

        return (OperatorAction(self.coefficient, out),)


@dataclass(frozen=True, slots=True)
class DiskDiagonalHopProjector(BaseLocalOperator):
    """Diagonal projector onto a mobile disk on one directed diagonal bond."""

    layout: VariableLayout
    source_site: int
    target_site: int
    coefficient: complex = 1.0
    occupied_value: int = 1
    empty_value: int = 0
    name: str = "disk_diagonal_hop_projector"

    def __post_init__(self) -> None:
        source_var = int(self.layout.site_variable_index(int(self.source_site)))
        target_var = int(self.layout.site_variable_index(int(self.target_site)))
        self.layout.local_space(source_var).validate_value(int(self.occupied_value))
        self.layout.local_space(target_var).validate_value(int(self.empty_value))
        object.__setattr__(self, "_source_var", source_var)
        object.__setattr__(self, "_target_var", target_var)
        object.__setattr__(
            self,
            "_affected",
            np.asarray([source_var, target_var], dtype=np.int64),
        )

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._affected.copy()

    def diagonal_value(self, config: npt.ArrayLike) -> complex | None:
        arr = self._as_config(config)
        if int(arr[self._source_var]) == int(self.occupied_value) and int(
            arr[self._target_var]
        ) == int(self.empty_value):
            return complex(self.coefficient)
        return None
