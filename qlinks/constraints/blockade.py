from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qlinks.constraints.base import BaseConstraint, ConstraintResult
from qlinks.lattice import LatticeGraph
from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class NearestNeighborBlockadeConstraint(BaseConstraint):
    """
    PXP/Rydberg-blockade-style constraint on one lattice bond.

        not (n_i == occupied_value and n_j == occupied_value)

    Usually occupied_value = 1.
    """

    layout: VariableLayout
    site_i: int
    site_j: int
    occupied_value: int = 1
    name: str = "nearest_neighbor_blockade"

    def __post_init__(self) -> None:
        vi = self.layout.site_variable_index(self.site_i)
        vj = self.layout.site_variable_index(self.site_j)

        self.layout.local_space(vi).validate_value(self.occupied_value)
        self.layout.local_space(vj).validate_value(self.occupied_value)

        object.__setattr__(self, "_variable_indices", np.asarray([vi, vj], dtype=np.int64))

    @classmethod
    def from_lattice(
        cls,
        lattice: LatticeGraph,
        layout: VariableLayout,
        occupied_value: int = 1,
    ) -> tuple[NearestNeighborBlockadeConstraint, ...]:
        constraints: list[NearestNeighborBlockadeConstraint] = []

        seen: set[tuple[int, int]] = set()
        for link in lattice.links:
            i, j = sorted((link.source, link.target))
            if (i, j) in seen:
                continue
            seen.add((i, j))

            constraints.append(
                cls(
                    layout=layout,
                    site_i=i,
                    site_j=j,
                    occupied_value=occupied_value,
                )
            )

        return tuple(constraints)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def check(self, config: npt.ArrayLike) -> ConstraintResult:
        arr = self._as_config(config)

        vi, vj = self._variable_indices
        ni = int(arr[vi])
        nj = int(arr[vj])

        violated = ni == self.occupied_value and nj == self.occupied_value

        return ConstraintResult(
            satisfied=not violated,
            name=self.name,
            residual=(ni, nj),
            message=(
                f"{self.name}(sites=({self.site_i}, {self.site_j})): "
                f"values=({ni}, {nj}), occupied={self.occupied_value}"
            ),
        )

    def partial_check(
            self,
            config: npt.ArrayLike,
            assigned_mask: npt.ArrayLike,
    ) -> bool:
        arr = np.asarray(config, dtype=np.int64)
        assigned = np.asarray(assigned_mask, dtype=bool)

        variable_indices = self._variable_indices
        assigned_local = assigned[variable_indices]

        # If fewer than two variables are assigned, this edge cannot yet violate blockade.
        if not np.all(assigned_local):
            return True

        values = arr[variable_indices]
        return not np.all(values == self.occupied_value)
