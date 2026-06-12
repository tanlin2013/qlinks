from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from qlinks.constraints.base import BaseConstraint, ConstraintResult
from qlinks.lattice import LatticeGraph
from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class DimerCoveringConstraint(BaseConstraint):
    """
    Dimer covering constraint at one site.

        sum_{links incident to site} n_l == required_count

    Usually required_count = 1 for a fully packed dimer model.
    """

    layout: VariableLayout
    site_id: int
    link_ids: npt.NDArray[np.int64]
    required_count: int = 1
    name: str = "dimer_covering"

    def __post_init__(self) -> None:
        link_ids = np.asarray(self.link_ids, dtype=np.int64)

        if link_ids.ndim != 1:
            raise ValueError("link_ids must be one-dimensional.")
        if link_ids.size == 0:
            raise ValueError("A dimer constraint needs at least one incident link.")

        variable_indices = np.asarray(
            [self.layout.link_variable_index(int(link_id)) for link_id in link_ids],
            dtype=np.int64,
        )

        object.__setattr__(self, "link_ids", link_ids)
        object.__setattr__(self, "required_count", int(self.required_count))
        object.__setattr__(self, "_variable_indices", variable_indices)
        object.__setattr__(self, "_n_incident_links", int(variable_indices.size))

    @classmethod
    def from_lattice_site(
        cls,
        lattice: LatticeGraph,
        layout: VariableLayout,
        site_id: int,
        required_count: int = 1,
    ) -> DimerCoveringConstraint:
        return cls(
            layout=layout,
            site_id=site_id,
            link_ids=lattice.incident_links(site_id),
            required_count=required_count,
        )

    @classmethod
    def all_sites(
        cls,
        lattice: LatticeGraph,
        layout: VariableLayout,
        required_counts: int | Sequence[int] | npt.NDArray[np.int64] = 1,
    ) -> tuple[DimerCoveringConstraint, ...]:
        if isinstance(required_counts, int):
            count_array = np.full(lattice.num_sites, required_counts, dtype=np.int64)
        else:
            count_array = np.asarray(required_counts, dtype=np.int64)

        if count_array.shape != (lattice.num_sites,):
            raise ValueError(
                f"required_counts must have shape ({lattice.num_sites},), got {count_array.shape}."
            )

        constraints: list[DimerCoveringConstraint] = []
        for site_id in range(lattice.num_sites):
            if lattice.incident_links(site_id).size == 0:
                continue
            constraints.append(
                cls.from_lattice_site(
                    lattice=lattice,
                    layout=layout,
                    site_id=site_id,
                    required_count=int(count_array[site_id]),
                )
            )

        return tuple(constraints)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def value(self, config: npt.ArrayLike) -> int:
        arr = self._as_config(config)
        return int(np.sum(arr[self._variable_indices]))

    def check(self, config: npt.ArrayLike) -> ConstraintResult:
        actual = self.value(config)
        satisfied = actual == self.required_count

        return ConstraintResult(
            satisfied=satisfied,
            name=self.name,
            residual=actual,
            message=(
                f"{self.name}(site={self.site_id}): "
                f"count={actual}, required={self.required_count}"
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

        assigned_count = int(np.count_nonzero(assigned_local))
        assigned_values = arr[variable_indices[assigned_local]]

        occupied_so_far = int(np.sum(assigned_values))
        unassigned_count = self._n_incident_links - assigned_count

        # Too many dimers already touching this site.
        if occupied_so_far > self.required_count:
            return False

        # Even filling all remaining incident links cannot reach required_count.
        if occupied_so_far + unassigned_count < self.required_count:
            return False

        # If fully assigned, require exact equality.
        if unassigned_count == 0:
            return occupied_so_far == self.required_count

        return True
