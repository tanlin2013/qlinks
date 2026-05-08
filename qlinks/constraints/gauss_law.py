from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from qlinks.constraints.base import BaseConstraint, ConstraintResult
from qlinks.lattice import LatticeGraph
from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class GaussLawConstraint(BaseConstraint):
    """
    Local Gauss-law-like constraint at one lattice site.

    Convention:

        sum_l B[site, l] * E_l == charge

    where B is the oriented incidence matrix with

        B[source, link] = -1
        B[target, link] = +1

    If the layout is link-only, link_id == variable_index.
    More generally, this class maps link_id -> variable_index through layout.
    """

    layout: VariableLayout
    site_id: int
    link_ids: npt.NDArray[np.int64]
    signs: npt.NDArray[np.int64]
    charge: int
    name: str = "gauss_law"

    def __post_init__(self) -> None:
        link_ids = np.asarray(self.link_ids, dtype=np.int64)
        signs = np.asarray(self.signs, dtype=np.int64)

        if link_ids.ndim != 1:
            raise ValueError("link_ids must be one-dimensional.")
        if signs.ndim != 1:
            raise ValueError("signs must be one-dimensional.")
        if link_ids.size != signs.size:
            raise ValueError("link_ids and signs must have the same length.")
        if link_ids.size == 0:
            raise ValueError("A Gauss-law constraint needs at least one incident link.")
        if not np.all(np.isin(signs, [-1, 1])):
            raise ValueError("Gauss-law signs must be +1 or -1.")

        variable_indices = np.asarray(
            [self.layout.link_variable_index(int(link_id)) for link_id in link_ids],
            dtype=np.int64,
        )

        object.__setattr__(self, "link_ids", link_ids)
        object.__setattr__(self, "signs", signs)
        object.__setattr__(self, "charge", int(self.charge))
        object.__setattr__(self, "_variable_indices", variable_indices)

    @classmethod
    def from_lattice_site(
        cls,
        lattice: LatticeGraph,
        layout: VariableLayout,
        site_id: int,
        charge: int = 0,
    ) -> GaussLawConstraint:
        incidence = lattice.incidence_matrix().tocsr()

        # Works for scipy.sparse csr_array.
        row = incidence[site_id, :].tocoo()

        link_ids = row.col.astype(np.int64)
        signs = row.data.astype(np.int64)

        order = np.argsort(link_ids)
        link_ids = link_ids[order]
        signs = signs[order]

        return cls(
            layout=layout,
            site_id=site_id,
            link_ids=link_ids,
            signs=signs,
            charge=charge,
        )

    @classmethod
    def all_sites(
        cls,
        lattice: LatticeGraph,
        layout: VariableLayout,
        charges: int | Sequence[int] | npt.NDArray[np.int64] = 0,
    ) -> tuple[GaussLawConstraint, ...]:
        if isinstance(charges, int):
            charge_array = np.full(lattice.num_sites, charges, dtype=np.int64)
        else:
            charge_array = np.asarray(charges, dtype=np.int64)

        if charge_array.shape != (lattice.num_sites,):
            raise ValueError(
                f"charges must have shape ({lattice.num_sites},), got {charge_array.shape}."
            )

        constraints: list[GaussLawConstraint] = []
        for site_id in range(lattice.num_sites):
            if lattice.incident_links(site_id).size == 0:
                continue
            constraints.append(
                cls.from_lattice_site(
                    lattice=lattice,
                    layout=layout,
                    site_id=site_id,
                    charge=int(charge_array[site_id]),
                )
            )

        return tuple(constraints)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def value(self, config: npt.ArrayLike) -> int:
        arr = self._as_config(config)
        return int(np.dot(self.signs, arr[self._variable_indices]))

    def check(self, config: npt.ArrayLike) -> ConstraintResult:
        actual = self.value(config)
        satisfied = actual == self.charge

        return ConstraintResult(
            satisfied=satisfied,
            name=self.name,
            residual=actual,
            message=(
                f"{self.name}(site={self.site_id}): " f"divergence={actual}, charge={self.charge}"
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
        orientations = self.signs.astype(np.int64)

        assigned_local = assigned[variable_indices]

        assigned_indices = variable_indices[assigned_local]
        assigned_orientations = orientations[assigned_local]

        current = int(np.sum(assigned_orientations * arr[assigned_indices]))

        remaining_orientations = orientations[~assigned_local]

        min_remaining = 0
        max_remaining = 0

        # For spin-half flux values {-1, +1}.
        # Each unassigned term is orientation * E, so it can contribute -1 or +1.
        for orientation in remaining_orientations:
            vals = np.array(
                [
                    int(orientation) * (-1),
                    int(orientation) * (+1),
                ],
                dtype=np.int64,
            )
            min_remaining += int(np.min(vals))
            max_remaining += int(np.max(vals))

        target = int(self.charge)

        if target < current + min_remaining:
            return False

        if target > current + max_remaining:
            return False

        if np.all(assigned_local):
            return current == target

        return True
