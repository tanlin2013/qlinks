from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.constraints.base import (
    BaseConstraint,
    ConstraintPropagation,
    ConstraintResult,
)
from qlinks.lattice import LatticeGraph
from qlinks.variables import VariableLayout

ChargeNormalization = Literal["integer_flux", "spin_half"]


def internal_charge_value(
    charge: int,
    *,
    charge_normalization: ChargeNormalization,
) -> int:
    """
    Convert user-facing charge into the raw integer target used by configs.

    integer_flux:
        config values are interpreted as physical E_l = ±1.
        charge target is used directly.

    spin_half:
        config values are stored as s_l = ±1, but physical E_l = s_l / 2.
        user-facing charge q is converted to raw target 2q.
    """
    if charge_normalization == "integer_flux":
        return int(charge)

    if charge_normalization == "spin_half":
        return 2 * int(charge)

    raise ValueError("charge_normalization must be 'integer_flux' or 'spin_half'.")


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
    charge_normalization: ChargeNormalization = "spin_half"

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

        internal_charge = internal_charge_value(
            int(self.charge),
            charge_normalization=self.charge_normalization,
        )

        object.__setattr__(self, "link_ids", link_ids)
        object.__setattr__(self, "signs", signs)
        object.__setattr__(self, "charge", int(self.charge))
        object.__setattr__(self, "_variable_indices", variable_indices)
        object.__setattr__(self, "_internal_charge", internal_charge)
        object.__setattr__(self, "_n_incident_links", int(variable_indices.size))

    @classmethod
    def from_lattice_site(
        cls,
        lattice: LatticeGraph,
        layout: VariableLayout,
        site_id: int,
        charge: int = 0,
        charge_normalization: ChargeNormalization = "spin_half",
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
            charge_normalization=charge_normalization,
        )

    @classmethod
    def all_sites(
        cls,
        lattice: LatticeGraph,
        layout: VariableLayout,
        charges: int | Sequence[int] | npt.NDArray[np.int64] = 0,
        charge_normalization: ChargeNormalization = "spin_half",
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
                    charge_normalization=charge_normalization,
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
        target = self._internal_charge

        residual = actual - target
        satisfied = residual == 0

        return ConstraintResult(
            satisfied=satisfied,
            name=self.name,
            residual=residual,
            message=(
                f"{self.name}(site={self.site_id}): "
                f"divergence={actual}, charge={self.charge}, "
                f"charge_normalization={self.charge_normalization}, "
                f"internal_charge={target}, residual={residual}"
            ),
        )

    def partial_check(
        self,
        config: npt.ArrayLike,
        assigned_mask: npt.ArrayLike,
    ) -> bool:
        return self.propagate(config, assigned_mask).consistent

    def propagate(
        self,
        config: npt.ArrayLike,
        assigned_mask: npt.ArrayLike,
    ) -> ConstraintPropagation:
        arr = np.asarray(config, dtype=np.int64)
        assigned = np.asarray(assigned_mask, dtype=bool)

        variable_indices = self._variable_indices
        signs = self.signs

        assigned_local = assigned[variable_indices]
        unassigned_variables = variable_indices[~assigned_local]
        unassigned_signs = signs[~assigned_local]

        current = int(
            np.dot(
                signs[assigned_local],
                arr[variable_indices[assigned_local]],
            )
        )

        # For spin-half flux values {-1, +1}, every unassigned incident link
        # can still contribute either -1 or +1 after multiplying by the
        # incidence sign.
        remaining_count = int(unassigned_variables.size)
        min_remaining = -remaining_count
        max_remaining = remaining_count

        target = self._internal_charge
        missing = int(target - current)

        if missing < min_remaining:
            return ConstraintPropagation.contradiction()

        if missing > max_remaining:
            return ConstraintPropagation.contradiction()

        # Sum of remaining +/-1 contributions must have the same parity as the
        # number of remaining links.  This prunes impossible charge sectors
        # before the site is fully assigned.
        if (missing + remaining_count) % 2 != 0:
            return ConstraintPropagation.contradiction()

        if remaining_count == 0:
            if missing != 0:
                return ConstraintPropagation.contradiction()
            return ConstraintPropagation.no_change()

        if missing == remaining_count:
            # Every remaining signed contribution must be +1, so value=sign.
            return ConstraintPropagation(
                forced_assignments=tuple(
                    (int(variable_index), int(sign))
                    for variable_index, sign in zip(
                        unassigned_variables, unassigned_signs, strict=True
                    )
                )
            )

        if missing == -remaining_count:
            # Every remaining signed contribution must be -1, so value=-sign.
            return ConstraintPropagation(
                forced_assignments=tuple(
                    (int(variable_index), int(-sign))
                    for variable_index, sign in zip(
                        unassigned_variables, unassigned_signs, strict=True
                    )
                )
            )

        return ConstraintPropagation.no_change()
