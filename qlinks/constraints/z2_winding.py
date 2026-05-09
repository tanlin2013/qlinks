from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from qlinks.constraints.base import BaseSectorCondition
from qlinks.lattice import BoundaryCondition, TriangularLattice
from qlinks.variables import VariableLayout

TriangularCycleDirection = Literal["a", "b"]
Z2ValueConvention = Literal["binary", "flux_pm"]


@dataclass(frozen=True, slots=True)
class TriangularZ2WindingSector(BaseSectorCondition):
    """
    Z2 topological winding sector for triangular-lattice QDM/QLM on a torus.

    The invariant is the parity of occupied/electric-positive links crossing a
    non-contractible cut.

    direction:
        "a":
            cut links wrapping across the a1 / x periodic seam.

        "b":
            cut links wrapping across the a2 / y periodic seam.

    value_convention:
        "binary":
            config values are already n_l in {0, 1}. Use n_l directly.

        "flux_pm":
            config values are E_l in {-1, +1}. Convert to n_l = (E_l + 1) // 2.
    """

    layout: VariableLayout
    lattice: TriangularLattice
    direction: TriangularCycleDirection
    target: int
    value_convention: Z2ValueConvention = "binary"
    name: str = "triangular_z2_winding_sector"

    def __post_init__(self) -> None:
        if self.lattice.boundary_condition != BoundaryCondition.PERIODIC:
            raise ValueError("TriangularZ2WindingSector requires periodic boundary conditions.")

        if self.direction not in ("a", "b"):
            raise ValueError("direction must be 'a' or 'b'.")

        if self.target not in (0, 1):
            raise ValueError("target must be 0 or 1.")

        if self.value_convention not in ("binary", "flux_pm"):
            raise ValueError("value_convention must be 'binary' or 'flux_pm'.")

        link_ids: list[int] = []

        for link in self.lattice.links:
            if not link.wrap:
                continue

            # TriangularLattice link kinds are:
            #   a: (x, y) -> (x + 1, y)
            #   b: (x, y) -> (x, y + 1)
            #   c: (x, y) -> (x - 1, y + 1)
            #
            # For a first clean convention:
            #   direction="a" uses wrapping a-links.
            #   direction="b" uses wrapping b-links.
            #
            # The c-links also wind on some seams depending on embedding, but
            # this simple convention is enough if your lattice construction
            # consistently tags wrap=True and kind='a'/'b' for the primitive cuts.
            if self.direction == "a" and link.kind == "a":
                link_ids.append(int(link.id))

            if self.direction == "b" and link.kind == "b":
                link_ids.append(int(link.id))

        if len(link_ids) == 0:
            raise ValueError(f"No wrapping links found for triangular {self.direction}-cycle.")

        variable_indices = np.asarray(
            [self.layout.link_variable_index(link_id) for link_id in link_ids],
            dtype=np.int64,
        )

        object.__setattr__(self, "_link_ids", np.asarray(link_ids, dtype=np.int64))
        object.__setattr__(self, "_variable_indices", variable_indices)
        object.__setattr__(self, "target", int(self.target))

    @property
    def link_ids(self) -> npt.NDArray[np.int64]:
        return self._link_ids.copy()

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def _occupation_values(self, values: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        if self.value_convention == "binary":
            return values.astype(np.int64, copy=False)

        if self.value_convention == "flux_pm":
            return ((values + 1) // 2).astype(np.int64, copy=False)

        raise ValueError(f"Unsupported value convention: {self.value_convention}")

    def value(self, config: npt.ArrayLike) -> int:
        arr = self._as_config(config)

        values = arr[self._variable_indices]
        occ = self._occupation_values(values)

        return int(np.sum(occ) % 2)

    def is_satisfied(self, config: npt.ArrayLike) -> bool:
        return self.value(config) == self.target

    def partial_check(
        self,
        config: npt.ArrayLike,
        assigned_mask: npt.ArrayLike,
    ) -> bool:
        """
        Z2 parity generally cannot be pruned until all cut variables are assigned.

        If unassigned variables remain, either parity is still possible unless
        one adds a more detailed parity-reachability check. The safe behavior is
        to wait.
        """
        assigned = np.asarray(assigned_mask, dtype=bool)

        if np.all(assigned[self._variable_indices]):
            return self.is_satisfied(config)

        return True
