from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from qlinks.constraints.base import BaseSectorCondition, ConstraintResult
from qlinks.lattice import BoundaryCondition, HoneycombLattice, SquareLattice
from qlinks.variables import VariableLayout

Direction = Literal["x", "y"]
FluxNormalization = Literal["integer_flux", "spin_half"]


def internal_flux_winding_value(
    winding: int,
    *,
    flux_normalization: FluxNormalization,
) -> int:
    """
    Convert user-facing winding target into raw integer flux target.

    integer_flux:
        stored flux s_l ∈ {-1,+1} is physical E_l.
        target = winding

    spin_half:
        stored flux s_l ∈ {-1,+1}, physical E_l = s_l / 2.
        target = 2 * winding
    """
    if flux_normalization == "integer_flux":
        return int(winding)

    if flux_normalization == "spin_half":
        return 2 * int(winding)

    raise ValueError("flux_normalization must be 'integer_flux' or 'spin_half'.")


@dataclass(frozen=True, slots=True)
class SquareWindingSector(BaseSectorCondition):
    """
    Simple diagonal winding-flux sector for SquareLattice link variables.

    This fixes the sum of link variables along a non-contractible cut.

    For direction="x":
        Sum x-links crossing from x = lx - 1 to x = 0 over all y.

    For direction="y":
        Sum y-links crossing from y = ly - 1 to y = 0 over all x.

    This class is intentionally square-lattice specific. Other lattices should
    define their own topological-sector objects.
    """

    layout: VariableLayout
    lattice: SquareLattice
    direction: Direction
    target: int
    name: str = "square_winding_sector"
    flux_normalization: FluxNormalization = "spin_half"

    def __post_init__(self) -> None:
        if self.lattice.boundary_condition != BoundaryCondition.PERIODIC:
            raise ValueError("SquareWindingSector requires a periodic SquareLattice.")

        if self.direction not in ("x", "y"):
            raise ValueError("direction must be 'x' or 'y'.")

        link_ids: list[int] = []

        for link in self.lattice.links:
            if self.direction == "x":
                if link.kind == "x" and link.wrap:
                    link_ids.append(link.id)
            else:
                if link.kind == "y" and link.wrap:
                    link_ids.append(link.id)

        if len(link_ids) == 0:
            raise ValueError(f"No wrapping {self.direction}-links found.")

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

    def value(self, config: npt.ArrayLike) -> int:
        arr = self._as_config(config)
        return int(np.sum(arr[self._variable_indices]))

    def internal_target(self) -> int:
        values_seen: set[int] = set()

        for variable_index in self._variable_indices:
            values_seen.update(
                int(v) for v in self.layout.local_space(int(variable_index)).values.tolist()
            )

        # QLM flux sector: values {-1,+1}
        if values_seen <= {-1, 1}:
            return internal_flux_winding_value(
                self.target,
                flux_normalization=self.flux_normalization,
            )

        # QDM cut-count sector: values {0,1}; target is already a count.
        return int(self.target)

    def check(self, config: npt.ArrayLike) -> ConstraintResult:
        actual = self.value(config)
        target = self.internal_target()
        residual = actual - target
        satisfied = residual == 0

        return ConstraintResult(
            satisfied=satisfied,
            name=self.name,
            residual=residual,
            message=(
                f"{self.name}(direction={self.direction}): "
                f"value={actual}, target={self.target}, "
                f"internal_target={target}, residual={residual}"
            ),
        )

    def partial_check(self, config, assigned_mask) -> bool:
        arr = np.asarray(config, dtype=np.int64)
        assigned = np.asarray(assigned_mask, dtype=bool)

        variable_indices = self._variable_indices
        assigned_local = assigned[variable_indices]

        current = int(np.sum(arr[variable_indices[assigned_local]]))

        min_remaining = 0
        max_remaining = 0

        for idx in variable_indices[~assigned_local]:
            values = self.layout.local_space(int(idx)).values
            min_remaining += int(np.min(values))
            max_remaining += int(np.max(values))

        target = self.internal_target()

        if target < current + min_remaining:
            return False

        if target > current + max_remaining:
            return False

        if np.all(assigned_local):
            return current == target

        return True


@dataclass(frozen=True, slots=True)
class SquareQDMElectricWindingSector(BaseSectorCondition):
    """
    Signed QDM winding sector compatible with the staggered-charge QLM mapping.

    QDM variables:
        n_l in {0, 1}

    Electric-flux mapping:
        E_l = eta(source(l)) * (2 n_l - 1)

    where:
        eta(x, y) = (-1)^(x + y)

    The winding is computed across wrapping links:

        direction='x':
            sum over x-wrapping links

        direction='y':
            sum over y-wrapping links

    This is the sector convention to compare with QLM winding sectors.
    """

    layout: VariableLayout
    lattice: SquareLattice
    direction: Direction
    target: int
    name: str = "square_qdm_electric_winding_sector"

    def __post_init__(self) -> None:
        if self.lattice.boundary_condition != BoundaryCondition.PERIODIC:
            raise ValueError("SquareQDMElectricWindingSector requires a periodic SquareLattice.")

        if self.direction not in ("x", "y"):
            raise ValueError("direction must be 'x' or 'y'.")

        link_ids: list[int] = []
        signs: list[int] = []

        for link in self.lattice.links:
            if link.kind != self.direction:
                continue

            if not link.wrap:
                continue

            source_site = self.lattice.sites[link.source]
            x, y = source_site.cell

            eta = 1 if (x + y) % 2 == 0 else -1

            link_ids.append(link.id)
            signs.append(eta)

        if len(link_ids) == 0:
            raise ValueError(f"No wrapping {self.direction}-links found.")

        variable_indices = np.asarray(
            [self.layout.link_variable_index(link_id) for link_id in link_ids],
            dtype=np.int64,
        )

        object.__setattr__(self, "_link_ids", np.asarray(link_ids, dtype=np.int64))
        object.__setattr__(self, "_variable_indices", variable_indices)
        object.__setattr__(self, "_signs", np.asarray(signs, dtype=np.int64))
        object.__setattr__(self, "target", int(self.target))

    @property
    def link_ids(self) -> npt.NDArray[np.int64]:
        return self._link_ids.copy()

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    @property
    def signs(self) -> npt.NDArray[np.int64]:
        return self._signs.copy()

    def value(self, config: npt.ArrayLike) -> int:
        arr = self._as_config(config)

        n = arr[self._variable_indices]

        # E_l = eta_l * (2 n_l - 1)
        electric_flux = self._signs * (2 * n - 1)

        return int(np.sum(electric_flux))

    def partial_check(
        self,
        config: npt.ArrayLike,
        assigned_mask: npt.ArrayLike,
    ) -> bool:
        arr = np.asarray(config, dtype=np.int64)
        assigned = np.asarray(assigned_mask, dtype=bool)

        variable_indices = self._variable_indices
        signs = self._signs

        assigned_local = assigned[variable_indices]

        assigned_indices = variable_indices[assigned_local]
        assigned_signs = signs[assigned_local]

        n = arr[assigned_indices]
        current = int(np.sum(assigned_signs * (2 * n - 1)))

        remaining_signs = signs[~assigned_local]

        min_remaining = -int(np.sum(np.abs(remaining_signs)))
        max_remaining = int(np.sum(np.abs(remaining_signs)))

        target = int(self.target)

        if target < current + min_remaining:
            return False

        if target > current + max_remaining:
            return False

        if np.all(assigned_local):
            return current == target

        return True


@dataclass(frozen=True, slots=True)
class HoneycombElectricWindingSector(BaseSectorCondition):
    layout: VariableLayout
    lattice: HoneycombLattice
    direction: Literal["x", "y"]
    target: int
    value_convention: Literal["binary", "flux_pm"] = "binary"
    name: str = "honeycomb_electric_winding_sector"
    flux_normalization: FluxNormalization = "spin_half"

    def __post_init__(self) -> None:
        if self.lattice.boundary_condition != BoundaryCondition.PERIODIC:
            raise ValueError("HoneycombElectricWindingSector requires PBC.")

        if self.direction not in ("x", "y"):
            raise ValueError("direction must be 'x' or 'y'.")

        if self.value_convention not in ("binary", "flux_pm"):
            raise ValueError("value_convention must be 'binary' or 'flux_pm'.")

        link_ids = [
            int(link.id) for link in self.lattice.links if link.wrap and link.kind == self.direction
        ]

        if len(link_ids) == 0:
            raise ValueError(f"No wrapping {self.direction}-links found.")

        variable_indices = np.asarray(
            [self.layout.link_variable_index(link_id) for link_id in link_ids],
            dtype=np.int64,
        )

        object.__setattr__(self, "_link_ids", np.asarray(link_ids, dtype=np.int64))
        object.__setattr__(self, "_variable_indices", variable_indices)
        object.__setattr__(self, "target", int(self.target))

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def _electric_values(self, values: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        if self.value_convention == "binary":
            return 2 * values - 1

        if self.value_convention == "flux_pm":
            return values

        raise ValueError(f"Unsupported value convention: {self.value_convention}")

    def value(self, config: npt.ArrayLike) -> int:
        arr = self._as_config(config)
        values = arr[self._variable_indices]
        electric = self._electric_values(values)
        return int(np.sum(electric))

    def internal_target(self) -> int:
        if self.value_convention == "flux_pm":
            return internal_flux_winding_value(
                self.target,
                flux_normalization=self.flux_normalization,
            )

        # QDM binary convention uses E = 2n - 1 internally.
        # The target is already in the raw integer electric-winding convention.
        return int(self.target)

    def is_satisfied(self, config: npt.ArrayLike) -> bool:
        return self.value(config) == self.internal_target()

    def check(self, config: npt.ArrayLike) -> ConstraintResult:
        actual = self.value(config)
        target = self.internal_target()
        residual = actual - target
        satisfied = residual == 0

        return ConstraintResult(
            satisfied=satisfied,
            name=self.name,
            residual=residual,
            message=(
                f"{self.name}(direction={self.direction}): "
                f"value={actual}, target={self.target}, "
                f"internal_target={target}, residual={residual}"
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

        assigned_values = arr[variable_indices[assigned_local]]
        current = int(np.sum(self._electric_values(assigned_values)))

        remaining = int(np.count_nonzero(~assigned_local))

        # Each unassigned link contributes either -1 or +1.
        min_possible = current - remaining
        max_possible = current + remaining

        target = self.internal_target()

        if target < min_possible:
            return False

        if target > max_possible:
            return False

        if remaining == 0:
            return current == target

        return True
