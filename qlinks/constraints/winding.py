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


def _signed_direction_links_annihilating_plaquettes(
    *,
    lattice,
    direction: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return signed direction links whose covector annihilates plaquettes.

    The returned ``link_ids`` and ``signs`` define a covector ``w`` with

        w[link_ids] = signs

    such that

        w @ lattice.plaquette_incidence_matrix() == 0.

    This is more robust than using wrapping links or all direction links with
    positive signs, especially on small PBC lattices such as 2x2.
    """
    link_ids = [int(link.id) for link in lattice.links if link.kind == direction]

    if len(link_ids) == 0:
        raise ValueError(f"No {direction}-links found.")

    selected_link_set = set(link_ids)
    signs_by_link: dict[int, int] = {}

    for seed_link_id in link_ids:
        if seed_link_id in signs_by_link:
            continue

        signs_by_link[seed_link_id] = 1
        search_stack = [seed_link_id]

        while search_stack:
            current_link_id = search_stack.pop()

            for plaquette_id in lattice.plaquette_ids:
                plaquette_links = lattice.plaquette_links(int(plaquette_id))
                plaquette_orientations = lattice.plaquette_orientations(int(plaquette_id))

                local_entries = [
                    (int(link_id), int(orientation))
                    for link_id, orientation in zip(
                        plaquette_links,
                        plaquette_orientations,
                        strict=True,
                    )
                    if int(link_id) in selected_link_set
                ]

                if len(local_entries) != 2:
                    continue

                first_link_id, first_orientation = local_entries[0]
                second_link_id, second_orientation = local_entries[1]

                if current_link_id not in (first_link_id, second_link_id):
                    continue

                if current_link_id == first_link_id:
                    known_link_id = first_link_id
                    known_orientation = first_orientation
                    unknown_link_id = second_link_id
                    unknown_orientation = second_orientation
                else:
                    known_link_id = second_link_id
                    known_orientation = second_orientation
                    unknown_link_id = first_link_id
                    unknown_orientation = first_orientation

                inferred_sign = -(
                    signs_by_link[known_link_id] * known_orientation // unknown_orientation
                )

                if unknown_link_id in signs_by_link:
                    if signs_by_link[unknown_link_id] != inferred_sign:
                        raise ValueError(
                            "Inconsistent winding-sign constraints for " f"direction={direction!r}."
                        )
                else:
                    signs_by_link[unknown_link_id] = inferred_sign
                    search_stack.append(unknown_link_id)

    ordered_link_ids = np.asarray(link_ids, dtype=np.int64)
    signs = np.asarray(
        [signs_by_link[int(link_id)] for link_id in ordered_link_ids],
        dtype=np.int64,
    )

    plaquette_incidence = lattice.plaquette_incidence_matrix().toarray()
    covector = np.zeros(lattice.num_links, dtype=np.int64)
    covector[ordered_link_ids] = signs

    winding_change = covector @ plaquette_incidence

    if np.any(winding_change != 0):
        raise ValueError(
            "Derived winding covector does not annihilate plaquette boundaries: "
            f"{winding_change.tolist()}"
        )

    return ordered_link_ids, signs


@dataclass(frozen=True, slots=True)
class SquareWindingSector(BaseSectorCondition):
    """Square-lattice electric winding sector.

    The winding covector is a signed direction-link covector chosen so that it
    annihilates every plaquette boundary. This guarantees that local plaquette
    flips preserve the sector, including on small PBC lattices.
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

        link_ids, signs = _signed_direction_links_annihilating_plaquettes(
            lattice=self.lattice,
            direction=self.direction,
        )

        variable_indices = np.asarray(
            [self.layout.link_variable_index(int(link_id)) for link_id in link_ids],
            dtype=np.int64,
        )

        object.__setattr__(self, "_link_ids", link_ids)
        object.__setattr__(self, "_signs", signs)
        object.__setattr__(self, "_variable_indices", variable_indices)
        object.__setattr__(self, "target", int(self.target))

    @property
    def link_ids(self) -> npt.NDArray[np.int64]:
        return self._link_ids.copy()

    @property
    def signs(self) -> np.ndarray:
        """Signs used in the winding covector."""
        return self._signs.copy()

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def value(self, config: npt.ArrayLike) -> int:
        arr = self._as_config(config)
        return int(np.dot(self._signs, arr[self._variable_indices]))

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
        signs = self._signs

        assigned_local = assigned[variable_indices]
        current = int(
            np.dot(
                signs[assigned_local],
                arr[variable_indices[assigned_local]],
            )
        )

        min_remaining = 0
        max_remaining = 0

        for local_position, variable_index in enumerate(variable_indices):
            if assigned_local[local_position]:
                continue

            sign = int(signs[local_position])
            values = self.layout.local_space(int(variable_index)).values

            signed_values = sign * values
            min_remaining += int(np.min(signed_values))
            max_remaining += int(np.max(signed_values))

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

        base_link_ids, base_signs = _signed_direction_links_annihilating_plaquettes(
            lattice=self.lattice,
            direction=self.direction,
        )

        qdm_signs: list[int] = []

        for link_id, base_sign in zip(base_link_ids, base_signs, strict=True):
            link = self.lattice.links[int(link_id)]
            source_site = self.lattice.sites[int(link.source)]
            source_x, source_y = source_site.cell

            staggered_sign = 1 if (int(source_x) + int(source_y)) % 2 == 0 else -1

            qdm_signs.append(int(base_sign) * staggered_sign)

        signs = np.asarray(qdm_signs, dtype=np.int64)

        variable_indices = np.asarray(
            [self.layout.link_variable_index(int(link_id)) for link_id in base_link_ids],
            dtype=np.int64,
        )

        object.__setattr__(self, "_link_ids", base_link_ids)
        object.__setattr__(self, "_signs", signs)
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

    @property
    def signs(self) -> npt.NDArray[np.int64]:
        return self._signs.copy()

    def value(self, configuration: np.ndarray) -> int:
        return int(
            np.dot(
                self._signs,
                configuration[self._variable_indices],
            )
        )

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
    """
    Electric winding sector for the honeycomb QLM on a periodic lattice.

    This sector fixes one of the two conserved electric-flux winding numbers
    on a honeycomb torus. The winding number is defined as the signed electric
    flux through a non-contractible cut of the periodic lattice. The signs are
    determined by the oriented-link convention of the lattice, so the sector is
    invariant under local plaquette flips.

    Notes
    -----
    The ``direction`` argument labels the two independent periodic directions
    of the integer unit-cell coordinates, not the Cartesian directions of the
    plotting embedding.

    Specifically,

        direction="x"

    means the winding sector associated with the first unit-cell direction,
    and

        direction="y"

    means the winding sector associated with the second unit-cell direction.

    For a honeycomb lattice these cell directions are generally oblique in the
    visual embedding. They should be understood as the two primitive torus
    cycles, or equivalently as a chosen basis of H_1(T^2, Z). Choosing a
    different pair of independent non-contractible cycles would give an
    equivalent winding basis, with sector labels related by an integer change
    of basis.

    The primitive vectors and basis offsets used for plotting do not define
    the winding sector. The winding sector is defined by the combinatorial
    periodic cell coordinates and the oriented link/cut convention.

    Parameters
    ----------
    layout:
        Variable layout whose link variables represent spin-half electric
        fluxes.

    lattice:
        Periodic honeycomb lattice.

    direction:
        Either ``"x"`` or ``"y"``. These refer to the two unit-cell periodic
        directions, not Cartesian plot axes.

    target:
        Target winding value in the chosen direction.

    flux_normalization:
        Normalization convention for the electric flux variables. For example,
        ``"spin_half"`` corresponds to link values ``{-1, +1}`` representing
        twice the physical spin-half electric field.
    """

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
            raise ValueError(
                "direction must be 'x' or 'y', referring to the two periodic "
                "unit-cell directions, not Cartesian plotting axes."
            )

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
