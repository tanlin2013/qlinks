from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from typing import Literal

import numpy as np
import numpy.typing as npt

from qlinks.constraints.base import BaseSectorCondition, ConstraintResult
from qlinks.lattice import BoundaryCondition, HoneycombLattice, SquareLattice
from qlinks.variables import VariableLayout

Direction = Literal["x", "y"]
FluxNormalization = Literal["integer_flux", "spin_half"]
WindingTarget = int | Fraction | str


def normalize_winding_target(target: WindingTarget) -> Fraction:
    """
    Normalize a user-facing winding target.

    Accepts:
        1
        Fraction(3, 2)
        "3/2"

    Avoid floats to prevent precision ambiguity.
    """
    if isinstance(target, Fraction):
        return target

    if isinstance(target, Integral):
        return Fraction(int(target), 1)

    if isinstance(target, str):
        return Fraction(target)

    raise TypeError(
        "winding target must be an int, Fraction, or fraction string such as '3/2'."
    )


def internal_flux_winding_value(
    winding: WindingTarget,
    *,
    flux_normalization: FluxNormalization,
) -> int:
    """
    Convert user-facing winding target into raw integer flux target.

    integer_flux:
        stored flux s_l ∈ {-1,+1} is the physical electric field.
        raw_target = winding

    spin_half:
        stored flux s_l ∈ {-1,+1} represents twice the physical spin-half
        electric field, E_l = s_l / 2.
        raw_target = 2 * winding
    """
    target = normalize_winding_target(winding)

    if flux_normalization == "integer_flux":
        raw = target
    elif flux_normalization == "spin_half":
        raw = 2 * target
    else:
        raise ValueError("flux_normalization must be 'integer_flux' or 'spin_half'.")

    if raw.denominator != 1:
        raise ValueError(
            f"Target {target} is incompatible with flux_normalization="
            f"{flux_normalization!r}: raw target {raw} is not an integer."
        )

    return int(raw)


def user_winding_value_from_internal(
    raw_target: int,
    *,
    flux_normalization: FluxNormalization,
) -> Fraction:
    """
    Convert raw integer winding value back to the user-facing convention.
    """
    raw = Fraction(int(raw_target), 1)

    if flux_normalization == "integer_flux":
        return raw

    if flux_normalization == "spin_half":
        return raw / 2

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


def allowed_signed_sum_targets(
    *,
    layout: VariableLayout,
    variable_indices: npt.ArrayLike,
    signs: npt.ArrayLike,
    value_transform=None,
) -> tuple[int, ...]:
    """
    Return all possible raw values of sum_i signs[i] * f(x_i).

    Parameters
    ----------
    layout:
        Variable layout.

    variable_indices:
        Variables included in the diagonal quantum number.

    signs:
        Integer signs/covector coefficients.

    value_transform:
        Optional function mapping local-space values to the internal electric
        values used by the sector. If None, values are used directly.
    """
    variable_indices = np.asarray(variable_indices, dtype=np.int64)
    signs = np.asarray(signs, dtype=np.int64)

    if variable_indices.shape != signs.shape:
        raise ValueError("variable_indices and signs must have the same shape.")

    values: set[int] = {0}

    for variable_index, sign in zip(variable_indices, signs, strict=True):
        local_values = np.asarray(
            layout.local_space(int(variable_index)).values,
            dtype=np.int64,
        )

        if value_transform is not None:
            local_values = np.asarray(value_transform(local_values), dtype=np.int64)

        next_values: set[int] = set()

        for current in values:
            for local_value in local_values:
                next_values.add(int(current + int(sign) * int(local_value)))

        values = next_values

    return tuple(sorted(values))


def user_targets_from_raw_flux_targets(
    raw_targets: tuple[int, ...],
    *,
    flux_normalization: FluxNormalization,
) -> tuple[int, ...]:
    """
    Convert raw stored-flux winding values into user-facing winding labels.

    integer_flux:
        raw target equals user target.

    spin_half:
        stored flux s_l in {-1,+1} represents twice the physical spin-half
        electric field, so raw target = 2 * user target. Only even raw targets
        are valid integer user labels under the current API.
    """
    if flux_normalization == "integer_flux":
        return tuple(int(x) for x in raw_targets)

    if flux_normalization == "spin_half":
        return tuple(int(x // 2) for x in raw_targets if int(x) % 2 == 0)

    raise ValueError("flux_normalization must be 'integer_flux' or 'spin_half'.")


def raw_targets_from_user_targets(
    user_targets: tuple[int, ...],
    *,
    flux_normalization: FluxNormalization,
) -> tuple[int, ...]:
    return tuple(
        internal_flux_winding_value(
            int(target),
            flux_normalization=flux_normalization,
        )
        for target in user_targets
    )


@dataclass(frozen=True, slots=True)
class WindingCutData:
    """
    Cached data defining one winding cut.

    link_ids:
        Physical lattice links participating in the winding cut.

    signs:
        Integer covector signs used in the winding value.

    variable_indices:
        VariableLayout indices corresponding to link_ids.
    """

    link_ids: npt.NDArray[np.int64]
    signs: npt.NDArray[np.int64]
    variable_indices: npt.NDArray[np.int64]


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
    target: WindingTarget
    name: str = "square_winding_sector"
    flux_normalization: FluxNormalization = "spin_half"

    def __post_init__(self) -> None:
        cut = self.cut_data(
            layout=self.layout,
            lattice=self.lattice,
            direction=self.direction,
        )

        object.__setattr__(self, "_link_ids", cut.link_ids)
        object.__setattr__(self, "_signs", cut.signs)
        object.__setattr__(self, "_variable_indices", cut.variable_indices)
        object.__setattr__(
            self,
            "target",
            normalize_winding_target(self.target),
        )

        self.validate_target(
            target=self.target,
            layout=self.layout,
            lattice=self.lattice,
            direction=self.direction,
            flux_normalization=self.flux_normalization,
        )

    @classmethod
    def cut_data(
        cls,
        *,
        layout: VariableLayout,
        lattice: SquareLattice,
        direction: Direction,
    ) -> WindingCutData:
        if lattice.boundary_condition != BoundaryCondition.PERIODIC:
            raise ValueError("SquareWindingSector requires a periodic SquareLattice.")

        if direction not in ("x", "y"):
            raise ValueError("direction must be 'x' or 'y'.")

        link_ids, signs = _signed_direction_links_annihilating_plaquettes(
            lattice=lattice,
            direction=direction,
        )

        variable_indices = np.asarray(
            [layout.link_variable_index(int(link_id)) for link_id in link_ids],
            dtype=np.int64,
        )

        return WindingCutData(
            link_ids=np.asarray(link_ids, dtype=np.int64),
            signs=np.asarray(signs, dtype=np.int64),
            variable_indices=variable_indices,
        )

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
                int(v)
                for v in self.layout.local_space(int(variable_index)).values.tolist()
            )

        if values_seen <= {-1, 1}:
            return internal_flux_winding_value(
                self.target,
                flux_normalization=self.flux_normalization,
            )

        return internal_flux_winding_value(
            self.target,
            flux_normalization="integer_flux",
        )

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

    @classmethod
    def allowed_internal_targets(
        cls,
        *,
        layout: VariableLayout,
        lattice: SquareLattice,
        direction: Direction,
    ) -> tuple[int, ...]:
        cut = cls.cut_data(
            layout=layout,
            lattice=lattice,
            direction=direction,
        )

        return allowed_signed_sum_targets(
            layout=layout,
            variable_indices=cut.variable_indices,
            signs=cut.signs,
        )

    @classmethod
    def allowed_targets(
        cls,
        *,
        layout: VariableLayout,
        lattice: SquareLattice,
        direction: Direction,
        flux_normalization: FluxNormalization = "spin_half",
    ) -> tuple[Fraction, ...]:
        cut = cls.cut_data(
            layout=layout,
            lattice=lattice,
            direction=direction,
        )

        raw_targets = allowed_signed_sum_targets(
            layout=layout,
            variable_indices=cut.variable_indices,
            signs=cut.signs,
        )

        values_seen: set[int] = set()
        for variable_index in cut.variable_indices:
            values_seen.update(
                int(v)
                for v in layout.local_space(int(variable_index)).values.tolist()
            )

        # QLM flux sector: stored values {-1,+1}.
        # Convert raw stored-flux labels to user-facing labels.
        if values_seen <= {-1, 1}:
            return tuple(
                user_winding_value_from_internal(
                    raw_target,
                    flux_normalization=flux_normalization,
                )
                for raw_target in raw_targets
            )

        # QDM/count-like sector: target is raw integer count-like label.
        return tuple(Fraction(raw_target, 1) for raw_target in raw_targets)

    @classmethod
    def validate_target(
        cls,
        *,
        target: WindingTarget,
        layout: VariableLayout,
        lattice: SquareLattice,
        direction: Direction,
        flux_normalization: FluxNormalization = "spin_half",
    ) -> None:
        normalized_target = normalize_winding_target(target)

        allowed = cls.allowed_targets(
            layout=layout,
            lattice=lattice,
            direction=direction,
            flux_normalization=flux_normalization,
        )

        if normalized_target not in allowed:
            raise ValueError(
                f"Illegal {cls.__name__} target {normalized_target} for "
                f"direction={direction!r}. Allowed targets are {allowed}."
            )


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
    target: WindingTarget
    name: str = "square_qdm_electric_winding_sector"

    def __post_init__(self) -> None:
        cut = self.cut_data(
            layout=self.layout,
            lattice=self.lattice,
            direction=self.direction,
        )

        object.__setattr__(self, "_link_ids", cut.link_ids)
        object.__setattr__(self, "_signs", cut.signs)
        object.__setattr__(self, "_variable_indices", cut.variable_indices)
        object.__setattr__(self, "target", int(self.target))

        self.validate_target(
            target=int(self.target),
            layout=self.layout,
            lattice=self.lattice,
            direction=self.direction,
        )

    @classmethod
    def cut_data(
        cls,
        *,
        layout: VariableLayout,
        lattice: SquareLattice,
        direction: Direction,
    ) -> WindingCutData:
        if lattice.boundary_condition != BoundaryCondition.PERIODIC:
            raise ValueError(
                "SquareQDMElectricWindingSector requires a periodic SquareLattice."
            )

        if direction not in ("x", "y"):
            raise ValueError("direction must be 'x' or 'y'.")

        base_link_ids, base_signs = _signed_direction_links_annihilating_plaquettes(
            lattice=lattice,
            direction=direction,
        )

        qdm_signs: list[int] = []

        for link_id, base_sign in zip(base_link_ids, base_signs, strict=True):
            link = lattice.links[int(link_id)]
            source_site = lattice.sites[int(link.source)]
            source_x, source_y = source_site.cell

            staggered_sign = 1 if (int(source_x) + int(source_y)) % 2 == 0 else -1
            qdm_signs.append(int(base_sign) * staggered_sign)

        signs = np.asarray(qdm_signs, dtype=np.int64)

        variable_indices = np.asarray(
            [layout.link_variable_index(int(link_id)) for link_id in base_link_ids],
            dtype=np.int64,
        )

        return WindingCutData(
            link_ids=np.asarray(base_link_ids, dtype=np.int64),
            signs=signs,
            variable_indices=variable_indices,
        )

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
        arr = self._as_config(configuration)
        n = arr[self._variable_indices]
        return int(np.sum(self._signs * (2 * n - 1)))

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

    @classmethod
    def allowed_targets(
        cls,
        *,
        layout: VariableLayout,
        lattice: SquareLattice,
        direction: Direction,
    ) -> tuple[int, ...]:
        cut = cls.cut_data(
            layout=layout,
            lattice=lattice,
            direction=direction,
        )

        return allowed_signed_sum_targets(
            layout=layout,
            variable_indices=cut.variable_indices,
            signs=cut.signs,
            value_transform=lambda values: 2 * values - 1,
        )

    @classmethod
    def validate_target(
        cls,
        *,
        target: int,
        layout: VariableLayout,
        lattice: SquareLattice,
        direction: Direction,
    ) -> None:
        allowed = cls.allowed_targets(
            layout=layout,
            lattice=lattice,
            direction=direction,
        )

        if int(target) not in allowed:
            raise ValueError(
                f"Illegal {cls.__name__} target {target} for "
                f"direction={direction!r}. Allowed targets are {allowed}."
            )


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
    target: WindingTarget
    value_convention: Literal["binary", "flux_pm"] = "binary"
    name: str = "honeycomb_electric_winding_sector"
    flux_normalization: FluxNormalization = "spin_half"

    def __post_init__(self) -> None:
        if self.value_convention not in ("binary", "flux_pm"):
            raise ValueError("value_convention must be 'binary' or 'flux_pm'.")

        cut = self.cut_data(
            layout=self.layout,
            lattice=self.lattice,
            direction=self.direction,
        )

        object.__setattr__(self, "_link_ids", cut.link_ids)
        object.__setattr__(self, "_signs", cut.signs)
        object.__setattr__(self, "_variable_indices", cut.variable_indices)
        object.__setattr__(
            self,
            "target",
            normalize_winding_target(self.target),
        )

        self.validate_target(
            target=self.target,
            layout=self.layout,
            lattice=self.lattice,
            direction=self.direction,
            value_convention=self.value_convention,
            flux_normalization=self.flux_normalization,
        )

    @classmethod
    def cut_data(
        cls,
        *,
        layout: VariableLayout,
        lattice: HoneycombLattice,
        direction: Literal["x", "y"],
    ) -> WindingCutData:
        if lattice.boundary_condition != BoundaryCondition.PERIODIC:
            raise ValueError("HoneycombElectricWindingSector requires PBC.")

        if direction not in ("x", "y"):
            raise ValueError(
                "direction must be 'x' or 'y', referring to the two periodic "
                "unit-cell directions, not Cartesian plotting axes."
            )

        link_ids = [
            int(link.id)
            for link in lattice.links
            if link.wrap and link.kind == direction
        ]

        if len(link_ids) == 0:
            raise ValueError(f"No wrapping {direction}-links found.")

        variable_indices = np.asarray(
            [layout.link_variable_index(int(link_id)) for link_id in link_ids],
            dtype=np.int64,
        )

        signs = np.ones(len(link_ids), dtype=np.int64)

        return WindingCutData(
            link_ids=np.asarray(link_ids, dtype=np.int64),
            signs=signs,
            variable_indices=variable_indices,
        )

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
        return int(np.dot(self._signs, electric))

    def internal_target(self) -> int:
        if self.value_convention == "flux_pm":
            return internal_flux_winding_value(
                self.target,
                flux_normalization=self.flux_normalization,
            )

        return internal_flux_winding_value(
            self.target,
            flux_normalization="integer_flux",
        )

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

    @staticmethod
    def electric_values_from_convention(
        values: npt.NDArray[np.int64],
        *,
        value_convention: Literal["binary", "flux_pm"],
    ) -> npt.NDArray[np.int64]:
        if value_convention == "binary":
            return 2 * values - 1

        if value_convention == "flux_pm":
            return values

        raise ValueError("value_convention must be 'binary' or 'flux_pm'.")

    @classmethod
    def allowed_internal_targets(
        cls,
        *,
        layout: VariableLayout,
        lattice: HoneycombLattice,
        direction: Literal["x", "y"],
        value_convention: Literal["binary", "flux_pm"] = "binary",
    ) -> tuple[int, ...]:
        cut = cls.cut_data(
            layout=layout,
            lattice=lattice,
            direction=direction,
        )

        return allowed_signed_sum_targets(
            layout=layout,
            variable_indices=cut.variable_indices,
            signs=cut.signs,
            value_transform=lambda values: cls.electric_values_from_convention(
                values,
                value_convention=value_convention,
            ),
        )

    @classmethod
    def allowed_targets(
        cls,
        *,
        layout: VariableLayout,
        lattice: HoneycombLattice,
        direction: Literal["x", "y"],
        value_convention: Literal["binary", "flux_pm"] = "binary",
        flux_normalization: FluxNormalization = "spin_half",
    ) -> tuple[int, ...]:
        raw_targets = cls.allowed_internal_targets(
            layout=layout,
            lattice=lattice,
            direction=direction,
            value_convention=value_convention,
        )

        if value_convention == "flux_pm":
            return tuple(
                user_winding_value_from_internal(
                    raw,
                    flux_normalization=flux_normalization,
                )
                for raw in raw_targets
            )

            # Binary convention uses E = 2n - 1 internally and target is raw.
        return tuple(Fraction(raw, 1) for raw in raw_targets)

    @classmethod
    def validate_target(
        cls,
        *,
        target: int,
        layout: VariableLayout,
        lattice: HoneycombLattice,
        direction: Literal["x", "y"],
        value_convention: Literal["binary", "flux_pm"] = "binary",
        flux_normalization: FluxNormalization = "spin_half",
    ) -> None:
        normalized_target = normalize_winding_target(target)

        allowed = cls.allowed_targets(
            layout=layout,
            lattice=lattice,
            direction=direction,
            value_convention=value_convention,
            flux_normalization=flux_normalization,
        )

        if normalized_target not in allowed:
            raise ValueError(
                f"Illegal {cls.__name__} target {normalized_target} for "
                f"direction={direction!r}. Allowed targets are {allowed}."
            )
