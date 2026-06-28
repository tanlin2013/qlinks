from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from qlinks.constraints.base import BaseSectorCondition, ConstraintPropagation
from qlinks.lattice import BoundaryCondition, KagomeLattice, TriangularLattice
from qlinks.variables import VariableLayout

TriangularCycleDirection = Literal["a", "b"]
Z2ValueConvention = Literal["binary", "flux_pm"]

_TRIANGULAR_KIND_DISPLACEMENTS: dict[str, tuple[int, int]] = {
    "a": (1, 0),
    "b": (0, 1),
    "c": (-1, 1),
}


@dataclass(frozen=True, slots=True)
class Z2CutData:
    """Cached link and variable indices for one Z2 winding cut.

    Attributes:
        link_ids: Physical link ids crossing the cut.
        variable_indices: Layout variable indices associated with ``link_ids``.
    """

    link_ids: npt.NDArray[np.int64]
    variable_indices: npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class TriangularZ2WindingSector(BaseSectorCondition):
    """
    Z2 topological winding sector for triangular-lattice QDM/QLM on a torus.

    The sector is the parity of occupied/electric-positive links crossing a
    non-contractible cut. The labels ``direction="a"`` and ``direction="b"``
    refer to the two independent periodic cell directions of the triangular
    torus, not to filtering links by ``link.kind``.

    The cut is constructed from the periodic image shift of each link. This is
    important because triangular ``c`` links can also cross the primitive seams.
    Counting only wrapping links of kind ``"a"`` or ``"b"`` is not invariant
    under all triangular QDM rhombus flips.
    """

    layout: VariableLayout
    lattice: TriangularLattice
    direction: TriangularCycleDirection
    target: int
    value_convention: Z2ValueConvention = "binary"
    name: str = "triangular_z2_winding_sector"

    @classmethod
    def _link_image_shift(
        cls,
        *,
        lattice: TriangularLattice,
        link_id: int,
    ) -> tuple[int, int]:
        link = lattice.links[int(link_id)]

        if link.kind not in _TRIANGULAR_KIND_DISPLACEMENTS:
            raise ValueError(f"Unsupported triangular link kind: {link.kind!r}")

        dx, dy = _TRIANGULAR_KIND_DISPLACEMENTS[str(link.kind)]

        source = lattice.sites[int(link.source)]
        target = lattice.sites[int(link.target)]

        sx, sy = (int(source.cell[0]), int(source.cell[1]))
        tx, ty = (int(target.cell[0]), int(target.cell[1]))

        expected_tx = sx + dx
        expected_ty = sy + dy

        delta_x = expected_tx - tx
        delta_y = expected_ty - ty

        if delta_x % lattice.lx != 0:
            raise ValueError(
                f"Link {link_id} has inconsistent x image shift: "
                f"expected target x {expected_tx}, stored target x {tx}."
            )

        if delta_y % lattice.ly != 0:
            raise ValueError(
                f"Link {link_id} has inconsistent y image shift: "
                f"expected target y {expected_ty}, stored target y {ty}."
            )

        return delta_x // lattice.lx, delta_y // lattice.ly

    @classmethod
    def cut_data(
        cls,
        *,
        layout: VariableLayout,
        lattice: TriangularLattice,
        direction: TriangularCycleDirection,
    ) -> Z2CutData:
        if lattice.boundary_condition != BoundaryCondition.PERIODIC:
            raise ValueError("TriangularZ2WindingSector requires periodic boundary conditions.")

        if direction not in ("a", "b"):
            raise ValueError("direction must be 'a' or 'b'.")

        direction_axis = 0 if direction == "a" else 1

        link_ids: list[int] = []

        for link in lattice.links:
            image_shift = cls._link_image_shift(
                lattice=lattice,
                link_id=int(link.id),
            )

            # Z2 winding only needs parity. A shift of -1 and +1 are the same
            # modulo 2.
            if int(image_shift[direction_axis]) % 2 != 0:
                link_ids.append(int(link.id))

        if len(link_ids) == 0:
            raise ValueError(f"No links cross the triangular {direction!r} Z2 cut.")

        variable_indices = np.asarray(
            [layout.link_variable_index(int(link_id)) for link_id in link_ids],
            dtype=np.int64,
        )

        return Z2CutData(
            link_ids=np.asarray(link_ids, dtype=np.int64),
            variable_indices=variable_indices,
        )

    def __post_init__(self) -> None:
        cut = self.cut_data(
            layout=self.layout,
            lattice=self.lattice,
            direction=self.direction,
        )

        object.__setattr__(self, "_link_ids", cut.link_ids)
        object.__setattr__(self, "_variable_indices", cut.variable_indices)
        object.__setattr__(self, "target", int(self.target))
        object.__setattr__(self, "_n_cut_links", int(cut.variable_indices.size))

        self.validate_target(
            target=self.target,
            layout=self.layout,
            lattice=self.lattice,
            direction=self.direction,
            value_convention=self.value_convention,
        )

    @property
    def link_ids(self) -> npt.NDArray[np.int64]:
        return self._link_ids.copy()

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def _occupation_values(
        self,
        values: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
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
        return self.propagate(config, assigned_mask).consistent

    def propagate(
        self,
        config: npt.ArrayLike,
        assigned_mask: npt.ArrayLike,
    ) -> ConstraintPropagation:
        arr = np.asarray(config, dtype=np.int64)
        assigned = np.asarray(assigned_mask, dtype=bool)
        assigned_local = assigned[self._variable_indices]

        values = arr[self._variable_indices[assigned_local]]
        current_parity = int(np.sum(self._occupation_values(values)) % 2)

        unassigned_variables = self._variable_indices[~assigned_local]

        if unassigned_variables.size == 0:
            if current_parity != int(self.target):
                return ConstraintPropagation.contradiction()
            return ConstraintPropagation.no_change()

        # When one cut variable remains, the Z2 target determines its
        # occupation parity.  Force it for the usual binary / flux_pm local
        # spaces when the needed parity selects a unique local value.
        if unassigned_variables.size == 1:
            variable_index = int(unassigned_variables[0])
            needed_parity = (int(self.target) - current_parity) % 2
            local_values = np.asarray(
                self.layout.local_space(variable_index).values,
                dtype=np.int64,
            )
            local_occupations = self._occupation_values(local_values)
            allowed_values = local_values[(local_occupations % 2) == needed_parity]
            unique_allowed_values = np.unique(allowed_values.astype(np.int64, copy=False))
            if unique_allowed_values.size == 0:
                return ConstraintPropagation.contradiction()
            if unique_allowed_values.size == 1:
                return ConstraintPropagation(
                    forced_assignments=((variable_index, int(unique_allowed_values[0])),)
                )

        return ConstraintPropagation.no_change()

    @classmethod
    def allowed_targets(
        cls,
        *,
        layout: VariableLayout,
        lattice: TriangularLattice,
        direction: TriangularCycleDirection,
        value_convention: Z2ValueConvention = "binary",
    ) -> tuple[int, ...]:
        # This also validates the direction/lattice and builds the cut.
        _ = cls.cut_data(
            layout=layout,
            lattice=lattice,
            direction=direction,
        )

        if value_convention not in ("binary", "flux_pm"):
            raise ValueError("value_convention must be 'binary' or 'flux_pm'.")

        return (0, 1)

    @classmethod
    def validate_target(
        cls,
        *,
        target: int,
        layout: VariableLayout,
        lattice: TriangularLattice,
        direction: TriangularCycleDirection,
        value_convention: Z2ValueConvention = "binary",
    ) -> None:
        allowed = cls.allowed_targets(
            layout=layout,
            lattice=lattice,
            direction=direction,
            value_convention=value_convention,
        )

        if int(target) not in allowed:
            raise ValueError(
                f"Illegal {cls.__name__} target {target} for "
                f"direction={direction!r}. Allowed targets are {allowed}."
            )


KagomeCycleDirection = Literal["a", "b"]


@dataclass(frozen=True, slots=True)
class KagomeZ2WindingSector(BaseSectorCondition):
    """Z2 topological winding sector for kagome QDM/QLM on a torus.

    The sector is the parity of occupied/electric-positive kagome links crossing
    a non-contractible cut of the triangular Bravais torus.  The labels
    ``direction="a"`` and ``direction="b"`` refer to the two primitive cell
    directions, not to a bond kind.
    """

    layout: VariableLayout
    lattice: KagomeLattice
    direction: KagomeCycleDirection
    target: int
    value_convention: Z2ValueConvention = "binary"
    name: str = "kagome_z2_winding_sector"

    @classmethod
    def _link_image_shift(
        cls,
        *,
        lattice: KagomeLattice,
        link_id: int,
    ) -> tuple[int, int]:
        link = lattice.links[int(link_id)]
        dx, dy = lattice.link_cell_displacement(str(link.kind))

        source = lattice.sites[int(link.source)]
        target = lattice.sites[int(link.target)]

        sx, sy = (int(source.cell[0]), int(source.cell[1]))
        tx, ty = (int(target.cell[0]), int(target.cell[1]))

        expected_tx = sx + dx
        expected_ty = sy + dy

        delta_x = expected_tx - tx
        delta_y = expected_ty - ty

        if delta_x % lattice.lx != 0:
            raise ValueError(
                f"Link {link_id} has inconsistent x image shift: "
                f"expected target x {expected_tx}, stored target x {tx}."
            )

        if delta_y % lattice.ly != 0:
            raise ValueError(
                f"Link {link_id} has inconsistent y image shift: "
                f"expected target y {expected_ty}, stored target y {ty}."
            )

        return delta_x // lattice.lx, delta_y // lattice.ly

    @classmethod
    def cut_data(
        cls,
        *,
        layout: VariableLayout,
        lattice: KagomeLattice,
        direction: KagomeCycleDirection,
    ) -> Z2CutData:
        if lattice.boundary_condition != BoundaryCondition.PERIODIC:
            raise ValueError("KagomeZ2WindingSector requires periodic boundary conditions.")

        if direction not in ("a", "b"):
            raise ValueError("direction must be 'a' or 'b'.")

        direction_axis = 0 if direction == "a" else 1
        link_ids: list[int] = []

        for link in lattice.links:
            image_shift = cls._link_image_shift(
                lattice=lattice,
                link_id=int(link.id),
            )
            if int(image_shift[direction_axis]) % 2 != 0:
                link_ids.append(int(link.id))

        if len(link_ids) == 0:
            raise ValueError(f"No links cross the kagome {direction!r} Z2 cut.")

        variable_indices = np.asarray(
            [layout.link_variable_index(int(link_id)) for link_id in link_ids],
            dtype=np.int64,
        )

        return Z2CutData(
            link_ids=np.asarray(link_ids, dtype=np.int64),
            variable_indices=variable_indices,
        )

    def __post_init__(self) -> None:
        cut = self.cut_data(
            layout=self.layout,
            lattice=self.lattice,
            direction=self.direction,
        )

        object.__setattr__(self, "_link_ids", cut.link_ids)
        object.__setattr__(self, "_variable_indices", cut.variable_indices)
        object.__setattr__(self, "target", int(self.target))
        object.__setattr__(self, "_n_cut_links", int(cut.variable_indices.size))

        self.validate_target(
            target=self.target,
            layout=self.layout,
            lattice=self.lattice,
            direction=self.direction,
            value_convention=self.value_convention,
        )

    @property
    def link_ids(self) -> npt.NDArray[np.int64]:
        return self._link_ids.copy()

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def _occupation_values(
        self,
        values: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
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
        return self.propagate(config, assigned_mask).consistent

    def propagate(
        self,
        config: npt.ArrayLike,
        assigned_mask: npt.ArrayLike,
    ) -> ConstraintPropagation:
        arr = np.asarray(config, dtype=np.int64)
        assigned = np.asarray(assigned_mask, dtype=bool)
        assigned_local = assigned[self._variable_indices]

        values = arr[self._variable_indices[assigned_local]]
        current_parity = int(np.sum(self._occupation_values(values)) % 2)
        unassigned_variables = self._variable_indices[~assigned_local]

        if unassigned_variables.size == 0:
            if current_parity != int(self.target):
                return ConstraintPropagation.contradiction()
            return ConstraintPropagation.no_change()

        if unassigned_variables.size == 1:
            variable_index = int(unassigned_variables[0])
            needed_parity = (int(self.target) - current_parity) % 2
            local_values = np.asarray(
                self.layout.local_space(variable_index).values,
                dtype=np.int64,
            )
            local_occupations = self._occupation_values(local_values)
            allowed_values = local_values[(local_occupations % 2) == needed_parity]
            unique_allowed_values = np.unique(allowed_values.astype(np.int64, copy=False))
            if unique_allowed_values.size == 0:
                return ConstraintPropagation.contradiction()
            if unique_allowed_values.size == 1:
                return ConstraintPropagation(
                    forced_assignments=((variable_index, int(unique_allowed_values[0])),)
                )

        return ConstraintPropagation.no_change()

    @classmethod
    def allowed_targets(
        cls,
        *,
        layout: VariableLayout,
        lattice: KagomeLattice,
        direction: KagomeCycleDirection,
        value_convention: Z2ValueConvention = "binary",
    ) -> tuple[int, ...]:
        _ = cls.cut_data(
            layout=layout,
            lattice=lattice,
            direction=direction,
        )

        if value_convention not in ("binary", "flux_pm"):
            raise ValueError("value_convention must be 'binary' or 'flux_pm'.")

        return (0, 1)
