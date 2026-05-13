from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qlinks.lattice import SquareLattice
from qlinks.operators.base import BaseLocalOperator, OperatorAction
from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class ToricCodeStarFlipOperator(BaseLocalOperator):
    """
    Toric-code star operator A_v in the Z basis.

    It flips all links incident on site v.
    """

    layout: VariableLayout
    lattice: SquareLattice
    site_id: int
    coefficient: complex = -1.0
    name: str = "toric_code_star_flip"

    def __post_init__(self) -> None:
        link_ids = np.asarray(
            self.lattice.incident_links(int(self.site_id)),
            dtype=np.int64,
        ).reshape(-1)

        variable_indices = self._link_variable_indices(link_ids)

        if variable_indices.size == 0:
            raise ValueError(f"Site {self.site_id} has no incident links.")

        self._validate_local_spaces(
            variable_indices,
            {-1, 1},
            operator_name=type(self).__name__,
        )

        object.__setattr__(self, "_link_ids", self._cached_array(link_ids))
        object.__setattr__(self, "_variable_indices", self._cached_array(variable_indices))

    @property
    def link_ids(self) -> npt.NDArray[np.int64]:
        return self._copy_indices(self._link_ids)

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._copy_indices(self._variable_indices)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._copy_indices(self._variable_indices)

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]:
        arr = self._as_config(config)

        new = arr.copy()
        new[self._variable_indices] *= -1

        return (OperatorAction(self.coefficient, new),)


@dataclass(frozen=True, slots=True)
class ToricCodePlaquetteFluxOperator(BaseLocalOperator):
    """
    Toric-code plaquette operator B_p in the Z basis.

    It is diagonal:

        B_p |z> = prod_{l in boundary(p)} z_l |z>
    """

    layout: VariableLayout
    lattice: SquareLattice
    plaquette_id: int
    coefficient: complex = -1.0
    name: str = "toric_code_plaquette_flux"

    def __post_init__(self) -> None:
        plaquette = self.lattice.plaquettes[int(self.plaquette_id)]

        link_ids = np.asarray(plaquette.links, dtype=np.int64)
        variable_indices = self._link_variable_indices(link_ids)

        if variable_indices.size == 0:
            raise ValueError(f"Plaquette {self.plaquette_id} has no links.")

        self._validate_local_spaces(
            variable_indices,
            {-1, 1},
            operator_name=type(self).__name__,
        )

        object.__setattr__(self, "_link_ids", self._cached_array(link_ids))
        object.__setattr__(self, "_variable_indices", self._cached_array(variable_indices))

    @property
    def link_ids(self) -> npt.NDArray[np.int64]:
        return self._copy_indices(self._link_ids)

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._copy_indices(self._variable_indices)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._copy_indices(self._variable_indices)

    def diagonal_value(self, config: npt.ArrayLike) -> complex:
        arr = self._as_config(config)
        flux = int(np.prod(arr[self._variable_indices]))
        return complex(self.coefficient) * flux

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]:
        arr = self._as_config(config)
        return (OperatorAction(self.diagonal_value(arr), arr.copy()),)
