from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qlinks.lattice import SquareLattice
from qlinks.operators.base import BaseLocalOperator, OperatorAction
from qlinks.variables import VariableKind, VariableLayout


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
        )

        if link_ids.ndim != 1:
            link_ids = link_ids.reshape(-1)

        variable_indices = np.asarray(
            [
                self.layout.link_variable_index(int(link_id))
                for link_id in link_ids
            ],
            dtype=np.int64,
        )

        if variable_indices.size == 0:
            raise ValueError(f"Site {self.site_id} has no incident links.")

        for variable_index in variable_indices:
            values = set(
                int(v)
                for v in self.layout.local_space(int(variable_index)).values.tolist()
            )
            if values != {-1, 1}:
                raise ValueError(
                    "ToricCodeStarFlipOperator requires link variables {-1, +1}."
                )

        object.__setattr__(self, "_link_ids", link_ids)
        object.__setattr__(self, "_variable_indices", variable_indices)

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

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

        variable_indices = np.asarray(
            [
                self.layout.variable_index(VariableKind.LINK, int(link_id))
                for link_id in plaquette.links
            ],
            dtype=np.int64,
        )

        if variable_indices.size == 0:
            raise ValueError(f"Plaquette {self.plaquette_id} has no links.")

        for variable_index in variable_indices:
            values = set(
                int(v)
                for v in self.layout.local_space(int(variable_index)).values.tolist()
            )
            if values != {-1, 1}:
                raise ValueError(
                    "ToricCodePlaquetteFluxOperator requires link variables {-1, +1}."
                )

        object.__setattr__(self, "_variable_indices", variable_indices)

    @property
    def variable_indices(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self._variable_indices.copy()

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]:
        arr = self._as_config(config)

        flux = int(np.prod(arr[self._variable_indices]))
        coefficient = complex(self.coefficient) * flux

        # Diagonal action: return same configuration.
        return (OperatorAction(coefficient, arr.copy()),)
