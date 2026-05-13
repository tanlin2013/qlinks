from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qlinks.lattice import LatticeGraph
from qlinks.operators.base import BaseLocalOperator, OperatorAction
from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class PXPSpinFlipOperator(BaseLocalOperator):
    """
    Constrained spin flip for PXP/Rydberg-blockade models.

    The spin at `site_id` is flipped only if all neighboring sites are not
    occupied.

    With binary variables:
        0 <-> 1

    and usually:
        occupied_value = 1
    """

    layout: VariableLayout
    lattice: LatticeGraph
    site_id: int
    coefficient: complex = 1.0
    occupied_value: int = 1
    name: str = "pxp_spin_flip"

    def __post_init__(self) -> None:
        site_variable = self._site_variable_index(self.site_id)

        neighbor_sites = np.asarray(
            self.lattice.neighbors(self.site_id),
            dtype=np.int64,
        )
        neighbor_variables = self._site_variable_indices(neighbor_sites)

        self._validate_local_space_values(
            site_variable,
            {0, 1},
            operator_name=type(self).__name__,
        )

        self.layout.local_space(site_variable).validate_value(self.occupied_value)

        for variable_index in neighbor_variables:
            self._validate_local_space_values(
                int(variable_index),
                {0, 1},
                operator_name=type(self).__name__,
            )
            self.layout.local_space(int(variable_index)).validate_value(self.occupied_value)

        object.__setattr__(self, "_site_variable", int(site_variable))
        object.__setattr__(self, "_neighbor_sites", self._cached_array(neighbor_sites))
        object.__setattr__(self, "_neighbor_variables", self._cached_array(neighbor_variables))

    @property
    def site_variable(self) -> int:
        return int(self._site_variable)

    @property
    def neighbor_sites(self) -> npt.NDArray[np.int64]:
        return self._copy_indices(self._neighbor_sites)

    @property
    def neighbor_variables(self) -> npt.NDArray[np.int64]:
        return self._copy_indices(self._neighbor_variables)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return np.asarray(
            [self._site_variable, *self._neighbor_variables.tolist()],
            dtype=np.int64,
        )

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]:
        arr = self._as_config(config)

        if np.any(arr[self._neighbor_variables] == self.occupied_value):
            return ()

        new = arr.copy()
        new[self._site_variable] = 1 - new[self._site_variable]

        return (OperatorAction(self.coefficient, new),)
