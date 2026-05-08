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
        site_variable = self.layout.site_variable_index(self.site_id)
        neighbor_sites = self.lattice.neighbors(self.site_id)

        neighbor_variables = np.asarray(
            [self.layout.site_variable_index(int(site)) for site in neighbor_sites],
            dtype=np.int64,
        )

        values = set(self.layout.local_space(site_variable).values.tolist())
        if values != {0, 1}:
            raise ValueError("PXPSpinFlipOperator requires binary site variables {0, 1}.")

        self.layout.local_space(site_variable).validate_value(self.occupied_value)

        for variable_index in neighbor_variables:
            self.layout.local_space(int(variable_index)).validate_value(self.occupied_value)

        object.__setattr__(self, "_site_variable", site_variable)
        object.__setattr__(self, "_neighbor_sites", neighbor_sites)
        object.__setattr__(self, "_neighbor_variables", neighbor_variables)

    @property
    def site_variable(self) -> int:
        return int(self._site_variable)

    @property
    def neighbor_sites(self) -> npt.NDArray[np.int64]:
        return self._neighbor_sites.copy()

    @property
    def neighbor_variables(self) -> npt.NDArray[np.int64]:
        return self._neighbor_variables.copy()

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
