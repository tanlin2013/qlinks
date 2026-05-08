from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qlinks.constraints.base import BaseSectorCondition
from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class TotalValueSector(BaseSectorCondition):
    """
    Fix the total signed value over selected variables.

    Examples:
        total particle number
        total magnetization
        total electric flux on selected links
    """

    layout: VariableLayout
    target: int
    variable_indices: npt.NDArray[np.int64] | None = None
    coefficients: npt.NDArray[np.int64] | None = None
    name: str = "total_value_sector"

    def __post_init__(self) -> None:
        if self.variable_indices is None:
            variable_indices = np.arange(self.layout.n_variables, dtype=np.int64)
        else:
            variable_indices = np.asarray(self.variable_indices, dtype=np.int64)

        if variable_indices.ndim != 1:
            raise ValueError("variable_indices must be one-dimensional.")

        if self.coefficients is None:
            coefficients = np.ones(variable_indices.size, dtype=np.int64)
        else:
            coefficients = np.asarray(self.coefficients, dtype=np.int64)

        if coefficients.ndim != 1:
            raise ValueError("coefficients must be one-dimensional.")
        if coefficients.size != variable_indices.size:
            raise ValueError("coefficients and variable_indices must have the same length.")

        for variable_index in variable_indices:
            self.layout.spec(int(variable_index))

        object.__setattr__(self, "variable_indices", variable_indices)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "target", int(self.target))

    def value(self, config: npt.ArrayLike) -> int:
        arr = self._as_config(config)
        return int(np.dot(self.coefficients, arr[self.variable_indices]))


@dataclass(frozen=True, slots=True)
class ParitySector(BaseSectorCondition):
    """
    Fix the parity of the sum over selected variables.

    target should be 0 or 1.
    """

    layout: VariableLayout
    target: int
    variable_indices: npt.NDArray[np.int64] | None = None
    name: str = "parity_sector"

    def __post_init__(self) -> None:
        if self.target not in (0, 1):
            raise ValueError("ParitySector.target must be 0 or 1.")

        if self.variable_indices is None:
            variable_indices = np.arange(self.layout.n_variables, dtype=np.int64)
        else:
            variable_indices = np.asarray(self.variable_indices, dtype=np.int64)

        if variable_indices.ndim != 1:
            raise ValueError("variable_indices must be one-dimensional.")

        for variable_index in variable_indices:
            self.layout.spec(int(variable_index))

        object.__setattr__(self, "variable_indices", variable_indices)
        object.__setattr__(self, "target", int(self.target))

    def value(self, config: npt.ArrayLike) -> int:
        arr = self._as_config(config)
        return int(np.sum(arr[self.variable_indices]) % 2)
