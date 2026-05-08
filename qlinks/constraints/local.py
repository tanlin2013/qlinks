from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qlinks.constraints.base import BaseConstraint, ConstraintResult
from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class FixedValueConstraint(BaseConstraint):
    """
    Require selected variables to take fixed values.

    Useful for boundary conditions, pinned charges, frozen links, etc.
    """

    layout: VariableLayout
    variable_indices: npt.NDArray[np.int64]
    values: npt.NDArray[np.int64]
    name: str = "fixed_value"

    def __post_init__(self) -> None:
        variable_indices = np.asarray(self.variable_indices, dtype=np.int64)
        values = np.asarray(self.values, dtype=np.int64)

        if variable_indices.ndim != 1:
            raise ValueError("variable_indices must be one-dimensional.")
        if values.ndim != 1:
            raise ValueError("values must be one-dimensional.")
        if variable_indices.size != values.size:
            raise ValueError("variable_indices and values must have the same length.")
        if variable_indices.size == 0:
            raise ValueError("At least one fixed variable is required.")

        for variable_index, value in zip(variable_indices, values, strict=True):
            self.layout.local_space(int(variable_index)).validate_value(int(value))

        object.__setattr__(self, "variable_indices", variable_indices)
        object.__setattr__(self, "values", values)

    @classmethod
    def single(cls, layout: VariableLayout, variable_index: int, value: int) -> FixedValueConstraint:
        return cls(
            layout=layout,
            variable_indices=np.asarray([variable_index], dtype=np.int64),
            values=np.asarray([value], dtype=np.int64),
        )

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self.variable_indices.copy()

    def check(self, config: npt.ArrayLike) -> ConstraintResult:
        arr = self._as_config(config)

        actual = arr[self.variable_indices]
        satisfied = bool(np.array_equal(actual, self.values))

        return ConstraintResult(
            satisfied=satisfied,
            name=self.name,
            residual=actual.copy(),
            message=f"{self.name}: actual={actual.tolist()}, target={self.values.tolist()}",
        )


@dataclass(frozen=True, slots=True)
class LocalSumConstraint(BaseConstraint):
    """
    Require a signed sum over selected variables to equal a target.

        sum_i coefficients[i] * config[variable_indices[i]] == target

    This is a generic building block for simple local constraints.
    """

    layout: VariableLayout
    variable_indices: npt.NDArray[np.int64]
    coefficients: npt.NDArray[np.int64]
    target: int
    name: str = "local_sum"

    def __post_init__(self) -> None:
        variable_indices = np.asarray(self.variable_indices, dtype=np.int64)
        coefficients = np.asarray(self.coefficients, dtype=np.int64)

        if variable_indices.ndim != 1:
            raise ValueError("variable_indices must be one-dimensional.")
        if coefficients.ndim != 1:
            raise ValueError("coefficients must be one-dimensional.")
        if variable_indices.size != coefficients.size:
            raise ValueError("variable_indices and coefficients must have the same length.")
        if variable_indices.size == 0:
            raise ValueError("At least one variable is required.")

        for variable_index in variable_indices:
            self.layout.spec(int(variable_index))

        object.__setattr__(self, "variable_indices", variable_indices)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "target", int(self.target))

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self.variable_indices.copy()

    def value(self, config: npt.ArrayLike) -> int:
        arr = self._as_config(config)
        return int(np.dot(self.coefficients, arr[self.variable_indices]))

    def check(self, config: npt.ArrayLike) -> ConstraintResult:
        actual = self.value(config)
        satisfied = actual == self.target

        return ConstraintResult(
            satisfied=satisfied,
            name=self.name,
            residual=actual,
            message=f"{self.name}: value={actual}, target={self.target}",
        )
