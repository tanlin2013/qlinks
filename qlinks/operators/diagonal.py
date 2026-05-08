from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qlinks.operators.base import BaseLocalOperator, OperatorAction
from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class ConstantDiagonalOperator(BaseLocalOperator):
    """
    Diagonal operator c * I.
    """

    layout: VariableLayout
    coefficient: complex
    name: str = "constant_diagonal"

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]:
        arr = self._as_config(config)
        return (OperatorAction(self.coefficient, arr.copy()),)


@dataclass(frozen=True, slots=True)
class LocalValueDiagonalOperator(BaseLocalOperator):
    """
    Diagonal operator proportional to one variable value.

        coefficient * config[variable_index]
    """

    layout: VariableLayout
    variable_index: int
    coefficient: complex = 1.0
    name: str = "local_value_diagonal"

    def __post_init__(self) -> None:
        self.layout.spec(self.variable_index)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return np.asarray([self.variable_index], dtype=np.int64)

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]:
        arr = self._as_config(config)
        value = int(arr[self.variable_index])
        coeff = complex(self.coefficient) * value
        return (OperatorAction(coeff, arr.copy()),)


@dataclass(frozen=True, slots=True)
class LocalSumDiagonalOperator(BaseLocalOperator):
    """
    Diagonal operator

        coefficient * sum_i weights[i] * config[variable_indices[i]]
    """

    layout: VariableLayout
    variable_indices: npt.NDArray[np.int64]
    weights: npt.NDArray[np.int64] | None = None
    coefficient: complex = 1.0
    name: str = "local_sum_diagonal"

    def __post_init__(self) -> None:
        variable_indices = np.asarray(self.variable_indices, dtype=np.int64)

        if variable_indices.ndim != 1:
            raise ValueError("variable_indices must be one-dimensional.")

        if variable_indices.size == 0:
            raise ValueError("At least one variable is required.")

        for variable_index in variable_indices:
            self.layout.spec(int(variable_index))

        if self.weights is None:
            weights = np.ones(variable_indices.size, dtype=np.int64)
        else:
            weights = np.asarray(self.weights, dtype=np.int64)

        if weights.ndim != 1:
            raise ValueError("weights must be one-dimensional.")

        if weights.size != variable_indices.size:
            raise ValueError("weights and variable_indices must have the same length.")

        object.__setattr__(self, "variable_indices", variable_indices)
        object.__setattr__(self, "weights", weights)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self.variable_indices.copy()

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]:
        arr = self._as_config(config)
        value = int(np.dot(self.weights, arr[self.variable_indices]))
        coeff = complex(self.coefficient) * value
        return (OperatorAction(coeff, arr.copy()),)


@dataclass(frozen=True, slots=True)
class PatternDiagonalOperator(BaseLocalOperator):
    """
    Diagonal projector-like operator.

    If selected variables match `pattern`, return coefficient * config.
    Otherwise return no action.

    Example:
        A flippability counter for a plaquette can be represented as a sum of
        pattern diagonal operators.
    """

    layout: VariableLayout
    variable_indices: npt.NDArray[np.int64]
    pattern: npt.NDArray[np.int64]
    coefficient: complex = 1.0
    name: str = "pattern_diagonal"

    def __post_init__(self) -> None:
        variable_indices = np.asarray(self.variable_indices, dtype=np.int64)
        pattern = np.asarray(self.pattern, dtype=np.int64)

        if variable_indices.ndim != 1:
            raise ValueError("variable_indices must be one-dimensional.")

        if pattern.ndim != 1:
            raise ValueError("pattern must be one-dimensional.")

        if variable_indices.size != pattern.size:
            raise ValueError("variable_indices and pattern must have the same length.")

        if variable_indices.size == 0:
            raise ValueError("At least one variable is required.")

        for variable_index, value in zip(variable_indices, pattern, strict=True):
            self.layout.local_space(int(variable_index)).validate_value(int(value))

        object.__setattr__(self, "variable_indices", variable_indices)
        object.__setattr__(self, "pattern", pattern)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self.variable_indices.copy()

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]:
        arr = self._as_config(config)

        if np.array_equal(arr[self.variable_indices], self.pattern):
            return (OperatorAction(self.coefficient, arr.copy()),)

        return ()
