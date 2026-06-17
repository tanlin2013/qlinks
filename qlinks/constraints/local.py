from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qlinks.constraints.base import (
    BaseConstraint,
    ConstraintPropagation,
    ConstraintResult,
)
from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class BoundedLocalCountConstraint(BaseConstraint):
    """Binary local count constraint with incremental propagation.

    The constraint enforces

        min_count <= sum(config[variable_indices]) <= max_count

    where ``min_count=None`` means there is no lower bound.  All participating
    variables must have binary local space ``{0, 1}``.  Besides the usual
    partial feasibility check, the constraint can force remaining variables:

    * if the current count already reaches ``max_count``, every unassigned
      variable in the support must be 0;
    * if the lower bound can only be reached by occupying all remaining
      variables, every unassigned variable in the support must be 1.
    """

    layout: VariableLayout
    variable_indices: npt.NDArray[np.int64]
    min_count: int | None
    max_count: int
    name: str = "bounded_local_count"

    def __post_init__(self) -> None:
        variable_indices = np.asarray(self.variable_indices, dtype=np.int64)

        if variable_indices.ndim != 1:
            raise ValueError("variable_indices must be one-dimensional.")
        if variable_indices.size == 0:
            raise ValueError("At least one variable is required.")
        if np.any(variable_indices < 0) or np.any(variable_indices >= self.layout.n_variables):
            raise ValueError("variable_indices contains indices outside the layout.")

        _, first_indices = np.unique(variable_indices, return_index=True)
        variable_indices = variable_indices[np.sort(first_indices)].astype(np.int64, copy=False)

        for variable_index in variable_indices:
            local_values = self.layout.local_space(int(variable_index)).values
            values = np.asarray(local_values, dtype=np.int64)
            if not np.array_equal(values, np.array([0, 1], dtype=np.int64)):
                raise ValueError(
                    "BoundedLocalCountConstraint requires binary local spaces with values [0, 1]."
                )

        min_count = None if self.min_count is None else int(self.min_count)
        max_count = int(self.max_count)

        if min_count is not None and min_count < 0:
            raise ValueError("min_count must be non-negative or None.")
        if max_count < 0:
            raise ValueError("max_count must be non-negative.")
        if min_count is not None and min_count > max_count:
            raise ValueError("min_count cannot exceed max_count.")
        if min_count is not None and min_count > variable_indices.size:
            raise ValueError("min_count cannot exceed the number of variables.")

        object.__setattr__(self, "variable_indices", variable_indices)
        object.__setattr__(self, "min_count", min_count)
        object.__setattr__(self, "max_count", max_count)

    @classmethod
    def exact(
        cls,
        *,
        layout: VariableLayout,
        variable_indices: npt.ArrayLike,
        count: int,
        name: str = "local_count",
    ) -> BoundedLocalCountConstraint:
        return cls(
            layout=layout,
            variable_indices=np.asarray(variable_indices, dtype=np.int64),
            min_count=int(count),
            max_count=int(count),
            name=name,
        )

    @classmethod
    def at_most(
        cls,
        *,
        layout: VariableLayout,
        variable_indices: npt.ArrayLike,
        max_count: int,
        name: str = "local_count",
    ) -> BoundedLocalCountConstraint:
        return cls(
            layout=layout,
            variable_indices=np.asarray(variable_indices, dtype=np.int64),
            min_count=None,
            max_count=int(max_count),
            name=name,
        )

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self.variable_indices.copy()

    def value(self, config: npt.ArrayLike) -> int:
        arr = self._as_config(config)
        return int(np.sum(arr[self.variable_indices]))

    def check(self, config: npt.ArrayLike) -> ConstraintResult:
        occupied = self.value(config)
        satisfied = occupied <= self.max_count and (
            self.min_count is None or occupied >= self.min_count
        )
        if self.min_count is None:
            rule = f"count<={self.max_count}"
        elif self.min_count == self.max_count:
            rule = f"count={self.min_count}"
        else:
            rule = f"{self.min_count}<=count<={self.max_count}"
        return ConstraintResult(
            satisfied=satisfied,
            name=self.name,
            residual=occupied,
            message=f"{self.name}: count={occupied}, rule={rule}",
        )

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

        variable_indices = self.variable_indices
        assigned_local = assigned[variable_indices]
        unassigned_variables = variable_indices[~assigned_local]

        occupied = int(np.sum(arr[variable_indices[assigned_local]]))
        unassigned = int(unassigned_variables.size)

        if occupied > self.max_count:
            return ConstraintPropagation.contradiction()

        if self.min_count is not None and occupied + unassigned < self.min_count:
            return ConstraintPropagation.contradiction()

        if unassigned == 0:
            if self.min_count is not None and occupied < self.min_count:
                return ConstraintPropagation.contradiction()
            return ConstraintPropagation.no_change()

        forced: list[tuple[int, int]] = []

        if occupied == self.max_count:
            forced.extend((int(variable_index), 0) for variable_index in unassigned_variables)

        if self.min_count is not None and occupied + unassigned == self.min_count:
            forced.extend((int(variable_index), 1) for variable_index in unassigned_variables)

        if not forced:
            return ConstraintPropagation.no_change()

        # The two forcing rules can conflict only if the bounds are inconsistent
        # with the current partial assignment; report that as a contradiction.
        forced_by_variable: dict[int, int] = {}
        for variable_index, value in forced:
            previous = forced_by_variable.get(variable_index)
            if previous is not None and previous != value:
                return ConstraintPropagation.contradiction()
            forced_by_variable[variable_index] = value

        return ConstraintPropagation(forced_assignments=tuple(sorted(forced_by_variable.items())))


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
    def single(
        cls, layout: VariableLayout, variable_index: int, value: int
    ) -> FixedValueConstraint:
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

        forced: list[tuple[int, int]] = []
        for variable_index, value in zip(self.variable_indices, self.values, strict=True):
            variable_index = int(variable_index)
            value = int(value)
            if assigned[variable_index]:
                if int(arr[variable_index]) != value:
                    return ConstraintPropagation.contradiction()
            else:
                forced.append((variable_index, value))

        if not forced:
            return ConstraintPropagation.no_change()

        return ConstraintPropagation(forced_assignments=tuple(forced))


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
