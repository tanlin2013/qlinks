from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qlinks.constraints.base import BaseSectorCondition, ConstraintPropagation
from qlinks.variables import VariableLayout


def _unique_extremal_value(
    *,
    values: npt.NDArray[np.int64],
    contributions: npt.NDArray[np.int64],
    extremum: int,
) -> tuple[bool, int]:
    extremal_values = values[contributions == int(extremum)]
    unique_values = np.unique(extremal_values.astype(np.int64, copy=False))
    if unique_values.size != 1:
        return False, 0
    return True, int(unique_values[0])


def _propagate_discrete_sum(
    *,
    config: npt.ArrayLike,
    assigned_mask: npt.ArrayLike,
    variable_indices: npt.NDArray[np.int64],
    contributions: npt.NDArray[np.int64],
    target: int,
    min_contributions: npt.NDArray[np.int64],
    max_contributions: npt.NDArray[np.int64],
    min_values: npt.NDArray[np.int64],
    max_values: npt.NDArray[np.int64],
    min_value_is_unique: npt.NDArray[np.bool_],
    max_value_is_unique: npt.NDArray[np.bool_],
) -> ConstraintPropagation:
    assigned = np.asarray(assigned_mask, dtype=bool)

    assigned_local = assigned[variable_indices]
    current = int(np.sum(contributions[assigned_local]))

    unassigned_local = ~assigned_local
    unassigned_variables = variable_indices[unassigned_local]

    min_remaining = int(np.sum(min_contributions[unassigned_local]))
    max_remaining = int(np.sum(max_contributions[unassigned_local]))

    min_possible = current + min_remaining
    max_possible = current + max_remaining

    if int(target) < min_possible:
        return ConstraintPropagation.contradiction()

    if int(target) > max_possible:
        return ConstraintPropagation.contradiction()

    if unassigned_variables.size == 0:
        if current != int(target):
            return ConstraintPropagation.contradiction()
        return ConstraintPropagation.no_change()

    forced: list[tuple[int, int]] = []

    if int(target) == min_possible:
        for variable_index, value, unique in zip(
            unassigned_variables,
            min_values[unassigned_local],
            min_value_is_unique[unassigned_local],
            strict=True,
        ):
            if unique:
                forced.append((int(variable_index), int(value)))

    if int(target) == max_possible:
        for variable_index, value, unique in zip(
            unassigned_variables,
            max_values[unassigned_local],
            max_value_is_unique[unassigned_local],
            strict=True,
        ):
            if unique:
                forced.append((int(variable_index), int(value)))

    if not forced:
        return ConstraintPropagation.no_change()

    forced_by_variable: dict[int, int] = {}
    for variable_index, value in forced:
        previous = forced_by_variable.get(variable_index)
        if previous is not None and previous != value:
            return ConstraintPropagation.contradiction()
        forced_by_variable[variable_index] = value

    return ConstraintPropagation(forced_assignments=tuple(sorted(forced_by_variable.items())))


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

        min_contributions: list[int] = []
        max_contributions: list[int] = []
        min_values: list[int] = []
        max_values: list[int] = []
        min_value_is_unique: list[bool] = []
        max_value_is_unique: list[bool] = []

        for variable_index, coefficient in zip(variable_indices, coefficients, strict=True):
            local_values = np.asarray(
                self.layout.local_space(int(variable_index)).values,
                dtype=np.int64,
            )
            contribution_values = int(coefficient) * local_values
            min_contribution = int(np.min(contribution_values))
            max_contribution = int(np.max(contribution_values))
            min_unique, min_value = _unique_extremal_value(
                values=local_values,
                contributions=contribution_values,
                extremum=min_contribution,
            )
            max_unique, max_value = _unique_extremal_value(
                values=local_values,
                contributions=contribution_values,
                extremum=max_contribution,
            )

            min_contributions.append(min_contribution)
            max_contributions.append(max_contribution)
            min_values.append(min_value)
            max_values.append(max_value)
            min_value_is_unique.append(min_unique)
            max_value_is_unique.append(max_unique)

        object.__setattr__(self, "variable_indices", variable_indices)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "target", int(self.target))
        object.__setattr__(
            self, "_min_contributions", np.asarray(min_contributions, dtype=np.int64)
        )
        object.__setattr__(
            self, "_max_contributions", np.asarray(max_contributions, dtype=np.int64)
        )
        object.__setattr__(self, "_min_values", np.asarray(min_values, dtype=np.int64))
        object.__setattr__(self, "_max_values", np.asarray(max_values, dtype=np.int64))
        object.__setattr__(
            self,
            "_min_value_is_unique",
            np.asarray(min_value_is_unique, dtype=bool),
        )
        object.__setattr__(
            self,
            "_max_value_is_unique",
            np.asarray(max_value_is_unique, dtype=bool),
        )

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return np.asarray(self.variable_indices, dtype=np.int64).copy()

    def value(self, config: npt.ArrayLike) -> int:
        arr = self._as_config(config)
        return int(np.dot(self.coefficients, arr[self.variable_indices]))

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
        contributions = self.coefficients * arr[self.variable_indices]
        return _propagate_discrete_sum(
            config=config,
            assigned_mask=assigned_mask,
            variable_indices=self.variable_indices,
            contributions=contributions,
            target=int(self.target),
            min_contributions=self._min_contributions,
            max_contributions=self._max_contributions,
            min_values=self._min_values,
            max_values=self._max_values,
            min_value_is_unique=self._min_value_is_unique,
            max_value_is_unique=self._max_value_is_unique,
        )


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

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return np.asarray(self.variable_indices, dtype=np.int64).copy()

    def value(self, config: npt.ArrayLike) -> int:
        arr = self._as_config(config)
        return int(np.sum(arr[self.variable_indices]) % 2)

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

        assigned_local = assigned[self.variable_indices]
        assigned_values = arr[self.variable_indices[assigned_local]]
        current_parity = int(np.sum(assigned_values) % 2)

        unassigned_variables = self.variable_indices[~assigned_local]
        if unassigned_variables.size == 0:
            if current_parity != int(self.target):
                return ConstraintPropagation.contradiction()
            return ConstraintPropagation.no_change()

        # With one remaining variable, parity determines its parity class.  We
        # can force the variable only when exactly one local value has that
        # parity.  For binary local spaces this gives immediate pruning.
        if unassigned_variables.size == 1:
            variable_index = int(unassigned_variables[0])
            needed_parity = (int(self.target) - current_parity) % 2
            values = np.asarray(
                self.layout.local_space(variable_index).values,
                dtype=np.int64,
            )
            allowed_values = values[(values % 2) == needed_parity]
            unique_allowed_values = np.unique(allowed_values.astype(np.int64, copy=False))
            if unique_allowed_values.size == 0:
                return ConstraintPropagation.contradiction()
            if unique_allowed_values.size == 1:
                return ConstraintPropagation(
                    forced_assignments=((variable_index, int(unique_allowed_values[0])),)
                )

        return ConstraintPropagation.no_change()
