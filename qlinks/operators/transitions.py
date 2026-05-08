from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qlinks.operators.base import BaseLocalOperator, OperatorAction
from qlinks.variables import VariableKind, VariableLayout


@dataclass(frozen=True, slots=True)
class SetVariablesOperator(BaseLocalOperator):
    """
    General local transition operator.

    If

        config[variable_indices] == initial_values

    then produce a new config with

        new_config[variable_indices] = final_values

    Otherwise return no action.
    """

    layout: VariableLayout
    variable_indices: npt.NDArray[np.int64]
    initial_values: npt.NDArray[np.int64]
    final_values: npt.NDArray[np.int64]
    coefficient: complex = 1.0
    name: str = "set_variables"

    def __post_init__(self) -> None:
        variable_indices = np.asarray(self.variable_indices, dtype=np.int64)
        initial_values = np.asarray(self.initial_values, dtype=np.int64)
        final_values = np.asarray(self.final_values, dtype=np.int64)

        if variable_indices.ndim != 1:
            raise ValueError("variable_indices must be one-dimensional.")

        if initial_values.ndim != 1:
            raise ValueError("initial_values must be one-dimensional.")

        if final_values.ndim != 1:
            raise ValueError("final_values must be one-dimensional.")

        if not (
            variable_indices.size == initial_values.size == final_values.size
        ):
            raise ValueError(
                "variable_indices, initial_values, and final_values must have the same length."
            )

        if variable_indices.size == 0:
            raise ValueError("At least one variable is required.")

        for variable_index, initial, final in zip(
            variable_indices,
            initial_values,
            final_values,
            strict=True,
        ):
            local_space = self.layout.local_space(int(variable_index))
            local_space.validate_value(int(initial))
            local_space.validate_value(int(final))

        object.__setattr__(self, "variable_indices", variable_indices)
        object.__setattr__(self, "initial_values", initial_values)
        object.__setattr__(self, "final_values", final_values)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self.variable_indices.copy()

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]:
        arr = self._as_config(config)

        if not np.array_equal(arr[self.variable_indices], self.initial_values):
            return ()

        new = arr.copy()
        new[self.variable_indices] = self.final_values

        return (OperatorAction(self.coefficient, new),)


@dataclass(frozen=True, slots=True)
class BinaryFlipOperator(BaseLocalOperator):
    """
    Flip one binary variable 0 <-> 1.
    """

    layout: VariableLayout
    variable_index: int
    coefficient: complex = 1.0
    name: str = "binary_flip"

    def __post_init__(self) -> None:
        values = set(self.layout.local_space(self.variable_index).values.tolist())
        if values != {0, 1}:
            raise ValueError("BinaryFlipOperator requires local values {0, 1}.")

    @classmethod
    def on_site(
        cls,
        layout: VariableLayout,
        site_id: int,
        coefficient: complex = 1.0,
    ) -> BinaryFlipOperator:
        return cls(
            layout=layout,
            variable_index=layout.variable_index(VariableKind.SITE, site_id),
            coefficient=coefficient,
        )

    @classmethod
    def on_link(
        cls,
        layout: VariableLayout,
        link_id: int,
        coefficient: complex = 1.0,
    ) -> BinaryFlipOperator:
        return cls(
            layout=layout,
            variable_index=layout.variable_index(VariableKind.LINK, link_id),
            coefficient=coefficient,
        )

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return np.asarray([self.variable_index], dtype=np.int64)

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]:
        arr = self._as_config(config)

        new = arr.copy()
        new[self.variable_index] = 1 - new[self.variable_index]

        return (OperatorAction(self.coefficient, new),)


@dataclass(frozen=True, slots=True)
class NegationFlipOperator(BaseLocalOperator):
    """
    Flip one sign variable v -> -v.

    Useful for internal integer representation of spin-1/2 flux variables
    with values {-1, +1}.
    """

    layout: VariableLayout
    variable_index: int
    coefficient: complex = 1.0
    name: str = "negation_flip"

    def __post_init__(self) -> None:
        values = set(self.layout.local_space(self.variable_index).values.tolist())
        for value in values:
            if -value not in values:
                raise ValueError(
                    "NegationFlipOperator requires a local space closed under v -> -v."
                )

    @classmethod
    def on_link(
        cls,
        layout: VariableLayout,
        link_id: int,
        coefficient: complex = 1.0,
    ) -> NegationFlipOperator:
        return cls(
            layout=layout,
            variable_index=layout.variable_index(VariableKind.LINK, link_id),
            coefficient=coefficient,
        )

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return np.asarray([self.variable_index], dtype=np.int64)

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]:
        arr = self._as_config(config)

        new = arr.copy()
        new[self.variable_index] = -new[self.variable_index]

        return (OperatorAction(self.coefficient, new),)


@dataclass(frozen=True, slots=True)
class MultiNegationFlipOperator(BaseLocalOperator):
    """
    Simultaneously apply v -> -v to several variables.

    Useful as a simple QLM-like plaquette flip when the link variables are
    represented as {-1, +1}.
    """

    layout: VariableLayout
    variable_indices: npt.NDArray[np.int64]
    coefficient: complex = 1.0
    name: str = "multi_negation_flip"

    def __post_init__(self) -> None:
        variable_indices = np.asarray(self.variable_indices, dtype=np.int64)

        if variable_indices.ndim != 1:
            raise ValueError("variable_indices must be one-dimensional.")

        if variable_indices.size == 0:
            raise ValueError("At least one variable is required.")

        for variable_index in variable_indices:
            values = set(self.layout.local_space(int(variable_index)).values.tolist())
            for value in values:
                if -value not in values:
                    raise ValueError(
                        "MultiNegationFlipOperator requires local spaces closed under v -> -v."
                    )

        object.__setattr__(self, "variable_indices", variable_indices)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return self.variable_indices.copy()

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]:
        arr = self._as_config(config)

        new = arr.copy()
        new[self.variable_indices] = -new[self.variable_indices]

        return (OperatorAction(self.coefficient, new),)
    