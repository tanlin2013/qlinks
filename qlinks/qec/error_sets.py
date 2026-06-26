from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from qlinks.models.base import ModelBuildResult
from qlinks.operators import (
    LocalOperator,
    LocalSquareValueDiagonalOperator,
    LocalValueDiagonalOperator,
    OperatorSum,
    PatternDiagonalOperator,
    SetVariablesOperator,
)
from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class ErrorOperator:
    """Named physical error operator used by QEC diagnostics."""

    name: str
    operator: Any
    support_variables: tuple[int, ...] = ()
    kind: str = "custom"

    @classmethod
    def from_operator(
        cls,
        operator: Any,
        *,
        name: str | None = None,
        kind: str = "custom",
        support_variables: Sequence[int] | None = None,
    ) -> ErrorOperator:
        """Create metadata from a local operator or matrix-like object."""
        if name is None:
            name = str(getattr(operator, "name", type(operator).__name__))

        if support_variables is None:
            affected_variables = getattr(operator, "affected_variables", None)
            if callable(affected_variables):
                support_variables = tuple(int(i) for i in affected_variables())
            else:
                support_variables = ()

        return cls(
            name=name,
            operator=operator,
            support_variables=tuple(int(i) for i in support_variables),
            kind=kind,
        )

    @property
    def weight(self) -> int:
        """Number of distinct variables in the declared support."""
        return len(set(self.support_variables))


@dataclass(frozen=True, slots=True)
class LocalErrorSet:
    """Collection of physical error operators with stable names and supports."""

    errors: tuple[ErrorOperator, ...]
    name: str = "local_errors"

    def __len__(self) -> int:
        return len(self.errors)

    def __iter__(self):
        return iter(self.errors)

    def __getitem__(self, index: int) -> ErrorOperator:
        return self.errors[index]

    @classmethod
    def from_operators(
        cls,
        operators: Sequence[Any],
        *,
        names: Sequence[str] = (),
        kind: str = "custom",
        name: str = "local_errors",
    ) -> LocalErrorSet:
        """Wrap arbitrary local operators or matrix-like objects as an error set."""
        if names and len(names) != len(operators):
            raise ValueError("names must be empty or have one entry per operator.")

        errors = []
        for index, operator in enumerate(operators):
            operator_name = names[index] if names else None
            errors.append(
                ErrorOperator.from_operator(
                    operator,
                    name=operator_name,
                    kind=kind,
                )
            )

        return cls(
            errors=tuple(errors),
            name=name,
        )

    @classmethod
    def from_model(
        cls,
        model: Any | None = None,
        *,
        build_result: ModelBuildResult | None = None,
        builder: str = "sparse",
        name: str = "model_local_terms",
    ) -> LocalErrorSet:
        """Use a model's local Hamiltonian terms as a first physical error set.

        This is a lightweight bridge from existing model APIs.  For a more
        exhaustive noise algebra, use :meth:`from_layout`.
        """
        if build_result is not None:
            operators = build_result.operators
        else:
            if model is None:
                raise ValueError("Either model or build_result is required.")

            layout = model.layout
            make_operators = getattr(model, "make_operators", None)
            if callable(make_operators):
                operators = make_operators(layout, builder=builder)
            else:
                terms = model.make_terms(layout, builder=builder)
                operators = tuple(operator for term in terms for operator in term.operators)

        return cls.from_operators(
            operators,
            kind="model_term",
            name=name,
        )

    @classmethod
    def from_layout(
        cls,
        layout: VariableLayout,
        *,
        variable_indices: Sequence[int] | None = None,
        max_weight: int = 1,
        include_value_diagonal: bool = True,
        include_square_diagonal: bool = False,
        include_projectors: bool = False,
        include_transitions: bool = True,
        include_identity_transitions: bool = False,
        name: str = "single_variable_errors",
    ) -> LocalErrorSet:
        """Generate a generic single-variable local error set.

        The current implementation intentionally supports ``max_weight=1``.
        Higher-weight algebras can be assembled later from tensor products or
        model-specific constrained local operators.
        """
        if max_weight != 1:
            raise NotImplementedError("LocalErrorSet.from_layout currently supports max_weight=1.")

        if variable_indices is None:
            indices = tuple(range(layout.n_variables))
        else:
            indices = tuple(int(i) for i in variable_indices)

        operators: list[LocalOperator] = []
        names: list[str] = []

        for variable_index in indices:
            layout.spec(variable_index)

            if include_value_diagonal:
                operators.append(
                    LocalValueDiagonalOperator(
                        layout=layout,
                        variable_index=variable_index,
                        name=f"value_{variable_index}",
                    )
                )
                names.append(f"value_{variable_index}")

            if include_square_diagonal:
                operators.append(
                    LocalSquareValueDiagonalOperator(
                        layout=layout,
                        variable_index=variable_index,
                        name=f"square_value_{variable_index}",
                    )
                )
                names.append(f"square_value_{variable_index}")

            values = [int(value) for value in layout.local_space(variable_index).values]

            if include_projectors:
                for value in values:
                    operators.append(
                        PatternDiagonalOperator(
                            layout=layout,
                            variable_indices=np.asarray([variable_index], dtype=np.int64),
                            pattern=np.asarray([value], dtype=np.int64),
                            name=f"projector_{variable_index}_{value}",
                        )
                    )
                    names.append(f"projector_{variable_index}_{value}")

            if include_transitions:
                for initial in values:
                    for final in values:
                        if initial == final and not include_identity_transitions:
                            continue
                        operators.append(
                            SetVariablesOperator(
                                layout=layout,
                                variable_indices=np.asarray([variable_index], dtype=np.int64),
                                initial_values=np.asarray([initial], dtype=np.int64),
                                final_values=np.asarray([final], dtype=np.int64),
                                name=f"transition_{variable_index}_{initial}_to_{final}",
                            )
                        )
                        names.append(f"transition_{variable_index}_{initial}_to_{final}")

        return cls.from_operators(
            operators,
            names=names,
            kind="single_variable",
            name=name,
        )

    def by_max_weight(self, max_weight: int) -> LocalErrorSet:
        """Return the subset with support size at most ``max_weight``."""
        return LocalErrorSet(
            errors=tuple(error for error in self.errors if error.weight <= max_weight),
            name=f"{self.name}_w{int(max_weight)}",
        )

    def with_identity(self, operator: Any, *, name: str = "identity") -> LocalErrorSet:
        """Return a copy with an explicit identity-like operator prepended."""
        return LocalErrorSet(
            errors=(
                ErrorOperator.from_operator(
                    operator,
                    name=name,
                    kind="identity",
                ),
                *self.errors,
            ),
            name=self.name,
        )


def combine_error_operators(
    name: str,
    operators: Sequence[LocalOperator],
    *,
    kind: str = "operator_sum",
) -> ErrorOperator:
    """Build one error entry from a formal sum of local operators."""
    return ErrorOperator.from_operator(
        OperatorSum.from_terms(operators, name=name),
        name=name,
        kind=kind,
    )
