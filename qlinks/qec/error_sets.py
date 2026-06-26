from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any

import numpy as np
import numpy.typing as npt

from qlinks.models.base import ModelBuildResult
from qlinks.operators import (
    LocalOperator,
    LocalSquareValueDiagonalOperator,
    LocalValueDiagonalOperator,
    OperatorAction,
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
class LocalOperatorProduct:
    """Sequential product of configuration-space local operators.

    The product acts right-to-left, matching matrix multiplication:
    ``LocalOperatorProduct((A, B))`` represents ``A @ B``.  It is mainly used
    to generate small-weight local error algebras from single-variable factors.
    """

    terms: tuple[LocalOperator, ...]
    name: str = "local_operator_product"

    @classmethod
    def from_terms(
        cls,
        terms: Sequence[LocalOperator],
        *,
        name: str = "local_operator_product",
    ) -> LocalOperatorProduct:
        if len(terms) == 0:
            raise ValueError("LocalOperatorProduct needs at least one term.")
        return cls(terms=tuple(terms), name=name)

    @property
    def layout(self):
        return self.terms[0].layout

    def affected_variables(self) -> npt.NDArray[np.int64]:
        affected: set[int] = set()
        for term in self.terms:
            affected.update(int(i) for i in term.affected_variables())
        return np.asarray(sorted(affected), dtype=np.int64)

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]:
        partials: list[tuple[complex, npt.NDArray[np.int64]]] = [
            (1.0 + 0.0j, np.asarray(config, dtype=np.int64))
        ]

        for term in reversed(self.terms):
            next_partials: list[tuple[complex, npt.NDArray[np.int64]]] = []
            for coefficient, current_config in partials:
                for action in term.apply(current_config):
                    new_coefficient = coefficient * action.coefficient
                    if new_coefficient != 0:
                        next_partials.append((new_coefficient, action.config))
            partials = next_partials
            if not partials:
                break

        return tuple(
            OperatorAction(coefficient=coefficient, config=action_config)
            for coefficient, action_config in partials
        )


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
        max_errors: int | None = None,
        name: str | None = None,
    ) -> LocalErrorSet:
        """Generate a generic local error set up to a variable weight.

        For ``max_weight=1`` this returns the same single-variable operators as
        before.  For larger weights, it forms products of one elementary
        single-variable operator on each chosen variable subset.  This gives a
        simple model-agnostic local algebra useful for KL/code-distance scans.
        Use ``max_errors`` to truncate large exploratory algebras in notebooks.
        """
        if max_weight < 1:
            raise ValueError("max_weight must be at least 1.")

        if variable_indices is None:
            indices = tuple(range(layout.n_variables))
        else:
            indices = tuple(int(i) for i in variable_indices)

        if max_weight > len(indices):
            raise ValueError(
                f"max_weight={max_weight} exceeds the number of selected variables "
                f"({len(indices)})."
            )

        per_variable = _single_variable_error_operators(
            layout,
            indices,
            include_value_diagonal=include_value_diagonal,
            include_square_diagonal=include_square_diagonal,
            include_projectors=include_projectors,
            include_transitions=include_transitions,
            include_identity_transitions=include_identity_transitions,
        )

        errors: list[ErrorOperator] = []
        limit = None if max_errors is None else int(max_errors)
        if limit is not None and limit < 1:
            raise ValueError("max_errors must be positive when provided.")

        for weight in range(1, max_weight + 1):
            for subset in combinations(indices, weight):
                factor_lists = [per_variable[variable_index] for variable_index in subset]
                for factors in product(*factor_lists):
                    if weight == 1:
                        operator = factors[0]
                        operator_name = operator.name
                    else:
                        operator_name = "prod_" + "__".join(factor.name for factor in factors)
                        operator = LocalOperatorProduct.from_terms(factors, name=operator_name)

                    errors.append(
                        ErrorOperator.from_operator(
                            operator,
                            name=operator_name,
                            kind=f"weight_{weight}",
                            support_variables=subset,
                        )
                    )
                    if limit is not None and len(errors) >= limit:
                        set_name = name or f"layout_errors_w{max_weight}_truncated"
                        return cls(errors=tuple(errors), name=set_name)

        set_name = name or f"layout_errors_w{max_weight}"
        return cls(errors=tuple(errors), name=set_name)

    def by_max_weight(self, max_weight: int) -> LocalErrorSet:
        """Return the subset with support size at most ``max_weight``."""
        return LocalErrorSet(
            errors=tuple(error for error in self.errors if error.weight <= max_weight),
            name=f"{self.name}_w{int(max_weight)}",
        )

    def by_exact_weight(self, weight: int) -> LocalErrorSet:
        """Return the subset with support size exactly ``weight``."""
        return LocalErrorSet(
            errors=tuple(error for error in self.errors if error.weight == int(weight)),
            name=f"{self.name}_exact_w{int(weight)}",
        )

    @property
    def max_weight(self) -> int:
        """Largest declared support weight in the set."""
        return max((error.weight for error in self.errors), default=0)

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


def _single_variable_error_operators(
    layout: VariableLayout,
    indices: Sequence[int],
    *,
    include_value_diagonal: bool,
    include_square_diagonal: bool,
    include_projectors: bool,
    include_transitions: bool,
    include_identity_transitions: bool,
) -> dict[int, tuple[LocalOperator, ...]]:
    per_variable: dict[int, tuple[LocalOperator, ...]] = {}

    for variable_index in indices:
        layout.spec(variable_index)
        operators: list[LocalOperator] = []

        if include_value_diagonal:
            operators.append(
                LocalValueDiagonalOperator(
                    layout=layout,
                    variable_index=variable_index,
                    name=f"value_{variable_index}",
                )
            )

        if include_square_diagonal:
            operators.append(
                LocalSquareValueDiagonalOperator(
                    layout=layout,
                    variable_index=variable_index,
                    name=f"square_value_{variable_index}",
                )
            )

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

        if not operators:
            raise ValueError(
                f"No elementary operators were requested for variable {variable_index}."
            )

        per_variable[int(variable_index)] = tuple(operators)

    return per_variable


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
