from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.variables import VariableLayout


@dataclass(frozen=True, slots=True)
class OperatorAction:
    """
    One operator action on one computational configuration.

    coefficient:
        Matrix element contributed by this action.

    config:
        Resulting configuration after the operator acts.
        For diagonal operators, this is usually a copy of the input config.
    """

    coefficient: complex
    config: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        arr = np.asarray(self.config, dtype=np.int64)

        if arr.ndim != 1:
            raise ValueError("OperatorAction.config must be one-dimensional.")

        object.__setattr__(self, "coefficient", complex(self.coefficient))
        object.__setattr__(self, "config", arr)


class LocalOperator(Protocol):
    """
    Interface for all configuration-space operators.
    """

    layout: VariableLayout
    name: str

    def affected_variables(self) -> npt.NDArray[np.int64]:
        ...

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]:
        ...


class BaseLocalOperator:
    """
    Convenience base class.

    This is intentionally not a dataclass, to avoid dataclass-inheritance issues.
    """

    layout: VariableLayout
    name: str

    def _as_config(
        self,
        config: npt.ArrayLike,
        *,
        validate: bool = True,
    ) -> npt.NDArray[np.int64]:
        arr = np.asarray(config, dtype=np.int64)

        if validate:
            self.layout.validate_config(arr)
        elif arr.shape != self.layout.shape:
            raise ValueError(f"Expected config shape {self.layout.shape}, got {arr.shape}.")

        return arr

    def affected_variables(self) -> npt.NDArray[np.int64]:
        return np.arange(self.layout.n_variables, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class OperatorSum:
    """
    Formal sum of local operators.

    This is still only an action-level object. Sparse matrix construction is
    handled by the next layer.
    """

    terms: tuple[LocalOperator, ...]
    name: str = "operator_sum"

    @classmethod
    def from_terms(cls, terms: Sequence[LocalOperator], name: str = "operator_sum") -> OperatorSum:
        return cls(terms=tuple(terms), name=name)

    def affected_variables(self) -> npt.NDArray[np.int64]:
        affected: set[int] = set()

        for term in self.terms:
            affected.update(int(i) for i in term.affected_variables())

        return np.asarray(sorted(affected), dtype=np.int64)

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]:
        actions: list[OperatorAction] = []

        for term in self.terms:
            actions.extend(term.apply(config))

        return tuple(actions)


def combine_duplicate_actions(
    actions: Sequence[OperatorAction],
    *,
    atol: float = 0.0,
) -> tuple[OperatorAction, ...]:
    """
    Combine actions that produce the same output configuration.

    This is useful before sparse assembly when two terms lead to the same
    row/column matrix element.
    """

    combined: dict[bytes, tuple[complex, npt.NDArray[np.int64]]] = {}

    for action in actions:
        key = np.ascontiguousarray(action.config, dtype=np.int64).tobytes()

        if key in combined:
            coeff, cfg = combined[key]
            combined[key] = (coeff + action.coefficient, cfg)
        else:
            combined[key] = (action.coefficient, action.config.copy())

    out: list[OperatorAction] = []

    for coeff, cfg in combined.values():
        if abs(coeff) > atol:
            out.append(OperatorAction(coeff, cfg))

    return tuple(out)
