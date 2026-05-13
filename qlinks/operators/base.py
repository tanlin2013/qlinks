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

    def affected_variables(self) -> npt.NDArray[np.int64]: ...

    def apply(self, config: npt.ArrayLike) -> tuple[OperatorAction, ...]: ...


class BaseLocalOperator:
    """
    Convenience base class for configuration-space local operators.

    Design rule
    -----------
    Concrete operators should precompute all geometry/layout-dependent data in
    __post_init__(), so apply() only performs cheap indexed array operations.
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

    # ------------------------------------------------------------------
    # Precomputation helpers for subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _int_array(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
        arr = np.asarray(values, dtype=np.int64)

        if arr.ndim != 1:
            raise ValueError("Expected a one-dimensional integer array.")

        return arr

    @staticmethod
    def _deduplicate_int_array(
        values: npt.ArrayLike,
    ) -> npt.NDArray[np.int64]:
        arr = BaseLocalOperator._int_array(values)

        if arr.size == 0:
            return arr

        _, first_indices = np.unique(arr, return_index=True)
        return arr[np.sort(first_indices)].astype(np.int64, copy=False)

    @staticmethod
    def _cached_array(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """
        Return a private immutable-ish int64 array for storing on frozen operators.

        We return a normal ndarray because NumPy arrays are mutable, but all
        public accessors should return copies.
        """
        return np.asarray(values, dtype=np.int64).reshape(-1)

    def _site_variable_index(self, site_id: int) -> int:
        return int(self.layout.site_variable_index(int(site_id)))

    def _link_variable_index(self, link_id: int) -> int:
        return int(self.layout.link_variable_index(int(link_id)))

    def _site_variable_indices(
        self,
        site_ids: npt.ArrayLike,
    ) -> npt.NDArray[np.int64]:
        site_ids_arr = self._int_array(site_ids)

        return np.asarray(
            [self._site_variable_index(int(site_id)) for site_id in site_ids_arr],
            dtype=np.int64,
        )

    def _link_variable_indices(
        self,
        link_ids: npt.ArrayLike,
    ) -> npt.NDArray[np.int64]:
        link_ids_arr = self._int_array(link_ids)

        return np.asarray(
            [self._link_variable_index(int(link_id)) for link_id in link_ids_arr],
            dtype=np.int64,
        )

    def _validate_local_space_values(
        self,
        variable_index: int,
        expected_values: set[int],
        *,
        operator_name: str | None = None,
    ) -> None:
        actual_values = set(
            int(v)
            for v in self.layout.local_space(int(variable_index)).values.tolist()
        )

        if actual_values != set(expected_values):
            name = operator_name or getattr(self, "name", type(self).__name__)
            expected = sorted(int(v) for v in expected_values)
            actual = sorted(int(v) for v in actual_values)

            raise ValueError(
                f"{name} requires local-space values {expected}, "
                f"got {actual} at variable {int(variable_index)}."
            )

    def _validate_local_spaces(
        self,
        variable_indices: npt.ArrayLike,
        expected_values: set[int],
        *,
        operator_name: str | None = None,
    ) -> None:
        for variable_index in self._int_array(variable_indices):
            self._validate_local_space_values(
                int(variable_index),
                expected_values,
                operator_name=operator_name,
            )

    @staticmethod
    def _copy_indices(values: npt.ArrayLike) -> npt.NDArray[np.int64]:
        return np.asarray(values, dtype=np.int64).copy()


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
