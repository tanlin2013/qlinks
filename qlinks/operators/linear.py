from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from qlinks.basis import Basis
from qlinks.operators.base import (
    LocalOperator,
    OperatorAction,
    combine_duplicate_actions,
)


@dataclass(frozen=True)
class BasisOperator:
    """
    Matrix-like wrapper around configuration-space LocalOperator objects.

    This object does not explicitly build the sparse matrix. Instead, it applies
    the local operators directly by looping over the basis states.

    Examples
    --------
    >>> O = BasisOperator(basis, operators)
    >>> y = O @ v
    >>> value = v.T @ O @ v
    >>> value_complex = v.conj() @ O @ v
    >>> y_t = O.T @ v
    >>> y_h = O.H @ v
    """

    basis: Basis
    operators: tuple[LocalOperator, ...]
    combine_duplicates: bool = True
    drop_zero_atol: float = 0.0
    dtype: npt.DTypeLike = np.complex128

    # Make NumPy defer ndarray @ BasisOperator to BasisOperator.__rmatmul__.
    __array_priority__ = 10_000

    def __post_init__(self) -> None:
        object.__setattr__(self, "operators", tuple(self.operators))

    @classmethod
    def from_operator(
        cls,
        basis: Basis,
        operator: LocalOperator,
        **kwargs,
    ) -> BasisOperator:
        return cls(
            basis=basis,
            operators=(operator,),
            **kwargs,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (self.basis.n_states, self.basis.n_states)

    @property
    def T(self) -> TransposedBasisOperator:  # noqa: N802
        return TransposedBasisOperator(
            parent=self,
            conjugate=False,
        )

    @property
    def H(self) -> TransposedBasisOperator:  # noqa: N802
        """
        Hermitian adjoint view.
        """
        return TransposedBasisOperator(
            parent=self,
            conjugate=True,
        )

    def _column_actions(
        self,
        config: npt.NDArray[np.int64],
    ) -> tuple[OperatorAction, ...]:
        actions: list[OperatorAction] = []

        for operator in self.operators:
            actions.extend(operator.apply(config))

        if self.combine_duplicates:
            return combine_duplicate_actions(
                actions,
                atol=self.drop_zero_atol,
            )

        return tuple(actions)

    def matvec(self, vector: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """
        Compute self @ vector.
        """
        x = np.asarray(vector, dtype=self.dtype)

        if x.ndim != 1:
            raise ValueError("matvec expects a one-dimensional vector.")

        if x.shape[0] != self.basis.n_states:
            raise ValueError(f"Expected vector length {self.basis.n_states}, got {x.shape[0]}.")

        y = np.zeros(self.basis.n_states, dtype=self.dtype)

        for col, config in enumerate(self.basis.iter_states(copy=False)):
            amplitude = x[col]

            if amplitude == 0:
                continue

            for action in self._column_actions(config):
                if abs(action.coefficient) <= self.drop_zero_atol:
                    continue

                row = self.basis.get_index(action.config)
                if row is None:
                    continue

                y[row] += action.coefficient * amplitude

        return y

    def matmat(self, matrix: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """
        Compute self @ matrix, where matrix has shape (n_basis, n_vecs).
        """
        x = np.asarray(matrix, dtype=self.dtype)

        if x.ndim != 2:
            raise ValueError("matmat expects a two-dimensional array.")

        if x.shape[0] != self.basis.n_states:
            raise ValueError(f"Expected matrix shape ({self.basis.n_states}, k), got {x.shape}.")

        y = np.zeros_like(x, dtype=self.dtype)

        for col, config in enumerate(self.basis.iter_states(copy=False)):
            amplitudes = x[col, :]

            if np.all(amplitudes == 0):
                continue

            for action in self._column_actions(config):
                if abs(action.coefficient) <= self.drop_zero_atol:
                    continue

                row = self.basis.get_index(action.config)
                if row is None:
                    continue

                y[row, :] += action.coefficient * amplitudes

        return y

    def rmatvec(self, vector: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """
        Compute vector @ self.
        """
        x = np.asarray(vector, dtype=self.dtype)

        if x.ndim != 1:
            raise ValueError("rmatvec expects a one-dimensional vector.")

        if x.shape[0] != self.basis.n_states:
            raise ValueError(f"Expected vector length {self.basis.n_states}, got {x.shape[0]}.")

        y = np.zeros(self.basis.n_states, dtype=self.dtype)

        for col, config in enumerate(self.basis.iter_states(copy=False)):
            for action in self._column_actions(config):
                if abs(action.coefficient) <= self.drop_zero_atol:
                    continue

                row = self.basis.get_index(action.config)
                if row is None:
                    continue

                # A[row, col] += coefficient
                # (x @ A)[col] += x[row] * A[row, col]
                y[col] += x[row] * action.coefficient

        return y

    def rmatmat(self, matrix: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """
        Compute matrix @ self, where matrix has shape (n_vecs, n_basis).
        """
        x = np.asarray(matrix, dtype=self.dtype)

        if x.ndim != 2:
            raise ValueError("rmatmat expects a two-dimensional array.")

        if x.shape[1] != self.basis.n_states:
            raise ValueError(f"Expected matrix shape (k, {self.basis.n_states}), got {x.shape}.")

        y = np.zeros_like(x, dtype=self.dtype)

        for col, config in enumerate(self.basis.iter_states(copy=False)):
            for action in self._column_actions(config):
                if abs(action.coefficient) <= self.drop_zero_atol:
                    continue

                row = self.basis.get_index(action.config)
                if row is None:
                    continue

                # A[row, col] += coefficient
                # (X @ A)[:, col] += X[:, row] * A[row, col]
                y[:, col] += x[:, row] * action.coefficient

        return y

    def __matmul__(self, rhs):
        arr = np.asarray(rhs)

        if arr.ndim == 1:
            return self.matvec(arr)

        if arr.ndim == 2:
            return self.matmat(arr)

        raise ValueError("Right operand must be a vector or matrix.")

    def __rmatmul__(self, lhs):
        arr = np.asarray(lhs)

        if arr.ndim == 1:
            return self.rmatvec(arr)

        if arr.ndim == 2:
            return self.rmatmat(arr)

        raise ValueError("Left operand must be a vector or matrix.")

    def expectation(
        self,
        vector: npt.ArrayLike,
        *,
        conjugate: bool = True,
    ) -> complex:
        """
        Compute <v|O|v> by default.

        If conjugate=False, compute v.T @ O @ v.
        """
        v = np.asarray(vector, dtype=self.dtype)

        if conjugate:
            return complex(v.conj() @ self @ v)

        return complex(v.T @ self @ v)


@dataclass(frozen=True)
class TransposedBasisOperator:
    """
    Transpose or adjoint view of BasisOperator.
    """

    parent: BasisOperator
    conjugate: bool = False

    __array_priority__ = 10_000

    @property
    def shape(self) -> tuple[int, int]:
        n, m = self.parent.shape
        return (m, n)

    @property
    def T(self) -> BasisOperator | TransposedBasisOperator:  # noqa: N802
        if self.conjugate:
            return TransposedBasisOperator(
                parent=self.parent,
                conjugate=True,
            )
        return self.parent

    @property
    def H(self) -> BasisOperator:  # noqa: N802
        return self.parent

    def _coefficient(self, coefficient: complex) -> complex:
        if self.conjugate:
            return complex(coefficient).conjugate()
        return complex(coefficient)

    def matvec(self, vector: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """
        Compute A.T @ vector or A.H @ vector.
        """
        x = np.asarray(vector, dtype=self.parent.dtype)

        if x.ndim != 1:
            raise ValueError("matvec expects a one-dimensional vector.")

        if x.shape[0] != self.parent.basis.n_states:
            raise ValueError(
                f"Expected vector length {self.parent.basis.n_states}, got {x.shape[0]}."
            )

        y = np.zeros(self.parent.basis.n_states, dtype=self.parent.dtype)

        for col, config in enumerate(self.parent.basis.iter_states(copy=False)):
            for action in self.parent._column_actions(config):
                if abs(action.coefficient) <= self.parent.drop_zero_atol:
                    continue

                row = self.parent.basis.get_index(action.config)
                if row is None:
                    continue

                # A[row, col] += coeff
                # A.T[col, row] += coeff
                y[col] += self._coefficient(action.coefficient) * x[row]

        return y

    def rmatvec(self, vector: npt.ArrayLike) -> npt.NDArray[np.complex128]:
        """
        Compute vector @ A.T or vector @ A.H.
        """
        x = np.asarray(vector, dtype=self.parent.dtype)

        if x.ndim != 1:
            raise ValueError("rmatvec expects a one-dimensional vector.")

        if x.shape[0] != self.parent.basis.n_states:
            raise ValueError(
                f"Expected vector length {self.parent.basis.n_states}, got {x.shape[0]}."
            )

        y = np.zeros(self.parent.basis.n_states, dtype=self.parent.dtype)

        for col, config in enumerate(self.parent.basis.iter_states(copy=False)):
            amplitude = x[col]

            if amplitude == 0:
                continue

            for action in self.parent._column_actions(config):
                if abs(action.coefficient) <= self.parent.drop_zero_atol:
                    continue

                row = self.parent.basis.get_index(action.config)
                if row is None:
                    continue

                # (x @ A.T)[row] += x[col] * A[row, col]
                y[row] += amplitude * self._coefficient(action.coefficient)

        return y

    def __matmul__(self, rhs):
        arr = np.asarray(rhs)

        if arr.ndim == 1:
            return self.matvec(arr)

        if arr.ndim == 2:
            return np.column_stack([self.matvec(arr[:, i]) for i in range(arr.shape[1])])

        raise ValueError("Right operand must be a vector or matrix.")

    def __rmatmul__(self, lhs):
        arr = np.asarray(lhs)

        if arr.ndim == 1:
            return self.rmatvec(arr)

        if arr.ndim == 2:
            return np.vstack([self.rmatvec(arr[i, :]) for i in range(arr.shape[0])])

        raise ValueError("Left operand must be a vector or matrix.")


def as_basis_operator(
    basis: Basis,
    operators: LocalOperator | Sequence[LocalOperator],
    **kwargs,
) -> BasisOperator:
    """
    Convenience constructor.
    """
    if isinstance(operators, tuple | list):
        terms = tuple(operators)
    else:
        terms = (operators,)

    return BasisOperator(
        basis=basis,
        operators=terms,
        **kwargs,
    )
