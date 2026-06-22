from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.backends import SparseBackend, SparseBackendName, get_sparse_backend
from qlinks.basis import Basis
from qlinks.builders.triplets import SparseTripletBuffer
from qlinks.operators import LocalOperator, OperatorAction, combine_duplicate_actions

MissingActionPolicy = Literal["skip", "raise"]


@dataclass(frozen=True, slots=True)
class SparseBuildStats:
    """Counters collected while building a sparse matrix.

    Attributes:
        n_basis: Number of basis states.
        n_terms: Number of local operators.
        n_raw_actions: Number of raw operator actions evaluated.
        n_kept_actions: Number of nonzero actions inserted into the matrix.
        n_missing_actions: Number of actions whose target state was outside the
            basis.
        nnz: Number of stored nonzero entries after duplicate summation.
    """

    n_basis: int
    n_terms: int
    n_raw_actions: int
    n_kept_actions: int
    n_missing_actions: int
    nnz: int


@dataclass(frozen=True, slots=True)
class SparseBuildResult:
    """Sparse matrix together with build statistics.

    Attributes:
        matrix: Built sparse matrix.
        stats: Counters describing the build.
    """

    matrix: Any
    stats: SparseBuildStats


@dataclass(frozen=True, slots=True)
class _PreparedLocalOperators:
    diagonal_value_functions: tuple[Callable[[npt.ArrayLike], complex | None], ...]
    fallback_operators: tuple[LocalOperator, ...]


def _prepare_local_operators(
    operators: Sequence[LocalOperator],
) -> _PreparedLocalOperators:
    """Classify local operators once before entering the sparse-builder hot loop."""
    diagonal_value_functions: list[Callable[[npt.ArrayLike], complex | None]] = []
    fallback_operators: list[LocalOperator] = []

    for operator in operators:
        diagonal_value = getattr(operator, "diagonal_value", None)

        if callable(diagonal_value):
            diagonal_value_functions.append(diagonal_value)
            continue

        fallback_operators.append(operator)

    return _PreparedLocalOperators(
        diagonal_value_functions=tuple(diagonal_value_functions),
        fallback_operators=tuple(fallback_operators),
    )


@dataclass(frozen=True, slots=True)
class SparseHamiltonianBuilder:
    """Generic sparse builder for configuration-space local operators.

    The builder applies each operator to each basis configuration and inserts
    the resulting matrix element ``H[row, col]``.  It is the most general
    builder and works with any :class:`LocalOperator`, at the cost of allocating
    full target configurations for non-diagonal actions.

    Attributes:
        dtype: Matrix dtype.
        on_missing: Policy for actions outside the basis.  ``"skip"`` drops
            them; ``"raise"`` turns them into a ``KeyError``.
        combine_duplicates: Whether to combine multiple actions from the same
            column to the same target before insertion.
        drop_zero_atol: Absolute threshold for dropping small coefficients.
        backend: Sparse backend name or backend object.
    """

    dtype: npt.DTypeLike = np.complex128
    on_missing: MissingActionPolicy = "skip"
    combine_duplicates: bool = True
    drop_zero_atol: float = 0.0
    backend: SparseBackendName | SparseBackend = "scipy"

    def build(
        self,
        basis: Basis,
        operators: Sequence[LocalOperator],
    ) -> Any:
        return self.build_with_stats(basis, operators).matrix

    def build_with_stats(
        self,
        basis: Basis,
        operators: Sequence[LocalOperator],
    ) -> SparseBuildResult:
        if self.on_missing not in ("skip", "raise"):
            raise ValueError("on_missing must be either 'skip' or 'raise'.")

        sparse_backend = get_sparse_backend(self.backend)
        n = basis.n_states

        if n == 0:
            matrix = sparse_backend.empty_csr((0, 0), dtype=self.dtype)
            stats = SparseBuildStats(
                n_basis=0,
                n_terms=len(operators),
                n_raw_actions=0,
                n_kept_actions=0,
                n_missing_actions=0,
                nnz=0,
            )
            return SparseBuildResult(matrix=matrix, stats=stats)

        triplets = SparseTripletBuffer()
        append_row = triplets.rows.append
        append_col = triplets.cols.append
        append_data = triplets.data.append

        n_raw_actions = 0
        n_kept_actions = 0
        n_missing_actions = 0

        states = basis.states
        encode_config = basis.encoder.encode
        index = basis.index
        prepared_operators = _prepare_local_operators(operators)

        for col, config in enumerate(states):
            column_actions: list[OperatorAction] = []
            diagonal_coefficient = 0.0 + 0.0j

            for diagonal_value in prepared_operators.diagonal_value_functions:
                value = diagonal_value(config)

                if value is not None:
                    n_raw_actions += 1
                    diagonal_coefficient += complex(value)

            for operator in prepared_operators.fallback_operators:
                actions = operator.apply(config)
                n_raw_actions += len(actions)
                column_actions.extend(actions)

            if abs(diagonal_coefficient) > self.drop_zero_atol:
                append_row(col)
                append_col(col)
                append_data(diagonal_coefficient)
                n_kept_actions += 1

            if self.combine_duplicates:
                actions_to_insert = combine_duplicate_actions(
                    column_actions,
                    atol=self.drop_zero_atol,
                )
            else:
                actions_to_insert = tuple(column_actions)

            for action in actions_to_insert:
                if abs(action.coefficient) <= self.drop_zero_atol:
                    continue

                row = index.get(encode_config(action.config))

                if row is None:
                    n_missing_actions += 1

                    if self.on_missing == "raise":
                        raise KeyError(
                            "Operator action produced a configuration outside the basis: "
                            f"{action.config.tolist()}"
                        )

                    continue

                append_row(row)
                append_col(col)
                append_data(action.coefficient)
                n_kept_actions += 1

        matrix = sparse_backend.coo_matrix(
            data=sparse_backend.as_data_array(triplets.data, dtype=self.dtype),
            rows=sparse_backend.as_index_array(triplets.rows),
            cols=sparse_backend.as_index_array(triplets.cols),
            shape=(n, n),
            dtype=self.dtype,
        )

        stats = SparseBuildStats(
            n_basis=n,
            n_terms=len(operators),
            n_raw_actions=n_raw_actions,
            n_kept_actions=n_kept_actions,
            n_missing_actions=n_missing_actions,
            nnz=matrix.nnz,
        )

        return SparseBuildResult(matrix=matrix, stats=stats)


def build_sparse_hamiltonian(
    basis: Basis,
    operators: Sequence[LocalOperator],
    *,
    dtype: npt.DTypeLike = np.complex128,
    on_missing: MissingActionPolicy = "skip",
    combine_duplicates: bool = True,
    drop_zero_atol: float = 0.0,
    backend: SparseBackendName | SparseBackend = "scipy",
) -> Any:
    """Build a sparse Hamiltonian from local operators.

    Args:
        basis: Basis that fixes the matrix row/column order.
        operators: Local operators to sum.
        dtype: Matrix dtype.
        on_missing: Policy for actions outside the basis.
        combine_duplicates: Whether to combine duplicate column actions before
            inserting them.
        drop_zero_atol: Absolute threshold for dropping small coefficients.
        backend: Sparse backend name or backend object.

    Returns:
        Sparse Hamiltonian matrix.
    """
    builder = SparseHamiltonianBuilder(
        dtype=dtype,
        on_missing=on_missing,
        combine_duplicates=combine_duplicates,
        drop_zero_atol=drop_zero_atol,
        backend=backend,
    )

    return builder.build(basis, operators)


def is_hermitian_sparse(
    matrix: Any,
    *,
    atol: float = 1e-12,
    backend: SparseBackendName | SparseBackend = "auto",
) -> bool:
    """Return whether a sparse matrix is Hermitian within tolerance.

    Args:
        matrix: Sparse matrix to test.
        atol: Absolute tolerance for the largest entry of ``A - A.conjugate().T``.
        backend: Sparse backend used to inspect matrix data.

    Returns:
        ``True`` if the matrix is numerically Hermitian.
    """
    diff = matrix - matrix.conjugate().T

    if diff.nnz == 0:
        return True

    sparse_backend = get_sparse_backend(backend)
    return sparse_backend.max_abs_data(diff) <= atol
