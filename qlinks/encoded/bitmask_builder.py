from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.backends import SparseBackend, SparseBackendName, get_sparse_backend
from qlinks.encoded.binary_basis import BinaryEncodedBasis
from qlinks.encoded.bitmask_operators import BitmaskOperator

MissingActionPolicy = Literal["skip", "raise"]


@dataclass(frozen=True, slots=True)
class BitmaskSparseBuildStats:
    """Counters collected by :class:`BitmaskSparseHamiltonianBuilder`.

    Attributes:
        n_basis: Number of encoded basis states.
        n_terms: Number of bitmask operators.
        n_raw_actions: Number of raw bitmask actions evaluated.
        n_kept_actions: Number of nonzero actions inserted into the matrix.
        n_missing_actions: Number of actions whose target code was outside the
            basis.
        nnz: Number of stored nonzero entries after sparse assembly.
    """

    n_basis: int
    n_terms: int
    n_raw_actions: int
    n_kept_actions: int
    n_missing_actions: int
    nnz: int


@dataclass(frozen=True, slots=True)
class BitmaskSparseBuildResult:
    """Bitmask sparse matrix together with build statistics.

    Attributes:
        matrix: Built sparse matrix.
        stats: Counters describing the bitmask build.
    """

    matrix: Any
    stats: BitmaskSparseBuildStats


@dataclass(frozen=True, slots=True)
class _PreparedBitmaskOperators:
    diagonal_value_code_functions: tuple[Callable[[int], complex | None], ...]
    single_action_code_functions: tuple[Callable[[int], tuple[complex, int] | None], ...]
    fallback_operators: tuple[BitmaskOperator, ...]


def _prepare_bitmask_operators(
    operators: Sequence[BitmaskOperator],
) -> _PreparedBitmaskOperators:
    """Classify bitmask operators once before entering the hot loop."""
    diagonal_value_code_functions: list[Callable[[int], complex | None]] = []
    single_action_code_functions: list[Callable[[int], tuple[complex, int] | None]] = []
    fallback_operators: list[BitmaskOperator] = []

    for operator in operators:
        diagonal_value_code = getattr(operator, "diagonal_value_code", None)

        if callable(diagonal_value_code):
            diagonal_value_code_functions.append(diagonal_value_code)
            continue

        single_action_code = getattr(operator, "single_action_code", None)

        if callable(single_action_code):
            single_action_code_functions.append(single_action_code)
            continue

        fallback_operators.append(operator)

    return _PreparedBitmaskOperators(
        diagonal_value_code_functions=tuple(diagonal_value_code_functions),
        single_action_code_functions=tuple(single_action_code_functions),
        fallback_operators=tuple(fallback_operators),
    )


@dataclass(frozen=True, slots=True)
class BitmaskSparseHamiltonianBuilder:
    """Sparse builder for :class:`BinaryEncodedBasis`.

    Bitmask operators act directly on integer-encoded configurations, avoiding
    NumPy configuration arrays in the matrix-assembly hot loop.

    Attributes:
        dtype: Matrix dtype.
        on_missing: Policy for actions outside the encoded basis.
        drop_zero_atol: Absolute threshold for dropping small coefficients.
        backend: Sparse backend name or backend object.
    """

    dtype: npt.DTypeLike = np.complex128
    on_missing: MissingActionPolicy = "skip"
    drop_zero_atol: float = 0.0
    backend: SparseBackendName | SparseBackend = "scipy"

    def build(
        self,
        basis: BinaryEncodedBasis,
        operators: Sequence[BitmaskOperator],
    ) -> Any:
        return self.build_with_stats(basis, operators).matrix

    def build_with_stats(
        self,
        basis: BinaryEncodedBasis,
        operators: Sequence[BitmaskOperator],
    ) -> BitmaskSparseBuildResult:
        if self.on_missing not in ("skip", "raise"):
            raise ValueError("on_missing must be either 'skip' or 'raise'.")

        sparse_backend = get_sparse_backend(self.backend)
        n = basis.n_states

        if n == 0:
            matrix = sparse_backend.empty_csr((0, 0), dtype=self.dtype)
            stats = BitmaskSparseBuildStats(
                n_basis=0,
                n_terms=len(operators),
                n_raw_actions=0,
                n_kept_actions=0,
                n_missing_actions=0,
                nnz=0,
            )
            return BitmaskSparseBuildResult(matrix=matrix, stats=stats)

        rows: list[int] = []
        cols: list[int] = []
        data: list[complex] = []

        n_raw_actions = 0
        n_kept_actions = 0
        n_missing_actions = 0

        codes = basis.codes
        index = basis.index
        prepared_operators = _prepare_bitmask_operators(operators)

        for col, code_obj in enumerate(codes):
            code = int(code_obj)
            diagonal_coefficient = 0.0 + 0.0j

            for diagonal_value_code in prepared_operators.diagonal_value_code_functions:
                diagonal_value = diagonal_value_code(code)

                if diagonal_value is not None:
                    n_raw_actions += 1
                    diagonal_coefficient += complex(diagonal_value)

            for single_action_code in prepared_operators.single_action_code_functions:
                single_action = single_action_code(code)

                if single_action is None:
                    continue

                n_raw_actions += 1
                coefficient, new_code = single_action

                if abs(coefficient) <= self.drop_zero_atol:
                    continue

                row = index.get(int(new_code))

                if row is None:
                    n_missing_actions += 1

                    if self.on_missing == "raise":
                        raise KeyError(
                            "Bitmask operator produced a code outside the basis: " f"{new_code}"
                        )

                    continue

                rows.append(row)
                cols.append(col)
                data.append(coefficient)
                n_kept_actions += 1

            for operator in prepared_operators.fallback_operators:
                actions = operator.apply_code(code)
                n_raw_actions += len(actions)

                for action in actions:
                    if abs(action.coefficient) <= self.drop_zero_atol:
                        continue

                    row = index.get(int(action.code))

                    if row is None:
                        n_missing_actions += 1

                        if self.on_missing == "raise":
                            raise KeyError(
                                "Bitmask operator produced a code outside the basis: "
                                f"{action.code}"
                            )

                        continue

                    rows.append(row)
                    cols.append(col)
                    data.append(action.coefficient)
                    n_kept_actions += 1

            if abs(diagonal_coefficient) > self.drop_zero_atol:
                rows.append(col)
                cols.append(col)
                data.append(diagonal_coefficient)
                n_kept_actions += 1

        matrix = sparse_backend.coo_matrix(
            data=sparse_backend.as_data_array(data, dtype=self.dtype),
            rows=sparse_backend.as_index_array(rows),
            cols=sparse_backend.as_index_array(cols),
            shape=(n, n),
            dtype=self.dtype,
        )

        matrix.sum_duplicates()
        matrix.eliminate_zeros()

        stats = BitmaskSparseBuildStats(
            n_basis=n,
            n_terms=len(operators),
            n_raw_actions=n_raw_actions,
            n_kept_actions=n_kept_actions,
            n_missing_actions=n_missing_actions,
            nnz=matrix.nnz,
        )

        return BitmaskSparseBuildResult(matrix=matrix, stats=stats)


def build_bitmask_sparse_hamiltonian(
    basis: BinaryEncodedBasis,
    operators: Sequence[BitmaskOperator],
    *,
    dtype: npt.DTypeLike = np.complex128,
    on_missing: MissingActionPolicy = "skip",
    drop_zero_atol: float = 0.0,
    backend: SparseBackendName | SparseBackend = "scipy",
) -> Any:
    """Build a sparse Hamiltonian from bitmask operators.

    Args:
        basis: Encoded basis that fixes matrix row/column order.
        operators: Bitmask operators to sum.
        dtype: Matrix dtype.
        on_missing: Policy for actions outside the encoded basis.
        drop_zero_atol: Absolute threshold for dropping small coefficients.
        backend: Sparse backend name or backend object.

    Returns:
        Sparse Hamiltonian matrix.
    """
    builder = BitmaskSparseHamiltonianBuilder(
        dtype=dtype,
        on_missing=on_missing,
        drop_zero_atol=drop_zero_atol,
        backend=backend,
    )
    return builder.build(basis, operators)
