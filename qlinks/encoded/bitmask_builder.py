from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.backends import SparseBackend, SparseBackendName, get_sparse_backend
from qlinks.encoded.binary_basis import BinaryEncodedBasis
from qlinks.encoded.bitmask_operators import BitmaskOperator

MissingActionPolicy = Literal["skip", "raise"]


@dataclass(frozen=True, slots=True)
class BitmaskSparseBuildStats:
    n_basis: int
    n_terms: int
    n_raw_actions: int
    n_kept_actions: int
    n_missing_actions: int
    nnz: int


@dataclass(frozen=True, slots=True)
class BitmaskSparseBuildResult:
    matrix: Any
    stats: BitmaskSparseBuildStats


@dataclass(frozen=True, slots=True)
class BitmaskSparseHamiltonianBuilder:
    """
    Sparse Hamiltonian builder for BinaryEncodedBasis.

    This avoids constructing NumPy config arrays during matrix assembly.
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

        for col in range(basis.n_states):
            code = basis.code(col)

            for operator in operators:
                actions = operator.apply_code(code)
                n_raw_actions += len(actions)

                for action in actions:
                    if abs(action.coefficient) <= self.drop_zero_atol:
                        continue

                    row = basis.get_index(action.code)

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
    builder = BitmaskSparseHamiltonianBuilder(
        dtype=dtype,
        on_missing=on_missing,
        drop_zero_atol=drop_zero_atol,
        backend=backend,
    )
    return builder.build(basis, operators)
