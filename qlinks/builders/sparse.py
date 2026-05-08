from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.backends import SparseBackend, SparseBackendName, get_sparse_backend
from qlinks.basis import Basis
from qlinks.operators import LocalOperator, OperatorAction, combine_duplicate_actions

MissingActionPolicy = Literal["skip", "raise"]


@dataclass(frozen=True, slots=True)
class SparseBuildStats:
    n_basis: int
    n_terms: int
    n_raw_actions: int
    n_kept_actions: int
    n_missing_actions: int
    nnz: int


@dataclass(frozen=True, slots=True)
class SparseBuildResult:
    matrix: Any
    stats: SparseBuildStats


@dataclass(frozen=True, slots=True)
class SparseHamiltonianBuilder:
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

        rows: list[int] = []
        cols: list[int] = []
        data: list[complex] = []

        n_raw_actions = 0
        n_kept_actions = 0
        n_missing_actions = 0

        for col, config in enumerate(basis.iter_states(copy=False)):
            column_actions: list[OperatorAction] = []

            for operator in operators:
                actions = operator.apply(config)
                n_raw_actions += len(actions)
                column_actions.extend(actions)

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

                row = basis.get_index(action.config)

                if row is None:
                    n_missing_actions += 1

                    if self.on_missing == "raise":
                        raise KeyError(
                            "Operator action produced a configuration outside the basis: "
                            f"{action.config.tolist()}"
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
    diff = matrix - matrix.conjugate().T

    if diff.nnz == 0:
        return True

    sparse_backend = get_sparse_backend(backend)
    return sparse_backend.max_abs_data(diff) <= atol
