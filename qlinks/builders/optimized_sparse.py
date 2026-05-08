from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.backends import SparseBackend, SparseBackendName, get_sparse_backend
from qlinks.basis import Basis
from qlinks.operators import LocalUpdateAction, LocalUpdateOperator

MissingActionPolicy = Literal["skip", "raise"]


@dataclass(frozen=True, slots=True)
class OptimizedSparseBuildStats:
    n_basis: int
    n_terms: int
    n_raw_actions: int
    n_kept_actions: int
    n_missing_actions: int
    nnz: int
    n_scratch_arrays: int


@dataclass(frozen=True, slots=True)
class OptimizedSparseBuildResult:
    matrix: Any
    stats: OptimizedSparseBuildStats


@dataclass(frozen=True, slots=True)
class OptimizedSparseHamiltonianBuilder:
    """
    Sparse Hamiltonian builder using LocalUpdateAction.

    Main optimization
    -----------------
    Operators do not allocate full output configurations. They only return
    local updates. The builder owns one scratch array and reuses it for every
    action.

    Matrix convention
    -----------------
    For each column basis state |config_col>, and action

        coefficient, variable_indices, new_values

    we construct

        config_row = config_col with local updates applied

    and insert

        H[row, col] += coefficient
    """

    dtype: npt.DTypeLike = np.complex128
    on_missing: MissingActionPolicy = "skip"
    drop_zero_atol: float = 0.0
    backend: SparseBackendName | SparseBackend = "scipy"

    def build(
        self,
        basis: Basis,
        operators: Sequence[LocalUpdateOperator],
    ) -> Any:
        return self.build_with_stats(basis, operators).matrix

    def build_with_stats(
        self,
        basis: Basis,
        operators: Sequence[LocalUpdateOperator],
    ) -> OptimizedSparseBuildResult:
        if self.on_missing not in ("skip", "raise"):
            raise ValueError("on_missing must be either 'skip' or 'raise'.")

        sparse_backend = get_sparse_backend(self.backend)
        n = basis.n_states

        if n == 0:
            matrix = sparse_backend.empty_csr((0, 0), dtype=self.dtype)
            stats = OptimizedSparseBuildStats(
                n_basis=0,
                n_terms=len(operators),
                n_raw_actions=0,
                n_kept_actions=0,
                n_missing_actions=0,
                nnz=0,
                n_scratch_arrays=0,
            )
            return OptimizedSparseBuildResult(matrix=matrix, stats=stats)

        rows: list[int] = []
        cols: list[int] = []
        data: list[complex] = []

        scratch = np.empty(basis.n_variables, dtype=np.int64)

        n_raw_actions = 0
        n_kept_actions = 0
        n_missing_actions = 0

        for col, config in enumerate(basis.iter_states(copy=False)):
            for operator in operators:
                actions = operator.apply_update(config)
                n_raw_actions += len(actions)

                for action in actions:
                    if abs(action.coefficient) <= self.drop_zero_atol:
                        continue

                    self._validate_action_against_layout(basis, action)

                    scratch[:] = config
                    scratch[action.variable_indices] = action.new_values

                    row = basis.get_index(scratch)

                    if row is None:
                        n_missing_actions += 1

                        if self.on_missing == "raise":
                            raise KeyError(
                                "Operator update produced a configuration outside the basis: "
                                f"{scratch.tolist()}"
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

        stats = OptimizedSparseBuildStats(
            n_basis=n,
            n_terms=len(operators),
            n_raw_actions=n_raw_actions,
            n_kept_actions=n_kept_actions,
            n_missing_actions=n_missing_actions,
            nnz=matrix.nnz,
            n_scratch_arrays=1,
        )

        return OptimizedSparseBuildResult(matrix=matrix, stats=stats)

    def _validate_action_against_layout(
        self,
        basis: Basis,
        action: LocalUpdateAction,
    ) -> None:
        """
        Validate only the updated values.

        This is much cheaper than validating the full scratch config every time.
        """
        for variable_index, value in zip(
            action.variable_indices,
            action.new_values,
            strict=True,
        ):
            basis.layout.local_space(int(variable_index)).validate_value(int(value))


def build_optimized_sparse_hamiltonian(
    basis: Basis,
    operators: Sequence[LocalUpdateOperator],
    *,
    dtype: npt.DTypeLike = np.complex128,
    on_missing: MissingActionPolicy = "skip",
    drop_zero_atol: float = 0.0,
    backend: SparseBackendName | SparseBackend = "scipy",
) -> Any:
    builder = OptimizedSparseHamiltonianBuilder(
        dtype=dtype,
        on_missing=on_missing,
        drop_zero_atol=drop_zero_atol,
        backend=backend,
    )
    return builder.build(basis, operators)
