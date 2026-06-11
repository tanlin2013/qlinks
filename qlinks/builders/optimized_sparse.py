from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

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
class _PreparedUpdateOperators:
    single_update_functions: tuple[
        Callable[
            [npt.ArrayLike],
            tuple[complex, npt.NDArray[np.int64], npt.NDArray[np.int64]] | None,
        ],
        ...,
    ]
    fallback_operators: tuple[LocalUpdateOperator, ...]


def _prepare_update_operators(
    operators: Sequence[LocalUpdateOperator],
) -> _PreparedUpdateOperators:
    """Classify update operators once before entering the optimized hot loop."""
    single_update_functions: list[
        Callable[
            [npt.ArrayLike],
            tuple[complex, npt.NDArray[np.int64], npt.NDArray[np.int64]] | None,
        ]
    ] = []
    fallback_operators: list[LocalUpdateOperator] = []

    for operator in operators:
        supports_single_update = bool(getattr(operator, "supports_single_update", True))

        if supports_single_update:
            single_update = getattr(operator, "single_update", None)

            if callable(single_update):
                single_update_functions.append(single_update)
                continue

        fallback_operators.append(operator)

    return _PreparedUpdateOperators(
        single_update_functions=tuple(single_update_functions),
        fallback_operators=tuple(fallback_operators),
    )


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

        states = basis.states
        encode_config = basis.encoder.encode
        index = basis.index
        prepared_operators = _prepare_update_operators(operators)

        for col, config in enumerate(states):
            for single_update in prepared_operators.single_update_functions:
                update = single_update(config)

                if update is None:
                    continue

                n_raw_actions += 1
                coefficient, variable_indices, new_values = update

                if abs(coefficient) <= self.drop_zero_atol:
                    continue

                self._validate_update_values(
                    basis,
                    variable_indices=variable_indices,
                    new_values=new_values,
                )

                scratch[:] = config
                scratch[variable_indices] = new_values

                row = index.get(encode_config(scratch))

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
                data.append(coefficient)
                n_kept_actions += 1

            for operator in prepared_operators.fallback_operators:
                actions = operator.apply_update(config)
                n_raw_actions += len(actions)

                for action in actions:
                    if abs(action.coefficient) <= self.drop_zero_atol:
                        continue

                    self._validate_action_against_layout(basis, action)

                    scratch[:] = config
                    scratch[action.variable_indices] = action.new_values

                    row = index.get(encode_config(scratch))

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

    def _validate_update_values(
        self,
        basis: Basis,
        *,
        variable_indices: npt.ArrayLike,
        new_values: npt.ArrayLike,
    ) -> None:
        """Validate only the updated values."""
        for variable_index, value in zip(
            np.asarray(variable_indices, dtype=np.int64),
            np.asarray(new_values, dtype=np.int64),
            strict=True,
        ):
            basis.layout.local_space(int(variable_index)).validate_value(int(value))

    def _validate_action_against_layout(
        self,
        basis: Basis,
        action: LocalUpdateAction,
    ) -> None:
        """
        Validate only the updated values.

        This is much cheaper than validating the full scratch config every time.
        """
        self._validate_update_values(
            basis,
            variable_indices=action.variable_indices,
            new_values=action.new_values,
        )


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
