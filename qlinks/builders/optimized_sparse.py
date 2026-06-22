from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

import numpy as np
import numpy.typing as npt

from qlinks.backends import SparseBackend, SparseBackendName, get_sparse_backend
from qlinks.basis import Basis
from qlinks.builders.triplets import SparseTripletBuffer
from qlinks.operators import (
    DiagonalLocalOperator,
    LocalUpdateAction,
    LocalUpdateOperator,
)

MissingActionPolicy = Literal["skip", "raise"]


@dataclass(frozen=True, slots=True)
class OptimizedSparseBuildStats:
    """Counters collected by :class:`OptimizedSparseHamiltonianBuilder`.

    Attributes:
        n_basis: Number of basis states.
        n_terms: Number of local update/diagonal operators.
        n_raw_actions: Number of raw local actions evaluated.
        n_kept_actions: Number of nonzero actions inserted into the matrix.
        n_missing_actions: Number of actions whose target state was outside the
            basis.
        nnz: Number of stored nonzero entries after sparse assembly.
        n_scratch_arrays: Number of reusable scratch arrays allocated by the
            builder.
    """

    n_basis: int
    n_terms: int
    n_raw_actions: int
    n_kept_actions: int
    n_missing_actions: int
    nnz: int
    n_scratch_arrays: int


@dataclass(frozen=True, slots=True)
class OptimizedSparseBuildResult:
    """Optimized sparse matrix together with build statistics.

    Attributes:
        matrix: Built sparse matrix.
        stats: Counters describing the optimized build.
    """

    matrix: Any
    stats: OptimizedSparseBuildStats


@dataclass(frozen=True, slots=True)
class _PreparedUpdateOperators:
    diagonal_value_functions: tuple[Callable[[npt.ArrayLike], complex | None], ...]
    single_update_functions: tuple[
        Callable[
            [npt.ArrayLike],
            tuple[complex, npt.NDArray[np.int64], npt.NDArray[np.int64]] | None,
        ],
        ...,
    ]
    fallback_operators: tuple[LocalUpdateOperator, ...]


def _prepare_update_operators(
    operators: Sequence[object],
) -> _PreparedUpdateOperators:
    """Classify operators once before entering the optimized hot loop."""
    diagonal_value_functions: list[Callable[[npt.ArrayLike], complex | None]] = []
    single_update_functions: list[
        Callable[
            [npt.ArrayLike],
            tuple[complex, npt.NDArray[np.int64], npt.NDArray[np.int64]] | None,
        ]
    ] = []
    fallback_operators: list[LocalUpdateOperator] = []

    for operator in operators:
        diagonal_value = getattr(operator, "diagonal_value", None)

        if callable(diagonal_value):
            diagonal_value_functions.append(diagonal_value)
            continue

        supports_single_update = bool(getattr(operator, "supports_single_update", True))

        if supports_single_update:
            single_update = getattr(operator, "single_update", None)

            if callable(single_update):
                single_update_functions.append(single_update)
                continue

        apply_update = getattr(operator, "apply_update", None)

        if not callable(apply_update):
            raise TypeError(
                "OptimizedSparseHamiltonianBuilder operators must provide either "
                "diagonal_value(config) or apply_update(config)."
            )

        fallback_operators.append(operator)

    return _PreparedUpdateOperators(
        diagonal_value_functions=tuple(diagonal_value_functions),
        single_update_functions=tuple(single_update_functions),
        fallback_operators=tuple(fallback_operators),
    )


@dataclass(frozen=True, slots=True)
class OptimizedSparseHamiltonianBuilder:
    """Sparse builder for update-style local operators.

    Operators return compact local updates rather than full output
    configurations.  The builder owns a reusable scratch configuration and
    inserts ``H[row, col] += coefficient`` after applying each update to the
    current column configuration.

    Attributes:
        dtype: Matrix dtype.
        on_missing: Policy for actions outside the basis.
        drop_zero_atol: Absolute threshold for dropping small coefficients.
        backend: Sparse backend name or backend object.
    """

    dtype: npt.DTypeLike = np.complex128
    on_missing: MissingActionPolicy = "skip"
    drop_zero_atol: float = 0.0
    backend: SparseBackendName | SparseBackend = "scipy"

    def build(
        self,
        basis: Basis,
        operators: Sequence[LocalUpdateOperator | DiagonalLocalOperator],
    ) -> Any:
        return self.build_with_stats(basis, operators).matrix

    def build_with_stats(
        self,
        basis: Basis,
        operators: Sequence[LocalUpdateOperator | DiagonalLocalOperator],
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

        triplets = SparseTripletBuffer()
        append_row = triplets.rows.append
        append_col = triplets.cols.append
        append_data = triplets.data.append

        scratch = np.empty(basis.n_variables, dtype=np.int64)

        n_raw_actions = 0
        n_kept_actions = 0
        n_missing_actions = 0

        states = basis.states
        encode_config = basis.encoder.encode
        index = basis.index
        prepared_operators = _prepare_update_operators(operators)

        for col, config in enumerate(states):
            diagonal_coefficient = 0.0 + 0.0j

            for diagonal_value in prepared_operators.diagonal_value_functions:
                value = diagonal_value(config)

                if value is None:
                    continue

                n_raw_actions += 1
                diagonal_coefficient += complex(value)

            if abs(diagonal_coefficient) > self.drop_zero_atol:
                append_row(col)
                append_col(col)
                append_data(diagonal_coefficient)
                n_kept_actions += 1

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

                append_row(row)
                append_col(col)
                append_data(coefficient)
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
    operators: Sequence[LocalUpdateOperator | DiagonalLocalOperator],
    *,
    dtype: npt.DTypeLike = np.complex128,
    on_missing: MissingActionPolicy = "skip",
    drop_zero_atol: float = 0.0,
    backend: SparseBackendName | SparseBackend = "scipy",
) -> Any:
    """Build a sparse Hamiltonian using update-style operators.

    Args:
        basis: Basis that fixes the matrix row/column order.
        operators: Operators supporting ``diagonal_value`` or ``apply_update``.
        dtype: Matrix dtype.
        on_missing: Policy for actions outside the basis.
        drop_zero_atol: Absolute threshold for dropping small coefficients.
        backend: Sparse backend name or backend object.

    Returns:
        Sparse Hamiltonian matrix.
    """
    builder = OptimizedSparseHamiltonianBuilder(
        dtype=dtype,
        on_missing=on_missing,
        drop_zero_atol=drop_zero_atol,
        backend=backend,
    )
    return builder.build(basis, operators)
