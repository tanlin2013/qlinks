from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, MutableMapping

import numpy as np
import scipy.sparse as scipy_sparse
from numpy.typing import NDArray

from qlinks.open_system.backend import (
    OpenSystemBackend,
    OpenSystemBackendName,
    as_backend_dense_array,
    as_backend_sparse_matrix,
    get_open_system_backend,
)
from qlinks.open_system.states import random_pure_state

ArrayC = NDArray[np.complex128]
ArrayF = NDArray[np.float64]
StateSampler = Callable[[np.random.Generator], Any]


@dataclass(frozen=True, slots=True)
class _SparseJumpRateEvaluator:
    """Sparse jump-rate evaluator that avoids full J|psi> work buffers."""

    jumps: tuple[scipy_sparse.csr_array, ...]
    active_rows: tuple[NDArray[np.int64], ...]
    row_columns: tuple[tuple[NDArray[np.int64], ...], ...]
    row_values: tuple[tuple[NDArray[np.complex128], ...], ...]
    single_entry_columns: tuple[NDArray[np.int64] | None, ...]
    single_entry_weights: tuple[NDArray[np.float64] | None, ...]
    single_entry_rate_matrix: scipy_sparse.csr_array | None
    expanded_rate_operator: scipy_sparse.csr_array | None
    expanded_rate_jump_indices: NDArray[np.int64]
    expanded_rate_row_splits: NDArray[np.int64]
    generic_jump_indices: NDArray[np.int64]

    @property
    def n_jumps(self) -> int:
        return len(self.jumps)


@dataclass(frozen=True, slots=True)
class _McwfPreparedOperators:
    backend: OpenSystemBackend
    hamiltonian: Any
    jumps: tuple[Any, ...]
    effective_hamiltonian_matrix: Any
    total_jump_rate_operator: Any | None = None
    sparse_jump_rate_evaluator: _SparseJumpRateEvaluator | None = None
    uses_sparse_operators: bool = False
    uses_sparse_rate_evaluator: bool = False

    def __getstate__(self) -> tuple[Any, ...]:
        return (
            self.backend.name,
            self.hamiltonian,
            self.jumps,
            self.effective_hamiltonian_matrix,
            self.total_jump_rate_operator,
            self.sparse_jump_rate_evaluator,
            self.uses_sparse_operators,
            self.uses_sparse_rate_evaluator,
        )

    def __setstate__(self, state: tuple[Any, ...]) -> None:
        (
            backend_name,
            hamiltonian,
            jumps,
            effective_hamiltonian_matrix,
            total_jump_rate_operator,
            sparse_jump_rate_evaluator,
            uses_sparse_operators,
            uses_sparse_rate_evaluator,
        ) = state
        object.__setattr__(self, "backend", get_open_system_backend(backend_name))
        object.__setattr__(self, "hamiltonian", hamiltonian)
        object.__setattr__(self, "jumps", jumps)
        object.__setattr__(self, "effective_hamiltonian_matrix", effective_hamiltonian_matrix)
        object.__setattr__(self, "total_jump_rate_operator", total_jump_rate_operator)
        object.__setattr__(self, "sparse_jump_rate_evaluator", sparse_jump_rate_evaluator)
        object.__setattr__(self, "uses_sparse_operators", uses_sparse_operators)
        object.__setattr__(self, "uses_sparse_rate_evaluator", uses_sparse_rate_evaluator)


@dataclass(frozen=True, slots=True)
class _JumpEvaluation:
    jumped_states: tuple[Any, ...]
    rates: NDArray[np.float64]

    @property
    def n_jumps(self) -> int:
        return int(self.rates.size)

    def probabilities(self, step_size: float) -> NDArray[np.float64]:
        if self.rates.size == 0:
            return np.zeros(0, dtype=np.float64)

        return np.maximum(step_size * self.rates, 0.0)


def _prepare_mcwf_operators(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
    prefer_sparse_operators: bool = True,
    prefer_sparse_rate_evaluator: bool = True,
) -> _McwfPreparedOperators:
    backend_obj = get_open_system_backend(backend)
    use_sparse_operators = (
        backend_obj.name == "scipy"
        and prefer_sparse_operators
        and (
            _is_sparse_like_operator(hamiltonian)
            or any(_is_sparse_like_operator(jump) for jump in jumps)
        )
    )

    if use_sparse_operators:
        hamiltonian_backend = as_backend_sparse_matrix(
            hamiltonian,
            backend=backend_obj,
            format="csr",
            dtype=np.complex128,
        )
        jump_operators = tuple(
            as_backend_sparse_matrix(
                jump,
                backend=backend_obj,
                format="csr",
                dtype=np.complex128,
            )
            for jump in jumps
        )
    else:
        hamiltonian_backend = as_backend_dense_array(
            hamiltonian,
            backend=backend_obj,
            dtype=np.complex128,
        )
        jump_operators = tuple(
            as_backend_dense_array(
                jump,
                backend=backend_obj,
                dtype=np.complex128,
            )
            for jump in jumps
        )

    total_jump_rate_operator = _total_jump_rate_operator(
        jump_operators,
        shape=tuple(int(axis) for axis in hamiltonian_backend.shape),
    )
    effective_hamiltonian_matrix = _effective_hamiltonian_from_total_rate_operator(
        hamiltonian_backend,
        total_jump_rate_operator,
    )

    sparse_jump_rate_evaluator = (
        _build_sparse_jump_rate_evaluator(jump_operators)
        if (backend_obj.name == "scipy" and use_sparse_operators and prefer_sparse_rate_evaluator)
        else None
    )

    return _McwfPreparedOperators(
        backend=backend_obj,
        hamiltonian=hamiltonian_backend,
        jumps=jump_operators,
        effective_hamiltonian_matrix=effective_hamiltonian_matrix,
        total_jump_rate_operator=total_jump_rate_operator,
        sparse_jump_rate_evaluator=sparse_jump_rate_evaluator,
        uses_sparse_operators=use_sparse_operators,
        uses_sparse_rate_evaluator=sparse_jump_rate_evaluator is not None,
    )


def _is_sparse_like_operator(operator: Any) -> bool:
    return (
        scipy_sparse.issparse(operator)
        or hasattr(operator, "asformat")
        or hasattr(operator, "tocsr")
    )


def _as_numpy_or_scipy_sparse(operator: Any) -> Any:
    if scipy_sparse.issparse(operator):
        return operator.tocsr().astype(np.complex128)

    if hasattr(operator, "tocsr"):
        return operator.tocsr().astype(np.complex128)

    if hasattr(operator, "asformat"):
        return operator.asformat("csr").astype(np.complex128)

    return np.asarray(operator, dtype=np.complex128)


def _build_sparse_jump_rate_evaluator(
    jumps: tuple[Any, ...],
    *,
    max_active_row_fraction: float = 0.5,
    min_single_entry_matrix_jumps: int = 8,
) -> _SparseJumpRateEvaluator | None:
    """Return a sparse row evaluator when it should beat full sparse matmul.

    The standard sparse-matrix product ``J @ states`` allocates a full
    ``(dim, n_trajectories)`` dense block for every jump. Cage-Lindblad jumps
    are often very row-sparse, so it is cheaper to evaluate only nonzero output
    rows and sum their squared norms.
    """
    if not jumps or not all(scipy_sparse.issparse(jump) for jump in jumps):
        return None

    dim = int(jumps[0].shape[0])
    csr_jumps: list[scipy_sparse.csr_array] = []
    active_rows: list[NDArray[np.int64]] = []
    row_columns: list[tuple[NDArray[np.int64], ...]] = []
    row_values: list[tuple[NDArray[np.complex128], ...]] = []
    single_entry_columns: list[NDArray[np.int64] | None] = []
    single_entry_weights: list[NDArray[np.float64] | None] = []
    total_active_rows = 0

    for jump in jumps:
        jump_csr = jump.tocsr().astype(np.complex128)
        rows = np.flatnonzero(np.diff(jump_csr.indptr) > 0).astype(np.int64, copy=False)
        jump_row_columns: list[NDArray[np.int64]] = []
        jump_row_values: list[NDArray[np.complex128]] = []
        one_entry_columns: list[int] = []
        one_entry_weights: list[float] = []
        has_only_one_entry_rows = True

        indptr = jump_csr.indptr
        indices = jump_csr.indices
        data = jump_csr.data
        for row in rows:
            start = int(indptr[row])
            stop = int(indptr[row + 1])
            columns = indices[start:stop].astype(np.int64, copy=True)
            values = data[start:stop].astype(np.complex128, copy=True)
            jump_row_columns.append(columns)
            jump_row_values.append(values)

            if columns.size == 1:
                one_entry_columns.append(int(columns[0]))
                one_entry_weights.append(float(abs(values[0]) ** 2))
            else:
                has_only_one_entry_rows = False

        csr_jumps.append(jump_csr)
        active_rows.append(rows)
        row_columns.append(tuple(jump_row_columns))
        row_values.append(tuple(jump_row_values))
        if has_only_one_entry_rows:
            single_entry_columns.append(np.asarray(one_entry_columns, dtype=np.int64))
            single_entry_weights.append(np.asarray(one_entry_weights, dtype=np.float64))
        else:
            single_entry_columns.append(None)
            single_entry_weights.append(None)
        total_active_rows += int(rows.size)

    if dim <= 0:
        return None

    average_active_fraction = total_active_rows / float(dim * len(jumps))
    if average_active_fraction > max_active_row_fraction:
        return None

    single_entry_rate_matrix: scipy_sparse.csr_array | None = None
    single_entry_matrix_jump_count = sum(
        columns is not None and columns.size > 0 for columns in single_entry_columns
    )
    if single_entry_matrix_jump_count >= min_single_entry_matrix_jumps:
        matrix_row_blocks: list[NDArray[np.int64]] = []
        matrix_col_blocks: list[NDArray[np.int64]] = []
        matrix_value_blocks: list[NDArray[np.float64]] = []
        for jump_index, (columns, weights) in enumerate(
            zip(single_entry_columns, single_entry_weights, strict=True)
        ):
            if columns is None or weights is None or columns.size == 0:
                continue

            matrix_row_blocks.append(np.full(columns.size, jump_index, dtype=np.int64))
            matrix_col_blocks.append(columns)
            matrix_value_blocks.append(weights)

        if matrix_value_blocks:
            single_entry_rate_matrix = scipy_sparse.csr_array(
                (
                    np.concatenate(matrix_value_blocks),
                    (np.concatenate(matrix_row_blocks), np.concatenate(matrix_col_blocks)),
                ),
                shape=(len(jumps), dim),
                dtype=np.float64,
            )
            single_entry_rate_matrix.sum_duplicates()
            single_entry_rate_matrix.eliminate_zeros()

    generic_jump_indices = (
        np.asarray(
            [index for index, columns in enumerate(single_entry_columns) if columns is None],
            dtype=np.int64,
        )
        if single_entry_rate_matrix is not None
        else np.arange(len(jumps), dtype=np.int64)
    )

    (
        expanded_rate_operator,
        expanded_rate_jump_indices,
        expanded_rate_row_splits,
    ) = _build_expanded_sparse_rate_operator(
        row_columns=tuple(row_columns),
        row_values=tuple(row_values),
        jump_indices=generic_jump_indices,
        dim=dim,
    )

    return _SparseJumpRateEvaluator(
        jumps=tuple(csr_jumps),
        active_rows=tuple(active_rows),
        row_columns=tuple(row_columns),
        row_values=tuple(row_values),
        single_entry_columns=tuple(single_entry_columns),
        single_entry_weights=tuple(single_entry_weights),
        single_entry_rate_matrix=single_entry_rate_matrix,
        expanded_rate_operator=expanded_rate_operator,
        expanded_rate_jump_indices=expanded_rate_jump_indices,
        expanded_rate_row_splits=expanded_rate_row_splits,
        generic_jump_indices=generic_jump_indices,
    )


def _build_expanded_sparse_rate_operator(
    *,
    row_columns: tuple[tuple[NDArray[np.int64], ...], ...],
    row_values: tuple[tuple[NDArray[np.complex128], ...], ...],
    jump_indices: NDArray[np.int64],
    dim: int,
) -> tuple[scipy_sparse.csr_array | None, NDArray[np.int64], NDArray[np.int64]]:
    """Return one stacked sparse row operator for generic jump-rate evaluation.

    Each row of the expanded operator is one nonzero output row of one jump.
    A single sparse matmul gives all row amplitudes; grouped row-norm sums then
    recover ``||J_mu psi||^2`` for each jump.  This removes the Python loop over
    active rows in the common Cage-Lindblad case where jumps have multi-entry
    rows and therefore cannot use the single-entry rate matrix.
    """
    if jump_indices.size == 0:
        return None, np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64)

    data_blocks: list[NDArray[np.complex128]] = []
    row_blocks: list[NDArray[np.int64]] = []
    column_blocks: list[NDArray[np.int64]] = []
    expanded_jump_indices: list[int] = []
    row_splits: list[int] = [0]
    expanded_row = 0

    for jump_index_raw in jump_indices:
        jump_index = int(jump_index_raw)
        rows_for_jump = 0
        for columns, values in zip(row_columns[jump_index], row_values[jump_index], strict=True):
            if columns.size == 0:
                continue

            data_blocks.append(values.astype(np.complex128, copy=False))
            column_blocks.append(columns.astype(np.int64, copy=False))
            row_blocks.append(np.full(columns.size, expanded_row, dtype=np.int64))
            expanded_row += 1
            rows_for_jump += 1

        if rows_for_jump > 0:
            expanded_jump_indices.append(jump_index)
            row_splits.append(expanded_row)

    if not data_blocks:
        return None, np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64)

    operator = scipy_sparse.csr_array(
        (
            np.concatenate(data_blocks),
            (np.concatenate(row_blocks), np.concatenate(column_blocks)),
        ),
        shape=(expanded_row, dim),
        dtype=np.complex128,
    )
    operator.sum_duplicates()
    operator.eliminate_zeros()

    return (
        operator,
        np.asarray(expanded_jump_indices, dtype=np.int64),
        np.asarray(row_splits, dtype=np.int64),
    )


@dataclass(frozen=True, slots=True)
class McwfOptions:
    """Options for Monte Carlo wave-function sampling."""

    backend: OpenSystemBackendName = "scipy"
    n_trajectories: int = 128
    seed: int | None = None
    return_backend_arrays: bool = False
    store_states: bool = True
    store_trajectories: bool = False
    normalize_each_step: bool = True
    max_jump_probability: float = 0.1
    prefer_sparse_operators: bool = True
    prefer_sparse_rate_evaluator: bool = True
    use_total_rate_first: bool = True
    use_event_driven_jumps: bool = False
    store_density_matrices: bool = True
    store_state_snapshots: bool = False
    trajectory_chunk_size: int | None = None
    trajectory_chunk_workers: int | None = None
    timing_collector: MutableMapping[str, float] | None = None

    adaptive_time_step: bool = False
    adaptive_safety_factor: float = 0.8
    min_step_size: float = 1.0e-12
    max_substeps_per_interval: int = 100_000

    def validate(self) -> None:
        """Validate MCWF time-step control options."""
        if self.n_trajectories <= 0:
            raise ValueError("options.n_trajectories must be positive.")

        if self.trajectory_chunk_size is not None and self.trajectory_chunk_size <= 0:
            raise ValueError("trajectory_chunk_size must be positive when set.")

        if self.trajectory_chunk_workers is not None and self.trajectory_chunk_workers <= 0:
            raise ValueError("trajectory_chunk_workers must be positive when set.")

        if self.max_jump_probability <= 0.0:
            raise ValueError("max_jump_probability must be positive.")

        if not (0.0 < self.adaptive_safety_factor <= 1.0):
            raise ValueError("adaptive_safety_factor must be in (0, 1].")

        if self.min_step_size <= 0.0:
            raise ValueError("min_step_size must be positive.")

        if self.max_substeps_per_interval <= 0:
            raise ValueError("max_substeps_per_interval must be positive.")

        if not self.adaptive_time_step and self.max_substeps_per_interval != 100_000:
            # Optional: probably do not need this check.
            pass


def _add_timing(
    timing_collector: MutableMapping[str, float] | None,
    stage: str,
    elapsed_seconds: float,
) -> None:
    if timing_collector is None:
        return

    timing_collector[stage] = float(timing_collector.get(stage, 0.0)) + float(elapsed_seconds)


def _perf_counter() -> float:
    return time.perf_counter()


@dataclass(frozen=True, slots=True)
class TrajectoryResult:
    """Single Monte Carlo wave-function trajectory."""

    times: NDArray[np.float64]
    states: list[Any]
    jump_times: NDArray[np.float64]
    jump_indices: NDArray[np.int64]
    norm_errors: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    """Ensemble-averaged MCWF result."""

    times: NDArray[np.float64]
    rho_t: list[ArrayC]
    trajectories: tuple[TrajectoryResult, ...] | None = None
    state_snapshots: tuple[ArrayC, ...] | None = None


def projector(state: Any) -> Any:
    """Return |state><state|."""
    state_column = state.reshape(-1, 1)
    return state_column @ state_column.conj().T


def expectation(state: Any, operator: Any) -> complex:
    """Return <state|operator|state>."""
    return complex(np.vdot(state, operator @ state))


def effective_hamiltonian(hamiltonian: Any, jumps: list[Any] | tuple[Any, ...]) -> Any:
    """Return H_eff = H - i/2 sum_mu J_mu^dagger J_mu."""
    total_rate_operator = _total_jump_rate_operator(
        tuple(jumps),
        shape=tuple(int(axis) for axis in hamiltonian.shape),
    )
    return _effective_hamiltonian_from_total_rate_operator(
        hamiltonian,
        total_rate_operator,
    )


def _effective_hamiltonian_from_total_rate_operator(
    hamiltonian: Any,
    total_rate_operator: Any | None,
) -> Any:
    """Return ``H - 0.5j * Gamma`` for ``Gamma=sum J^dagger J``."""
    effective = hamiltonian.copy()
    if total_rate_operator is None:
        return effective

    if scipy_sparse.issparse(effective) or scipy_sparse.issparse(total_rate_operator):
        effective_sparse = (
            effective.tocsr()
            if scipy_sparse.issparse(effective)
            else scipy_sparse.csr_array(effective, dtype=np.complex128)
        )
        total_rate_sparse = (
            total_rate_operator.tocsr()
            if scipy_sparse.issparse(total_rate_operator)
            else scipy_sparse.csr_array(total_rate_operator, dtype=np.complex128)
        )
        return (effective_sparse - 0.5j * total_rate_sparse).tocsr()

    return effective - 0.5j * total_rate_operator


def _total_jump_rate_operator(
    jumps: list[Any] | tuple[Any, ...],
    *,
    shape: tuple[int, int],
) -> Any | None:
    """Return ``Gamma=sum_mu J_mu^dagger J_mu`` or ``None`` with no jumps."""
    if not jumps:
        return None

    if all(scipy_sparse.issparse(jump) for jump in jumps):
        sparse_gram_sum = _sparse_jump_gram_sum_csr(jumps, shape=shape)
        if sparse_gram_sum is not None:
            return sparse_gram_sum

    total = jumps[0].conj().T @ jumps[0]
    for jump in jumps[1:]:
        total = total + (jump.conj().T @ jump)
    return total


def _sparse_jump_gram_sum_csr(
    jumps: list[Any] | tuple[Any, ...],
    *,
    shape: tuple[int, int],
    max_row_nnz: int = 32,
) -> scipy_sparse.csr_array | None:
    """Return ``sum_mu J_mu.conj().T @ J_mu`` for row-sparse jumps.

    This avoids doing many tiny sparse matrix multiplications and repeated sparse
    additions during MCWF operator preparation.  The row-wise construction is
    only used when every nonzero output row is sufficiently small; otherwise we
    fall back to SciPy's sparse multiplication, which is better for row-dense
    matrices.
    """
    if not jumps:
        return scipy_sparse.csr_array(shape, dtype=np.complex128)

    if shape[0] != shape[1]:
        return None

    dim = int(shape[0])
    row_blocks: list[NDArray[np.int64]] = []
    col_blocks: list[NDArray[np.int64]] = []
    value_blocks: list[NDArray[np.complex128]] = []

    for jump in jumps:
        if not scipy_sparse.issparse(jump):
            return None

        jump_csr = jump.tocsr().astype(np.complex128, copy=False)
        if jump_csr.shape != shape:
            return None

        indptr = jump_csr.indptr
        indices = jump_csr.indices
        data = jump_csr.data
        row_counts = np.diff(indptr)
        if row_counts.size and int(np.max(row_counts)) > max_row_nnz:
            return None

        active_rows = np.flatnonzero(row_counts > 0)
        for row in active_rows:
            start = int(indptr[row])
            stop = int(indptr[row + 1])
            columns = indices[start:stop].astype(np.int64, copy=False)
            values = data[start:stop]
            nnz = int(columns.size)
            if nnz == 0:
                continue

            row_blocks.append(np.repeat(columns, nnz))
            col_blocks.append(np.tile(columns, nnz))
            value_blocks.append((values.conj().reshape(nnz, 1) * values.reshape(1, nnz)).ravel())

    if not value_blocks:
        return scipy_sparse.csr_array(shape, dtype=np.complex128)

    rows = np.concatenate(row_blocks)
    columns = np.concatenate(col_blocks)
    values = np.concatenate(value_blocks).astype(np.complex128, copy=False)
    gram = scipy_sparse.csr_array((values, (rows, columns)), shape=(dim, dim), dtype=np.complex128)
    gram.sum_duplicates()
    gram.eliminate_zeros()
    return gram


def jump_probabilities(
    state: Any,
    jumps: list[Any] | tuple[Any, ...],
    step_size: float,
    *,
    backend: OpenSystemBackend,
) -> NDArray[np.float64]:
    """Return first-order jump probabilities dt <psi|J^dagger J|psi>."""
    return _evaluate_jumps(
        state,
        jumps,
        backend=backend,
    ).probabilities(step_size)


def _evaluate_jumps(
    state: Any,
    jumps: list[Any] | tuple[Any, ...],
    *,
    backend: OpenSystemBackend,
) -> _JumpEvaluation:
    """Return J_mu|psi> and rates ||J_mu|psi>||^2 for one MCWF state."""
    if not jumps:
        return _JumpEvaluation(
            jumped_states=(),
            rates=np.zeros(0, dtype=np.float64),
        )

    array_module = backend.array_module
    jumped_states: list[Any] = []
    rates: list[float] = []

    for jump in jumps:
        jumped_state = jump @ state
        rate = array_module.real(array_module.vdot(jumped_state, jumped_state))
        rate_float = float(backend.to_numpy(rate))
        jumped_states.append(jumped_state)
        rates.append(max(rate_float, 0.0))

    return _JumpEvaluation(
        jumped_states=tuple(jumped_states),
        rates=np.asarray(rates, dtype=np.float64),
    )


def choose_jump(
    probabilities: NDArray[np.float64],
    rng: np.random.Generator,
) -> int:
    """Choose a jump index according to normalized probabilities."""
    total_probability = float(np.sum(probabilities))

    if total_probability <= 0.0:
        raise ValueError("Total jump probability must be positive when choosing a jump.")

    if len(probabilities) == 1:
        return 0

    return int(
        rng.choice(
            len(probabilities),
            p=probabilities / total_probability,
        )
    )


def evolve_no_jump_first_order(
    state: Any,
    effective_hamiltonian_matrix: Any,
    step_size: float,
) -> Any:
    """First-order no-jump evolution under H_eff."""
    return state - 1j * step_size * (effective_hamiltonian_matrix @ state)


def run_quantum_jump_trajectory(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    state_initial: Any,
    times: NDArray[np.float64],
    rng: np.random.Generator | int | None = None,
    backend: OpenSystemBackendName = "scipy",
    return_backend_arrays: bool = False,
    store_states: bool = True,
    normalize_each_step: bool = True,
    max_jump_probability: float = 0.1,
    prefer_sparse_operators: bool = True,
    prefer_sparse_rate_evaluator: bool = True,
    use_total_rate_first: bool = True,
    adaptive_time_step: bool = False,
    adaptive_safety_factor: float = 0.8,
    min_step_size: float = 1.0e-12,
    max_substeps_per_interval: int = 100_000,
) -> TrajectoryResult:
    """Run one Monte Carlo wave-function trajectory.

    This uses a first-order no-jump propagator. It is therefore deliberately
    conservative: if the total jump probability in one time step is too large,
    it raises an error and asks the caller to refine the time grid.
    """
    prepared = _prepare_mcwf_operators(
        hamiltonian=hamiltonian,
        jumps=jumps,
        backend=backend,
        prefer_sparse_operators=prefer_sparse_operators,
        prefer_sparse_rate_evaluator=prefer_sparse_rate_evaluator,
    )
    return _run_quantum_jump_trajectory_prepared(
        prepared=prepared,
        state_initial=state_initial,
        times=times,
        rng=rng,
        return_backend_arrays=return_backend_arrays,
        store_states=store_states,
        normalize_each_step=normalize_each_step,
        max_jump_probability=max_jump_probability,
        use_total_rate_first=use_total_rate_first,
        adaptive_time_step=adaptive_time_step,
        adaptive_safety_factor=adaptive_safety_factor,
        min_step_size=min_step_size,
        max_substeps_per_interval=max_substeps_per_interval,
    )


def _run_quantum_jump_trajectory_prepared(
    *,
    prepared: _McwfPreparedOperators,
    state_initial: Any,
    times: NDArray[np.float64],
    rng: np.random.Generator | int | None = None,
    return_backend_arrays: bool = False,
    store_states: bool = True,
    state_callback: Callable[[int, Any], None] | None = None,
    normalize_each_step: bool = True,
    max_jump_probability: float = 0.1,
    use_total_rate_first: bool = True,
    adaptive_time_step: bool = False,
    adaptive_safety_factor: float = 0.8,
    min_step_size: float = 1.0e-12,
    max_substeps_per_interval: int = 100_000,
) -> TrajectoryResult:
    times = np.asarray(times, dtype=np.float64)
    _validate_times_for_mcwf(times)

    generator = _rng_from_seed(rng)
    backend_obj = prepared.backend

    if backend_obj.name == "scipy":
        return _run_quantum_jump_trajectory_prepared_scipy(
            prepared=prepared,
            state_initial=state_initial,
            times=times,
            rng=generator,
            return_backend_arrays=return_backend_arrays,
            store_states=store_states,
            state_callback=state_callback,
            normalize_each_step=normalize_each_step,
            max_jump_probability=max_jump_probability,
            use_total_rate_first=use_total_rate_first,
            adaptive_time_step=adaptive_time_step,
            adaptive_safety_factor=adaptive_safety_factor,
            min_step_size=min_step_size,
            max_substeps_per_interval=max_substeps_per_interval,
        )
    jump_operators = prepared.jumps
    has_jump_operators = len(jump_operators) > 0

    state = _normalize_backend_state(
        backend_obj.asarray(state_initial, dtype=np.complex128),
        backend=backend_obj,
    )
    effective_hamiltonian_matrix = prepared.effective_hamiltonian_matrix

    states: list[Any] = []
    if store_states:
        states.append(
            _maybe_to_numpy(state, backend=backend_obj, enabled=not return_backend_arrays)
        )

    if state_callback is not None:
        state_callback(0, state)

    jump_times: list[float] = []
    jump_indices: list[int] = []
    norm_errors: list[float] = []

    for time_index in range(times.size - 1):
        interval_start = float(times[time_index])
        interval_stop = float(times[time_index + 1])
        current_time = interval_start
        substeps = 0

        while current_time < interval_stop:
            remaining_step = interval_stop - current_time
            step_size = remaining_step

            if has_jump_operators:
                jump_evaluation = _evaluate_jumps(
                    state,
                    jump_operators,
                    backend=backend_obj,
                )
                probabilities = jump_evaluation.probabilities(step_size)
                total_jump_probability = float(np.sum(probabilities))
            else:
                jump_evaluation = None
                probabilities = np.zeros(0, dtype=np.float64)
                total_jump_probability = 0.0

            if total_jump_probability > max_jump_probability:
                if not adaptive_time_step:
                    raise RuntimeError(
                        "Time step is too large for first-order MCWF: "
                        f"total jump probability={total_jump_probability:.6e}, "
                        f"allowed maximum={max_jump_probability:.6e}. "
                        "Use a finer time grid, enable adaptive_time_step=True, "
                        "or increase max_jump_probability only if you know this is "
                        "acceptable."
                    )

                scale = adaptive_safety_factor * max_jump_probability / total_jump_probability
                step_size = max(remaining_step * scale, min_step_size)

                if step_size >= remaining_step:
                    # Should only happen from roundoff, but keep the branch safe.
                    step_size = remaining_step

                assert jump_evaluation is not None
                probabilities = jump_evaluation.probabilities(step_size)
                total_jump_probability = float(np.sum(probabilities))

                if total_jump_probability > max_jump_probability:
                    raise RuntimeError(
                        "Adaptive MCWF failed to reduce the step enough: "
                        f"total jump probability={total_jump_probability:.6e}, "
                        f"allowed maximum={max_jump_probability:.6e}. "
                        "Try a smaller min_step_size or max_jump_probability."
                    )

            if step_size < min_step_size and remaining_step > min_step_size:
                raise RuntimeError(
                    "Adaptive MCWF reached min_step_size before completing "
                    "the requested time interval."
                )

            if jump_evaluation is not None and generator.random() < total_jump_probability:
                jump_index = choose_jump(probabilities, generator)
                state = jump_evaluation.jumped_states[jump_index]
                jump_times.append(float(current_time + step_size))
                jump_indices.append(jump_index)
            else:
                state = evolve_no_jump_first_order(
                    state,
                    effective_hamiltonian_matrix,
                    step_size,
                )

            state_norm = _backend_state_norm(state, backend=backend_obj)
            norm_errors.append(abs(state_norm - 1.0))

            if normalize_each_step:
                if state_norm == 0.0:
                    raise RuntimeError(
                        "The MCWF state reached zero norm. " "Try a smaller time step."
                    )
                state = state / state_norm

            current_time += step_size
            substeps += 1

            if substeps > max_substeps_per_interval:
                raise RuntimeError(
                    "Adaptive MCWF exceeded max_substeps_per_interval. "
                    "Try increasing max_jump_probability, increasing "
                    "max_substeps_per_interval, or checking jump rates."
                )

        if store_states:
            states.append(
                _maybe_to_numpy(
                    state,
                    backend=backend_obj,
                    enabled=not return_backend_arrays,
                )
            )

        if state_callback is not None:
            state_callback(time_index + 1, state)

    return TrajectoryResult(
        times=times,
        states=states,
        jump_times=np.asarray(jump_times, dtype=np.float64),
        jump_indices=np.asarray(jump_indices, dtype=np.int64),
        norm_errors=np.asarray(norm_errors, dtype=np.float64),
    )


def _run_quantum_jump_trajectory_prepared_scipy(
    *,
    prepared: _McwfPreparedOperators,
    state_initial: Any,
    times: NDArray[np.float64],
    rng: np.random.Generator,
    return_backend_arrays: bool = False,
    store_states: bool = True,
    state_callback: Callable[[int, Any], None] | None = None,
    normalize_each_step: bool = True,
    max_jump_probability: float = 0.1,
    use_total_rate_first: bool = True,
    adaptive_time_step: bool = False,
    adaptive_safety_factor: float = 0.8,
    min_step_size: float = 1.0e-12,
    max_substeps_per_interval: int = 100_000,
) -> TrajectoryResult:
    """Run one trajectory with a NumPy-specialized inner loop."""
    del return_backend_arrays  # SciPy backend states are already NumPy arrays.

    state = _normalize_numpy_state(np.asarray(state_initial, dtype=np.complex128))
    effective_hamiltonian_matrix = _as_numpy_or_scipy_sparse(prepared.effective_hamiltonian_matrix)
    total_jump_rate_operator = (
        _as_numpy_or_scipy_sparse(prepared.total_jump_rate_operator)
        if prepared.total_jump_rate_operator is not None
        else None
    )
    jump_operators = tuple(_as_numpy_or_scipy_sparse(jump) for jump in prepared.jumps)
    sparse_jump_rate_evaluator = prepared.sparse_jump_rate_evaluator
    has_jump_operators = len(jump_operators) > 0

    states: list[Any] = [None] * int(times.size) if store_states else []
    if store_states:
        states[0] = state.copy()

    if state_callback is not None:
        state_callback(0, state)

    jump_times: list[float] = []
    jump_indices: list[int] = []
    norm_errors: list[float] = []

    for time_index in range(times.size - 1):
        interval_start = float(times[time_index])
        interval_stop = float(times[time_index + 1])
        current_time = interval_start
        substeps = 0

        while current_time < interval_stop:
            remaining_step = interval_stop - current_time
            step_size = remaining_step

            jumped_states: tuple[ArrayC, ...] = ()
            jumped_state_single: ArrayC | None = None
            probabilities = np.zeros(0, dtype=np.float64)
            total_jump_probability = 0.0

            if has_jump_operators:
                if _should_use_total_rate_first(
                    use_total_rate_first,
                    total_jump_rate_operator=total_jump_rate_operator,
                    sparse_jump_rate_evaluator=sparse_jump_rate_evaluator,
                ):
                    total_jump_probability = step_size * _evaluate_total_jump_rate_numpy(
                        state,
                        total_jump_rate_operator,
                    )
                elif sparse_jump_rate_evaluator is not None:
                    rates = _evaluate_sparse_jump_rates_numpy(
                        state,
                        sparse_jump_rate_evaluator,
                    )
                    probabilities = step_size * rates
                    total_jump_probability = float(np.sum(probabilities))
                elif len(jump_operators) == 1:
                    jumped_state_single = jump_operators[0] @ state
                    rate = max(float(np.vdot(jumped_state_single, jumped_state_single).real), 0.0)
                    total_jump_probability = step_size * rate
                else:
                    jumped_states, rates = _evaluate_jumps_numpy(state, jump_operators)
                    probabilities = step_size * rates
                    total_jump_probability = float(np.sum(probabilities))

                if total_jump_probability > max_jump_probability:
                    if not adaptive_time_step:
                        raise RuntimeError(
                            "Time step is too large for first-order MCWF: "
                            f"total jump probability={total_jump_probability:.6e}, "
                            f"allowed maximum={max_jump_probability:.6e}. "
                            "Use a finer time grid, enable adaptive_time_step=True, "
                            "or increase max_jump_probability only if you know this is "
                            "acceptable."
                        )

                    scale = adaptive_safety_factor * max_jump_probability / total_jump_probability
                    step_size = max(remaining_step * scale, min_step_size)

                    if step_size >= remaining_step:
                        step_size = remaining_step

                    if _should_use_total_rate_first(
                        use_total_rate_first,
                        total_jump_rate_operator=total_jump_rate_operator,
                        sparse_jump_rate_evaluator=sparse_jump_rate_evaluator,
                    ):
                        total_jump_probability = step_size * _evaluate_total_jump_rate_numpy(
                            state,
                            total_jump_rate_operator,
                        )
                    elif len(jump_operators) == 1 and sparse_jump_rate_evaluator is None:
                        total_jump_probability = step_size * rate
                    else:
                        probabilities = step_size * rates
                        total_jump_probability = float(np.sum(probabilities))

                    if total_jump_probability > max_jump_probability:
                        raise RuntimeError(
                            "Adaptive MCWF failed to reduce the step enough: "
                            f"total jump probability={total_jump_probability:.6e}, "
                            f"allowed maximum={max_jump_probability:.6e}. "
                            "Try a smaller min_step_size or max_jump_probability."
                        )

            if step_size < min_step_size and remaining_step > min_step_size:
                raise RuntimeError(
                    "Adaptive MCWF reached min_step_size before completing "
                    "the requested time interval."
                )

            if has_jump_operators and rng.random() < total_jump_probability:
                if len(jump_operators) == 1:
                    jump_index = 0
                    if jumped_state_single is None:
                        jumped_state_single = jump_operators[0] @ state
                    state = jumped_state_single
                else:
                    if probabilities.size == 0:
                        if sparse_jump_rate_evaluator is not None:
                            rates = _evaluate_sparse_jump_rates_numpy(
                                state,
                                sparse_jump_rate_evaluator,
                            )
                        else:
                            jumped_states, rates = _evaluate_jumps_numpy(state, jump_operators)
                        probabilities = step_size * rates
                    jump_index = choose_jump(probabilities, rng)
                    if sparse_jump_rate_evaluator is not None or not jumped_states:
                        state = jump_operators[jump_index] @ state
                    else:
                        state = jumped_states[jump_index]
                jump_times.append(float(current_time + step_size))
                jump_indices.append(jump_index)
            else:
                state = state - 1j * step_size * (effective_hamiltonian_matrix @ state)

            state_norm = float(np.linalg.norm(state))
            norm_errors.append(abs(state_norm - 1.0))

            if normalize_each_step:
                if state_norm == 0.0:
                    raise RuntimeError("The MCWF state reached zero norm. Try a smaller time step.")
                state = state / state_norm

            current_time += step_size
            substeps += 1

            if substeps > max_substeps_per_interval:
                raise RuntimeError(
                    "Adaptive MCWF exceeded max_substeps_per_interval. "
                    "Try increasing max_jump_probability, increasing "
                    "max_substeps_per_interval, or checking jump rates."
                )

        if store_states:
            states[time_index + 1] = state.copy()

        if state_callback is not None:
            state_callback(time_index + 1, state)

    return TrajectoryResult(
        times=times,
        states=states,
        jump_times=np.asarray(jump_times, dtype=np.float64),
        jump_indices=np.asarray(jump_indices, dtype=np.int64),
        norm_errors=np.asarray(norm_errors, dtype=np.float64),
    )


def _evaluate_jumps_numpy(
    state: ArrayC,
    jumps: tuple[ArrayC, ...],
) -> tuple[tuple[ArrayC, ...], NDArray[np.float64]]:
    """Return J_mu|psi> and rates for one NumPy MCWF state."""
    if not jumps:
        return (), np.zeros(0, dtype=np.float64)

    jumped_states = tuple(jump @ state for jump in jumps)
    rates = np.fromiter(
        (
            max(float(np.vdot(jumped_state, jumped_state).real), 0.0)
            for jumped_state in jumped_states
        ),
        dtype=np.float64,
        count=len(jumped_states),
    )
    return jumped_states, rates


def _can_use_vectorized_mcwf_ensemble(
    *,
    options: McwfOptions,
    backend: OpenSystemBackend,
) -> bool:
    """Return whether the ensemble can use the NumPy vectorized MCWF path."""
    return (
        backend.name == "scipy" and not options.store_trajectories and options.normalize_each_step
    )


def _sample_lindblad_mcwf_vectorized_scipy(
    *,
    prepared: _McwfPreparedOperators,
    dim: int,
    times: NDArray[np.float64],
    state_initial: Any | None,
    state_sampler: StateSampler | None,
    options: McwfOptions,
    rng: np.random.Generator,
) -> EnsembleResult:
    """Sample an MCWF ensemble with trajectory states stored as columns."""
    timing_collector = options.timing_collector

    start = _perf_counter()
    states = _initial_state_matrix_for_ensemble(
        dim=dim,
        n_trajectories=options.n_trajectories,
        state_initial=state_initial,
        state_sampler=state_sampler,
        rng=rng,
    )
    _add_timing(timing_collector, "mcwf.initial_state_matrix", _perf_counter() - start)

    start = _perf_counter()
    effective_hamiltonian_matrix = _as_numpy_or_scipy_sparse(prepared.effective_hamiltonian_matrix)
    total_jump_rate_operator = (
        _as_numpy_or_scipy_sparse(prepared.total_jump_rate_operator)
        if prepared.total_jump_rate_operator is not None
        else None
    )
    jump_operators = tuple(_as_numpy_or_scipy_sparse(jump) for jump in prepared.jumps)
    sparse_jump_rate_evaluator = prepared.sparse_jump_rate_evaluator
    _add_timing(timing_collector, "mcwf.operator_view_conversion", _perf_counter() - start)

    rho_t: list[ArrayC] = []
    if options.store_density_matrices:
        start = _perf_counter()
        rho_t.append(_density_matrix_from_state_matrix(states))
        _add_timing(timing_collector, "mcwf.density_accumulation", _perf_counter() - start)

    state_snapshots: list[ArrayC] | None = None
    if options.store_state_snapshots:
        state_snapshots = [states.copy()]

    event_survival_thresholds: NDArray[np.float64] | None = None
    if options.use_event_driven_jumps:
        event_survival_thresholds = _draw_mcwf_survival_thresholds(
            rng,
            options.n_trajectories,
        )

    for time_index in range(times.size - 1):
        interval_start = float(times[time_index])
        interval_stop = float(times[time_index + 1])
        current_time = interval_start
        substeps = 0

        if options.use_event_driven_jumps:
            assert event_survival_thresholds is not None
            states, event_survival_thresholds = _advance_state_matrix_event_driven_numpy(
                states=states,
                survival_thresholds=event_survival_thresholds,
                step_size=interval_stop - interval_start,
                effective_hamiltonian_matrix=effective_hamiltonian_matrix,
                jump_operators=jump_operators,
                sparse_jump_rate_evaluator=sparse_jump_rate_evaluator,
                rng=rng,
                max_substeps=options.max_substeps_per_interval,
                min_step_size=options.min_step_size,
                max_jump_probability=options.max_jump_probability,
                adaptive_safety_factor=options.adaptive_safety_factor,
                timing_collector=timing_collector,
            )

            if options.store_state_snapshots:
                assert state_snapshots is not None
                state_snapshots.append(states.copy())

            if options.store_density_matrices:
                start = _perf_counter()
                rho_t.append(_density_matrix_from_state_matrix(states))
                _add_timing(
                    timing_collector,
                    "mcwf.density_accumulation",
                    _perf_counter() - start,
                )

            continue

        while current_time < interval_stop:
            remaining_step = interval_stop - current_time
            step_size = remaining_step

            probabilities = np.zeros((0, options.n_trajectories), dtype=np.float64)
            total_jump_probabilities = np.zeros(options.n_trajectories, dtype=np.float64)
            rates: NDArray[np.float64] | None = None

            if jump_operators:
                (
                    probabilities,
                    total_jump_probabilities,
                    rates,
                ) = _evaluate_vectorized_jump_probabilities_numpy(
                    states=states,
                    step_size=step_size,
                    jump_operators=jump_operators,
                    sparse_jump_rate_evaluator=sparse_jump_rate_evaluator,
                    total_jump_rate_operator=total_jump_rate_operator,
                    use_total_rate_first=options.use_total_rate_first,
                    timing_collector=timing_collector,
                )

                max_total_jump_probability = float(np.max(total_jump_probabilities))
                if max_total_jump_probability > options.max_jump_probability:
                    if not options.adaptive_time_step:
                        raise RuntimeError(
                            "Time step is too large for first-order MCWF: "
                            f"total jump probability={max_total_jump_probability:.6e}, "
                            f"allowed maximum={options.max_jump_probability:.6e}. "
                            "Use a finer time grid, enable adaptive_time_step=True, "
                            "or increase max_jump_probability only if you know this is "
                            "acceptable."
                        )

                    scale = (
                        options.adaptive_safety_factor
                        * options.max_jump_probability
                        / max_total_jump_probability
                    )
                    step_size = max(remaining_step * scale, options.min_step_size)
                    if step_size >= remaining_step:
                        step_size = remaining_step

                    (
                        probabilities,
                        total_jump_probabilities,
                        rates,
                    ) = _evaluate_vectorized_jump_probabilities_numpy(
                        states=states,
                        step_size=step_size,
                        jump_operators=jump_operators,
                        sparse_jump_rate_evaluator=sparse_jump_rate_evaluator,
                        total_jump_rate_operator=total_jump_rate_operator,
                        use_total_rate_first=options.use_total_rate_first,
                        timing_collector=timing_collector,
                    )
                    max_total_jump_probability = float(np.max(total_jump_probabilities))
                    if max_total_jump_probability > options.max_jump_probability:
                        raise RuntimeError(
                            "Adaptive MCWF failed to reduce the step enough: "
                            f"total jump probability={max_total_jump_probability:.6e}, "
                            f"allowed maximum={options.max_jump_probability:.6e}. "
                            "Try a smaller min_step_size or max_jump_probability."
                        )

            if step_size < options.min_step_size and remaining_step > options.min_step_size:
                raise RuntimeError(
                    "Adaptive MCWF reached min_step_size before completing "
                    "the requested time interval."
                )

            start = _perf_counter()
            next_states = states - 1j * step_size * (effective_hamiltonian_matrix @ states)
            _add_timing(timing_collector, "mcwf.no_jump_propagation", _perf_counter() - start)

            if jump_operators:
                start = _perf_counter()
                jump_mask = rng.random(options.n_trajectories) < total_jump_probabilities
                if np.any(jump_mask):
                    if rates is None:
                        channel_start = _perf_counter()
                        if sparse_jump_rate_evaluator is not None:
                            selected_rates = _evaluate_sparse_jump_rates_state_matrix_numpy(
                                states[:, jump_mask],
                                sparse_jump_rate_evaluator,
                            )
                        else:
                            selected_rates = _evaluate_jump_rates_state_matrix_numpy(
                                states[:, jump_mask],
                                jump_operators,
                            )
                        selected_probabilities = step_size * selected_rates
                        selected_total_probabilities = np.sum(selected_probabilities, axis=0)
                        channel_elapsed = _perf_counter() - channel_start
                        _add_timing(
                            timing_collector,
                            "mcwf.channel_rate_evaluation",
                            channel_elapsed,
                        )
                        _add_timing(timing_collector, "mcwf.rate_evaluation", channel_elapsed)
                    else:
                        selected_probabilities = probabilities[:, jump_mask]
                        selected_total_probabilities = total_jump_probabilities[jump_mask]
                    selected_for_active = _choose_jump_indices_vectorized(
                        probabilities=selected_probabilities,
                        total_probabilities=selected_total_probabilities,
                        rng=rng,
                    )
                    selected_jump_indices = np.zeros(options.n_trajectories, dtype=np.int64)
                    selected_jump_indices[jump_mask] = selected_for_active
                    _add_timing(timing_collector, "mcwf.jump_selection", _perf_counter() - start)

                    start = _perf_counter()
                    _apply_selected_jumps_to_state_matrix_numpy(
                        next_states=next_states,
                        states=states,
                        jumps=jump_operators,
                        jump_mask=jump_mask,
                        selected_jump_indices=selected_jump_indices,
                    )
                    _add_timing(
                        timing_collector,
                        "mcwf.selected_jump_application",
                        _perf_counter() - start,
                    )
                else:
                    _add_timing(timing_collector, "mcwf.jump_selection", _perf_counter() - start)

            start = _perf_counter()
            norms = np.linalg.norm(next_states, axis=0)
            if np.any(norms == 0.0):
                raise RuntimeError(
                    "At least one MCWF state reached zero norm. " "Try a smaller time step."
                )

            states = next_states / norms.reshape(1, -1)
            _add_timing(timing_collector, "mcwf.normalization", _perf_counter() - start)

            current_time += step_size
            substeps += 1
            if substeps > options.max_substeps_per_interval:
                raise RuntimeError(
                    "Adaptive MCWF exceeded max_substeps_per_interval. "
                    "Try increasing max_jump_probability, increasing "
                    "max_substeps_per_interval, or checking jump rates."
                )

        if options.store_state_snapshots:
            assert state_snapshots is not None
            state_snapshots.append(states.copy())

        if options.store_density_matrices:
            start = _perf_counter()
            rho_t.append(_density_matrix_from_state_matrix(states))
            _add_timing(timing_collector, "mcwf.density_accumulation", _perf_counter() - start)

    return EnsembleResult(
        times=times,
        rho_t=rho_t,
        trajectories=None,
        state_snapshots=(tuple(state_snapshots) if state_snapshots is not None else None),
    )


def _draw_mcwf_survival_thresholds(
    rng: np.random.Generator,
    size: int,
) -> NDArray[np.float64]:
    """Draw survival thresholds for norm-threshold MCWF jumps."""
    thresholds = rng.random(size)
    np.maximum(thresholds, np.finfo(np.float64).tiny, out=thresholds)
    return thresholds.astype(np.float64, copy=False)


def _normalize_state_matrix_columns_numpy(states: ArrayC) -> tuple[ArrayC, NDArray[np.float64]]:
    """Normalize state columns and return the original column norms."""
    norms = np.linalg.norm(states, axis=0)
    if np.any(norms == 0.0):
        raise RuntimeError("At least one MCWF state reached zero norm.")

    return states / norms.reshape(1, -1), norms.astype(np.float64, copy=False)


def _first_order_survival_crossing_times(
    *,
    states: ArrayC,
    derivatives: ArrayC,
    thresholds: NDArray[np.float64],
    max_times: NDArray[np.float64],
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """Locate first-order no-jump norm-threshold crossings.

    Starting from normalized columns ``psi`` and first-order no-jump
    derivatives ``dpsi/dt``, the unnormalized survival over one segment is

        ||psi + t dpsi/dt||^2 = 1 + b t + a t^2.

    The returned crossing time is the first nonnegative root of
    ``1 + b t + a t^2 = threshold`` that lies before ``max_times``.  The
    returned instantaneous total jump rates are ``-b`` clipped at zero.
    """
    thresholds = np.asarray(thresholds, dtype=np.float64)
    max_times = np.asarray(max_times, dtype=np.float64)

    quadratic_coefficients = np.einsum(
        "ij,ij->j",
        derivatives.conj(),
        derivatives,
        optimize=True,
    ).real
    linear_coefficients = (
        2.0
        * np.einsum(
            "ij,ij->j",
            states.conj(),
            derivatives,
            optimize=True,
        ).real
    )
    constant_terms = 1.0 - thresholds
    total_rates = np.maximum(-linear_coefficients, 0.0)

    crossing_times = np.full(thresholds.size, np.inf, dtype=np.float64)

    linear_only_mask = np.abs(quadratic_coefficients) <= np.finfo(np.float64).eps
    valid_linear_mask = linear_only_mask & (linear_coefficients < 0.0)
    crossing_times[valid_linear_mask] = (
        -constant_terms[valid_linear_mask] / linear_coefficients[valid_linear_mask]
    )

    quadratic_mask = ~linear_only_mask
    if np.any(quadratic_mask):
        quadratic_values = quadratic_coefficients[quadratic_mask]
        linear_values = linear_coefficients[quadratic_mask]
        constant_values = constant_terms[quadratic_mask]
        discriminants = linear_values * linear_values - 4.0 * quadratic_values * constant_values
        valid_discriminant_mask = discriminants >= 0.0
        candidate_roots = np.full(quadratic_values.size, np.inf, dtype=np.float64)
        if np.any(valid_discriminant_mask):
            square_roots = np.sqrt(np.maximum(discriminants[valid_discriminant_mask], 0.0))
            valid_quadratic_values = quadratic_values[valid_discriminant_mask]
            valid_linear_values = linear_values[valid_discriminant_mask]
            lower_roots = (-valid_linear_values - square_roots) / (2.0 * valid_quadratic_values)
            upper_roots = (-valid_linear_values + square_roots) / (2.0 * valid_quadratic_values)
            positive_lower_mask = lower_roots >= 0.0
            positive_upper_mask = upper_roots >= 0.0
            valid_roots = np.full(lower_roots.size, np.inf, dtype=np.float64)
            valid_roots[positive_lower_mask] = lower_roots[positive_lower_mask]
            valid_roots[positive_upper_mask] = np.minimum(
                valid_roots[positive_upper_mask],
                upper_roots[positive_upper_mask],
            )
            candidate_roots[valid_discriminant_mask] = valid_roots

        crossing_times[quadratic_mask] = candidate_roots

    crossing_mask = (
        np.isfinite(crossing_times) & (crossing_times >= 0.0) & (crossing_times <= max_times)
    )
    crossing_times = np.minimum(crossing_times, max_times)
    np.maximum(crossing_times, 0.0, out=crossing_times)
    return crossing_mask, crossing_times, total_rates


def _event_driven_segment_limits(
    *,
    remaining_times: NDArray[np.float64],
    total_rates: NDArray[np.float64],
    derivative_norms: NDArray[np.float64],
    max_jump_probability: float,
    adaptive_safety_factor: float,
) -> NDArray[np.float64]:
    """Choose stable first-order no-jump segment lengths.

    The norm-threshold MCWF event rule is correct only for the no-jump
    propagator used to evaluate the survival norm.  A single large first-order
    Euler step can make ``||psi_tilde||^2`` increase because of its quadratic
    term, which suppresses norm crossings.  This helper caps each event segment
    using both the instantaneous total jump rate and the full no-jump derivative
    norm.  The derivative cap is important for states with zero instantaneous
    jump rate that are rotated into jump-active subspaces by the Hamiltonian.
    """
    remaining_times = np.asarray(remaining_times, dtype=np.float64)
    total_rates = np.asarray(total_rates, dtype=np.float64)
    derivative_norms = np.asarray(derivative_norms, dtype=np.float64)
    segment_limits = remaining_times.copy()
    probability_cap = max_jump_probability * adaptive_safety_factor

    positive_rate_mask = total_rates > 0.0
    if np.any(positive_rate_mask):
        rate_limited_steps = probability_cap / total_rates[positive_rate_mask]
        segment_limits[positive_rate_mask] = np.minimum(
            segment_limits[positive_rate_mask],
            rate_limited_steps,
        )

    positive_derivative_mask = derivative_norms > 0.0
    if np.any(positive_derivative_mask):
        derivative_limited_steps = (
            np.sqrt(probability_cap) / derivative_norms[positive_derivative_mask]
        )
        segment_limits[positive_derivative_mask] = np.minimum(
            segment_limits[positive_derivative_mask],
            derivative_limited_steps,
        )

    np.maximum(segment_limits, 0.0, out=segment_limits)
    return segment_limits


def _advance_state_matrix_event_driven_numpy(
    *,
    states: ArrayC,
    survival_thresholds: NDArray[np.float64],
    step_size: float,
    effective_hamiltonian_matrix: Any,
    jump_operators: tuple[Any, ...],
    sparse_jump_rate_evaluator: _SparseJumpRateEvaluator | None,
    rng: np.random.Generator,
    max_substeps: int,
    min_step_size: float,
    max_jump_probability: float,
    adaptive_safety_factor: float,
    timing_collector: MutableMapping[str, float] | None,
) -> tuple[ArrayC, NDArray[np.float64]]:
    """Advance a vectorized MCWF ensemble with norm-threshold jump times.

    This is the standard MCWF waiting-time construction adapted to the
    existing first-order no-jump propagator.  Each trajectory carries a
    survival threshold ``r`` across output intervals.  A jump occurs when the
    unnormalized no-jump norm first satisfies ``||psi_tilde(t)||^2 <= r``.
    If an output time is reached before the jump, the state is normalized and
    the residual threshold is rescaled by the survival probability reached in
    that interval.

    This replaces the earlier piecewise-constant hazard approximation
    ``tau = -log(u) / <Gamma>`` and avoids redrawing waiting times merely
    because an output sample was requested.
    """
    if step_size < 0.0:
        raise ValueError("step_size must be nonnegative.")

    if survival_thresholds.shape != (states.shape[1],):
        raise ValueError("survival_thresholds must have one entry per trajectory.")

    survival_thresholds = np.asarray(survival_thresholds, dtype=np.float64).copy()
    np.clip(
        survival_thresholds,
        np.finfo(np.float64).tiny,
        np.nextafter(1.0, 0.0),
        out=survival_thresholds,
    )

    if step_size == 0.0:
        return states, survival_thresholds

    if not jump_operators:
        start = _perf_counter()
        next_states = states - 1j * step_size * (effective_hamiltonian_matrix @ states)
        _add_timing(timing_collector, "mcwf.no_jump_propagation", _perf_counter() - start)

        start = _perf_counter()
        next_states, _ = _normalize_state_matrix_columns_numpy(next_states)
        _add_timing(timing_collector, "mcwf.normalization", _perf_counter() - start)
        return next_states, survival_thresholds

    remaining = np.full(states.shape[1], float(step_size), dtype=np.float64)
    active_indices = np.arange(states.shape[1], dtype=np.int64)
    event_substeps = 0

    while active_indices.size:
        if event_substeps > max_substeps:
            raise RuntimeError(
                "Event-driven MCWF exceeded max_substeps_per_interval. "
                "Try increasing max_substeps_per_interval or checking jump rates."
            )

        active_states = states[:, active_indices]
        active_remaining = remaining[active_indices]
        active_thresholds = survival_thresholds[active_indices]

        start = _perf_counter()
        derivatives = -1j * (effective_hamiltonian_matrix @ active_states)
        _, _, total_rates = _first_order_survival_crossing_times(
            states=active_states,
            derivatives=derivatives,
            thresholds=active_thresholds,
            max_times=active_remaining,
        )
        derivative_norms = np.linalg.norm(derivatives, axis=0)
        segment_limits = _event_driven_segment_limits(
            remaining_times=active_remaining,
            total_rates=total_rates,
            derivative_norms=derivative_norms,
            max_jump_probability=max_jump_probability,
            adaptive_safety_factor=adaptive_safety_factor,
        )
        crossing_mask, crossing_times, _ = _first_order_survival_crossing_times(
            states=active_states,
            derivatives=derivatives,
            thresholds=active_thresholds,
            max_times=segment_limits,
        )
        segment_steps = np.where(crossing_mask, crossing_times, segment_limits)
        next_active_states = active_states + derivatives * segment_steps.reshape(1, -1)
        _add_timing(timing_collector, "mcwf.no_jump_propagation", _perf_counter() - start)

        if np.any((segment_steps < min_step_size) & (active_remaining > min_step_size)):
            raise RuntimeError(
                "Event-driven MCWF reached min_step_size before completing "
                "the requested time interval."
            )

        start = _perf_counter()
        norm_squares = np.einsum(
            "ij,ij->j",
            next_active_states.conj(),
            next_active_states,
            optimize=True,
        ).real
        norm_squares = np.asarray(norm_squares, dtype=np.float64)
        if np.any(norm_squares <= 0.0):
            raise RuntimeError("At least one MCWF state reached zero norm.")
        normalized_active_states = next_active_states / np.sqrt(norm_squares).reshape(1, -1)
        _add_timing(timing_collector, "mcwf.normalization", _perf_counter() - start)

        if np.any(crossing_mask):
            event_columns = np.flatnonzero(crossing_mask)
            event_source_states = normalized_active_states[:, event_columns]

            channel_start = _perf_counter()
            if sparse_jump_rate_evaluator is not None:
                event_rates = _evaluate_sparse_jump_rates_state_matrix_numpy(
                    event_source_states,
                    sparse_jump_rate_evaluator,
                )
            else:
                event_rates = _evaluate_jump_rates_state_matrix_numpy(
                    event_source_states,
                    jump_operators,
                )
            event_total_rates = np.sum(event_rates, axis=0)
            if np.any(event_total_rates <= 0.0):
                raise RuntimeError(
                    "A norm-threshold MCWF event was detected with zero channel rate."
                )
            elapsed = _perf_counter() - channel_start
            _add_timing(timing_collector, "mcwf.channel_rate_evaluation", elapsed)
            _add_timing(timing_collector, "mcwf.rate_evaluation", elapsed)

            jump_start = _perf_counter()
            selected_for_events = _choose_jump_indices_vectorized(
                probabilities=event_rates,
                total_probabilities=event_total_rates,
                rng=rng,
            )
            selected_jump_indices = np.zeros(active_indices.size, dtype=np.int64)
            selected_jump_indices[event_columns] = selected_for_events
            _add_timing(timing_collector, "mcwf.jump_selection", _perf_counter() - jump_start)

            start = _perf_counter()
            _apply_selected_jumps_to_state_matrix_numpy(
                next_states=normalized_active_states,
                states=normalized_active_states.copy(),
                jumps=jump_operators,
                jump_mask=crossing_mask,
                selected_jump_indices=selected_jump_indices,
            )
            _add_timing(
                timing_collector,
                "mcwf.selected_jump_application",
                _perf_counter() - start,
            )

            start = _perf_counter()
            normalized_active_states, _ = _normalize_state_matrix_columns_numpy(
                normalized_active_states,
            )
            _add_timing(timing_collector, "mcwf.normalization", _perf_counter() - start)

            survival_thresholds[active_indices[crossing_mask]] = _draw_mcwf_survival_thresholds(
                rng,
                int(np.count_nonzero(crossing_mask)),
            )

        no_event_mask = ~crossing_mask
        if np.any(no_event_mask):
            residual_thresholds = active_thresholds[no_event_mask] / norm_squares[no_event_mask]
            np.clip(
                residual_thresholds,
                np.finfo(np.float64).tiny,
                np.nextafter(1.0, 0.0),
                out=residual_thresholds,
            )
            survival_thresholds[active_indices[no_event_mask]] = residual_thresholds

        states[:, active_indices] = normalized_active_states
        remaining[active_indices] -= segment_steps
        active_indices = active_indices[remaining[active_indices] > min_step_size]
        event_substeps += 1

    return states, survival_thresholds


def _sample_lindblad_mcwf_chunked_vectorized_scipy(
    *,
    prepared: _McwfPreparedOperators,
    dim: int,
    times: NDArray[np.float64],
    state_initial: Any | None,
    state_sampler: StateSampler | None,
    options: McwfOptions,
    rng: np.random.Generator,
) -> EnsembleResult:
    """Sample a vectorized MCWF ensemble in independent trajectory chunks.

    This keeps the per-chunk state matrix small while preserving the public
    ensemble result.  Density matrices are merged as weighted averages.  State
    snapshots, when requested, are concatenated column-wise in trajectory order.
    If ``options.trajectory_chunk_workers`` is greater than one, chunks are run
    in worker processes and merged deterministically in trajectory order.
    """
    chunk_size = _effective_trajectory_chunk_size(options)
    if chunk_size is None:
        return _sample_lindblad_mcwf_vectorized_scipy(
            prepared=prepared,
            dim=dim,
            times=times,
            state_initial=state_initial,
            state_sampler=state_sampler,
            options=options,
            rng=rng,
        )

    timing_collector = options.timing_collector
    n_trajectories = int(options.n_trajectories)
    chunk_slices = _trajectory_chunk_slices(n_trajectories, chunk_size)
    chunk_seeds = rng.integers(
        low=0,
        high=np.iinfo(np.int64).max,
        size=len(chunk_slices),
        dtype=np.int64,
    )

    rho_t = (
        [np.zeros((dim, dim), dtype=np.complex128) for _ in range(times.size)]
        if options.store_density_matrices
        else []
    )
    state_snapshots = (
        [np.empty((dim, n_trajectories), dtype=np.complex128) for _ in range(times.size)]
        if options.store_state_snapshots
        else None
    )

    worker_count = _effective_trajectory_chunk_workers(options, n_chunks=len(chunk_slices))
    if worker_count <= 1:
        for (chunk_start, chunk_stop), chunk_seed in zip(chunk_slices, chunk_seeds, strict=True):
            chunk_n = int(chunk_stop - chunk_start)
            chunk_options = replace(
                options,
                n_trajectories=chunk_n,
                trajectory_chunk_size=None,
                trajectory_chunk_workers=None,
            )
            chunk_rng = np.random.default_rng(int(chunk_seed))

            chunk_result = _sample_lindblad_mcwf_vectorized_scipy(
                prepared=prepared,
                dim=dim,
                times=times,
                state_initial=state_initial,
                state_sampler=state_sampler,
                options=chunk_options,
                rng=chunk_rng,
            )

            _merge_mcwf_chunk_result(
                parent_rho_t=rho_t,
                parent_state_snapshots=state_snapshots,
                chunk_result=chunk_result,
                chunk_start=chunk_start,
                chunk_stop=chunk_stop,
                n_trajectories=n_trajectories,
                timing_collector=timing_collector,
            )
    else:
        worker_start = _perf_counter()
        tasks = [
            (
                prepared,
                dim,
                times,
                state_initial,
                state_sampler,
                replace(
                    options,
                    n_trajectories=int(chunk_stop - chunk_start),
                    trajectory_chunk_size=None,
                    trajectory_chunk_workers=None,
                    timing_collector=None,
                ),
                int(chunk_seed),
            )
            for (chunk_start, chunk_stop), chunk_seed in zip(chunk_slices, chunk_seeds, strict=True)
        ]
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            chunk_outputs = list(executor.map(_sample_lindblad_mcwf_chunk_worker, tasks))
        _add_timing(timing_collector, "mcwf.chunk_parallel_wall", _perf_counter() - worker_start)

        for (chunk_start, chunk_stop), (chunk_result, chunk_timing) in zip(
            chunk_slices,
            chunk_outputs,
            strict=True,
        ):
            _merge_mcwf_timing(timing_collector, chunk_timing)
            _merge_mcwf_timing(
                timing_collector,
                chunk_timing,
                prefix="mcwf.worker.",
            )
            _merge_mcwf_chunk_result(
                parent_rho_t=rho_t,
                parent_state_snapshots=state_snapshots,
                chunk_result=chunk_result,
                chunk_start=chunk_start,
                chunk_stop=chunk_stop,
                n_trajectories=n_trajectories,
                timing_collector=timing_collector,
            )

    return EnsembleResult(
        times=times,
        rho_t=rho_t,
        trajectories=None,
        state_snapshots=(tuple(state_snapshots) if state_snapshots is not None else None),
    )


def _sample_lindblad_mcwf_chunk_worker(
    task: tuple[
        _McwfPreparedOperators,
        int,
        NDArray[np.float64],
        Any | None,
        StateSampler | None,
        McwfOptions,
        int,
    ],
) -> tuple[EnsembleResult, dict[str, float]]:
    """Run one MCWF chunk in a worker process.

    The function is intentionally module-level so it is picklable under the
    ``spawn`` multiprocessing start method used on macOS and Windows.
    """
    (
        prepared,
        dim,
        times,
        state_initial,
        state_sampler,
        options,
        seed,
    ) = task
    timing: dict[str, float] = {}
    chunk_options = replace(options, timing_collector=timing)
    result = _sample_lindblad_mcwf_vectorized_scipy(
        prepared=prepared,
        dim=dim,
        times=times,
        state_initial=state_initial,
        state_sampler=state_sampler,
        options=chunk_options,
        rng=np.random.default_rng(seed),
    )
    return result, timing


def _merge_mcwf_chunk_result(
    *,
    parent_rho_t: list[ArrayC],
    parent_state_snapshots: list[ArrayC] | None,
    chunk_result: EnsembleResult,
    chunk_start: int,
    chunk_stop: int,
    n_trajectories: int,
    timing_collector: MutableMapping[str, float] | None,
) -> None:
    start = _perf_counter()
    chunk_n = int(chunk_stop - chunk_start)
    if parent_rho_t:
        weight = chunk_n / float(n_trajectories)
        for time_index, density_matrix in enumerate(chunk_result.rho_t):
            parent_rho_t[time_index] += weight * density_matrix

    if parent_state_snapshots is not None:
        if chunk_result.state_snapshots is None:
            raise RuntimeError("chunked MCWF did not return requested state snapshots.")
        for time_index, state_matrix in enumerate(chunk_result.state_snapshots):
            parent_state_snapshots[time_index][:, chunk_start:chunk_stop] = state_matrix
    _add_timing(timing_collector, "mcwf.chunk_merge", _perf_counter() - start)


def _merge_mcwf_timing(
    timing_collector: MutableMapping[str, float] | None,
    child_timing: MutableMapping[str, float],
    *,
    prefix: str = "",
) -> None:
    if timing_collector is None:
        return

    for stage, elapsed_seconds in child_timing.items():
        _add_timing(timing_collector, f"{prefix}{stage}", elapsed_seconds)


def _effective_trajectory_chunk_size(options: McwfOptions) -> int | None:
    chunk_size = options.trajectory_chunk_size
    if chunk_size is None or chunk_size >= options.n_trajectories:
        return None

    return int(chunk_size)


def _effective_trajectory_chunk_workers(options: McwfOptions, *, n_chunks: int) -> int:
    workers = options.trajectory_chunk_workers
    if workers is None or workers <= 1 or n_chunks <= 1:
        return 1

    return min(int(workers), int(n_chunks))


def _trajectory_chunk_slices(
    n_trajectories: int,
    chunk_size: int,
) -> list[tuple[int, int]]:
    return [
        (start, min(start + chunk_size, n_trajectories))
        for start in range(0, n_trajectories, chunk_size)
    ]


def _evaluate_vectorized_jump_probabilities_numpy(
    *,
    states: ArrayC,
    step_size: float,
    jump_operators: tuple[Any, ...],
    sparse_jump_rate_evaluator: _SparseJumpRateEvaluator | None,
    total_jump_rate_operator: Any | None,
    use_total_rate_first: bool,
    timing_collector: MutableMapping[str, float] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Return first-order jump probabilities for the vectorized MCWF path."""
    start = _perf_counter()
    if _should_use_total_rate_first(
        use_total_rate_first,
        total_jump_rate_operator=total_jump_rate_operator,
        sparse_jump_rate_evaluator=sparse_jump_rate_evaluator,
    ):
        total_rates = _evaluate_total_jump_rates_state_matrix_numpy(
            states,
            total_jump_rate_operator,
        )
        total_jump_probabilities = step_size * total_rates
        elapsed = _perf_counter() - start
        _add_timing(timing_collector, "mcwf.total_rate_evaluation", elapsed)
        _add_timing(timing_collector, "mcwf.rate_evaluation", elapsed)
        return (
            np.zeros((0, states.shape[1]), dtype=np.float64),
            total_jump_probabilities,
            None,
        )

    if sparse_jump_rate_evaluator is not None:
        rates = _evaluate_sparse_jump_rates_state_matrix_numpy(
            states,
            sparse_jump_rate_evaluator,
        )
    else:
        rates = _evaluate_jump_rates_state_matrix_numpy(states, jump_operators)
    probabilities = step_size * rates
    total_jump_probabilities = np.sum(probabilities, axis=0)
    elapsed = _perf_counter() - start
    _add_timing(timing_collector, "mcwf.channel_rate_evaluation", elapsed)
    _add_timing(timing_collector, "mcwf.rate_evaluation", elapsed)
    return probabilities, total_jump_probabilities, rates


def _should_use_total_rate_first(
    enabled: bool,
    *,
    total_jump_rate_operator: Any | None,
    sparse_jump_rate_evaluator: _SparseJumpRateEvaluator | None,
) -> bool:
    """Return whether to use total-rate-first sampling for this operator mix.

    The total-rate path is structural: it avoids per-channel work on no-jump
    steps.  It is not always the fastest kernel, however.  A pure single-entry
    sparse rate matrix can be cheaper than applying ``Gamma=sum J†J``.  In that
    case, keep the already vectorized channel-rate path.
    """
    if not enabled or total_jump_rate_operator is None:
        return False

    if sparse_jump_rate_evaluator is None:
        return True

    if (
        sparse_jump_rate_evaluator.single_entry_rate_matrix is not None
        and sparse_jump_rate_evaluator.expanded_rate_operator is None
    ):
        return False

    channel_nnz = 0
    if sparse_jump_rate_evaluator.single_entry_rate_matrix is not None:
        channel_nnz += int(sparse_jump_rate_evaluator.single_entry_rate_matrix.nnz)
    if sparse_jump_rate_evaluator.expanded_rate_operator is not None:
        channel_nnz += int(sparse_jump_rate_evaluator.expanded_rate_operator.nnz)

    if channel_nnz <= 0:
        return True

    if scipy_sparse.issparse(total_jump_rate_operator):
        total_nnz = int(total_jump_rate_operator.nnz)
        return total_nnz < channel_nnz

    return True


def _evaluate_total_jump_rates_state_matrix_numpy(
    states: ArrayC,
    total_jump_rate_operator: Any,
) -> NDArray[np.float64]:
    """Return ``<psi_a|Gamma|psi_a>`` for state columns."""
    acted_states = total_jump_rate_operator @ states
    rates = np.einsum(
        "ij,ij->j",
        states.conj(),
        acted_states,
        optimize=True,
    ).real
    rates = np.asarray(rates, dtype=np.float64)
    np.maximum(rates, 0.0, out=rates)
    return rates


def _evaluate_total_jump_rate_numpy(
    state: ArrayC,
    total_jump_rate_operator: Any,
) -> float:
    """Return ``<psi|Gamma|psi>`` for one state."""
    value = np.vdot(state, total_jump_rate_operator @ state).real
    return max(float(value), 0.0)


def _evaluate_sparse_jump_rates_numpy(
    state: ArrayC,
    evaluator: _SparseJumpRateEvaluator,
) -> NDArray[np.float64]:
    """Return ||J_mu psi||^2 without forming full dense J_mu|psi> vectors."""
    rates = np.zeros(evaluator.n_jumps, dtype=np.float64)

    if evaluator.single_entry_rate_matrix is not None:
        state_weights = np.abs(state) ** 2
        rates += np.asarray(evaluator.single_entry_rate_matrix @ state_weights).reshape(-1)

    if evaluator.expanded_rate_operator is not None:
        row_values = evaluator.expanded_rate_operator @ state
        row_rates = np.abs(row_values) ** 2
        grouped_rates = np.add.reduceat(
            row_rates,
            evaluator.expanded_rate_row_splits[:-1],
        )
        rates[evaluator.expanded_rate_jump_indices] += grouped_rates
    else:
        for jump_index in evaluator.generic_jump_indices:
            columns = evaluator.single_entry_columns[int(jump_index)]
            weights = evaluator.single_entry_weights[int(jump_index)]
            if columns is not None and weights is not None:
                rates[int(jump_index)] = max(float(weights @ np.abs(state[columns]) ** 2), 0.0)
                continue

            rate = 0.0
            for row_columns, row_values in zip(
                evaluator.row_columns[int(jump_index)],
                evaluator.row_values[int(jump_index)],
                strict=True,
            ):
                value = np.dot(row_values, state[row_columns])
                rate += float(abs(value) ** 2)
            rates[int(jump_index)] = max(rate, 0.0)

    return rates


def _evaluate_sparse_jump_rates_state_matrix_numpy(
    states: ArrayC,
    evaluator: _SparseJumpRateEvaluator,
) -> NDArray[np.float64]:
    """Return ||J_mu psi_a||^2 without full dense J_mu|psi_a> blocks."""
    rates = np.zeros((evaluator.n_jumps, states.shape[1]), dtype=np.float64)

    if evaluator.single_entry_rate_matrix is not None:
        state_weights = np.abs(states) ** 2
        rates += np.asarray(evaluator.single_entry_rate_matrix @ state_weights)

    if evaluator.expanded_rate_operator is not None:
        expanded_values = evaluator.expanded_rate_operator @ states
        expanded_row_rates = np.abs(expanded_values) ** 2
        grouped_rates = np.add.reduceat(
            expanded_row_rates,
            evaluator.expanded_rate_row_splits[:-1],
            axis=0,
        )
        rates[evaluator.expanded_rate_jump_indices, :] += grouped_rates
    else:
        for jump_index in evaluator.generic_jump_indices:
            jump_index_int = int(jump_index)
            columns = evaluator.single_entry_columns[jump_index_int]
            weights = evaluator.single_entry_weights[jump_index_int]
            if columns is not None and weights is not None:
                if columns.size:
                    rates[jump_index_int, :] = weights @ np.abs(states[columns, :]) ** 2
                continue

            for row_columns, row_values in zip(
                evaluator.row_columns[jump_index_int],
                evaluator.row_values[jump_index_int],
                strict=True,
            ):
                row_values_for_states = row_values @ states[row_columns, :]
                rates[jump_index_int, :] += np.abs(row_values_for_states) ** 2

    np.maximum(rates, 0.0, out=rates)
    return rates


def _evaluate_jump_rates_state_matrix_numpy(
    states: ArrayC,
    jumps: tuple[Any, ...],
) -> NDArray[np.float64]:
    """Return rates ||J_mu psi_a||^2 without retaining J_mu|psi_a| blocks."""
    rates = np.empty((len(jumps), states.shape[1]), dtype=np.float64)

    for jump_index, jump in enumerate(jumps):
        jumped_states = jump @ states
        rates[jump_index, :] = np.einsum(
            "ij,ij->j",
            jumped_states.conj(),
            jumped_states,
            optimize=True,
        ).real

    np.maximum(rates, 0.0, out=rates)
    return rates


def _apply_selected_jumps_to_state_matrix_numpy(
    *,
    next_states: ArrayC,
    states: ArrayC,
    jumps: tuple[Any, ...],
    jump_mask: NDArray[np.bool_],
    selected_jump_indices: NDArray[np.int64],
) -> None:
    """Apply only the jump channels that were actually selected."""
    active_jump_indices = np.unique(selected_jump_indices[jump_mask])

    for jump_index in active_jump_indices:
        selected_mask = jump_mask & (selected_jump_indices == jump_index)
        if np.any(selected_mask):
            next_states[:, selected_mask] = jumps[int(jump_index)] @ states[:, selected_mask]


def _choose_jump_indices_vectorized(
    *,
    probabilities: NDArray[np.float64],
    total_probabilities: NDArray[np.float64],
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    cumulative_probabilities = np.cumsum(probabilities, axis=0)
    draws = rng.random(total_probabilities.size) * total_probabilities
    selected = np.sum(
        draws.reshape(1, -1) >= cumulative_probabilities,
        axis=0,
        dtype=np.int64,
    )
    return np.minimum(selected, probabilities.shape[0] - 1).astype(np.int64, copy=False)


def _initial_state_matrix_for_ensemble(
    *,
    dim: int,
    n_trajectories: int,
    state_initial: Any | None,
    state_sampler: StateSampler | None,
    rng: np.random.Generator,
) -> ArrayC:
    if state_initial is not None:
        state = _normalize_numpy_state(np.asarray(state_initial, dtype=np.complex128))
        return np.repeat(state.reshape(-1, 1), n_trajectories, axis=1)

    states = np.empty((dim, n_trajectories), dtype=np.complex128)
    child_seeds = rng.integers(
        low=0,
        high=np.iinfo(np.int64).max,
        size=n_trajectories,
        dtype=np.int64,
    )

    for trajectory_index, child_seed in enumerate(child_seeds):
        trajectory_rng = np.random.default_rng(int(child_seed))
        state = _initial_state_for_trajectory(
            dim=dim,
            state_initial=None,
            state_sampler=state_sampler,
            rng=trajectory_rng,
        )
        states[:, trajectory_index] = _normalize_numpy_state(state)

    return states


def _normalize_numpy_state(state: ArrayC) -> ArrayC:
    norm = float(np.linalg.norm(state))

    if norm == 0.0:
        raise ValueError("state must be nonzero.")

    return np.asarray(state / norm, dtype=np.complex128)


def density_matrix_from_state_matrix(states: ArrayC) -> ArrayC:
    """Return the ensemble density matrix represented by state columns.

    ``states[:, trajectory_index]`` is one normalized MCWF trajectory state.
    The returned matrix is the average projector over columns.
    """
    states = np.asarray(states, dtype=np.complex128)
    if states.ndim != 2:
        raise ValueError("states must be a 2D array with one state per column.")

    if states.shape[1] == 0:
        raise ValueError("states must contain at least one trajectory column.")

    return (states @ states.conj().T) / float(states.shape[1])


def _density_matrix_from_state_matrix(states: ArrayC) -> ArrayC:
    return density_matrix_from_state_matrix(states)


def sample_lindblad_mcwf(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    times: NDArray[np.float64],
    state_initial: Any | None = None,
    state_sampler: StateSampler | None = None,
    options: McwfOptions | None = None,
    rng: np.random.Generator | int | None = None,
) -> EnsembleResult:
    """Estimate Lindblad evolution by averaging MCWF trajectories.

    Parameters
    ----------
    hamiltonian:
        Hamiltonian matrix.

    jumps:
        Lindblad jump operators.

    times:
        Strictly increasing time grid.

    state_initial:
        Fixed initial pure state. Used for every trajectory.

    state_sampler:
        Optional callable that samples an initial pure state from a NumPy RNG.
        Mutually exclusive with state_initial.

    options:
        MCWF options.

    rng:
        Optional RNG or seed. Overrides options.seed when provided.

    Returns
    -------
    EnsembleResult
        Contains ensemble-averaged density matrices ``rho_t`` when requested,
        optional low-rank state snapshots, and optional individual trajectories.
    """
    if options is None:
        options = McwfOptions()

    options.validate()

    if state_initial is not None and state_sampler is not None:
        raise ValueError("Pass only one of state_initial or state_sampler.")

    times = np.asarray(times, dtype=np.float64)
    _validate_times_for_mcwf(times)

    generator = _rng_from_seed(options.seed if rng is None else rng)
    start = _perf_counter()
    prepared = _prepare_mcwf_operators(
        hamiltonian=hamiltonian,
        jumps=jumps,
        backend=options.backend,
        prefer_sparse_operators=options.prefer_sparse_operators,
        prefer_sparse_rate_evaluator=options.prefer_sparse_rate_evaluator,
    )
    _add_timing(options.timing_collector, "mcwf.operator_preparation", _perf_counter() - start)
    backend_obj = prepared.backend
    hamiltonian_backend = prepared.hamiltonian

    dim = int(hamiltonian_backend.shape[0])

    if hamiltonian_backend.shape != (dim, dim):
        raise ValueError("hamiltonian must be a square matrix.")

    for jump in prepared.jumps:
        if jump.shape != (dim, dim):
            raise ValueError("Every jump operator must have shape (dim, dim).")

    if _can_use_vectorized_mcwf_ensemble(
        options=options,
        backend=backend_obj,
    ):
        if _effective_trajectory_chunk_size(options) is not None:
            return _sample_lindblad_mcwf_chunked_vectorized_scipy(
                prepared=prepared,
                dim=dim,
                times=times,
                state_initial=state_initial,
                state_sampler=state_sampler,
                options=options,
                rng=generator,
            )

        return _sample_lindblad_mcwf_vectorized_scipy(
            prepared=prepared,
            dim=dim,
            times=times,
            state_initial=state_initial,
            state_sampler=state_sampler,
            options=options,
            rng=generator,
        )

    rho_t = (
        [np.zeros((dim, dim), dtype=np.complex128) for _ in range(times.size)]
        if options.store_density_matrices
        else []
    )
    state_snapshots = (
        [np.zeros((dim, options.n_trajectories), dtype=np.complex128) for _ in range(times.size)]
        if options.store_state_snapshots
        else None
    )

    stored_trajectories: list[TrajectoryResult] | None
    if options.store_trajectories:
        stored_trajectories = []
    else:
        stored_trajectories = None

    child_seeds = generator.integers(
        low=0,
        high=np.iinfo(np.int64).max,
        size=options.n_trajectories,
        dtype=np.int64,
    )

    store_trajectory_states = bool(options.store_trajectories and options.store_states)

    for trajectory_index, child_seed in enumerate(child_seeds):
        trajectory_rng = np.random.default_rng(int(child_seed))

        trajectory_state_initial = _initial_state_for_trajectory(
            dim=dim,
            state_initial=state_initial,
            state_sampler=state_sampler,
            rng=trajectory_rng,
        )

        state_callback: Callable[[int, Any], None] | None = None
        if options.store_density_matrices or state_snapshots is not None:

            def collect_ensemble_output(time_index: int, state: Any) -> None:
                state_numpy = _state_to_numpy(state, backend=backend_obj)
                if options.store_density_matrices:
                    rho_t[time_index] += projector(state_numpy)

                if state_snapshots is not None:
                    state_snapshots[time_index][:, trajectory_index] = state_numpy

            state_callback = collect_ensemble_output

        trajectory = _run_quantum_jump_trajectory_prepared(
            prepared=prepared,
            state_initial=trajectory_state_initial,
            times=times,
            rng=trajectory_rng,
            return_backend_arrays=options.return_backend_arrays,
            store_states=store_trajectory_states,
            state_callback=state_callback,
            normalize_each_step=options.normalize_each_step,
            max_jump_probability=options.max_jump_probability,
            adaptive_time_step=options.adaptive_time_step,
            adaptive_safety_factor=options.adaptive_safety_factor,
            min_step_size=options.min_step_size,
            max_substeps_per_interval=options.max_substeps_per_interval,
        )

        if stored_trajectories is not None:
            stored_trajectories.append(trajectory)

    if options.store_density_matrices:
        rho_t = [density_matrix / float(options.n_trajectories) for density_matrix in rho_t]

    return EnsembleResult(
        times=times,
        rho_t=rho_t,
        trajectories=(tuple(stored_trajectories) if stored_trajectories is not None else None),
        state_snapshots=(tuple(state_snapshots) if state_snapshots is not None else None),
    )


def observable_vs_time(
    rho_t: Iterable[ArrayC],
    observable: ArrayC,
) -> ArrayF:
    """Return Tr[rho(t) observable] for each density matrix."""
    observable = np.asarray(observable, dtype=np.complex128)

    values = [np.real_if_close(np.trace(density_matrix @ observable)) for density_matrix in rho_t]

    return np.asarray(values, dtype=np.float64)


def _validate_times_for_mcwf(times: NDArray[np.float64]) -> None:
    if times.ndim != 1:
        raise ValueError("times must be a 1D array.")

    if times.size < 2:
        raise ValueError("times must contain at least two points.")

    if not np.all(np.isfinite(times)):
        raise ValueError("times must be finite.")

    if not np.all(np.diff(times) > 0.0):
        raise ValueError("times must be strictly increasing.")


def _rng_from_seed(
    rng: np.random.Generator | int | None,
) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng

    return np.random.default_rng(rng)


def _initial_state_for_trajectory(
    *,
    dim: int,
    state_initial: Any | None,
    state_sampler: StateSampler | None,
    rng: np.random.Generator,
) -> ArrayC:
    if state_initial is not None:
        return np.asarray(state_initial, dtype=np.complex128)

    if state_sampler is not None:
        return np.asarray(state_sampler(rng), dtype=np.complex128)

    return random_pure_state(dim, rng=rng)


def _backend_state_norm(
    state: Any,
    *,
    backend: OpenSystemBackend,
) -> float:
    norm = backend.array_module.linalg.norm(state)
    return float(backend.to_numpy(norm))


def _normalize_backend_state(
    state: Any,
    *,
    backend: OpenSystemBackend,
) -> Any:
    norm = _backend_state_norm(state, backend=backend)

    if norm == 0.0:
        raise ValueError("state must be nonzero.")

    return state / norm


def _maybe_to_numpy(
    value: Any,
    *,
    backend: OpenSystemBackend,
    enabled: bool,
) -> Any:
    if not enabled:
        return value.copy()

    return backend.to_numpy(value.copy())


def _state_to_numpy(
    state: Any,
    *,
    backend: OpenSystemBackend,
) -> ArrayC:
    if isinstance(state, np.ndarray):
        return np.asarray(state, dtype=np.complex128)

    return np.asarray(backend.to_numpy(state), dtype=np.complex128)
