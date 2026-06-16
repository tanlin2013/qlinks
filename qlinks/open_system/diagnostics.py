from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.linalg as scipy_linalg
import scipy.sparse as scipy_sparse
from scipy.sparse.csgraph import connected_components

from qlinks.open_system.backend import (
    OpenSystemBackend,
    OpenSystemBackendName,
)
from qlinks.open_system.operators import (
    build_liouvillian,
    lindblad_rhs_density_matrix,
)


@dataclass(frozen=True, slots=True)
class EvolutionDiagnostics:
    trace_errors: np.ndarray
    hermiticity_errors: np.ndarray
    min_eigenvalues: np.ndarray
    purities: np.ndarray
    fidelities: np.ndarray | None
    lindblad_residuals: np.ndarray | None
    times: np.ndarray | None = None
    source: str = "density_matrices"
    density_check_mode: str = "full"
    trajectory_counts: np.ndarray | None = None
    state_norm_errors: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class JumpSpanDiagnostics:
    """Hilbert-Schmidt span diagnostics for a Lindblad jump list."""

    dim: int
    n_jumps: int
    span_rank: int
    dependent_jump_count: int
    compression_ratio: float
    rank_tolerance: float
    absolute_rank_threshold: float
    gram_eigenvalues: np.ndarray
    effective_rank: float
    participation_rank: float
    total_jump_nnz: int | None
    span_matrix_nnz: int | None
    max_normalized_overlap: float
    mean_normalized_overlap: float

    @property
    def has_exact_dependencies(self) -> bool:
        return self.dependent_jump_count > 0

    def to_summary_dict(self, *, n_eigenvalues: int = 8) -> dict[str, object]:
        """Return a compact JSON-friendly benchmark summary."""
        leading = self.gram_eigenvalues[: max(int(n_eigenvalues), 0)]
        trailing = self.gram_eigenvalues[-max(int(n_eigenvalues), 0) :]
        return {
            "dim": self.dim,
            "n_jumps": self.n_jumps,
            "span_rank": self.span_rank,
            "dependent_jump_count": self.dependent_jump_count,
            "compression_ratio": self.compression_ratio,
            "rank_tolerance": self.rank_tolerance,
            "absolute_rank_threshold": self.absolute_rank_threshold,
            "effective_rank": self.effective_rank,
            "participation_rank": self.participation_rank,
            "total_jump_nnz": self.total_jump_nnz,
            "span_matrix_nnz": self.span_matrix_nnz,
            "max_normalized_overlap": self.max_normalized_overlap,
            "mean_normalized_overlap": self.mean_normalized_overlap,
            "leading_gram_eigenvalues": [float(value) for value in leading],
            "trailing_gram_eigenvalues": [float(value) for value in trailing],
        }


def diagnose_jump_span(
    jumps: Sequence[Any],
    *,
    rank_tolerance: float = 1.0e-10,
) -> JumpSpanDiagnostics:
    """Diagnose exact/near linear dependencies among jump operators.

    The jump-operator span is measured in the Hilbert-Schmidt inner product,
    ``<J_i, J_j> = Tr(J_i† J_j)``.  The rank of this Gram matrix is the number
    of independent jump directions.  If this rank is much smaller than the raw
    number of jumps, a future compression pass can rotate/drop jumps before MCWF
    sampling without changing the Lindblad dissipator.
    """
    jump_tuple = tuple(jumps)
    if not jump_tuple:
        return JumpSpanDiagnostics(
            dim=0,
            n_jumps=0,
            span_rank=0,
            dependent_jump_count=0,
            compression_ratio=0.0,
            rank_tolerance=float(rank_tolerance),
            absolute_rank_threshold=0.0,
            gram_eigenvalues=np.zeros(0, dtype=np.float64),
            effective_rank=0.0,
            participation_rank=0.0,
            total_jump_nnz=0,
            span_matrix_nnz=0,
            max_normalized_overlap=0.0,
            mean_normalized_overlap=0.0,
        )

    first_shape = tuple(int(axis) for axis in jump_tuple[0].shape)
    if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
        raise ValueError("Jump operators must be square matrices.")

    dim = first_shape[0]
    for jump in jump_tuple:
        if tuple(int(axis) for axis in jump.shape) != first_shape:
            raise ValueError("All jump operators must have the same shape.")

    span_matrix, total_jump_nnz = _jump_span_matrix(jump_tuple, dim=dim)
    if scipy_sparse.issparse(span_matrix):
        gram_matrix = (span_matrix.conj().T @ span_matrix).toarray()
        span_matrix_nnz = int(span_matrix.nnz)
    else:
        gram_matrix = span_matrix.conj().T @ span_matrix
        span_matrix_nnz = None

    gram_matrix = np.asarray(gram_matrix, dtype=np.complex128)
    gram_matrix = 0.5 * (gram_matrix + gram_matrix.conj().T)
    eigenvalues = np.linalg.eigvalsh(gram_matrix).real
    eigenvalues = np.sort(np.maximum(eigenvalues, 0.0))[::-1]

    largest_eigenvalue = float(eigenvalues[0]) if eigenvalues.size else 0.0
    absolute_threshold = float(rank_tolerance) * max(largest_eigenvalue, 1.0)
    span_rank = int(np.count_nonzero(eigenvalues > absolute_threshold))
    probabilities = eigenvalues[eigenvalues > 0.0]
    total_weight = float(np.sum(probabilities))
    if total_weight > 0.0:
        normalized = probabilities / total_weight
        effective_rank = float(np.exp(-np.sum(normalized * np.log(normalized))))
        participation_rank = float(
            total_weight * total_weight / np.sum(probabilities * probabilities)
        )
    else:
        effective_rank = 0.0
        participation_rank = 0.0

    normalized_overlaps = _normalized_gram_offdiagonal_values(gram_matrix)
    if normalized_overlaps.size:
        max_overlap = float(np.max(normalized_overlaps))
        mean_overlap = float(np.mean(normalized_overlaps))
    else:
        max_overlap = 0.0
        mean_overlap = 0.0

    return JumpSpanDiagnostics(
        dim=dim,
        n_jumps=len(jump_tuple),
        span_rank=span_rank,
        dependent_jump_count=len(jump_tuple) - span_rank,
        compression_ratio=(float(span_rank) / float(len(jump_tuple))),
        rank_tolerance=float(rank_tolerance),
        absolute_rank_threshold=absolute_threshold,
        gram_eigenvalues=eigenvalues.astype(np.float64, copy=False),
        effective_rank=effective_rank,
        participation_rank=participation_rank,
        total_jump_nnz=total_jump_nnz,
        span_matrix_nnz=span_matrix_nnz,
        max_normalized_overlap=max_overlap,
        mean_normalized_overlap=mean_overlap,
    )


def _jump_span_matrix(
    jumps: tuple[Any, ...],
    *,
    dim: int,
) -> tuple[Any, int | None]:
    if all(
        scipy_sparse.issparse(jump)
        or hasattr(jump, "tocoo")
        or hasattr(jump, "tocsr")
        or hasattr(jump, "asformat")
        for jump in jumps
    ):
        data_blocks: list[np.ndarray] = []
        row_blocks: list[np.ndarray] = []
        column_blocks: list[np.ndarray] = []
        total_jump_nnz = 0
        for jump_index, jump in enumerate(jumps):
            if hasattr(jump, "tocoo"):
                coo = jump.tocoo()
            elif hasattr(jump, "tocsr"):
                coo = jump.tocsr().tocoo()
            elif hasattr(jump, "asformat"):
                coo = jump.asformat("coo")
            else:
                coo = scipy_sparse.coo_array(jump)
            coo = coo.astype(np.complex128)
            flat_rows = np.asarray(coo.row, dtype=np.int64) * dim + np.asarray(
                coo.col, dtype=np.int64
            )
            data_blocks.append(np.asarray(coo.data, dtype=np.complex128))
            row_blocks.append(flat_rows)
            column_blocks.append(np.full(coo.nnz, jump_index, dtype=np.int64))
            total_jump_nnz += int(coo.nnz)

        if not data_blocks:
            return scipy_sparse.csc_array((dim * dim, len(jumps)), dtype=np.complex128), 0

        span = scipy_sparse.csc_array(
            (
                np.concatenate(data_blocks),
                (np.concatenate(row_blocks), np.concatenate(column_blocks)),
            ),
            shape=(dim * dim, len(jumps)),
            dtype=np.complex128,
        )
        span.sum_duplicates()
        span.eliminate_zeros()
        return span, total_jump_nnz

    dense_columns = [np.asarray(jump, dtype=np.complex128).reshape(dim * dim) for jump in jumps]
    return np.column_stack(dense_columns), None


def _normalized_gram_offdiagonal_values(gram_matrix: np.ndarray) -> np.ndarray:
    diagonal = np.maximum(np.real(np.diag(gram_matrix)), 0.0)
    scales = np.sqrt(np.outer(diagonal, diagonal))
    valid = scales > 0.0
    normalized = np.zeros_like(np.abs(gram_matrix), dtype=np.float64)
    normalized[valid] = np.abs(gram_matrix[valid]) / scales[valid]
    offdiagonal_mask = ~np.eye(gram_matrix.shape[0], dtype=bool)
    return normalized[offdiagonal_mask]


def analyze_lindblad_evolution(
    density_matrices: Sequence[Any] | None = None,
    *,
    ensemble_result: Any | None = None,
    state_snapshots: Sequence[Any] | None = None,
    trajectories: Sequence[Any] | None = None,
    times: npt.ArrayLike | None = None,
    target_state: npt.ArrayLike | None = None,
    hamiltonian=None,
    jumps=None,
    atol: float = 1e-10,
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
    density_check_mode: str = "auto",
) -> EvolutionDiagnostics:
    """Analyze density diagnostics along a Lindblad/MCWF evolution.

    The function accepts the original dense-density path, but it can also read
    MCWF ensemble outputs directly. In particular, an ``EnsembleResult`` with
    ``rho_t`` disabled and ``state_snapshots`` enabled can be diagnosed without
    rebuilding dense density matrices unless explicitly requested.

    Parameters
    ----------
    density_matrices:
        Sequence of density matrices. This is the legacy input path.

    ensemble_result:
        Optional MCWF ``EnsembleResult``. The analyzer prefers ``rho_t`` when it
        is populated, then ``state_snapshots``, then stored trajectories.

    state_snapshots:
        Optional sequence of state matrices with shape ``(dim, n_trajectories)``
        representing low-rank MCWF ensemble snapshots.

    trajectories:
        Optional sequence of trajectory results with stored ``states``. This is
        mainly a compatibility path for scalar MCWF output.

    density_check_mode:
        ``"auto"`` uses full density checks when dense density matrices are
        available or when Lindblad residuals are requested. For snapshots only,
        it uses ``"low_rank"`` by default. ``"full"`` materializes density
        matrices from snapshots and can compute Lindblad residuals.
        ``"low_rank"`` computes trace, purity, and target fidelity directly
        from snapshot state matrices and reports ``NaN`` for min eigenvalues.
    """
    (
        density_matrices_resolved,
        state_snapshots_resolved,
        times_resolved,
        source,
    ) = _resolve_lindblad_evolution_inputs(
        density_matrices=density_matrices,
        ensemble_result=ensemble_result,
        state_snapshots=state_snapshots,
        trajectories=trajectories,
        times=times,
    )

    if density_check_mode not in {"auto", "full", "low_rank"}:
        raise ValueError('density_check_mode must be "auto", "full", or "low_rank".')

    if density_matrices_resolved is not None:
        mode = "full" if density_check_mode == "auto" else density_check_mode
        if mode == "low_rank":
            raise ValueError(
                'density_check_mode="low_rank" requires state snapshots, not density matrices.'
            )
        return _analyze_density_matrix_sequence(
            density_matrices_resolved,
            target_state=target_state,
            hamiltonian=hamiltonian,
            jumps=jumps,
            atol=atol,
            backend=backend,
            times=times_resolved,
            source=source,
            density_check_mode="full",
        )

    if state_snapshots_resolved is None:
        streamed_fidelities = _streamed_fidelities_from_ensemble_result(ensemble_result)
        if streamed_fidelities is not None:
            return _analyze_streamed_fidelity_series(
                streamed_fidelities,
                times=times_resolved,
                source="ensemble_result.target_fidelities",
            )

        raise ValueError(
            "Provide density_matrices, state_snapshots, trajectories, or an EnsembleResult "
            "containing rho_t/state_snapshots/trajectories/target_fidelities."
        )

    mode = density_check_mode
    if mode == "auto":
        mode = "full" if hamiltonian is not None and jumps is not None else "low_rank"

    snapshots = tuple(_as_state_snapshot(snapshot) for snapshot in state_snapshots_resolved)
    _validate_times_length(times_resolved, len(snapshots))

    trajectory_counts = np.array([snapshot.shape[1] for snapshot in snapshots], dtype=np.int64)
    state_norm_errors = np.array(
        [_state_snapshot_norm_error(snapshot) for snapshot in snapshots],
        dtype=np.float64,
    )

    if mode == "full":
        density_matrices_from_snapshots = [
            _density_matrix_from_state_matrix(snapshot) for snapshot in snapshots
        ]
        return _analyze_density_matrix_sequence(
            density_matrices_from_snapshots,
            target_state=target_state,
            hamiltonian=hamiltonian,
            jumps=jumps,
            atol=atol,
            backend=backend,
            times=times_resolved,
            source=source,
            density_check_mode="full",
            trajectory_counts=trajectory_counts,
            state_norm_errors=state_norm_errors,
        )

    if hamiltonian is not None and jumps is not None:
        raise ValueError(
            "Lindblad residuals require density_check_mode='full' when analyzing "
            "state snapshots."
        )

    return _analyze_state_snapshot_sequence_low_rank(
        snapshots,
        target_state=target_state,
        times=times_resolved,
        source=source,
        trajectory_counts=trajectory_counts,
        state_norm_errors=state_norm_errors,
    )


def _analyze_density_matrix_sequence(
    density_matrices: Sequence[Any],
    *,
    target_state: npt.ArrayLike | None,
    hamiltonian: Any,
    jumps: Any,
    atol: float,
    backend: OpenSystemBackendName | OpenSystemBackend,
    times: np.ndarray | None,
    source: str,
    density_check_mode: str,
    trajectory_counts: np.ndarray | None = None,
    state_norm_errors: np.ndarray | None = None,
) -> EvolutionDiagnostics:
    density_matrix_tuple = tuple(density_matrices)
    _validate_times_length(times, len(density_matrix_tuple))

    density_diagnostics = [
        verify_density_matrix(
            density_matrix,
            target_state=target_state,
            atol=atol,
        )
        for density_matrix in density_matrix_tuple
    ]

    lindblad_residuals = None
    if hamiltonian is not None and jumps is not None:
        lindblad_residuals = np.array(
            [
                np.linalg.norm(
                    lindblad_rhs_density_matrix(
                        density_matrix,
                        hamiltonian=hamiltonian,
                        jumps=jumps,
                        backend=backend,
                    )
                )
                for density_matrix in density_matrix_tuple
            ],
            dtype=np.float64,
        )

    fidelities = None
    if target_state is not None:
        fidelities = np.array(
            [diagnostic.fidelity_with_target for diagnostic in density_diagnostics],
            dtype=np.float64,
        )

    return EvolutionDiagnostics(
        trace_errors=np.array([d.trace_error for d in density_diagnostics], dtype=np.float64),
        hermiticity_errors=np.array(
            [d.hermiticity_error for d in density_diagnostics],
            dtype=np.float64,
        ),
        min_eigenvalues=np.array([d.min_eigenvalue for d in density_diagnostics], dtype=np.float64),
        purities=np.array([d.purity for d in density_diagnostics], dtype=np.float64),
        fidelities=fidelities,
        lindblad_residuals=lindblad_residuals,
        times=times,
        source=source,
        density_check_mode=density_check_mode,
        trajectory_counts=trajectory_counts,
        state_norm_errors=state_norm_errors,
    )


def _resolve_lindblad_evolution_inputs(
    *,
    density_matrices: Sequence[Any] | None,
    ensemble_result: Any | None,
    state_snapshots: Sequence[Any] | None,
    trajectories: Sequence[Any] | None,
    times: npt.ArrayLike | None,
) -> tuple[
    Sequence[Any] | None,
    Sequence[Any] | None,
    np.ndarray | None,
    str,
]:
    n_explicit_sources = sum(
        source is not None
        for source in (density_matrices, ensemble_result, state_snapshots, trajectories)
    )
    if n_explicit_sources != 1:
        raise ValueError(
            "Pass exactly one of density_matrices, ensemble_result, "
            "state_snapshots, or trajectories."
        )

    times_array = None if times is None else np.asarray(times, dtype=np.float64)

    if ensemble_result is not None:
        if times_array is None:
            result_times = getattr(ensemble_result, "times", None)
            if result_times is not None:
                times_array = np.asarray(result_times, dtype=np.float64)

        result_rho_t = tuple(getattr(ensemble_result, "rho_t", ()) or ())
        if result_rho_t:
            return result_rho_t, None, times_array, "ensemble_result.rho_t"

        result_snapshots = getattr(ensemble_result, "state_snapshots", None)
        if result_snapshots is not None:
            return None, tuple(result_snapshots), times_array, "ensemble_result.state_snapshots"

        result_trajectories = getattr(ensemble_result, "trajectories", None)
        if result_trajectories is not None:
            snapshots = _state_snapshots_from_trajectories(tuple(result_trajectories))
            return None, snapshots, times_array, "ensemble_result.trajectories"

        return None, None, times_array, "ensemble_result"

    if density_matrices is not None:
        density_matrix_tuple = tuple(density_matrices)
        return density_matrix_tuple, None, times_array, "density_matrices"

    if state_snapshots is not None:
        return None, tuple(state_snapshots), times_array, "state_snapshots"

    assert trajectories is not None
    return (
        None,
        _state_snapshots_from_trajectories(tuple(trajectories)),
        times_array,
        "trajectories",
    )


def _streamed_fidelities_from_ensemble_result(
    ensemble_result: Any | None,
) -> np.ndarray | None:
    if ensemble_result is None:
        return None

    target_fidelities = getattr(ensemble_result, "target_fidelities", None)
    if not target_fidelities:
        return None

    if "target" in target_fidelities:
        return np.asarray(target_fidelities["target"], dtype=np.float64)

    if len(target_fidelities) == 1:
        return np.asarray(next(iter(target_fidelities.values())), dtype=np.float64)

    raise ValueError(
        "EnsembleResult contains multiple target_fidelities entries; use the "
        "'target' key for analyze_lindblad_evolution or pass state_snapshots."
    )


def _analyze_streamed_fidelity_series(
    fidelities: np.ndarray,
    *,
    times: np.ndarray | None,
    source: str,
) -> EvolutionDiagnostics:
    fidelity_values = np.asarray(fidelities, dtype=np.float64)
    if fidelity_values.ndim != 1 or fidelity_values.size == 0:
        raise ValueError("streamed fidelity series must be a non-empty 1D array.")
    _validate_times_length(times, fidelity_values.size)

    missing_values = np.full(fidelity_values.size, np.nan, dtype=np.float64)
    return EvolutionDiagnostics(
        trace_errors=missing_values.copy(),
        hermiticity_errors=missing_values.copy(),
        min_eigenvalues=missing_values.copy(),
        purities=missing_values.copy(),
        fidelities=fidelity_values,
        lindblad_residuals=None,
        times=times,
        source=source,
        density_check_mode="streamed_fidelity",
        trajectory_counts=None,
        state_norm_errors=None,
    )


def _state_snapshots_from_trajectories(trajectories: Sequence[Any]) -> tuple[np.ndarray, ...]:
    if not trajectories:
        raise ValueError("trajectories must not be empty.")

    n_times = len(getattr(trajectories[0], "states", ()))
    if n_times == 0:
        raise ValueError("trajectories must contain stored states.")

    snapshots: list[np.ndarray] = []
    for time_index in range(n_times):
        states_at_time = []
        for trajectory in trajectories:
            states = getattr(trajectory, "states", None)
            if states is None or len(states) != n_times:
                raise ValueError("Every trajectory must store the same number of states.")
            states_at_time.append(np.asarray(states[time_index], dtype=np.complex128))
        snapshots.append(np.column_stack(states_at_time))

    return tuple(snapshots)


def _analyze_state_snapshot_sequence_low_rank(
    snapshots: Sequence[np.ndarray],
    *,
    target_state: npt.ArrayLike | None,
    times: np.ndarray | None,
    source: str,
    trajectory_counts: np.ndarray,
    state_norm_errors: np.ndarray,
) -> EvolutionDiagnostics:
    target = _normalized_target_state(target_state) if target_state is not None else None

    trace_errors = []
    purities = []
    fidelities = [] if target is not None else None

    for snapshot in snapshots:
        column_norms = np.sum(np.abs(snapshot) ** 2, axis=0)
        trace_errors.append(float(abs(float(np.mean(column_norms)) - 1.0)))
        purities.append(_state_snapshot_purity(snapshot))
        if target is not None:
            overlaps = target.conj() @ snapshot
            fidelities.append(float(np.real(np.mean(np.abs(overlaps) ** 2))))

    n_outputs = len(snapshots)
    return EvolutionDiagnostics(
        trace_errors=np.array(trace_errors, dtype=np.float64),
        hermiticity_errors=np.zeros(n_outputs, dtype=np.float64),
        min_eigenvalues=np.full(n_outputs, np.nan, dtype=np.float64),
        purities=np.array(purities, dtype=np.float64),
        fidelities=(None if fidelities is None else np.array(fidelities, dtype=np.float64)),
        lindblad_residuals=None,
        times=times,
        source=source,
        density_check_mode="low_rank",
        trajectory_counts=trajectory_counts,
        state_norm_errors=state_norm_errors,
    )


def _as_state_snapshot(snapshot: npt.ArrayLike) -> np.ndarray:
    snapshot_array = np.asarray(snapshot, dtype=np.complex128)
    if snapshot_array.ndim != 2:
        raise ValueError("Each state snapshot must be a two-dimensional state matrix.")
    if snapshot_array.shape[1] == 0:
        raise ValueError("Each state snapshot must contain at least one trajectory column.")
    return snapshot_array


def _density_matrix_from_state_matrix(states: np.ndarray) -> np.ndarray:
    return (states @ states.conj().T) / float(states.shape[1])


def _state_snapshot_norm_error(snapshot: np.ndarray) -> float:
    column_norms = np.sum(np.abs(snapshot) ** 2, axis=0)
    return float(np.max(np.abs(column_norms - 1.0)))


def _state_snapshot_purity(snapshot: np.ndarray) -> float:
    dim, n_trajectories = snapshot.shape
    if dim <= n_trajectories:
        density_matrix = _density_matrix_from_state_matrix(snapshot)
        return float(np.real(np.trace(density_matrix @ density_matrix)))

    gram = snapshot.conj().T @ snapshot
    return float(np.real(np.sum(np.abs(gram) ** 2)) / float(n_trajectories**2))


def _normalized_target_state(target_state: npt.ArrayLike) -> np.ndarray:
    target = np.asarray(target_state, dtype=np.complex128)
    if target.ndim != 1:
        raise ValueError("target_state must be one-dimensional.")
    norm = np.linalg.norm(target)
    if norm == 0:
        raise ValueError("target_state must be nonzero.")
    return target / norm


def _validate_times_length(times: np.ndarray | None, n_outputs: int) -> None:
    if times is None:
        return
    if times.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    if times.size != n_outputs:
        raise ValueError("times length must match the number of evolution outputs.")


@dataclass(frozen=True, slots=True)
class DensityMatrixVerification:
    trace: complex
    trace_error: float
    hermiticity_error: float
    min_eigenvalue: float
    purity: float
    fidelity_with_target: float | None
    is_hermitian: bool
    is_trace_one: bool
    is_positive_semidefinite: bool
    is_density_matrix: bool


def verify_density_matrix(
    rho: npt.ArrayLike,
    *,
    target_state: npt.ArrayLike | None = None,
    atol: float = 1e-10,
) -> DensityMatrixVerification:
    rho_array = np.asarray(rho, dtype=np.complex128)

    if rho_array.ndim != 2 or rho_array.shape[0] != rho_array.shape[1]:
        raise ValueError("rho must be a square matrix.")

    trace = np.trace(rho_array)
    trace_error = float(abs(trace - 1.0))

    hermitian_part = 0.5 * (rho_array + rho_array.conj().T)
    hermiticity_error = float(np.linalg.norm(rho_array - rho_array.conj().T))

    eigenvalues = np.linalg.eigvalsh(hermitian_part)
    min_eigenvalue = float(np.min(eigenvalues))

    purity_value = float(np.real(np.trace(rho_array @ rho_array)))

    fidelity = None
    if target_state is not None:
        psi = np.asarray(target_state, dtype=np.complex128)

        if psi.ndim != 1:
            raise ValueError("target_state must be one-dimensional.")

        norm = np.linalg.norm(psi)
        if norm == 0:
            raise ValueError("target_state must be nonzero.")

        psi = psi / norm
        fidelity = float(np.real(np.vdot(psi, rho_array @ psi)))

    is_hermitian = hermiticity_error <= atol
    is_trace_one = trace_error <= atol
    is_positive = min_eigenvalue >= -atol

    return DensityMatrixVerification(
        trace=complex(trace),
        trace_error=trace_error,
        hermiticity_error=hermiticity_error,
        min_eigenvalue=min_eigenvalue,
        purity=purity_value,
        fidelity_with_target=fidelity,
        is_hermitian=is_hermitian,
        is_trace_one=is_trace_one,
        is_positive_semidefinite=is_positive,
        is_density_matrix=(is_hermitian and is_trace_one and is_positive),
    )


@dataclass(frozen=True, slots=True)
class LindbladFinalStateVerification:
    density_matrix: DensityMatrixVerification
    lindblad_residual: float
    relative_lindblad_residual: float


def verify_lindblad_final_state(
    rho: npt.ArrayLike,
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    target_state: npt.ArrayLike | None = None,
    atol: float = 1e-10,
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
) -> LindbladFinalStateVerification:
    rho_array = np.asarray(rho, dtype=np.complex128)

    density = verify_density_matrix(
        rho_array,
        target_state=target_state,
        atol=atol,
    )

    rhs = lindblad_rhs_density_matrix(
        rho_array,
        hamiltonian=hamiltonian,
        jumps=jumps,
        backend=backend,
    )

    residual = float(np.linalg.norm(rhs))
    relative = residual / max(1.0, float(np.linalg.norm(rho_array)))

    return LindbladFinalStateVerification(
        density_matrix=density,
        lindblad_residual=residual,
        relative_lindblad_residual=relative,
    )


@dataclass(frozen=True, slots=True)
class MonitorKernelClosureDiagnostics:
    """Diagnostics for monitor-kernel closure under Hamiltonian mixing.

    The monitor kernel is ``intersection_i ker(M_i)``.  For a monitor-recycler
    design ``L_i = V_i M_i``, this kernel is always contained in the jump
    kernel and therefore measures what the recyclers cannot see directly.

    The first Hamiltonian-closure layer appends the constraints ``M_i H``.  If
    this sharply reduces the bad kernel, attraction is possible but can be slow
    because the Hamiltonian must rotate bad monitor-kernel states into the
    monitored subspace before dissipation acts.
    """

    dim: int
    n_monitors: int
    closure_order: int

    max_target_monitor_residual: float
    target_monitor_residuals: tuple[float, ...]

    monitor_kernel_dimension: int
    target_projection_onto_monitor_kernel: float
    target_distance_from_monitor_kernel: float
    target_in_monitor_kernel: bool
    bad_monitor_kernel_dimension: int
    bad_monitor_kernel_iprs: tuple[float, ...]

    bad_kernel_hamiltonian_leakage_norms: tuple[float, ...]
    min_bad_kernel_hamiltonian_leakage_norm: float | None
    mean_bad_kernel_hamiltonian_leakage_norm: float | None
    max_bad_kernel_hamiltonian_leakage_norm: float | None

    closure_kernel_dimension: int
    target_projection_onto_closure_kernel: float
    target_distance_from_closure_kernel: float
    target_in_closure_kernel: bool
    bad_closure_kernel_dimension: int
    bad_closure_kernel_iprs: tuple[float, ...]

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "dim": self.dim,
            "n_monitors": self.n_monitors,
            "closure_order": self.closure_order,
            "max_target_monitor_residual": self.max_target_monitor_residual,
            "monitor_kernel_dimension": self.monitor_kernel_dimension,
            "target_projection_onto_monitor_kernel": (self.target_projection_onto_monitor_kernel),
            "target_distance_from_monitor_kernel": (self.target_distance_from_monitor_kernel),
            "target_in_monitor_kernel": self.target_in_monitor_kernel,
            "bad_monitor_kernel_dimension": self.bad_monitor_kernel_dimension,
            "bad_monitor_kernel_iprs": self.bad_monitor_kernel_iprs,
            "bad_kernel_hamiltonian_leakage_norms": (self.bad_kernel_hamiltonian_leakage_norms),
            "min_bad_kernel_hamiltonian_leakage_norm": (
                self.min_bad_kernel_hamiltonian_leakage_norm
            ),
            "mean_bad_kernel_hamiltonian_leakage_norm": (
                self.mean_bad_kernel_hamiltonian_leakage_norm
            ),
            "max_bad_kernel_hamiltonian_leakage_norm": (
                self.max_bad_kernel_hamiltonian_leakage_norm
            ),
            "closure_kernel_dimension": self.closure_kernel_dimension,
            "target_projection_onto_closure_kernel": (self.target_projection_onto_closure_kernel),
            "target_distance_from_closure_kernel": (self.target_distance_from_closure_kernel),
            "target_in_closure_kernel": self.target_in_closure_kernel,
            "bad_closure_kernel_dimension": self.bad_closure_kernel_dimension,
            "bad_closure_kernel_iprs": self.bad_closure_kernel_iprs,
        }

    def to_rich(self):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "MonitorKernelClosureDiagnostics.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.dim))
        overview.add_row("number of monitors", str(self.n_monitors))
        overview.add_row("closure order", str(self.closure_order))

        monitor_table = Table(title="Monitor kernel")
        monitor_table.add_column("quantity", style="bold")
        monitor_table.add_column("value", justify="right")
        monitor_table.add_row("max ||M_i psi||", _format_float(self.max_target_monitor_residual))
        monitor_table.add_row("dim intersection ker M_i", str(self.monitor_kernel_dimension))
        monitor_table.add_row(
            "bad monitor-kernel dimension",
            str(self.bad_monitor_kernel_dimension),
        )
        monitor_table.add_row(
            "min H-leakage from bad kernel",
            _format_float_or_none(self.min_bad_kernel_hamiltonian_leakage_norm),
        )
        monitor_table.add_row(
            "mean H-leakage from bad kernel",
            _format_float_or_none(self.mean_bad_kernel_hamiltonian_leakage_norm),
        )
        monitor_table.add_row(
            "max H-leakage from bad kernel",
            _format_float_or_none(self.max_bad_kernel_hamiltonian_leakage_norm),
        )

        closure_table = Table(title="Hamiltonian closure")
        closure_table.add_column("quantity", style="bold")
        closure_table.add_column("value", justify="right")
        closure_table.add_row("dim ker{M_i, M_i H}", str(self.closure_kernel_dimension))
        closure_table.add_row(
            "bad closure-kernel dimension",
            str(self.bad_closure_kernel_dimension),
        )
        closure_table.add_row(
            "target distance from closure kernel",
            _format_float(self.target_distance_from_closure_kernel),
        )

        return Panel(
            Group(overview, monitor_table, closure_table),
            title=Text("Monitor-kernel closure diagnostics", style="bold cyan"),
            border_style="cyan",
        )


def diagnose_monitor_kernel_closure(
    *,
    hamiltonian: Any,
    monitors: Sequence[Any],
    target_state: npt.ArrayLike,
    closure_order: int = 1,
    tolerance: float = 1e-10,
) -> MonitorKernelClosureDiagnostics:
    """Diagnose whether local monitors are closed by Hamiltonian mixing.

    This is designed for monitor-recycler jumps ``L_i = V_i M_i``.  Recyclers
    cannot act on states in ``intersection_i ker(M_i)``, so the size of this
    kernel and its leakage under ``H`` are the first diagnostics to inspect.

    Currently ``closure_order`` supports ``0`` or ``1``.  Order 1 appends the
    constraints ``M_i H`` and computes the common kernel of ``{M_i, M_i H}``.
    """
    if closure_order not in (0, 1):
        raise ValueError("closure_order currently supports only 0 or 1.")

    hamiltonian_sparse = _as_scipy_csr_matrix(hamiltonian)
    monitor_sparse = tuple(_as_scipy_csr_matrix(monitor) for monitor in monitors)

    target = np.asarray(target_state, dtype=np.complex128)
    if target.ndim != 1:
        raise ValueError("target_state must be one-dimensional.")

    target_norm = float(np.linalg.norm(target))
    if target_norm == 0.0:
        raise ValueError("target_state must be nonzero.")

    target = target / target_norm
    dim = int(target.size)

    if hamiltonian_sparse.shape != (dim, dim):
        raise ValueError("hamiltonian shape must be compatible with target_state.")

    for monitor in monitor_sparse:
        if monitor.shape != (dim, dim):
            raise ValueError(
                "Every monitor must have shape " "(len(target_state), len(target_state))."
            )

    target_monitor_vectors = tuple(monitor @ target for monitor in monitor_sparse)
    target_monitor_residuals = tuple(
        float(np.linalg.norm(vector)) for vector in target_monitor_vectors
    )
    max_target_monitor_residual = max(target_monitor_residuals) if target_monitor_residuals else 0.0

    monitor_kernel_basis = _common_kernel_basis_from_sparse_operators(
        operators=monitor_sparse,
        dim=dim,
        tolerance=tolerance,
    )
    monitor_kernel_dimension = int(monitor_kernel_basis.shape[1])
    target_projection_onto_monitor_kernel = _projection_norm_onto_basis(
        vector=target,
        basis=monitor_kernel_basis,
    )
    target_distance_from_monitor_kernel = float(
        np.sqrt(max(0.0, 1.0 - target_projection_onto_monitor_kernel**2))
    )
    target_in_monitor_kernel = (
        target_distance_from_monitor_kernel <= np.sqrt(tolerance)
        or max_target_monitor_residual <= tolerance
    )

    bad_monitor_kernel_basis = _kernel_basis_orthogonal_to_target(
        basis=monitor_kernel_basis,
        target=target,
        tolerance=tolerance,
    )
    bad_monitor_kernel_dimension = int(bad_monitor_kernel_basis.shape[1])
    bad_monitor_kernel_iprs = tuple(
        _state_ipr(bad_monitor_kernel_basis[:, index])
        for index in range(bad_monitor_kernel_basis.shape[1])
    )
    bad_leakages = _monitor_hamiltonian_leakage_norms(
        hamiltonian=hamiltonian_sparse,
        monitors=monitor_sparse,
        basis=bad_monitor_kernel_basis,
    )

    if bad_leakages.size:
        min_bad_leakage = float(np.min(bad_leakages))
        mean_bad_leakage = float(np.mean(bad_leakages))
        max_bad_leakage = float(np.max(bad_leakages))
    else:
        min_bad_leakage = None
        mean_bad_leakage = None
        max_bad_leakage = None

    if closure_order == 0:
        closure_operators = monitor_sparse
    else:
        closure_operators = monitor_sparse + tuple(
            (monitor @ hamiltonian_sparse).tocsr() for monitor in monitor_sparse
        )

    closure_kernel_basis = _common_kernel_basis_from_sparse_operators(
        operators=closure_operators,
        dim=dim,
        tolerance=tolerance,
    )
    closure_kernel_dimension = int(closure_kernel_basis.shape[1])
    target_projection_onto_closure_kernel = _projection_norm_onto_basis(
        vector=target,
        basis=closure_kernel_basis,
    )
    target_distance_from_closure_kernel = float(
        np.sqrt(max(0.0, 1.0 - target_projection_onto_closure_kernel**2))
    )
    target_in_closure_kernel = target_distance_from_closure_kernel <= np.sqrt(tolerance)

    bad_closure_kernel_basis = _kernel_basis_orthogonal_to_target(
        basis=closure_kernel_basis,
        target=target,
        tolerance=tolerance,
    )
    bad_closure_kernel_dimension = int(bad_closure_kernel_basis.shape[1])
    bad_closure_kernel_iprs = tuple(
        _state_ipr(bad_closure_kernel_basis[:, index])
        for index in range(bad_closure_kernel_basis.shape[1])
    )

    return MonitorKernelClosureDiagnostics(
        dim=dim,
        n_monitors=len(monitor_sparse),
        closure_order=int(closure_order),
        max_target_monitor_residual=max_target_monitor_residual,
        target_monitor_residuals=target_monitor_residuals,
        monitor_kernel_dimension=monitor_kernel_dimension,
        target_projection_onto_monitor_kernel=target_projection_onto_monitor_kernel,
        target_distance_from_monitor_kernel=target_distance_from_monitor_kernel,
        target_in_monitor_kernel=target_in_monitor_kernel,
        bad_monitor_kernel_dimension=bad_monitor_kernel_dimension,
        bad_monitor_kernel_iprs=bad_monitor_kernel_iprs,
        bad_kernel_hamiltonian_leakage_norms=tuple(float(value) for value in bad_leakages),
        min_bad_kernel_hamiltonian_leakage_norm=min_bad_leakage,
        mean_bad_kernel_hamiltonian_leakage_norm=mean_bad_leakage,
        max_bad_kernel_hamiltonian_leakage_norm=max_bad_leakage,
        closure_kernel_dimension=closure_kernel_dimension,
        target_projection_onto_closure_kernel=target_projection_onto_closure_kernel,
        target_distance_from_closure_kernel=target_distance_from_closure_kernel,
        target_in_closure_kernel=target_in_closure_kernel,
        bad_closure_kernel_dimension=bad_closure_kernel_dimension,
        bad_closure_kernel_iprs=bad_closure_kernel_iprs,
    )


@dataclass(frozen=True, slots=True)
class DarkSubspaceDiagnostics:
    """Diagnostics for whether a dark target is unique/attractive."""

    dim: int
    n_jumps: int

    target_norm: float
    target_jump_residuals: tuple[float, ...]
    max_target_jump_residual: float
    target_liouvillian_residual: float

    common_jump_kernel_dimension: int
    target_projection_onto_common_kernel: float
    target_distance_from_common_kernel: float
    target_in_common_jump_kernel: bool
    bad_common_jump_kernel_dimension: int
    bad_common_jump_kernel_iprs: tuple[float, ...]

    liouvillian_zero_mode_count: int | None
    liouvillian_spectral_gap: float | None
    liouvillian_eigenvalues: tuple[complex, ...]

    likely_unique_dark_state: bool | None

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "dim": self.dim,
            "n_jumps": self.n_jumps,
            "max_target_jump_residual": self.max_target_jump_residual,
            "target_liouvillian_residual": self.target_liouvillian_residual,
            "common_jump_kernel_dimension": self.common_jump_kernel_dimension,
            "target_projection_onto_common_kernel": (self.target_projection_onto_common_kernel),
            "target_distance_from_common_kernel": (self.target_distance_from_common_kernel),
            "target_in_common_jump_kernel": self.target_in_common_jump_kernel,
            "bad_common_jump_kernel_dimension": (self.bad_common_jump_kernel_dimension),
            "bad_common_jump_kernel_iprs": self.bad_common_jump_kernel_iprs,
            "liouvillian_zero_mode_count": self.liouvillian_zero_mode_count,
            "liouvillian_spectral_gap": self.liouvillian_spectral_gap,
            "likely_unique_dark_state": self.likely_unique_dark_state,
        }

    def to_rich(self):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "DarkSubspaceDiagnostics.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()

        overview.add_row("Hilbert dimension", str(self.dim))
        overview.add_row("number of jumps", str(self.n_jumps))
        overview.add_row(
            "likely unique dark state",
            str(self.likely_unique_dark_state),
        )

        target = Table(title="Target checks")
        target.add_column("quantity", style="bold")
        target.add_column("value", justify="right")
        target.add_column("status", justify="center")

        target.add_row(
            "max ||J_mu psi||",
            _format_float(self.max_target_jump_residual),
            _status_for_residual(self.max_target_jump_residual),
        )
        target.add_row(
            "||L(rho_psi)||",
            _format_float(self.target_liouvillian_residual),
            _status_for_residual(self.target_liouvillian_residual),
        )

        jump_kernel = Table(title="Common jump kernel")
        jump_kernel.add_column("quantity", style="bold")
        jump_kernel.add_column("value", justify="right")

        jump_kernel.add_row(
            "dim intersection ker J_mu",
            str(self.common_jump_kernel_dimension),
        )
        jump_kernel.add_row(
            "projection of psi onto kernel",
            _format_float(self.target_projection_onto_common_kernel),
        )
        jump_kernel.add_row(
            "distance of psi from kernel",
            _format_float(self.target_distance_from_common_kernel),
        )
        jump_kernel.add_row(
            "target in common kernel",
            str(self.target_in_common_jump_kernel),
        )
        jump_kernel.add_row(
            "bad common-kernel dimension",
            str(self.bad_common_jump_kernel_dimension),
        )
        jump_kernel.add_row(
            "bad-kernel IPRs",
            _format_float_tuple(self.bad_common_jump_kernel_iprs),
        )

        liouvillian = Table(title="Liouvillian zero modes")
        liouvillian.add_column("quantity", style="bold")
        liouvillian.add_column("value", justify="right")

        liouvillian.add_row(
            "zero-mode count",
            (
                "not checked"
                if self.liouvillian_zero_mode_count is None
                else str(self.liouvillian_zero_mode_count)
            ),
        )
        liouvillian.add_row(
            "spectral gap",
            _format_float_or_none(self.liouvillian_spectral_gap),
        )

        return Panel(
            Group(overview, target, jump_kernel, liouvillian),
            title=Text("Dark-subspace diagnostics", style="bold cyan"),
            border_style="cyan",
        )


def diagnose_dark_subspace(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    target_state: npt.ArrayLike,
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
    kernel_tolerance: float = 1e-10,
    liouvillian_zero_tolerance: float = 1e-9,
    check_liouvillian_spectrum: bool = True,
    max_liouvillian_dense_dimension: int = 4096,
) -> DarkSubspaceDiagnostics:
    """Diagnose whether a dark target is likely unique/attractive.

    This is intended for small systems. It computes:

        1. target jump residuals ||J_mu psi||;
        2. common jump kernel dim intersection_mu ker J_mu;
        3. bad common-kernel dimension after removing the target direction;
        4. target Liouvillian residual ||L(|psi><psi|)||;
        5. optional Liouvillian zero-mode count.

    The Liouvillian spectrum check is dense and should only be used for small
    Hilbert spaces.
    """
    # _backend_obj = get_open_system_backend(backend)

    hamiltonian_sparse = _as_scipy_csr_matrix(hamiltonian)
    jumps_sparse = tuple(_as_scipy_csr_matrix(jump) for jump in jumps)

    target = np.asarray(target_state, dtype=np.complex128)
    if target.ndim != 1:
        raise ValueError("target_state must be one-dimensional.")

    target_norm = float(np.linalg.norm(target))
    if target_norm == 0.0:
        raise ValueError("target_state must be nonzero.")

    target = target / target_norm
    dim = int(target.size)

    if hamiltonian_sparse.shape != (dim, dim):
        raise ValueError("hamiltonian shape must be compatible with target_state.")

    for jump in jumps_sparse:
        if jump.shape != (dim, dim):
            raise ValueError(
                "Every jump operator must have shape " "(len(target_state), len(target_state))."
            )

    target_jump_vectors = tuple(jump @ target for jump in jumps_sparse)
    target_jump_residuals = tuple(float(np.linalg.norm(vector)) for vector in target_jump_vectors)
    max_target_jump_residual = max(target_jump_residuals) if target_jump_residuals else 0.0

    common_kernel_basis = _common_kernel_basis_from_sparse_operators(
        operators=jumps_sparse,
        dim=dim,
        tolerance=kernel_tolerance,
    )

    common_jump_kernel_dimension = int(common_kernel_basis.shape[1])

    target_projection_onto_common_kernel = _projection_norm_onto_basis(
        vector=target,
        basis=common_kernel_basis,
    )
    target_distance_from_common_kernel = float(
        np.sqrt(
            max(
                0.0,
                1.0 - target_projection_onto_common_kernel**2,
            )
        )
    )
    target_in_common_jump_kernel = (
        target_distance_from_common_kernel <= np.sqrt(kernel_tolerance)
        or max_target_jump_residual <= kernel_tolerance
    )

    bad_common_kernel_basis = _kernel_basis_orthogonal_to_target(
        basis=common_kernel_basis,
        target=target,
        tolerance=kernel_tolerance,
    )
    bad_common_jump_kernel_dimension = int(bad_common_kernel_basis.shape[1])
    bad_common_jump_kernel_iprs = tuple(
        _state_ipr(bad_common_kernel_basis[:, index])
        for index in range(bad_common_kernel_basis.shape[1])
    )

    target_liouvillian_residual = _rank_one_lindblad_rhs_norm(
        hamiltonian=hamiltonian_sparse,
        jumps=jumps_sparse,
        target=target,
        precomputed_jump_targets=target_jump_vectors,
    )

    liouvillian_zero_mode_count: int | None = None
    liouvillian_spectral_gap: float | None = None
    liouvillian_eigenvalues: tuple[complex, ...] = ()

    if check_liouvillian_spectrum:
        liouvillian_dimension = dim * dim

        if liouvillian_dimension > max_liouvillian_dense_dimension:
            raise ValueError(
                "Dense Liouvillian spectrum check is too expensive: "
                f"dim^2={liouvillian_dimension}, "
                f"max_liouvillian_dense_dimension="
                f"{max_liouvillian_dense_dimension}. "
                "Set check_liouvillian_spectrum=False or increase the limit."
            )

        liouvillian = build_liouvillian(
            hamiltonian_sparse,
            list(jumps_sparse),
            backend="scipy",
            sparse_format="csr",
        )
        liouvillian_dense = liouvillian.toarray()
        eigenvalues = scipy_linalg.eigvals(liouvillian_dense)

        eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
        eigenvalue_abs = np.abs(eigenvalues)

        liouvillian_zero_mode_count = int(
            np.count_nonzero(eigenvalue_abs <= liouvillian_zero_tolerance)
        )

        nonzero_abs = eigenvalue_abs[eigenvalue_abs > liouvillian_zero_tolerance]
        if nonzero_abs.size == 0:
            liouvillian_spectral_gap = None
        else:
            liouvillian_spectral_gap = float(np.min(nonzero_abs))

        # Store the smallest few eigenvalues for inspection.
        order = np.argsort(eigenvalue_abs)
        liouvillian_eigenvalues = tuple(
            complex(eigenvalues[index]) for index in order[: min(16, eigenvalues.size)]
        )

    likely_unique_dark_state: bool | None
    if liouvillian_zero_mode_count is None:
        likely_unique_dark_state = None
    else:
        likely_unique_dark_state = (
            liouvillian_zero_mode_count == 1
            and target_liouvillian_residual <= liouvillian_zero_tolerance
        )

    return DarkSubspaceDiagnostics(
        dim=dim,
        n_jumps=len(jumps_sparse),
        target_norm=target_norm,
        target_jump_residuals=target_jump_residuals,
        max_target_jump_residual=max_target_jump_residual,
        target_liouvillian_residual=target_liouvillian_residual,
        common_jump_kernel_dimension=common_jump_kernel_dimension,
        target_projection_onto_common_kernel=target_projection_onto_common_kernel,
        target_distance_from_common_kernel=target_distance_from_common_kernel,
        target_in_common_jump_kernel=target_in_common_jump_kernel,
        bad_common_jump_kernel_dimension=bad_common_jump_kernel_dimension,
        bad_common_jump_kernel_iprs=bad_common_jump_kernel_iprs,
        liouvillian_zero_mode_count=liouvillian_zero_mode_count,
        liouvillian_spectral_gap=liouvillian_spectral_gap,
        liouvillian_eigenvalues=liouvillian_eigenvalues,
        likely_unique_dark_state=likely_unique_dark_state,
    )


@dataclass(frozen=True, slots=True)
class AbsorbingProjectorJumpDiagnostics:
    """Diagnostics for one jump relative to a target projector."""

    jump_index: int
    target_residual: float
    outflow_norm: float
    inflow_norm: float
    commutator_norm: float
    dissipator_adjoint_projector_norm: float

    @property
    def is_dark_on_target(self) -> bool:
        return self.target_residual < 1e-10

    @property
    def has_inflow(self) -> bool:
        return self.inflow_norm > 1e-10


@dataclass(frozen=True, slots=True)
class AbsorbingProjectorSymmetryDiagnostics:
    """Diagnostics for the absorbing-state projector symmetry P_psi."""

    dim: int
    n_jumps: int
    hamiltonian_commutator_norm: float
    liouvillian_adjoint_projector_norm: float
    max_target_residual: float
    max_outflow_norm: float
    max_inflow_norm: float
    max_jump_projector_commutator_norm: float
    jump_diagnostics: tuple[AbsorbingProjectorJumpDiagnostics, ...]

    absorbing_projector_is_conserved: bool
    target_is_dark: bool
    has_recycling_inflow: bool
    has_absorbing_projector_symmetry: bool

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "dim": self.dim,
            "n_jumps": self.n_jumps,
            "hamiltonian_commutator_norm": self.hamiltonian_commutator_norm,
            "liouvillian_adjoint_projector_norm": (self.liouvillian_adjoint_projector_norm),
            "max_target_residual": self.max_target_residual,
            "max_outflow_norm": self.max_outflow_norm,
            "max_inflow_norm": self.max_inflow_norm,
            "max_jump_projector_commutator_norm": (self.max_jump_projector_commutator_norm),
            "absorbing_projector_is_conserved": (self.absorbing_projector_is_conserved),
            "target_is_dark": self.target_is_dark,
            "has_recycling_inflow": self.has_recycling_inflow,
            "has_absorbing_projector_symmetry": (self.has_absorbing_projector_symmetry),
            "jump_diagnostics": tuple(
                {
                    "jump_index": diagnostic.jump_index,
                    "target_residual": diagnostic.target_residual,
                    "outflow_norm": diagnostic.outflow_norm,
                    "inflow_norm": diagnostic.inflow_norm,
                    "commutator_norm": diagnostic.commutator_norm,
                    "dissipator_adjoint_projector_norm": (
                        diagnostic.dissipator_adjoint_projector_norm
                    ),
                }
                for diagnostic in self.jump_diagnostics
            ),
        }

    def to_rich(self):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "AbsorbingProjectorSymmetryDiagnostics.to_rich() "
                "requires rich. Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()

        overview.add_row("Hilbert dimension", str(self.dim))
        overview.add_row("number of jumps", str(self.n_jumps))
        overview.add_row(
            "target is dark",
            str(self.target_is_dark),
        )
        overview.add_row(
            "has recycling inflow",
            str(self.has_recycling_inflow),
        )
        overview.add_row(
            "P_psi conserved",
            str(self.absorbing_projector_is_conserved),
        )
        overview.add_row(
            "absorbing-projector symmetry",
            str(self.has_absorbing_projector_symmetry),
        )

        global_table = Table(title="Global projector diagnostics")
        global_table.add_column("quantity", style="bold")
        global_table.add_column("value", justify="right")

        global_table.add_row(
            "||[H, P_psi]||",
            _format_float(self.hamiltonian_commutator_norm),
        )
        global_table.add_row(
            "||L†(P_psi)||",
            _format_float(self.liouvillian_adjoint_projector_norm),
        )
        global_table.add_row(
            "max ||J psi||",
            _format_float(self.max_target_residual),
        )
        global_table.add_row(
            "max ||(I-P) J P||",
            _format_float(self.max_outflow_norm),
        )
        global_table.add_row(
            "max ||P J (I-P)||",
            _format_float(self.max_inflow_norm),
        )
        global_table.add_row(
            "max ||[J, P]||",
            _format_float(self.max_jump_projector_commutator_norm),
        )

        jumps = Table(title="Jump-by-jump projector diagnostics")
        jumps.add_column("jump", justify="right")
        jumps.add_column("||J psi||", justify="right")
        jumps.add_column("outflow", justify="right")
        jumps.add_column("inflow", justify="right")
        jumps.add_column("||[J,P]||", justify="right")
        jumps.add_column("||D†_J(P)||", justify="right")

        for diagnostic in self.jump_diagnostics:
            jumps.add_row(
                str(diagnostic.jump_index),
                _format_float(diagnostic.target_residual),
                _format_float(diagnostic.outflow_norm),
                _format_float(diagnostic.inflow_norm),
                _format_float(diagnostic.commutator_norm),
                _format_float(diagnostic.dissipator_adjoint_projector_norm),
            )

        return Panel(
            Group(overview, global_table, jumps),
            title=Text(
                "Absorbing-projector symmetry diagnostics",
                style="bold cyan",
            ),
            border_style="cyan",
        )


def diagnose_absorbing_projector_symmetry(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    target_state: npt.ArrayLike,
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
    tolerance: float = 1e-10,
) -> AbsorbingProjectorSymmetryDiagnostics:
    """Diagnose whether P_psi is an absorbing-state projector symmetry.

    The target projector is

        P_psi = |psi><psi|.

    The relevant obstruction to attraction is:

        J_mu |psi> = 0
        and
        P_psi J_mu (I - P_psi) = 0

    for all jumps. Then the target is dark, but there is no jump-induced
    inflow from psi_perp into psi. Equivalently, P_psi is conserved by the
    Heisenberg-picture Lindbladian.
    """
    # _backend_obj = get_open_system_backend(backend)

    hamiltonian_sparse = _as_scipy_csr_matrix(hamiltonian)
    jumps_sparse = tuple(_as_scipy_csr_matrix(jump) for jump in jumps)

    target = np.asarray(target_state, dtype=np.complex128)
    if target.ndim != 1:
        raise ValueError("target_state must be one-dimensional.")

    target_norm = float(np.linalg.norm(target))
    if target_norm == 0.0:
        raise ValueError("target_state must be nonzero.")

    target = target / target_norm
    dim = int(target.size)

    if hamiltonian_sparse.shape != (dim, dim):
        raise ValueError("hamiltonian shape must be compatible with target_state.")

    for jump in jumps_sparse:
        if jump.shape != (dim, dim):
            raise ValueError(
                "Every jump operator must have shape " "(len(target_state), len(target_state))."
            )

    hamiltonian_target = hamiltonian_sparse @ target
    hamiltonian_commutator_norm = _low_rank_operator_frobenius_norm(
        (
            (1.0, hamiltonian_target, target),
            (-1.0, target, hamiltonian_target),
        )
    )

    jump_diagnostics: list[AbsorbingProjectorJumpDiagnostics] = []

    liouvillian_adjoint_terms: list[tuple[complex, np.ndarray, np.ndarray]] = [
        (1j, hamiltonian_target, target),
        (-1j, target, hamiltonian_target),
    ]

    for jump_index, jump in enumerate(jumps_sparse):
        jump_target = jump @ target
        jump_dagger_target = jump.conj().T @ target
        jump_dagger_jump_target = jump.conj().T @ jump_target

        target_residual = float(np.linalg.norm(jump_target))
        outflow_norm = _orthogonal_component_norm(jump_target, target)
        inflow_norm = _orthogonal_component_norm(jump_dagger_target, target)
        commutator_norm = _low_rank_operator_frobenius_norm(
            (
                (1.0, jump_target, target),
                (-1.0, target, jump_dagger_target),
            )
        )

        dissipator_terms = (
            (1.0, jump_dagger_target, jump_dagger_target),
            (-0.5, jump_dagger_jump_target, target),
            (-0.5, target, jump_dagger_jump_target),
        )
        dissipator_adjoint_projector_norm = _low_rank_operator_frobenius_norm(dissipator_terms)

        liouvillian_adjoint_terms.extend(dissipator_terms)

        jump_diagnostics.append(
            AbsorbingProjectorJumpDiagnostics(
                jump_index=jump_index,
                target_residual=target_residual,
                outflow_norm=outflow_norm,
                inflow_norm=inflow_norm,
                commutator_norm=commutator_norm,
                dissipator_adjoint_projector_norm=dissipator_adjoint_projector_norm,
            )
        )

    max_target_residual = max(
        (diagnostic.target_residual for diagnostic in jump_diagnostics),
        default=0.0,
    )
    max_outflow_norm = max(
        (diagnostic.outflow_norm for diagnostic in jump_diagnostics),
        default=0.0,
    )
    max_inflow_norm = max(
        (diagnostic.inflow_norm for diagnostic in jump_diagnostics),
        default=0.0,
    )
    max_jump_projector_commutator_norm = max(
        (diagnostic.commutator_norm for diagnostic in jump_diagnostics),
        default=0.0,
    )

    liouvillian_adjoint_projector_norm = _low_rank_operator_frobenius_norm(
        tuple(liouvillian_adjoint_terms)
    )

    target_is_dark = max_target_residual <= tolerance
    has_recycling_inflow = max_inflow_norm > tolerance
    absorbing_projector_is_conserved = liouvillian_adjoint_projector_norm <= tolerance

    has_absorbing_projector_symmetry = (
        target_is_dark and not has_recycling_inflow and absorbing_projector_is_conserved
    )

    return AbsorbingProjectorSymmetryDiagnostics(
        dim=dim,
        n_jumps=len(jumps_sparse),
        hamiltonian_commutator_norm=hamiltonian_commutator_norm,
        liouvillian_adjoint_projector_norm=liouvillian_adjoint_projector_norm,
        max_target_residual=max_target_residual,
        max_outflow_norm=max_outflow_norm,
        max_inflow_norm=max_inflow_norm,
        max_jump_projector_commutator_norm=(max_jump_projector_commutator_norm),
        jump_diagnostics=tuple(jump_diagnostics),
        absorbing_projector_is_conserved=absorbing_projector_is_conserved,
        target_is_dark=target_is_dark,
        has_recycling_inflow=has_recycling_inflow,
        has_absorbing_projector_symmetry=(has_absorbing_projector_symmetry),
    )


def _as_scipy_csr_matrix(matrix: Any) -> scipy_sparse.csr_array:
    if scipy_sparse.issparse(matrix):
        return matrix.tocsr().astype(np.complex128)

    if hasattr(matrix, "get"):
        matrix = matrix.get()

    if hasattr(matrix, "toarray"):
        return scipy_sparse.csr_array(matrix.toarray(), dtype=np.complex128)

    if hasattr(matrix, "tocsr"):
        return matrix.tocsr().astype(np.complex128)

    return scipy_sparse.csr_array(np.asarray(matrix, dtype=np.complex128))


def _common_jump_kernel_basis_from_sparse_jumps(
    *,
    jumps: tuple[scipy_sparse.spmatrix, ...] | tuple[scipy_sparse.sparray, ...],
    dim: int,
    tolerance: float,
) -> np.ndarray:
    return _common_kernel_basis_from_sparse_operators(
        operators=jumps,
        dim=dim,
        tolerance=tolerance,
    )


def _common_kernel_basis_from_sparse_operators(
    *,
    operators: tuple[scipy_sparse.spmatrix, ...] | tuple[scipy_sparse.sparray, ...],
    dim: int,
    tolerance: float,
) -> np.ndarray:
    if len(operators) == 0:
        return np.eye(dim, dtype=np.complex128)

    rate_operator = scipy_sparse.csr_array((dim, dim), dtype=np.complex128)
    for operator in operators:
        rate_operator = rate_operator + operator.conj().T @ operator

    rate_operator = rate_operator.tocsr()
    graph = (abs(rate_operator) > tolerance).astype(np.int8)
    graph = (graph + graph.T).astype(np.int8)
    n_components, labels = connected_components(graph, directed=False)
    eigenvalue_threshold = max(tolerance, tolerance * tolerance)
    kernel_vectors: list[np.ndarray] = []

    for component_index in range(n_components):
        component_indices = np.flatnonzero(labels == component_index)
        if component_indices.size == 0:
            continue

        block = rate_operator[np.ix_(component_indices, component_indices)].toarray()
        block = 0.5 * (block + block.conj().T)

        if component_indices.size == 1:
            if float(np.real(block[0, 0])) <= eigenvalue_threshold:
                vector = np.zeros(dim, dtype=np.complex128)
                vector[component_indices[0]] = 1.0
                kernel_vectors.append(vector)
            continue

        eigenvalues, eigenvectors = np.linalg.eigh(block)
        for local_index in np.flatnonzero(eigenvalues <= eigenvalue_threshold):
            vector = np.zeros(dim, dtype=np.complex128)
            vector[component_indices] = eigenvectors[:, local_index]
            kernel_vectors.append(vector)

    if not kernel_vectors:
        return np.zeros((dim, 0), dtype=np.complex128)

    return np.column_stack(kernel_vectors).astype(np.complex128, copy=False)


def _monitor_hamiltonian_leakage_norms(
    *,
    hamiltonian: scipy_sparse.csr_array,
    monitors: tuple[scipy_sparse.spmatrix, ...] | tuple[scipy_sparse.sparray, ...],
    basis: np.ndarray,
) -> np.ndarray:
    if basis.size == 0 or basis.shape[1] == 0:
        return np.zeros(0, dtype=np.float64)

    hamiltonian_basis = hamiltonian @ basis
    squared_norms = np.zeros(basis.shape[1], dtype=np.float64)
    for monitor in monitors:
        image = monitor @ hamiltonian_basis
        squared_norms += np.sum(np.abs(image) ** 2, axis=0).real

    return np.sqrt(np.maximum(squared_norms, 0.0)).astype(np.float64, copy=False)


def _rank_one_lindblad_rhs_norm(
    *,
    hamiltonian: scipy_sparse.spmatrix | scipy_sparse.sparray,
    jumps: tuple[scipy_sparse.spmatrix, ...] | tuple[scipy_sparse.sparray, ...],
    target: np.ndarray,
    precomputed_jump_targets: tuple[np.ndarray, ...] | None = None,
) -> float:
    hamiltonian_target = hamiltonian @ target
    terms: list[tuple[complex, np.ndarray, np.ndarray]] = [
        (-1j, hamiltonian_target, target),
        (1j, target, hamiltonian_target),
    ]

    if precomputed_jump_targets is None:
        jump_targets = tuple(jump @ target for jump in jumps)
    else:
        jump_targets = precomputed_jump_targets

    for jump, jump_target in zip(jumps, jump_targets):
        jump_dagger_jump_target = jump.conj().T @ jump_target
        terms.extend(
            (
                (1.0, jump_target, jump_target),
                (-0.5, jump_dagger_jump_target, target),
                (-0.5, target, jump_dagger_jump_target),
            )
        )

    return _low_rank_operator_frobenius_norm(tuple(terms))


def _orthogonal_component_norm(vector: np.ndarray, basis_vector: np.ndarray) -> float:
    vector_norm_squared = float(np.real(np.vdot(vector, vector)))
    projection = np.vdot(basis_vector, vector)
    return float(np.sqrt(max(0.0, vector_norm_squared - abs(projection) ** 2)))


def _low_rank_operator_frobenius_norm(
    terms: Sequence[tuple[complex, np.ndarray, np.ndarray]],
) -> float:
    if len(terms) == 0:
        return 0.0

    norm_squared = 0.0 + 0.0j
    for coefficient_i, left_i, right_i in terms:
        for coefficient_j, left_j, right_j in terms:
            norm_squared += (
                np.conj(coefficient_i)
                * coefficient_j
                * np.vdot(left_i, left_j)
                * np.vdot(right_j, right_i)
            )

    return float(np.sqrt(max(0.0, float(np.real(norm_squared)))))


def _common_jump_kernel_basis(
    *,
    jumps: tuple[np.ndarray, ...],
    dim: int,
    tolerance: float,
) -> np.ndarray:
    if len(jumps) == 0:
        return np.eye(dim, dtype=np.complex128)

    stacked = np.vstack(jumps)
    return _nullspace_basis(stacked, tolerance=tolerance)


def _nullspace_basis(
    matrix: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    if matrix.size == 0:
        return np.eye(matrix.shape[1], dtype=np.complex128)

    _left_vectors, singular_values, right_vectors_dagger = np.linalg.svd(
        matrix,
        full_matrices=True,
    )

    n_columns = matrix.shape[1]
    rank = int(np.count_nonzero(singular_values > tolerance))

    if rank >= n_columns:
        return np.zeros((n_columns, 0), dtype=np.complex128)

    return (
        right_vectors_dagger.conj()
        .T[:, rank:]
        .astype(
            np.complex128,
            copy=False,
        )
    )


def _projection_norm_onto_basis(
    *,
    vector: np.ndarray,
    basis: np.ndarray,
) -> float:
    if basis.shape[1] == 0:
        return 0.0

    coefficients = basis.conj().T @ vector
    return float(np.linalg.norm(coefficients))


def _kernel_basis_orthogonal_to_target(
    *,
    basis: np.ndarray,
    target: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    if basis.shape[1] == 0:
        return np.zeros((target.size, 0), dtype=np.complex128)

    target = target / np.linalg.norm(target)

    projected = basis - np.outer(target, target.conj() @ basis)

    # Remove numerically zero columns before QR/SVD.
    column_norms = np.linalg.norm(projected, axis=0)
    keep = column_norms > tolerance

    if not np.any(keep):
        return np.zeros((target.size, 0), dtype=np.complex128)

    projected = projected[:, keep]

    return _orthonormal_column_basis(
        projected,
        tolerance=tolerance,
    )


def _orthonormal_column_basis(
    matrix: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.complex128)

    left_vectors, singular_values, _right_vectors_dagger = np.linalg.svd(
        matrix,
        full_matrices=False,
    )

    rank = int(np.count_nonzero(singular_values > tolerance))

    if rank == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.complex128)

    return left_vectors[:, :rank].astype(np.complex128, copy=False)


def _state_ipr(state: np.ndarray) -> float:
    norm = float(np.linalg.norm(state))

    if norm == 0.0:
        return 0.0

    normalized = state / norm
    probabilities = np.abs(normalized) ** 2
    return float(np.sum(probabilities**2))


def _format_float(value: float) -> str:
    return f"{value:.3e}"


def _format_float_or_none(value: float | None) -> str:
    if value is None:
        return "not checked"

    return _format_float(float(value))


def _format_float_tuple(
    values: tuple[float, ...],
    *,
    max_items: int = 8,
) -> str:
    if len(values) == 0:
        return "∅"

    if len(values) <= max_items:
        return ", ".join(_format_float(value) for value in values)

    head = ", ".join(_format_float(value) for value in values[:max_items])
    return f"{head}, ... ({len(values)} total)"


def _status_for_residual(
    value: float | None,
    *,
    excellent: float = 1e-12,
    acceptable: float = 1e-8,
) -> str:
    if value is None:
        return "[dim]n/a[/dim]"

    if value <= excellent:
        return "[green]ok[/green]"

    if value <= acceptable:
        return "[yellow]warn[/yellow]"

    return "[red]large[/red]"
