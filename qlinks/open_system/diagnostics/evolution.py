from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from qlinks.open_system.backend import OpenSystemBackend, OpenSystemBackendName
from qlinks.open_system.diagnostics.verification import verify_density_matrix
from qlinks.open_system.operators import lindblad_rhs_density_matrix


@dataclass(frozen=True, slots=True)
class EvolutionDiagnostics:
    """Diagnostics for density-matrix or MCWF evolution output.

    Attributes:
        trace_errors: Absolute errors of ``Tr(rho)`` from one.
        hermiticity_errors: Frobenius norm of anti-Hermitian parts.
        min_eigenvalues: Smallest density-matrix eigenvalue at each time.
        purities: ``Tr(rho^2)`` values.
        fidelities: Optional fidelity with a target state/density matrix.
        lindblad_residuals: Optional norm of the Lindblad RHS at each time.
        times: Optional time grid.
        source: Description of the analyzed data source.
        density_check_mode: Density check strategy used by the analyzer.
        trajectory_counts: Optional number of trajectories per time point.
        state_norm_errors: Optional norm errors for pure-state trajectories.
    """

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
    """Analyze diagnostics along Lindblad or MCWF evolution output.

    The function accepts dense density matrices directly, but it can also read
    MCWF ensemble outputs.  When an ensemble stores low-rank state snapshots but
    not ``rho_t``, diagnostics can be computed without materializing dense
    density matrices unless ``density_check_mode="full"`` is requested.

    Args:
        density_matrices: Optional sequence of density matrices.
        ensemble_result: Optional MCWF ensemble result.  The analyzer prefers
            populated ``rho_t``, then ``state_snapshots``, then stored
            trajectories.
        state_snapshots: Optional sequence of matrices with shape
            ``(dim, n_trajectories)``.
        trajectories: Optional trajectory results with stored states.
        times: Optional time grid.
        target_state: Optional pure state for fidelity diagnostics.
        hamiltonian: Optional Hamiltonian for Lindblad residual diagnostics.
        jumps: Optional jump operators for Lindblad residual diagnostics.
        atol: Numerical tolerance for density-matrix checks.
        backend: Open-system backend name or object.
        density_check_mode: ``"auto"``, ``"full"``, or ``"low_rank"``.
            ``"low_rank"`` avoids materializing density matrices from MCWF
            snapshots and reports ``NaN`` for minimum eigenvalues.

    Returns:
        Evolution diagnostics arrays.
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
