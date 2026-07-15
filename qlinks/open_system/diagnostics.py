from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import scipy.linalg as scipy_linalg
import scipy.sparse as scipy_sparse
import scipy.sparse.linalg as scipy_sparse_linalg
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


def _orthonormal_target_basis(
    target_states: npt.ArrayLike,
    *,
    dim: int | None = None,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return orthonormal target states as columns."""
    matrix = np.asarray(target_states, dtype=np.complex128)

    if matrix.ndim == 1:
        if dim is not None and matrix.size != dim:
            raise ValueError(f"target state has dimension {matrix.size}; expected {dim}.")
        matrix = matrix.reshape(matrix.size, 1)
    elif matrix.ndim == 2:
        if dim is not None:
            if matrix.shape[0] == dim:
                pass
            elif matrix.shape[1] == dim:
                matrix = matrix.T
            else:
                raise ValueError(
                    "target states must have shape (dim, n_states) or " "(n_states, dim)."
                )
        elif matrix.shape[0] < matrix.shape[1]:
            # Common notebook convention for a small manifold is one state per row.
            matrix = matrix.T
    else:
        raise ValueError("target states must be one- or two-dimensional.")

    if matrix.shape[1] == 0:
        raise ValueError("target_states must contain at least one state.")

    q, r = np.linalg.qr(matrix)
    diagonal = np.abs(np.diag(r))
    rank = int(np.count_nonzero(diagonal > tolerance))
    if rank == 0:
        raise ValueError("target_states have numerical rank zero.")

    return np.asarray(q[:, :rank], dtype=np.complex128)


def target_manifold_projector(
    target_states: npt.ArrayLike,
    *,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return the projector onto a target manifold.

    ``target_states`` may be one target vector, a ``(dim, n_states)`` matrix, or
    an ``(n_states, dim)`` matrix.  The columns/rows are orthonormalized before
    building the projector, so linearly dependent target vectors are harmless.
    """
    target_basis = _orthonormal_target_basis(target_states, tolerance=tolerance)
    return target_basis @ target_basis.conj().T


def target_manifold_weight(
    density_matrix: npt.ArrayLike,
    *,
    target_states: npt.ArrayLike | None = None,
    target_basis: npt.ArrayLike | None = None,
    projector: npt.ArrayLike | None = None,
    tolerance: float = 1.0e-10,
) -> float:
    """Return ``Tr(P_target rho)`` for one density matrix.

    Pass either ``target_states``/``target_basis`` or an explicit ``projector``.
    When a target basis is supplied, the computation uses
    ``Tr(Q^dagger rho Q)`` and avoids materializing the full projector.
    """
    rho = np.asarray(density_matrix, dtype=np.complex128)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("density_matrix must be square.")

    n_target_specs = sum(spec is not None for spec in (target_states, target_basis, projector))
    if n_target_specs != 1:
        raise ValueError("Pass exactly one of target_states, target_basis, or projector.")

    if projector is not None:
        p_target = np.asarray(projector, dtype=np.complex128)
        if p_target.shape != rho.shape:
            raise ValueError("projector must have the same shape as density_matrix.")
        value = np.trace(p_target @ rho)
        return float(np.real_if_close(value).real)

    raw_basis = target_basis if target_basis is not None else target_states
    q = _orthonormal_target_basis(raw_basis, dim=rho.shape[0], tolerance=tolerance)
    value = np.trace(q.conj().T @ rho @ q)
    return float(np.real_if_close(value).real)


def target_manifold_weight_series(
    *,
    density_matrices: Sequence[npt.ArrayLike] | None = None,
    evolution_result: Any | None = None,
    ensemble_result: Any | None = None,
    state_snapshots: Sequence[npt.ArrayLike] | None = None,
    target_states: npt.ArrayLike | None = None,
    target_basis: npt.ArrayLike | None = None,
    projector: npt.ArrayLike | None = None,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return ``Tr(P_target rho(t))`` for evolution or MCWF output.

    Exactly one data source must be supplied: ``density_matrices``, a
    ``LindbladEvolutionResult`` via ``evolution_result``, an ``EnsembleResult``
    via ``ensemble_result``, or MCWF ``state_snapshots``.  For state snapshots,
    each snapshot is expected to be a ``(dim, n_trajectories)`` state matrix and
    the returned weight is the trajectory average of ``<psi|P_target|psi>``.
    """
    sources = (density_matrices, evolution_result, ensemble_result, state_snapshots)
    if sum(source is not None for source in sources) != 1:
        raise ValueError(
            "Pass exactly one of density_matrices, evolution_result, "
            "ensemble_result, or state_snapshots."
        )

    resolved_density_matrices = density_matrices
    resolved_state_snapshots = state_snapshots

    if evolution_result is not None:
        resolved_density_matrices = getattr(evolution_result, "density_matrices", None)
        if resolved_density_matrices is None:
            raise ValueError("evolution_result does not contain density_matrices.")

    if ensemble_result is not None:
        rho_t = tuple(getattr(ensemble_result, "rho_t", ()) or ())
        if rho_t:
            resolved_density_matrices = rho_t
        else:
            snapshots = getattr(ensemble_result, "state_snapshots", None)
            if snapshots is None:
                raise ValueError(
                    "ensemble_result must contain rho_t or state_snapshots for "
                    "target-manifold weights."
                )
            resolved_state_snapshots = tuple(snapshots)

    n_target_specs = sum(spec is not None for spec in (target_states, target_basis, projector))
    if n_target_specs != 1:
        raise ValueError("Pass exactly one of target_states, target_basis, or projector.")

    if resolved_density_matrices is not None:
        return np.asarray(
            [
                target_manifold_weight(
                    density_matrix,
                    target_states=target_states,
                    target_basis=target_basis,
                    projector=projector,
                    tolerance=tolerance,
                )
                for density_matrix in resolved_density_matrices
            ],
            dtype=np.float64,
        )

    assert resolved_state_snapshots is not None
    snapshot_tuple = tuple(resolved_state_snapshots)
    if len(snapshot_tuple) == 0:
        raise ValueError("state_snapshots must be non-empty.")

    first_snapshot = np.asarray(snapshot_tuple[0], dtype=np.complex128)
    if first_snapshot.ndim != 2:
        raise ValueError("state snapshots must be 2D state matrices.")

    if projector is not None:
        p_target = np.asarray(projector, dtype=np.complex128)
        if p_target.shape != (first_snapshot.shape[0], first_snapshot.shape[0]):
            raise ValueError("projector has incompatible shape for state_snapshots.")
        weights = []
        for snapshot in snapshot_tuple:
            states = np.asarray(snapshot, dtype=np.complex128)
            values = np.einsum("ij,ij->j", states.conj(), p_target @ states)
            weights.append(float(np.real(np.mean(values))))
        return np.asarray(weights, dtype=np.float64)

    raw_basis = target_basis if target_basis is not None else target_states
    q = _orthonormal_target_basis(
        raw_basis,
        dim=first_snapshot.shape[0],
        tolerance=tolerance,
    )
    weights = []
    for snapshot in snapshot_tuple:
        states = np.asarray(snapshot, dtype=np.complex128)
        if states.ndim != 2 or states.shape[0] != q.shape[0]:
            raise ValueError("state snapshot has incompatible shape.")
        overlaps = q.conj().T @ states
        values = np.sum(np.abs(overlaps) ** 2, axis=0)
        weights.append(float(np.real(np.mean(values))))
    return np.asarray(weights, dtype=np.float64)


def _resolve_density_matrix_or_snapshot_series(
    *,
    density_matrices: Sequence[npt.ArrayLike] | None = None,
    evolution_result: Any | None = None,
    ensemble_result: Any | None = None,
    state_snapshots: Sequence[npt.ArrayLike] | None = None,
) -> tuple[str, tuple[Any, ...]]:
    """Resolve one density/state time-series source.

    The returned source kind is either ``"density_matrices"`` or
    ``"state_snapshots"``.  This helper intentionally accepts duck-typed solver
    outputs so diagnostics can be used with lightweight notebook objects.
    """
    sources = (density_matrices, evolution_result, ensemble_result, state_snapshots)
    if sum(source is not None for source in sources) != 1:
        raise ValueError(
            "Pass exactly one of density_matrices, evolution_result, "
            "ensemble_result, or state_snapshots."
        )

    if evolution_result is not None:
        resolved_density_matrices = getattr(evolution_result, "density_matrices", None)
        if resolved_density_matrices is None:
            raise ValueError("evolution_result does not contain density_matrices.")
        return "density_matrices", tuple(resolved_density_matrices)

    if ensemble_result is not None:
        rho_t = tuple(getattr(ensemble_result, "rho_t", ()) or ())
        if rho_t:
            return "density_matrices", rho_t
        snapshots = getattr(ensemble_result, "state_snapshots", None)
        if snapshots is None:
            raise ValueError(
                "ensemble_result must contain rho_t or state_snapshots for "
                "target-manifold diagnostics."
            )
        return "state_snapshots", tuple(snapshots)

    if density_matrices is not None:
        return "density_matrices", tuple(density_matrices)

    assert state_snapshots is not None
    return "state_snapshots", tuple(state_snapshots)


def target_manifold_density_matrix(
    density_matrix: npt.ArrayLike,
    *,
    target_states: npt.ArrayLike | None = None,
    target_basis: npt.ArrayLike | None = None,
    normalize: bool = True,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return the density matrix reduced to the target manifold basis.

    The returned matrix is ``Q^dagger rho Q`` where columns of ``Q`` are an
    orthonormal target basis.  When ``normalize=True`` this is conditioned on
    being in the manifold by dividing by ``Tr(Q^dagger rho Q)``.  If the weight
    is numerically zero, the unnormalized zero matrix is returned.
    """
    rho = np.asarray(density_matrix, dtype=np.complex128)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("density_matrix must be square.")

    if (target_states is None) == (target_basis is None):
        raise ValueError("Pass exactly one of target_states or target_basis.")

    raw_basis = target_basis if target_basis is not None else target_states
    q = _orthonormal_target_basis(raw_basis, dim=rho.shape[0], tolerance=tolerance)
    reduced = np.asarray(q.conj().T @ rho @ q, dtype=np.complex128)

    if normalize:
        weight = float(np.real_if_close(np.trace(reduced)).real)
        if abs(weight) > tolerance:
            reduced = reduced / weight

    return reduced


def _target_manifold_density_matrix_from_snapshot(
    snapshot: npt.ArrayLike,
    *,
    target_basis: npt.ArrayLike,
    normalize: bool,
    tolerance: float,
) -> np.ndarray:
    states = np.asarray(snapshot, dtype=np.complex128)
    if states.ndim != 2:
        raise ValueError("state snapshots must be 2D state matrices.")

    q = _orthonormal_target_basis(
        target_basis,
        dim=states.shape[0],
        tolerance=tolerance,
    )
    overlaps = q.conj().T @ states
    reduced = np.asarray(overlaps @ overlaps.conj().T, dtype=np.complex128)
    reduced /= float(states.shape[1])

    if normalize:
        weight = float(np.real_if_close(np.trace(reduced)).real)
        if abs(weight) > tolerance:
            reduced = reduced / weight

    return reduced


def target_manifold_density_matrix_series(
    *,
    density_matrices: Sequence[npt.ArrayLike] | None = None,
    evolution_result: Any | None = None,
    ensemble_result: Any | None = None,
    state_snapshots: Sequence[npt.ArrayLike] | None = None,
    target_states: npt.ArrayLike | None = None,
    target_basis: npt.ArrayLike | None = None,
    normalize: bool = True,
    tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return ``Q^dagger rho(t) Q`` for a target manifold basis.

    The result has shape ``(n_times, manifold_dimension, manifold_dimension)``.
    With ``normalize=True`` each time slice is conditioned by its target-manifold
    weight, i.e. it is divided by ``Tr(P_target rho(t))`` when the weight is
    nonzero.
    """
    if (target_states is None) == (target_basis is None):
        raise ValueError("Pass exactly one of target_states or target_basis.")

    raw_basis = target_basis if target_basis is not None else target_states
    source_kind, values = _resolve_density_matrix_or_snapshot_series(
        density_matrices=density_matrices,
        evolution_result=evolution_result,
        ensemble_result=ensemble_result,
        state_snapshots=state_snapshots,
    )
    if not values:
        raise ValueError(f"{source_kind} must be non-empty.")

    if source_kind == "density_matrices":
        return np.stack(
            [
                target_manifold_density_matrix(
                    density_matrix,
                    target_basis=raw_basis,
                    normalize=normalize,
                    tolerance=tolerance,
                )
                for density_matrix in values
            ]
        )

    first_snapshot = np.asarray(values[0], dtype=np.complex128)
    if first_snapshot.ndim != 2:
        raise ValueError("state snapshots must be 2D state matrices.")
    q = _orthonormal_target_basis(
        raw_basis,
        dim=first_snapshot.shape[0],
        tolerance=tolerance,
    )
    return np.stack(
        [
            _target_manifold_density_matrix_from_snapshot(
                snapshot,
                target_basis=q,
                normalize=normalize,
                tolerance=tolerance,
            )
            for snapshot in values
        ]
    )


def target_manifold_populations_series(**kwargs: Any) -> np.ndarray:
    """Return diagonal populations of the target-manifold density matrices."""
    reduced = target_manifold_density_matrix_series(**kwargs)
    return np.real(np.diagonal(reduced, axis1=1, axis2=2))


def target_manifold_coherence_series(
    *,
    norm: Literal["fro", "l1"] = "fro",
    **kwargs: Any,
) -> np.ndarray:
    """Return off-diagonal coherence of target-manifold density matrices.

    ``norm="fro"`` returns the Frobenius norm of off-diagonal entries, while
    ``norm="l1"`` returns their elementwise absolute sum.
    """
    reduced = target_manifold_density_matrix_series(**kwargs)
    offdiag = reduced.copy()
    indices = np.arange(offdiag.shape[1])
    offdiag[:, indices, indices] = 0.0

    if norm == "fro":
        return np.linalg.norm(offdiag.reshape(offdiag.shape[0], -1), axis=1)
    if norm == "l1":
        return np.sum(np.abs(offdiag), axis=(1, 2))
    raise ValueError("norm must be 'fro' or 'l1'.")


def target_manifold_purity_series(**kwargs: Any) -> np.ndarray:
    """Return ``Tr(rho_target(t)^2)`` for target-manifold density matrices."""
    reduced = target_manifold_density_matrix_series(**kwargs)
    values = np.einsum("tij,tji->t", reduced, reduced)
    return np.real_if_close(values).real.astype(np.float64, copy=False)


def target_manifold_entropy_series(
    *,
    base: float | None = None,
    tolerance: float = 1.0e-12,
    **kwargs: Any,
) -> np.ndarray:
    """Return von Neumann entropy of target-manifold density matrices.

    The input density matrices are usually conditioned by leaving
    ``normalize=True`` in ``kwargs``.  Eigenvalues below ``tolerance`` are
    ignored in the logarithm.
    """
    reduced = target_manifold_density_matrix_series(**kwargs)
    entropies: list[float] = []
    log_base = 1.0 if base is None else float(np.log(base))
    for rho in reduced:
        hermitian = 0.5 * (rho + rho.conj().T)
        eigvals = np.linalg.eigvalsh(hermitian)
        eigvals = np.real(eigvals[eigvals > tolerance])
        if eigvals.size == 0:
            entropies.append(0.0)
            continue
        entropy = -float(np.sum(eigvals * np.log(eigvals)))
        entropies.append(entropy / log_base)
    return np.asarray(entropies, dtype=np.float64)


def jump_activity(
    density_matrix: npt.ArrayLike,
    jumps: Sequence[npt.ArrayLike],
) -> float:
    """Return total Lindblad jump activity ``sum_mu Tr(J_mu^dag J_mu rho)``."""
    rho = np.asarray(density_matrix, dtype=np.complex128)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("density_matrix must be square.")

    total = 0.0
    for jump in jumps:
        if scipy_sparse.issparse(jump):
            jump_matrix = jump.tocsr()
            jumped = jump_matrix @ rho
            value = np.trace(jumped @ jump_matrix.conj().T)
        else:
            jump_matrix = np.asarray(jump, dtype=np.complex128)
            if jump_matrix.shape != rho.shape:
                raise ValueError("jump has incompatible shape.")
            jumped = jump_matrix @ rho
            value = np.trace(jumped @ jump_matrix.conj().T)
        total += float(np.real_if_close(value).real)
    return total


def _jump_activity_from_snapshot(
    snapshot: npt.ArrayLike,
    jumps: Sequence[npt.ArrayLike],
) -> float:
    states = np.asarray(snapshot, dtype=np.complex128)
    if states.ndim != 2:
        raise ValueError("state snapshots must be 2D state matrices.")

    total = 0.0
    for jump in jumps:
        jumped = jump @ states
        values = np.sum(np.abs(jumped) ** 2, axis=0)
        total += float(np.real(np.mean(values)))
    return total


def jump_activity_series(
    *,
    jumps: Sequence[npt.ArrayLike],
    density_matrices: Sequence[npt.ArrayLike] | None = None,
    evolution_result: Any | None = None,
    ensemble_result: Any | None = None,
    state_snapshots: Sequence[npt.ArrayLike] | None = None,
) -> np.ndarray:
    """Return total jump activity for each time point.

    For density matrices this evaluates
    ``sum_mu Tr(J_mu^dagger J_mu rho(t))``.  For MCWF state snapshots it returns
    the trajectory average of ``sum_mu ||J_mu |psi(t)>||^2``.
    """
    source_kind, values = _resolve_density_matrix_or_snapshot_series(
        density_matrices=density_matrices,
        evolution_result=evolution_result,
        ensemble_result=ensemble_result,
        state_snapshots=state_snapshots,
    )
    if not values:
        raise ValueError(f"{source_kind} must be non-empty.")

    if source_kind == "density_matrices":
        return np.asarray([jump_activity(rho, jumps) for rho in values], dtype=np.float64)

    return np.asarray(
        [_jump_activity_from_snapshot(snapshot, jumps) for snapshot in values],
        dtype=np.float64,
    )


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


@dataclass(frozen=True, slots=True)
class DensityMatrixVerification:
    """Numerical checks for a candidate density matrix.

    Attributes:
        trace: Matrix trace.
        trace_error: Absolute error of the trace from one.
        hermiticity_error: Frobenius norm of ``rho - rho^dagger``.
        min_eigenvalue: Smallest Hermitian eigenvalue after symmetrization.
        purity: ``Tr(rho^2)``.
        fidelity_with_target: Optional fidelity ``<psi|rho|psi>``.
        is_hermitian: Whether the Hermiticity check passes.
        is_trace_one: Whether the trace-one check passes.
        is_positive_semidefinite: Whether the minimum eigenvalue is above
            ``-atol``.
        is_density_matrix: Combined validity flag.
    """

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
    """Check whether an array is a valid density matrix.

    Args:
        rho: Candidate density matrix.
        target_state: Optional pure state used to compute ``<psi|rho|psi>``.
        atol: Absolute tolerance for trace, Hermiticity, and positivity checks.

    Returns:
        Verification record with scalar diagnostics and boolean flags.

    Raises:
        ValueError: If ``rho`` is not square or ``target_state`` is invalid.
    """
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
    """Verification of a final Lindblad density matrix.

    Attributes:
        density_matrix: Basic density-matrix validity diagnostics.
        lindblad_residual: Norm of the Lindblad RHS evaluated at the final
            state.
        relative_lindblad_residual: Residual normalized by the density-matrix
            norm scale.
    """

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
    """Verify density-matrix validity and Lindblad stationarity.

    Args:
        rho: Candidate final density matrix.
        hamiltonian: Hamiltonian matrix.
        jumps: Lindblad jump operators.
        target_state: Optional pure state for fidelity diagnostics.
        atol: Absolute tolerance passed to :func:`verify_density_matrix`.
        backend: Open-system backend name or object.

    Returns:
        Final-state verification record.
    """
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
    liouvillian_zero_mode_count_is_lower_bound: bool
    liouvillian_spectral_gap: float | None
    liouvillian_decay_gap: float | None
    liouvillian_peripheral_mode_count: int | None
    liouvillian_spectrum_method: str
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
            "liouvillian_zero_mode_count_is_lower_bound": (
                self.liouvillian_zero_mode_count_is_lower_bound
            ),
            "liouvillian_spectral_gap": self.liouvillian_spectral_gap,
            "liouvillian_decay_gap": self.liouvillian_decay_gap,
            "liouvillian_peripheral_mode_count": self.liouvillian_peripheral_mode_count,
            "liouvillian_spectrum_method": self.liouvillian_spectrum_method,
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
                else (
                    str(self.liouvillian_zero_mode_count)
                    + ("+" if self.liouvillian_zero_mode_count_is_lower_bound else "")
                )
            ),
        )
        liouvillian.add_row(
            "spectrum method",
            self.liouvillian_spectrum_method,
        )
        liouvillian.add_row(
            "absolute spectral gap",
            _format_float_or_none(self.liouvillian_spectral_gap),
        )
        liouvillian.add_row(
            "decay gap",
            _format_float_or_none(self.liouvillian_decay_gap),
        )
        liouvillian.add_row(
            "peripheral mode count",
            (
                "not checked"
                if self.liouvillian_peripheral_mode_count is None
                else str(self.liouvillian_peripheral_mode_count)
            ),
        )

        return Panel(
            Group(overview, target, jump_kernel, liouvillian),
            title=Text("Dark-subspace diagnostics", style="bold cyan"),
            border_style="cyan",
        )


@dataclass(frozen=True, slots=True)
class DarkManifoldDiagnostics:
    """Diagnostics for an attractive dark manifold/DFS target.

    The target is a column-orthonormal basis ``M`` and the target projector is
    ``P_M = M M†``.  Unlike :class:`DarkSubspaceDiagnostics`, this report does
    not expect a unique pure steady state.  Internal zero or imaginary-axis
    Liouvillian modes generated by the projected Hamiltonian on the target
    manifold are treated as expected modes; only additional non-decaying modes
    outside the target manifold are flagged as obstructions.
    """

    dim: int
    n_jumps: int
    manifold_dimension: int

    hamiltonian_closure_residual: float
    target_jump_residuals: tuple[float, ...]
    max_target_jump_residual: float
    target_density_liouvillian_residual: float
    inflow_norm: float

    common_jump_kernel_dimension: int
    target_projection_onto_common_kernel: float
    target_distance_from_common_kernel: float
    target_in_common_jump_kernel: bool
    bad_common_jump_kernel_dimension: int
    bad_common_jump_kernel_iprs: tuple[float, ...]

    internal_hamiltonian_eigenvalues: tuple[complex, ...]
    expected_internal_liouvillian_eigenvalues: tuple[complex, ...]
    expected_internal_zero_mode_count: int
    expected_internal_peripheral_mode_count: int

    liouvillian_zero_mode_count: int | None
    liouvillian_zero_mode_count_is_lower_bound: bool
    liouvillian_spectral_gap: float | None
    liouvillian_decay_gap: float | None
    liouvillian_peripheral_mode_count: int | None
    liouvillian_spectrum_method: str
    liouvillian_eigenvalues: tuple[complex, ...]

    matched_internal_nondecaying_mode_count: int | None
    missing_internal_nondecaying_mode_count: int | None
    extra_nondecaying_mode_count: int | None
    extra_zero_mode_count: int | None
    external_decay_gap: float | None

    likely_attractive_dark_manifold: bool | None

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "dim": self.dim,
            "n_jumps": self.n_jumps,
            "manifold_dimension": self.manifold_dimension,
            "h_closure_residual": self.hamiltonian_closure_residual,
            "max_target_jump_residual": self.max_target_jump_residual,
            "target_density_liouvillian_residual": self.target_density_liouvillian_residual,
            "inflow_norm": self.inflow_norm,
            "common_jump_kernel_dimension": self.common_jump_kernel_dimension,
            "target_projection_onto_common_kernel": self.target_projection_onto_common_kernel,
            "target_distance_from_common_kernel": self.target_distance_from_common_kernel,
            "target_in_common_jump_kernel": self.target_in_common_jump_kernel,
            "bad_common_jump_kernel_dimension": self.bad_common_jump_kernel_dimension,
            "bad_common_jump_kernel_iprs": self.bad_common_jump_kernel_iprs,
            "internal_hamiltonian_eigenvalues": [
                complex(value) for value in self.internal_hamiltonian_eigenvalues
            ],
            "expected_internal_zero_mode_count": self.expected_internal_zero_mode_count,
            "expected_internal_peripheral_mode_count": (
                self.expected_internal_peripheral_mode_count
            ),
            "liouvillian_zero_mode_count": self.liouvillian_zero_mode_count,
            "liouvillian_zero_mode_count_is_lower_bound": (
                self.liouvillian_zero_mode_count_is_lower_bound
            ),
            "liouvillian_spectral_gap": self.liouvillian_spectral_gap,
            "liouvillian_decay_gap": self.liouvillian_decay_gap,
            "liouvillian_peripheral_mode_count": self.liouvillian_peripheral_mode_count,
            "liouvillian_spectrum_method": self.liouvillian_spectrum_method,
            "matched_internal_nondecaying_mode_count": (
                self.matched_internal_nondecaying_mode_count
            ),
            "missing_internal_nondecaying_mode_count": (
                self.missing_internal_nondecaying_mode_count
            ),
            "extra_nondecaying_mode_count": self.extra_nondecaying_mode_count,
            "extra_zero_mode_count": self.extra_zero_mode_count,
            "external_decay_gap": self.external_decay_gap,
            "likely_attractive_dark_manifold": self.likely_attractive_dark_manifold,
        }

    def __rich__(self):
        return self.to_rich()

    def to_rich(self):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "DarkManifoldDiagnostics.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.dim))
        overview.add_row("number of jumps", str(self.n_jumps))
        overview.add_row("manifold dimension", str(self.manifold_dimension))
        overview.add_row(
            "likely attractive dark manifold",
            str(self.likely_attractive_dark_manifold),
        )

        target = Table(title="Target manifold checks")
        target.add_column("quantity", style="bold")
        target.add_column("value", justify="right")
        target.add_column("status", justify="center")
        target.add_row(
            "||(I-P_M) H P_M||",
            _format_float(self.hamiltonian_closure_residual),
            _status_for_residual(self.hamiltonian_closure_residual),
        )
        target.add_row(
            "max ||J_mu P_M||",
            _format_float(self.max_target_jump_residual),
            _status_for_residual(self.max_target_jump_residual),
        )
        target.add_row(
            "||L(P_M/m)||",
            _format_float(self.target_density_liouvillian_residual),
            _status_for_residual(self.target_density_liouvillian_residual),
        )
        target.add_row(
            "inflow ||P_M J Q_M||",
            _format_float(self.inflow_norm),
            "[green]yes[/green]" if self.inflow_norm > 1e-12 else "[yellow]none[/yellow]",
        )

        jump_kernel = Table(title="Common jump kernel")
        jump_kernel.add_column("quantity", style="bold")
        jump_kernel.add_column("value", justify="right")
        jump_kernel.add_row("dim intersection ker J_mu", str(self.common_jump_kernel_dimension))
        jump_kernel.add_row(
            "target projection onto kernel",
            _format_float(self.target_projection_onto_common_kernel),
        )
        jump_kernel.add_row(
            "target distance from kernel",
            _format_float(self.target_distance_from_common_kernel),
        )
        jump_kernel.add_row("target in kernel", str(self.target_in_common_jump_kernel))
        jump_kernel.add_row(
            "bad complement kernel dim",
            str(self.bad_common_jump_kernel_dimension),
        )
        jump_kernel.add_row(
            "bad-kernel IPRs",
            _format_float_tuple(self.bad_common_jump_kernel_iprs),
        )

        internal = Table(title="Internal non-decaying modes")
        internal.add_column("quantity", style="bold")
        internal.add_column("value", justify="right")
        internal.add_row(
            "expected zero modes",
            str(self.expected_internal_zero_mode_count),
        )
        internal.add_row(
            "expected peripheral modes",
            str(self.expected_internal_peripheral_mode_count),
        )
        internal.add_row(
            "matched internal modes",
            _format_optional_int(self.matched_internal_nondecaying_mode_count),
        )
        internal.add_row(
            "missing internal modes",
            _format_optional_int(self.missing_internal_nondecaying_mode_count),
        )

        liouvillian = Table(title="Liouvillian spectrum")
        liouvillian.add_column("quantity", style="bold")
        liouvillian.add_column("value", justify="right")
        liouvillian.add_row("spectrum method", self.liouvillian_spectrum_method)
        liouvillian.add_row(
            "zero-mode count",
            _format_optional_int(
                self.liouvillian_zero_mode_count,
                lower_bound=self.liouvillian_zero_mode_count_is_lower_bound,
            ),
        )
        liouvillian.add_row(
            "peripheral mode count",
            _format_optional_int(self.liouvillian_peripheral_mode_count),
        )
        liouvillian.add_row(
            "extra non-decaying modes",
            _format_optional_int(self.extra_nondecaying_mode_count),
        )
        liouvillian.add_row("extra zero modes", _format_optional_int(self.extra_zero_mode_count))
        liouvillian.add_row(
            "absolute spectral gap",
            _format_float_or_none(self.liouvillian_spectral_gap),
        )
        liouvillian.add_row("decay gap", _format_float_or_none(self.liouvillian_decay_gap))
        liouvillian.add_row("external decay gap", _format_float_or_none(self.external_decay_gap))

        return Panel(
            Group(overview, target, jump_kernel, internal, liouvillian),
            title=Text("Dark-manifold diagnostics", style="bold cyan"),
            border_style="cyan",
        )


@dataclass(frozen=True, slots=True)
class CommonKernelHamiltonianInvariantSectorReport:
    """Cheap obstruction diagnostic inside the common jump kernel.

    The common jump-kernel condition ``cap_mu ker J_mu = M`` is sufficient but
    stronger than necessary.  A complement vector in the common kernel is only a
    Hamiltonian-stable dark obstruction if its whole Krylov orbit under ``H``
    remains inside the common jump kernel.  This report computes the largest
    such subspace inside ``(cap_mu ker J_mu) cap M^perp`` using a small dense
    nullspace problem.
    """

    dim: int
    n_jumps: int
    manifold_dimension: int
    common_jump_kernel_dimension: int
    bad_common_jump_kernel_dimension: int
    bad_h_invariant_kernel_dimension: int
    h_leakage_norm_from_bad_kernel: float
    h_leakage_norm_from_invariant_kernel: float
    h_target_coupling_norm_from_bad_kernel: float
    h_bad_block_norm: float
    h_invariant_block_eigenvalues: tuple[complex, ...]
    bad_h_invariant_kernel_iprs: tuple[float, ...]
    target_in_common_jump_kernel: bool
    kernel_tolerance: float

    @property
    def has_bad_common_kernel(self) -> bool:
        return self.bad_common_jump_kernel_dimension > 0

    @property
    def has_bad_h_invariant_kernel(self) -> bool:
        return self.bad_h_invariant_kernel_dimension > 0

    @property
    def likely_attractive_by_h_invariant_kernel(self) -> bool:
        return self.target_in_common_jump_kernel and not self.has_bad_h_invariant_kernel

    def to_summary_dict(self) -> dict[str, object]:
        return {
            "dim": self.dim,
            "n_jumps": self.n_jumps,
            "manifold_dimension": self.manifold_dimension,
            "common_jump_kernel_dimension": self.common_jump_kernel_dimension,
            "bad_common_jump_kernel_dimension": self.bad_common_jump_kernel_dimension,
            "bad_h_invariant_kernel_dimension": self.bad_h_invariant_kernel_dimension,
            "h_leakage_norm_from_bad_kernel": self.h_leakage_norm_from_bad_kernel,
            "h_leakage_norm_from_invariant_kernel": (self.h_leakage_norm_from_invariant_kernel),
            "h_target_coupling_norm_from_bad_kernel": (self.h_target_coupling_norm_from_bad_kernel),
            "h_bad_block_norm": self.h_bad_block_norm,
            "h_invariant_block_eigenvalues": tuple(
                complex(value) for value in self.h_invariant_block_eigenvalues
            ),
            "bad_h_invariant_kernel_iprs": self.bad_h_invariant_kernel_iprs,
            "target_in_common_jump_kernel": self.target_in_common_jump_kernel,
            "kernel_tolerance": self.kernel_tolerance,
            "has_bad_common_kernel": self.has_bad_common_kernel,
            "has_bad_h_invariant_kernel": self.has_bad_h_invariant_kernel,
            "likely_attractive_by_h_invariant_kernel": (
                self.likely_attractive_by_h_invariant_kernel
            ),
        }

    def __rich__(self):
        return self.to_rich()

    def to_rich(self):
        try:
            from rich.console import Group
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text
        except ImportError as exc:
            raise ImportError(
                "CommonKernelHamiltonianInvariantSectorReport.to_rich() requires rich. "
                "Install it with `pip install rich`."
            ) from exc

        overview = Table.grid(padding=(0, 2))
        overview.add_column(style="bold")
        overview.add_column()
        overview.add_row("Hilbert dimension", str(self.dim))
        overview.add_row("number of jumps", str(self.n_jumps))
        overview.add_row("manifold dimension", str(self.manifold_dimension))
        overview.add_row("common kernel dimension", str(self.common_jump_kernel_dimension))
        overview.add_row("bad common kernel dimension", str(self.bad_common_jump_kernel_dimension))
        overview.add_row(
            "bad H-invariant kernel dimension",
            str(self.bad_h_invariant_kernel_dimension),
        )
        overview.add_row(
            "likely attractive by H-invariant kernel",
            str(self.likely_attractive_by_h_invariant_kernel),
        )

        leakage = Table(title="Hamiltonian leakage")
        leakage.add_column("quantity", style="bold")
        leakage.add_column("value", justify="right")
        leakage.add_row(
            "||(I-P_K) H B_bad||",
            _format_float(self.h_leakage_norm_from_bad_kernel),
        )
        leakage.add_row(
            "||(I-P_K) H B_inv||",
            _format_float(self.h_leakage_norm_from_invariant_kernel),
        )
        leakage.add_row(
            "||P_M H B_bad||",
            _format_float(self.h_target_coupling_norm_from_bad_kernel),
        )
        leakage.add_row("||B_bad† H B_bad||", _format_float(self.h_bad_block_norm))
        leakage.add_row(
            "bad invariant IPRs",
            _format_float_tuple(self.bad_h_invariant_kernel_iprs),
        )

        return Panel(
            Group(overview, leakage),
            title=Text("Common-kernel H-invariant sector", style="bold cyan"),
            border_style=("green" if self.likely_attractive_by_h_invariant_kernel else "yellow"),
        )


def bad_h_invariant_common_kernel_basis(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    target_states: npt.ArrayLike,
    kernel_tolerance: float = 1.0e-10,
) -> np.ndarray:
    """Return the bad H-invariant sector inside the common jump kernel.

    The returned columns form an orthonormal basis for the largest subspace of
    ``(cap_mu ker J_mu) cap M^perp`` whose Hamiltonian orbit stays inside the
    common jump kernel.  An empty ``(dim, 0)`` array means the selected jumps
    have no Hamiltonian-stable dark obstruction outside the target manifold.

    This helper exposes the obstruction basis used internally by
    :func:`diagnose_common_kernel_h_invariant_sector`, so jump-design routines
    can add a targeted completion stage without recomputing or interpreting a
    Liouvillian spectrum.
    """
    hamiltonian_sparse = _as_scipy_csr_matrix(hamiltonian)
    dim = int(hamiltonian_sparse.shape[0])
    if hamiltonian_sparse.shape != (dim, dim):
        raise ValueError("hamiltonian must be a square matrix.")

    manifold_basis = _orthonormal_target_state_matrix(
        target_states,
        dim=dim,
        tolerance=kernel_tolerance,
    )

    jumps_sparse = tuple(_as_scipy_csr_matrix(jump) for jump in jumps)
    for jump in jumps_sparse:
        if jump.shape != (dim, dim):
            raise ValueError("Every jump operator must have shape (dim, dim).")

    common_kernel_basis = _common_kernel_basis_from_sparse_operators(
        operators=jumps_sparse,
        dim=dim,
        tolerance=kernel_tolerance,
    )
    bad_basis = _kernel_basis_orthogonal_to_manifold(
        basis=common_kernel_basis,
        manifold_basis=manifold_basis,
        tolerance=kernel_tolerance,
    )
    bad_dimension = int(bad_basis.shape[1])
    if bad_dimension == 0:
        return np.zeros((dim, 0), dtype=np.complex128)

    h_bad = np.asarray(hamiltonian_sparse @ bad_basis, dtype=np.complex128)
    if common_kernel_basis.shape[1] == 0:
        projected_to_common = np.zeros_like(h_bad)
    else:
        projected_to_common = common_kernel_basis @ (common_kernel_basis.conj().T @ h_bad)
    leakage = h_bad - projected_to_common

    bad_block = bad_basis.conj().T @ h_bad
    bad_block = 0.5 * (bad_block + bad_block.conj().T)

    invariant_coefficients = _largest_h_invariant_subspace_inside_leakage_kernel(
        leakage=leakage,
        bad_block=bad_block,
        tolerance=kernel_tolerance,
    )

    invariant_basis = bad_basis @ invariant_coefficients
    return _orthonormal_column_basis(invariant_basis, tolerance=kernel_tolerance)


def diagnose_common_kernel_h_invariant_sector(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    target_states: npt.ArrayLike,
    kernel_tolerance: float = 1.0e-10,
) -> CommonKernelHamiltonianInvariantSectorReport:
    """Diagnose Hamiltonian-stable obstructions inside the common jump kernel.

    This is a cheap alternative to a Liouvillian spectrum check.  It first
    computes the common jump kernel ``K = cap_mu ker J_mu`` and the bad
    complement ``B = K cap M^perp``.  It then computes the largest subspace of
    ``B`` whose Hamiltonian Krylov orbit stays inside ``K``.  Only this
    H-invariant part is a purely dark Hamiltonian-stable complement sector.
    """
    hamiltonian_sparse = _as_scipy_csr_matrix(hamiltonian)
    dim = int(hamiltonian_sparse.shape[0])
    if hamiltonian_sparse.shape != (dim, dim):
        raise ValueError("hamiltonian must be a square matrix.")

    manifold_basis = _orthonormal_target_state_matrix(
        target_states,
        dim=dim,
        tolerance=kernel_tolerance,
    )
    manifold_dimension = int(manifold_basis.shape[1])

    jumps_sparse = tuple(_as_scipy_csr_matrix(jump) for jump in jumps)
    for jump in jumps_sparse:
        if jump.shape != (dim, dim):
            raise ValueError("Every jump operator must have shape (dim, dim).")

    common_kernel_basis = _common_kernel_basis_from_sparse_operators(
        operators=jumps_sparse,
        dim=dim,
        tolerance=kernel_tolerance,
    )
    common_jump_kernel_dimension = int(common_kernel_basis.shape[1])

    target_projection_onto_common_kernel, target_distance_from_common_kernel = (
        _subspace_projection_and_distance(
            subspace_basis=manifold_basis,
            containing_basis=common_kernel_basis,
        )
    )
    max_target_jump_residual = max(
        (float(np.linalg.norm(jump @ manifold_basis)) for jump in jumps_sparse),
        default=0.0,
    )
    target_in_common_jump_kernel = (
        target_distance_from_common_kernel <= np.sqrt(kernel_tolerance)
        or max_target_jump_residual <= kernel_tolerance
    )

    bad_basis = _kernel_basis_orthogonal_to_manifold(
        basis=common_kernel_basis,
        manifold_basis=manifold_basis,
        tolerance=kernel_tolerance,
    )
    bad_dimension = int(bad_basis.shape[1])

    if bad_dimension == 0:
        return CommonKernelHamiltonianInvariantSectorReport(
            dim=dim,
            n_jumps=len(jumps_sparse),
            manifold_dimension=manifold_dimension,
            common_jump_kernel_dimension=common_jump_kernel_dimension,
            bad_common_jump_kernel_dimension=0,
            bad_h_invariant_kernel_dimension=0,
            h_leakage_norm_from_bad_kernel=0.0,
            h_leakage_norm_from_invariant_kernel=0.0,
            h_target_coupling_norm_from_bad_kernel=0.0,
            h_bad_block_norm=0.0,
            h_invariant_block_eigenvalues=(),
            bad_h_invariant_kernel_iprs=(),
            target_in_common_jump_kernel=bool(target_in_common_jump_kernel),
            kernel_tolerance=float(kernel_tolerance),
        )

    h_bad = np.asarray(hamiltonian_sparse @ bad_basis, dtype=np.complex128)
    if common_jump_kernel_dimension == 0:
        projected_to_common = np.zeros_like(h_bad)
    else:
        projected_to_common = common_kernel_basis @ (common_kernel_basis.conj().T @ h_bad)
    leakage = h_bad - projected_to_common
    h_leakage_norm_from_bad_kernel = float(np.linalg.norm(leakage))
    h_target_coupling_norm_from_bad_kernel = float(np.linalg.norm(manifold_basis.conj().T @ h_bad))

    bad_block = bad_basis.conj().T @ h_bad
    bad_block = 0.5 * (bad_block + bad_block.conj().T)
    h_bad_block_norm = float(np.linalg.norm(bad_block))

    invariant_coefficients = _largest_h_invariant_subspace_inside_leakage_kernel(
        leakage=leakage,
        bad_block=bad_block,
        tolerance=kernel_tolerance,
    )

    invariant_basis = bad_basis @ invariant_coefficients
    invariant_basis = _orthonormal_column_basis(invariant_basis, tolerance=kernel_tolerance)
    invariant_dimension = int(invariant_basis.shape[1])

    if invariant_dimension == 0:
        h_leakage_norm_from_invariant_kernel = 0.0
        h_invariant_block_eigenvalues: tuple[complex, ...] = ()
        bad_h_invariant_kernel_iprs: tuple[float, ...] = ()
    else:
        h_invariant = np.asarray(hamiltonian_sparse @ invariant_basis, dtype=np.complex128)
        projected_h_invariant = common_kernel_basis @ (common_kernel_basis.conj().T @ h_invariant)
        h_leakage_norm_from_invariant_kernel = float(
            np.linalg.norm(h_invariant - projected_h_invariant)
        )
        invariant_block = invariant_basis.conj().T @ h_invariant
        invariant_block = 0.5 * (invariant_block + invariant_block.conj().T)
        h_invariant_block_eigenvalues = tuple(
            complex(value) for value in np.linalg.eigvalsh(invariant_block)
        )
        bad_h_invariant_kernel_iprs = tuple(
            _state_ipr(invariant_basis[:, index]) for index in range(invariant_dimension)
        )

    return CommonKernelHamiltonianInvariantSectorReport(
        dim=dim,
        n_jumps=len(jumps_sparse),
        manifold_dimension=manifold_dimension,
        common_jump_kernel_dimension=common_jump_kernel_dimension,
        bad_common_jump_kernel_dimension=bad_dimension,
        bad_h_invariant_kernel_dimension=invariant_dimension,
        h_leakage_norm_from_bad_kernel=h_leakage_norm_from_bad_kernel,
        h_leakage_norm_from_invariant_kernel=h_leakage_norm_from_invariant_kernel,
        h_target_coupling_norm_from_bad_kernel=h_target_coupling_norm_from_bad_kernel,
        h_bad_block_norm=h_bad_block_norm,
        h_invariant_block_eigenvalues=h_invariant_block_eigenvalues,
        bad_h_invariant_kernel_iprs=bad_h_invariant_kernel_iprs,
        target_in_common_jump_kernel=bool(target_in_common_jump_kernel),
        kernel_tolerance=float(kernel_tolerance),
    )


def diagnose_dark_manifold(
    *,
    hamiltonian: Any,
    jumps: list[Any] | tuple[Any, ...],
    target_states: npt.ArrayLike,
    backend: OpenSystemBackendName | OpenSystemBackend = "scipy",
    kernel_tolerance: float = 1e-10,
    liouvillian_zero_tolerance: float = 1e-9,
    check_liouvillian_spectrum: bool = True,
    max_liouvillian_dense_dimension: int = 4096,
    liouvillian_spectrum_method: Literal["auto", "dense", "sparse", "none"] = "auto",
    sparse_liouvillian_eigenvalue_count: int = 32,
) -> DarkManifoldDiagnostics:
    """Diagnose whether a target manifold is an attractive dark manifold.

    The columns of ``target_states`` span the target manifold.  They need not be
    orthonormal; this function orthonormalizes them and uses the target
    projector ``P_M``.  The diagnostic accepts the internal non-decaying
    Liouvillian modes generated by the projected Hamiltonian ``M† H M`` and
    reports additional zero/peripheral modes as possible complement obstructions.
    """
    # _backend_obj = get_open_system_backend(backend)

    hamiltonian_sparse = _as_scipy_csr_matrix(hamiltonian)
    dim = int(hamiltonian_sparse.shape[0])
    if hamiltonian_sparse.shape != (dim, dim):
        raise ValueError("hamiltonian must be a square matrix.")

    manifold_basis = _orthonormal_target_state_matrix(
        target_states,
        dim=dim,
        tolerance=kernel_tolerance,
    )
    manifold_dimension = int(manifold_basis.shape[1])
    jumps_sparse = tuple(_as_scipy_csr_matrix(jump) for jump in jumps)
    for jump in jumps_sparse:
        if jump.shape != (dim, dim):
            raise ValueError("Every jump operator must have shape (dim, dim).")

    hamiltonian_action = np.asarray(hamiltonian_sparse @ manifold_basis, dtype=np.complex128)
    internal_hamiltonian = manifold_basis.conj().T @ hamiltonian_action
    projected_hamiltonian_action = manifold_basis @ internal_hamiltonian
    hamiltonian_closure_residual = float(
        np.linalg.norm(hamiltonian_action - projected_hamiltonian_action)
    )

    target_jump_matrices = tuple(jump @ manifold_basis for jump in jumps_sparse)
    target_jump_residuals = tuple(float(np.linalg.norm(matrix)) for matrix in target_jump_matrices)
    max_target_jump_residual = max(target_jump_residuals) if target_jump_residuals else 0.0

    target_density = (manifold_basis @ manifold_basis.conj().T) / float(manifold_dimension)
    target_density_liouvillian_residual = float(
        np.linalg.norm(
            lindblad_rhs_density_matrix(
                target_density,
                hamiltonian=hamiltonian_sparse,
                jumps=list(jumps_sparse),
                backend=backend,
            )
        )
    )

    inflow_norm = _manifold_inflow_norm(
        jumps=jumps_sparse,
        manifold_basis=manifold_basis,
    )

    common_kernel_basis = _common_kernel_basis_from_sparse_operators(
        operators=jumps_sparse,
        dim=dim,
        tolerance=kernel_tolerance,
    )
    common_jump_kernel_dimension = int(common_kernel_basis.shape[1])

    target_projection_onto_common_kernel, target_distance_from_common_kernel = (
        _subspace_projection_and_distance(
            subspace_basis=manifold_basis,
            containing_basis=common_kernel_basis,
        )
    )
    target_in_common_jump_kernel = (
        target_distance_from_common_kernel <= np.sqrt(kernel_tolerance)
        or max_target_jump_residual <= kernel_tolerance
    )

    bad_common_kernel_basis = _kernel_basis_orthogonal_to_manifold(
        basis=common_kernel_basis,
        manifold_basis=manifold_basis,
        tolerance=kernel_tolerance,
    )
    bad_common_jump_kernel_dimension = int(bad_common_kernel_basis.shape[1])
    bad_common_jump_kernel_iprs = tuple(
        _state_ipr(bad_common_kernel_basis[:, index])
        for index in range(bad_common_kernel_basis.shape[1])
    )

    internal_hamiltonian = 0.5 * (internal_hamiltonian + internal_hamiltonian.conj().T)
    internal_hamiltonian_eigenvalues = tuple(
        complex(value) for value in np.linalg.eigvalsh(internal_hamiltonian)
    )
    expected_internal_liouvillian_eigenvalues = _internal_liouvillian_eigenvalues(
        internal_hamiltonian_eigenvalues
    )
    expected_internal_zero_mode_count = int(
        sum(
            abs(value) <= liouvillian_zero_tolerance
            for value in expected_internal_liouvillian_eigenvalues
        )
    )
    expected_internal_peripheral_mode_count = (
        len(expected_internal_liouvillian_eigenvalues) - expected_internal_zero_mode_count
    )

    liouvillian_zero_mode_count: int | None = None
    liouvillian_zero_mode_count_is_lower_bound = False
    liouvillian_spectral_gap: float | None = None
    liouvillian_decay_gap: float | None = None
    liouvillian_peripheral_mode_count: int | None = None
    liouvillian_eigenvalues: tuple[complex, ...] = ()
    actual_liouvillian_spectrum_method = "none"
    matched_internal_nondecaying_mode_count: int | None = None
    missing_internal_nondecaying_mode_count: int | None = None
    extra_nondecaying_mode_count: int | None = None
    extra_zero_mode_count: int | None = None
    external_decay_gap: float | None = None

    if check_liouvillian_spectrum and liouvillian_spectrum_method != "none":
        liouvillian_dimension = dim * dim
        liouvillian = build_liouvillian(
            hamiltonian_sparse,
            list(jumps_sparse),
            backend="scipy",
            sparse_format="csr",
        )

        if liouvillian_spectrum_method == "auto":
            actual_liouvillian_spectrum_method = (
                "dense" if liouvillian_dimension <= max_liouvillian_dense_dimension else "sparse"
            )
        else:
            actual_liouvillian_spectrum_method = liouvillian_spectrum_method

        if actual_liouvillian_spectrum_method == "dense":
            if liouvillian_dimension > max_liouvillian_dense_dimension:
                raise ValueError(
                    "Dense Liouvillian spectrum check is too expensive: "
                    f"dim^2={liouvillian_dimension}, "
                    f"max_liouvillian_dense_dimension={max_liouvillian_dense_dimension}. "
                    "Use liouvillian_spectrum_method='sparse' or 'auto', "
                    "or set check_liouvillian_spectrum=False."
                )
            eigenvalues = scipy_linalg.eigvals(liouvillian.toarray())
            eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
            is_partial_spectrum = False
        elif actual_liouvillian_spectrum_method == "sparse":
            eigenvalues = _sparse_liouvillian_near_zero_eigenvalues(
                liouvillian,
                n_eigenvalues=sparse_liouvillian_eigenvalue_count,
                zero_tolerance=liouvillian_zero_tolerance,
            )
            is_partial_spectrum = True
        else:
            raise ValueError(
                "liouvillian_spectrum_method must be 'auto', 'dense', 'sparse', or 'none'."
            )

        (
            liouvillian_zero_mode_count,
            liouvillian_zero_mode_count_is_lower_bound,
            liouvillian_spectral_gap,
            liouvillian_decay_gap,
            liouvillian_peripheral_mode_count,
            liouvillian_eigenvalues,
        ) = _summarize_liouvillian_eigenvalues(
            eigenvalues,
            zero_tolerance=liouvillian_zero_tolerance,
            is_partial_spectrum=is_partial_spectrum,
            requested_count=sparse_liouvillian_eigenvalue_count,
        )

        if not is_partial_spectrum:
            nondecaying_values = tuple(
                complex(value)
                for value in eigenvalues
                if abs(complex(value).real) <= liouvillian_zero_tolerance
            )
            match = _match_expected_internal_nondecaying_modes(
                observed=nondecaying_values,
                expected=expected_internal_liouvillian_eigenvalues,
                tolerance=liouvillian_zero_tolerance,
            )
            matched_internal_nondecaying_mode_count = match["matched"]
            missing_internal_nondecaying_mode_count = match["missing"]
            extra_nondecaying_mode_count = match["extra"]
            extra_zero_mode_count = max(
                0,
                int(liouvillian_zero_mode_count) - expected_internal_zero_mode_count,
            )
            external_decay_gap = _external_decay_gap_from_spectrum(
                eigenvalues=eigenvalues,
                matched_observed_indices=match["matched_observed_indices"],
                zero_tolerance=liouvillian_zero_tolerance,
            )

    likely_attractive_dark_manifold: bool | None
    if extra_nondecaying_mode_count is None:
        likely_attractive_dark_manifold = None
    else:
        likely_attractive_dark_manifold = (
            hamiltonian_closure_residual <= liouvillian_zero_tolerance
            and max_target_jump_residual <= liouvillian_zero_tolerance
            and target_density_liouvillian_residual <= liouvillian_zero_tolerance
            and extra_nondecaying_mode_count == 0
        )

    return DarkManifoldDiagnostics(
        dim=dim,
        n_jumps=len(jumps_sparse),
        manifold_dimension=manifold_dimension,
        hamiltonian_closure_residual=hamiltonian_closure_residual,
        target_jump_residuals=target_jump_residuals,
        max_target_jump_residual=max_target_jump_residual,
        target_density_liouvillian_residual=target_density_liouvillian_residual,
        inflow_norm=inflow_norm,
        common_jump_kernel_dimension=common_jump_kernel_dimension,
        target_projection_onto_common_kernel=target_projection_onto_common_kernel,
        target_distance_from_common_kernel=target_distance_from_common_kernel,
        target_in_common_jump_kernel=target_in_common_jump_kernel,
        bad_common_jump_kernel_dimension=bad_common_jump_kernel_dimension,
        bad_common_jump_kernel_iprs=bad_common_jump_kernel_iprs,
        internal_hamiltonian_eigenvalues=internal_hamiltonian_eigenvalues,
        expected_internal_liouvillian_eigenvalues=expected_internal_liouvillian_eigenvalues,
        expected_internal_zero_mode_count=expected_internal_zero_mode_count,
        expected_internal_peripheral_mode_count=expected_internal_peripheral_mode_count,
        liouvillian_zero_mode_count=liouvillian_zero_mode_count,
        liouvillian_zero_mode_count_is_lower_bound=bool(liouvillian_zero_mode_count_is_lower_bound),
        liouvillian_spectral_gap=liouvillian_spectral_gap,
        liouvillian_decay_gap=liouvillian_decay_gap,
        liouvillian_peripheral_mode_count=liouvillian_peripheral_mode_count,
        liouvillian_spectrum_method=actual_liouvillian_spectrum_method,
        liouvillian_eigenvalues=liouvillian_eigenvalues,
        matched_internal_nondecaying_mode_count=matched_internal_nondecaying_mode_count,
        missing_internal_nondecaying_mode_count=missing_internal_nondecaying_mode_count,
        extra_nondecaying_mode_count=extra_nondecaying_mode_count,
        extra_zero_mode_count=extra_zero_mode_count,
        external_decay_gap=external_decay_gap,
        likely_attractive_dark_manifold=likely_attractive_dark_manifold,
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
    liouvillian_spectrum_method: Literal["auto", "dense", "sparse", "none"] = "auto",
    sparse_liouvillian_eigenvalue_count: int = 16,
) -> DarkSubspaceDiagnostics:
    """Diagnose whether a dark target is likely unique/attractive.

    This is intended for small systems. It computes:

        1. target jump residuals ||J_mu psi||;
        2. common jump kernel dim intersection_mu ker J_mu;
        3. bad common-kernel dimension after removing the target direction;
        4. target Liouvillian residual ||L(|psi><psi|)||;
        5. optional Liouvillian zero-mode count.

    The Liouvillian spectrum check uses a dense solver for small Liouvillians and
    a sparse shift-invert Arnoldi solver for larger ones when
    ``liouvillian_spectrum_method="auto"``.  The sparse zero-mode count is a
    lower bound if all requested eigenvalues are numerically zero; increase
    ``sparse_liouvillian_eigenvalue_count`` to resolve more zero modes.
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
    liouvillian_zero_mode_count_is_lower_bound = False
    liouvillian_spectral_gap: float | None = None
    liouvillian_decay_gap: float | None = None
    liouvillian_peripheral_mode_count: int | None = None
    liouvillian_eigenvalues: tuple[complex, ...] = ()
    actual_liouvillian_spectrum_method = "none"

    if check_liouvillian_spectrum and liouvillian_spectrum_method != "none":
        liouvillian_dimension = dim * dim
        liouvillian = build_liouvillian(
            hamiltonian_sparse,
            list(jumps_sparse),
            backend="scipy",
            sparse_format="csr",
        )

        if liouvillian_spectrum_method == "auto":
            actual_liouvillian_spectrum_method = (
                "dense" if liouvillian_dimension <= max_liouvillian_dense_dimension else "sparse"
            )
        else:
            actual_liouvillian_spectrum_method = liouvillian_spectrum_method

        if actual_liouvillian_spectrum_method == "dense":
            if liouvillian_dimension > max_liouvillian_dense_dimension:
                raise ValueError(
                    "Dense Liouvillian spectrum check is too expensive: "
                    f"dim^2={liouvillian_dimension}, "
                    f"max_liouvillian_dense_dimension="
                    f"{max_liouvillian_dense_dimension}. "
                    "Use liouvillian_spectrum_method='sparse' or 'auto', "
                    "or set check_liouvillian_spectrum=False."
                )

            eigenvalues = scipy_linalg.eigvals(liouvillian.toarray())
            eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
        elif actual_liouvillian_spectrum_method == "sparse":
            eigenvalues = _sparse_liouvillian_near_zero_eigenvalues(
                liouvillian,
                n_eigenvalues=sparse_liouvillian_eigenvalue_count,
                zero_tolerance=liouvillian_zero_tolerance,
            )
        else:
            raise ValueError(
                "liouvillian_spectrum_method must be 'auto', 'dense', 'sparse', or 'none'."
            )

        (
            liouvillian_zero_mode_count,
            liouvillian_zero_mode_count_is_lower_bound,
            liouvillian_spectral_gap,
            liouvillian_decay_gap,
            liouvillian_peripheral_mode_count,
            liouvillian_eigenvalues,
        ) = _summarize_liouvillian_eigenvalues(
            eigenvalues,
            zero_tolerance=liouvillian_zero_tolerance,
            is_partial_spectrum=(actual_liouvillian_spectrum_method == "sparse"),
            requested_count=sparse_liouvillian_eigenvalue_count,
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
        liouvillian_zero_mode_count_is_lower_bound=bool(liouvillian_zero_mode_count_is_lower_bound),
        liouvillian_spectral_gap=liouvillian_spectral_gap,
        liouvillian_decay_gap=liouvillian_decay_gap,
        liouvillian_peripheral_mode_count=liouvillian_peripheral_mode_count,
        liouvillian_spectrum_method=actual_liouvillian_spectrum_method,
        liouvillian_eigenvalues=liouvillian_eigenvalues,
        likely_unique_dark_state=likely_unique_dark_state,
    )


def _sparse_liouvillian_near_zero_eigenvalues(
    liouvillian: Any,
    *,
    n_eigenvalues: int,
    zero_tolerance: float,
) -> np.ndarray:
    """Return a partial spectrum close to zero for a sparse Liouvillian."""
    matrix = (
        liouvillian.tocsr()
        if hasattr(liouvillian, "tocsr")
        else scipy_sparse.csr_array(liouvillian)
    )
    dimension = int(matrix.shape[0])
    if dimension <= 2:
        return scipy_linalg.eigvals(matrix.toarray())

    k = max(1, min(int(n_eigenvalues), dimension - 2))

    # Shift-invert close to zero is usually far more reliable than ``which='SM'``
    # for non-Hermitian Liouvillians, but sigma=0 can fail because the
    # Liouvillian is singular.  Use a tiny positive real shift and fall back to
    # smallest-magnitude Arnoldi if the factorization is ill-conditioned.
    sigma = max(float(zero_tolerance) * 0.1, 1.0e-14)
    try:
        values = scipy_sparse_linalg.eigs(
            matrix,
            k=k,
            sigma=sigma,
            which="LM",
            return_eigenvectors=False,
        )
    except Exception:
        values = scipy_sparse_linalg.eigs(
            matrix,
            k=k,
            which="SM",
            return_eigenvectors=False,
        )

    return np.asarray(values, dtype=np.complex128)


def _summarize_liouvillian_eigenvalues(
    eigenvalues: npt.ArrayLike,
    *,
    zero_tolerance: float,
    is_partial_spectrum: bool,
    requested_count: int,
) -> tuple[int, bool, float | None, float | None, int | None, tuple[complex, ...]]:
    values = np.asarray(eigenvalues, dtype=np.complex128)
    if values.size == 0:
        return 0, False, None, None, None, ()

    abs_values = np.abs(values)
    zero_mask = abs_values <= zero_tolerance
    zero_count = int(np.count_nonzero(zero_mask))
    zero_count_is_lower_bound = bool(
        is_partial_spectrum and zero_count >= min(int(requested_count), values.size)
    )

    nonzero_abs = abs_values[~zero_mask]
    absolute_gap = float(np.min(nonzero_abs)) if nonzero_abs.size else None

    nonzero_real_parts = np.real(values[~zero_mask])
    decaying = nonzero_real_parts < -zero_tolerance
    decay_gap = float(-np.max(nonzero_real_parts[decaying])) if np.any(decaying) else None

    peripheral_mask = (~zero_mask) & (np.abs(np.real(values)) <= zero_tolerance)
    peripheral_count = int(np.count_nonzero(peripheral_mask))

    order = np.lexsort((np.real(values), abs_values))
    shown = tuple(complex(values[index]) for index in order[: min(16, values.size)])

    return (
        zero_count,
        zero_count_is_lower_bound,
        absolute_gap,
        decay_gap,
        peripheral_count,
        shown,
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

    # Evaluate the Hamiltonian commutator using only the component orthogonal
    # to the rank-one projector.  Writing the two commutator terms as
    # ``-i H|psi><psi| + i |psi><psi|H`` suffers from catastrophic
    # cancellation when ``|psi>`` is an eigenstate with a nonzero energy.  In
    # that case the two large rank-one terms cancel exactly, but the low-rank
    # Frobenius contraction can leave a spurious residual around sqrt(eps).
    # Subtracting the Rayleigh quotient first preserves the same commutator and
    # makes exact eigenstates numerically dark.
    target_energy = np.vdot(target, hamiltonian_target)
    hamiltonian_target_perp = hamiltonian_target - target_energy * target
    terms: list[tuple[complex, np.ndarray, np.ndarray]] = [
        (-1j, hamiltonian_target_perp, target),
        (1j, target, hamiltonian_target_perp),
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


def _orthonormal_target_state_matrix(
    target_states: npt.ArrayLike,
    *,
    dim: int,
    tolerance: float,
) -> np.ndarray:
    matrix = np.asarray(target_states, dtype=np.complex128)
    if matrix.ndim == 1:
        if matrix.size != dim:
            raise ValueError("target_states vector has incompatible dimension.")
        matrix = matrix.reshape(dim, 1)
    elif matrix.ndim == 2:
        if matrix.shape[0] == dim:
            pass
        elif matrix.shape[1] == dim:
            matrix = matrix.T
        else:
            raise ValueError("target_states must have shape (dim, n_states) or (n_states, dim).")
    else:
        raise ValueError("target_states must be one- or two-dimensional.")

    if matrix.shape[1] == 0:
        raise ValueError("target_states must contain at least one state.")

    return _orthonormal_column_basis(matrix, tolerance=tolerance)


def _subspace_projection_and_distance(
    *,
    subspace_basis: np.ndarray,
    containing_basis: np.ndarray,
) -> tuple[float, float]:
    if subspace_basis.shape[1] == 0:
        return 0.0, 0.0
    if containing_basis.shape[1] == 0:
        return 0.0, float(np.sqrt(subspace_basis.shape[1]))

    projected = containing_basis @ (containing_basis.conj().T @ subspace_basis)
    projection_norm = float(np.linalg.norm(projected))
    distance = float(np.linalg.norm(subspace_basis - projected))
    return projection_norm, distance


def _kernel_basis_orthogonal_to_manifold(
    *,
    basis: np.ndarray,
    manifold_basis: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Return the part of ``basis`` orthogonal to ``manifold_basis``.

    ``basis`` is typically an orthonormal common jump-kernel basis whose
    columns are arbitrary SVD vectors.  Projecting each column separately can
    produce spurious complement vectors when the whole subspace equals the
    target manifold but individual columns are only numerically aligned.  Use
    the principal-angle/nullspace formulation instead: vectors in
    ``span(basis)`` orthogonal to the manifold are ``basis @ c`` with
    ``manifold_basis† basis c = 0``.
    """
    if basis.shape[1] == 0:
        return np.zeros((manifold_basis.shape[0], 0), dtype=np.complex128)

    overlap = manifold_basis.conj().T @ basis
    overlap_scale = float(np.linalg.norm(overlap, ord="fro"))
    cutoff = max(float(tolerance), float(np.sqrt(tolerance)) * max(1.0, overlap_scale))
    coefficients = _nullspace_basis(overlap, tolerance=cutoff)

    if coefficients.shape[1] == 0:
        return np.zeros((manifold_basis.shape[0], 0), dtype=np.complex128)

    complement = basis @ coefficients
    return _orthonormal_column_basis(complement, tolerance=tolerance)


def _manifold_inflow_norm(
    *,
    jumps: tuple[scipy_sparse.spmatrix, ...] | tuple[scipy_sparse.sparray, ...],
    manifold_basis: np.ndarray,
) -> float:
    total = 0.0
    for jump in jumps:
        adjoint_action = np.asarray(jump.conj().T @ manifold_basis, dtype=np.complex128)
        projected_out = adjoint_action - manifold_basis @ (manifold_basis.conj().T @ adjoint_action)
        total += float(np.linalg.norm(projected_out) ** 2)
    return float(np.sqrt(max(total, 0.0)))


def _internal_liouvillian_eigenvalues(
    internal_hamiltonian_eigenvalues: Sequence[complex],
) -> tuple[complex, ...]:
    energies = tuple(complex(value) for value in internal_hamiltonian_eigenvalues)
    return tuple(-1j * (left - right) for left in energies for right in energies)


def _match_expected_internal_nondecaying_modes(
    *,
    observed: Sequence[complex],
    expected: Sequence[complex],
    tolerance: float,
) -> dict[str, object]:
    observed_values = [complex(value) for value in observed]
    expected_values = [complex(value) for value in expected]
    unmatched = set(range(len(observed_values)))
    matched_indices: list[int] = []

    for expected_value in expected_values:
        best_index = None
        best_distance = float("inf")
        for observed_index in unmatched:
            distance = abs(observed_values[observed_index] - expected_value)
            if distance < best_distance:
                best_distance = float(distance)
                best_index = observed_index
        if best_index is not None and best_distance <= tolerance:
            unmatched.remove(best_index)
            matched_indices.append(best_index)

    matched = len(matched_indices)
    return {
        "matched": matched,
        "missing": max(0, len(expected_values) - matched),
        "extra": max(0, len(observed_values) - matched),
        "matched_observed_indices": tuple(sorted(matched_indices)),
    }


def _external_decay_gap_from_spectrum(
    *,
    eigenvalues: npt.ArrayLike,
    matched_observed_indices: Sequence[int],
    zero_tolerance: float,
) -> float | None:
    values = np.asarray(eigenvalues, dtype=np.complex128)
    nondecaying_indices = [
        index for index, value in enumerate(values) if abs(complex(value).real) <= zero_tolerance
    ]
    matched_nondecaying_indices = set(matched_observed_indices)
    external_indices = [
        index for index in range(values.size) if index not in matched_nondecaying_indices
    ]
    if not external_indices:
        return None

    external_real_parts = np.real(values[external_indices])
    decaying = external_real_parts < -zero_tolerance
    if not np.any(decaying):
        return None
    if any(index in external_indices for index in nondecaying_indices):
        return 0.0
    return float(-np.max(external_real_parts[decaying]))


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


def _largest_h_invariant_subspace_inside_leakage_kernel(
    *,
    leakage: np.ndarray,
    bad_block: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Return bad-coordinate vectors in ``ker(leakage)`` invariant under ``bad_block``.

    The direct Krylov certificate stacks ``leakage @ bad_block**n`` for every
    ``n`` and then computes one large dense nullspace.  That is fragile for the
    triangular-QDM production cases: the stacked matrix is very tall and can
    trigger LAPACK ``SVD did not converge`` failures.  This computes the same
    largest invariant subspace by fixed-point intersections,

    ``S <- {x in S : bad_block @ x in S}``, starting from ``S = ker(leakage)``.

    Each nullspace problem has only ``bad_dimension`` rows and the current
    subspace dimension columns, so the dense linear algebra remains bounded by
    the selected jump set rather than by the full Krylov stack.
    """
    bad_dimension = int(bad_block.shape[0])
    if bad_dimension == 0:
        return np.zeros((0, 0), dtype=np.complex128)

    if not np.all(np.isfinite(leakage)) or not np.all(np.isfinite(bad_block)):
        raise ValueError("H-invariant kernel diagnostic received non-finite matrix entries.")

    current = _nullspace_basis_by_gram(leakage, tolerance=tolerance)
    current = _orthonormal_column_basis(current, tolerance=tolerance)
    if current.shape[1] == 0:
        return np.zeros((bad_dimension, 0), dtype=np.complex128)

    for _iteration in range(bad_dimension):
        image = bad_block @ current
        projected = current @ (current.conj().T @ image)
        escape = image - projected
        if float(np.linalg.norm(escape)) <= tolerance:
            return current.astype(np.complex128, copy=False)

        surviving_coordinates = _nullspace_basis_by_gram(
            escape,
            tolerance=tolerance,
        )
        if surviving_coordinates.shape[1] == 0:
            return np.zeros((bad_dimension, 0), dtype=np.complex128)

        next_current = current @ surviving_coordinates
        next_current = _orthonormal_column_basis(next_current, tolerance=tolerance)
        if next_current.shape[1] == 0:
            return np.zeros((bad_dimension, 0), dtype=np.complex128)

        if next_current.shape[1] == current.shape[1]:
            current = next_current
            continue
        current = next_current

    return current.astype(np.complex128, copy=False)


def _nullspace_basis_by_gram(
    matrix: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    """Return a right-nullspace basis using the Hermitian Gram matrix.

    This is a robust fallback for tall matrices where a direct SVD can be both
    slow and numerically fragile.  The Gram path squares the condition number,
    so it is used only where a production diagnostic prefers a conservative,
    non-crashing certificate over an exact singular spectrum.
    """
    if matrix.size == 0:
        return np.eye(matrix.shape[1], dtype=np.complex128)
    matrix = np.asarray(matrix, dtype=np.complex128)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Cannot compute a nullspace for a matrix with non-finite entries.")

    gram = matrix.conj().T @ matrix
    return _kernel_basis_from_hermitian_gram(gram, tolerance=tolerance)


def _kernel_basis_from_hermitian_gram(
    gram: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    gram = np.asarray(gram, dtype=np.complex128)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram matrix must be square.")
    dimension = int(gram.shape[0])
    if dimension == 0:
        return np.zeros((0, 0), dtype=np.complex128)
    if not np.all(np.isfinite(gram)):
        raise ValueError("Cannot diagonalize a Gram matrix with non-finite entries.")

    gram = 0.5 * (gram + gram.conj().T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = scipy_linalg.eigh(
            gram,
            check_finite=True,
            driver="evd",
        )

    eigenvalues = np.maximum(np.real(eigenvalues), 0.0)
    eigenvalue_scale = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
    roundoff_threshold = (
        100.0 * np.finfo(np.float64).eps * max(1.0, eigenvalue_scale) * max(1, dimension)
    )
    eigenvalue_threshold = max(float(tolerance) * float(tolerance), roundoff_threshold)
    keep = eigenvalues <= eigenvalue_threshold
    if not np.any(keep):
        return np.zeros((dimension, 0), dtype=np.complex128)
    return eigenvectors[:, keep].astype(np.complex128, copy=False)


def _range_basis_from_hermitian_gram(
    matrix: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    """Orthonormalize columns from a right Gram eigendecomposition fallback."""
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.complex128)
    matrix = np.asarray(matrix, dtype=np.complex128)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Cannot orthonormalize a matrix with non-finite entries.")

    gram = matrix.conj().T @ matrix
    gram = 0.5 * (gram + gram.conj().T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = scipy_linalg.eigh(
            gram,
            check_finite=True,
            driver="evd",
        )

    eigenvalues = np.maximum(np.real(eigenvalues), 0.0)
    eigenvalue_scale = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
    roundoff_threshold = (
        100.0 * np.finfo(np.float64).eps * max(1.0, eigenvalue_scale) * max(1, matrix.shape[1])
    )
    eigenvalue_threshold = max(float(tolerance) * float(tolerance), roundoff_threshold)
    keep = eigenvalues > eigenvalue_threshold
    if not np.any(keep):
        return np.zeros((matrix.shape[0], 0), dtype=np.complex128)

    kept_eigenvectors = eigenvectors[:, keep]
    kept_eigenvalues = eigenvalues[keep]
    basis = matrix @ (kept_eigenvectors / np.sqrt(kept_eigenvalues)[None, :])
    return basis.astype(np.complex128, copy=False)


def _nullspace_basis(
    matrix: np.ndarray,
    *,
    tolerance: float,
) -> np.ndarray:
    if matrix.size == 0:
        return np.eye(matrix.shape[1], dtype=np.complex128)

    matrix = np.asarray(matrix, dtype=np.complex128)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Cannot compute a nullspace for a matrix with non-finite entries.")

    # A full SVD is only needed for underdetermined matrices.  For tall
    # stacked-jump matrices, economy SVD keeps the complete right-singular
    # space while avoiding a huge unused left-unitary allocation.
    full_matrices = matrix.shape[0] < matrix.shape[1]
    try:
        _left_vectors, singular_values, right_vectors_dagger = np.linalg.svd(
            matrix,
            full_matrices=full_matrices,
        )
    except np.linalg.LinAlgError:
        # LAPACK occasionally fails to converge on very tall, ill-conditioned
        # diagnostic matrices even though the right Gram matrix is small.  Fall
        # back to the Hermitian path so production diagnostics report a
        # conservative kernel instead of crashing.
        return _nullspace_basis_by_gram(matrix, tolerance=tolerance)

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

    matrix = np.asarray(matrix, dtype=np.complex128)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Cannot orthonormalize a matrix with non-finite entries.")

    try:
        left_vectors, singular_values, _right_vectors_dagger = np.linalg.svd(
            matrix,
            full_matrices=False,
        )
    except np.linalg.LinAlgError:
        return _range_basis_from_hermitian_gram(matrix, tolerance=tolerance)

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


def _format_optional_int(value: int | None, *, lower_bound: bool = False) -> str:
    if value is None:
        return "not checked"
    return f"{value}{'+' if lower_bound else ''}"


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
