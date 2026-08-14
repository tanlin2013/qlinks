from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as scipy_sparse

from qlinks.open_system.diagnostics.target_manifold import (
    _resolve_density_matrix_or_snapshot_series,
)


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
