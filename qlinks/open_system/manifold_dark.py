"""Dark-operator basis, dressing, and shared detector linear algebra."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from qlinks.open_system.manifold_detector_types import (
    DarkOperatorTerm,
    DressedManifoldDarkDetectorCandidate,
    DressedManifoldDarkDetectorReport,
    ManifoldDarkOperatorBasisReport,
    ManifoldDarkOperatorCandidate,
    RecycledManifoldDarkDetectorCandidate,
)


def _as_csr(operator: Any) -> sp.csr_array:
    if hasattr(operator, "tocsr"):
        return operator.tocsr()
    return sp.csr_array(operator)


def _normalize_state_columns(
    states: npt.ArrayLike,
    *,
    tolerance: float,
) -> tuple[npt.NDArray[np.complex128], float]:
    matrix = np.asarray(states, dtype=np.complex128)

    if matrix.ndim == 1:
        matrix = matrix.reshape(matrix.size, 1)
    elif matrix.ndim != 2:
        raise ValueError("states must be one- or two-dimensional.")

    if matrix.shape[0] < matrix.shape[1]:
        # This is only a convenience heuristic.  Most callers pass columns, but
        # small test/state lists often come as rows.
        row_norms = np.linalg.norm(matrix, axis=1)
        column_norms = np.linalg.norm(matrix, axis=0)
        if np.count_nonzero(row_norms > tolerance) <= np.count_nonzero(column_norms > tolerance):
            matrix = matrix.T

    if matrix.shape[1] == 0:
        raise ValueError("states must contain at least one vector.")

    q, r = np.linalg.qr(matrix)
    diagonal = np.abs(np.diag(r))
    rank = int(np.count_nonzero(diagonal > tolerance))
    if rank == 0:
        raise ValueError("states have numerical rank zero.")

    q = q[:, :rank].astype(np.complex128, copy=False)
    gram_residual = float(np.linalg.norm(q.conj().T @ q - np.eye(rank)))
    return q, gram_residual


def _combined_operator_frobenius_norm(
    *,
    operators: tuple[sp.csr_array, ...],
    coefficients: npt.NDArray[np.complex128],
) -> float:
    if len(operators) == 0:
        return 0.0
    combined = sp.csr_array(operators[0].shape, dtype=np.complex128)
    for coefficient, operator in zip(coefficients, operators, strict=True):
        if abs(coefficient) == 0.0:
            continue
        combined = combined + coefficient * operator
    return float(sp.linalg.norm(combined))


def _coefficient_ipr(coefficients: npt.ArrayLike) -> float:
    values = np.asarray(coefficients, dtype=np.complex128)
    norm_squared = float(np.vdot(values, values).real)
    if norm_squared <= 0.0:
        return 0.0
    return float(np.sum(np.abs(values) ** 4) / (norm_squared * norm_squared))


def _effective_coefficient_count(coefficients: npt.ArrayLike) -> float:
    ipr = _coefficient_ipr(coefficients)
    if ipr <= 0.0:
        return float("inf")
    return float(1.0 / ipr)


def _phase_fixed_normalized_vector(
    vector: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> npt.NDArray[np.complex128] | None:
    norm = float(np.linalg.norm(vector))
    if norm <= tolerance:
        return None
    normalized = np.asarray(vector / norm, dtype=np.complex128)
    pivot = int(np.argmax(np.abs(normalized)))
    pivot_value = normalized[pivot]
    if abs(pivot_value) > tolerance:
        normalized = normalized * np.exp(-1j * np.angle(pivot_value))
    return normalized


def _deduplicate_coefficient_vectors(
    vectors: list[npt.NDArray[np.complex128]],
    *,
    overlap_tolerance: float,
) -> list[npt.NDArray[np.complex128]]:
    unique: list[npt.NDArray[np.complex128]] = []
    for vector in vectors:
        if any(abs(np.vdot(existing, vector)) >= 1.0 - overlap_tolerance for existing in unique):
            continue
        unique.append(vector)
    return unique


def _sparse_ipr_dark_detector_columns(
    *,
    nullspace: npt.NDArray[np.complex128],
    max_candidates: int | None,
    tolerance: float,
    overlap_tolerance: float,
) -> npt.NDArray[np.complex128]:
    """Return nullspace vectors biased toward small operator support.

    The ordinary SVD basis is arbitrary inside a degenerate dark-detector
    nullspace.  To get more interpretable detector readouts, project each
    coordinate unit vector onto the dark nullspace and rank the resulting
    vectors by coefficient IPR.  This is a cheap deterministic proxy for a
    sparse/IPR-optimized basis: a high score means the detector is concentrated
    on fewer supplied local operators.
    """
    if nullspace.ndim != 2:
        raise ValueError("nullspace must be two-dimensional.")
    n_operators, nullity = nullspace.shape
    if n_operators == 0 or nullity == 0:
        return np.zeros((n_operators, 0), dtype=np.complex128)

    projected: list[npt.NDArray[np.complex128]] = []
    for operator_index in range(n_operators):
        row = np.asarray(nullspace[operator_index, :], dtype=np.complex128)
        # Projection of the coordinate vector e_i onto span(nullspace).
        vector = nullspace @ row.conj()
        normalized = _phase_fixed_normalized_vector(vector, tolerance=tolerance)
        if normalized is not None:
            projected.append(normalized)

    projected.sort(
        key=lambda vector: (
            -_coefficient_ipr(vector),
            int(np.count_nonzero(np.abs(vector) > tolerance)),
            int(np.argmax(np.abs(vector))),
        )
    )
    unique = _deduplicate_coefficient_vectors(
        projected,
        overlap_tolerance=overlap_tolerance,
    )

    # If coordinate projections produced fewer vectors than requested, append
    # the orthonormal SVD basis as a robust fallback.
    for column_index in range(nullity):
        normalized = _phase_fixed_normalized_vector(
            np.asarray(nullspace[:, column_index], dtype=np.complex128),
            tolerance=tolerance,
        )
        if normalized is not None:
            unique = _deduplicate_coefficient_vectors(
                unique + [normalized],
                overlap_tolerance=overlap_tolerance,
            )

    if max_candidates is not None:
        unique = unique[: max(int(max_candidates), 0)]
    if len(unique) == 0:
        return np.zeros((n_operators, 0), dtype=np.complex128)
    return np.column_stack(unique).astype(np.complex128, copy=False)


def _right_nullspace_from_constraint_matrix(
    constraint_matrix: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> tuple[npt.NDArray[np.float64], float, int, npt.NDArray[np.complex128]]:
    """Return singular values, cutoff, rank, and right nullspace.

    The detector constraint matrix is usually tall in production cage runs:
    ``(hilbert_dimension * manifold_dimension) x n_operators``.  Computing a
    full/economy SVD of this tall matrix can dominate the dark-detector stage,
    even though we only need the right nullspace in operator-coefficient space.
    The Hermitian Gram matrix ``C^† C`` has size ``n_operators x n_operators``
    and its eigenvectors are the right singular vectors of ``C``.  This path is
    therefore substantially cheaper for ``coordinate_ipr`` and regional-unit
    workflows with many basis states but modest local-operator families.
    """
    if constraint_matrix.ndim != 2:
        raise ValueError("constraint_matrix must be two-dimensional.")
    n_operators = int(constraint_matrix.shape[1])
    if n_operators == 0:
        return (
            np.zeros(0, dtype=np.float64),
            float(tolerance),
            0,
            np.zeros((0, 0), dtype=np.complex128),
        )

    gram = np.asarray(
        constraint_matrix.conj().T @ constraint_matrix,
        dtype=np.complex128,
    )
    # Symmetrize away tiny BLAS roundoff so eigh sees an exactly Hermitian input.
    gram = 0.5 * (gram + gram.conj().T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
    except np.linalg.LinAlgError:
        full_matrices = constraint_matrix.shape[0] < constraint_matrix.shape[1]
        _, singular_values, vh = np.linalg.svd(
            constraint_matrix,
            full_matrices=full_matrices,
        )
        if singular_values.size == 0:
            cutoff = float(tolerance)
            rank = 0
        else:
            cutoff = float(tolerance * max(float(singular_values[0]), 1.0))
            rank = int(np.count_nonzero(singular_values > cutoff))
        return (
            np.asarray(singular_values, dtype=np.float64),
            cutoff,
            rank,
            vh.conj().T[:, rank:].astype(np.complex128, copy=False),
        )

    eigenvalues = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    singular_values_ascending = np.sqrt(eigenvalues)
    singular_values = singular_values_ascending[::-1].copy()
    if singular_values.size == 0:
        cutoff = float(tolerance)
    else:
        cutoff = float(tolerance * max(float(singular_values[0]), 1.0))
    dark_mask = singular_values_ascending <= cutoff
    rank = int(n_operators - np.count_nonzero(dark_mask))
    nullspace = np.asarray(eigenvectors[:, dark_mask], dtype=np.complex128)
    return singular_values, cutoff, rank, nullspace


def diagnose_manifold_dark_operator_basis(
    *,
    states: npt.ArrayLike,
    operators: tuple[Any, ...] | list[Any],
    operator_names: tuple[str, ...] | list[str] | None = None,
    tolerance: float = 1.0e-10,
    coefficient_tolerance: float = 1.0e-8,
    max_candidates: int | None = 16,
    candidate_strategy: Literal["svd_basis", "coordinate_ipr"] = "svd_basis",
    candidate_overlap_tolerance: float = 1.0e-7,
) -> ManifoldDarkOperatorBasisReport:
    """Find linear combinations of supplied operators annihilating a manifold.

    Args:
        states: Target manifold basis with shape ``(dim, n_states)`` or rows as
            states.  The columns are orthonormalized before the nullspace solve.
        operators: Operator basis matrices with the same Hilbert dimension.
        operator_names: Optional names for the operators.
        tolerance: Absolute/relative SVD tolerance used for the dark-detector
            nullspace.
        coefficient_tolerance: Coefficient magnitude threshold for term readout.
        max_candidates: Maximum number of nullspace candidates to store.  Use
            ``None`` to keep all candidates.
        candidate_strategy: ``"svd_basis"`` keeps the numerical nullspace basis.
            ``"coordinate_ipr"`` projects individual supplied operators onto the
            dark nullspace and ranks the results by coefficient IPR, producing
            more localized/interpretable detector combinations when the dark
            solution space is degenerate.
        candidate_overlap_tolerance: Deduplication tolerance for
            ``candidate_strategy="coordinate_ipr"``.

    Returns:
        A report whose candidate coefficient columns define
        ``D=sum_a c_a O_a`` with ``D P_M ~= 0``.
    """
    operator_matrices = tuple(_as_csr(operator) for operator in operators)
    if len(operator_matrices) == 0:
        raise ValueError("operators must contain at least one matrix.")

    state_basis, gram_residual = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(state_basis.shape[0])
    manifold_dimension = int(state_basis.shape[1])

    for operator in operator_matrices:
        if operator.shape != (dim, dim):
            raise ValueError(
                "operator has incompatible shape: " f"{operator.shape} != {(dim, dim)}."
            )

    if operator_names is None:
        names = tuple(f"O_{index}" for index in range(len(operator_matrices)))
    else:
        names = tuple(str(name) for name in operator_names)
        if len(names) != len(operator_matrices):
            raise ValueError("operator_names length must match operators length.")

    action_columns = [
        np.asarray(operator @ state_basis, dtype=np.complex128).reshape(-1)
        for operator in operator_matrices
    ]
    constraint_matrix = np.column_stack(action_columns).astype(np.complex128, copy=False)

    singular_values, cutoff, rank, nullspace = _right_nullspace_from_constraint_matrix(
        constraint_matrix,
        tolerance=float(tolerance),
    )
    detector_nullity = int(nullspace.shape[1])

    if candidate_strategy not in {"svd_basis", "coordinate_ipr"}:
        raise ValueError('candidate_strategy must be "svd_basis" or "coordinate_ipr".')
    if candidate_strategy == "svd_basis":
        candidate_columns = nullspace
        if max_candidates is not None:
            candidate_columns = candidate_columns[:, : max(int(max_candidates), 0)]
    else:
        candidate_columns = _sparse_ipr_dark_detector_columns(
            nullspace=nullspace,
            max_candidates=max_candidates,
            tolerance=max(float(tolerance), float(coefficient_tolerance)),
            overlap_tolerance=float(candidate_overlap_tolerance),
        )

    candidates: list[ManifoldDarkOperatorCandidate] = []
    for candidate_index in range(candidate_columns.shape[1]):
        coefficients = np.asarray(candidate_columns[:, candidate_index], dtype=np.complex128)
        coefficient_norm = float(np.linalg.norm(coefficients))
        if coefficient_norm == 0.0:
            continue
        coefficients = coefficients / coefficient_norm
        residual = float(np.linalg.norm(constraint_matrix @ coefficients))
        operator_norm = _combined_operator_frobenius_norm(
            operators=operator_matrices,
            coefficients=coefficients,
        )
        relative_residual = residual / max(operator_norm, 1.0)
        coefficient_ipr = _coefficient_ipr(coefficients)
        effective_operator_count = _effective_coefficient_count(coefficients)

        terms = tuple(
            DarkOperatorTerm(
                operator_index=int(index),
                operator_name=names[index],
                coefficient=complex(coefficient),
                weight=float(abs(coefficient)),
            )
            for index, coefficient in sorted(
                enumerate(coefficients),
                key=lambda item: -abs(item[1]),
            )
            if abs(coefficient) > coefficient_tolerance
        )

        candidates.append(
            ManifoldDarkOperatorCandidate(
                candidate_index=int(candidate_index),
                coefficients=coefficients,
                action_residual=residual,
                relative_action_residual=float(relative_residual),
                operator_frobenius_norm=operator_norm,
                coefficient_ipr=coefficient_ipr,
                effective_operator_count=effective_operator_count,
                terms=terms,
            )
        )

    return ManifoldDarkOperatorBasisReport(
        operator_names=names,
        manifold_dimension=manifold_dimension,
        hilbert_dimension=dim,
        gram_residual=gram_residual,
        constraint_matrix_shape=tuple(int(value) for value in constraint_matrix.shape),
        constraint_rank=rank,
        detector_nullity=detector_nullity,
        singular_values=np.asarray(singular_values, dtype=np.float64),
        cutoff=cutoff,
        candidates=tuple(candidates),
        tolerance=float(tolerance),
        candidate_strategy=candidate_strategy,
    )


def _combined_operator(
    *,
    operators: tuple[sp.csr_array, ...],
    coefficients: npt.NDArray[np.complex128],
) -> sp.csr_array:
    if len(operators) == 0:
        raise ValueError("operators must contain at least one matrix.")
    combined = sp.csr_array(operators[0].shape, dtype=np.complex128)
    for coefficient, operator in zip(coefficients, operators, strict=True):
        if abs(coefficient) == 0.0:
            continue
        combined = combined + coefficient * operator
    return combined.tocsr()


def _projected_inflow_norm(
    *,
    jump: sp.csr_array,
    state_basis: npt.NDArray[np.complex128],
) -> tuple[float, float]:
    """Return ``||P J (I-P)||_F`` and ``||P J P||_F`` for ``P=QQ^dag``."""
    adjoint_action = np.asarray(jump.conj().T @ state_basis, dtype=np.complex128)
    left_projected_norm_sq = float(np.linalg.norm(adjoint_action) ** 2)
    target_block = np.asarray(state_basis.conj().T @ (jump @ state_basis), dtype=np.complex128)
    target_block_norm_sq = float(np.linalg.norm(target_block) ** 2)
    inflow_sq = max(left_projected_norm_sq - target_block_norm_sq, 0.0)
    return float(np.sqrt(inflow_sq)), float(np.sqrt(target_block_norm_sq))


def _multi_jump_projected_inflow_norm(
    *,
    jumps: tuple[sp.csr_array, ...] | list[sp.csr_array],
    state_basis: npt.NDArray[np.complex128],
) -> float:
    """Return the incoherent total inflow norm for a jump family.

    This is the cheap part of :func:`diagnose_dark_manifold`: it avoids common
    kernel and Liouvillian checks, but still measures the actual final jump
    matrices rather than the pre-bundled candidate scores.
    """
    total = 0.0
    for jump in jumps:
        inflow_norm, _ = _projected_inflow_norm(jump=jump, state_basis=state_basis)
        total += float(inflow_norm) ** 2
    return float(np.sqrt(max(total, 0.0)))


def _diagonal_vector_if_diagonal(
    operator: sp.csr_array,
    *,
    tolerance: float,
) -> npt.NDArray[np.complex128] | None:
    """Return the diagonal when a sparse operator has no off-diagonal support."""
    coo = operator.tocoo()
    off_diagonal_mask = coo.row != coo.col
    if np.any(np.abs(coo.data[off_diagonal_mask]) > tolerance):
        return None
    return np.asarray(operator.diagonal(), dtype=np.complex128)


def _embedded_matrix_unit_metrics_with_diagonal_right_factor(
    *,
    embedding_context: Any,
    target_local_index: int,
    source_local_index: int,
    right_diagonal: npt.NDArray[np.complex128],
    state_basis: npt.NDArray[np.complex128],
    zero_tolerance: float,
) -> tuple[float, float, float, int, float, int] | None:
    """Fast score ``J = |target><source|_R D`` for diagonal ``D``."""
    transition_mask = (embedding_context.target_local_indices == int(target_local_index)) & (
        embedding_context.source_local_indices == int(source_local_index)
    )
    if not np.any(transition_mask):
        return None

    source_indices = embedding_context.source_full_indices[transition_mask]
    target_indices = embedding_context.target_full_indices[transition_mask]
    jump_values = right_diagonal[source_indices]
    jump_mask = np.abs(jump_values) > zero_tolerance
    jump_nnz = int(np.count_nonzero(jump_mask))
    if jump_nnz == 0:
        return None

    source_indices = source_indices[jump_mask]
    target_indices = target_indices[jump_mask]
    jump_values = jump_values[jump_mask]

    # Matrix units have unit entries on each constrained-basis transition.
    recycler_nnz = int(np.count_nonzero(transition_mask))
    recycler_frobenius_norm = float(np.sqrt(recycler_nnz))

    adjoint_action = np.zeros_like(state_basis, dtype=np.complex128)
    conjugated_values = np.conj(jump_values)
    for state_index in range(state_basis.shape[1]):
        np.add.at(
            adjoint_action[:, state_index],
            source_indices,
            conjugated_values * state_basis[target_indices, state_index],
        )

    target_block = adjoint_action.conj().T @ state_basis
    target_block_norm_sq = float(np.linalg.norm(target_block) ** 2)
    adjoint_norm_sq = float(np.linalg.norm(adjoint_action) ** 2)
    inflow_norm = float(np.sqrt(max(adjoint_norm_sq - target_block_norm_sq, 0.0)))
    jump_frobenius_norm = float(np.linalg.norm(jump_values))
    return (
        inflow_norm,
        float(np.sqrt(max(target_block_norm_sq, 0.0))),
        jump_frobenius_norm,
        jump_nnz,
        recycler_frobenius_norm,
        recycler_nnz,
    )


def _embedded_local_operator_metrics_with_diagonal_right_factor(
    *,
    embedding_context: Any,
    local_operator: npt.NDArray[np.complex128],
    right_diagonal: npt.NDArray[np.complex128],
    state_basis: npt.NDArray[np.complex128],
    zero_tolerance: float,
) -> tuple[float, float, float, int, float, int] | None:
    """Score ``J = R D`` without materializing sparse matrices when ``D`` is diagonal.

    The generic recycled-detector scan used to build every embedded local
    recycler ``R``, multiply it by the detector ``D``, and then multiply the
    resulting sparse matrix by the target-manifold basis.  In QDM production
    runs the detector basis is normally diagonal plaquette-projector data.  For
    that common case, the nonzero entries of ``J`` are just the embedded local
    entries of ``R`` scaled by the source-basis diagonal of ``D``.  Computing the
    projected inflow directly from these arrays avoids hundreds of thousands of
    tiny CSR constructions and sparse products.
    """
    if local_operator.shape != (embedding_context.local_dim, embedding_context.local_dim):
        raise ValueError(
            "local_operator has incompatible shape: "
            f"{local_operator.shape} != "
            f"{(embedding_context.local_dim, embedding_context.local_dim)}."
        )
    if embedding_context.source_full_indices.size == 0:
        return None

    local_values = np.asarray(
        local_operator[
            embedding_context.target_local_indices,
            embedding_context.source_local_indices,
        ],
        dtype=np.complex128,
    )
    recycler_mask = np.abs(local_values) > zero_tolerance
    recycler_nnz = int(np.count_nonzero(recycler_mask))
    if recycler_nnz == 0:
        return None

    source_indices = embedding_context.source_full_indices
    target_indices = embedding_context.target_full_indices
    jump_values = local_values * right_diagonal[source_indices]
    jump_mask = np.abs(jump_values) > zero_tolerance
    jump_nnz = int(np.count_nonzero(jump_mask))
    if jump_nnz == 0:
        return None

    jump_values = jump_values[jump_mask]
    source_indices = source_indices[jump_mask]
    target_indices = target_indices[jump_mask]

    adjoint_action = np.zeros_like(state_basis, dtype=np.complex128)
    conjugated_values = np.conj(jump_values)
    for state_index in range(state_basis.shape[1]):
        np.add.at(
            adjoint_action[:, state_index],
            source_indices,
            conjugated_values * state_basis[target_indices, state_index],
        )

    target_block = adjoint_action.conj().T @ state_basis
    target_block_norm_sq = float(np.linalg.norm(target_block) ** 2)
    adjoint_norm_sq = float(np.linalg.norm(adjoint_action) ** 2)
    inflow_norm = float(np.sqrt(max(adjoint_norm_sq - target_block_norm_sq, 0.0)))

    jump_frobenius_norm = float(np.linalg.norm(jump_values))
    recycler_frobenius_norm = float(np.linalg.norm(local_values[recycler_mask]))
    return (
        inflow_norm,
        float(np.sqrt(max(target_block_norm_sq, 0.0))),
        jump_frobenius_norm,
        jump_nnz,
        recycler_frobenius_norm,
        recycler_nnz,
    )


def _embedded_matrix_unit_times_diagonal_as_csr(
    *,
    embedding_context: Any,
    target_local_index: int,
    source_local_index: int,
    right_diagonal: npt.NDArray[np.complex128],
    dim: int,
    zero_tolerance: float,
) -> sp.csr_array:
    """Build ``|target><source|_R D`` directly for diagonal ``D``."""
    transition_mask = (embedding_context.target_local_indices == int(target_local_index)) & (
        embedding_context.source_local_indices == int(source_local_index)
    )
    if not np.any(transition_mask):
        return sp.csr_array((dim, dim), dtype=np.complex128)

    source_indices = embedding_context.source_full_indices[transition_mask]
    target_indices = embedding_context.target_full_indices[transition_mask]
    jump_values = np.asarray(right_diagonal[source_indices], dtype=np.complex128)
    jump_mask = np.abs(jump_values) > zero_tolerance
    if not np.any(jump_mask):
        return sp.csr_array((dim, dim), dtype=np.complex128)

    return sp.csr_array(
        (
            jump_values[jump_mask],
            (target_indices[jump_mask], source_indices[jump_mask]),
        ),
        shape=(dim, dim),
        dtype=np.complex128,
    )


def _embedded_local_operator_times_diagonal_as_csr(
    *,
    embedding_context: Any,
    local_operator: npt.NDArray[np.complex128],
    right_diagonal: npt.NDArray[np.complex128],
    dim: int,
    zero_tolerance: float,
) -> sp.csr_array:
    """Build embedded ``R D`` directly when ``D`` is diagonal."""
    if local_operator.shape != (embedding_context.local_dim, embedding_context.local_dim):
        raise ValueError(
            "local_operator has incompatible shape: "
            f"{local_operator.shape} != "
            f"{(embedding_context.local_dim, embedding_context.local_dim)}."
        )
    if embedding_context.source_full_indices.size == 0:
        return sp.csr_array((dim, dim), dtype=np.complex128)

    local_values = np.asarray(
        local_operator[
            embedding_context.target_local_indices,
            embedding_context.source_local_indices,
        ],
        dtype=np.complex128,
    )
    recycler_mask = np.abs(local_values) > zero_tolerance
    if not np.any(recycler_mask):
        return sp.csr_array((dim, dim), dtype=np.complex128)

    source_indices = embedding_context.source_full_indices
    target_indices = embedding_context.target_full_indices
    jump_values = local_values * right_diagonal[source_indices]
    jump_mask = np.abs(jump_values) > zero_tolerance
    if not np.any(jump_mask):
        return sp.csr_array((dim, dim), dtype=np.complex128)

    return sp.csr_array(
        (
            jump_values[jump_mask],
            (target_indices[jump_mask], source_indices[jump_mask]),
        ),
        shape=(dim, dim),
        dtype=np.complex128,
    )


def _recycled_candidate_sort_key(
    candidate: RecycledManifoldDarkDetectorCandidate,
    *,
    dark_tolerance: float,
) -> tuple[bool, float, float, int, int, int, int]:
    return (
        candidate.relative_dark_residual > dark_tolerance,
        -candidate.inflow_norm,
        candidate.relative_dark_residual,
        candidate.jump_nnz,
        candidate.detector_index,
        candidate.region_index,
        candidate.recycler_index,
    )


def _append_ranked_recycled_candidate(
    candidates: list[RecycledManifoldDarkDetectorCandidate],
    candidate: RecycledManifoldDarkDetectorCandidate,
    *,
    max_report_candidates: int | None,
    dark_tolerance: float,
) -> None:
    candidates.append(candidate)
    if max_report_candidates is None:
        return
    limit = max(int(max_report_candidates), 0)
    if limit == 0:
        candidates.clear()
        return
    if len(candidates) <= limit:
        return
    candidates.sort(
        key=lambda item: _recycled_candidate_sort_key(
            item,
            dark_tolerance=dark_tolerance,
        )
    )
    del candidates[limit:]


def _normalize_detector_coefficients(
    detector_coefficients: npt.ArrayLike,
    *,
    n_operators: int,
) -> npt.NDArray[np.complex128]:
    coefficients = np.asarray(detector_coefficients, dtype=np.complex128)
    if coefficients.ndim == 1:
        if coefficients.shape[0] != n_operators:
            raise ValueError(
                "detector_coefficients has incompatible length: "
                f"{coefficients.shape[0]} != {n_operators}."
            )
        coefficients = coefficients.reshape(n_operators, 1)
    elif coefficients.ndim == 2:
        if coefficients.shape[0] == n_operators:
            pass
        elif coefficients.shape[1] == n_operators:
            coefficients = coefficients.T
        else:
            raise ValueError(
                "detector_coefficients must have shape "
                "(n_operators, n_detectors) or (n_detectors, n_operators)."
            )
    else:
        raise ValueError("detector_coefficients must be one- or two-dimensional.")

    if coefficients.shape[1] == 0:
        raise ValueError("detector_coefficients must contain at least one detector.")

    normalized = coefficients.copy()
    for column_index in range(normalized.shape[1]):
        norm = float(np.linalg.norm(normalized[:, column_index]))
        if norm == 0.0:
            raise ValueError("detector_coefficients contains a zero detector column.")
        normalized[:, column_index] /= norm
    return normalized


def diagnose_dressed_manifold_dark_detectors(
    *,
    states: npt.ArrayLike,
    detector_operators: tuple[Any, ...] | list[Any],
    left_multipliers: tuple[Any, ...] | list[Any],
    detector_coefficients: npt.ArrayLike | None = None,
    dark_operator_report: ManifoldDarkOperatorBasisReport | None = None,
    detector_operator_names: tuple[str, ...] | list[str] | None = None,
    left_multiplier_names: tuple[str, ...] | list[str] | None = None,
    detector_names: tuple[str, ...] | list[str] | None = None,
    tolerance: float = 1.0e-10,
    dark_tolerance: float = 1.0e-10,
    inflow_tolerance: float = 1.0e-12,
    max_detectors: int | None = None,
    sort_by_inflow: bool = True,
) -> DressedManifoldDarkDetectorReport:
    """Test paper-style dressed jumps ``J = V D`` for a dark manifold.

    Args:
        states: Target manifold basis.  Columns are orthonormalized.
        detector_operators: Operator basis ``O_a`` used to assemble
            ``D=sum_a c_a O_a``.
        left_multipliers: Candidate left multipliers ``V_beta``.
        detector_coefficients: Optional coefficient matrix for the detectors.
            If omitted, coefficients are taken from ``dark_operator_report``.
        dark_operator_report: Optional report from
            :func:`diagnose_manifold_dark_operator_basis`.
        detector_operator_names: Names for ``detector_operators``.  Only used
            to build default detector names.
        left_multiplier_names: Names for the left multipliers.
        detector_names: Optional explicit detector names.
        tolerance: Orthonormalization and shape-check tolerance.
        dark_tolerance: Relative dark residual threshold.
        inflow_tolerance: Direct-inflow threshold.
        max_detectors: Optional maximum number of detectors to test.
        sort_by_inflow: If true, store candidates with largest inflow first.

    Returns:
        A report of dressed candidates.  A candidate with small dark residual
        and positive inflow satisfies the necessary direct-inflow condition for
        manifold attraction, but does not by itself rule out invariant sectors
        in the complement.
    """
    detector_matrices = tuple(_as_csr(operator) for operator in detector_operators)
    multiplier_matrices = tuple(_as_csr(operator) for operator in left_multipliers)
    if len(detector_matrices) == 0:
        raise ValueError("detector_operators must contain at least one matrix.")
    if len(multiplier_matrices) == 0:
        raise ValueError("left_multipliers must contain at least one matrix.")

    state_basis, gram_residual = _normalize_state_columns(states, tolerance=tolerance)
    dim = int(state_basis.shape[0])
    manifold_dimension = int(state_basis.shape[1])

    for operator in detector_matrices + multiplier_matrices:
        if operator.shape != (dim, dim):
            raise ValueError(
                "operator has incompatible shape: " f"{operator.shape} != {(dim, dim)}."
            )

    if detector_coefficients is None:
        if dark_operator_report is None:
            raise ValueError(
                "Pass detector_coefficients or dark_operator_report to define detectors."
            )
        detector_coefficients = np.column_stack(
            [candidate.coefficients for candidate in dark_operator_report.candidates]
        )

    coefficients = _normalize_detector_coefficients(
        detector_coefficients,
        n_operators=len(detector_matrices),
    )
    if max_detectors is not None:
        coefficients = coefficients[:, : max(int(max_detectors), 0)]

    if detector_operator_names is None:
        operator_names = tuple(f"O_{index}" for index in range(len(detector_matrices)))
    else:
        operator_names = tuple(str(name) for name in detector_operator_names)
        if len(operator_names) != len(detector_matrices):
            raise ValueError("detector_operator_names length must match detector_operators.")

    if detector_names is None:
        names = tuple(
            _default_detector_name(
                coefficients=coefficients[:, detector_index],
                operator_names=operator_names,
            )
            for detector_index in range(coefficients.shape[1])
        )
    else:
        names = tuple(str(name) for name in detector_names)
        if len(names) != coefficients.shape[1]:
            raise ValueError("detector_names length must match detector count.")

    if left_multiplier_names is None:
        multiplier_names = tuple(f"V_{index}" for index in range(len(multiplier_matrices)))
    else:
        multiplier_names = tuple(str(name) for name in left_multiplier_names)
        if len(multiplier_names) != len(multiplier_matrices):
            raise ValueError("left_multiplier_names length must match left_multipliers.")

    candidates: list[DressedManifoldDarkDetectorCandidate] = []
    for detector_index in range(coefficients.shape[1]):
        detector = _combined_operator(
            operators=detector_matrices,
            coefficients=coefficients[:, detector_index],
        )
        detector_action_residual = float(np.linalg.norm(detector @ state_basis))
        detector_norm = float(sp.linalg.norm(detector))
        detector_relative_residual = detector_action_residual / max(detector_norm, 1.0)
        for multiplier_index, multiplier in enumerate(multiplier_matrices):
            jump = (multiplier @ detector).tocsr()
            dark_residual = float(np.linalg.norm(jump @ state_basis))
            jump_norm = float(sp.linalg.norm(jump))
            relative_dark_residual = dark_residual / max(jump_norm, 1.0)
            inflow_norm, target_block_norm = _projected_inflow_norm(
                jump=jump,
                state_basis=state_basis,
            )
            candidates.append(
                DressedManifoldDarkDetectorCandidate(
                    candidate_index=len(candidates),
                    detector_index=int(detector_index),
                    detector_name=names[detector_index],
                    left_multiplier_index=int(multiplier_index),
                    left_multiplier_name=multiplier_names[multiplier_index],
                    dark_residual=dark_residual,
                    relative_dark_residual=float(relative_dark_residual),
                    inflow_norm=inflow_norm,
                    jump_frobenius_norm=jump_norm,
                    target_block_norm=target_block_norm,
                    detector_action_residual=detector_action_residual,
                    detector_relative_action_residual=float(detector_relative_residual),
                )
            )

    if sort_by_inflow:
        candidates = sorted(
            candidates,
            key=lambda candidate: (
                candidate.relative_dark_residual > dark_tolerance,
                -candidate.inflow_norm,
                candidate.relative_dark_residual,
            ),
        )
        candidates = [
            DressedManifoldDarkDetectorCandidate(
                candidate_index=index,
                detector_index=candidate.detector_index,
                detector_name=candidate.detector_name,
                left_multiplier_index=candidate.left_multiplier_index,
                left_multiplier_name=candidate.left_multiplier_name,
                dark_residual=candidate.dark_residual,
                relative_dark_residual=candidate.relative_dark_residual,
                inflow_norm=candidate.inflow_norm,
                jump_frobenius_norm=candidate.jump_frobenius_norm,
                target_block_norm=candidate.target_block_norm,
                detector_action_residual=candidate.detector_action_residual,
                detector_relative_action_residual=candidate.detector_relative_action_residual,
            )
            for index, candidate in enumerate(candidates)
        ]

    return DressedManifoldDarkDetectorReport(
        manifold_dimension=manifold_dimension,
        hilbert_dimension=dim,
        gram_residual=gram_residual,
        detector_names=names,
        left_multiplier_names=multiplier_names,
        dark_tolerance=float(dark_tolerance),
        inflow_tolerance=float(inflow_tolerance),
        candidates=tuple(candidates),
    )


def _orthogonal_complement_basis(
    basis: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> npt.NDArray[np.complex128]:
    """Return an orthonormal basis of the complement of ``span(basis)``."""
    q, _ = _normalize_state_columns(basis, tolerance=tolerance)
    _u, singular_values, vh = np.linalg.svd(q.conj().T, full_matrices=True)
    singular_scale = float(singular_values[0]) if singular_values.size else 1.0
    cutoff = max(float(tolerance), float(tolerance) * singular_scale)
    rank = int(np.count_nonzero(singular_values > cutoff))
    return vh.conj().T[:, rank:].astype(np.complex128, copy=False)


def _right_kernel_basis(
    matrix: npt.NDArray[np.complex128],
    *,
    tolerance: float,
) -> npt.NDArray[np.complex128]:
    """Return an orthonormal basis for the right kernel of ``matrix``."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    n_columns = int(matrix.shape[1])
    if n_columns == 0:
        return np.zeros((0, 0), dtype=np.complex128)

    full_matrices = matrix.shape[0] < matrix.shape[1]
    _u, singular_values, vh = np.linalg.svd(matrix, full_matrices=full_matrices)
    if singular_values.size == 0:
        rank = 0
    else:
        cutoff = max(float(tolerance), float(np.sqrt(tolerance)) * float(singular_values[0]))
        rank = int(np.count_nonzero(singular_values > cutoff))
    if rank >= n_columns:
        return np.zeros((n_columns, 0), dtype=np.complex128)
    return vh.conj().T[:, rank:].astype(np.complex128, copy=False)


def _default_detector_name(
    *,
    coefficients: npt.NDArray[np.complex128],
    operator_names: tuple[str, ...],
    max_terms: int = 4,
) -> str:
    terms = []
    for index, coefficient in sorted(
        enumerate(coefficients),
        key=lambda item: -abs(item[1]),
    )[:max_terms]:
        if abs(coefficient) <= 1.0e-8:
            continue
        terms.append(f"{coefficient:.3g}·{operator_names[index]}")
    if len(terms) == 0:
        return "0"
    if np.count_nonzero(np.abs(coefficients) > 1.0e-8) > max_terms:
        terms.append("…")
    return " + ".join(terms)
