from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.linalg as scipy_linalg
import scipy.sparse as scipy_sparse
from scipy.sparse.csgraph import connected_components


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
