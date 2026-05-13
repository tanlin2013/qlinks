from __future__ import annotations

import numpy as np
import scipy.linalg as scipy_linalg
import scipy.sparse as scipy_sparse
from numpy.typing import NDArray


def as_dense_array(matrix: object) -> NDArray[np.complex128]:
    """Convert a dense or sparse matrix to a dense complex NumPy array."""
    if scipy_sparse.issparse(matrix):
        return matrix.toarray().astype(np.complex128, copy=False)

    return np.asarray(matrix, dtype=np.complex128)


def nullspace_svd(
    matrix: object,
    *,
    tolerance: float = 1e-10,
) -> NDArray[np.complex128]:
    """
    Return an orthonormal basis for the nullspace of a matrix.

    The returned array has shape ``(n_columns, nullity)``.
    """
    dense_matrix = as_dense_array(matrix)

    if dense_matrix.ndim != 2:
        raise ValueError("matrix must be 2D.")

    row_count, column_count = dense_matrix.shape

    if column_count == 0:
        return np.zeros((0, 0), dtype=np.complex128)

    if row_count == 0:
        return np.eye(column_count, dtype=np.complex128)

    _left_vectors, singular_values, right_vectors_h = scipy_linalg.svd(
        dense_matrix,
        full_matrices=True,
    )

    rank = int(np.sum(singular_values > tolerance))
    nullity = column_count - rank

    if nullity <= 0:
        return np.zeros((column_count, 0), dtype=np.complex128)

    return right_vectors_h.conj().T[:, rank:]
