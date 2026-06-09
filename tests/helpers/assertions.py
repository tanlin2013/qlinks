from __future__ import annotations

import numpy as np
from scipy import sparse


def as_csr(matrix: sparse.spmatrix | sparse.sparray) -> sparse.csr_array:
    return sparse.csr_array(matrix)


def assert_sparse_allclose(
    actual: sparse.spmatrix | sparse.sparray,
    expected: sparse.spmatrix | sparse.sparray,
    *,
    atol: float = 1.0e-12,
) -> None:
    actual_csr = as_csr(actual)
    expected_csr = as_csr(expected)

    difference = actual_csr - expected_csr
    difference.eliminate_zeros()

    if difference.nnz == 0:
        return

    max_abs = np.max(np.abs(difference.data))
    assert max_abs < atol


def assert_hermitian_sparse(
    matrix: sparse.spmatrix | sparse.sparray,
    *,
    atol: float = 1.0e-12,
) -> None:
    csr = as_csr(matrix)
    difference = csr - csr.conj().T
    difference.eliminate_zeros()

    if difference.nnz == 0:
        return

    assert np.max(np.abs(difference.data)) < atol


def assert_same_sparse_matrix(
    actual: sparse.spmatrix | sparse.sparray,
    expected: sparse.spmatrix | sparse.sparray,
) -> None:
    actual_csr = as_csr(actual)
    expected_csr = as_csr(expected)
    difference = actual_csr - expected_csr
    difference.eliminate_zeros()
    assert difference.nnz == 0
