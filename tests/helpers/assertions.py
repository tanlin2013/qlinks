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


def assert_optional_sparse_allclose(
    actual: sparse.spmatrix | sparse.sparray | None,
    expected: sparse.spmatrix | sparse.sparray | None,
    *,
    atol: float = 1.0e-12,
) -> None:
    if actual is None or expected is None:
        assert actual is None and expected is None
        return

    assert_sparse_allclose(actual, expected, atol=atol)


def assert_same_binary_basis_order(sparse_result, bitmask_result) -> None:
    """Assert that a binary bitmask basis preserves array-basis order."""
    sparse_states = sparse_result.basis.states
    bitmask_states = bitmask_result.basis.to_array_basis().states

    np.testing.assert_array_equal(bitmask_states, sparse_states)


def assert_same_physical_flux_basis_order(sparse_result, bitmask_result) -> None:
    """
    Assert that a binary bitmask basis matches a physical flux basis order.

    Bitmask QLM builders internally encode physical flux values {-1, +1}
    as binary values {0, 1}. This helper converts the bitmask basis back to
    physical flux values and compares it against the sparse-builder basis.
    """
    sparse_states = sparse_result.basis.states
    bitmask_binary_states = bitmask_result.basis.to_array_basis().states
    bitmask_flux_states = 2 * bitmask_binary_states - 1

    np.testing.assert_array_equal(bitmask_flux_states, sparse_states)
